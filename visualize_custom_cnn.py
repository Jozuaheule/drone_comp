
import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import hdf5plugin

class GateDataset(Dataset):
    def __init__(self, h5_path, grid_size=7, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.S = grid_size
        with h5py.File(h5_path, 'r') as f:
            self.images = f['images'][:]
            self.keys = sorted(f['targets'].keys(), key=int)
            self.targets_data = [f['targets'][key][:] for key in self.keys]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target_data = self.targets_data[idx]

        image_tensor = torch.from_numpy(image).float() / 255.0

        label_matrix = torch.zeros((self.S, self.S, 5))
        if target_data.size > 0:
            num_gates = target_data.shape[0]
            for i in range(num_gates):
                gate_coords_with_flags = target_data[i]
                gate_coords = gate_coords_with_flags.reshape(4, 3)[:, :2]
                xmin = gate_coords[:, 0].min()
                ymin = gate_coords[:, 1].min()
                xmax = gate_coords[:, 0].max()
                ymax = gate_coords[:, 1].max()
                
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin

                grid_x = int(self.S * x_center)
                grid_y = int(self.S * y_center)

                x_cell = self.S * x_center - grid_x
                y_cell = self.S * y_center - grid_y

                if label_matrix[grid_y, grid_x, 0] == 0:
                    label_matrix[grid_y, grid_x, 0] = 1
                    label_matrix[grid_y, grid_x, 1:] = torch.tensor([x_cell, y_cell, w, h])

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label_matrix

class CustomCNN(nn.Module):
    def __init__(self, grid_size=7):
        super(CustomCNN, self).__init__()
        self.S = grid_size
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 30 * 40, 4096)
        self.fc2 = nn.Linear(4096, self.S * self.S * 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.reshape(-1, self.S, self.S, 5)

def get_bboxes_from_grid(grid_preds, grid_targets, S=7, conf_thresh=0.5):
    pred_bboxes = []
    target_bboxes = []

    for i in range(S):
        for j in range(S):
            # Process predictions
            if grid_preds[i, j, 0] > conf_thresh:
                xc, yc, w, h = grid_preds[i, j, 1:]
                x = (j + xc) / S - w / 2
                y = (i + yc) / S - h / 2
                pred_bboxes.append([x, y, w, h])

            # Process targets
            if grid_targets[i, j, 0] == 1:
                xc, yc, w, h = grid_targets[i, j, 1:]
                x = (j + xc) / S - w / 2
                y = (i + yc) / S - h / 2
                target_bboxes.append([x, y, w, h])
                
    return pred_bboxes, target_bboxes

def visualize_predictions(model, loader, device, num_images=16):
    model.eval()
    images, targets = next(iter(loader))
    images, targets = images.to(device), targets.to(device)

    with torch.no_grad():
        preds = model(images)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_images):
        ax = axes[i]
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Un-normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        pred_bboxes, target_bboxes = get_bboxes_from_grid(preds[i], targets[i])
        
        # Plot predicted boxes
        for bbox in pred_bboxes:
            x, y, w, h = bbox
            x, w = x * 640, w * 640
            y, h = y * 480, h * 480
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Plot target boxes
        for bbox in target_bboxes:
            x, y, w, h = bbox
            x, w = x * 640, w * 640
            y, h = y * 480, h * 480
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('visualization.png')
    print("Visualization saved as visualization.png")

def main():
    DATA_ROOT = '/Users/m.j.j.heule/Documents/4. TU Delft/Master/AI for Aerodpace Engineering/drone_comp/ae4353-y25'
    MODEL_PATH = 'custom_cnn_model.pth'
    
    # Use one of the validation files for visualization
    h5_path = os.path.join(DATA_ROOT, 'piloted_flight-03p-ellipse.h5')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transforms = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = GateDataset(h5_path, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = CustomCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    print("Visualizing model predictions...")
    visualize_predictions(model, loader, device)

if __name__ == '__main__':
    main()
