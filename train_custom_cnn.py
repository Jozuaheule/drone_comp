
import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
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

from torchvision.ops import sigmoid_focal_loss, complete_box_iou_loss

class CustomLoss(nn.Module):
    def __init__(self, grid_size=7, lambda_coord=5, lambda_noobj=0.5, lambda_corner=0.1):
        super(CustomLoss, self).__init__()
        self.S = grid_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_corner = lambda_corner
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, predictions, target):
        exists_box = target[..., 0] == 1
        no_exists_box = target[..., 0] == 0

        # Confidence loss (Focal Loss)
        box_preds = predictions[exists_box]
        box_targets = target[exists_box]
        obj_loss = sigmoid_focal_loss(box_preds[:, 0], box_targets[:, 0], reduction='mean')

        noobj_preds = predictions[no_exists_box]
        noobj_targets = target[no_exists_box]
        noobj_loss = sigmoid_focal_loss(noobj_preds[:, 0], noobj_targets[:, 0], reduction='mean')

        # Box coordinates loss (CIoU)
        box_preds_coords = box_preds[:, 1:5]
        box_targets_coords = box_targets[:, 1:5]

        # Ensure width and height are non-negative and non-zero
        w_pred = torch.abs(box_preds_coords[..., 2]) + 1e-6
        h_pred = torch.abs(box_preds_coords[..., 3]) + 1e-6
        x_pred = box_preds_coords[..., 0]
        y_pred = box_preds_coords[..., 1]

        w_targ = box_targets_coords[..., 2]
        h_targ = box_targets_coords[..., 3]
        x_targ = box_targets_coords[..., 0]
        y_targ = box_targets_coords[..., 1]

        # Convert to (x1, y1, x2, y2) format for CIoU loss
        box_preds_x1y1x2y2 = torch.zeros_like(box_preds_coords)
        box_preds_x1y1x2y2[..., 0] = x_pred - w_pred / 2
        box_preds_x1y1x2y2[..., 1] = y_pred - h_pred / 2
        box_preds_x1y1x2y2[..., 2] = x_pred + w_pred / 2
        box_preds_x1y1x2y2[..., 3] = y_pred + h_pred / 2
        
        box_targets_x1y1x2y2 = torch.zeros_like(box_targets_coords)
        box_targets_x1y1x2y2[..., 0] = x_targ - w_targ / 2
        box_targets_x1y1x2y2[..., 1] = y_targ - h_targ / 2
        box_targets_x1y1x2y2[..., 2] = x_targ + w_targ / 2
        box_targets_x1y1x2y2[..., 3] = y_targ + h_targ / 2

        ciou_loss = torch.mean(complete_box_iou_loss(box_preds_x1y1x2y2, box_targets_x1y1x2y2))

        # Corner loss
        corner_loss = self.l1(box_preds_x1y1x2y2, box_targets_x1y1x2y2)

        total_loss = (
            self.lambda_coord * ciou_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_corner * corner_loss
        )
        return total_loss

def main():
    DATA_ROOT = '/Users/m.j.j.heule/Documents/4. TU Delft/Master/AI for Aerodpace Engineering/drone_comp/ae4353-y25'
    
    files_to_use = [
        'autonomous_flight-08a-lemniscate.h5', 'piloted_flight-03p-ellipse.h5', 'piloted_flight-08p-lemniscate.h5',
        'autonomous_flight-01a-ellipse.h5', 'autonomous_flight-13a-trackRATM.h5'
    ]
    h5_paths = [os.path.join(DATA_ROOT, fname) for fname in files_to_use]

    print(f"Training on {len(h5_paths)} files: {files_to_use}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transforms = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datasets = [GateDataset(f, grid_size=7, transform=data_transforms) for f in h5_paths]
    full_dataset = torch.utils.data.ConcatDataset(datasets)

    # Create a random subset for training
    total_size = len(full_dataset)
    subset_indices = np.random.choice(total_size, 50, replace=False)
    subset = torch.utils.data.Subset(full_dataset, subset_indices)

    # Create training and validation sets
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_set, val_set = torch.utils.data.random_split(subset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    model = CustomCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)
    loss_fn = CustomLoss()

    num_epochs = 10
    best_val_loss = float('inf')
    
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)                                                                                              
                                                                                                                                                       
        # Validation                                                                                                                                 
        model.eval()                                                                                                                                 
        val_loss = 0                                                                                                                                 
        with torch.no_grad():                                                                                                                        
            for images, targets in val_loader:                                                                                                       
                images = images.to(device)                                                                                                           
                targets = targets.to(device)                                                                                                         
                predictions = model(images)                                                                                                          
                val_loss += loss_fn(predictions, targets).item()                                                                                     
                                                                                                                                                
        avg_val_loss = val_loss / len(val_loader)                                                                                                    
                                                                                                                                                        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")                                                  
                                                                                                                                                            
        if avg_val_loss < best_val_loss:                                                                                                             
            best_val_loss = avg_val_loss                                                                                                             
            torch.save(model.state_dict(), 'custom_cnn_model.pth')                                                                                   
            patience_counter = 0                                                                                                                     
            print("🏆 New best model saved!")                                                                                                        
        else:                                                                                                                                        
            patience_counter += 1                                                                                                                   
            if patience_counter >= patience:                                                                                                         
                print(f"Early stopping at epoch {epoch+1}")                                                                                          
                break                                      

    torch.save(model.state_dict(), 'custom_cnn_model.pth')
    
    print("\n✅ Training complete! Model saved as custom_cnn_model.pth")

if __name__ == '__main__':
    main()
