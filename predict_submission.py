
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import hdf5plugin

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

class TestDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            self.images = f['images'][:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_tensor = torch.from_numpy(image).float() / 255.0
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor

def get_bboxes_from_grid_pred(grid_preds, S=7, conf_thresh=0.5):
    pred_bboxes = []
    best_conf = 0
    best_box = None
    for i in range(S):
        for j in range(S):
            if grid_preds[i, j, 0] > best_conf:
                best_conf = grid_preds[i, j, 0]
                xc, yc, w, h = grid_preds[i, j, 1:]
                x = (j + xc) / S - w / 2
                y = (i + yc) / S - h / 2
                best_box = [x, y, w, h]
    
    if best_box is not None:
        pred_bboxes.append(best_box)
        
    return pred_bboxes

def main():
    DATA_ROOT = '/Users/m.j.j.heule/Documents/4. TU Delft/Master/AI for Aerodpace Engineering/drone_comp/ae4353-y25'
    TEST_SET_PATH = os.path.join(DATA_ROOT, 'test_set.h5')
    MODEL_PATH = 'custom_cnn_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load the trained model
    model = CustomCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 2. Create the test dataset and dataloader
    data_transforms = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = TestDataset(TEST_SET_PATH, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. Generate predictions
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc='Generating predictions'):
            images = images.to(device)
            outputs = model(images)
            for i in range(outputs.shape[0]):
                bboxes = get_bboxes_from_grid_pred(outputs[i])
                predictions.append(bboxes)

    # 4. Format predictions into the submission string format
    prediction_strings = []
    for bboxes in predictions:
        pred_str = ''
        for bbox in bboxes:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y
            x3, y3 = x + w, y + h
            x4, y4 = x, y + h
            pred_str += f'{x1} {y1} 2.0 {x2} {y2} 2.0 {x3} {y3} 2.0 {x4} {y4} 2.0 '
        prediction_strings.append(pred_str.strip())

    # 5. Create the submission DataFrame
    submission_df = pd.DataFrame({
        'Id': range(len(prediction_strings)),
        'PredictionString': prediction_strings
    })

    # 6. Save the submission file
    submission_df.to_csv('submission.csv', index=False)

    print("\n✅ Submission file 'submission.csv' created successfully!")

if __name__ == '__main__':
    main()
