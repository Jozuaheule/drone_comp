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
import torchvision


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


def get_bboxes_from_grid_pred_improved(grid_preds, S=14, conf_thresh=0.25, nms_thresh=0.4):
    """
    Improved bbox extraction with:
    - Multiple anchor support
    - NMS to remove duplicates  
    - Adaptive confidence thresholding
    """
    pred_bboxes = []
    
    # Handle multiple anchors
    if len(grid_preds.shape) == 4:
        num_anchors = grid_preds.shape[2]
    else:
        num_anchors = 1
        grid_preds = grid_preds.unsqueeze(2)
    
    for i in range(S):
        for j in range(S):
            for a in range(num_anchors):
                conf = torch.sigmoid(grid_preds[i, j, a, 0])
                
                if conf > conf_thresh:
                    xc, yc, w, h = grid_preds[i, j, a, 1:]
                    
                    # Convert to absolute coordinates with sigmoid for x,y offsets
                    x = (j + torch.sigmoid(xc)) / S
                    y = (i + torch.sigmoid(yc)) / S
                    w = torch.abs(w)
                    h = torch.abs(h)
                    
                    # Convert to x1y1x2y2 format
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    
                    pred_bboxes.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])
    
    # Apply NMS if multiple boxes detected
    if len(pred_bboxes) > 1:
        pred_bboxes_tensor = torch.tensor(pred_bboxes)
        boxes = pred_bboxes_tensor[:, :4]
        scores = pred_bboxes_tensor[:, 4]
        
        # Apply NMS
        keep_indices = torchvision.ops.nms(boxes, scores, nms_thresh)
        pred_bboxes_tensor = pred_bboxes_tensor[keep_indices]
        
        # Convert back to xywh format
        final_boxes = []
        for box in pred_bboxes_tensor:
            x1, y1, x2, y2, conf = box
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            final_boxes.append([x.item(), y.item(), w.item(), h.item()])
        
        return final_boxes
    elif len(pred_bboxes) == 1:
        box = pred_bboxes[0]
        x1, y1, x2, y2, conf = box
        return [[x1, y1, x2 - x1, y2 - y1]]
    
    return []


def predict_with_tta(model, image, device, S=14):
    """Test-Time Augmentation for more robust predictions"""
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = model(image.to(device))
        predictions.append(pred)
    
    # Horizontal flip
    image_flipped = torch.flip(image, dims=[3])
    with torch.no_grad():
        pred_flipped = model(image_flipped.to(device))
        # Un-flip the prediction
        pred_flipped[..., 1] = 1 - pred_flipped[..., 1]  # Flip x-coordinate
        predictions.append(pred_flipped)
    
    # Average predictions
    avg_pred = torch.mean(torch.stack(predictions), dim=0)
    return avg_pred


def main():
    DATA_ROOT = '/Users/m.j.j.heule/Documents/4. TU Delft/Master/AI for Aerodpace Engineering/drone_comp/ae4353-y25'
    TEST_SET_PATH = os.path.join(DATA_ROOT, 'test_set.h5')
    MODEL_PATH = 'improved_gate_model_best.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    grid_size = checkpoint.get('grid_size', 14)
    num_anchors = checkpoint.get('num_anchors', 2)
    
    print(f"Loading model with grid_size={grid_size}, num_anchors={num_anchors}")

    # Import and initialize model
    from improved_gate_model import ImprovedGateCNN
    model = ImprovedGateCNN(grid_size=grid_size, num_anchors=num_anchors).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create test dataset
    data_transforms = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestDataset(TEST_SET_PATH, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    print(f"Test set size: {len(test_dataset)}")

    # Generate predictions
    predictions = []
    use_tta = False  # Set to True for Test-Time Augmentation (slower but more accurate)
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for images in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            
            if use_tta:
                outputs = predict_with_tta(model, images, device, S=grid_size)
            else:
                outputs = model(images)
            
            # Process each image in batch
            for i in range(outputs.shape[0]):
                bboxes = get_bboxes_from_grid_pred_improved(
                    outputs[i], 
                    S=grid_size, 
                    conf_thresh=0.25,  # Lower threshold to catch more gates
                    nms_thresh=0.4
                )
                predictions.append(bboxes)

    # Format predictions into submission string format
    prediction_strings = []
    for bboxes in predictions:
        pred_str = ''
        for bbox in bboxes:
            x, y, w, h = bbox
            # Convert to corner format
            x1, y1 = x, y
            x2, y2 = x + w, y
            x3, y3 = x + w, y + h
            x4, y4 = x, y + h
            
            # Add confidence score (2.0 as default, can be adjusted)
            pred_str += f'{x1:.6f} {y1:.6f} 2.0 {x2:.6f} {y2:.6f} 2.0 {x3:.6f} {y3:.6f} 2.0 {x4:.6f} {y4:.6f} 2.0 '
        
        prediction_strings.append(pred_str.strip())

    # Statistics
    num_detections = sum(1 for ps in prediction_strings if ps)
    print(f"\nDetected gates in {num_detections}/{len(predictions)} images")

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'Id': range(len(prediction_strings)),
        'PredictionString': prediction_strings
    })

    # Save submission file
    submission_df.to_csv('improved_submission.csv', index=False)

    print("\n" + "="*60)
    print("✅ Submission file 'improved_submission.csv' created successfully!")
    print("="*60)
    
    # Show sample predictions
    print("\nSample predictions (first 5):")
    for i in range(min(5, len(prediction_strings))):
        num_boxes = len(predictions[i])
        print(f"  Image {i}: {num_boxes} gate(s) detected")

if __name__ == '__main__':
    main()