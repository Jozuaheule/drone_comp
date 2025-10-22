import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss, complete_box_iou_loss


class ImprovedGateCNN(nn.Module):
    """Enhanced architecture with deeper network, residual connections, and attention"""
    def __init__(self, grid_size=14, num_anchors=2):
        super(ImprovedGateCNN, self).__init__()
        self.S = grid_size
        self.num_anchors = num_anchors
        
        # Encoder with residual blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Calculate feature map size after pooling
        # Input: 480x640 -> after 5 pooling: 15x20
        self.feature_size = 512 * 15 * 20
        
        # More powerful FC layers with residual connection
        self.fc1 = nn.Linear(self.feature_size, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, self.S * self.S * num_anchors * 5)
        
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.bn_fc2 = nn.BatchNorm1d(2048)
        
    def forward(self, x):
        # Encoder with progressive downsampling
        x = self.pool(self.conv1(x))  # 240x320
        x = self.pool(self.conv2(x))  # 120x160
        x = self.pool(self.conv3(x))  # 60x80
        x = self.pool(self.conv4(x))  # 30x40
        x = self.pool(self.conv5(x))  # 15x20
        
        # Apply spatial attention
        attention = self.spatial_attention(x)
        x = x * attention
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers with dropout and batch norm
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Reshape to grid format: [batch, grid_y, grid_x, num_anchors, 5]
        return x.reshape(-1, self.S, self.S, self.num_anchors, 5)


class EnhancedLoss(nn.Module):
    """Improved loss function with better balance and IoU handling"""
    def __init__(self, grid_size=14, lambda_coord=5.0, lambda_noobj=0.5, 
                 lambda_obj=1.0, lambda_size=2.0):
        super(EnhancedLoss, self).__init__()
        self.S = grid_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_size = lambda_size

    def forward(self, predictions, target):
        """
        predictions: [batch, S, S, num_anchors, 5]
        target: [batch, S, S, num_anchors, 5]
        """
        # Flatten anchors into grid for easier processing
        batch_size = predictions.size(0)
        num_anchors = predictions.size(3)
        
        # Reshape to [batch, S, S*num_anchors, 5]
        pred_flat = predictions.view(batch_size, self.S, -1, 5)
        target_flat = target.view(batch_size, self.S, -1, 5)
        
        # Object mask: where target confidence = 1
        exists_box = target_flat[..., 0] == 1
        no_exists_box = target_flat[..., 0] == 0
        
        # 1. Objectness loss (Focal Loss for class imbalance)
        if exists_box.sum() > 0:
            obj_preds = predictions[exists_box]
            obj_targets = target[exists_box]
            obj_loss = sigmoid_focal_loss(
                obj_preds[:, 0], 
                obj_targets[:, 0], 
                alpha=0.25,
                gamma=2.0,
                reduction='mean'
            )
        else:
            obj_loss = torch.tensor(0.0, device=predictions.device)
        
        # 2. No-object loss
        if no_exists_box.sum() > 0:
            noobj_preds = predictions[no_exists_box]
            noobj_targets = target[no_exists_box]
            noobj_loss = sigmoid_focal_loss(
                noobj_preds[:, 0],
                noobj_targets[:, 0],
                alpha=0.75,
                gamma=2.0,
                reduction='mean'
            )
        else:
            noobj_loss = torch.tensor(0.0, device=predictions.device)
        
        # 3. Coordinate loss (only for boxes that exist)
        if exists_box.sum() > 0:
            box_preds = predictions[exists_box][:, 1:5]
            box_targets = target[exists_box][:, 1:5]
            
            # Ensure positive width/height
            w_pred = torch.abs(box_preds[..., 2]) + 1e-6
            h_pred = torch.abs(box_preds[..., 3]) + 1e-6
            x_pred = box_preds[..., 0]
            y_pred = box_preds[..., 1]
            
            w_targ = box_targets[..., 2] + 1e-6
            h_targ = box_targets[..., 3] + 1e-6
            x_targ = box_targets[..., 0]
            y_targ = box_targets[..., 1]
            
            # Convert to x1y1x2y2 format for CIoU
            pred_x1 = x_pred - w_pred / 2
            pred_y1 = y_pred - h_pred / 2
            pred_x2 = x_pred + w_pred / 2
            pred_y2 = y_pred + h_pred / 2
            
            targ_x1 = x_targ - w_targ / 2
            targ_y1 = y_targ - h_targ / 2
            targ_x2 = x_targ + w_targ / 2
            targ_y2 = y_targ + h_targ / 2
            
            pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
            targ_boxes = torch.stack([targ_x1, targ_y1, targ_x2, targ_y2], dim=-1)
            
            # CIoU loss
            ciou_loss = complete_box_iou_loss(pred_boxes, targ_boxes, reduction='mean')
            
            # Additional size loss (encourage correct width/height prediction)
            size_loss = F.mse_loss(torch.sqrt(w_pred), torch.sqrt(w_targ)) + \
                       F.mse_loss(torch.sqrt(h_pred), torch.sqrt(h_targ))
            
            # Center coordinate loss
            center_loss = F.mse_loss(x_pred, x_targ) + F.mse_loss(y_pred, y_targ)
        else:
            ciou_loss = torch.tensor(0.0, device=predictions.device)
            size_loss = torch.tensor(0.0, device=predictions.device)
            center_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combine all losses
        total_loss = (
            self.lambda_obj * obj_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_coord * (ciou_loss + center_loss) +
            self.lambda_size * size_loss
        )
        
        return total_loss, {
            'obj_loss': obj_loss.item(),
            'noobj_loss': noobj_loss.item(),
            'ciou_loss': ciou_loss.item(),
            'size_loss': size_loss.item(),
            'total_loss': total_loss.item()
        }


def get_bboxes_from_grid_pred_improved(grid_preds, S=14, conf_thresh=0.3, nms_thresh=0.4):
    """
    Improved bbox extraction with:
    - Multiple anchor support
    - NMS to remove duplicates
    - Confidence thresholding
    """
    pred_bboxes = []
    
    # Handle multiple anchors
    num_anchors = grid_preds.shape[2] if len(grid_preds.shape) == 4 else 1
    
    for i in range(S):
        for j in range(S):
            for a in range(num_anchors):
                if num_anchors > 1:
                    conf = torch.sigmoid(grid_preds[i, j, a, 0])
                    if conf > conf_thresh:
                        xc, yc, w, h = grid_preds[i, j, a, 1:]
                else:
                    conf = torch.sigmoid(grid_preds[i, j, 0])
                    if conf > conf_thresh:
                        xc, yc, w, h = grid_preds[i, j, 1:]
                
                # Convert to absolute coordinates
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
    if len(pred_bboxes) > 0:
        pred_bboxes = torch.tensor(pred_bboxes)
        boxes = pred_bboxes[:, :4]
        scores = pred_bboxes[:, 4]
        
        # Simple NMS
        keep_indices = torchvision.ops.nms(boxes, scores, nms_thresh)
        pred_bboxes = pred_bboxes[keep_indices]
        
        # Convert back to xywh format
        final_boxes = []
        for box in pred_bboxes:
            x1, y1, x2, y2, conf = box
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            final_boxes.append([x.item(), y.item(), w.item(), h.item()])
        
        return final_boxes
    
    return []
