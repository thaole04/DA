import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

######################################
# Model: YoloNoAnchorLite (Light version)
######################################
class YoloNoAnchorLite(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchorLite, self).__init__()
        self.num_classes = num_classes
        
        # Block 1: Input 256x256 -> 128x128, channels: 3 -> 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 2: 128x128 -> 64x64, channels: 16 -> 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 3: Giữ kích thước 64x64, channels: 32 -> 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Block 4: 64x64 -> 32x32, channels: 64 -> 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Output layer: dự đoán [objectness, x, y, w, h, class score]
        self.out_conv = nn.Conv2d(128, (5 + num_classes), kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        x = self.conv1(x)   # 256 -> 128
        x = self.conv2(x)   # 128 -> 64
        x = self.conv3(x)   # kích thước giữ nguyên 64
        x = self.conv4(x)   # 64 -> 32
        x = self.out_conv(x)
        return x

######################################
# Dataset: Đọc ảnh và nhãn theo định dạng YOLO
######################################
class YOLODataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Đường dẫn tới thư mục chứa "images" và "labels".
            transform (callable, optional): Các phép biến đổi được áp dụng cho ảnh.
        """
        self.image_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        base_name = os.path.basename(img_path).replace(".jpg", ".txt")
        label_path = os.path.join(self.label_dir, base_name)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    # Dòng label: class x_center y_center width height (đã chuẩn hóa)
                    if len(parts) == 5:
                        cls, x_center, y_center, w, h = parts
                        boxes.append([float(cls), float(x_center), float(y_center), float(w), float(h)])
        # Nếu không có nhãn, trả về tensor rỗng
        target = torch.tensor(boxes) if boxes else torch.empty((0, 5))
        return image, target

######################################
# Hàm loss cho YOLO (phiên bản đơn giản)
######################################
def compute_yolo_loss(predictions, targets, grid_size=32, lambda_coord=5, lambda_noobj=0.5):
    """
    predictions: tensor shape (B, 5+num_classes, S, S) với S = grid_size (ở đây giả định S=8)
    targets: danh sách length=B, mỗi phần tử là tensor shape (num_boxes, 5) với định dạng:
             [class, x_center, y_center, width, height] (các giá trị đã được chuẩn hóa)
    """
    B = predictions.size(0)
    device = predictions.device
    num_classes = predictions.size(1) - 5

    # Tạo target cho objectness, bounding box và class theo kích thước của grid
    target_obj = torch.zeros((B, grid_size, grid_size), device=device)
    target_bbox = torch.zeros((B, 4, grid_size, grid_size), device=device)
    target_class = torch.zeros((B, grid_size, grid_size), device=device)
    
    # Giả sử mỗi ảnh chỉ có 1 đối tượng
    for i in range(B):
        if targets[i].numel() == 0:
            continue
        gt = targets[i][0]  # [cls, x, y, w, h]
        gt_class = 1  # Vì chỉ có 1 lớp, target class = 1
        gt_x, gt_y, gt_w, gt_h = gt[1], gt[2], gt[3], gt[4]
        # Xác định grid cell chứa trung tâm của box
        cell_j = int(gt_x * grid_size)
        cell_i = int(gt_y * grid_size)
        cell_j = min(cell_j, grid_size - 1)
        cell_i = min(cell_i, grid_size - 1)
        # Tính offset trong cell (trong khoảng [0,1])
        t_x = gt_x * grid_size - cell_j
        t_y = gt_y * grid_size - cell_i
        target_obj[i, cell_i, cell_j] = 1
        target_bbox[i, :, cell_i, cell_j] = torch.tensor([t_x, t_y, gt_w, gt_h], device=device)
        target_class[i, cell_i, cell_j] = gt_class

    # Tách các thành phần dự đoán
    pred_obj   = predictions[:, 0, :, :]      # objectness
    pred_bbox  = predictions[:, 1:5, :, :]      # bbox: x, y, w, h
    pred_class = predictions[:, 5, :, :]        # class score

    # Objectness loss
    bce_loss = F.binary_cross_entropy(torch.sigmoid(pred_obj), target_obj, reduction='none')
    obj_mask = target_obj == 1
    noobj_mask = target_obj == 0
    loss_obj = bce_loss[obj_mask].sum() + lambda_noobj * bce_loss[noobj_mask].sum()

    # Localization loss (chỉ tính cho grid chứa đối tượng)
    mask = obj_mask.unsqueeze(1).float()
    pred_xy = torch.sigmoid(pred_bbox[:, :2, :, :])
    pred_wh = pred_bbox[:, 2:4, :, :]
    target_xy = target_bbox[:, :2, :, :]
    target_wh = target_bbox[:, 2:4, :, :]
    loss_loc = F.mse_loss(pred_xy * mask, target_xy * mask, reduction='sum')
    loss_loc += F.mse_loss(pred_wh * mask, target_wh * mask, reduction='sum')

    # Classification loss (với 1 lớp, dùng BCE)
    loss_class = F.binary_cross_entropy(torch.sigmoid(pred_class)[obj_mask],
                                        target_class[obj_mask], reduction='sum')
    
    total_loss = lambda_coord * loss_loc + loss_obj + loss_class
    return total_loss

######################################
# Main: Training và Validation
######################################
def main():
    root_dir = "train_20000_256"  # Thư mục chứa 'images' và 'labels'
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Tạo dataset và chia thành train và validation
    dataset = YOLODataset(root_dir, transform=transform)
    total_images = len(dataset)
    valid_size = 1000 if total_images >= 1000 else int(total_images * 0.2)
    train_size = total_images - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloNoAnchorLite(num_classes=1).to(device)
    
    # Nếu có file trọng số huấn luyện, load trọng số
    weight_path = "yolo_no_anchor_lite_model.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Loaded pretrained weights from", weight_path)
    else:
        print("Pretrained weight file not found. Using random initialization.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            # targets là danh sách các tensor, mỗi tensor shape (num_boxes, 5)
            targets = list(targets)
            optimizer.zero_grad()
            outputs = model(images)
            loss = compute_yolo_loss(outputs, targets, grid_size=32)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_loss:.4f}")
        
        # Validation sau mỗi epoch
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, targets in valid_loader:
                images = images.to(device)
                targets = list(targets)
                outputs = model(images)
                loss = compute_yolo_loss(outputs, targets, grid_size=32)
                valid_loss += loss.item()
            avg_valid_loss = valid_loss / len(valid_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_valid_loss:.4f}")

    # Lưu model sau training
    torch.save(model.state_dict(), "yolo_no_anchor_model_lite.pth")
    print("Training complete. Model saved as 'yolo_no_anchor_model_lite.pth'.")

if __name__ == "__main__":
    main()
