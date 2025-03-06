import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Import model LicensePlateModel (đảm bảo đường dẫn đúng)
from model.LicensePlateModel import LicensePlateModel

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Thư mục chứa 2 folder 'images' và 'labels'.
            transform (callable, optional): Transform cho ảnh.
        """
        self.images_dir = os.path.join(root_dir, "images")
        self.labels_dir = os.path.join(root_dir, "labels")
        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load ảnh
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Đọc file label với cùng tên ảnh (định dạng YOLO: class x_center y_center width height)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, base_name + ".txt")
        with open(label_path, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"File {label_path} không có đúng 5 phần tử (class x_center y_center width height)")
            # Ở đây ta bỏ qua class (parts[0]) và chỉ lấy các giá trị bounding box
            bbox = np.array(parts[1:], dtype=np.float32)
        
        bbox = torch.tensor(bbox, dtype=torch.float32)
        return image, bbox

def train_model():
    # Tham số huấn luyện
    num_epochs = 20
    batch_size = 16
    learning_rate = 1e-3
    train_dir = "train"  # thư mục chứa images/ và labels/

    # Kiểm tra device: GPU nếu có, ngược lại dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transform: Resize (nếu cần) và chuyển thành tensor
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
    ])
    
    # Dataset và DataLoader
    dataset = LicensePlateDataset(root_dir=train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Khởi tạo model (model dự đoán 1 bounding box với 4 giá trị)
    model = LicensePlateModel(input_size=(256, 512), dropout=True)
    model = model.to(device)

    # Loss và Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # outputs có kích thước (B, 4)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    print("Training complete.")
    # Lưu model sau khi training
    torch.save(model.state_dict(), "license_plate_model.pth")

if __name__ == "__main__":
    train_model()
