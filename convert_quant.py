import os
import glob
import torch
import torch.quantization
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import model của bạn (đảm bảo file model.py nằm cùng thư mục)
from quantized_model import YoloNoAnchorQuantized

# Dataset dùng cho calibration (chỉ cần ảnh, không cần label)
class YOLODataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Đường dẫn chứa thư mục con 'images' (với ảnh dùng cho calibration)
            transform (callable, optional): Các phép biến đổi ảnh.
        """
        self.image_dir = os.path.join(root, "images")
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Hàm hiệu chuẩn (calibration) để thu thập thông tin cho observer
def calibrate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            model(images)
    return model

def main():
    # Chỉ sử dụng CPU cho quantization
    device = torch.device("cpu")
    
    # Khởi tạo model và load trạng thái đã train (đảm bảo model đã được huấn luyện)
    model = YoloNoAnchorQuantized(num_classes=1)
    model.load_state_dict(torch.load("yolo_no_anchor_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # --- Bước 1: Fuse các module ---
    fuse_list = [
        ['conv1', 'bn1', 'relu1'],
        ['conv2', 'bn2', 'relu2'],
        ['conv3', 'bn3', 'relu3'],
        ['conv4', 'bn4', 'relu4'],
        ['conv5', 'bn5', 'relu5'],
        ['conv6', 'bn6', 'relu6'],
        ['conv7', 'bn7', 'relu7'],
        ['conv8', 'bn8', 'relu8'],
        ['conv9', 'bn9', 'relu9'],
        ['conv10', 'bn10', 'relu10'],
        ['conv11', 'bn11', 'relu11']
    ]
    torch.quantization.fuse_modules(model, fuse_list, inplace=True)
    
    # --- Bước 2: Cấu hình quantization ---
    model.qconfig = torch.quantization.get_default_qconfig('x86')
    print("QConfig:", model.qconfig)
    torch.quantization.prepare(model, inplace=True)
    
    # --- Bước 3: Calibration ---
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Sử dụng một tập dữ liệu nhỏ cho calibration (ví dụ: thư mục "train_20000_256")
    calib_dataset = YOLODataset(root="train_20000_256", transform=transform)
    calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    print("Đang hiệu chuẩn (calibrate) model...")
    calibrate(model, calib_loader, device)
    
    # --- Bước 4: Chuyển đổi sang model quantized ---
    torch.quantization.convert(model, inplace=True)
    
    # Lưu model quantized
    quantized_model_path = "yolo_no_anchor_model_quantized.pth"
    torch.save(model, quantized_model_path)
    # Lưu weight của model quantized
    quantized_weight_path = "yolo_no_anchor_model_quantized_weight.pth"
    torch.save(model.state_dict(), quantized_weight_path)
    print("Model quantized đã được lưu tại:", quantized_model_path)

if __name__ == "__main__":
    main()
