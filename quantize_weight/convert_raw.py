import os
import glob
import torch
import torch.quantization
from torch.quantization import MovingAverageMinMaxObserver, MinMaxObserver, HistogramObserver
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import model của bạn (đảm bảo file model.py nằm cùng thư mục)
from model import YoloNoAnchor

# Dataset dùng cho calibration (chỉ cần ảnh, không cần label)
class YOLODataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Đường dẫn chứa thư mục con 'images' (với ảnh dùng cho calibration)
            transform (callable, optional): Các phép biến đổi ảnh.
        """
        # self.image_dir = os.path.join(root, "images")
        self.image_dir = os.path.join(root)
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
    with torch.no_grad():
        for images in data_loader:
            # images = torch.quantize_per_tensor(images, scale=1/255, zero_point=0, dtype=torch.quint8)
            images = images.float() / 255.0  # Chuyển đổi ảnh về khoảng [0, 1]
            images = images.to(device)
            model(images)
    return model

def main():
    # Chỉ sử dụng CPU cho quantization
    device = torch.device("cpu")
    
    # Khởi tạo model và load trạng thái đã train (đảm bảo model đã được huấn luyện)
    model = YoloNoAnchor(num_classes=1)
    model.to(device)
    model.load_state_dict(torch.load("1e-3/yolo_no_anchor_model.pth", map_location=device))
    model.eval()
    # --- Bước 2: Cấu hình quantization ---
    model.qconfig = torch.quantization.QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric, 
            dtype=torch.qint8
        ),
        weight=MovingAverageMinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric, 
            dtype=torch.qint8
        )
    )
    print("QConfig:", model.qconfig)
    
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
    model_fuse = torch.quantization.fuse_modules(model, fuse_list, inplace=True)
    
    model_prepared = torch.quantization.prepare(model_fuse, inplace=True)
    
    # --- Bước 3: Calibration ---
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Sử dụng một tập dữ liệu nhỏ cho calibration (ví dụ: thư mục "train_20000_256")
    calib_dataset = YOLODataset(root="train_20000_256/images", transform=transform)
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    print("Đang hiệu chuẩn (calibrate) model...")
    # # calib with dummy data
    # for i in range(10):
    #     dummy_input = torch.randn(1, 3, 256, 256).to(device)
    #     model(dummy_input)
    # calib with real data
    # dummy_input = torch.randn(1, 3, 256, 256).to(device)
    # model(dummy_input)
    calibrate(model_prepared, calib_loader, device)
    
    # --- Bước 4: Chuyển đổi sang model quantized ---
    model_prepared.eval()
    model_converted = torch.quantization.convert(model_prepared, inplace=True)
    
    # Lưu model quantized
    quantized_weight_path = "raw_quant.pth"
    torch.save(model_converted.state_dict(), quantized_weight_path)
    print("Model quantized đã được lưu tại:", quantized_weight_path)

if __name__ == "__main__":
    main()
