from quantized_model import YoloNoAnchorQuantized
from torch.quantization.observer import MovingAverageMinMaxObserver
import torch
import torch.quantization as quant
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
# -----------------------------
# Dataset: Đọc ảnh và label (YOLO format)
# -----------------------------
class YOLODataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Đường dẫn chứa thư mục con 'images' và 'labels'.
            transform (callable, optional): Các phép biến đổi ảnh.
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
                    if len(parts) == 5:
                        cls, x_center, y_center, w, h = parts
                        boxes.append([float(cls), float(x_center), float(y_center), float(w), float(h)])
        target = torch.tensor(boxes) if boxes else torch.empty((0, 5))
        return image, target
def convert():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # Tạo dataset và tách thành train/validation
    dataset = YOLODataset('train_20000_256', transform=transform)
    calib_num = int(0.1 * len(dataset))
    train_num = len(dataset) - calib_num
    train_dataset, calib_dataset = random_split(dataset, [train_num, calib_num])
    train_dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)

    # Load model đã train (model gốc không quantize)
    model = YoloNoAnchorQuantized(num_classes=1)
    model.load_state_dict(torch.load("yolo_no_anchor_model.pth", map_location="cpu"), strict=False)
    model.eval()

    # Định nghĩa danh sách các module cần fuse (chỉ fuse Conv và BatchNorm)
    fuse_modules = [
        ["conv1", "bn1", "relu1"],
        ["conv2", "bn2", "relu2"],
        ["conv3", "bn3", "relu3"],
        ["conv4", "bn4", "relu4"],
        ["conv5", "bn5", "relu5"],
        ["conv6", "bn6", "relu6"],
        ["conv7", "bn7", "relu7"],
        ["conv8", "bn8", "relu8"],
        ["conv9", "bn9", "relu9"],
        ["conv10", "bn10", "relu10"],
        ["conv11", "bn11", "relu11"],
    ]

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Fuse các module (Conv + BN) theo danh sách fuse_modules
    model_fused = quant.fuse_modules(model, fuse_modules, inplace=False)

    # Prepare model cho quantization (prepare sẽ thêm observer vào các module đã được fuse)
    model_prepared = quant.prepare(model_fused, inplace=False)

    # Calibration: chạy một vài input qua model (ở đây dùng dummy input)
    for images, _ in calib_loader:
        model_prepared(images)

    # Convert model sang dạng quantized
    model_quantized = quant.convert(model_prepared, inplace=False)
    
    # Lưu state_dict của model đã quantized
    torch.save(model_quantized.state_dict(), "yolo_no_anchor_quantized.pth")
    print("Model đã được quantized và lưu thành công!")

def main():
    print("Quantizing model...")
    convert()

if __name__ == "__main__":
    main()
