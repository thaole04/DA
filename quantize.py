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
    total_images = len(dataset)
    calib_size = int(0.1 * total_images)
    train_size = total_images - calib_size
    train_dataset, calib_dataset = random_split(dataset, [train_size, calib_size])
    calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Load model đã train (model gốc không quantize)
    model = YoloNoAnchorQuantized(num_classes=1)
    model.load_state_dict(torch.load("yolo_no_anchor_model.pth", map_location="cpu"), strict=False)
    model.eval()

    # Định nghĩa danh sách các module cần fuse (chỉ fuse Conv và BatchNorm)
    fuse_modules = [
        ["stage1_conv1.0", "stage1_conv1.1"],
        ["stage1_conv2.0", "stage1_conv2.1"],
        ["stage1_conv3.0", "stage1_conv3.1"],
        ["stage1_conv4.0", "stage1_conv4.1"],
        ["stage1_conv5.0", "stage1_conv5.1"],
        ["stage1_conv6.0", "stage1_conv6.1"],
        ["stage1_conv7.0", "stage1_conv7.1"],
        ["stage1_conv8.0", "stage1_conv8.1"],
        ["stage2_a_conv1.0", "stage2_a_conv1.1"],
        ["stage2_a_conv2.0", "stage2_a_conv2.1"],
        ["stage2_a_conv3.0", "stage2_a_conv3.1"]
    ]

    model.qconfig = quant.get_default_qconfig('fbgemm')

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
