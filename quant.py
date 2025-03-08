import os
import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
from PIL import Image

# Đặt engine cho quantization (dành cho CPU)
torch.backends.quantized.engine = 'fbgemm'

############################################
# Model: YoloNoAnchor (không dùng anchor box)
############################################
class YoloNoAnchor(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchor, self).__init__()
        self.num_classes = num_classes
        # Stage 1
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.stage1_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.stage1_conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv5 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.stage1_conv6 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv8 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2)
        )
        # Stage 2a
        self.stage2_a_maxpl = nn.MaxPool2d(2,2)
        self.stage2_a_conv1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage2_a_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage2_a_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Output layer: Dự đoán [objectness, x, y, w, h, class score]
        self.output_conv = nn.Conv2d(256, (5 + num_classes), 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.stage1_conv1(x)
        x = self.stage1_conv2(x)
        x = self.stage1_conv3(x)
        x = self.stage1_conv4(x)
        x = self.stage1_conv5(x)
        x = self.stage1_conv6(x)
        x = self.stage1_conv7(x)
        x = self.stage1_conv8(x)
        x = self.stage2_a_maxpl(x)
        x = self.stage2_a_conv1(x)
        x = self.stage2_a_conv2(x)
        x = self.stage2_a_conv3(x)
        x = self.output_conv(x)
        return x

############################################
# Hàm fuse modules: fuse các cặp Conv+BN trong các Sequential
############################################
def fuse_model(model):
    # Chỉ fuse các module Conv2d và BatchNorm2d trong các Sequential
    torch.quantization.fuse_modules(model.stage1_conv1, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv2, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv3, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv4, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv5, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv6, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv7, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv8, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage2_a_conv1, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage2_a_conv2, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage2_a_conv3, ['0', '1'], inplace=True)
    return model

############################################
# Main: Quantize model và lưu file weight quantized
############################################
def main():
    # Khởi tạo model
    model = YoloNoAnchor(num_classes=1)
    
    # Nếu có file trọng số đã huấn luyện, load vào model
    weight_path = "yolo_no_anchor_model.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        print("Loaded pretrained model weights from", weight_path)
    else:
        print("Pretrained weight file not found. Using randomly initialized model.")

    # Chuyển model sang chế độ eval trước khi fuse
    model.eval()
    
    # Fuse các module (Conv2d + BatchNorm2d) – bắt buộc khi thực hiện static quantization
    model = fuse_model(model)
    
    # Đặt cấu hình qconfig cho static quantization (sử dụng 'fbgemm' cho CPU)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print("Model qconfig:", model.qconfig)
    
    # Prepare model cho static quantization
    model_prepared = torch.quantization.prepare(model, inplace=False)
    
    # Calibration: chạy một vài forward pass với dummy input (nên dùng tập dữ liệu calibration thực tế)
    dummy_input = torch.randn(1, 3, 256, 256)
    for _ in range(10):
        model_prepared(dummy_input)
    
    # Convert model sang dạng quantized (int8)
    model_int8 = torch.quantization.convert(model_prepared, inplace=False)
    
    # Kiểm tra model_int8 ở chế độ eval
    model_int8.eval()
    
    # Lưu model quantized (state_dict)
    quantized_weight_path = "yolo_no_anchor_model_quantized.pth"
    torch.save(model_int8.state_dict(), quantized_weight_path)
    print("Quantized model saved as", quantized_weight_path)

if __name__ == "__main__":
    main()
