import torch
import torch.quantization
import torch.nn as nn

class YoloNoAnchor(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchor, self).__init__()
        self.num_classes = num_classes

        # --- Stage 1 ---
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.stage1_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.stage1_conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv5 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.stage1_conv6 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv8 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # --- Stage 2a (đơn giản hóa) ---
        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)
        self.stage2_a_conv1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage2_a_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage2_a_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # --- Lớp output ---
        # Vì loại bỏ anchor box nên mỗi grid cell dự đoán trực tiếp:
        # [objectness, x, y, w, h, class score] => (5+num_classes) kênh
        self.output_conv = nn.Conv2d(256, (5 + num_classes), 1, 1, 0, bias=True)

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

def fuse_model(model):
    """
    Fuse các module conv+bn trong các Sequential có thể fuse.
    Chúng ta sẽ fuse các module [0, 1] (Conv và BatchNorm) trong các block:
      - stage1_conv1, stage1_conv2, stage1_conv3, stage1_conv4,
        stage1_conv5, stage1_conv6, stage1_conv7, stage1_conv8,
        stage2_a_conv1, stage2_a_conv2, stage2_a_conv3.
    Lưu ý: Các block có LeakyReLU và MaxPool không được fuse thêm.
    """
    # Fuse các stage của phần Stage1
    torch.quantization.fuse_modules(model.stage1_conv1, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv2, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv3, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv4, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv5, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv6, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv7, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage1_conv8, ['0', '1'], inplace=True)
    # Fuse các stage của phần Stage2a
    torch.quantization.fuse_modules(model.stage2_a_conv1, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage2_a_conv2, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(model.stage2_a_conv3, ['0', '1'], inplace=True)
    return model

def calibrate_model(model, calibration_data, num_batches=10):
    """
    Chạy model với một vài batch dữ liệu để hiệu chỉnh (calibrate) scale cho quantization.
    calibration_data: DataLoader hoặc list các tensor input
    """
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(calibration_data):
            # Nếu calibration_data là DataLoader với tuple (images, _)
            if isinstance(data, (list, tuple)):
                images = data[0]
            else:
                images = data
            model(images)
            if i >= num_batches - 1:
                break

def main():
    # Load model đã train
    model = YoloNoAnchor(num_classes=1)
    weight_path = "yolo_no_anchor_model.pth"
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()

    # Gán quantization config (sử dụng 'fbgemm' cho CPU)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print("Quantization config:", model.qconfig)

    # Fuse các module cần thiết
    model = fuse_model(model)

    # Chuẩn bị cho static quantization
    model_prepared = torch.quantization.prepare(model, inplace=False)

    # Calibrate model: ở đây ta dùng dummy input làm ví dụ calibration
    # (Trong thực tế, bạn nên dùng một tập dữ liệu calibration nhỏ)
    dummy_input = torch.randn(1, 3, 256, 256)
    model_prepared(dummy_input)

    # Hoặc nếu bạn có calibration DataLoader, ví dụ:
    # from torch.utils.data import DataLoader
    # calibration_loader = DataLoader(your_calibration_dataset, batch_size=1, shuffle=True)
    # calibrate_model(model_prepared, calibration_loader, num_batches=10)

    # Chuyển model sang quantized version (int8)
    model_int8 = torch.quantization.convert(model_prepared, inplace=False)
    
    # Lưu model quantized
    quantized_weight_path = "yolo_no_anchor_model_int8.pth"
    torch.save(model_int8.state_dict(), quantized_weight_path)
    print("Quantized model saved as", quantized_weight_path)

if __name__ == "__main__":
    main()
