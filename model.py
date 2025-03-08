import torch
import torch.nn as nn

class YoloNoAnchorLite(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchorLite, self).__init__()
        self.num_classes = num_classes

        # Block 1: Xuất (3 -> 16)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # Kích thước giảm: 256 -> 128
        )
        # Block 2: (16 -> 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )
        # Block 3: (32 -> 64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
            # Không pool để giữ lại thông tin không quá mất đi độ phân giải
        )
        # Block 4: (64 -> 128)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )
        # Lớp output: Dự đoán 6 kênh: objectness, x, y, w, h, class score
        self.out_conv = nn.Conv2d(128, (5 + num_classes), kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)   # kích thước: 256 -> 128
        x = self.conv2(x)   # 128 -> 64
        x = self.conv3(x)   # kích thước giữ nguyên: 64
        x = self.conv4(x)   # 64 -> 32
        x = self.out_conv(x)
        return x

if __name__ == "__main__":
    # Test model với dummy input
    model = YoloNoAnchorLite(num_classes=1)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print("Output shape:", output.shape)
