import torch
import torch.nn as nn

# -----------------------------
# Model: YOLO không sử dụng anchor box
# -----------------------------
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

if __name__ == "__main__":
    net = Yolo(1)
