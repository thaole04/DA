import torch
import torch.nn as nn
import torch.quantization as quant
from torchsummary import summary

class YoloNoAnchorQuantized(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchorQuantized, self).__init__()
        self.num_classes = num_classes
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

        # --- Stage 1 ---
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1, bias=False),  # giảm từ 16 -> 8
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )
        self.stage1_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1, bias=False),  # giảm từ 32 -> 16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )
        self.stage1_conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),  # giảm từ 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=False)
        )
        self.stage1_conv4 = nn.Sequential(
            nn.Conv2d(32, 16, 1, 1, 0, bias=False),  # giảm từ 64 -> 32, sau đó giảm xuống 16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=False)
        )
        self.stage1_conv5 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),  # giảm từ 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )
        self.stage1_conv6 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # giảm từ 128 -> 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=False)
        )
        self.stage1_conv7 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0, bias=False),  # giảm từ 128 -> 64, sau đó giảm xuống 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=False)
        )
        self.stage1_conv8 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # giảm từ 128 -> 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )

        # --- Stage 2a (đơn giản hóa) ---
        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)  # 16 -> 8
        self.stage2_a_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),  # giảm từ 256 -> 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=False)
        )
        self.stage2_a_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0, bias=False),  # giảm từ 256 -> 128, sau đó giảm xuống 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=False)
        )
        self.stage2_a_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),  # giảm từ 256 -> 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=False)
        )
        # --- Lớp output ---
        self.output_conv = nn.Conv2d(128, (5 + num_classes), 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.quant(x)
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
        x = self.dequant(x)
        return x
