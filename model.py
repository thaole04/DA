import torch
import torch.nn as nn
from torchsummary import summary

class YoloNoAnchor(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchor, self).__init__()
        self.num_classes = num_classes

        # --- Stage 1 ---
        # Block 1:
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2:
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3:
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Block 4:
        self.conv4 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4   = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Block 5:
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5   = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 6:
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6   = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(inplace=True)
        
        # Block 7:
        self.conv7 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7   = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU(inplace=True)
        
        # Block 8:
        self.conv8 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8   = nn.BatchNorm2d(64)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Stage 2a (đơn giản hóa) ---
        self.pool_stage2a = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 9:
        self.conv9 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9   = nn.BatchNorm2d(128)
        self.relu9 = nn.ReLU(inplace=True)
        
        # Conv block 10:
        self.conv10 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn10   = nn.BatchNorm2d(64)
        self.relu10 = nn.ReLU(inplace=True)
        
        # Conv block 11:
        self.conv11 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11   = nn.BatchNorm2d(128)
        self.relu11 = nn.ReLU(inplace=True)
        
        # --- Lớp output ---
        self.out_conv = nn.Conv2d(128, 5 + num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # Stage 1, Block 1:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2:
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Block 3:
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Block 4:
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # Block 5:
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        # Block 6:
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        
        # Block 7:
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        
        # Block 8:
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.pool8(x)
        
        # Stage 2a:
        x = self.pool_stage2a(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        
        # Lớp output:
        x = self.out_conv(x)
        return x

if __name__ == '__main__':
    model = YoloNoAnchor()
    summary(model, (3, 256, 256))
