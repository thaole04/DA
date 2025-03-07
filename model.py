import torch
import torch.nn as nn

class Yolo(nn.Module):
    def __init__(self, num_classes,
                 anchors=[(392.89, 166.93),
                        (243.84, 194.15),
                        (170.57, 75.62),
                        (95.37, 49.03),
                        (43.02, 28.53)]):
        super(Yolo, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors

        # Stage 1 - Giảm channels đáng kể
        self.stage1_conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv4 = nn.Sequential(nn.Conv2d(64, 32, 1, 1, 0, bias=False), nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv5 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv6 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv7 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv8 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))

        # Loại bỏ Stage 2b và Stage 3, đơn giản hóa Stage 2a
        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)
        self.stage2_a_conv1 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Giảm channel
                                            nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv2 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128), # Giảm channel
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), # Giảm channel
                                            nn.LeakyReLU(0.1, inplace=True))

        # Output Layer - Chỉ cần 5 + 1 = 6 channels cho mỗi anchor box
        self.output_conv = nn.Conv2d(256, len(self.anchors) * (5 + num_classes), 1, 1, 0, bias=True) # num_classes = 1

    def forward(self, input):
        output = self.stage1_conv1(input)
        output = self.stage1_conv2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)

        # Stage 2a đã được đơn giản hóa
        output = self.stage2_a_maxpl(output)
        output = self.stage2_a_conv1(output)
        output = self.stage2_a_conv2(output)
        output = self.stage2_a_conv3(output)

        # Output Layer
        output = self.output_conv(output)
        return output
    

if __name__ == "__main__":
    net = Yolo(1)