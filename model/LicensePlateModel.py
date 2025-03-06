from model.ConvBatchnormReLU import ConvBatchnormReLU
from model.EncoderStage import EncoderStage
import torch

class LicensePlateModel(torch.nn.Module):
    def __init__(self, input_size=(256, 512), dropout=True):
        super(LicensePlateModel, self).__init__()

        self.input_size = input_size
        self.output_size = (self.input_size[0] // 8, self.input_size[1] // 8)

        # Encoder stages
        self.encoder_stage_1 = EncoderStage(in_channels=3,  mid_channels=8,  out_channels=16, dropout=dropout)
        self.encoder_stage_2 = EncoderStage(in_channels=16, mid_channels=16, out_channels=32, dropout=dropout)
        self.encoder_stage_3 = EncoderStage(in_channels=32, mid_channels=32, out_channels=64, dropout=dropout)

        # Output branch for bounding box detection (chỉ 1 bounding box)
        # Sử dụng AdaptiveAvgPool2d để giảm kích thước feature map về 1x1,
        # sau đó dùng Linear layer để dự đoán 4 giá trị: [x, y, width, height].
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, 4)  # Mỗi ảnh dự đoán 4 giá trị bounding box

    def forward(self, x):
        # Encoder stages
        x1 = self.encoder_stage_1(x)
        x2 = self.encoder_stage_2(x1)
        x3 = self.encoder_stage_3(x2)

        # Output branch cho bounding box detection
        x_pool = self.avgpool(x3)              # Kích thước: (B, 64, 1, 1)
        x_flat = x_pool.view(x_pool.size(0), -1) # Flatten thành (B, 64)
        bbox = self.fc(x_flat)                 # Dự đoán bounding box, kích thước: (B, 4)

        return bbox
