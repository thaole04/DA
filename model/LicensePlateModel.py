from model.ConvBatchnormReLU import ConvBatchnormReLU
from model.EncoderStage import EncoderStage
import torch

class LicensePlateModel(torch.nn.Module):

    def __init__(self, input_size=(256, 512), num_boxes=1, dropout=True):

        super(LicensePlateModel, self).__init__()

        self.input_size = input_size
        self.output_size = (self.input_size[0] // 8, self.input_size[1] // 8)

        # Encoder stages
        self.encoder_stage_1 = EncoderStage(in_channels=3,  mid_channels=8,  out_channels=16, dropout=dropout)
        self.encoder_stage_2 = EncoderStage(in_channels=16, mid_channels=16, out_channels=32, dropout=dropout)
        self.encoder_stage_3 = EncoderStage(in_channels=32, mid_channels=32, out_channels=64, dropout=dropout)

        # Output branch for bounding box detection
        self.bbox_out = torch.nn.Sequential(
            ConvBatchnormReLU(in_channels=64, out_channels=32, padding=1),
            ConvBatchnormReLU(in_channels=32, out_channels=16, padding=1),
            ConvBatchnormReLU(in_channels=16, out_channels=8,  padding=1),
            torch.nn.Conv2d  (in_channels=8,  out_channels=num_boxes * 4, kernel_size=3, padding=1, bias=True),  # 4 coordinates for each bounding box
        )

    def forward(self, x):
        # Encoder stages
        x1 = self.encoder_stage_1(x)
        x2 = self.encoder_stage_2(x1)
        x3 = self.encoder_stage_3(x2)

        # Output branch for bounding box detection
        bbox = self.bbox_out(x3)

        return bbox