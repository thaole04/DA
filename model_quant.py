from model import YoloNoAnchor
import torch
import os

if os.name == 'nt':
    # Windows
    torch.backends.quantized.engine = 'fbgemm'
else:
    # Linux
    torch.backends.quantized.engine = 'qnnpack'

class YoloNoAnchorQuantized(YoloNoAnchor):
    def __init__(self, num_classes=1):
        super(YoloNoAnchor, self).__init__()
        self.num_classes = num_classes
        self.quant   = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
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

        return self.dequant(x)
