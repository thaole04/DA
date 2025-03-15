import torch
import torch.nn as nn
import torch.quantization
import numpy as np
from quantized_model import YoloNoAnchorQuantized  # File chứa model quantized của bạn

##########################################
# Phần 1: Tính toán output block1 với PyTorch
##########################################
def get_first_layer_output_quant_int(model, x):
    # Chú ý: Ở đây chúng ta không gọi dequant, nên output vẫn ở dạng quantized.
    x = model.quant(x)
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu1(x)
    x = model.pool1(x)
    return x  # output dạng quantized

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Khởi tạo và convert model quantized
quant_model = YoloNoAnchorQuantized(num_classes=1)
quant_model.eval()
quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# Fuse conv1, bn1, relu1
quant_model = torch.quantization.fuse_modules(quant_model, [["conv1", "bn1", "relu1"]])
quant_model = torch.quantization.prepare(quant_model)
quant_model = torch.quantization.convert(quant_model)
quant_model.load_state_dict(torch.load("yolo_no_anchor_quantized.pth", map_location="cpu"), strict=False)

import cv2
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
        transforms.ToTensor(),
])
frame = cv2.imread("test.jpg")
frame_resized = cv2.resize(frame, (256, 256))
frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(frame_rgb)
input_tensor = transform(pil_img).unsqueeze(0)

# Lấy output của block1 ở dạng quantized (vẫn ở dạng quantized, chưa dequantize)
output_quant_torch = get_first_layer_output_quant_int(quant_model, input_tensor)
# Lấy giá trị int8 (sử dụng int_repr)
output_quant_torch_int = output_quant_torch.int_repr().detach().cpu().numpy()

print("Output block1 từ PyTorch (int8):")
print(output_quant_torch_int)
print("Shape:", output_quant_torch_int.shape)