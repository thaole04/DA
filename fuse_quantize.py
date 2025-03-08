import torch
import torch.ao.quantization as quant
from torch.utils.data import DataLoader
from train import YoloNoAnchor

# Hàm fuse các layer của model
def fuse_model(model):
    # Fuse các block trong Stage 1
    quant.fuse_modules(model.stage1_conv1, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv2, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv3, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv4, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv5, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv6, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv7, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv8, ['0', '1'], inplace=True)
    # Fuse các block trong Stage 2a
    quant.fuse_modules(model.stage2_a_conv1, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage2_a_conv2, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage2_a_conv3, ['0', '1'], inplace=True)

# Main: Tải mô hình, fuse, calibrate (nếu có calibration dataset) và quantize
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tải mô hình đã được train (file yolo_no_anchor_model.pth)
    model = YoloNoAnchor(num_classes=1).to(device)
    model.load_state_dict(torch.load("yolo_no_anchor_model.pth", map_location=device))
    model.eval()

    # Fuse các layer: Conv + BatchNorm + LeakyReLU
    fuse_model(model)
    print("Model fused!")

    # Đặt cấu hình lượng tử hóa: dùng qconfig mặc định cho fbgemm (tối ưu cho CPU)
    model.qconfig = quant.get_default_qconfig("fbgemm")
    # Prepare mô hình cho lượng tử hóa (PTQ)
    quant.prepare(model, inplace=True)
    
    # (Tùy chọn) Calibration: Chạy qua một số batch để thu thập thống kê
    # Nếu có dataset calibration, bạn có thể load một vài batch như sau:
    # calibration_dataloader = DataLoader(calibration_dataset, batch_size=16, shuffle=True)
    # with torch.no_grad():
    #     for images, _ in calibration_dataloader:
    #         images = images.to(device)
    #         model(images)
    
    # Chuyển mô hình sang dạng lượng tử hóa
    quant.convert(model, inplace=True)
    print("Model quantized!")

    # Lưu mô hình lượng tử hóa
    torch.save(model.state_dict(), "yolo_no_anchor_quantized.pth")
    print("Quantized model saved as 'yolo_no_anchor_quantized.pth'.")

if __name__ == "__main__":
    main()

