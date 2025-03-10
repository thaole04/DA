import os
import random
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.quantization as quant
from quantized_model import YoloNoAnchorQuantized

# -----------------------------
# Hàm decode output từ model
# -----------------------------
def decode_predictions(predictions, conf_threshold=0.3, grid_size=8, img_size=256):
    """
    predictions: tensor shape (1, 6, grid_size, grid_size)
    Trả về danh sách các box dạng (x1, y1, x2, y2, confidence)
    """
    preds = predictions[0]  # shape: (6, grid_size, grid_size)
    boxes = []
    obj = torch.sigmoid(preds[0])  # objectness score
    for i in range(grid_size):
        for j in range(grid_size):
            conf = obj[i, j].item()
            if conf > conf_threshold:
                tx = torch.sigmoid(preds[1, i, j]).item()
                ty = torch.sigmoid(preds[2, i, j]).item()
                tw = preds[3, i, j].item()
                th = preds[4, i, j].item()
                tw = max(tw, 0)
                th = max(th, 0)
                
                cell_size = img_size / grid_size  # ví dụ: 256/8 = 32
                cx = (j + tx) * cell_size
                cy = (i + ty) * cell_size
                box_w = tw * img_size
                box_h = th * img_size
                x1 = int(cx - box_w / 2)
                y1 = int(cy - box_h / 2)
                x2 = int(cx + box_w / 2)
                y2 = int(cy + box_h / 2)
                boxes.append((x1, y1, x2, y2, conf))
    return boxes

# -----------------------------
# Main: Test model sử dụng ảnh từ thư mục
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo model wrapper đã tích hợp QuantStub/DeQuantStub
    model = YoloNoAnchorQuantized(num_classes=1).to(device)
    model.eval()
    
    # Danh sách các module cần fuse (chỉ fuse Conv và BatchNorm)
    fuse_modules = [
        ["conv1", "bn1", "relu1"],
        ["conv2", "bn2", "relu2"],
        ["conv3", "bn3", "relu3"],
        ["conv4", "bn4", "relu4"],
        ["conv5", "bn5", "relu5"],
        ["conv6", "bn6", "relu6"],
        ["conv7", "bn7", "relu7"],
        ["conv8", "bn8", "relu8"],
        ["conv9", "bn9", "relu9"],
        ["conv10", "bn10", "relu10"],
        ["conv11", "bn11", "relu11"],
    ]
    
    # Đặt qconfig sử dụng MinMaxObserver theo dạng symmetric int8
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Fuse các module
    model = quant.fuse_modules(model, fuse_modules, inplace=False)
    
    # Prepare và convert model sang dạng quantized
    model = quant.prepare(model, inplace=False)
    # Chạy một vài forward pass dummy để calibration (nếu cần)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    model = quant.convert(model, inplace=False)
    
    # Load trọng số đã quantized đã được lưu từ quá trình quantize
    weight_path = "yolo_no_anchor_quantized.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    
    # Thiết lập pipeline tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Đọc danh sách ảnh từ thư mục "images"
    images_folder = "test_0"  # thay đổi nếu cần
    image_paths = [os.path.join(images_folder, f) for f in os.listdir(images_folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_paths:
        print("Không tìm thấy ảnh trong thư mục", images_folder)
        return
    
    print("Nhấn 'n' để load ảnh ngẫu nhiên, nhấn 'q' để thoát.")
    
    while True:
        # Lấy ảnh ngẫu nhiên
        img_path = random.choice(image_paths)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        boxes = decode_predictions(outputs, conf_threshold=0.5, grid_size=8, img_size=256)
        if boxes:
            best_box = max(boxes, key=lambda x: x[4])
            x1, y1, x2, y2, conf = best_box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"Conf: {conf:.2f}", (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("YOLO No Anchor Detection", frame_resized)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            continue
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
