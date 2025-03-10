import cv2
import torch
import os
import random
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchsummary import summary

# -----------------------------
# Model: YOLO không sử dụng anchor box
# -----------------------------
from model import YoloNoAnchor


# -----------------------------
# Hàm decode output từ model
# -----------------------------
def decode_predictions(predictions, conf_threshold=0.3, grid_size=8, img_size=256):
    """
    predictions: tensor shape (1, 6, grid_size, grid_size)
    Trả về danh sách các box dạng (x1, y1, x2, y2, confidence)
    
    Cách decode:
      - Với mỗi grid cell, x và y được dự đoán qua hàm sigmoid (offset trong cell)
      - Chuyển offset và chỉ số cell thành tọa độ trung tâm tuyệt đối.
      - w, h được dự đoán trực tiếp (giả sử các giá trị đã được huấn luyện ổn định, nếu cần có thể áp dụng sigmoid hoặc clamp).
      - Chuyển từ tọa độ trung tâm và kích thước thành góc trên bên trái và góc dưới bên phải.
    """
    preds = predictions[0]  # shape: (6, grid_size, grid_size)
    boxes = []
    obj = torch.sigmoid(preds[0])  # objectness score
    for i in range(grid_size):
        for j in range(grid_size):
            conf = obj[i, j].item()
            if conf > conf_threshold:
                # Dự đoán offset (x, y) trong cell, áp dụng sigmoid
                tx = torch.sigmoid(preds[1, i, j]).item()
                ty = torch.sigmoid(preds[2, i, j]).item()
                # Dự đoán w, h (có thể cần clamp nếu âm)
                tw = preds[3, i, j].item()
                th = preds[4, i, j].item()
                tw = max(tw, 0)
                th = max(th, 0)
                
                cell_size = img_size / grid_size  # ví dụ: 256/8 = 32
                # Tọa độ trung tâm tuyệt đối:
                cx = (j + tx) * cell_size
                cy = (i + ty) * cell_size
                # Chuyển w, h từ giá trị chuẩn hóa thành kích thước tuyệt đối:
                box_w = tw * img_size
                box_h = th * img_size
                # Tính tọa độ box: chuyển từ trung tâm sang góc trái và phải
                x1 = int(cx - box_w / 2)
                y1 = int(cy - box_h / 2)
                x2 = int(cx + box_w / 2)
                y2 = int(cy + box_h / 2)
                boxes.append((x1, y1, x2, y2, conf))
    return boxes

# -----------------------------
# Main: Test model sử dụng webcam
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloNoAnchor(num_classes=1).to(device)
    
    # Load weights đã train
    weight_path = "yolo_no_anchor_model.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Hiển thị cấu trúc model
    summary(model, (3, 256, 256))
    # Hiển thị thông tin số lượng tham số
    print(f"Số lượng tham số: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Đọc danh sách ảnh từ thư mục "images"
    images_folder = "test_1"  # thay đổi nếu cần
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
