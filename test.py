import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchsummary import summary

# -----------------------------
# Model: YOLO không sử dụng anchor box
# -----------------------------
class YoloNoAnchor(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchor, self).__init__()
        self.num_classes = num_classes

        # --- Stage 1 ---
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1, bias=False),  # giảm từ 16 -> 8
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )
        self.stage1_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1, bias=False),  # giảm từ 32 -> 16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )
        self.stage1_conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),  # giảm từ 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv4 = nn.Sequential(
            nn.Conv2d(32, 16, 1, 1, 0, bias=False),  # giảm từ 64 -> 32, sau đó giảm xuống 16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv5 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),  # giảm từ 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )
        self.stage1_conv6 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # giảm từ 128 -> 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv7 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0, bias=False),  # giảm từ 128 -> 64, sau đó giảm xuống 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv8 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # giảm từ 128 -> 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )

        # --- Stage 2a (đơn giản hóa) ---
        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)  # 16 -> 8
        self.stage2_a_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),  # giảm từ 256 -> 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage2_a_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0, bias=False),  # giảm từ 256 -> 128, sau đó giảm xuống 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage2_a_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),  # giảm từ 256 -> 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # --- Lớp output ---
        self.output_conv = nn.Conv2d(128, (5 + num_classes), 1, 1, 0, bias=True)

    def forward(self, x):
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
        return x

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
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được frame từ webcam.")
            break

        # Resize frame về 256x256
        frame_resized = cv2.resize(frame, (256, 256))
        # Chuyển đổi BGR sang RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Decode dự đoán
        boxes = decode_predictions(outputs, conf_threshold=0.3, grid_size=8, img_size=256)
        
        # Vì mỗi ảnh chỉ có 1 đối tượng, chọn box có confidence cao nhất (nếu có)
        if boxes:
            best_box = max(boxes, key=lambda x: x[4])
            x1, y1, x2, y2, conf = best_box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"Conf: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("YOLO No Anchor Detection", frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
