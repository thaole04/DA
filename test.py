import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

######################################
# Model: YoloNoAnchorLite (phiên bản nhẹ)
######################################
class YoloNoAnchorLite(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchorLite, self).__init__()
        self.num_classes = num_classes
        
        # Block 1: Input 256x256 -> 128x128, channels: 3 -> 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 2: 128x128 -> 64x64, channels: 16 -> 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 3: Giữ kích thước 64x64, channels: 32 -> 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Block 4: 64x64 -> 32x32, channels: 64 -> 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Lớp output: Dự đoán 6 kênh: objectness, x, y, w, h, class score
        self.out_conv = nn.Conv2d(128, (5 + num_classes), kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        x = self.conv1(x)   # 256 -> 128
        x = self.conv2(x)   # 128 -> 64
        x = self.conv3(x)   # 64 giữ nguyên
        x = self.conv4(x)   # 64 -> 32
        x = self.out_conv(x)  # Output shape: (B, 6, 32, 32)
        return x

######################################
# Hàm decode predictions
######################################
def decode_predictions(predictions, conf_threshold=0.3, grid_size=32, img_size=256):
    """
    predictions: tensor có shape (1, 6, grid_size, grid_size)
    Trả về danh sách bounding box dưới dạng (x1, y1, x2, y2, confidence)

    Cách decode:
      - objectness: sigmoid(pred[0])
      - (x, y): offset trong cell (qua sigmoid) cộng với vị trí cell để tính trung tâm tuyệt đối.
      - (w, h): dự đoán trực tiếp, nhân với img_size để có kích thước tuyệt đối.
    """
    preds = predictions[0]  # shape: (6, grid_size, grid_size)
    boxes = []
    obj = torch.sigmoid(preds[0])
    cell_size = img_size / grid_size  # Với grid_size=32, cell_size = 8 pixels
    for i in range(grid_size):
        for j in range(grid_size):
            conf = obj[i, j].item()
            if conf > conf_threshold:
                tx = torch.sigmoid(preds[1, i, j]).item()
                ty = torch.sigmoid(preds[2, i, j]).item()
                tw = preds[3, i, j].item()
                th = preds[4, i, j].item()
                # Đảm bảo kích thước không âm
                tw = max(tw, 0)
                th = max(th, 0)
                # Tính tọa độ trung tâm tuyệt đối
                cx = (j + tx) * cell_size
                cy = (i + ty) * cell_size
                # Chuyển từ trung tâm và kích thước sang góc trên bên trái và góc dưới bên phải
                box_w = tw * img_size
                box_h = th * img_size
                x1 = int(cx - box_w / 2)
                y1 = int(cy - box_h / 2)
                x2 = int(cx + box_w / 2)
                y2 = int(cy + box_h / 2)
                boxes.append((x1, y1, x2, y2, conf))
    return boxes

######################################
# Main: Test model với webcam
######################################
def main():
    device = torch.device("cpu")
    
    # Khởi tạo model và load trọng số đã huấn luyện
    model = YoloNoAnchorLite(num_classes=1).to(device)
    weight_path = "yolo_no_anchor_model_lite.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được frame từ webcam.")
            break

        # Resize frame về 256x256 và chuyển đổi sang RGB
        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Decode bounding boxes với grid_size=32
        boxes = decode_predictions(outputs, conf_threshold=0.6, grid_size=32, img_size=256)
        if boxes:
            # Vì mỗi ảnh chỉ có 1 biển số, chọn box có confidence cao nhất
            best_box = max(boxes, key=lambda x: x[4])
            x1, y1, x2, y2, conf = best_box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"Conf: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Quantized YOLO Inference", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
