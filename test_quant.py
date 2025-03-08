import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------
# Định nghĩa model YoloNoAnchor (giống với kiến trúc dùng để quantize)
# -----------------------------
class YoloNoAnchor(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchor, self).__init__()
        self.num_classes = num_classes
        # Stage 1
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.stage1_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.stage1_conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv5 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.stage1_conv6 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1_conv8 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2)
        )
        # Stage 2a
        self.stage2_a_maxpl = nn.MaxPool2d(2,2)
        self.stage2_a_conv1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage2_a_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage2_a_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Output layer: Dự đoán [objectness, x, y, w, h, class score]
        self.output_conv = nn.Conv2d(256, (5 + num_classes), 1, stride=1, padding=0, bias=True)

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
    predictions: tensor có shape (1, 6, grid_size, grid_size)
    Trả về danh sách bounding box dưới dạng (x1, y1, x2, y2, confidence)
    
    Cách decode:
      - objectness: sigmoid(pred[0])
      - (x, y): offset trong cell (qua sigmoid) cộng với vị trí cell để tính trung tâm tuyệt đối
      - (w, h): dự đoán trực tiếp, nhân với img_size để có kích thước tuyệt đối
    """
    preds = predictions[0]  # shape: (6, grid_size, grid_size)
    boxes = []
    obj = torch.sigmoid(preds[0])
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
                cell_size = img_size / grid_size  # Ví dụ: 256/8 = 32
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
# Main: Test model quantized với webcam
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo model và load weight quantized
    model = YoloNoAnchor(num_classes=1).to(device)
    state_dict = torch.load("yolo_no_anchor_model_quantized.pth", map_location="cpu")
    for key, value in state_dict.items():
        if hasattr(value, "dequantize"):
            state_dict[key] = value.dequantize()
    model.load_state_dict(state_dict)
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

        # Resize frame về 256x256 và chuyển sang RGB
        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Decode bounding box từ output
        boxes = decode_predictions(outputs, conf_threshold=0.3, grid_size=8, img_size=256)
        # Vì mỗi ảnh chỉ có 1 biển số, chọn bounding box có confidence cao nhất (nếu có)
        if boxes:
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
