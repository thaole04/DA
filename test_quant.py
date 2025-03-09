import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchsummary import summary
import torch.quantization as quant

# Import model wrapper đã tích hợp QuantStub/DeQuantStub
from quantized_model import YoloNoAnchorQuantized

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
      - w, h được dự đoán trực tiếp (nếu cần có thể áp dụng sigmoid hoặc clamp).
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
                # Tính tọa độ box: từ trung tâm sang góc trái và góc phải
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
    
    # Khởi tạo model wrapper đã tích hợp QuantStub/DeQuantStub
    model = YoloNoAnchorQuantized(num_classes=1).to(device)
    model.eval()

   # Định nghĩa danh sách các module cần fuse (chỉ fuse Conv và BatchNorm)
    fuse_modules = [
        ["stage1_conv1.0", "stage1_conv1.1"],
        ["stage1_conv2.0", "stage1_conv2.1"],
        ["stage1_conv3.0", "stage1_conv3.1"],
        ["stage1_conv4.0", "stage1_conv4.1"],
        ["stage1_conv5.0", "stage1_conv5.1"],
        ["stage1_conv6.0", "stage1_conv6.1"],
        ["stage1_conv7.0", "stage1_conv7.1"],
        ["stage1_conv8.0", "stage1_conv8.1"],
        ["stage2_a_conv1.0", "stage2_a_conv1.1"],
        ["stage2_a_conv2.0", "stage2_a_conv2.1"],
        ["stage2_a_conv3.0", "stage2_a_conv3.1"]
    ]

    model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, reduce_range=False
        ),
        weight=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        )
    )

    # Fuse các module (Conv + BN) theo danh sách fuse_modules
    model = quant.fuse_modules(model, fuse_modules, inplace=False)

    model = quant.prepare(model, inplace=False)
    model = quant.convert(model, inplace=False)
    
    # Load trọng số đã quantized (model đã được fuse & convert trước đó)
    weight_path = "yolo_no_anchor_quantized.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    frame = cv2.imread("test.jpg")

    # Resize frame về 256x256
    frame_resized = cv2.resize(frame, (256, 256))
    # Chuyển đổi BGR sang RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Decode dự đoán
    boxes = decode_predictions(outputs, conf_threshold=0.5, grid_size=8, img_size=256)
    
    # Nếu có dự đoán, chọn box có confidence cao nhất
    if boxes:
        best_box = max(boxes, key=lambda x: x[4])
        x1, y1, x2, y2, conf = best_box
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_resized, f"Conf: {conf:.2f}", (x1, max(y1-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLO No Anchor Detection", frame_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
