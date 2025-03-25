import os
import random
import cv2
import glob
import torch
import torchvision.transforms as transforms
from torch.quantization import MovingAverageMinMaxObserver, MinMaxObserver, HistogramObserver
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torch.ao.quantization import get_default_qconfig
from quantized_model import YoloNoAnchorQuantized
from model import YoloNoAnchor
# from torchsummary import summary
def print_first_layer_output(model, input_tensor):
    # Tạo biến chứa output từ layer đầu tiên
    outputs = {}

    # Định nghĩa hook function để lưu output của layer đầu tiên
    def hook_fn(module, input, output):
        outputs['first_layer'] = output

    # Giả sử sau fuse, layer đầu tiên là module 'conv1' (đã được fuse với bn1 và relu1)
    hook_handle = model.conv1.register_forward_hook(hook_fn)
    
    # Chạy forward pass để hook được kích hoạt
    _ = model(input_tensor)
    
    # Sau khi lấy output, hủy hook để không ảnh hưởng đến forward sau này
    hook_handle.remove()
    
    # In ra kết quả của layer đầu tiên 
    print("Output sau layer đầu tiên (conv1 fuse block):")
    print(outputs['first_layer'])
    print("Output sau layer đầu tiên đã dequant:")
    print(outputs['first_layer'].int_repr())
class YOLODataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Đường dẫn chứa thư mục con 'images' và 'labels'.
            transform (callable, optional): Các phép biến đổi ảnh.
        """
        self.image_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        base_name = os.path.basename(img_path).replace(".jpg", ".txt")
        label_path = os.path.join(self.label_dir, base_name)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x_center, y_center, w, h = parts
                        boxes.append([float(cls), float(x_center), float(y_center), float(w), float(h)])
        target = torch.tensor(boxes) if boxes else torch.empty((0, 5))
        return image, target
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
# Main: Test model sử dụng ảnh từ thư mục
# -----------------------------
def main():
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    model_quant = YoloNoAnchor(num_classes=1).to(device)
    model_quant.eval()
    fuse_list = [
        ['conv1', 'bn1', 'relu1'],
        ['conv2', 'bn2', 'relu2'],
        ['conv3', 'bn3', 'relu3'],
        ['conv4', 'bn4', 'relu4'],
        ['conv5', 'bn5', 'relu5'],
        ['conv6', 'bn6', 'relu6'],
        ['conv7', 'bn7', 'relu7'],
        ['conv8', 'bn8', 'relu8'],
        ['conv9', 'bn9', 'relu9'],
        ['conv10', 'bn10', 'relu10'],
        ['conv11', 'bn11', 'relu11']
    ]
    torch.quantization.fuse_modules(model_quant, fuse_list, inplace=True)
    model_quant.qconfig = torch.quantization.QConfig(
        activation=HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
        weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
    )
    print("QConfig:", model_quant.qconfig)
    torch.quantization.prepare(model_quant, inplace=True)
    torch.quantization.convert(model_quant, inplace=True)
    # Load trọng số đã quantized
    model_quant.load_state_dict(torch.load("HistogramObserver_both.pth", map_location=device))

    # Thiết lập pipeline tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    haha = 2
    if haha == 0:
        # Đọc ảnh từ file
        img_path = "xemay.jpg"
        frame = cv2.imread(img_path)
        if frame is None:
            print("Không thể đọc ảnh", img_path)
            return
        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        input_tensor = torch.quantize_per_tensor(input_tensor, scale=1/255, zero_point=0, dtype=torch.quint8)
        # quint8 => 0 -> 256
        # print(input_tensor)
        # print("--------------------------------")
        print(input_tensor.int_repr())

        with torch.no_grad():
            print_first_layer_output(model_quant, input_tensor)
            outputs = model_quant(input_tensor)
        
        boxes = decode_predictions(outputs, conf_threshold=0.5, grid_size=8, img_size=256)
        if boxes:
            best_box = max(boxes, key=lambda x: x[4])
            x1, y1, x2, y2, conf = best_box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"Conf: {conf:.2f}", (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Detection", frame_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif haha == 1:
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
            # convert to quantized model
            input_tensor = torch.quantize_per_tensor(input_tensor, scale=1/255, zero_point=0, dtype=torch.quint8)
            # print(input_tensor)
            
            with torch.no_grad():
                # tính thời gian xử lý
                # start = cv2.getTickCount()
                print_first_layer_output(model_quant, input_tensor)
                outputs = model_quant(input_tensor)
                # end = cv2.getTickCount()
                # print("Time: %.2fms" % ((end - start) / cv2.getTickFrequency() * 1000))
                # print("fps: %.2f" % (cv2.getTickFrequency() / (end - start)))
                # print outputs
                # print(outputs)
            
            boxes = decode_predictions(outputs, conf_threshold=0.5, grid_size=8, img_size=256)
            if boxes:
                best_box = max(boxes, key=lambda x: x[4])
                x1, y1, x2, y2, conf = best_box
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"Conf: {conf:.2f}", (x1, max(y1-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Detection", frame_resized)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                continue
            cv2.destroyAllWindows()
    elif haha == 2:
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
            input_tensor = torch.quantize_per_tensor(input_tensor, scale=1/255, zero_point=0, dtype=torch.quint8)
            
            with torch.no_grad():
                print_first_layer_output(model_quant, input_tensor)
                outputs = model_quant(input_tensor)
            
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
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
