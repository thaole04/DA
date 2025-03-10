import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Hàm tính IoU cho 2 bounding box theo định dạng (x1, y1, x2, y2)
def compute_iou(box1, box2):
    # Tính tọa độ giao nhau
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

########################################
# Model: YoloNoAnchor (gốc, không dùng anchor)
########################################
from model import YoloNoAnchor

########################################
# Hàm decode_predictions: giải mã output thành bounding box
########################################
def decode_predictions(predictions, conf_threshold=0.3, grid_size=8, img_size=256):
    """
    predictions: Tensor shape (1, 5+num_classes, grid_size, grid_size)
      Với model gốc, với input 256x256, output có kích thước 8x8.
    Returns:
      List các bounding box theo định dạng (x1, y1, x2, y2, confidence)
    """
    preds = predictions[0]  # shape: (5+num_classes, grid_size, grid_size)
    boxes = []
    obj = torch.sigmoid(preds[0])
    cell_size = img_size / grid_size  # 256/8 = 32
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

########################################
# Hàm evaluate_iou: đánh giá IoU trung bình trên tập dữ liệu
########################################
def evaluate_iou(model, dataset, device, transform, grid_size=8, img_size=256, conf_threshold=0.3):
    iou_list = []
    for img, target in dataset:
        if target.numel() == 0:
            continue
        # Lấy ground truth: dùng box đầu tiên (giả sử mỗi ảnh chỉ có 1 đối tượng)
        gt = target[0]  # [cls, x, y, w, h]
        gt_x = gt[1].item() * img_size
        gt_y = gt[2].item() * img_size
        gt_w = gt[3].item() * img_size
        gt_h = gt[4].item() * img_size
        gt_box = (int(gt_x - gt_w/2), int(gt_y - gt_h/2), int(gt_x + gt_w/2), int(gt_y + gt_h/2))
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)
        boxes = decode_predictions(pred, conf_threshold=conf_threshold, grid_size=grid_size, img_size=img_size)
        if boxes:
            best_box = max(boxes, key=lambda x: x[4])
            iou = compute_iou(gt_box, best_box[:4])
            iou_list.append(iou)
    if len(iou_list) == 0:
        return 0.0
    return sum(iou_list) / len(iou_list)

########################################
# Hàm load_model: tải model với trọng số từ file
########################################
def load_model(model_path, device):
    model = YoloNoAnchor(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

########################################
# Main: Tính IoU cho model gốc và model quantized
########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Tạo dataset từ thư mục "train/images" và "train/labels"
    image_paths = sorted(glob.glob(os.path.join("../go_to_quantize_qta/train_prepared", "images", "*.jpg")))
    dataset = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        base_name = os.path.basename(path).replace(".jpg", ".txt")
        label_path = os.path.join("../go_to_quantize_qta/train_prepared", "labels", base_name)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = parts
                        boxes.append([float(cls), float(x), float(y), float(w), float(h)])
        target = torch.tensor(boxes) if boxes else torch.empty((0, 5))
        dataset.append((img, target))
    
    # Load model gốc và model quantized
    original_model_path = "yolo_no_anchor_model.pth"

    original_model = YoloNoAnchor(num_classes=1).to(device)
    
    original_model.load_state_dict(torch.load(original_model_path, map_location=device))
    
    iou_original = evaluate_iou(original_model, dataset, device, transform, grid_size=8, img_size=256)
    
    print("Average IoU (Original Model):", iou_original)

if __name__ == "__main__":
    main()
