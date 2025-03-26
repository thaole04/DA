import numpy as np
import torch
import os
import sys
import argparse
sys.path.append(os.path.abspath('./yolov5'))
from utils.general import non_max_suppression
from models.experimental import attempt_load
import cv2
import time
import torchvision.transforms as transforms
from PIL import Image

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

def detect():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detect_model = torch.load("detect_model.pth", map_location=device, weights_only=False)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_path = "xemay.jpg"
    # Đọc ảnh gốc
    frame = cv2.imread(img_path)
    if frame is None:
        print("Không thể đọc ảnh", img_path)
        return

    # Lưu lại kích thước gốc
    orig_h, orig_w = frame.shape[:2]

    # Resize ảnh cho việc dự đoán
    frame_resized = cv2.resize(frame, (256, 256))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = detect_model(input_tensor)
        
    boxes = decode_predictions(outputs, conf_threshold=0.5, grid_size=8, img_size=256)
    if boxes:
        best_box = max(boxes, key=lambda x: x[4])
        x1, y1, x2, y2, conf = best_box
        
        # Tính hệ số scale giữa ảnh gốc và ảnh đã resize
        scale_x = orig_w / 256
        scale_y = orig_h / 256
        
        # Chuyển đổi tọa độ box từ ảnh resize sang ảnh gốc
        x1_orig = int(x1 * scale_x)
        y1_orig = int(y1 * scale_y)
        x2_orig = int(x2 * scale_x)
        y2_orig = int(y2 * scale_y)
        
        # return ảnh chỉ cắt phần biển số
        return frame[y1_orig:y2_orig, x1_orig:x2_orig]
    return None


class Detection:
    def __init__(self, weights_path='.pt',size=(640,640),device='cuda',iou_thres=None,conf_thres=None):
        cwd = os.path.dirname(__file__)
        self.device=device
        self.char_model, self.names = self.load_model(weights_path)
        self.size=size
        
        self.iou_thres=iou_thres
        self.conf_thres=conf_thres

    def detect(self, frame):
        
        results, resized_img = self.char_detection_yolo(frame)

        return results, resized_img
    
    def preprocess_image(self, original_image):

        resized_img = self.ResizeImg(original_image,size=self.size)
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.float()
        image = image / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img
    
    def char_detection_yolo(self, image, classes=None, \
                            agnostic_nms=True, max_det=1000):

        img,resized_img = self.preprocess_image(image.copy())
        pred = self.char_model(img, augment=False)[0]
        
        detections = non_max_suppression(pred, conf_thres=self.conf_thres,
                                            iou_thres=self.iou_thres,
                                            classes=classes,
                                            agnostic=agnostic_nms,
                                            multi_label=True,
                                            labels=(),
                                            max_det=max_det)
        results=[]
        for i, det in enumerate(detections):
            # det[:, :4]=scale_coords(resized_img.shape,det[:, :4],image.shape).round()
            det=det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    # xc,yc,w_,h_=(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,(xyxy[2]-xyxy[0]),(xyxy[3]-xyxy[1])
                    result=[self.names[int(cls)], str(conf), (xyxy[0],xyxy[1],xyxy[2],xyxy[3])]
                    results.append(result)
        # print(results)
        return results, resized_img
        
    def ResizeImg(self, img, size):
        h1, w1, _ = img.shape
        # print(h1, w1, _)
        h, w = size
        if w1 < h1 * (w / h):
            # print(w1/h1)
            img_rs = cv2.resize(img, (int(float(w1 / h1) * h), h))
            mask = np.zeros((h, w - (int(float(w1 / h1) * h)), 3), np.uint8)
            img = cv2.hconcat([img_rs, mask])
            trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
            trans_y = 0
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
        else:
            img_rs = cv2.resize(img, (w, int(float(h1 / w1) * w)))
            mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
            img = cv2.vconcat([img_rs, mask])
            trans_x = 0
            trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
    def load_model(self,path, train = False):
        model = torch.load(path, map_location=self.device, weights_only=False)['model']
        model = model.float()
        model.to(self.device).eval()
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if train:
            model.train()
        else:
            model.eval()
        return model, names


def sort_license_plate_chars(detections, plate_type):
    chars_info = []
    if len(detections) == 0:
        return ''
    for label, confidence, box in detections:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        chars_info.append({
            'char': label,
            'confidence': confidence,
            'center_x': center_x,
            'center_y': center_y,
            'box': box
        })
    
    if plate_type == 'rectangle':
        # Biển số 1 dòng: sắp xếp tất cả ký tự theo trục x
        chars_info.sort(key=lambda x: x['center_x'])
        result = ''.join(char['char'] for char in chars_info)
        
    elif plate_type == 'square':
        # Biển số 2 dòng
        y_coords = [info['center_y'] for info in chars_info]
        y_min, y_max = min(y_coords), max(y_coords)
        y_middle = (y_max + y_min) / 2
        
        # Phân chia ký tự thành 2 dòng
        upper_line = []
        lower_line = []
        
        for char_info in chars_info:
            if char_info['center_y'] < y_middle:
                upper_line.append(char_info)
            else:
                lower_line.append(char_info)
        
        # Sắp xếp các ký tự trong mỗi dòng theo trục x
        upper_line.sort(key=lambda x: x['center_x'])
        lower_line.sort(key=lambda x: x['center_x'])
        
        # Ghép các ký tự thành chuỗi với dấu - ở giữa
        result = ''.join(char['char'] for char in upper_line) + '-' + ''.join(char['char'] for char in lower_line)
        
    else:
        raise ValueError("plate_type phải là 'rectangle' hoặc 'square'")
    
    return result

def recognize_plate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_model = Detection(size=[128, 128],weights_path='weight.pt',device=device,iou_thres=0.5,conf_thres=0.1)
    print("Load char model xong")
    detect_model = torch.load("detect_model.pth", map_location=device, weights_only=False)
    print("Load detect model xong")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_path = "xemay.jpg"
    # Đọc ảnh gốc
    frame = cv2.imread(img_path)
    if frame is None:
        print("Không thể đọc ảnh", img_path)
        return

    # Lưu lại kích thước gốc
    orig_h, orig_w = frame.shape[:2]

    # Resize ảnh cho việc dự đoán
    frame_resized = cv2.resize(frame, (256, 256))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = detect_model(input_tensor)
    plate_img = None
        
    boxes = decode_predictions(outputs, conf_threshold=0.5, grid_size=8, img_size=256)
    if boxes:
        best_box = max(boxes, key=lambda x: x[4])
        x1, y1, x2, y2, conf = best_box
        
        # Tính hệ số scale giữa ảnh gốc và ảnh đã resize
        scale_x = orig_w / 256
        scale_y = orig_h / 256
        
        # Chuyển đổi tọa độ box từ ảnh resize sang ảnh gốc
        x1_orig = int(x1 * scale_x)
        y1_orig = int(y1 * scale_y)
        x2_orig = int(x2 * scale_x)
        y2_orig = int(y2 * scale_y)
        
        # return ảnh chỉ cắt phần biển số
        plate_img = frame[y1_orig:y2_orig, x1_orig:x2_orig]
    if plate_img is not None:
        results,_ = char_model.detect(plate_img.copy())
        if plate_img.shape[0] > (plate_img.shape[1]/2):
            plate_text = sort_license_plate_chars(results, 'square')
        else:
            plate_text = sort_license_plate_chars(results, 'rectangle')
        plate_text = plate_text.upper()
        print("License plate: {}".format(plate_text))
    else:
        print("Không tìm thấy biển")

if __name__ == '__main__':
    recognize_plate()