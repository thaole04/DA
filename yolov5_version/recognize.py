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

class Detection:
    def __init__(self, weights_path='.pt',size=(640,640),device='cpu',iou_thres=None,conf_thres=None):
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
        # print(self.device)
        # model = attempt_load(path, map_location=self.device)  # load FP32 model
        model = torch.load(path, map_location=self.device)['model']
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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='Vietnamese_imgs', help='file/dir')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--video', type=str, default='demo0.mp4', help='video path')
    opt = parser.parse_args()

    return opt



if __name__ == '__main__':
    opt = parse_opt()
    char_model = Detection(size=[128, 128],weights_path='weight.pt',device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
    path=opt.source

    
    count = 0
    times = 0
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.png'):
            count += 1
            img = cv2.imread(os.path.join(path, file))
            time_start = time.time()
            results,_ = char_model.detect(img.copy())
            time_end = time.time()
            times += time_end - time_start
            if img.shape[0] > (img.shape[1]/2):
                plate_text = sort_license_plate_chars(results, 'square')
            else:
                plate_text = sort_license_plate_chars(results, 'rectangle')
            plate_text = plate_text.upper()
            print("License plate of {}: {}".format(file, plate_text))
            # draw_img = cv2.putText(img, "{}".format(plate_text), (10, 30),
            #                         cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                         (255, 0, 255), 2)
            # # save img
            # cv2.imwrite(os.path.join('./output', 'result_' + file), draw_img)
            
            print("Time to process image {}: {:.4f} seconds".format(file, time_end - time_start))
            
    print("Average time to process 1 image: {:.4f} seconds".format(times / count))
    print("Total images: ", count)