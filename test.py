import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

#########################################
# Model: YoloNoAnchorDeeper
#########################################
class YoloNoAnchorDeeper(nn.Module):
    def __init__(self, num_classes=1):
        super(YoloNoAnchorDeeper, self).__init__()
        self.num_classes = num_classes
        # Block 1: 256x256 -> 128x128, channels: 3 -> 16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 2: 128x128 -> 64x64, channels: 16 -> 32
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 3: 64x64 -> 32x32, channels: 32 -> 64
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 4: 32x32 -> 16x16, channels: 64 -> 128
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 5: 16x16 -> 8x8, channels: 128 -> 256
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Output layer: Predict 6 channels: [objectness, x, y, w, h, class score]
        self.out_conv = nn.Conv2d(256, (5 + num_classes), kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        x = self.block1(x)  # 256 -> 128
        x = self.block2(x)  # 128 -> 64
        x = self.block3(x)  # 64 -> 32
        x = self.block4(x)  # 32 -> 16
        x = self.block5(x)  # 16 -> 8
        x = self.out_conv(x)  # Output shape: (B, 6, 8, 8)
        return x

#########################################
# Decode function: convert model output into bounding boxes
#########################################
def decode_predictions(predictions, conf_threshold=0.3, grid_size=8, img_size=256):
    """
    Args:
      predictions: Tensor of shape (1, 6, grid_size, grid_size)
      conf_threshold: Minimum objectness confidence to consider a box.
      grid_size: The spatial resolution of the output grid (8 for this model).
      img_size: Size of the input image (assumed square, e.g., 256).
    
    Returns:
      A list of bounding boxes as tuples: (x1, y1, x2, y2, confidence)
      
    Decoding details:
      - The first channel is the objectness score (apply sigmoid).
      - The next two channels are x and y offsets within each grid cell (apply sigmoid).
      - The following two channels are width and height predictions, which are scaled by img_size.
    """
    preds = predictions[0]  # shape: (6, grid_size, grid_size)
    boxes = []
    # Objectness score
    obj = torch.sigmoid(preds[0])
    cell_size = img_size / grid_size  # e.g., 256/8 = 32 pixels per cell
    for i in range(grid_size):
        for j in range(grid_size):
            conf = obj[i, j].item()
            if conf > conf_threshold:
                # Offsets for center coordinates
                tx = torch.sigmoid(preds[1, i, j]).item()
                ty = torch.sigmoid(preds[2, i, j]).item()
                # Predicted width and height
                tw = preds[3, i, j].item()
                th = preds[4, i, j].item()
                # Ensure non-negative dimensions
                tw = max(tw, 0)
                th = max(th, 0)
                # Compute absolute center coordinates
                cx = (j + tx) * cell_size
                cy = (i + ty) * cell_size
                # Compute box width and height in absolute pixels
                box_w = tw * img_size
                box_h = th * img_size
                # Convert center, width, height to top-left and bottom-right coordinates
                x1 = int(cx - box_w / 2)
                y1 = int(cy - box_h / 2)
                x2 = int(cx + box_w / 2)
                y2 = int(cy + box_h / 2)
                boxes.append((x1, y1, x2, y2, conf))
    return boxes

#########################################
# Main: Test the model using the webcam
#########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model and load pretrained weights
    model = YoloNoAnchorDeeper(num_classes=1).to(device)
    weight_path = "yolo_no_anchor_model_deeper.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame from webcam.")
            break

        # Resize frame to 256x256 and convert from BGR to RGB
        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Decode the model outputs into bounding boxes
        boxes = decode_predictions(outputs, conf_threshold=0.3, grid_size=8, img_size=256)
        if boxes:
            # Since each image contains one object, choose the box with the highest confidence
            best_box = max(boxes, key=lambda x: x[4])
            x1, y1, x2, y2, conf = best_box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"Conf: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Test YoloNoAnchorDeeper", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
