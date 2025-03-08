import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.ao.quantization as quant
from train import YoloNoAnchor  # assuming your model is defined in train.py

# Define a function to fuse modules (modify as needed for your model)
def fuse_model(model):
    # Fuse Stage 1 modules
    quant.fuse_modules(model.stage1_conv1, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv2, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv3, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv4, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv5, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv6, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv7, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage1_conv8, ['0', '1'], inplace=True)
    # Fuse Stage 2a modules if possible (if they are suitable for fusing)
    quant.fuse_modules(model.stage2_a_conv1, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage2_a_conv2, ['0', '1'], inplace=True)
    quant.fuse_modules(model.stage2_a_conv3, ['0', '1'], inplace=True)
    return model

def test_quantized_model():
    device = torch.device("cpu")  # Quantized models usually run on CPU
    # 1. Instantiate the model
    model = YoloNoAnchor(num_classes=1)
    model.eval()

    # 2. Fuse layers
    model = fuse_model(model)

    # 3. Set the quantization configuration and prepare the model
    model.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(model, inplace=True)
    
    # (Optional) Calibration: run a few batches if you have a calibration dataset.
    # For now, we'll skip this step assuming the weights were quantized properly.
    
    # 4. Convert the model to quantized version
    quantized_model = quant.convert(model, inplace=True)
    
    # 5. Load the quantized state_dict
    weight_path = "yolo_no_anchor_quantized.pth"
    quantized_model.load_state_dict(torch.load(weight_path, map_location=device))
    quantized_model.eval()

    # 6. Define transformation for input image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Resize and preprocess frame
        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = quantized_model(input_tensor)
        
        # (Your decoding function remains the same)
        boxes = decode_predictions(outputs, conf_threshold=0.3, grid_size=8, img_size=256)
        
        if boxes:
            best_box = max(boxes, key=lambda x: x[4])
            x1, y1, x2, y2, conf = best_box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"Conf: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Quantized YOLO", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_quantized_model()

