import torch
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
from model.LicensePlateModel import LicensePlateModel

# Load the model weights
def load_model(weights_path, input_size=(256, 512), num_boxes=1):
    model = LicensePlateModel(input_size=input_size, num_boxes=num_boxes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Resize the input image
def resize_image(image_path, output_size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(output_size, Image.LANCZOS)
    return image

# Preprocess the input image
def preprocess_image(image, input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Draw bounding box on the image
def draw_bounding_box(image, bbox):
    draw = ImageDraw.Draw(image)
    class_label, x_center, y_center, width, height = bbox
    x_center *= image.width
    y_center *= image.height
    width *= image.width
    height *= image.height
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
    return image

def main():
    weights_path = 'license_plate_model.pth'
    image_path = 'test.png'
    input_size = (256, 512)
    num_boxes = 1

    # Load the model
    model = load_model(weights_path, input_size, num_boxes)

    # Resize the image
    resized_image = resize_image(image_path, (512, 256))

    # Preprocess the image
    image_tensor = preprocess_image(resized_image, input_size)
    
    # Predict the bounding box
    with torch.no_grad():
        output = model(image_tensor)
    
    # Convert the output to numpy array
    bbox = output.squeeze().numpy()
    
    # Draw the bounding box on the image
    result_image = draw_bounding_box(resized_image, bbox)
    
    # Save or display the result image
    result_image.save('result_image.jpg')
    result_image.show()

if __name__ == "__main__":
    main()