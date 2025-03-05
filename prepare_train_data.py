import os
import cv2
import numpy as np

# Paths
input_image_dir = 'train/images'
input_label_dir = 'train/labels'
output_image_dir = 'train_prepared/images'
output_label_dir = 'train_prepared/labels'

# Ensure output directories exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Desired size
output_size = (512, 256)

def resize_and_adjust_bboxes(image_path, label_path, output_image_path, output_label_path, output_size):
    # Read image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Debug: Print original image size
    print(f"Original image size: {w}x{h}")

    # Resize image
    resized_image = cv2.resize(image, output_size)
    cv2.imwrite(output_image_path, resized_image)

    # Read and adjust bounding boxes
    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        bbox = list(map(float, parts[1:]))

        # Debug: Print original bounding box
        print(f"Original bbox: {bbox}")

        # No need to adjust normalized bbox coordinates for YOLO format
        new_bbox = bbox

        # Debug: Print new bounding box
        print(f"New bbox: {new_bbox}")

        new_line = f"{class_id} {' '.join(map(str, new_bbox))}\n"
        new_lines.append(new_line)

    # Write new label file
    with open(output_label_path, 'w') as f:
        f.writelines(new_lines)

# Process all files
for filename in os.listdir(input_image_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(input_image_dir, filename)
        label_path = os.path.join(input_label_dir, filename.replace('.png', '.txt'))
        output_image_path = os.path.join(output_image_dir, filename)
        output_label_path = os.path.join(output_label_dir, filename.replace('.png', '.txt'))

        resize_and_adjust_bboxes(image_path, label_path, output_image_path, output_label_path, output_size)

print("Dataset preparation complete.")