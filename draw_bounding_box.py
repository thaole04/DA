import os
import cv2

# Paths
image_dir = 'train_prepared/images'
label_dir = 'train_prepared/labels'
output_dir = 'train_bboxes'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def draw_bounding_boxes(image_path, label_path, output_path):
    # Read image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Read bounding boxes
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        bbox = list(map(float, parts[1:]))

        # Debug: Print original bounding box coordinates
        print(f"Original bbox: {bbox}")

        # Convert relative coordinates to absolute coordinates
        x_center, y_center, bbox_width, bbox_height = bbox
        x_center = int(x_center * w)
        y_center = int(y_center * h)
        bbox_width = int(bbox_width * w)
        bbox_height = int(bbox_height * h)

        # Debug: Print scaled bounding box coordinates
        print(f"Scaled bbox: x_center={x_center}, y_center={y_center}, width={bbox_width}, height={bbox_height}")

        # Calculate top-left and bottom-right corners
        x1 = x_center - bbox_width // 2
        y1 = y_center - bbox_height // 2
        x2 = x_center + bbox_width // 2
        y2 = y_center + bbox_height // 2

        # Debug: Print corner coordinates
        print(f"Drawing bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save image with bounding boxes
    cv2.imwrite(output_path, image)

# Process all files
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))
        output_path = os.path.join(output_dir, filename)

        draw_bounding_boxes(image_path, label_path, output_path)

print("Bounding boxes drawn and saved in", output_dir)