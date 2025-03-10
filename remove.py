import os

def remove_orphan_labels(images_dir="train_20000_256/images", labels_dir="train_20000_256/labels"):
    # Duyệt qua tất cả các file trong thư mục labels
    for label_file in os.listdir(labels_dir):
        # Kiểm tra file có đuôi .txt không
        if label_file.endswith('.txt'):
            # Lấy tên file ảnh tương ứng (giả định ảnh có đuôi .jpg)
            base_name = os.path.splitext(label_file)[0]
            image_file = base_name + '.jpg'
            image_path = os.path.join(images_dir, image_file)
            print(f"Kiểm tra file ảnh: {image_path}")
            label_path = os.path.join(labels_dir, label_file)
            
            # Nếu file ảnh không tồn tại, xóa file label
            if not os.path.exists(image_path):
                print(f"Xoá file label: {label_file} vì không tìm thấy file ảnh: {image_file}")
                os.remove(label_path)

if __name__ == "__main__":
    images_directory = "train_20000_256/images"
    labels_directory = "train_20000_256/labels"
    
    remove_orphan_labels(images_directory, labels_directory)
    print("Quá trình kiểm tra và xóa file label hoàn tất.")
