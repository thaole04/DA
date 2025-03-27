import pandas as pd
from fast_plate_ocr import ONNXPlateRecognizer
import os

def calculate_accuracy_with_error_tolerance(csv_file, image_dir, model_path='vietnam2_cnn_ocr.onnx', config_path='config/latin_plate_example.yaml'):
    """
    Tính toán độ chính xác của model nhận diện biển số xe với các mức độ sai khác nhau.

    Args:
        csv_file (str): Đường dẫn đến file CSV chứa 'image_path' và 'plate_text'.
        image_dir (str): Thư mục gốc chứa ảnh, nếu 'image_path' trong CSV là đường dẫn tương đối.
        model_path (str): Đường dẫn đến file model ONNX.
        config_path (str): Đường dẫn đến file config YAML.

    Returns:
        tuple: (perfect_accuracy, accuracy_error_le_1, accuracy_error_le_2, accuracy_error_le_3, total_predictions) # Thêm total_predictions vào return
               - perfect_accuracy (float): Tỷ lệ phần trăm độ chính xác hoàn toàn.
               - accuracy_error_le_1 (float): Tỷ lệ phần trăm độ chính xác với lỗi <= 1 ký tự.
               - accuracy_error_le_2 (float): Tỷ lệ phần trăm độ chính xác với lỗi <= 2 ký tự.
               - accuracy_error_le_3 (float): Tỷ lệ phần trăm độ chính xác với lỗi <= 3 ký tự.
               - total_predictions (int): Tổng số lượng dự đoán. # Giải thích thêm về total_predictions
    """

    df = pd.read_csv(csv_file)
    m = ONNXPlateRecognizer(model_path=model_path, config_path=config_path)
    # m = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model')
    correct_predictions = 0
    error_le_1 = 0
    error_le_2 = 0
    error_le_3 = 0
    total_predictions = 0

    for index, row in df.iterrows():
        image_path_csv = row['image_path']
        ground_truth_plate = str(row['plate_text']).strip().upper()

        full_image_path = os.path.join(image_dir, image_path_csv)

        if not os.path.exists(full_image_path):
            print(f"Cảnh báo: Không tìm thấy ảnh tại đường dẫn: {full_image_path}")
            continue

        try:
            predicted_plate = m.run(full_image_path)
            if isinstance(predicted_plate, list):
                if predicted_plate:
                    predicted_plate_str = predicted_plate[0]
                else:
                    predicted_plate_str = ""
            else:
                predicted_plate_str = predicted_plate

            predicted_plate_str = str(predicted_plate_str).strip().upper().replace('_', '')

            if predicted_plate_str == ground_truth_plate:
                correct_predictions += 1
                error_le_1 += 1
                error_le_2 += 1
                error_le_3 += 1
            else:
                mismatched_chars = 0
                min_len = min(len(predicted_plate_str), len(ground_truth_plate))
                max_len = max(len(predicted_plate_str), len(ground_truth_plate))

                for i in range(min_len):
                    if predicted_plate_str[i] != ground_truth_plate[i]:
                        mismatched_chars += 1
                mismatched_chars += (max_len - min_len)

                if mismatched_chars <= 1:
                    error_le_1 += 1
                    error_le_2 += 1
                    error_le_3 += 1
                elif mismatched_chars <= 2:
                    error_le_2 += 1
                    error_le_3 += 1
                elif mismatched_chars <= 3:
                    error_le_3 += 1

                if mismatched_chars > 0: # Chỉ in ra nếu có lỗi để dễ quan sát
                    print(f"Sai {mismatched_chars} ký tự - Ảnh: {image_path_csv}, Dự đoán: {predicted_plate_str}, Thực tế: {ground_truth_plate}")


            total_predictions += 1

        except Exception as e:
            print(f"Lỗi khi xử lý ảnh: {image_path_csv}, Lỗi: {e}")
            continue

    perfect_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    accuracy_error_le_1 = (error_le_1 / total_predictions) * 100 if total_predictions > 0 else 0
    accuracy_error_le_2 = (error_le_2 / total_predictions) * 100 if total_predictions > 0 else 0
    accuracy_error_le_3 = (error_le_3 / total_predictions) * 100 if total_predictions > 0 else 0

    return perfect_accuracy, accuracy_error_le_1, accuracy_error_le_2, accuracy_error_le_3, total_predictions # Thêm total_predictions vào return

if __name__ == "__main__":
    csv_file = '../images/val.csv'
    image_dir = '../images'

    perfect_accuracy, accuracy_error_le_1, accuracy_error_le_2, accuracy_error_le_3, total_predictions = calculate_accuracy_with_error_tolerance(csv_file, image_dir) # Nhận total_predictions khi gọi hàm

    print(f"Tổng số lượng dự đoán: {total_predictions}")
    print(f"Tỉ lệ đúng hoàn toàn: {perfect_accuracy:.2f}%")
    print(f"Tỉ lệ sai <= 1 ký tự: {accuracy_error_le_1:.2f}%")
    print(f"Tỉ lệ sai <= 2 ký tự: {accuracy_error_le_2:.2f}%")
    print(f"Tỉ lệ sai <= 3 ký tự: {accuracy_error_le_3:.2f}%")