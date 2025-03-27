from fast_plate_ocr import ONNXPlateRecognizer
import os
import time

m = ONNXPlateRecognizer(model_path='vietnam_cnn_ocr.onnx', config_path='config/vietnam_plate.yaml')
print('Model loaded')
# m.benchmark()
img_path = 'valid_imgs'
start_time = time.time()
count = 0
for img_name in os.listdir(img_path):
    count += 1
    plate = m.run(img_path + '/' + img_name)
    print('img_name:', img_name, 'plate:', plate)
print('Time:', time.time() - start_time, 's')
print('FPS:', count / (time.time() - start_time))
print('Time per image:', (time.time() - start_time) / count, 's')