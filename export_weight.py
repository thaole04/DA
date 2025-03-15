import torch
import struct
import numpy as np

def save_weights_to_bin(pth_file, bin_file):
    # Load state_dict từ file .pth
    state_dict = torch.load(pth_file, map_location="cpu")
    
    with open(bin_file, "wb") as f:
        # Ghi số lượng layer (số phần tử trong state_dict)
        num_layers = len(state_dict)
        f.write(struct.pack("I", num_layers))
        
        for name, param in state_dict.items():
            # Tách tensor ra khỏi đồ thị tính toán
            param = param.detach()
            
            # Ghi tên layer: độ dài tên và sau đó tên (UTF-8)
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("I", len(name_bytes)))
            f.write(name_bytes)
            
            # Ghi thông tin shape: số chiều và từng chiều (unsigned int)
            shape = param.shape
            num_dims = len(shape)
            f.write(struct.pack("I", num_dims))
            for dim in shape:
                f.write(struct.pack("I", dim))
            
            # Xác định dtype và chuyển tensor về numpy theo cách 2: ép int64 về int32
            # Mã dtype: 0: float32, 1: int8, 2: int32
            if param.dtype == torch.float32:
                dtype_code = 0
                np_array = param.cpu().numpy()
            elif param.dtype == torch.int8:
                dtype_code = 1
                np_array = param.cpu().numpy()
            elif param.dtype == torch.qint8:
                # Xử lý quantized tensor: lấy underlying int_repr()
                dtype_code = 1  # Gán cùng mã như int8
                np_array = param.int_repr().cpu().numpy()
            elif param.dtype == torch.int32:
                dtype_code = 2
                np_array = param.cpu().numpy()
            elif param.dtype == torch.int64:
                dtype_code = 2  # Ép kiểu về int32
                np_array = param.cpu().numpy().astype(np.int32)
            else:
                raise ValueError(f"Unsupported dtype: {param.dtype}")
            
            f.write(struct.pack("I", dtype_code))
            
            # Chuyển tensor (dạng numpy) về raw bytes và ghi vào file
            raw_bytes = np_array.tobytes()
            raw_length = len(raw_bytes)
            f.write(struct.pack("I", raw_length))
            f.write(raw_bytes)
            
            print(f"Saved layer '{name}' with shape {shape} and dtype {param.dtype}")

if __name__ == '__main__':
    pth_file = "yolo_no_anchor_quantized.pth"
    bin_file = "model_quantized.bin"
    save_weights_to_bin(pth_file, bin_file)
    print(f"Đã lưu file nhị phân: {bin_file}")
