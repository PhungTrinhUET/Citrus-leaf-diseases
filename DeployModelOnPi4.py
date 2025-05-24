import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import time

# ------------------ CẤU HÌNH ------------------
MODEL_PATH = "/home/edgedevice/Downloads/citrus_mbv3.tflite"
INPUT_WIDTH, INPUT_HEIGHT = 224, 224

# Thứ tự nhãn – hãy kiểm tra lại nếu cần điều chỉnh!
LABELS = ["Black spot", "Canker", "Greening", "Healthy", "Melanose"]

# ------------------ KHỞI TẠO CAMERA ------------------
# Sử dụng Picamera2 để capture ảnh với định dạng RGB888 và kích thước 224x224
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (INPUT_WIDTH, INPUT_HEIGHT)})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Cho camera ổn định

# ------------------ LOAD MODEL TFLITE ------------------
print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded successfully.")

# ------------------ HÀM TIỀN XỬ LÝ ------------------
def preprocess(frame):
    """
    - Nhận vào frame ảnh (RGB) kích thước 224x224.
    - Ép kiểu sang float32 (giá trị pixel giữ nguyên trong khoảng [0, 255]).
    - Thêm dimension batch: kết quả có shape (1, 224, 224, 3)
    """
    # Nếu ảnh được capture từ Picamera2 theo định dạng RGB, không cần chuyển đổi màu.
    frame_float = frame.astype(np.float32)
    return np.expand_dims(frame_float, axis=0)

# ------------------ HÀM DỰ ĐOÁN ------------------
def classify(frame):
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Giả sử shape: (5,)
    pred_index = np.argmax(output_data)
    confidence = output_data[pred_index]
    pred_label = LABELS[pred_index]
    return pred_label, confidence, output_data

# ------------------ VÒNG LẶP REALTIME ------------------
print("Starting realtime classification. Press 'q' to exit.")
while True:
    # Capture frame từ camera (Picamera2 trả về ảnh theo định dạng đã cấu hình: RGB888)
    frame = picam2.capture_array()
   
    # Chạy mô hình để dự đoán
    label, conf, prob_vector = classify(frame)
    result_text = f"{label} ({conf*100:.1f}%)"
   
    # Chuyển ảnh sang BGR để hiển thị qua OpenCV
    disp_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
   
    # Vẽ bounding box quanh toàn ảnh (vì đây là classifier)
    cv2.rectangle(disp_frame, (0, 0), (INPUT_WIDTH-1, INPUT_HEIGHT-1), (0, 255, 0), 2)
    cv2.putText(disp_frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
   
    # Hiển thị cửa sổ realtime
    cv2.imshow("Realtime Classification", disp_frame)
   
    # In ra vector xác suất để debug (nếu cần)
    # print("Output vector:", prob_vector)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
print("Realtime classification stopped.")
