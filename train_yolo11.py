import torch
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
# Huấn luyện mô hình
    model.train(
    data="D:/BTL_AI/BrainTumorYolov11/data.yaml",       # Đường dẫn đến file YAML định nghĩa dataset
    epochs=5,              # Số vòng lặp huấn luyện
    imgsz=640,              # Kích thước ảnh đầu vào
    batch=16,               # Batch size, chỉnh tùy RAM/GPU
    device=0,               # Dùng GPU số 0 (thường là GPU chính)
    project="runs",         # Thư mục chứa kết quả
    name="yolov8n_btl",     # Tên thư mục kết quả
)

