from ultralytics import YOLO
import os

# 1. 加载模型
MODEL_PATH = r"C:\Users\ASUS\PycharmProjects\YOLOv12\runs\detect\train4\weights\best.pt"
model = YOLO(MODEL_PATH)

# 2. 定义视频路径（支持mp4等格式，若填 "0" 则调用电脑摄像头实时检测）
VIDEO_PATH = r"C:\Users\ASUS\Downloads\c03.mp4"  # 实时检测：VIDEO_PATH = 0

# 3. 执行视频检测（save=True 自动保存检测后的视频）
results = model(VIDEO_PATH, conf=0.5, imgsz=640, save=True)

print(f"✅ 视频检测完成！结果已保存到：runs/detect/predict/")