from ultralytics.models.yolo import YOLO
from config import BEST_MODEL_PATH, VIDEO_PATH, CONF_THRESHOLD

# 1. 加载模型
model = YOLO(BEST_MODEL_PATH)

# 2. 执行视频检测（save=True 自动保存检测后的视频）
results = model(VIDEO_PATH, conf=CONF_THRESHOLD, imgsz=640, save=True)

print(f"✅ 视频检测完成！结果已保存到：runs/detect/predict/")
