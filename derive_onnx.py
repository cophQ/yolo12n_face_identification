from ultralytics.models.yolo import YOLO
from config import BEST_MODEL_PATH, IMG_SIZE, ONNX_OPSET, ONNX_SIMPLIFY

# 1. 加载模型
model = YOLO(BEST_MODEL_PATH)

# 2. 导出为 ONNX 格式（img sz 需与训练时一致）
export_results = model.export(
    format="onnx",
    imgsz=IMG_SIZE,
    batch=1,  # 部署时通常用批量1
    opset=ONNX_OPSET,  # ONNX算子集版本，兼容大多数框架
    simplify=ONNX_SIMPLIFY,  # 简化ONNX模型，减小体积并提升推理速度
)

print(f"✅ ONNX模型导出完成！保存路径：{export_results}")
# 导出的模型默认保存到：runs/detect/train4/weights/best.onnx
