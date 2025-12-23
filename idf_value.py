from ultralytics import YOLO
import os
import numpy as np  # 导入numpy，用于数组处理（可选，若需要复杂操作）

if __name__ == '__main__':
    # 1. 加载模型（原有路径保持不变）
    MODEL_PATH = r"C:\Users\ASUS\PycharmProjects\YOLOv12\runs\detect\train4\weights\best.pt"
    DATA_YAML_PATH = r"D:\dataset\WIDER_FACE_YOLO\data.yaml"
    model = YOLO(MODEL_PATH)

    # 2. 执行全面评估（保留原有所有参数配置）
    metrics = model.val(
        data=DATA_YAML_PATH,
        imgsz=640,
        batch=2,
        conf=0.5,
        iou=0.5,
        save_json=False,
        save_conf=True,
        save_txt=False
    )

    # 3. 处理数组类型指标，转换为标量后格式化输出
    print(f"✅ 评估完成！核心指标如下：")
    # 方式1：若为单类别任务，用.item()提取数组中的唯一标量（推荐）
    precision = metrics.box.p.item()  # 从numpy数组转换为Python float标量
    recall = metrics.box.r.item()
    map50 = metrics.box.map50.item()
    map50_95 = metrics.box.map.item()

    # 方式2：若为多类别任务，取所有类别的平均值（按需选择）
    # precision = metrics.box.p.mean()
    # recall = metrics.box.r.mean()
    # map50 = metrics.box.map50.mean()
    # map50_95 = metrics.box.map.mean()

    # 正常格式化打印
    print(f"精确率（Precision）：{precision:.4f}")
    print(f"召回率（Recall）：{recall:.4f}")
    print(f"mAP50：{map50:.4f}")
    print(f"mAP50-95：{map50_95:.4f}")
    print(f"✅ 评估报告保存到：runs/detect/val/")