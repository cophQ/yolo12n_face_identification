from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback
import os

# ==================== 核心配置（仅需改这部分！） ====================
# 转换后的data.yaml路径（关键！指向D:\dataset\WIDER_FACE_YOLO\data.yaml）
DATA_YAML_PATH = r"D:\dataset\WIDER_FACE_YOLO\data.yaml"
# 模型选择：修正为yolov12s（中等模型），拼写完整（添加v，替换n为s）
MODEL_TYPE = "yolo12n.pt"  # 关键修改：yolo12n.pt → yolov12s.pt
# 训练参数：适配yolov12s（显存占用更高，批次大小可选调低，8G显存建议4）
EPOCHS = 100          # 训练轮数（新手先跑10轮验证流程，再跑100轮）
BATCH_SIZE = 6        # 关键调整：从8调低到6（若显存不足可改为4，24G显存可保留8/16）
IMG_SIZE = 640        # 输入图片尺寸（人脸小目标可试800/1024，需匹配显存）
DEVICE = 0            # GPU编号（CPU填"cpu"，多GPU填[0,1]）
# SwanLab项目配置
SWANLAB_PROJECT = "YOLOv12-WIDER-FACE"  # 项目名
SWANLAB_EXP_NAME = "yolov12s-train-100e"  # 实验名：同步修改为yolov12s
# ==================================================================

def check_data_yaml():
    """检查data.yaml是否存在"""
    if not os.path.exists(DATA_YAML_PATH):
        raise FileNotFoundError(f"❌ 找不到data.yaml文件，请检查路径：{DATA_YAML_PATH}")
    print(f"✅ 找到data.yaml：{DATA_YAML_PATH}")

if __name__ == "__main__":
    # 1. 检查配置文件
    check_data_yaml()

    # 2. 加载YOLOv12s预训练模型（修正后将自动下载正确权重）
    #model = YOLO(MODEL_TYPE)
    #print(f"✅ 加载模型成功：{MODEL_TYPE}")

    ### 改为断点续训模式（添加下面两行）
    # *1. 确定last.pt的完整路径（建议使用绝对路径）
    RESUME_WEIGHTS_PATH = r"C:\Users\ASUS\PycharmProjects\YOLOv12\runs\detect\train4\weights\last.pt"

    # *2. 使用last.pt加载模型
    model = YOLO(RESUME_WEIGHTS_PATH)

    # 3. 集成SwanLab回调（自动记录训练指标）
    add_swanlab_callback(
        model,
        project=SWANLAB_PROJECT,
        experiment_name=SWANLAB_EXP_NAME,
        description="YOLOv12s训练WIDER FACE（手动终止后续训）",
    )

    # 4. 开始训练（针对人脸检测优化参数，解决AMP报错）
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        resume=True,  # 关键参数，启用断点续训
        # 人脸检测专属优化参数
        lr0=0.001,               # 初始学习率（中等模型建议0.001，与n一致即可）
        lrf=0.01,                # 学习率衰减因子
        box=7.0,                 # 框损失权重（人脸小目标增大权重）
        cls=0.5,                 # 分类损失权重
        weight_decay=0.0005,     # 权重衰减（防过拟合）
        warmup_epochs=3,         # 热身轮数（适应学习率）
        augment=True,            # 开启数据增强（默认True，必开）
        hsv_h=0.015,             # 色调增强（小幅，避免人脸颜色失真）
        hsv_s=0.7,               # 饱和度增强
        hsv_v=0.4,               # 明度增强
        mosaic=1.0,              # 马赛克增强（密集人脸适配）
        mixup=0.0,               # 混合增强（人脸不建议开，易混淆）
        copy_paste=0.0,          # 复制粘贴增强（同上）
        patience=10,             # 早停（10轮没提升就停止，防过拟合）
        save=True,               # 保存最佳模型
        val=True,                # 训练中验证（必开，SwanLab记录验证指标）
        amp=False,               # 关键修改：从True改为False，禁用AMP，解决初始化报错（新手优先）
        # 若你已确认GPU启用且PyTorch≥2.0.0，可改回amp=True
        workers=2                # 数据加载线程（Windows建议≤4）
    )

    # 5. 训练完成后自动验证（输出详细指标）
    print("\n===== 开始验证最佳模型 =====")
    metrics = model.val()  # 验证最佳权重（runs/detect/train/weights/best.pt）
    # 打印核心指标（mAP50是人脸检测关键）
    print(f"✅ 验证完成 | mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")

    # 6. 导出模型（可选，用于部署）
    print("\n===== 导出ONNX格式模型 =====")
    model.export(format="onnx", imgsz=IMG_SIZE)  # 导出到runs/detect/train/weights/
    print("✅ 模型导出完成！")