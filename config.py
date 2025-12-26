"""
项目配置文件 - 统一管理所有路径
支持从.env文件和环境变量读取配置
"""

import os
from pathlib import Path
from typing import Union, Optional
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()


class ConfigError(Exception):
    """配置错误异常"""

    pass


def validate_required_configs():
    """验证必须配置的参数"""
    required_configs = {
        "DATASET_ROOT": "数据集根目录路径",
        "BEST_MODEL_PATH": "最佳模型权重文件路径",
        "VIDEO_PATH": "视频路径（设置为 0 可使用摄像头）",
    }

    missing_configs = []
    for config_name, description in required_configs.items():
        if not os.getenv(config_name):
            missing_configs.append(f"{config_name}: {description}")

    if missing_configs:
        raise ConfigError(
            "缺少以下必须配置项，请在.env文件中设置：\n"
            + "\n".join(f"  • {config}" for config in missing_configs)
            + "\n\n参考.env.example文件了解详细配置方法。"
        )


def validate_paths():
    """验证路径是否存在"""
    # 验证数据集路径
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"数据集路径不存在: {DATASET_ROOT}")

    # 验证最佳模型路径
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"最佳模型权重文件不存在: {BEST_MODEL_PATH}")


def get_env_path(
    env_name: str, default_path: Optional[str] = None, required: bool = False
) -> Path:
    """
    从环境变量获取路径，如果不存在则使用默认值
    Args:
        env_name: 环境变量名
        default_path: 默认路径（可选）
        required: 是否必须配置，如果为True且未配置则抛出异常
    Returns:
        Path对象
    Raises:
        ConfigError: 当required为True且未配置时
    """
    env_value = os.getenv(env_name)
    if env_value:
        return Path(env_value)
    elif default_path:
        return Path(default_path)
    elif required:
        raise ConfigError(
            f"缺少必要的环境变量: {env_name}\n"
            f"请在.env文件中设置此变量，参考.env.example文件了解详细配置方法。"
        )
    else:
        raise ConfigError(f"缺少必要的环境变量: {env_name}")


def get_env_int(env_name: str, default_value: int) -> int:
    """从环境变量获取整数，如果不存在则使用默认值"""
    env_value = os.getenv(env_name)
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            raise ConfigError(f"环境变量 {env_name} 必须是整数: {env_value}")
    return default_value


def get_env_float(env_name: str, default_value: float) -> float:
    """从环境变量获取浮点数，如果不存在则使用默认值"""
    env_value = os.getenv(env_name)
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            raise ConfigError(f"环境变量 {env_name} 必须是数字: {env_value}")
    return default_value


def get_env_bool(env_name: str, default_value: bool) -> bool:
    """从环境变量获取布尔值，如果不存在则使用默认值"""
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.lower() in ("true", "1", "yes", "on")
    return default_value


# ==================== 数据集路径 ====================
# WIDER FACE YOLO格式数据集根目录
# 注意：由于不同操作系统的默认路径不同，建议在.env文件中明确设置
DATASET_ROOT = get_env_path(
    "DATASET_ROOT", None, required=True  # 不提供默认值，强制用户在.env中配置
)
# data.yaml文件路径（自动生成）
DATA_YAML_PATH = DATASET_ROOT / "data.yaml"

# ==================== 模型路径 ====================
# 预训练模型类型（用于从头训练）
MODEL_TYPE = os.getenv("MODEL_TYPE", "yolo12n.pt")

# 已训练模型权重路径，建议在.env文件中明确设置
BEST_MODEL_PATH = get_env_path(
    "BEST_MODEL_PATH",
    None,
    required=True,
    # 不提供默认值，强制用户在.env中配置
)
LAST_MODEL_PATH = get_env_path("LAST_MODEL_PATH", None, required=False)

# ==================== 训练配置 ====================
# 训练参数
EPOCHS = get_env_int("EPOCHS", 100)
BATCH_SIZE = get_env_int("BATCH_SIZE", 6)
IMG_SIZE = get_env_int("IMG_SIZE", 640)
DEVICE = os.getenv("DEVICE", "0")  # GPU编号，使用CPU改为 "cpu"

# SwanLab项目配置
SWANLAB_PROJECT = os.getenv("SWANLAB_PROJECT", "YOLOv12-WIDER-FACE")
SWANLAB_EXP_NAME = os.getenv("SWANLAB_EXP_NAME", "yolo12n-train-100e")

# ==================== 推理配置 ====================
# 视频路径（设置为 "0" 可使用摄像头）
VIDEO_PATH = get_env_path(
    "VIDEO_PATH", None, required=True  # 不提供默认值，强制用户在.env中配置
)

# 推理参数
CONF_THRESHOLD = get_env_float("CONF_THRESHOLD", 0.5)
IOU_THRESHOLD = get_env_float("IOU_THRESHOLD", 0.5)


# ==================== 导出配置 ====================
# ONNX导出参数
ONNX_OPSET = get_env_int("ONNX_OPSET", 12)
ONNX_SIMPLIFY = get_env_bool("ONNX_SIMPLIFY", True)

# ==================== 配置验证 ====================
# 验证必须配置项
try:
    validate_required_configs()
    validate_paths()
except (ConfigError, FileNotFoundError) as e:
    print(f"配置验证失败: {e}")
    raise
