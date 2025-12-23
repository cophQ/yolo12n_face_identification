import os
import shutil
import chardet
from PIL import Image
from pathlib import Path

# ==================== 核心配置（务必修改！） ====================
# 原始WIDER FACE数据集解压路径
WIDER_ROOT = r"D:\dataset\WIDER_FACE"
# 转换后YOLO格式数据集保存路径
YOLO_SAVE_ROOT = r"D:\dataset\WIDER_FACE_YOLO"
# ===============================================================

# 定义需要转换的数据集类型（训练集+验证集，测试集无标注无需转换）
DATA_TYPES = {
    "train": {
        "annot_path": os.path.join(WIDER_ROOT, "wider_face_split", "wider_face_train_bbx_gt.txt"),
        "img_root": os.path.join(WIDER_ROOT, "WIDER_train", "images"),
    },
    "val": {
        "annot_path": os.path.join(WIDER_ROOT, "wider_face_split", "wider_face_val_bbx_gt.txt"),
        "img_root": os.path.join(WIDER_ROOT, "WIDER_val", "images"),
    }
}

def auto_detect_encoding(file_path):
    """
    暴力检测文件编码（核心函数）
    :param file_path: 待检测文件路径
    :return: 检测到的编码（如gbk/utf-8），失败则返回gbk（WIDER FACE默认编码）
    """
    try:
        with open(file_path, "rb") as f:
            # 读取前10000字节（足够检测编码，避免读大文件卡顿）
            raw_data = f.read(10000)
            # chardet检测编码
            result = chardet.detect(raw_data)
            encoding = result["encoding"]
            # 处理检测失败的情况（兜底用gbk）
            if encoding is None or encoding == "ascii":
                encoding = "gbk"
            # 统一编码名称（如GB2312→gbk，cp1252→gbk）
            encoding = encoding.lower().replace("gb2312", "gbk").replace("cp1252", "gbk")
        print(f"✅ 文件 {os.path.basename(file_path)} 编码检测结果：{encoding}")
        return encoding
    except Exception as e:
        print(f"⚠️ 编码检测失败，兜底使用GBK | 错误：{e}")
        return "gbk"

def read_annot_file(file_path):
    """
    按自动检测的编码读取标注文件，避免乱码
    """
    encoding = auto_detect_encoding(file_path)
    try:
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            # errors="ignore"：忽略少量无法解码的字符（避免脚本崩溃）
            lines = [line.strip() for line in f if line.strip()]  # 去除空行和首尾空格
        return lines
    except Exception as e:
        print(f"❌ 读取文件失败，尝试用GBK重新读取 | 错误：{e}")
        with open(file_path, "r", encoding="gbk", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

def create_yolo_dirs():
    """创建YOLO标准目录结构"""
    dirs = [
        Path(YOLO_SAVE_ROOT) / "images" / "train",
        Path(YOLO_SAVE_ROOT) / "images" / "val",
        Path(YOLO_SAVE_ROOT) / "labels" / "train",
        Path(YOLO_SAVE_ROOT) / "labels" / "val",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 已创建YOLO目录结构：{YOLO_SAVE_ROOT}")

def convert_wider_to_yolo(data_type_param, annot_lines_param, img_root):
    """
    将WIDER FACE标注转换为YOLO格式
    :param data_type_param: train/val
    :param annot_lines_param: 读取后的标注文件行列表
    :param img_root: 对应数据集的图片根目录
    """
    img_save_dir = Path(YOLO_SAVE_ROOT) / "images" / data_type_param
    label_save_dir = Path(YOLO_SAVE_ROOT) / "labels" / data_type_param

    i = 0
    total_lines = len(annot_lines_param)
    success_count = 0  # 成功转换的图片-标注对数
    fail_count = 0     # 失败数

    print(f"\n🚀 开始转换 {data_type_param} 集（共{total_lines}行标注）...")

    while i < total_lines:
        # 1. 读取图片相对路径（标注文件中每行图片路径后紧跟人脸数量）
        img_rel_path = annot_lines_param[i]
        i += 1
        if i >= total_lines:
            break

        # 2. 读取人脸数量（处理可能的数字解析错误）
        try:
            num_faces = int(annot_lines_param[i])
            i += 1
        except ValueError:
            print(f"⚠️ 人脸数量解析失败，跳过该图片 | 行内容：{annot_lines_param[i]}")
            i += 1
            fail_count += 1
            continue

        # 3. 拼接图片绝对路径，检查图片是否存在
        img_abs_path = Path(img_root) / img_rel_path
        if not img_abs_path.exists():
            print(f"⚠️ 图片不存在，跳过 | 路径：{img_abs_path}")
            i += num_faces  # 跳过后续的人脸标注行
            fail_count += 1
            continue

        # 4. 读取图片尺寸（用于坐标归一化）
        try:
            with Image.open(img_abs_path) as img:
                img_w, img_h = img.size
            if img_w == 0 or img_h == 0:
                print(f"⚠️ 图片尺寸异常，跳过 | 路径：{img_abs_path}")
                i += num_faces
                fail_count += 1
                continue
        except Exception as e:
            print(f"⚠️ 读取图片尺寸失败，跳过 | 路径：{img_abs_path} | 错误：{e}")
            i += num_faces
            fail_count += 1
            continue

        # 5. 复制图片到YOLO目录（保留原文件名）
        img_save_path = img_save_dir / img_abs_path.name
        shutil.copy2(img_abs_path, img_save_path)  # copy2保留文件元信息

        # 6. 生成YOLO格式标注文件
        label_save_path = label_save_dir / (img_abs_path.stem + ".txt")
        with open(label_save_path, "w", encoding="utf-8") as f:
            for _ in range(num_faces):
                if i >= total_lines:
                    break
                # 读取单个人脸标注行：x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
                face_annot = annot_lines_param[i].split()
                i += 1

                # 过滤无效标注（invalid=1表示无效人脸，跳过）
                if len(face_annot) >= 8 and face_annot[7] == "1":
                    continue

                # 解析坐标（x1,y1是左上角坐标，w,h是宽高）
                try:
                    x1 = float(face_annot[0])
                    y1 = float(face_annot[1])
                    w = float(face_annot[2])
                    h = float(face_annot[3])
                except (ValueError, IndexError):
                    print(f"⚠️ 坐标解析失败，跳过该人脸 | 标注行：{annot_lines_param[i-1]}")
                    continue

                # 转换为YOLO格式（归一化中心坐标+宽高）
                x_center = (x1 + w/2) / img_w
                y_center = (y1 + h/2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                # 过滤异常坐标（避免归一化后超出0-1范围）
                if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                    continue
                if norm_w < 0 or norm_w > 1 or norm_h < 0 or norm_h > 1:
                    continue

                # 写入YOLO标注（类别0=人脸，保留6位小数）
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        # 7. 统计成功数
        success_count += 1

        # 8. 进度提示（每处理1000张图片打印一次）
        if success_count % 1000 == 0:
            print(f"📈 进度：已处理 {success_count} 张图片 | 失败 {fail_count} 张")

    # 打印转换统计
    print(f"\n✅ {data_type_param} 集转换完成 | 成功：{success_count} 张 | 失败：{fail_count} 张")
    print(f"📂 图片保存路径：{img_save_dir}")
    print(f"📂 标注保存路径：{label_save_dir}")

def generate_data_yaml():
    """自动生成YOLO训练用的data.yaml文件"""
    yaml_path = Path(YOLO_SAVE_ROOT) / "data.yaml"
    # 提前处理路径的反斜杠替换，避免在f-string中使用反斜杠
    train_path = str(Path(YOLO_SAVE_ROOT) / "images" / "train").replace("\\", "/")
    val_path = str(Path(YOLO_SAVE_ROOT) / "images" / "val").replace("\\", "/")
    # 构造yaml内容（不再在f-string内使用反斜杠）
    yaml_content = f"""# WIDER FACE YOLO格式数据集配置
train: {train_path}
val: {val_path}

# 类别配置
nc: 1  # 仅人脸一个类别
names: ['face']  # 类别名称（对应标注中的0类）
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n📄 已生成data.yaml文件：{yaml_path}")

if __name__ == "__main__":
    # 1. 创建YOLO目录
    create_yolo_dirs()

    # 2. 遍历训练集/验证集，逐个转换
    for data_type, config in DATA_TYPES.items():
        # 读取标注文件（自动检测编码）
        annot_lines = read_annot_file(config["annot_path"])
        # 转换为YOLO格式
        convert_wider_to_yolo(data_type, annot_lines, config["img_root"])

    # 3. 自动生成data.yaml
    generate_data_yaml()

    # 4. 最终提示
    print(f"\n🎉 所有转换完成！")
    print(f"📌 YOLO数据集路径：{YOLO_SAVE_ROOT}")
    print(f"📌 训练时只需指定data.yaml路径即可：{Path(YOLO_SAVE_ROOT) / 'data.yaml'}")