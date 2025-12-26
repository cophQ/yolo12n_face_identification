# YOLO 图像识别大模型
## 环境快速配置
1. 先下载[uv包管理器](https://docs.astral.sh/uv/getting-started/installation/)到电脑上
   
   如果是Linux/MacOS, 在终端运行:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   如果是Windows环境:
   ```bash
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   下载后重新打开终端以刷新缓存和环境变量。
2. 激活虚拟环境并下载需要的依赖
   
    如果是Linux/MacOS环境:
    ```bash
    uv sync && source .venv/bin/activate
    ```
    如果是Windows环境：
    ```bash
    uv sync && .venv\Scripts\activate
    ```

创建虚拟环境后注意将Python解释器选择为**虚拟环境下的Python解释器**。

## 配置文件设置
本项目使用`.env`文件来管理配置，避免硬编码路径，提高跨平台兼容性。请认真阅读`.env.example`来进行环境环境变量的设置。

### 配置步骤

1. 创建配置文件:
   ```bash
   cp .env.example .env
   ```
   **!!注意确保`.gitignore`中包含`.env`文件，不要造成泄露!!**
2. 编辑配置文件
   
   根据你的操作系统和实际情况修改`.env`文件中的配置项：
   | 类型     | 配置项          | 描述                            |
   | -------- | --------------- | ------------------------------- |
   | **必须** | DATASET_ROOT    | 数据集根目录路径                |
   | **必须** | BEST_MODEL_PATH | 最佳模型权重文件路径            |
   | **必须** | VIDEO_PATH      | 视频文件路径                    |
   | 可选     | 训练参数        | EPOCHS、BATCH_SIZE等            |
   | 可选     | 推理参数        | CONF_THRESHOLD、IOU_THRESHOLD等 |
   | 可选     | 其他路径配置    |                                 |

推荐使用相对路径和正斜杠(`/`)。程序启动时会自动验证必须配置项和路径是否存在，如果配置错误会显示详细的错误信息。
