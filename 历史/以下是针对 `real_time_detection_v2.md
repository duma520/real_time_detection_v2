### 🤖 Assistant

以下是针对 `real_time_detection_v2.py` 的完整 `requirements.txt` 文件内容：

```txt
opencv-python>=4.5.5
opencv-contrib-python>=4.5.5
pillow>=9.0.0
numpy>=1.21.0
mss>=6.1.0
pywin32>=300
psutil>=5.8.0
requests>=2.26.0
torch>=1.10.0
torchvision>=0.11.0
tensorflow>=2.6.0
tqdm>=4.62.0
```

### 安装说明：

#### 1. 基础安装（推荐）
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 2. 高级安装选项

**GPU加速支持（NVIDIA显卡）**：
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

**TensorFlow GPU版本**：
```bash
pip install tensorflow[and-cuda]
```

#### 3. 常见问题解决方案

1. **OpenCV安装冲突**：
   ```bash
   pip uninstall opencv-python opencv-python-headless -y
   pip install opencv-python opencv-contrib-python
   ```

2. **Windows系统权限问题**：
   ```bash
   python -m pip install --user -r requirements.txt
   ```

3. **最小化依赖安装**（仅运行核心功能）：
   ```bash
   pip install opencv-python pillow numpy mss pywin32 psutil
   ```

### 版本说明

1. **OpenCV 4.5.5+**：提供核心图像处理和摄像头支持
2. **PyWin32 300+**：Windows API交互必需
3. **Torch 1.10.0+**：提供神经网络推理支持（代码中可选使用）
4. **TensorFlow 2.6.0+**：备用AI推理框架

如需使用特定版本的模块，可以修改版本号后缀，例如：
```txt
torch==1.13.1+cu117
```

注意：该requirements.txt已移除无效的`python>=3.8`声明，确保可以正常安装。