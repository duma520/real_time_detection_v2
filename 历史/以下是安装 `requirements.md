### 🤖 Assistant

以下是安装 `requirements.txt` 的完整步骤指南（适用于 Windows/Linux/macOS 系统）：

---

### **1. 准备工作**
#### 1.1 确保已安装 Python（推荐 3.8-3.10）
```bash
python --version  # 检查版本
```

#### 1.2 升级 pip（可选但推荐）
```bash
pip install --upgrade pip
```

---

### **2. 安装依赖**
#### 方法一：直接安装（适合大多数用户）
```bash
pip install -r requirements.txt
```

#### 方法二：分步安装（若遇到兼容性问题）
```bash
# 先安装基础依赖
pip install opencv-python numpy Pillow tk torch torchvision tensorflow

# 再安装加速后端（根据硬件选择）
pip install onnxruntime-gpu pycuda numba openvino

# 其他工具
pip install psutil pywin32 mss requests tqdm
```

---

### **3. 解决常见问题**
#### ▶ **GPU 加速支持**
- **CUDA 安装**（NVIDIA 显卡必需）：
  1. 确认显卡驱动已安装：`nvidia-smi`
  2. 安装对应版本的 CUDA 和 cuDNN：
     - [CUDA 官方指南](https://developer.nvidia.com/cuda-downloads)
     - [cuDNN 安装](https://developer.nvidia.com/cudnn)

- **TensorRT 安装**（可选）：
  参考 [TensorRT 文档](https://developer.nvidia.com/tensorrt)

#### ▶ **OpenVINO 警告**
如果看到 `openvino.runtime` 弃用警告：
```python
import warnings
warnings.filterwarnings("ignore", message="The `openvino.runtime` module is deprecated")
```

#### ▶ **权限问题**
- Linux/macOS 用户可加 `--user`：
  ```bash
  pip install --user -r requirements.txt
  ```

---

### **4. 验证安装**
运行以下代码检查关键库是否正常：
```python
import cv2, torch, tensorflow as tf
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("TensorFlow:", tf.__version__, "GPU:", tf.config.list_physical_devices('GPU'))
```

---

### **5. 卸载（如需重装）**
```bash
pip freeze | xargs pip uninstall -y
```

---

### **附：各组件作用说明**
| 库名称 | 用途 | 是否必需 |
|--------|------|----------|
| `opencv-python` | 图像处理 | ✔️ |
| `torch/tensorflow` | 深度学习框架 | ✔️ |
| `onnxruntime-gpu` | ONNX 模型加速 | 可选 |
| `openvino` | Intel 加速 | 可选 |
| `pycuda` | CUDA 接口 | 可选（GPU 需要） |

遇到问题可提供完整的错误日志，我会协助排查！