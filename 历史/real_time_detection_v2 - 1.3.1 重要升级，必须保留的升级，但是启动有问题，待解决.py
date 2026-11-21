import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from mss import mss
import win32gui
import win32process
import psutil
import time
import platform
import sys
import subprocess
import threading
import queue
import importlib
import inspect
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, Tuple, List
import collections
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN自定义操作
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制使用 CPU
import json
import requests
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==================== 常量定义 ====================
VERSION = "v2.0.0"
RELEASE_DATE = "2025-04-01"
AUTHOR = "杜玛"
COPYRIGHT = "Copyright © 杜玛. All rights reserved."

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    "YOLOv8": {
        "repo_url": "https://github.com/ultralytics/ultralytics",
        "weights_dir": "models/yolov8",
        "model_size": ["n", "s", "m", "l", "x"]
    },
    "MobileNetV3": {
        "repo_url": "https://github.com/tensorflow/models",
        "weights_dir": "models/mobilenetv3",
        "versions": ["large", "small"]
    },
    "DeepLabV3+": {
        "repo_url": "https://github.com/tensorflow/models",
        "weights_dir": "models/deeplab",
        "backbone": ["mobilenetv3", "xception"]
    }
}

# ==================== 自定义数据集类 ====================
class ChangeDetectionDataset(Dataset):
    """变化检测数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_pairs = self._load_image_pairs()
        
    def _load_image_pairs(self):
        """加载图像对"""
        image_pairs = []
        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                images = sorted(os.listdir(subdir_path))
                for i in range(0, len(images)-1, 2):
                    if i+1 < len(images):
                        img1_path = os.path.join(subdir_path, images[i])
                        img2_path = os.path.join(subdir_path, images[i+1])
                        mask_path = os.path.join(subdir_path, "mask_"+images[i].split('.')[0]+".png")
                        if os.path.exists(mask_path):
                            image_pairs.append((img1_path, img2_path, mask_path))
        return image_pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, mask_path = self.image_pairs[idx]
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        mask = cv2.imread(mask_path, 0)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            mask = self.transform(mask)
            
        return img1, img2, mask

# ==================== 工具函数 ====================
def get_latest_release_info(repo_url):
    """获取GitHub仓库的最新发布信息"""
    api_url = repo_url.replace("github.com", "api.github.com/repos") + "/releases/latest"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"获取最新发布信息失败: {str(e)}")
        return None

def download_file(url: str, destination: str) -> bool:
    """下载文件到指定路径"""
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"下载文件失败: {str(e)}")
        return False

def check_file_integrity(file_path: str, expected_hash: str = None) -> bool:
    """检查文件完整性"""
    if not os.path.exists(file_path):
        return False
    
    if expected_hash is None:
        return True  # 如果没有提供哈希值，只检查文件是否存在
    
    # 计算文件哈希
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)  # 64KB chunks
            if not data:
                break
            sha256.update(data)
    file_hash = sha256.hexdigest()
    
    return file_hash == expected_hash

def backup_file(file_path: str) -> bool:
    """备份文件"""
    if not os.path.exists(file_path):
        return False
    
    backup_path = f"{file_path}.bak"
    try:
        shutil.copy2(file_path, backup_path)
        return True
    except Exception as e:
        print(f"备份文件失败: {str(e)}")
        return False

def get_latest_model_weights(model_name: str, model_size: str = None) -> Tuple[str, str]:
    """获取最新模型权重文件信息"""
    if model_name not in MODEL_CONFIG:
        return None, None
    
    config = MODEL_CONFIG[model_name]
    
    if model_name == "YOLOv8":
        # 获取YOLO最新版本信息
        latest_release = get_latest_release_info(config["repo_url"])
        if not latest_release:
            return None, None
            
        # 查找权重文件
        assets = latest_release.get("assets", [])
        model_size = model_size or "s"  # 默认使用small版本
        
        for asset in assets:
            if f"yolov8{model_size}.pt" in asset["name"].lower():
                return asset["browser_download_url"], latest_release["tag_name"]
    
    elif model_name == "MobileNetV3":
        # MobileNetV3直接从TensorFlow获取
        version = config["versions"][0]  # 默认使用large版本
        weights_url = f"https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_{version}_224_tf.h5"
        return weights_url, f"v3_{version}"

    elif model_name == "DeepLabV3+":
        # DeepLabV3+权重
        backbone = config["backbone"][0]  # 默认backbone
        weights_url = f"https://storage.googleapis.com/tensorflow/models/deeplabv3_{backbone}_cityscapes_train_2018_02_06.tar.gz"
        return weights_url, "v3+"
    
    return None, None

def ensure_model_files(model_name: str, model_size: str = None) -> Tuple[bool, str]:
    """确保模型文件存在且有效，返回(是否成功, 版本号)"""
    if model_name not in MODEL_CONFIG:
        return False, None
    
    config = MODEL_CONFIG[model_name]
    weights_url, version = get_latest_model_weights(model_name, model_size)
    
    if not weights_url:
        print(f"无法获取{model_name}最新权重文件URL")
        return False, None
    
    # 确定本地保存路径
    weights_dir = config["weights_dir"]
    os.makedirs(weights_dir, exist_ok=True)
    
    if model_name == "YOLO":
        weights_filename = f"yolov5{model_size}.pt" if model_size else "yolov5s.pt"
    elif model_name == "MobileNet":
        weights_filename = weights_url.split("/")[-1]
    
    weights_path = os.path.join(weights_dir, weights_filename)
    
    # 检查权重文件是否存在且有效
    if os.path.exists(weights_path):
        print(f"{model_name} 权重文件已存在")
        return True, version
    
    # 下载权重文件
    print(f"正在从 {weights_url} 下载权重文件...")
    if download_file(weights_url, weights_path):
        print(f"{model_name} 权重文件下载成功")
        return True, version
    else:
        print(f"{model_name} 权重文件下载失败")
        return False, None

def get_model_version(model_name: str) -> str:
    """获取模型版本号"""
    if model_name not in MODEL_CONFIG:
        return "unknown"
    
    config = MODEL_CONFIG[model_name]
    
    if model_name == "YOLO":
        latest_release = get_latest_release_info(config["repo_url"])
        return latest_release["tag_name"] if latest_release else "unknown"
    elif model_name == "MobileNet":
        return f"v{config['versions'][-1]}"
    
    return "unknown"

# ==================== 加速后端抽象基类 ====================
class AccelerationBackend(ABC):
    """加速后端抽象基类"""
    def __init__(self):
        self.name = "CPU"
        self.initialized = False
        self.backend_info = {}
        self.priority = 0  # 优先级，数字越大优先级越高
        self.training = False
        self.training_progress = 0.0
        self.training_status = "未训练"
        self.model_version = "unknown"
        self.train_params = {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "data_path": "data/train"
        }
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化加速后端"""
        pass
    
    @abstractmethod
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """处理帧并返回阈值化差异图像"""
        pass
    
    @abstractmethod
    def release(self):
        """释放资源"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """获取后端信息"""
        return self.backend_info
    
    def train_model(self, params):
        """训练模型"""
        self.train_params.update(params)
        self.training = True
        self.training_progress = 0.0
        self.training_status = "训练中"
        
        # 启动训练线程
        threading.Thread(target=self._train_thread, daemon=True).start()
        return True
    
    def _train_thread(self):
        """训练线程"""
        try:
            # 准备数据
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ChangeDetectionDataset(self.train_params["data_path"], transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.train_params["batch_size"], shuffle=True)
            
            # 训练循环
            total_steps = len(dataloader) * self.train_params["epochs"]
            current_step = 0
            
            for epoch in range(self.train_params["epochs"]):
                for batch_idx, (img1, img2, mask) in enumerate(dataloader):
                    if not self.training:
                        break
                    
                    # 在这里添加实际的训练逻辑
                    # 模拟训练进度
                    current_step += 1
                    self.training_progress = current_step / total_steps
                    self.training_status = f"训练中 Epoch {epoch+1}/{self.train_params['epochs']} - {batch_idx+1}/{len(dataloader)}"
                    
                    time.sleep(0.01)  # 模拟训练时间
                
                if not self.training:
                    break
            
            if self.training:
                self.training_status = "训练完成"
                self.save_trained_model()
            self.training = False
            
        except Exception as e:
            self.training_status = f"训练失败: {str(e)}"
            self.training = False
    
    def save_trained_model(self):
        """保存训练好的模型"""
        model_dir = f"trained_models/{self.name}"
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{model_dir}/model_{timestamp}.weights"
        
        # 实际应用中这里应该保存模型权重
        with open(model_path, 'w') as f:
            f.write(f"{self.name} model trained at {timestamp}")
        
        return model_path
    
    def get_training_progress(self):
        """获取训练进度"""
        return {
            "training": self.training,
            "progress": self.training_progress,
            "status": self.training_status
        }
    
    def stop_training(self):
        """停止训练"""
        self.training = False
        self.training_status = "训练已停止"

# ==================== 具体加速后端实现 ====================
class YOLOv8Backend(AccelerationBackend):
    def __init__(self):
        super().__init__()
        self.name = "YOLOv8"
        self.priority = 9  # 高优先级
        self.model_path = "models/yolov8/"
        self.model = None
        self.device = None
        self.model_size = "s"  # 默认使用small模型
    
    def initialize(self) -> bool:
        try:
            from ultralytics import YOLO
            import torch
            
            self.backend_info["yolov8_version"] = YOLO.version
            
            if not torch.cuda.is_available():
                return False
                
            # 自动获取最新模型
            success, version = ensure_model_files("YOLOv8", self.model_size)
            if not success:
                return False
                
            self.model_version = version
            
            # 加载模型
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            weights_path = os.path.join(self.model_path, f"yolov8{self.model_size}.pt")
            self.model = YOLO(weights_path)
            self.model.to(self.device)
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"YOLOv8初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.model is None:
            raise RuntimeError("YOLOv8后端未初始化")
        
        # 推理帧1
        results1 = self.model(frame1, verbose=False)
        # 推理帧2
        results2 = self.model(frame2, verbose=False)
        
        # 创建差异掩码
        h, w = frame1.shape[:2]
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        
        # 绘制检测框到掩码
        for box in results1[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(mask1, (x1, y1), (x2, y2), 255, -1)
        
        for box in results2[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(mask2, (x1, y1), (x2, y2), 255, -1)
        
        # 计算差异
        diff = cv2.absdiff(mask1, mask2)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def release(self):
        """释放资源"""
        if self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'device') and self.device is not None:
            import torch
            torch.cuda.empty_cache()
    
class MobileNetV3Backend(AccelerationBackend):
    def __init__(self):
        super().__init__()
        self.name = "MobileNetV3"
        self.priority = 6
        self.model_path = "models/mobilenetv3/"
        self.model_version = "large"  # 默认使用large版本
        self.model = None
    
    def initialize(self) -> bool:
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Concatenate
            
            self.backend_info["tf_version"] = tf.__version__
            
            # 检查GPU可用性
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return False
            
            # 构建变化检测模型
            input1 = Input(shape=(224, 224, 3))
            input2 = Input(shape=(224, 224, 3))
            
            if self.model_version == "large":
                base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            else:
                base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            # 冻结基础模型
            base_model.trainable = False
            
            # 处理两个输入
            x1 = base_model(input1)
            x1 = GlobalAveragePooling2D()(x1)
            
            x2 = base_model(input2)
            x2 = GlobalAveragePooling2D()(x2)
            
            # 合并特征
            merged = Concatenate()([x1, x2])
            predictions = Dense(1, activation='sigmoid')(merged)
            
            self.model = Model(inputs=[input1, input2], outputs=predictions)
            
            # 编译模型
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"MobileNetV3初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.model is None:
            raise RuntimeError("MobileNetV3后端未初始化")
        
        # 预处理帧
        img1 = cv2.resize(frame1, (224, 224))
        img1 = tf.keras.applications.mobilenet_v3.preprocess_input(img1)
        img1 = np.expand_dims(img1, axis=0)
        
        img2 = cv2.resize(frame2, (224, 224))
        img2 = tf.keras.applications.mobilenet_v3.preprocess_input(img2)
        img2 = np.expand_dims(img2, axis=0)
        
        # 获取特征图
        feature_extractor = tf.keras.Model(inputs=self.model.input, 
                                         outputs=self.model.layers[-2].output)
        features1 = feature_extractor.predict(img1)
        features2 = feature_extractor.predict(img2)
        
        # 计算差异
        diff = np.abs(features1 - features2).sum(axis=-1)
        diff = (diff * 255).astype(np.uint8)
        diff = cv2.resize(diff, (frame1.shape[1], frame1.shape[0]))
        
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def release(self):
        """释放资源"""
        if self.model is not None:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            self.model = None

class DeepLabV3Backend(AccelerationBackend):
    def __init__(self):
        super().__init__()
        self.name = "DeepLabV3+"
        self.priority = 7
        self.model_path = "models/deeplab/"
        self.backbone = "mobilenetv3"  # 默认backbone
        self.model = None
    
    def initialize(self) -> bool:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from deeplab import DeepLabV3Plus  # 需要安装deeplab库
            
            self.backend_info["tf_version"] = tf.__version__
            
            # 检查GPU可用性
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return False
            
            # 构建DeepLabV3+模型
            self.model = DeepLabV3Plus(
                input_shape=(512, 512, 3),
                backbone=self.backbone,
                num_classes=1  # 二分类问题
            )
            
            # 加载预训练权重
            weights_path = os.path.join(self.model_path, f"deeplab_{self.backbone}.h5")
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"DeepLabV3+初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.model is None:
            raise RuntimeError("DeepLabV3+后端未初始化")
        
        # 预处理帧
        img1 = cv2.resize(frame1, (512, 512))
        img1 = img1 / 127.5 - 1.0  # 归一化到[-1, 1]
        img1 = np.expand_dims(img1, axis=0)
        
        img2 = cv2.resize(frame2, (512, 512))
        img2 = img2 / 127.5 - 1.0
        img2 = np.expand_dims(img2, axis=0)
        
        # 计算差异
        pred1 = self.model.predict(img1)[0]
        pred2 = self.model.predict(img2)[0]
        
        diff = np.abs(pred1 - pred2).mean(axis=-1)
        diff = (diff * 255).astype(np.uint8)
        diff = cv2.resize(diff, (frame1.shape[1], frame1.shape[0]))
        
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def release(self):
        """释放资源"""
        if self.model is not None:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            self.model = None


class PyTorchBackend(AccelerationBackend):
    """PyTorch GPU加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "PyTorch"
        self.priority = 9
        self.device = None
        self.model = None
        self.criterion = None
        self.optimizer = None
    
    def initialize(self) -> bool:
        try:
            import torch
            self.backend_info["torch_version"] = torch.__version__
            
            if not torch.cuda.is_available():
                return False
                
            self.device = torch.device('cuda')
            # 测试CUDA是否真的可用
            test_tensor = torch.tensor([1.0]).cuda()
            
            # 初始化模型
            self.model = SimpleChangeDetector().to(self.device)
            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"PyTorch初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.device is None:
            raise RuntimeError("PyTorch后端未初始化")
        
        import torch
        tensor1 = torch.from_numpy(frame1).permute(2, 0, 1).float().to(self.device)
        tensor2 = torch.from_numpy(frame2).permute(2, 0, 1).float().to(self.device)
        
        gray1 = (0.299 * tensor1[0] + 0.587 * tensor1[1] + 0.114 * tensor1[2]).to(torch.uint8)
        gray2 = (0.299 * tensor2[0] + 0.587 * tensor2[1] + 0.114 * tensor2[2]).to(torch.uint8)
        
        diff = torch.abs(gray1.int() - gray2.int())
        thresh = (diff > threshold).to(torch.uint8) * 255
        
        return thresh.cpu().numpy()
    
    def _train_thread(self):
        """PyTorch训练线程"""
        try:
            # 准备数据
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ChangeDetectionDataset(self.train_params["data_path"], transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.train_params["batch_size"], shuffle=True)
            
            # 设置优化器
            if self.train_params["optimizer"] == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_params["learning_rate"])
            elif self.train_params["optimizer"] == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.train_params["learning_rate"])
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_params["learning_rate"])
            
            # 训练循环
            total_steps = len(dataloader) * self.train_params["epochs"]
            current_step = 0
            
            for epoch in range(self.train_params["epochs"]):
                self.model.train()
                running_loss = 0.0
                
                for batch_idx, (img1, img2, mask) in enumerate(dataloader):
                    if not self.training:
                        break
                    
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    mask = mask.to(self.device)
                    
                    # 清零梯度
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(img1, img2)
                    loss = self.criterion(outputs, mask.float())
                    
                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()
                    
                    # 更新进度
                    running_loss += loss.item()
                    current_step += 1
                    self.training_progress = current_step / total_steps
                    self.training_status = f"训练中 Epoch {epoch+1}/{self.train_params['epochs']} - Loss: {running_loss/(batch_idx+1):.4f}"
                    
                if not self.training:
                    break
            
            if self.training:
                self.training_status = "训练完成"
                self.save_trained_model()
            self.training = False
            
        except Exception as e:
            self.training_status = f"训练失败: {str(e)}"
            self.training = False
    
    def save_trained_model(self):
        """保存PyTorch模型"""
        model_dir = f"trained_models/{self.name}"
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{model_dir}/model_{timestamp}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_params': self.train_params
        }, model_path)
        
        return model_path
    
    def release(self):
        if hasattr(self, 'device') and self.device is not None:
            import torch
            torch.cuda.empty_cache()

class SimpleChangeDetector(nn.Module):
    """简单的变化检测模型"""
    def __init__(self):
        super(SimpleChangeDetector, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x

class CUDABackend(AccelerationBackend):
    """OpenCV CUDA加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "CUDA"
        self.priority = 8
        self.stream = None
        self.gpu_frame1 = None
        self.gpu_frame2 = None
    
    def initialize(self) -> bool:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                return False
            
            self.stream = cv2.cuda_Stream()
            self.gpu_frame1 = cv2.cuda_GpuMat()
            self.gpu_frame2 = cv2.cuda_GpuMat()
            self.backend_info["cuda_devices"] = cv2.cuda.getCudaEnabledDeviceCount()
            self.initialized = True
            return True
        except Exception as e:
            print(f"CUDA初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.stream is None:
            raise RuntimeError("CUDA后端未初始化")
        
        self.gpu_frame1.upload(frame1, self.stream)
        self.gpu_frame2.upload(frame2, self.stream)
        
        gpu_gray1 = cv2.cuda.cvtColor(self.gpu_frame1, cv2.COLOR_BGR2GRAY, stream=self.stream)
        gpu_gray2 = cv2.cuda.cvtColor(self.gpu_frame2, cv2.COLOR_BGR2GRAY, stream=self.stream)
        
        gpu_diff = cv2.cuda.absdiff(gpu_gray1, gpu_gray2, stream=self.stream)
        _, gpu_thresh = cv2.cuda.threshold(gpu_diff, threshold, 255, cv2.THRESH_BINARY, stream=self.stream)
        
        thresh = gpu_thresh.download(stream=self.stream)
        self.stream.waitForCompletion()
        return thresh
    
    def release(self):
        if self.stream is not None:
            self.stream = None
        self.gpu_frame1 = None
        self.gpu_frame2 = None

class OpenCLBackend(AccelerationBackend):
    """OpenCL加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "OpenCL"
        self.priority = 7
    
    def initialize(self) -> bool:
        try:
            if not cv2.ocl.haveOpenCL():
                return False
            
            cv2.ocl.setUseOpenCL(True)
            if not cv2.ocl.useOpenCL():
                return False
            
            self.backend_info["opencl_available"] = True
            self.initialized = True
            return True
        except Exception as e:
            print(f"OpenCL初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("OpenCL后端未初始化")
        
        umat1 = cv2.UMat(frame1)
        umat2 = cv2.UMat(frame2)
        
        gray1 = cv2.cvtColor(umat1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(umat2, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        return cv2.UMat.get(thresh) if isinstance(thresh, cv2.UMat) else thresh
    
    def release(self):
        cv2.ocl.setUseOpenCL(False)
        self.initialized = False

class NumbaBackend(AccelerationBackend):
    """Numba JIT加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "Numba"
        self.priority = 6
    
    def initialize(self) -> bool:
        try:
            import numba
            from numba import jit
            self.backend_info["numba_version"] = numba.__version__
            
            @jit(nopython=True, nogil=True)
            def numba_process(frame1, frame2, threshold):
                diff = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))
                return (diff > threshold).astype(np.uint8) * 255
            
            self._numba_process = numba_process
            self.initialized = True
            return True
        except Exception as e:
            print(f"Numba初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Numba后端未初始化")
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        return self._numba_process(gray1, gray2, threshold)
    
    def release(self):
        pass

class TensorRTBackend(AccelerationBackend):
    """TensorRT加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "TensorRT"
        self.priority = 10
        self.model_path = "models/tensorrt/"
        self.train_params = {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "Adam"
        }
        self.model = None
        self.engine = None
    
    def initialize(self) -> bool:
        try:
            import tensorrt as trt
            self.backend_info["tensorrt_version"] = trt.__version__
            
            # 检查是否有可用的GPU
            if not self._check_cuda_available():
                return False
                
            # 自动获取最新模型
            success, version = ensure_model_files("YOLO")
            if success:
                self.model_version = version
                
            self.initialized = True
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"TensorRT初始化失败: {str(e)}")
            return False
    
    def _check_cuda_available(self):
        try:
            import pycuda.driver as cuda
            cuda.init()
            return True
        except:
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        # 简化的TensorRT实现，实际使用时需要构建完整的引擎
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def _train_thread(self):
        """TensorRT训练线程"""
        try:
            # 准备数据
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ChangeDetectionDataset(self.train_params["data_path"], transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.train_params["batch_size"], shuffle=True)
            
            # 训练循环
            total_steps = len(dataloader) * self.train_params["epochs"]
            current_step = 0
            
            for epoch in range(self.train_params["epochs"]):
                for batch_idx, (img1, img2, mask) in enumerate(dataloader):
                    if not self.training:
                        break
                    
                    # 在这里添加实际的TensorRT训练逻辑
                    # 模拟训练进度
                    current_step += 1
                    self.training_progress = current_step / total_steps
                    self.training_status = f"训练中 Epoch {epoch+1}/{self.train_params['epochs']} - {batch_idx+1}/{len(dataloader)}"
                    
                    time.sleep(0.01)  # 模拟训练时间
                
                if not self.training:
                    break
            
            if self.training:
                self.training_status = "训练完成"
                self.save_trained_model()
            self.training = False
            
        except Exception as e:
            self.training_status = f"训练失败: {str(e)}"
            self.training = False
    
    def validate_model(self):
        """验证模型"""
        print("TensorRT 模型验证")
        # 这里添加实际验证代码
        return True
    
    def test_model(self):
        """测试模型"""
        print("TensorRT 模型测试")
        # 这里添加实际测试代码
        return True
    
    def release(self):
        if self.engine is not None:
            self.engine.destroy()
            self.engine = None

class ONNXBackend(AccelerationBackend):
    """ONNX Runtime加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "ONNX"
        self.priority = 7
        self.model_path = "models/onnx/"
        self.train_params = {
            "batch_size": 16,
            "epochs": 50,
            "learning_rate": 0.0005,
            "optimizer": "SGD"
        }
        self.session = None
    
    def initialize(self) -> bool:
        try:
            import onnxruntime as ort
            self.backend_info["onnx_version"] = ort.__version__
            
            # 检查是否有可用的GPU
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                self.provider = 'CUDAExecutionProvider'
            elif 'CPUExecutionProvider' in providers:
                self.provider = 'CPUExecutionProvider'
            else:
                return False
                
            # 自动获取最新模型
            success, version = ensure_model_files("YOLO")
            if success:
                self.model_version = version
                
            self.initialized = True
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"ONNX初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        # 简化的ONNX实现，实际使用时需要加载预训练模型
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def _train_thread(self):
        """ONNX训练线程"""
        try:
            # 准备数据
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ChangeDetectionDataset(self.train_params["data_path"], transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.train_params["batch_size"], shuffle=True)
            
            # 训练循环
            total_steps = len(dataloader) * self.train_params["epochs"]
            current_step = 0
            
            for epoch in range(self.train_params["epochs"]):
                for batch_idx, (img1, img2, mask) in enumerate(dataloader):
                    if not self.training:
                        break
                    
                    # 在这里添加实际的ONNX训练逻辑
                    # 模拟训练进度
                    current_step += 1
                    self.training_progress = current_step / total_steps
                    self.training_status = f"训练中 Epoch {epoch+1}/{self.train_params['epochs']} - {batch_idx+1}/{len(dataloader)}"
                    
                    time.sleep(0.01)  # 模拟训练时间
                
                if not self.training:
                    break
            
            if self.training:
                self.training_status = "训练完成"
                self.save_trained_model()
            self.training = False
            
        except Exception as e:
            self.training_status = f"训练失败: {str(e)}"
            self.training = False
    
    def validate_model(self):
        """验证模型"""
        print("ONNX 模型验证")
        # 这里添加实际验证代码
        return True
    
    def test_model(self):
        """测试模型"""
        print("ONNX 模型测试")
        # 这里添加实际测试代码
        return True
    
    def release(self):
        if self.session is not None:
            del self.session
            self.session = None

class YOLOBackend(AccelerationBackend):
    """YOLO目标检测后端"""
    def __init__(self):
        super().__init__()
        self.name = "YOLO"
        self.priority = 8
        self.model_path = "models/yolo/"
        self.train_params = {
            "batch_size": 8,
            "epochs": 300,
            "learning_rate": 0.0001,
            "optimizer": "AdamW"
        }
        self.model = None
        self.device = None
        self.model_size = "s"  # 默认使用small模型
    
    def initialize(self) -> bool:
        try:
            import torch
            from models.experimental import attempt_load
            self.backend_info["yolo_supported"] = True
            
            if not torch.cuda.is_available():
                return False
            
            # 自动获取最新模型
            success, version = ensure_model_files("YOLO", self.model_size)
            if not success:
                return False
                
            self.model_version = version
            
            # 加载模型
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            weights_path = os.path.join(self.model_path, f"yolov5{self.model_size}.pt")
            self.model = attempt_load(weights_path, device=self.device)
            self.model.eval()
            
            self.initialized = True
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"YOLO初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.model is None:
            raise RuntimeError("YOLO后端未初始化")
        
        import torch
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression
        
        # 预处理帧1
        img1 = letterbox(frame1, new_shape=640)[0]
        img1 = img1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img1 = np.ascontiguousarray(img1)
        img1 = torch.from_numpy(img1).to(self.device)
        img1 = img1.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img1.ndimension() == 3:
            img1 = img1.unsqueeze(0)
        
        # 预处理帧2
        img2 = letterbox(frame2, new_shape=640)[0]
        img2 = img2[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img2 = np.ascontiguousarray(img2)
        img2 = torch.from_numpy(img2).to(self.device)
        img2 = img2.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img2.ndimension() == 3:
            img2 = img2.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            pred1 = self.model(img1)[0]
            pred2 = self.model(img2)[0]
        
        # NMS
        pred1 = non_max_suppression(pred1, 0.25, 0.45)
        pred2 = non_max_suppression(pred2, 0.25, 0.45)
        
        # 创建差异掩码
        h, w = frame1.shape[:2]
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        
        # 绘制检测框到掩码
        for det in pred1[0]:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], frame1.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(mask1, (x1, y1), (x2, y2), 255, -1)
        
        for det in pred2[0]:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img2.shape[2:], det[:, :4], frame2.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(mask2, (x1, y1), (x2, y2), 255, -1)
        
        # 计算差异
        diff = cv2.absdiff(mask1, mask2)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def _train_thread(self):
        """YOLO训练线程"""
        try:
            import yaml
            from yolov5.train import run
            
            # 准备YOLO训练配置
            data_yaml = {
                'train': os.path.join(self.train_params["data_path"], 'train'),
                'val': os.path.join(self.train_params["data_path"], 'val'),
                'nc': 1,  # 变化检测只有一类
                'names': ['change']
            }
            
            # 保存YAML文件
            yaml_path = os.path.join(self.train_params["data_path"], 'change_detection.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(data_yaml, f)
            
            # 设置训练参数
            train_args = {
                'weights': os.path.join(self.model_path, f"yolov5{self.model_size}.pt"),
                'data': yaml_path,
                'epochs': self.train_params["epochs"],
                'batch-size': self.train_params["batch_size"],
                'img-size': 640,
                'device': '0' if torch.cuda.is_available() else 'cpu',
                'optimizer': self.train_params["optimizer"],
                'project': 'runs/train',
                'name': f'yolov5_change_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            }
            
            # 启动YOLO训练
            run(**train_args)
            
            self.training_status = "训练完成"
            self.save_trained_model()
            self.training = False
            
        except Exception as e:
            self.training_status = f"训练失败: {str(e)}"
            self.training = False
    
    def validate_model(self):
        """验证模型"""
        print("YOLO 模型验证")
        # 这里添加实际验证代码
        return True
    
    def test_model(self):
        """测试模型"""
        print("YOLO 模型测试")
        # 这里添加实际测试代码
        return True
    
    def release(self):
        if self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'device') and self.device is not None:
            import torch
            torch.cuda.empty_cache()

class MobileNetBackend(AccelerationBackend):
    """MobileNet轻量级后端"""
    def __init__(self):
        super().__init__()
        self.name = "MobileNet"
        self.priority = 6
        self.model_path = "models/mobilenet/"
        self.train_params = {
            "batch_size": 64,
            "epochs": 150,
            "learning_rate": 0.0002,
            "optimizer": "RMSprop"
        }
        self.model = None
        self.model_version = "v2.0"  # 默认使用2.0版本
    
    def initialize(self) -> bool:
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Concatenate
            
            self.backend_info["tf_version"] = tf.__version__
            
            # 检查是否有可用的GPU
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return False
            
            # 自动获取最新模型
            success, version = ensure_model_files("MobileNet")
            if not success:
                return False
                
            self.model_version = version
            
            # 构建变化检测模型
            input1 = Input(shape=(224, 224, 3))
            input2 = Input(shape=(224, 224, 3))
            
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            # 冻结基础模型
            base_model.trainable = False
            
            # 处理第一个输入
            x1 = base_model(input1)
            x1 = GlobalAveragePooling2D()(x1)
            
            # 处理第二个输入
            x2 = base_model(input2)
            x2 = GlobalAveragePooling2D()(x2)
            
            # 合并特征
            merged = Concatenate()([x1, x2])
            
            # 添加分类层
            predictions = Dense(1, activation='sigmoid')(merged)
            
            self.model = Model(inputs=[input1, input2], outputs=predictions)
            
            # 编译模型
            if self.train_params["optimizer"] == "Adam":
                optimizer = optimizers.Adam(learning_rate=self.train_params["learning_rate"])
            elif self.train_params["optimizer"] == "SGD":
                optimizer = optimizers.SGD(learning_rate=self.train_params["learning_rate"])
            else:
                optimizer = optimizers.RMSprop(learning_rate=self.train_params["learning_rate"])
            
            self.model.compile(optimizer=optimizer,
                              loss=losses.BinaryCrossentropy(),
                              metrics=['accuracy'])
            
            self.initialized = True
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"MobileNet初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.model is None:
            raise RuntimeError("MobileNet后端未初始化")
        
        import tensorflow as tf
        
        # 预处理帧1
        img1 = cv2.resize(frame1, (224, 224))
        img1 = tf.keras.applications.mobilenet_v2.preprocess_input(img1)
        img1 = np.expand_dims(img1, axis=0)
        
        # 预处理帧2
        img2 = cv2.resize(frame2, (224, 224))
        img2 = tf.keras.applications.mobilenet_v2.preprocess_input(img2)
        img2 = np.expand_dims(img2, axis=0)
        
        # 获取特征图
        feature_extractor = tf.keras.Model(inputs=self.model.input, 
                                         outputs=self.model.layers[-2].output)
        features1 = feature_extractor.predict(img1)
        features2 = feature_extractor.predict(img2)
        
        # 计算差异
        diff = np.abs(features1 - features2).sum(axis=-1)
        diff = (diff * 255).astype(np.uint8)
        diff = cv2.resize(diff, (frame1.shape[1], frame1.shape[0]))
        
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def _train_thread(self):
        """MobileNet训练线程"""
        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            # 数据增强
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )
            
            # 训练数据生成器
            train_generator = datagen.flow_from_directory(
                self.train_params["data_path"],
                target_size=(224, 224),
                batch_size=self.train_params["batch_size"],
                class_mode='binary',
                subset='training'
            )
            
            # 验证数据生成器
            val_generator = datagen.flow_from_directory(
                self.train_params["data_path"],
                target_size=(224, 224),
                batch_size=self.train_params["batch_size"],
                class_mode='binary',
                subset='validation'
            )
            
            # 训练模型
            steps_per_epoch = train_generator.samples // self.train_params["batch_size"]
            validation_steps = val_generator.samples // self.train_params["batch_size"]
            
            total_steps = steps_per_epoch * self.train_params["epochs"]
            current_step = 0
            
            # 训练回调
            class TrainingCallback(tf.keras.callbacks.Callback):
                def __init__(self, backend):
                    self.backend = backend
                
                def on_epoch_begin(self, epoch, logs=None):
                    if not self.backend.training:
                        self.model.stop_training = True
                
                def on_batch_end(self, batch, logs=None):
                    if not self.backend.training:
                        self.model.stop_training = True
                    current_step = (epoch * steps_per_epoch) + batch
                    self.backend.training_progress = current_step / total_steps
                    self.backend.training_status = f"训练中 Epoch {epoch+1}/{self.backend.train_params['epochs']} - Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f}"
            
            history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=self.train_params["epochs"],
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=[TrainingCallback(self)]
            )
            
            if self.training:
                self.training_status = "训练完成"
                self.save_trained_model()
            self.training = False
            
        except Exception as e:
            self.training_status = f"训练失败: {str(e)}"
            self.training = False
    
    def save_trained_model(self):
        """保存MobileNet模型"""
        model_dir = f"trained_models/{self.name}"
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{model_dir}/model_{timestamp}.h5"
        
        self.model.save(model_path)
        return model_path
    
    def validate_model(self):
        """验证模型"""
        print("MobileNet 模型验证")
        # 这里添加实际验证代码
        return True
    
    def test_model(self):
        """测试模型"""
        print("MobileNet 模型测试")
        # 这里添加实际测试代码
        return True
    
    def release(self):
        if self.model is not None:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            self.model = None

class OpenVINOBackend(AccelerationBackend):
    """OpenVINO加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "OpenVINO"
        self.priority = 9
        self.model_path = "models/openvino/"
        self.train_params = {
            "batch_size": 24,
            "epochs": 200,
            "learning_rate": 0.0003,
            "optimizer": "Adagrad"
        }
        self.core = None
        self.model = None
    
    def initialize(self) -> bool:
        try:
            from openvino import Core
            self.core = Core()
            devices = self.core.available_devices
            self.backend_info["openvino_devices"] = devices
            
            if not devices:
                return False
                
            # 自动获取最新模型
            success, version = ensure_model_files("YOLO")
            if success:
                self.model_version = version
                
            self.initialized = True
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"OpenVINO初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        # 简化的OpenVINO实现
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def _train_thread(self):
        """OpenVINO训练线程"""
        try:
            # 准备数据
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ChangeDetectionDataset(self.train_params["data_path"], transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.train_params["batch_size"], shuffle=True)
            
            # 训练循环
            total_steps = len(dataloader) * self.train_params["epochs"]
            current_step = 0
            
            for epoch in range(self.train_params["epochs"]):
                for batch_idx, (img1, img2, mask) in enumerate(dataloader):
                    if not self.training:
                        break
                    
                    # 在这里添加实际的OpenVINO训练逻辑
                    # 模拟训练进度
                    current_step += 1
                    self.training_progress = current_step / total_steps
                    self.training_status = f"训练中 Epoch {epoch+1}/{self.train_params['epochs']} - {batch_idx+1}/{len(dataloader)}"
                    
                    time.sleep(0.01)  # 模拟训练时间
                
                if not self.training:
                    break
            
            if self.training:
                self.training_status = "训练完成"
                self.save_trained_model()
            self.training = False
            
        except Exception as e:
            self.training_status = f"训练失败: {str(e)}"
            self.training = False
    
    def validate_model(self):
        """验证模型"""
        print("OpenVINO 模型验证")
        # 这里添加实际验证代码
        return True
    
    def test_model(self):
        """测试模型"""
        print("OpenVINO 模型测试")
        # 这里添加实际测试代码
        return True
    
    def release(self):
        if self.model is not None:
            del self.model
            self.model = None

class CPUBackend(AccelerationBackend):
    """CPU后备方案"""
    def __init__(self):
        super().__init__()
        self.name = "CPU"
        self.priority = 1
    
    def initialize(self) -> bool:
        self.initialized = True
        return True
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def release(self):
        pass

# ==================== 加速后端管理器 ====================
class AccelerationManager:
    """加速后端管理器"""
    def __init__(self):
        # 按优先级排序的后端列表
        self.backends = [
            TensorRTBackend(),
            OpenVINOBackend(),
            YOLOv8Backend(),
            PyTorchBackend(),
            DeepLabV3Backend(),
            CUDABackend(),
            ONNXBackend(),
            OpenCLBackend(),
            MobileNetV3Backend(),
            NumbaBackend(),
            CPUBackend()
        ]
        self.current_backend: Optional[AccelerationBackend] = None
        self.backend_combinations = []
    
    def detect_available_backends(self) -> List[AccelerationBackend]:
        """检测所有可用的加速后端"""
        available = []
        for backend in self.backends:
            if backend.initialize():
                available.append(backend)
                print(f"检测到加速后端: {backend.name} (版本: {backend.model_version})")
                print(f"后端信息: {backend.get_info()}")
        
        if not available:
            print("未检测到加速后端，使用CPU")
            available.append(self.backends[-1])  # 添加CPU后端
        
        return available
    
    def detect_best_backend(self) -> AccelerationBackend:
        """检测并返回最佳可用的加速后端"""
        available = self.detect_available_backends()
        if not available:
            return self.backends[-1]  # 返回CPU后端
        
        # 按优先级排序
        available.sort(key=lambda x: x.priority, reverse=True)
        return available[0]
    
    def detect_optimal_combination(self) -> List[AccelerationBackend]:
        """检测最优的后端组合"""
        available = self.detect_available_backends()
        if not available:
            return [self.backends[-1]]  # 返回CPU后端
        
        # 简单的组合策略：选择前3个最高优先级的后端
        available.sort(key=lambda x: x.priority, reverse=True)
        return available[:3]
    
    def set_backend(self, backend_name: str) -> bool:
        """设置特定的加速后端"""
        for backend in self.backends:
            if backend.name.lower() == backend_name.lower():
                if backend.initialize():
                    if self.current_backend:
                        self.current_backend.release()
                    self.current_backend = backend
                    return True
        return False
    
    def set_backend_combination(self, backends: List[str]) -> bool:
        """设置后端组合"""
        selected = []
        for name in backends:
            for backend in self.backends:
                if backend.name.lower() == name.lower() and backend.initialize():
                    selected.append(backend)
        
        if not selected:
            return False
        
        if self.current_backend:
            self.current_backend.release()
        
        self.backend_combinations = selected
        return True
    
    def get_current_backend(self) -> AccelerationBackend:
        """获取当前加速后端"""
        if self.current_backend is None:
            self.current_backend = self.detect_best_backend()
        return self.current_backend
    
    def release_all(self):
        """释放所有后端资源"""
        for backend in self.backends:
            backend.release()
        self.current_backend = None
        self.backend_combinations = []

# ==================== 算法管理器 ====================
class AlgorithmManager:
    """算法管理器，提供多种处理算法和组合"""
    def __init__(self):
        self.current_algorithm = "智能组合"
        self.available_algorithms = {
            "原始设置": self.original_algorithm,
            "高斯模糊": self.gaussian_blur_algorithm,
            "背景减除": self.background_subtraction_algorithm,
            "形态学处理": self.morphological_algorithm,
            "光流法": self.optical_flow_algorithm,
            "帧差分增强": self.frame_diff_enhanced_algorithm,
            "多尺度检测": self.multi_scale_algorithm,
            "深度学习": self.deep_learning_algorithm
        }
        
        # 算法组合预设
        self.algorithm_presets = {
            "快速检测": ["高斯模糊", "帧差分增强"],
            "精确检测": ["背景减除", "形态学处理"],
            "运动追踪": ["光流法", "多尺度检测"],
            "智能组合": []  # 动态选择
        }
        
        # 初始化背景减除器
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
        # 光流法参数
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # 智能组合缓存
        self.smart_combo_cache = None
        self.last_combo_time = 0
    
    def set_algorithm(self, algorithm_name: str):
        """设置当前使用的算法"""
        if algorithm_name in self.available_algorithms:
            self.current_algorithm = algorithm_name
            return True
        return False
    
    def set_algorithm_combo(self, combo_name: str):
        """设置算法组合"""
        if combo_name in self.algorithm_presets:
            self.current_algorithm = combo_name
            return True
        return False
    
    def smart_select_algorithm(self):
        """智能选择最佳算法组合"""
        current_time = time.time()
        if self.smart_combo_cache and current_time - self.last_combo_time < 10:
            return self.smart_combo_cache
        
        # 根据系统性能动态选择
        cpu_load = psutil.cpu_percent()
        mem_load = psutil.virtual_memory().percent
        
        if cpu_load > 70 or mem_load > 80:
            combo = ["高斯模糊", "帧差分增强"]  # 轻量级组合
        else:
            combo = ["背景减除", "形态学处理"]  # 精确组合
        
        self.smart_combo_cache = combo
        self.last_combo_time = current_time
        return combo
    
    def process_frame(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """处理帧的核心方法"""
        if self.current_algorithm == "智能组合":
            combo = self.smart_select_algorithm()
            result = frame1.copy()
            for algo in combo:
                result = self.available_algorithms[algo](result, frame2, threshold)
            return result
        elif self.current_algorithm in self.algorithm_presets:
            combo = self.algorithm_presets[self.current_algorithm]
            result = frame1.copy()
            for algo in combo:
                result = self.available_algorithms[algo](result, frame2, threshold)
            return result
        else:
            return self.available_algorithms[self.current_algorithm](frame1, frame2, threshold)
    
    # ===== 算法实现 =====
    def original_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """原始算法"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def gaussian_blur_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """高斯模糊预处理"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def background_subtraction_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """背景减除算法"""
        fg_mask = self.backSub.apply(frame2)
        _, thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def morphological_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """形态学处理算法"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh
    
    def optical_flow_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """光流法优化"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **self.flow_params)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def frame_diff_enhanced_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """增强型帧差分算法"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        diff = cv2.multiply(diff, 1.5)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def multi_scale_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """多尺度检测算法"""
        scales = [1.0, 0.75, 0.5]
        results = []
        
        for scale in scales:
            if scale != 1.0:
                resized1 = cv2.resize(frame1, None, fx=scale, fy=scale)
                resized2 = cv2.resize(frame2, None, fx=scale, fy=scale)
            else:
                resized1 = frame1.copy()
                resized2 = frame2.copy()
                
            gray1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            if scale != 1.0:
                thresh = cv2.resize(thresh, (frame1.shape[1], frame1.shape[0]))
            
            results.append(thresh)
        
        final_thresh = np.zeros_like(results[0])
        for thresh in results:
            final_thresh = cv2.bitwise_or(final_thresh, thresh)
        
        return final_thresh
    
    def deep_learning_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """深度学习算法"""
        # 使用YOLO后端处理
        backend = YOLOBackend()
        if backend.initialize():
            return backend.process_frames(frame1, frame2, threshold)
        else:
            # 如果YOLO不可用，使用MobileNet
            backend = MobileNetBackend()
            if backend.initialize():
                return backend.process_frames(frame1, frame2, threshold)
            else:
                # 如果都没有，使用原始算法
                return self.original_algorithm(frame1, frame2, threshold)

# ==================== 帧处理器 ====================
class FrameProcessor:
    """多线程/多进程帧处理器"""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = True
        self.use_multithread = True
        self.use_multiprocess = False
        self.current_mode = "auto"  # auto/single/multi
    
    def set_processing_mode(self, mode: str):
        """设置处理模式"""
        self.current_mode = mode
        if mode == "multi":
            self.use_multithread = True
            self.use_multiprocess = False
        elif mode == "single":
            self.use_multithread = False
            self.use_multiprocess = False
        elif mode == "auto":
            # 根据系统负载自动选择
            cpu_load = psutil.cpu_percent()
            if cpu_load < 70:
                self.use_multithread = True
                self.use_multiprocess = False
            else:
                self.use_multithread = False
                self.use_multiprocess = False
    
    def process_frame_task(self, frame1, frame2, threshold, backend):
        """线程/进程任务函数"""
        try:
            thresh = backend.process_frames(frame1, frame2, threshold)
            return thresh
        except Exception as e:
            print(f"处理失败: {e}")
            return None
    
    def process_frames(self, frame1, frame2, threshold, backend):
        """处理帧，根据设置选择处理方式"""
        if not self.use_multithread:
            return backend.process_frames(frame1, frame2, threshold)
            
        try:
            self.frame_queue.put((frame1, frame2))
            if not self.result_queue.empty():
                future = self.result_queue.get()
                new_thresh = future.result()
                if new_thresh is not None:
                    if new_thresh.dtype != np.uint8:
                        new_thresh = new_thresh.astype(np.uint8)
                    return new_thresh
            return backend.process_frames(frame1, frame2, threshold)
        except Exception as e:
            print(f"多线程处理失败: {str(e)}，使用单线程")
            return backend.process_frames(frame1, frame2, threshold)
    
    def start_processing(self, threshold, backend):
        """启动处理线程"""
        def worker():
            while self.running:
                try:
                    frame1, frame2 = self.frame_queue.get(timeout=0.1)
                    try:
                        future = self.executor.submit(
                            self.process_frame_task, 
                            frame1, frame2, threshold, backend
                        )
                        self.result_queue.put(future, timeout=0.1)
                    except Exception as e:
                        print(f"任务提交失败: {str(e)}")
                        self.frame_queue.put((frame1, frame2))
                except queue.Empty:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"工作线程错误: {str(e)} - {type(e).__name__}")
                    continue
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
    
    def stop_processing(self):
        """停止处理线程"""
        self.running = False
        self.executor.shutdown(wait=True)
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join()

# ==================== 主应用类 ====================
class ChangeDetectionApp:
    """变化检测主应用"""
    def __init__(self, root):
        try:
            self.root = root
            # 检查Tkinter是否正常
            if not hasattr(root, 'tk'):
                raise RuntimeError("Tkinter初始化失败")
       
            print("设置窗口标题和尺寸...")
            self.root.title(f"智能变化检测系统 ({VERSION} | {AUTHOR} | {COPYRIGHT})")
            self.root.geometry("800x900")
            self.root.minsize(600, 700)
        
            # 初始化变量
            self.threshold = 30
            self.min_contour_area = 500
            self.frame_rate = 30
            self.lock_aspect_ratio = True
            self.monitoring_process = None
            self.monitoring_camera = None
            self.monitor_area_roi = None
            self.performance_mode = "平衡"
            self.adaptive_fps_enabled = True
            self.dynamic_threshold_enabled = False
            self.fade_effect_enabled = False
            self.fade_frames = 10
            self.use_multithread = True
            self.cpu_monitor_enabled = True
            self.show_algorithm_debug = False
            self.current_image = None
            self.last_boxes = []
            self.fade_boxes = []
            self.frame_count = 0
            self.actual_fps = 0
            self.last_update_time = None
            self.last_fps_warning_time = 0
            self.fps_warning_interval = 30
            self.max_fps_limit = 60
            self.min_fps_limit = 5
            self.current_load_factor = 1.0
            self.log_messages = []
            self.max_log_entries = 100
        
            # 初始化管理器
            print("初始化管理器...")
            self.sct = mss()
            self.acceleration_manager = AccelerationManager()
            self.algorithm_manager = AlgorithmManager()
            self.frame_processor = FrameProcessor()
        
            # 创建UI
            print("创建用户界面...")
            self.create_ui()
        
            # 启动帧处理器
            print("启动帧处理器...")
            self.frame_processor.start_processing(self.threshold, self.acceleration_manager.get_current_backend())
        
            # 开始更新帧
            print("开始更新帧...")
            self.update_frame()

        except Exception as e:
            print(f"界面初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
        raise
    
    def create_ui(self):
        """创建用户界面"""
        # 主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 视频显示区域 (占3/4高度)
        self.video_frame = tk.Frame(self.main_frame, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        # 画布
        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 控制面板区域 (固定高度250像素)
        self.control_panel_frame = tk.Frame(self.main_frame, bd=2, relief=tk.RAISED, height=350)
        self.control_panel_frame.pack_propagate(False)  # 固定高度
        self.control_panel_frame.pack(fill=tk.X, pady=0)
        
        # 日志区域 (固定高度100像素)
        self.log_frame = tk.Frame(self.main_frame, bd=2, relief=tk.SUNKEN, height=100)
        self.log_frame.pack_propagate(False)  # 固定高度
        self.log_frame.pack(fill=tk.BOTH, pady=0)
        
        # 控制面板开关按钮
        self.toggle_button = tk.Button(
            self.main_frame, 
            text="▼ 显示控制面板 ▼", 
            command=self.toggle_control_panel,
            relief=tk.RAISED,
            bd=1
        )
        self.toggle_button.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 状态栏
        self.status_bar = tk.Label(self.root, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 创建控制面板内容
        self.create_control_panel()
        
        # 创建日志区域
        self.create_log_panel()
        
        # 控制面板初始状态
        self.control_panel_visible = True
    
    def create_control_panel(self):
        """创建控制面板内容"""
        # 使用Notebook实现多标签页
        self.tab_control = ttk.Notebook(self.control_panel_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # 监控设置标签页
        self.monitor_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.monitor_tab, text="监控")
        
        # 算法设置标签页
        self.algorithm_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.algorithm_tab, text="算法")
        
        # 性能设置标签页
        self.performance_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.performance_tab, text="性能")
        
        # 高级设置标签页
        self.advanced_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.advanced_tab, text="高级")
        
        # 模型训练标签页
        self.model_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.model_tab, text="模型训练")
        
        # 填充各标签页内容
        self.fill_monitor_tab()
        self.fill_algorithm_tab()
        self.fill_performance_tab()
        self.fill_advanced_tab()
        self.fill_model_tab()
    
    def fill_monitor_tab(self):
        """填充监控设置标签页"""
        frame = ttk.Frame(self.monitor_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 使用PanedWindow实现左右分割
        paned = tk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板 - 监控参数
        left_frame = ttk.Frame(paned)
        paned.add(left_frame)
        
        # 监控源选择
        source_frame = ttk.LabelFrame(left_frame, text="监控源", padding=5)
        source_frame.pack(fill=tk.X, pady=2)
        
        self.process_button = ttk.Button(source_frame, text="选择进程", command=self.toggle_process_monitoring)
        self.process_button.pack(side=tk.LEFT, padx=2)
        
        self.camera_button = ttk.Button(source_frame, text="选择镜头", command=self.toggle_camera_monitoring)
        self.camera_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(source_frame, text="选择区域", command=self.select_monitor_area).pack(side=tk.LEFT, padx=2)
        
        # 监控参数设置
        param_frame = ttk.LabelFrame(left_frame, text="监控参数", padding=5)
        param_frame.pack(fill=tk.X, pady=2)
        
        # 帧率控制
        ttk.Label(param_frame, text="帧率:").grid(row=0, column=0, sticky="e")
        self.frame_rate_scale = tk.Scale(param_frame, from_=1, to=120, orient=tk.HORIZONTAL, 
                                       command=lambda v: self.set_frame_rate(int(float(v))))
        self.frame_rate_scale.set(self.frame_rate)
        self.frame_rate_scale.grid(row=0, column=1, sticky="ew")
        
        self.frame_rate_entry = ttk.Entry(param_frame, width=5)
        self.frame_rate_entry.insert(0, str(self.frame_rate))
        self.frame_rate_entry.bind("<Return>", lambda e: self.set_frame_rate(int(e.widget.get())))
        self.frame_rate_entry.grid(row=0, column=2, padx=2)
        
        # 变化阈值
        ttk.Label(param_frame, text="阈值:").grid(row=1, column=0, sticky="e")
        self.threshold_scale = tk.Scale(param_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                      command=lambda v: self.set_threshold(int(float(v))))
        self.threshold_scale.set(self.threshold)
        self.threshold_scale.grid(row=1, column=1, sticky="ew")
        
        self.threshold_entry = ttk.Entry(param_frame, width=5)
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.bind("<Return>", lambda e: self.set_threshold(int(e.widget.get())))
        self.threshold_entry.grid(row=1, column=2, padx=2)
        
        # 最小变化区域
        ttk.Label(param_frame, text="最小区域:").grid(row=2, column=0, sticky="e")
        self.min_area_scale = tk.Scale(param_frame, from_=1, to=1000, orient=tk.HORIZONTAL,
                                     command=lambda v: self.set_min_contour_area(int(float(v))))
        self.min_area_scale.set(self.min_contour_area)
        self.min_area_scale.grid(row=2, column=1, sticky="ew")
        
        self.min_area_entry = ttk.Entry(param_frame, width=5)
        self.min_area_entry.insert(0, str(self.min_contour_area))
        self.min_area_entry.bind("<Return>", lambda e: self.set_min_contour_area(int(e.widget.get())))
        self.min_area_entry.grid(row=2, column=2, padx=2)
        
        # 右侧面板 - 显示设置
        right_frame = ttk.Frame(paned)
        paned.add(right_frame)
        
        # 显示设置
        display_frame = ttk.LabelFrame(right_frame, text="显示设置", padding=5)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        self.lock_aspect_var = tk.BooleanVar(value=self.lock_aspect_ratio)
        ttk.Checkbutton(display_frame, text="锁定宽高比", variable=self.lock_aspect_var,
                       command=lambda: setattr(self, 'lock_aspect_ratio', self.lock_aspect_var.get())).pack(anchor=tk.W)
        
        self.topmost_var = tk.BooleanVar(value=self.root.attributes('-topmost'))
        ttk.Checkbutton(display_frame, text="窗口置顶", variable=self.topmost_var,
                       command=lambda: self.root.attributes('-topmost', self.topmost_var.get())).pack(anchor=tk.W)
        
        self.fade_var = tk.BooleanVar(value=self.fade_effect_enabled)
        ttk.Checkbutton(display_frame, text="渐隐效果", variable=self.fade_var,
                       command=lambda: setattr(self, 'fade_effect_enabled', self.fade_var.get())).pack(anchor=tk.W)
        
        fade_speed_frame = ttk.Frame(display_frame)
        fade_speed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(fade_speed_frame, text="渐隐速度:").pack(side=tk.LEFT)
        self.fade_speed_scale = tk.Scale(fade_speed_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                                       command=lambda v: setattr(self, 'fade_frames', int(v)))
        self.fade_speed_scale.set(self.fade_frames)
        self.fade_speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def fill_algorithm_tab(self):
        """填充算法设置标签页"""
        frame = ttk.Frame(self.algorithm_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 算法选择
        algo_frame = ttk.LabelFrame(frame, text="算法选择", padding=5)
        algo_frame.pack(fill=tk.BOTH, pady=2)
        
        # 算法模式选择
        self.algo_mode = tk.StringVar(value="smart")
        ttk.Radiobutton(algo_frame, text="单一算法", variable=self.algo_mode, value="single").pack(side=tk.LEFT)
        ttk.Radiobutton(algo_frame, text="算法组合", variable=self.algo_mode, value="combo").pack(side=tk.LEFT)
        ttk.Radiobutton(algo_frame, text="智能组合", variable=self.algo_mode, value="smart").pack(side=tk.LEFT)
        
        # 算法选择下拉框
        self.algo_select = ttk.Combobox(algo_frame, state="readonly")
        self.algo_select['values'] = list(self.algorithm_manager.available_algorithms.keys())
        self.algo_select.current(0)
        self.algo_select.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.algo_select.bind("<<ComboboxSelected>>", self.update_algorithm)
        
        # 算法组合选择下拉框
        self.algo_combo_select = ttk.Combobox(algo_frame, state="readonly")
        self.algo_combo_select['values'] = list(self.algorithm_manager.algorithm_presets.keys())
        self.algo_combo_select.current(0)
        self.algo_combo_select.pack_forget()  # 初始隐藏
        
        # 算法模式切换事件
        self.algo_mode.trace("w", self.on_algo_mode_changed)
        
        # 算法调试信息
        self.debug_var = tk.BooleanVar(value=self.show_algorithm_debug)
        ttk.Checkbutton(algo_frame, text="调试信息", variable=self.debug_var,
                       command=lambda: setattr(self, 'show_algorithm_debug', self.debug_var.get())).pack(side=tk.RIGHT)
        
        # 后端选择
        backend_frame = ttk.LabelFrame(frame, text="加速后端", padding=5)
        backend_frame.pack(fill=tk.BOTH, pady=2)
        
        # 获取可用的后端
        available_backends = [b.name for b in self.acceleration_manager.detect_available_backends()]
        
        self.backend_var = tk.StringVar()
        self.backend_menu = ttk.Combobox(backend_frame, textvariable=self.backend_var, state="readonly")
        self.backend_menu['values'] = available_backends
        self.backend_menu.current(0)
        self.backend_menu.pack(fill=tk.X, padx=2)
        self.backend_menu.bind("<<ComboboxSelected>>", lambda e: self.acceleration_manager.set_backend(self.backend_var.get()))
        
        # 后端组合选择
        self.backend_combo_var = tk.StringVar()
        self.backend_combo_menu = ttk.Combobox(backend_frame, textvariable=self.backend_combo_var, state="readonly")
        self.backend_combo_menu['values'] = ["自动组合", "高性能", "平衡", "节能"]
        self.backend_combo_menu.current(0)
        self.backend_combo_menu.pack(fill=tk.X, padx=2)
        self.backend_combo_menu.bind("<<ComboboxSelected>>", self.on_backend_combo_changed)
    
    def fill_performance_tab(self):
        """填充性能设置标签页"""
        frame = ttk.Frame(self.performance_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 性能模式
        perf_frame = ttk.LabelFrame(frame, text="性能模式", padding=5)
        perf_frame.pack(fill=tk.X, pady=2)
        
        self.perf_mode_var = tk.StringVar(value=self.performance_mode)
        ttk.Radiobutton(perf_frame, text="性能优先", variable=self.perf_mode_var, value="性能").pack(side=tk.LEFT)
        ttk.Radiobutton(perf_frame, text="平衡模式", variable=self.perf_mode_var, value="平衡").pack(side=tk.LEFT)
        ttk.Radiobutton(perf_frame, text="画质优先", variable=self.perf_mode_var, value="画质").pack(side=tk.LEFT)
        self.perf_mode_var.trace("w", lambda *_: setattr(self, 'performance_mode', self.perf_mode_var.get()))
        
        # 自适应设置
        adapt_frame = ttk.LabelFrame(frame, text="自适应设置", padding=5)
        adapt_frame.pack(fill=tk.X, pady=2)
        
        self.adaptive_fps_var = tk.BooleanVar(value=self.adaptive_fps_enabled)
        ttk.Checkbutton(adapt_frame, text="自适应帧率", variable=self.adaptive_fps_var,
                       command=lambda: setattr(self, 'adaptive_fps_enabled', self.adaptive_fps_var.get())).pack(side=tk.LEFT)
        
        self.dynamic_thresh_var = tk.BooleanVar(value=self.dynamic_threshold_enabled)
        ttk.Checkbutton(adapt_frame, text="动态阈值", variable=self.dynamic_thresh_var,
                       command=lambda: setattr(self, 'dynamic_threshold_enabled', self.dynamic_thresh_var.get())).pack(side=tk.LEFT)
        
        # 处理模式
        process_frame = ttk.LabelFrame(frame, text="处理模式", padding=5)
        process_frame.pack(fill=tk.X, pady=2)
        
        self.process_mode_var = tk.StringVar(value="auto")
        ttk.Radiobutton(process_frame, text="自动选择", variable=self.process_mode_var, value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(process_frame, text="单线程", variable=self.process_mode_var, value="single").pack(side=tk.LEFT)
        ttk.Radiobutton(process_frame, text="多线程", variable=self.process_mode_var, value="multi").pack(side=tk.LEFT)
        self.process_mode_var.trace("w", lambda *_: self.frame_processor.set_processing_mode(self.process_mode_var.get()))
        
        # CPU监控
        self.cpu_monitor_var = tk.BooleanVar(value=self.cpu_monitor_enabled)
        ttk.Checkbutton(process_frame, text="CPU监控", variable=self.cpu_monitor_var,
                       command=lambda: setattr(self, 'cpu_monitor_enabled', self.cpu_monitor_var.get())).pack(side=tk.LEFT)
    
    def fill_advanced_tab(self):
        """填充高级设置标签页"""
        frame = ttk.Frame(self.advanced_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 帧跳过设置
        skip_frame = ttk.LabelFrame(frame, text="帧跳过", padding=5)
        skip_frame.pack(fill=tk.X, pady=2)
        
        self.frame_skip_var = tk.IntVar(value=0)
        tk.Scale(skip_frame, from_=0, to=5, orient=tk.HORIZONTAL, variable=self.frame_skip_var,
                showvalue=0).pack(fill=tk.X, expand=True)
        ttk.Label(skip_frame, text="跳过帧数: 0").pack()
        self.frame_skip_var.trace("w", lambda *_: skip_frame.children['!label'].config(text=f"跳过帧数: {self.frame_skip_var.get()}"))
        
        # 分辨率调节
        res_frame = ttk.LabelFrame(frame, text="分辨率", padding=5)
        res_frame.pack(fill=tk.X, pady=2)
        
        self.resolution_scale_var = tk.DoubleVar(value=1.0)
        tk.Scale(res_frame, from_=0.3, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                variable=self.resolution_scale_var, showvalue=0).pack(fill=tk.X, expand=True)
        ttk.Label(res_frame, text="分辨率比例: 1.0").pack()
        self.resolution_scale_var.trace("w", lambda *_: res_frame.children['!label'].config(text=f"分辨率比例: {self.resolution_scale_var.get():.1f}"))
        
        # 处理优化
        opt_frame = ttk.LabelFrame(frame, text="处理优化", padding=5)
        opt_frame.pack(fill=tk.X, pady=2)
        
        self.roi_tracking_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="ROI跟踪", variable=self.roi_tracking_var).pack(side=tk.LEFT)
        
        self.dynamic_interval_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="动态间隔", variable=self.dynamic_interval_var).pack(side=tk.LEFT)
    
    def fill_model_tab(self):
        """填充模型训练标签页"""
        frame = ttk.Frame(self.model_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 使用Notebook实现多模型标签页
        model_notebook = ttk.Notebook(frame)
        model_notebook.pack(fill=tk.BOTH, expand=True)
        
        # TensorRT模型训练页
        trt_frame = ttk.Frame(model_notebook)
        model_notebook.add(trt_frame, text="TensorRT")
        self.create_model_training_ui(trt_frame, "TensorRT")
        
        # ONNX模型训练页
        onnx_frame = ttk.Frame(model_notebook)
        model_notebook.add(onnx_frame, text="ONNX")
        self.create_model_training_ui(onnx_frame, "ONNX")
        
        # YOLO模型训练页
        yolo_frame = ttk.Frame(model_notebook)
        model_notebook.add(yolo_frame, text="YOLO")
        self.create_model_training_ui(yolo_frame, "YOLO")
        
        # MobileNet模型训练页
        mobilenet_frame = ttk.Frame(model_notebook)
        model_notebook.add(mobilenet_frame, text="MobileNet")
        self.create_model_training_ui(mobilenet_frame, "MobileNet")
        
        # OpenVINO模型训练页
        openvino_frame = ttk.Frame(model_notebook)
        model_notebook.add(openvino_frame, text="OpenVINO")
        self.create_model_training_ui(openvino_frame, "OpenVINO")
    
    def create_model_training_ui(self, parent, model_name):
        """为特定模型创建训练UI"""
        backend = next(b for b in self.acceleration_manager.backends if b.name == model_name)
        
        # 参数设置区域
        param_frame = ttk.LabelFrame(parent, text="训练参数", padding=5)
        param_frame.pack(fill=tk.X, pady=2)
        
        # 紧凑布局参数输入
        grid_frame = ttk.Frame(param_frame)
        grid_frame.pack(fill=tk.X)
        
        # 批量大小
        ttk.Label(grid_frame, text="批量大小:").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        batch_size_var = tk.IntVar(value=backend.train_params["batch_size"])
        batch_size_entry = ttk.Entry(grid_frame, textvariable=batch_size_var, width=8)
        batch_size_entry.grid(row=0, column=1, sticky="w", padx=2, pady=2)
        
        # 训练轮数
        ttk.Label(grid_frame, text="训练轮数:").grid(row=0, column=2, sticky="e", padx=2, pady=2)
        epochs_var = tk.IntVar(value=backend.train_params["epochs"])
        epochs_entry = ttk.Entry(grid_frame, textvariable=epochs_var, width=8)
        epochs_entry.grid(row=0, column=3, sticky="w", padx=2, pady=2)
        
        # 学习率
        ttk.Label(grid_frame, text="学习率:").grid(row=1, column=0, sticky="e", padx=2, pady=2)
        lr_var = tk.DoubleVar(value=backend.train_params["learning_rate"])
        lr_entry = ttk.Entry(grid_frame, textvariable=lr_var, width=8)
        lr_entry.grid(row=1, column=1, sticky="w", padx=2, pady=2)
        
        # 优化器
        ttk.Label(grid_frame, text="优化器:").grid(row=1, column=2, sticky="e", padx=2, pady=2)
        optimizer_var = tk.StringVar(value=backend.train_params["optimizer"])
        optimizer_combo = ttk.Combobox(grid_frame, textvariable=optimizer_var, 
                                     values=["Adam", "SGD", "RMSprop", "Adagrad", "AdamW"], width=8)
        optimizer_combo.grid(row=1, column=3, sticky="w", padx=2, pady=2)
        
        # 数据路径
        ttk.Label(grid_frame, text="数据路径:").grid(row=2, column=0, sticky="e", padx=2, pady=2)
        data_path_var = tk.StringVar(value=backend.train_params["data_path"])
        data_path_entry = ttk.Entry(grid_frame, textvariable=data_path_var, width=20)
        data_path_entry.grid(row=2, column=1, columnspan=3, sticky="we", padx=2, pady=2)
        
        # 进度条
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.training_progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(progress_frame, variable=self.training_progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, expand=True, padx=5)
        
        self.training_status_var = tk.StringVar(value="准备训练")
        ttk.Label(progress_frame, textvariable=self.training_status_var).pack()
        
        # 按钮区域
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        
        # 训练按钮
        train_btn = ttk.Button(btn_frame, text="训练模型", 
                             command=lambda: self.start_model_training(
                                 model_name,
                                 {
                                     "batch_size": batch_size_var.get(),
                                     "epochs": epochs_var.get(),
                                     "learning_rate": lr_var.get(),
                                     "optimizer": optimizer_var.get(),
                                     "data_path": data_path_var.get()
                                 }
                             ))
        train_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # 验证按钮
        val_btn = ttk.Button(btn_frame, text="验证模型", 
                           command=lambda: self.validate_model(model_name))
        val_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # 测试按钮
        test_btn = ttk.Button(btn_frame, text="测试模型", 
                            command=lambda: self.test_model(model_name))
        test_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # 停止按钮
        stop_btn = ttk.Button(btn_frame, text="停止训练", 
                            command=lambda: self.stop_model_training(model_name))
        stop_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
    
    def start_model_training(self, model_name, params):
        """开始模型训练"""
        backend = next(b for b in self.acceleration_manager.backends if b.name == model_name)
        if backend.train_model(params):
            self.log_message(f"{model_name} 模型训练开始")
            # 启动进度更新线程
            threading.Thread(target=self.update_training_progress, args=(model_name,), daemon=True).start()
        else:
            self.log_message(f"{model_name} 模型训练失败", "ERROR")
    
    def update_training_progress(self, model_name):
        """更新训练进度"""
        backend = next(b for b in self.acceleration_manager.backends if b.name == model_name)
        while backend.training:
            progress = backend.get_training_progress()
            self.training_progress_var.set(progress["progress"] * 100)
            self.training_status_var.set(progress["status"])
            time.sleep(0.1)
        
        # 训练完成后更新状态
        progress = backend.get_training_progress()
        self.training_progress_var.set(progress["progress"] * 100)
        self.training_status_var.set(progress["status"])
    
    def stop_model_training(self, model_name):
        """停止模型训练"""
        backend = next(b for b in self.acceleration_manager.backends if b.name == model_name)
        backend.stop_training()
        self.log_message(f"{model_name} 模型训练已停止")
    
    def validate_model(self, model_name):
        """验证模型"""
        backend = next(b for b in self.acceleration_manager.backends if b.name == model_name)
        if backend.validate_model():
            self.log_message(f"{model_name} 模型验证完成")
        else:
            self.log_message(f"{model_name} 模型验证失败", "ERROR")
    
    def test_model(self, model_name):
        """测试模型"""
        backend = next(b for b in self.acceleration_manager.backends if b.name == model_name)
        if backend.test_model():
            self.log_message(f"{model_name} 模型测试完成")
        else:
            self.log_message(f"{model_name} 模型测试失败", "ERROR")
    
    def create_log_panel(self):
        """创建日志面板"""
        # 日志折叠按钮
        self.log_header = ttk.Frame(self.log_frame, relief=tk.RAISED, borderwidth=1)
        self.log_header.pack(fill=tk.X, side=tk.TOP)
        
        ttk.Label(self.log_header, text="系统日志").pack(side=tk.LEFT, padx=5)
        
        self.log_toggle_btn = ttk.Button(self.log_header, text="▼", width=3, 
                                       command=self.toggle_log_visibility)
        self.log_toggle_btn.pack(side=tk.RIGHT, padx=2)
        
        # 日志文本框
        self.log_text = tk.Text(self.log_frame, wrap=tk.NONE, height=5)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 垂直滚动条
        v_scroll = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=v_scroll.set)
        
        # 水平滚动条
        h_scroll = ttk.Scrollbar(self.log_frame, orient=tk.HORIZONTAL, command=self.log_text.xview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.log_text.config(xscrollcommand=h_scroll.set)
        
        # 日志控制按钮
        btn_frame = tk.Frame(self.log_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=2)
        
        ttk.Button(btn_frame, text="清除", command=self.clear_log).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="复制", command=self.copy_log).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="保存", command=self.save_log).pack(fill=tk.X, pady=1)
        
        # 初始状态为展开
        self.log_visible = True
    
    def toggle_log_visibility(self):
        """切换日志显示状态"""
        if self.log_visible:
            self.log_text.pack_forget()
            self.log_toggle_btn.config(text="▲")
            self.log_visible = False
        else:
            self.log_text.pack(fill=tk.BOTH, expand=True)
            self.log_toggle_btn.config(text="▼")
            self.log_visible = True
    
    def toggle_control_panel(self):
        """切换控制面板显示状态"""
        if self.control_panel_visible:
            self.control_panel_frame.pack_forget()
            self.toggle_button.config(text="▲ 显示控制面板 ▲")
            self.control_panel_visible = False
        else:
            self.control_panel_frame.pack(fill=tk.X, before=self.toggle_button)
            self.toggle_button.config(text="▼ 隐藏控制面板 ▼")
            self.control_panel_visible = True
    
    def log_message(self, message, level="INFO"):
        """记录日志消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # 添加到日志列表
        self.log_messages.append(log_entry)
        if len(self.log_messages) > self.max_log_entries:
            self.log_messages.pop(0)
        
        # 更新日志显示
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 同时在状态栏显示重要消息
        if level in ["WARNING", "ERROR"]:
            self.status_bar.config(text=message, fg="red" if level == "ERROR" else "orange")
    
    def clear_log(self):
        """清除日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log_messages = []
    
    def copy_log(self):
        """复制日志到剪贴板"""
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_text.get(1.0, tk.END))
    
    def save_log(self):
        """保存日志到文件"""
        filename = f"change_detection_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w") as f:
            f.write(self.log_text.get(1.0, tk.END))
        self.log_message(f"日志已保存到: {filename}")
    
    def set_frame_rate(self, fps):
        """设置帧率"""
        self.frame_rate = max(1, min(360, fps))
        self.frame_rate_scale.set(self.frame_rate)
        self.frame_rate_entry.delete(0, tk.END)
        self.frame_rate_entry.insert(0, str(self.frame_rate))
        self.log_message(f"帧率设置为: {self.frame_rate}FPS")
    
    def set_threshold(self, threshold):
        """设置变化阈值"""
        self.threshold = max(1, min(100, threshold))
        self.threshold_scale.set(self.threshold)
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, str(self.threshold))
        self.log_message(f"变化阈值设置为: {self.threshold}")
    
    def set_min_contour_area(self, area):
        """设置最小变化区域"""
        self.min_contour_area = max(1, min(1000, area))
        self.min_area_scale.set(self.min_contour_area)
        self.min_area_entry.delete(0, tk.END)
        self.min_area_entry.insert(0, str(self.min_contour_area))
        self.log_message(f"最小变化区域设置为: {self.min_contour_area}像素")
    
    def on_algo_mode_changed(self, *args):
        """算法模式改变事件"""
        if self.algo_mode.get() == "combo":
            self.algo_select.pack_forget()
            self.algo_combo_select.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.algorithm_manager.set_algorithm_combo(self.algo_combo_select.get())
        else:
            self.algo_combo_select.pack_forget()
            self.algo_select.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            if self.algo_mode.get() == "single":
                self.algorithm_manager.set_algorithm(self.algo_select.get())
            else:
                self.algorithm_manager.set_algorithm("智能组合")
    
    def update_algorithm(self, event=None):
        """更新算法设置"""
        if self.algo_mode.get() == "single":
            self.algorithm_manager.set_algorithm(self.algo_select.get())
        elif self.algo_mode.get() == "combo":
            self.algorithm_manager.set_algorithm_combo(self.algo_combo_select.get())
        else:
            self.algorithm_manager.set_algorithm("智能组合")
        
        self.log_message(f"算法设置为: {self.algorithm_manager.current_algorithm}")
    
    def on_backend_combo_changed(self, event=None):
        """后端组合改变事件"""
        combo = self.backend_combo_var.get()
        if combo == "自动组合":
            backends = self.acceleration_manager.detect_optimal_combination()
            self.acceleration_manager.set_backend_combination([b.name for b in backends])
            self.log_message(f"已自动选择后端组合: {', '.join([b.name for b in backends])}")
        elif combo == "高性能":
            self.acceleration_manager.set_backend_combination(["TensorRT", "PyTorch", "CUDA"])
            self.log_message("已选择高性能后端组合: TensorRT, PyTorch, CUDA")
        elif combo == "平衡":
            self.acceleration_manager.set_backend_combination(["PyTorch", "OpenCL", "Numba"])
            self.log_message("已选择平衡后端组合: PyTorch, OpenCL, Numba")
        elif combo == "节能":
            self.acceleration_manager.set_backend_combination(["OpenCL", "CPU"])
            self.log_message("已选择节能后端组合: OpenCL, CPU")
    
    def adjust_performance_settings(self):
        """根据当前负载动态调节参数"""
        if not self.adaptive_fps_enabled:
            return
        
        # 获取当前CPU/GPU负载
        cpu_load = psutil.cpu_percent() / 100
        mem_load = psutil.virtual_memory().percent / 100
        current_load = max(cpu_load, mem_load)
        
        # 计算负载系数（0.5-1.5范围）
        self.current_load_factor = 0.5 + current_load
        
        # 根据性能模式调节
        if self.performance_mode == "性能":
            self.frame_rate = int(self.max_fps_limit * (1.8 - self.current_load_factor))
            if self.dynamic_threshold_enabled:
                self.threshold = min(100, int(30 * self.current_load_factor))
        elif self.performance_mode == "画质":
            self.frame_rate = max(self.min_fps_limit, int(self.max_fps_limit * (1.2 - self.current_load_factor/2)))
        else:  # 平衡
            self.frame_rate = int(self.max_fps_limit * (1.5 - self.current_load_factor))
        
        # 确保在合理范围内
        self.frame_rate = max(self.min_fps_limit, min(self.max_fps_limit, self.frame_rate))
        self.threshold = max(5, min(100, self.threshold))
        
        # 更新UI显示
        self.frame_rate_scale.set(self.frame_rate)
        self.frame_rate_entry.delete(0, tk.END)
        self.frame_rate_entry.insert(0, str(self.frame_rate))
        
        self.threshold_scale.set(self.threshold)
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, str(self.threshold))
    
    def update_video_display(self, frame):
        """更新视频显示"""
        # 将OpenCV图像转换为PIL格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # 计算缩放比例以保持宽高比
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        img_width, img_height = img.size
        
        if self.lock_aspect_ratio:
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
        else:
            new_width = canvas_width
            new_height = canvas_height
        
        # 缩放图像
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 转换为Tkinter PhotoImage
        self.current_image = ImageTk.PhotoImage(image=img)
        
        # 清除画布并显示新图像
        self.canvas.delete("all")
        self.canvas.create_image(
            (canvas_width - new_width) // 2,
            (canvas_height - new_height) // 2,
            anchor=tk.NW,
            image=self.current_image
        )
    
    def get_process_windows(self):
        """获取所有可见窗口的进程列表"""
        process_list = []
        
        def callback(hwnd, hwnd_list):
            if win32gui.IsWindowVisible(hwnd):
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    process = psutil.Process(pid)
                    window_title = win32gui.GetWindowText(hwnd)
                    if window_title:  # 只包含有标题的窗口
                        hwnd_list.append({
                            'hwnd': hwnd,
                            'pid': pid,
                            'title': window_title,
                            'name': process.name()
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return True
        
        win32gui.EnumWindows(callback, process_list)
        return process_list
    
    def get_available_cameras(self, max_test=5):
        """获取可用的摄像头列表"""
        cameras = []
        backends = [
            cv2.CAP_DSHOW,  # DirectShow
            cv2.CAP_MSMF,   # Microsoft Media Foundation
            cv2.CAP_ANY     # 自动选择
        ]

        for i in range(max_test):
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        cameras.append(i)
                        cap.release()
                        break  # 找到一个可用的后端就停止尝试
                    else:
                        cap.release()
                except:
                    continue
        return cameras
    
    def get_window_rect(self, hwnd):
        """获取窗口的矩形区域"""
        rect = win32gui.GetWindowRect(hwnd)
        return {
            "left": rect[0],
            "top": rect[1],
            "width": rect[2] - rect[0],
            "height": rect[3] - rect[1]
        }
    
    def create_process_selection_window(self):
        """创建进程选择窗口"""
        selection_window = tk.Toplevel(self.root)
        selection_window.title(f"选择需要监控的程序 ({VERSION} | {AUTHOR} | {COPYRIGHT})")
        selection_window.geometry("750x500")
        
        # 创建搜索框
        search_frame = tk.Frame(selection_window)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        search_label = tk.Label(search_frame, text="搜索:")
        search_label.pack(side=tk.LEFT)
        
        search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 填充进程列表的函数
        def update_process_list(treeview):
            processes = self.get_process_windows()
            treeview.delete(*treeview.get_children())
            
            for process in processes:
                treeview.insert('', 'end', values=(process['pid'], process['name'], process['title']))
        
        refresh_button = tk.Button(search_frame, text="刷新", command=lambda: update_process_list(tree))
        refresh_button.pack(side=tk.RIGHT, padx=5)
        
        # 创建进程列表
        tree = ttk.Treeview(selection_window, columns=('pid', 'name', 'title'), show='headings')
        tree.heading('pid', text='PID', command=lambda: self.sort_treeview(tree, 'pid', False))
        tree.heading('name', text='进程名', command=lambda: self.sort_treeview(tree, 'name', False))
        tree.heading('title', text='窗口标题', command=lambda: self.sort_treeview(tree, 'title', False))
        tree.column('pid', width=80, anchor='center')
        tree.column('name', width=150, anchor='center')
        tree.column('title', width=350, anchor='w')
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(selection_window, orient="vertical", command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # 初始填充列表
        update_process_list(tree)
        
        # 搜索功能
        def on_search(*args):
            query = search_var.get().lower()
            processes = self.get_process_windows()
            tree.delete(*tree.get_children())
            for process in processes:
                if (query in str(process['pid']).lower() or 
                    query in process['name'].lower() or 
                    query in process['title'].lower()):
                    tree.insert('', 'end', values=(process['pid'], process['name'], process['title']))
        
        search_var.trace("w", on_search)
        
        # 确定按钮
        def on_select():
            selected_item = tree.focus()
            if not selected_item:
                messagebox.showwarning("警告", "请先选择一个进程")
                return
            
            item_data = tree.item(selected_item)
            self.monitoring_process = {
                'hwnd': next(p['hwnd'] for p in self.get_process_windows() if p['pid'] == int(item_data['values'][0])),
                'pid': int(item_data['values'][0]),
                'name': item_data['values'][1],
                'title': item_data['values'][2]
            }
            self.monitoring_camera = None  # 清除摄像头监控
            
            # 更新按钮状态
            self.process_button.config(text=f"停止监控 {self.monitoring_process['name']}")
            self.camera_button.config(text="选择镜头监控")
            
            selection_window.destroy()
            self.log_message(f"已开始监控进程: {self.monitoring_process['name']}")
        
        # 取消按钮
        def on_cancel():
            selection_window.destroy()
        
        button_frame = tk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        select_button = tk.Button(button_frame, text="确定", command=on_select)
        select_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(button_frame, text="取消", command=on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=10)
        
        # 绑定双击事件
        def on_double_click(event):
            on_select()
        
        tree.bind("<Double-1>", on_double_click)
    
    def create_camera_selection_window(self):
        """创建摄像头选择窗口"""
        selection_window = tk.Toplevel(self.root)
        selection_window.title(f"选择监控镜头 ({VERSION} | {AUTHOR} | {COPYRIGHT})")
        selection_window.geometry("400x300")
        
        # 获取可用摄像头
        cameras = self.get_available_cameras()
        if not cameras:
            self.log_message("没有检测到可用的摄像头")
            selection_window.destroy()
            return
        
        # 创建摄像头列表
        tree = ttk.Treeview(selection_window, columns=('id', 'status'), show='headings')
        tree.heading('id', text='摄像头ID')
        tree.heading('status', text='状态')
        tree.column('id', width=100)
        tree.column('status', width=200)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 填充摄像头列表
        for cam_id in cameras:
            tree.insert('', 'end', values=(cam_id, "可用"))
        
        # 确定按钮
        def on_select():
            selected_item = tree.focus()
            if not selected_item:
                self.log_message("请先选择一个摄像头")
                return
            
            item_data = tree.item(selected_item)
            self.monitoring_camera = int(item_data['values'][0])
            self.monitoring_process = None  # 清除进程监控
            
            # 更新按钮状态
            self.camera_button.config(text=f"停止监控 摄像头{self.monitoring_camera}")
            self.process_button.config(text="选择监控进程")
            
            selection_window.destroy()
            self.log_message(f"已开始监控摄像头: {self.monitoring_camera}")
        
        # 取消按钮
        def on_cancel():
            selection_window.destroy()
        
        button_frame = tk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        select_button = tk.Button(button_frame, text="确定", command=on_select)
        select_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(button_frame, text="取消", command=on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=10)
    
    def toggle_process_monitoring(self):
        """开始/停止监控进程"""
        if self.monitoring_process:
            # 停止监控
            self.monitoring_process = None
            self.process_button.config(text="选择监控进程")
            self.log_message("已停止监控")
        else:
            # 开始监控 - 打开进程选择窗口
            self.monitoring_camera = None  # 清除摄像头监控
            self.camera_button.config(text="选择镜头监控")
            self.create_process_selection_window()
    
    def toggle_camera_monitoring(self):
        """开始/停止监控摄像头"""
        if self.monitoring_camera is not None:
            # 停止监控
            self.monitoring_camera = None
            self.camera_button.config(text="选择镜头监控")
            self.log_message("已停止监控")
        else:
            # 开始监控 - 打开摄像头选择窗口
            self.monitoring_process = None  # 清除进程监控
            self.process_button.config(text="选择监控进程")
            self.create_camera_selection_window()
    
    def select_monitor_area(self):
        """选择监控区域"""
        if not self.monitoring_process:
            messagebox.showwarning("警告", "请先选择要监控的进程")
            return
        
        # 获取窗口句柄
        hwnd = self.monitoring_process['hwnd']
        
        # 创建临时窗口用于选择区域
        temp_window = tk.Toplevel(self.root)
        temp_window.title("请在目标窗口上选择监控区域")
        temp_window.geometry("300x100")
        
        # 添加说明标签
        label = tk.Label(temp_window, text="请在目标窗口上拖动鼠标选择监控区域\n释放鼠标左键自动确认")
        label.pack(pady=20)
        
        # 设置窗口置顶
        temp_window.attributes('-topmost', True)
        
        # 获取窗口矩形
        rect = win32gui.GetWindowRect(hwnd)
        
        # 创建全屏透明窗口用于选择
        selection_window = tk.Toplevel(temp_window)
        selection_window.overrideredirect(True)
        selection_window.geometry(f"{rect[2]-rect[0]}x{rect[3]-rect[1]}+{rect[0]}+{rect[1]}")
        selection_window.attributes('-alpha', 0.3)
        selection_window.attributes('-topmost', True)
        
        # 选择区域变量
        start_x, start_y = 0, 0
        end_x, end_y = 0, 0
        rect_id = None
        
        def on_mouse_down(event):
            nonlocal start_x, start_y, rect_id
            start_x, start_y = event.x, event.y
            if rect_id:
                canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)
        
        def on_mouse_move(event):
            nonlocal end_x, end_y, rect_id
            end_x, end_y = event.x, event.y
            if rect_id:
                canvas.coords(rect_id, start_x, start_y, end_x, end_y)
        
        def on_mouse_up(event):
            nonlocal end_x, end_y
            end_x, end_y = event.x, event.y
        
            # 鼠标释放后自动确认选择
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            x2, y2 = max(start_x, end_x), max(start_y, end_y)
            
            # 确保选择的区域有效
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # 最小10像素的宽度和高度
                self.monitor_area_roi = (x1, y1, x2-x1, y2-y1)
                self.log_message(f"已设置监控区域: {self.monitor_area_roi}")
            else:
                self.log_message("选择区域太小，请重新选择")
                return
            
            temp_window.destroy()
            selection_window.destroy()
        
        # 创建画布
        canvas = tk.Canvas(selection_window, cursor="cross")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定事件
        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_move)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)
        
        # 确保画布可以接收键盘输入
        canvas.focus_set()
    
    def sort_treeview(self, treeview, col, reverse):
        """树状视图排序"""
        data = [(treeview.set(child, col), child) for child in treeview.get_children('')]
        
        # 尝试转换为数字排序
        try:
            data.sort(key=lambda x: int(x[0]), reverse=reverse)
        except ValueError:
            data.sort(reverse=reverse)
            
        for index, (val, child) in enumerate(data):
            treeview.move(child, '', index)
        
        treeview.heading(col, command=lambda: self.sort_treeview(treeview, col, not reverse))
    
    def get_next_frame(self):
        """获取下一帧"""
        if self.monitoring_camera is not None:
            cap = cv2.VideoCapture(self.monitoring_camera)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        else:
            rect = self.get_window_rect(self.monitoring_process['hwnd'])
            monitor_area = {
                "left": rect["left"],
                "top": rect["top"],
                "width": rect["width"],
                "height": rect["height"]
            }
            if self.monitor_area_roi:
                x, y, w, h = self.monitor_area_roi
                monitor_area["left"] += x
                monitor_area["top"] += y
                monitor_area["width"] = w
                monitor_area["height"] = h
            frame = np.array(self.sct.grab(monitor_area))
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    def update_frame(self):
        """更新帧显示"""
        # 1. 安全获取时间戳
        try:
            current_time = time.time()
        except:
            current_time = self.last_update_time or time.time()
        
        # 2. 计算时间差（带多重保护）
        time_diff = 0.033  # 默认30FPS的间隔
        if self.last_update_time is not None:
            time_diff = max(0.001, current_time - self.last_update_time)  # 最小1ms间隔
        
        # 3. 更新帧率计算
        self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1 / time_diff)
        self.last_update_time = current_time
        
        # 4. 帧率显示保护
        if not 0 < self.actual_fps < 1000:  # 合理范围检查
            self.actual_fps = 30.0
        
        # 5. 性能监控
        if self.frame_count % 10 == 0:
            self.adjust_performance_settings()
        
        # 6. 帧率过高警告
        if (self.frame_rate > 120 or abs(self.actual_fps - self.frame_rate) > 10) and \
            (current_time - self.last_fps_warning_time > self.fps_warning_interval):
            
            status_text = f"⚠️ 帧率过高: 设置 {self.frame_rate}FPS, 实际 {self.actual_fps:.1f}FPS"
            self.status_bar.config(text=status_text, fg="red")
            self.last_fps_warning_time = current_time
        
        # 7. 帧跳过逻辑
        if hasattr(self, 'frame_skip_var') and self.frame_skip_var.get() > 0:
            if not hasattr(self.update_frame, 'skip_counter'):
                self.update_frame.skip_counter = 0
            self.update_frame.skip_counter += 1
            if self.update_frame.skip_counter <= self.frame_skip_var.get():
                self.root.after(max(1, int(1000/self.frame_rate)), self.update_frame)
                return
            self.update_frame.skip_counter = 0
        
        # 8. 获取并处理帧
        if self.monitoring_camera is not None or self.monitoring_process:
            try:
                # 获取原始帧
                if self.monitoring_camera is not None:
                    cap = cv2.VideoCapture(self.monitoring_camera)
                    ret, original_frame = cap.read()
                    cap.release()
                    if not ret:
                        self.root.after(30, self.update_frame)
                        return
                else:
                    rect = self.get_window_rect(self.monitoring_process['hwnd'])
                    monitor_area = {
                        "left": rect["left"],
                        "top": rect["top"],
                        "width": rect["width"],
                        "height": rect["height"]
                    }
                    if self.monitor_area_roi:
                        x, y, w, h = self.monitor_area_roi
                        monitor_area["left"] += x
                        monitor_area["top"] += y
                        monitor_area["width"] = w
                        monitor_area["height"] = h
                    original_frame = np.array(self.sct.grab(monitor_area))
                    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)
                
                # 应用分辨率缩放
                if hasattr(self, 'resolution_scale_var'):
                    scale = self.resolution_scale_var.get()
                else:
                    scale = 1.0
                
                if scale < 1.0:
                    working_frame = cv2.resize(original_frame, (0,0), fx=scale, fy=scale)
                else:
                    working_frame = original_frame.copy()
                
                # 读取第二帧用于比较
                next_frame = self.get_next_frame()
                if next_frame is None or working_frame.shape != next_frame.shape:
                    self.root.after(max(1, int(1000/self.frame_rate)), self.update_frame)
                    return
                
                # 使用当前算法处理帧
                backend = self.acceleration_manager.get_current_backend()
                if self.use_multithread:
                    thresh = self.frame_processor.process_frames(working_frame, next_frame, self.threshold, backend)
                else:
                    thresh = self.algorithm_manager.process_frame(working_frame, next_frame, self.threshold)
                
                # 显示算法调试信息
                if self.show_algorithm_debug:
                    debug_info = f"算法: {self.algorithm_manager.current_algorithm}"
                    if self.algorithm_manager.current_algorithm in self.algorithm_manager.algorithm_presets:
                        debug_info += f" ({', '.join(self.algorithm_manager.algorithm_presets[self.algorithm_manager.current_algorithm])})"
                    self.log_message(debug_info, "DEBUG")
                
                # 查找轮廓
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 创建副本帧用于绘制
                display_frame = original_frame.copy()
                
                # 绘制检测框
                current_boxes = []
                for contour in contours:
                    if cv2.contourArea(contour) > self.min_contour_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        current_boxes.append((x, y, w, h))
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 渐隐效果处理
                if self.fade_effect_enabled:
                    # 更新渐隐框列表
                    self.fade_boxes = [(box, self.fade_frames) for box in current_boxes] + \
                                     [(box, count-1) for box, count in self.fade_boxes if count > 1]
                
                    # 绘制渐隐框
                    for box, count in self.fade_boxes:
                        if box not in current_boxes:  # 只绘制消失的框
                            alpha = count / self.fade_frames  # 计算透明度
                            color = (0, int(255*alpha), int(255*(1-alpha)))  # 从绿渐变到黄
                            cv2.rectangle(display_frame, 
                                         (box[0], box[1]), 
                                         (box[0]+box[2], box[1]+box[3]), 
                                         color, 
                                         1 + int(2*alpha))  # 线宽也逐渐变细
                
                # 如果缩放过，将结果放大回原始尺寸
                if scale < 1.0:
                    display_frame = cv2.resize(display_frame, (original_frame.shape[1], original_frame.shape[0]))
                
                # 更新显示
                self.update_video_display(display_frame)
                
                # 动态检测间隔
                if hasattr(self, 'dynamic_interval_var') and self.dynamic_interval_var.get():
                    if len(current_boxes) == 0:  # 没有检测到变化
                        self.root.after(max(1, int(2000/self.frame_rate)), self.update_frame)  # 降低检测频率
                        return
                
                # 更新状态栏
                backend = self.acceleration_manager.get_current_backend()
                status_text = (
                    f"运行中: {min(999, max(1, self.actual_fps)):.1f}FPS/{self.frame_rate}FPS | "
                    f"负载: {self.current_load_factor*100:.0f}% | "
                    f"模式: {self.performance_mode} | "
                    f"后端: {backend.name} (版本: {backend.model_version})"
                )
                self.status_bar.config(text=status_text)
                
            except Exception as e:
                self.log_message(f"处理出错: {str(e)}", "ERROR")
                if self.monitoring_process:
                    self.monitoring_process = None
                    self.process_button.config(text="选择监控进程")
                elif self.monitoring_camera is not None:
                    self.monitoring_camera = None
                    self.camera_button.config(text="选择镜头监控")
        else:
            # 没有选择监控源，显示提示信息
            current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            try:
                from PIL import Image, ImageDraw, ImageFont
                img_pil = Image.fromarray(current_frame)
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype("msyh.ttf", 30)
                draw.text((150, 200), "请选择监控源", font=font, fill=(255, 255, 255))
                font = ImageFont.truetype("msyh.ttf", 20)
                draw.text((100, 250), "1. 点击上方按钮选择监控进程", font=font, fill=(255, 255, 255))
                draw.text((100, 280), "2. 或选择监控摄像头", font=font, fill=(255, 255, 255))
                current_frame = np.array(img_pil)
            except:
                cv2.putText(current_frame, "Please select source", (150, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(current_frame, "1. Click button to monitor process", (50, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(current_frame, "2. Or select camera", (50, 290), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            self.update_video_display(current_frame)
        
        # 帧率控制
        effective_fps = max(1, min(self.frame_rate, self.max_fps_limit))
        delay = max(1, int(1000 / effective_fps))
        self.root.after(delay, self.update_frame)
        
        # 更新帧计数器
        self.frame_count += 1

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    try:
        root = tk.Tk()
        
        # 添加启动日志
        print("程序启动中...")
        print("初始化主界面...")
        
        # 设置超时检测
        def check_timeout():
            print("警告：程序初始化时间过长，可能是某些后端初始化卡住")
            print("尝试强制继续...")
            root.after(1000, root.destroy)  # 1秒后强制关闭
            
        timeout_timer = root.after(10000, check_timeout)  # 10秒超时
        
        # 初始化应用
        def init_app():
            try:
                print("正在创建应用实例...")
                app = ChangeDetectionApp(root)
                root.after_cancel(timeout_timer)  # 取消超时检测
                print("应用初始化完成，进入主循环")
                root.mainloop()
                
                # 释放资源
                print("程序退出，释放资源...")
                if hasattr(app, 'monitoring_camera') and app.monitoring_camera is not None:
                    cv2.VideoCapture(app.monitoring_camera).release()
                if hasattr(app, 'acceleration_manager'):
                    app.acceleration_manager.release_all()
                if hasattr(app, 'frame_processor'):
                    app.frame_processor.stop_processing()
                print("资源释放完成")
                
            except Exception as e:
                print(f"应用初始化失败: {str(e)}")
                import traceback
                traceback.print_exc()
                input("按Enter键退出...")
                root.destroy()
        
        root.after(100, init_app)  # 延迟100ms初始化
        
    except Exception as e:
        print(f"主程序崩溃: {str(e)}")
        import traceback
        traceback.print_exc()
        input("按Enter键退出...")