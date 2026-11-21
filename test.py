import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from mss import mss
import win32gui
import win32process
import win32con
import win32api
import psutil
import time
import platform
import sys
import subprocess
import threading
import queue
import os
import json
from typing import Optional, Dict, Any, Tuple, List
import collections
from abc import ABC, abstractmethod
import ctypes
from ctypes import wintypes

# ==================== 常量定义 ====================
VERSION = "v2.3.0"
RELEASE_DATE = "2025-04-02"
AUTHOR = "杜玛"
COPYRIGHT = "Copyright © 杜玛. All rights reserved."

# ==================== 覆盖窗口类 ====================
class OverlayWindow:
    """透明覆盖窗口管理器"""
    def __init__(self):
        self.POINT = wintypes.POINT
        self.RECT = wintypes.RECT
        self.windll = ctypes.windll
        self.overlay_hwnd = None
        self.overlay_thread = None
        self.running = False
        self.current_boxes = []
        self.box_color = (0, 255, 0)  # 默认绿色
        self.box_thickness = 2
        self.fade_boxes = []
        self.fade_frames = 10
        self.fade_enabled = True
        self.click_through = True
        self.parent_hwnd = None
        self.parent_rect = None
    
    def create_overlay(self, parent_hwnd):
        """创建透明覆盖窗口"""
        self.parent_hwnd = parent_hwnd
        self.running = True
        
        # 获取父窗口位置和大小
        rect = win32gui.GetWindowRect(parent_hwnd)
        self.parent_rect = {
            "left": rect[0],
            "top": rect[1],
            "width": rect[2] - rect[0],
            "height": rect[3] - rect[1]
        }
        
        # 创建覆盖窗口
        wc = win32gui.WNDCLASS()
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = "ChangeDetectionOverlay"
        wc.lpfnWndProc = self.wnd_proc
        class_atom = win32gui.RegisterClass(wc)
        
        # 窗口样式
        style = (win32con.WS_POPUP | win32con.WS_VISIBLE | 
                win32con.WS_CLIPSIBLINGS | win32con.WS_CLIPCHILDREN)
        ex_style = (win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | 
                   win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_TOPMOST)
        
        # 创建窗口
        self.overlay_hwnd = win32gui.CreateWindowEx(
            ex_style,
            class_atom,
            "ChangeDetectionOverlay",
            style,
            self.parent_rect["left"],
            self.parent_rect["top"],
            self.parent_rect["width"],
            self.parent_rect["height"],
            None, None, wc.hInstance, None
        )
        
        # 设置窗口透明
        win32gui.SetLayeredWindowAttributes(
            self.overlay_hwnd, 
            win32api.RGB(0, 0, 0), 
            0,  # 完全透明
            win32con.LWA_COLORKEY | win32con.LWA_ALPHA
        )
        
        # 启动更新线程
        self.overlay_thread = threading.Thread(target=self.update_overlay, daemon=True)
        self.overlay_thread.start()
    
    def wnd_proc(self, hwnd, msg, wparam, lparam):
        """窗口消息处理"""
        if msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        elif msg == win32con.WM_ERASEBKGND:
            return 1  # 防止背景擦除
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
    
    def update_overlay(self):
        """更新覆盖窗口内容"""
        hdc = win32gui.GetDC(self.overlay_hwnd)
        try:
            while self.running:
                # 检查父窗口是否仍然存在
                if not win32gui.IsWindow(self.parent_hwnd):
                    self.close()
                    break
                
                # 获取父窗口新位置
                rect = win32gui.GetWindowRect(self.parent_hwnd)
                new_rect = {
                    "left": rect[0],
                    "top": rect[1],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1]
                }
                
                # 如果窗口位置或大小改变，更新覆盖窗口
                if (new_rect["left"] != self.parent_rect["left"] or 
                    new_rect["top"] != self.parent_rect["top"] or
                    new_rect["width"] != self.parent_rect["width"] or
                    new_rect["height"] != self.parent_rect["height"]):
                    
                    self.parent_rect = new_rect
                    win32gui.SetWindowPos(
                        self.overlay_hwnd,
                        win32con.HWND_TOPMOST,
                        self.parent_rect["left"],
                        self.parent_rect["top"],
                        self.parent_rect["width"],
                        self.parent_rect["height"],
                        win32con.SWP_NOACTIVATE
                    )
                
                # 创建内存DC和位图
                mem_dc = win32gui.CreateCompatibleDC(hdc)
                bitmap = win32gui.CreateCompatibleBitmap(
                    hdc, 
                    self.parent_rect["width"], 
                    self.parent_rect["height"]
                )
                win32gui.SelectObject(mem_dc, bitmap)
                
                # 绘制透明背景
                win32gui.PatBlt(mem_dc, 0, 0, self.parent_rect["width"], self.parent_rect["height"], 
                                win32con.PATCOPY)
                
                # 如果有检测框，绘制它们
                if self.current_boxes or self.fade_boxes:
                    gdi32 = self.windll.gdi32
                    user32 = self.windll.user32
                    
                    # 创建画笔
                    pen = gdi32.CreatePen(win32con.PS_SOLID, self.box_thickness, 
                                         win32api.RGB(*self.box_color))
                    
                    # 选择画笔
                    old_pen = gdi32.SelectObject(mem_dc, pen)
                    
                    # 绘制当前检测框
                    for box in self.current_boxes:
                        x, y, w, h = box
                        gdi32.Rectangle(mem_dc, x, y, x + w, y + h)
                    
                    # 绘制渐隐框
                    if self.fade_enabled:
                        new_fade_boxes = []
                        for box, count in self.fade_boxes:
                            if count > 0:
                                alpha = count / self.fade_frames
                                color = (
                                    int(self.box_color[0] * alpha),
                                    int(self.box_color[1] * alpha),
                                    int(self.box_color[2] * alpha)
                                )
                                fade_pen = gdi32.CreatePen(win32con.PS_SOLID, 
                                                          max(1, int(self.box_thickness * alpha)), 
                                                          win32api.RGB(*color))
                                gdi32.SelectObject(mem_dc, fade_pen)
                                gdi32.Rectangle(mem_dc, box[0], box[1], box[0] + box[2], box[1] + box[3])
                                gdi32.DeleteObject(fade_pen)
                                new_fade_boxes.append((box, count - 1))
                        self.fade_boxes = new_fade_boxes
                    
                    # 恢复旧画笔并删除新画笔
                    gdi32.SelectObject(mem_dc, old_pen)
                    gdi32.DeleteObject(pen)
                
                # 更新窗口内容
                src_point = self.POINT(0, 0)
                size = win32gui.GetClientRect(self.overlay_hwnd)
                blend = win32gui.BLENDFUNCTION()
                blend.BlendOp = win32con.AC_SRC_OVER
                blend.BlendFlags = 0
                blend.SourceConstantAlpha = 255  # 完全不透明
                blend.AlphaFormat = win32con.AC_SRC_ALPHA
                
                user32.UpdateLayeredWindow(
                    self.overlay_hwnd, hdc, None, ctypes.byref(size),
                    mem_dc, ctypes.byref(src_point), 0, ctypes.byref(blend),
                    win32con.ULW_ALPHA
                )
                
                # 清理资源
                win32gui.DeleteObject(bitmap)
                win32gui.DeleteDC(mem_dc)
                
                # 控制刷新率
                time.sleep(0.03)  # ~30FPS
            
        finally:
            win32gui.ReleaseDC(self.overlay_hwnd, hdc)
    
    def update_boxes(self, boxes):
        """更新要显示的检测框"""
        self.current_boxes = boxes.copy()
        
        # 更新渐隐框
        if self.fade_enabled:
            # 找出新出现的框
            new_boxes = [box for box in boxes if box not in [b for b, _ in self.fade_boxes]]
            self.fade_boxes.extend([(box, self.fade_frames) for box in new_boxes])
    
    def set_color(self, color):
        """设置框的颜色 (B, G, R)"""
        self.box_color = color
    
    def set_thickness(self, thickness):
        """设置框的线宽"""
        self.box_thickness = thickness
    
    def set_fade(self, enabled, frames=10):
        """设置渐隐效果"""
        self.fade_enabled = enabled
        self.fade_frames = frames
    
    def set_click_through(self, enabled):
        """设置是否穿透鼠标点击"""
        self.click_through = enabled
        if self.overlay_hwnd:
            ex_style = win32gui.GetWindowLong(self.overlay_hwnd, win32con.GWL_EXSTYLE)
            if enabled:
                ex_style |= win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED
            else:
                ex_style &= ~(win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED)
            win32gui.SetWindowLong(self.overlay_hwnd, win32con.GWL_EXSTYLE, ex_style)
    
    def close(self):
        """关闭覆盖窗口"""
        self.running = False
        if self.overlay_hwnd and win32gui.IsWindow(self.overlay_hwnd):
            win32gui.PostMessage(self.overlay_hwnd, win32con.WM_CLOSE, 0, 0)
        if self.overlay_thread and self.overlay_thread.is_alive():
            self.overlay_thread.join(timeout=1)
        self.overlay_hwnd = None
        self.overlay_thread = None

# ==================== 加速后端抽象基类 ====================
class AccelerationBackend(ABC):
    """加速后端抽象基类"""
    def __init__(self):
        self.name = "CPU"
        self.initialized = False
        self.backend_info = {}
        self.priority = 0  # 优先级，数字越大优先级越高
    
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

# ==================== 具体加速后端实现 ====================
class PyTorchBackend(AccelerationBackend):
    """PyTorch GPU加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "PyTorch"
        self.priority = 9
        self.device = None
    
    def initialize(self) -> bool:
        try:
            import torch
            self.backend_info["torch_version"] = torch.__version__
            
            if not torch.cuda.is_available():
                self.log("CUDA不可用，回退到CPU")
                self.device = torch.device('cpu')
                self.priority = 2  # 降低CPU模式的优先级
            else:
                self.device = torch.device('cuda')
                # 测试CUDA是否真的可用
                try:
                    test_tensor = torch.tensor([1.0]).to(self.device)
                except Exception as e:
                    self.log(f"PyTorch CUDA测试失败: {str(e)}")
                    return False
            
            self.initialized = True
            return True
        except ImportError:
            self.log("PyTorch未安装")
            return False
        except Exception as e:
            self.log(f"PyTorch初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("PyTorch后端未初始化")
        
        import torch
        try:
            tensor1 = torch.from_numpy(frame1).permute(2, 0, 1).float().to(self.device)
            tensor2 = torch.from_numpy(frame2).permute(2, 0, 1).float().to(self.device)
            
            gray1 = (0.299 * tensor1[0] + 0.587 * tensor1[1] + 0.114 * tensor1[2]).to(torch.uint8)
            gray2 = (0.299 * tensor2[0] + 0.587 * tensor2[1] + 0.114 * tensor2[2]).to(torch.uint8)
            
            diff = torch.abs(gray1.int() - gray2.int())
            thresh = (diff > threshold).to(torch.uint8) * 255
            
            return thresh.cpu().numpy()
        except Exception as e:
            self.log(f"PyTorch处理失败: {str(e)}")
            # 失败时回退到CPU计算
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            return thresh
    
    def release(self):
        if hasattr(self, 'device') and self.device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()
    
    def log(self, message):
        """内部日志方法"""
        print(f"[PyTorchBackend] {message}")

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
                self.log("没有可用的CUDA设备")
                return False
            
            self.stream = cv2.cuda_Stream()
            self.gpu_frame1 = cv2.cuda_GpuMat()
            self.gpu_frame2 = cv2.cuda_GpuMat()
            self.backend_info["cuda_devices"] = cv2.cuda.getCudaEnabledDeviceCount()
            self.initialized = True
            return True
        except Exception as e:
            self.log(f"CUDA初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("CUDA后端未初始化")
        
        try:
            self.gpu_frame1.upload(frame1, self.stream)
            self.gpu_frame2.upload(frame2, self.stream)
            
            gpu_gray1 = cv2.cuda.cvtColor(self.gpu_frame1, cv2.COLOR_BGR2GRAY, stream=self.stream)
            gpu_gray2 = cv2.cuda.cvtColor(self.gpu_frame2, cv2.COLOR_BGR2GRAY, stream=self.stream)
            
            gpu_diff = cv2.cuda.absdiff(gpu_gray1, gpu_gray2, stream=self.stream)
            _, gpu_thresh = cv2.cuda.threshold(gpu_diff, threshold, 255, cv2.THRESH_BINARY, stream=self.stream)
            
            thresh = gpu_thresh.download(stream=self.stream)
            self.stream.waitForCompletion()
            return thresh
        except Exception as e:
            self.log(f"CUDA处理失败: {str(e)}")
            # 失败时回退到CPU计算
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            return thresh
    
    def release(self):
        if self.stream is not None:
            self.stream = None
        self.gpu_frame1 = None
        self.gpu_frame2 = None
    
    def log(self, message):
        """内部日志方法"""
        print(f"[CUDABackend] {message}")

class OpenCLBackend(AccelerationBackend):
    """OpenCL加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "OpenCL"
        self.priority = 7
    
    def initialize(self) -> bool:
        try:
            if not cv2.ocl.haveOpenCL():
                self.log("系统不支持OpenCL")
                return False
            
            cv2.ocl.setUseOpenCL(True)
            if not cv2.ocl.useOpenCL():
                self.log("OpenCL初始化失败")
                return False
            
            self.backend_info["opencl_available"] = True
            self.initialized = True
            return True
        except Exception as e:
            self.log(f"OpenCL初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("OpenCL后端未初始化")
        
        try:
            umat1 = cv2.UMat(frame1)
            umat2 = cv2.UMat(frame2)
            
            gray1 = cv2.cvtColor(umat1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(umat2, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            return cv2.UMat.get(thresh) if isinstance(thresh, cv2.UMat) else thresh
        except Exception as e:
            self.log(f"OpenCL处理失败: {str(e)}")
            # 失败时回退到CPU计算
            cv2.ocl.setUseOpenCL(False)
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            return thresh
    
    def release(self):
        cv2.ocl.setUseOpenCL(False)
        self.initialized = False
    
    def log(self, message):
        """内部日志方法"""
        print(f"[OpenCLBackend] {message}")

class CPUBackend(AccelerationBackend):
    """CPU后备方案"""
    def __init__(self):
        super().__init__()
        self.name = "CPU"
        self.priority = 1
    
    def initialize(self) -> bool:
        self.initialized = True
        self.backend_info["cpu_cores"] = os.cpu_count()
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
            PyTorchBackend(),
            CUDABackend(),
            OpenCLBackend(),
            CPUBackend()  # 最后是CPU后备方案
        ]
        self.current_backend = None
    
    def detect_available_backends(self) -> List[AccelerationBackend]:
        """检测所有可用的加速后端"""
        available = []
        for backend in self.backends:
            if backend.initialize():
                available.append(backend)
                print(f"检测到加速后端: {backend.name}")
                print(f"后端信息: {backend.get_info()}")
        
        if not available:
            print("未检测到加速后端，强制使用CPU")
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

# ==================== 算法管理器 ====================
class AlgorithmManager:
    """算法管理器，提供多种处理算法和组合"""
    def __init__(self):
        self.current_algorithm = "原始设置"
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
                resized1 = cv2.resize(frame1, (0,0), fx=scale, fy=scale)
                resized2 = cv2.resize(frame2, (0,0), fx=scale, fy=scale)
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
        # 简化的深度学习实现，实际使用时需要加载预训练模型
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh

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
        self.root = root
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
        
        # 覆盖窗口相关
        self.overlay = OverlayWindow()
        self.show_overlay = True
        self.overlay_color = (0, 255, 0)  # 绿色
        self.overlay_thickness = 2
        self.overlay_fade = True
        self.overlay_click_through = True
        
        # 初始化管理器
        self.sct = mss()
        self.acceleration_manager = AccelerationManager()
        self.algorithm_manager = AlgorithmManager()
        self.frame_processor = FrameProcessor()
        
        # 创建UI
        self.create_ui()
        
        # 启动帧处理器
        self.frame_processor.start_processing(self.threshold, self.acceleration_manager.get_current_backend())
        
        # 开始更新帧
        self.update_frame()
    
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
        self.control_panel_frame = tk.Frame(self.main_frame, bd=2, relief=tk.RAISED, height=250)
        self.control_panel_frame.pack_propagate(False)  # 固定高度
        self.control_panel_frame.pack(fill=tk.X, pady=0)
        
        # 日志区域 (固定高度100像素)
        self.log_frame = tk.Frame(self.main_frame, bd=2, relief=tk.SUNKEN, height=120)
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
        
        # 覆盖显示标签页
        self.overlay_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.overlay_tab, text="覆盖显示")
        
        # 填充各标签页内容
        self.fill_monitor_tab()
        self.fill_algorithm_tab()
        self.fill_performance_tab()
        self.fill_overlay_tab()
    
    def fill_monitor_tab(self):
        """填充监控设置标签页"""
        frame = ttk.Frame(self.monitor_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 监控源选择
        source_frame = ttk.LabelFrame(frame, text="监控源", padding=5)
        source_frame.pack(fill=tk.X, pady=2)
        
        self.process_button = ttk.Button(source_frame, text="选择进程", command=self.toggle_process_monitoring)
        self.process_button.pack(side=tk.LEFT, padx=2)
        
        self.camera_button = ttk.Button(source_frame, text="选择镜头", command=self.toggle_camera_monitoring)
        self.camera_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(source_frame, text="选择区域", command=self.select_monitor_area).pack(side=tk.LEFT, padx=2)
        
        # 监控参数设置
        param_frame = ttk.LabelFrame(frame, text="监控参数", padding=5)
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
        
        # 显示设置
        display_frame = ttk.LabelFrame(frame, text="显示设置", padding=5)
        display_frame.pack(fill=tk.X, pady=2)
        
        self.lock_aspect_var = tk.BooleanVar(value=self.lock_aspect_ratio)
        ttk.Checkbutton(display_frame, text="锁定宽高比", variable=self.lock_aspect_var,
                       command=lambda: setattr(self, 'lock_aspect_ratio', self.lock_aspect_var.get())).pack(side=tk.LEFT)
        
        self.topmost_var = tk.BooleanVar(value=self.root.attributes('-topmost'))
        ttk.Checkbutton(display_frame, text="窗口置顶", variable=self.topmost_var,
                       command=lambda: self.root.attributes('-topmost', self.topmost_var.get())).pack(side=tk.LEFT)
    
    def fill_algorithm_tab(self):
        """填充算法设置标签页"""
        frame = ttk.Frame(self.algorithm_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 算法选择
        algo_frame = ttk.LabelFrame(frame, text="算法选择", padding=5)
        algo_frame.pack(fill=tk.BOTH, pady=2)
        
        # 算法模式选择
        self.algo_mode = tk.StringVar(value="single")
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
    
    def fill_overlay_tab(self):
        """填充覆盖显示设置标签页"""
        frame = ttk.Frame(self.overlay_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 覆盖显示开关
        self.overlay_var = tk.BooleanVar(value=self.show_overlay)
        ttk.Checkbutton(frame, text="启用覆盖显示", variable=self.overlay_var,
                       command=self.toggle_overlay).pack(anchor=tk.W, pady=2)
        
        # 颜色选择
        color_frame = ttk.Frame(frame)
        color_frame.pack(fill=tk.X, pady=2)
        ttk.Label(color_frame, text="框颜色:").pack(side=tk.LEFT)
        
        self.color_var = tk.StringVar(value="绿色")
        color_menu = ttk.Combobox(color_frame, textvariable=self.color_var, 
                                 values=["红色", "绿色", "蓝色", "黄色", "青色", "紫色"])
        color_menu.pack(side=tk.LEFT, padx=5)
        color_menu.bind("<<ComboboxSelected>>", self.change_overlay_color)
        
        # 线宽设置
        width_frame = ttk.Frame(frame)
        width_frame.pack(fill=tk.X, pady=2)
        ttk.Label(width_frame, text="线宽:").pack(side=tk.LEFT)
        
        self.thickness_scale = tk.Scale(width_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                      command=lambda v: self.set_overlay_thickness(int(v)))
        self.thickness_scale.set(self.overlay_thickness)
        self.thickness_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 渐隐效果
        self.fade_var = tk.BooleanVar(value=self.overlay_fade)
        ttk.Checkbutton(frame, text="启用渐隐效果", variable=self.fade_var,
                       command=self.toggle_fade).pack(anchor=tk.W, pady=2)
        
        # 点击穿透
        self.click_through_var = tk.BooleanVar(value=self.overlay_click_through)
        ttk.Checkbutton(frame, text="穿透鼠标点击", variable=self.click_through_var,
                       command=self.toggle_click_through).pack(anchor=tk.W, pady=2)
    
    def create_log_panel(self):
        """创建日志面板"""
        # 日志文本框
        self.log_text = tk.Text(self.log_frame, wrap=tk.NONE)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
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
    
    def toggle_overlay(self):
        """切换覆盖显示状态"""
        self.show_overlay = self.overlay_var.get()
        if not self.show_overlay:
            self.overlay.close()
        elif self.monitoring_process:
            self.overlay.create_overlay(self.monitoring_process['hwnd'])
        
        self.log_message(f"覆盖显示 {'启用' if self.show_overlay else '禁用'}")
    
    def change_overlay_color(self, event=None):
        """更改覆盖显示颜色"""
        color_map = {
            "红色": (0, 0, 255),
            "绿色": (0, 255, 0),
            "蓝色": (255, 0, 0),
            "黄色": (0, 255, 255),
            "青色": (255, 255, 0),
            "紫色": (255, 0, 255)
        }
        self.overlay_color = color_map.get(self.color_var.get(), (0, 255, 0))
        self.overlay.set_color(self.overlay_color)
        self.log_message(f"覆盖颜色设置为: {self.color_var.get()}")
    
    def set_overlay_thickness(self, thickness):
        """设置覆盖线宽"""
        self.overlay_thickness = thickness
        self.overlay.set_thickness(thickness)
    
    def toggle_fade(self):
        """切换渐隐效果"""
        self.overlay_fade = self.fade_var.get()
        self.overlay.set_fade(self.overlay_fade)
        self.log_message(f"渐隐效果 {'启用' if self.overlay_fade else '禁用'}")
    
    def toggle_click_through(self):
        """切换点击穿透"""
        self.overlay_click_through = self.click_through_var.get()
        self.overlay.set_click_through(self.overlay_click_through)
        self.log_message(f"点击穿透 {'启用' if self.overlay_click_through else '禁用'}")
    
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
            
            # 创建覆盖窗口
            if self.show_overlay:
                self.overlay.create_overlay(self.monitoring_process['hwnd'])
                self.overlay.set_color(self.overlay_color)
                self.overlay.set_thickness(self.overlay_thickness)
                self.overlay.set_fade(self.overlay_fade)
                self.overlay.set_click_through(self.overlay_click_through)
            
            selection_window.destroy()
            self.log_message(f"已开始监控进程: {self.monitoring_process['name']}")
        
        # 取消按钮
        def on_cancel():
            selection_window.destroy()
        
        button_frame = tk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        select_button = tk.Button(button_frame,