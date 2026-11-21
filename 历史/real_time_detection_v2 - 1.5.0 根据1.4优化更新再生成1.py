import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from mss import mss
import win32gui
import win32con
import win32process
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

# ==================== 常量定义 ====================
VERSION = "v2.2.0"
RELEASE_DATE = "2025-04-02"
AUTHOR = "杜玛"
COPYRIGHT = "Copyright © 杜玛. All rights reserved."

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

# ==================== 覆盖窗口管理器 ====================
class OverlayWindow:
    """透明覆盖窗口管理器"""
    def __init__(self):
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
                    # 创建GDI+图形对象
                    from ctypes import windll, byref, c_int
                    from ctypes.wintypes import POINT, RECT
                    
                    gdi32 = windll.gdi32
                    user32 = windll.user32
                    
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
                src_point = POINT(0, 0)
                size = win32gui.GetClientRect(self.overlay_hwnd)
                blend = win32gui.BLENDFUNCTION()
                blend.BlendOp = win32con.AC_SRC_OVER
                blend.BlendFlags = 0
                blend.SourceConstantAlpha = 255  # 完全不透明
                blend.AlphaFormat = win32con.AC_SRC_ALPHA
                
                user32.UpdateLayeredWindow(
                    self.overlay_hwnd, hdc, None, byref(size),
                    mem_dc, byref(src_point), 0, byref(blend),
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
        self.monitoring_process = None
        self.monitoring_camera = None
        self.monitor_area_roi = None
        self.current_image = None
        self.frame_count = 0
        self.actual_fps = 0
        self.last_update_time = None
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
        self.backend = self.acceleration_manager.detect_best_backend()
        self.log_message(f"使用加速后端: {self.backend.name}")
        
        # 创建UI
        self.create_ui()
        
        # 开始更新帧
        self.update_frame()
    
    def create_ui(self):
        """创建用户界面"""
        # 主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 视频显示区域
        self.video_frame = tk.Frame(self.main_frame, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        # 画布
        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 控制面板区域
        self.control_panel_frame = tk.Frame(self.main_frame, bd=2, relief=tk.RAISED, height=350)
        self.control_panel_frame.pack_propagate(False)
        self.control_panel_frame.pack(fill=tk.X, pady=0)
        
        # 日志区域
        self.log_frame = tk.Frame(self.main_frame, bd=2, relief=tk.SUNKEN, height=100)
        self.log_frame.pack_propagate(False)
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
        
        # 覆盖设置标签页
        self.overlay_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.overlay_tab, text="覆盖显示")
        
        # 性能设置标签页
        self.performance_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.performance_tab, text="性能")
        
        # 填充各标签页内容
        self.fill_monitor_tab()
        self.fill_overlay_tab()
        self.fill_performance_tab()
    
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
    
    def fill_performance_tab(self):
        """填充性能设置标签页"""
        frame = ttk.Frame(self.performance_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 加速后端信息
        backend_frame = ttk.LabelFrame(frame, text="加速后端", padding=5)
        backend_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(backend_frame, text=f"当前后端: {self.backend.name}").pack(anchor=tk.W)
        ttk.Label(backend_frame, text=f"后端信息: {self.backend.get_info()}").pack(anchor=tk.W)
        
        # 后端切换
        ttk.Button(backend_frame, text="重新检测加速后端", 
                  command=self.redetect_backends).pack(pady=5)
    
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
    
    def redetect_backends(self):
        """重新检测加速后端"""
        self.acceleration_manager.release_all()
        self.backend = self.acceleration_manager.detect_best_backend()
        self.log_message(f"重新检测完成，当前后端: {self.backend.name}")
        self.log_message(f"后端信息: {self.backend.get_info()}")
    
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
        selection_window.title("选择需要监控的程序")
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
        tree.heading('pid', text='PID')
        tree.heading('name', text='进程名')
        tree.heading('title', text='窗口标题')
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
        selection_window.title("选择监控镜头")
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
            
            # 关闭覆盖窗口
            self.overlay.close()
            
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
            self.overlay.close()
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
            self.overlay.close()
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
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
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
        current_time = time.time()
        time_diff = 0.033  # 默认30FPS的间隔
        if self.last_update_time is not None:
            time_diff = max(0.001, current_time - self.last_update_time)  # 最小1ms间隔
        
        # 更新帧率计算
        self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1 / time_diff)
        self.last_update_time = current_time
        
        # 帧率显示保护
        if not 0 < self.actual_fps < 1000:  # 合理范围检查
            self.actual_fps = 30.0
        
        # 获取并处理帧
        if self.monitoring_camera is not None or self.monitoring_process:
            try:
                # 获取原始帧
                original_frame = self.get_next_frame()
                if original_frame is None:
                    self.root.after(30, self.update_frame)
                    return
                
                # 读取第二帧用于比较
                next_frame = self.get_next_frame()
                if next_frame is None or original_frame.shape != next_frame.shape:
                    self.root.after(30, self.update_frame)
                    return
                
                # 处理帧
                thresh = self.backend.process_frames(original_frame, next_frame, self.threshold)
                
                # 查找轮廓
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 创建检测框列表
                boxes = []
                for contour in contours:
                    if cv2.contourArea(contour) > self.min_contour_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        boxes.append((x, y, w, h))
                
                # 更新覆盖窗口
                if self.monitoring_process and self.show_overlay:
                    self.overlay.update_boxes(boxes)
                
                # 创建副本帧用于绘制
                display_frame = original_frame.copy()
                
                # 绘制检测框
                for box in boxes:
                    x, y, w, h = box
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 更新显示
                self.update_video_display(display_frame)
                
                # 更新状态栏
                status_text = (
                    f"运行中: {min(999, max(1, self.actual_fps)):.1f}FPS/{self.frame_rate}FPS | "
                    f"检测到 {len(boxes)} 个变化区域 | "
                    f"后端: {self.backend.name}"
                )
                self.status_bar.config(text=status_text)
                
            except Exception as e:
                self.log_message(f"处理出错: {str(e)}", "ERROR")
                if self.monitoring_process:
                    self.monitoring_process = None
                    self.process_button.config(text="选择监控进程")
                    self.overlay.close()
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
        delay = max(1, int(1000 / self.frame_rate))
        self.root.after(delay, self.update_frame)
        
        # 更新帧计数器
        self.frame_count += 1

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ChangeDetectionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"程序崩溃: {str(e)}")
        import traceback
        traceback.print_exc()
        input("按Enter键退出...")