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
import collections
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple

# ==================== 全局配置 ====================
VERSION = "1.0.0"
AUTHOR = "智能变化检测系统"
COPYRIGHT = "Copyright © 2023. All rights reserved."

# ==================== 主窗口类 ====================
class ChangeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.setup_variables()
        self.setup_acceleration()
        self.setup_processing()
        
        # 启动主循环
        self.update_frame()
        
    def setup_ui(self):
        """初始化用户界面"""
        self.root.title(f"智能变化检测系统 {VERSION}")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # 主框架布局
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 视频显示区域 (占3/4高度)
        self.video_frame = tk.Frame(self.main_frame, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 控制面板区域 (折叠时高度为30px，展开时250px)
        self.control_panel_frame = tk.Frame(self.main_frame, bd=1, relief=tk.RAISED)
        self.control_panel_frame.pack(fill=tk.X, pady=0)
        
        # 控制面板开关按钮
        self.toggle_button = tk.Button(
            self.control_panel_frame, 
            text="▼ 展开控制面板 ▼", 
            command=self.toggle_control_panel,
            height=1
        )
        self.toggle_button.pack(fill=tk.X)
        
        # 控制面板内容 (初始隐藏)
        self.control_panel = ttk.Notebook(self.control_panel_frame)
        self.create_control_panels()
        self.control_panel_visible = False
        
        # 日志区域 (固定100px高度)
        self.log_frame = tk.Frame(self.main_frame, bd=1, relief=tk.SUNKEN)
        self.log_frame.pack(fill=tk.X, pady=0, ipady=0)
        
        self.log_text = tk.Text(self.log_frame, height=4, wrap=tk.NONE)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 日志滚动条
        scroll_y = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x = ttk.Scrollbar(self.log_frame, orient=tk.HORIZONTAL, command=self.log_text.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.log_text.config(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        # 状态栏
        self.status_bar = tk.Label(self.root, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def toggle_control_panel(self):
        """切换控制面板的可见性"""
        if self.control_panel_visible:
            self.control_panel.pack_forget()
            self.toggle_button.config(text="▼ 展开控制面板 ▼")
            self.control_panel_visible = False
        else:
            self.control_panel.pack(fill=tk.BOTH, expand=True, pady=5)
            self.toggle_button.config(text="▲ 收起控制面板 ▲")
            self.control_panel_visible = True
    
    def create_control_panels(self):
        """创建控制面板的各选项卡"""
        # 1. 监控设置选项卡
        monitor_tab = ttk.Frame(self.control_panel)
        self.control_panel.add(monitor_tab, text="监控设置")
        
        # 监控源选择
        source_frame = tk.Frame(monitor_tab)
        source_frame.pack(fill=tk.X, pady=2)
        
        tk.Button(source_frame, text="选择进程", command=self.select_process).pack(side=tk.LEFT, padx=2)
        tk.Button(source_frame, text="选择摄像头", command=self.select_camera).pack(side=tk.LEFT, padx=2)
        tk.Button(source_frame, text="选择区域", command=self.select_area).pack(side=tk.LEFT, padx=2)
        
        # 2. 检测参数选项卡
        param_tab = ttk.Frame(self.control_panel)
        self.control_panel.add(param_tab, text="检测参数")
        
        # 帧率控制
        tk.Label(param_tab, text="帧率:").grid(row=0, column=0, sticky="e")
        self.frame_rate = tk.Scale(param_tab, from_=1, to=60, orient=tk.HORIZONTAL)
        self.frame_rate.set(30)
        self.frame_rate.grid(row=0, column=1, sticky="ew")
        
        # 变化阈值
        tk.Label(param_tab, text="变化阈值:").grid(row=1, column=0, sticky="e")
        self.threshold = tk.Scale(param_tab, from_=1, to=100, orient=tk.HORIZONTAL)
        self.threshold.set(30)
        self.threshold.grid(row=1, column=1, sticky="ew")
        
        # 最小变化区域
        tk.Label(param_tab, text="最小区域:").grid(row=2, column=0, sticky="e")
        self.min_area = tk.Scale(param_tab, from_=10, to=1000, orient=tk.HORIZONTAL)
        self.min_area.set(100)
        self.min_area.grid(row=2, column=1, sticky="ew")
        
        # 3. 算法选择选项卡
        algo_tab = ttk.Frame(self.control_panel)
        self.control_panel.add(algo_tab, text="算法选择")
        
        # 算法模式选择
        self.algo_mode = tk.StringVar(value="auto")
        tk.Radiobutton(algo_tab, text="自动选择", variable=self.algo_mode, value="auto").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(algo_tab, text="手动选择", variable=self.algo_mode, value="manual").grid(row=0, column=1, sticky="w")
        
        # 算法选择下拉框
        self.algo_combobox = ttk.Combobox(algo_tab, state="readonly")
        self.algo_combobox.grid(row=1, column=0, columnspan=2, sticky="ew")
        
        # 4. 性能优化选项卡
        perf_tab = ttk.Frame(self.control_panel)
        self.control_panel.add(perf_tab, text="性能优化")
        
        # 处理模式
        tk.Label(perf_tab, text="处理模式:").grid(row=0, column=0, sticky="e")
        self.proc_mode = ttk.Combobox(perf_tab, values=["自动", "多进程", "多线程", "单线程"])
        self.proc_mode.current(0)
        self.proc_mode.grid(row=0, column=1, sticky="ew")
        
        # 自适应选项
        self.adaptive_fps = tk.BooleanVar(value=True)
        tk.Checkbutton(perf_tab, text="自适应帧率", variable=self.adaptive_fps).grid(row=1, column=0, columnspan=2, sticky="w")
        
        self.adaptive_thresh = tk.BooleanVar(value=True)
        tk.Checkbutton(perf_tab, text="自适应阈值", variable=self.adaptive_thresh).grid(row=2, column=0, columnspan=2, sticky="w")
        
        # 使各列均匀分布
        for i in range(2):
            param_tab.columnconfigure(i, weight=1)
            algo_tab.columnconfigure(i, weight=1)
            perf_tab.columnconfigure(i, weight=1)
    
    def setup_variables(self):
        """初始化程序变量"""
        self.monitoring_process = None
        self.monitoring_camera = None
        self.monitor_area = None
        self.current_image = None
        
        # 性能相关
        self.frame_times = collections.deque(maxlen=60)
        self.last_update_time = None
        self.actual_fps = 0
        self.frame_count = 0
        self.cpu_load = 0
        
        # 检测结果
        self.last_boxes = []
        self.fade_boxes = []
        
        # 截图工具
        self.sct = mss()
    
    def setup_acceleration(self):
        """初始化加速后端"""
        self.accel_manager = AccelerationManager()
        self.current_backend = self.accel_manager.detect_best_backend()
        self.log_message(f"检测到最佳加速后端: {self.current_backend.name}")
    
    def setup_processing(self):
        """初始化处理模块"""
        self.frame_processor = FrameProcessor()
        self.algorithm_manager = AlgorithmManager()
        
        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.running = True
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self.process_frames_thread, daemon=True)
        self.processing_thread.start()
    
    def select_process(self):
        """选择要监控的进程"""
        # 实现进程选择逻辑
        pass
    
    def select_camera(self):
        """选择要监控的摄像头"""
        # 实现摄像头选择逻辑
        pass
    
    def select_area(self):
        """选择监控区域"""
        # 实现区域选择逻辑
        pass
    
    def update_frame(self):
        """更新视频帧显示"""
        try:
            # 获取当前时间
            current_time = time.time()
            
            # 计算实际帧率
            if self.last_update_time:
                time_diff = max(0.001, current_time - self.last_update_time)
                self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1 / time_diff)
            self.last_update_time = current_time
            
            # 获取帧
            frame = self.get_next_frame()
            if frame is None:
                self.root.after(30, self.update_frame)
                return
            
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 更新显示
            self.update_display(processed_frame)
            
            # 更新状态栏
            self.update_status()
            
            # 性能调节
            if self.frame_count % 10 == 0:
                self.adjust_performance()
            
            self.frame_count += 1
            
            # 计算下一帧延迟
            delay = max(1, int(1000 / self.frame_rate.get()))
            self.root.after(delay, self.update_frame)
            
        except Exception as e:
            self.log_message(f"更新帧出错: {str(e)}", "ERROR")
            self.root.after(1000, self.update_frame)
    
    def get_next_frame(self):
        """获取下一帧图像"""
        if self.monitoring_camera is not None:
            # 从摄像头获取帧
            cap = cv2.VideoCapture(self.monitoring_camera)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        elif self.monitoring_process:
            # 从进程窗口获取帧
            rect = self.get_window_rect(self.monitoring_process['hwnd'])
            if self.monitor_area:
                rect['left'] += self.monitor_area[0]
                rect['top'] += self.monitor_area[1]
                rect['width'] = self.monitor_area[2]
                rect['height'] = self.monitor_area[3]
            
            frame = np.array(self.sct.grab(rect))
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            # 没有监控源时显示提示
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "请选择监控源", (150, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return frame
    
    def process_frame(self, frame):
        """处理帧并检测变化"""
        # 获取两帧用于比较
        frame1 = frame.copy()
        time.sleep(1.0 / self.frame_rate.get())
        frame2 = self.get_next_frame()
        
        if frame2 is None or frame1.shape != frame2.shape:
            return frame1
        
        # 使用当前算法处理
        if self.algo_mode.get() == "auto":
            result = self.algorithm_manager.smart_process(frame1, frame2, self.threshold.get())
        else:
            result = self.algorithm_manager.process_frame(
                frame1, frame2, self.threshold.get(), 
                self.algo_combobox.get()
            )
        
        # 查找轮廓
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制检测结果
        display_frame = frame1.copy()
        current_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area.get():
                x, y, w, h = cv2.boundingRect(contour)
                current_boxes.append((x, y, w, h))
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 更新检测框历史
        self.last_boxes = current_boxes
        
        return display_frame
    
    def update_display(self, frame):
        """更新视频显示"""
        # 转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # 计算缩放比例
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
        self.current_image = ImageTk.PhotoImage(image=img)
        
        # 更新显示
        self.canvas.delete("all")
        self.canvas.create_image(
            (canvas_width - new_width) // 2,
            (canvas_height - new_height) // 2,
            anchor=tk.NW,
            image=self.current_image
        )
    
    def update_status(self):
        """更新状态栏信息"""
        status_text = (
            f"FPS: {min(999, max(1, self.actual_fps)):.1f}/{self.frame_rate.get()} | "
            f"CPU: {self.cpu_load}% | "
            f"算法: {self.current_backend.name} | "
            f"检测到: {len(self.last_boxes)}处变化"
        )
        self.status_bar.config(text=status_text)
    
    def adjust_performance(self):
        """根据系统负载调整性能参数"""
        # 获取CPU负载
        self.cpu_load = psutil.cpu_percent(interval=0.1)
        
        # 自适应帧率
        if self.adaptive_fps.get():
            target_fps = 60 - min(50, self.cpu_load * 0.5)
            self.frame_rate.set(max(5, min(60, int(target_fps))))
        
        # 自适应阈值
        if self.adaptive_thresh.get():
            avg_contours = len(self.last_boxes)
            if avg_contours > 20:  # 检测到太多变化，增加阈值
                new_thresh = min(100, self.threshold.get() + 5)
                self.threshold.set(new_thresh)
            elif avg_contours < 5:  # 检测到太少变化，降低阈值
                new_thresh = max(5, self.threshold.get() - 2)
                self.threshold.set(new_thresh)
    
    def process_frames_thread(self):
        """后台处理线程"""
        while self.running:
            try:
                # 从队列获取帧
                frame1, frame2 = self.frame_queue.get(timeout=0.1)
                
                # 处理帧
                future = self.executor.submit(
                    self.process_frame_task,
                    frame1, frame2, self.threshold.get()
                )
                self.result_queue.put(future)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.log_message(f"处理线程出错: {str(e)}", "ERROR")
                continue
    
    def process_frame_task(self, frame1, frame2, threshold):
        """帧处理任务函数"""
        try:
            return self.current_backend.process_frames(frame1, frame2, threshold)
        except Exception as e:
            self.log_message(f"处理帧出错: {str(e)}", "ERROR")
            return None
    
    def log_message(self, message, level="INFO"):
        """记录日志消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # 更新日志显示
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 限制日志大小
        if self.log_text.index(tk.END).split('.')[0] > 1000:
            self.log_text.delete(1.0, "100.0")
    
    def get_window_rect(self, hwnd):
        """获取窗口矩形区域"""
        rect = win32gui.GetWindowRect(hwnd)
        return {
            "left": rect[0],
            "top": rect[1],
            "width": rect[2] - rect[0],
            "height": rect[3] - rect[1]
        }
    
    def on_close(self):
        """关闭程序时的清理工作"""
        self.running = False
        self.executor.shutdown(wait=False)
        
        if self.monitoring_camera is not None:
            cv2.VideoCapture(self.monitoring_camera).release()
        
        self.accel_manager.release_all()
        self.root.destroy()

# ==================== 加速后端实现 ====================
class AccelerationBackend(ABC):
    """加速后端抽象基类"""
    def __init__(self):
        self.name = "CPU"
        self.initialized = False
        self.backend_info = {}
    
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

class CUDABackend(AccelerationBackend):
    """CUDA加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "CUDA"
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
        except:
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.stream is None:
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
            self.initialized = False
            raise RuntimeError(f"CUDA处理失败: {str(e)}")
    
    def release(self):
        if self.stream is not None:
            self.stream = None
        self.gpu_frame1 = None
        self.gpu_frame2 = None

class PyTorchBackend(AccelerationBackend):
    """PyTorch GPU加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "PyTorch"
        self.device = None
        self.torch_available = False
    
    def initialize(self) -> bool:
        try:
            import torch
            self.torch_available = True
            self.backend_info["torch_version"] = torch.__version__
            
            if not torch.cuda.is_available():
                self.device = torch.device('cpu')
                return False
            
            try:
                self.device = torch.device('cuda')
                test_tensor = torch.tensor([1.0]).cuda()
                self.initialized = True
                return True
            except Exception as e:
                self.device = torch.device('cpu')
                return False
            
        except Exception as e:
            self.torch_available = False
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.torch_available or not self.initialized or self.device is None:
            raise RuntimeError("PyTorch后端未初始化")
        
        try:
            import torch
            tensor1 = torch.from_numpy(frame1).permute(2, 0, 1).float().to(self.device)
            tensor2 = torch.from_numpy(frame2).permute(2, 0, 1).float().to(self.device)
            
            gray1 = (0.299 * tensor1[0] + 0.587 * tensor1[1] + 0.114 * tensor1[2]).to(torch.uint8)
            gray2 = (0.299 * tensor2[0] + 0.587 * tensor2[1] + 0.114 * tensor2[2]).to(torch.uint8)
            
            diff = torch.abs(gray1.int() - gray2.int())
            thresh = (diff > threshold).to(torch.uint8) * 255
            
            return thresh.cpu().numpy()
        except Exception as e:
            self.initialized = False
            raise RuntimeError(f"PyTorch处理失败: {str(e)}")
    
    def release(self):
        if hasattr(self, 'device') and self.device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()

class TensorRTBackend(AccelerationBackend):
    """TensorRT加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "TensorRT"
        self.engine = None
    
    def initialize(self) -> bool:
        try:
            import tensorrt as trt
            self.backend_info["tensorrt_version"] = trt.__version__
            
            # 这里需要实际加载TensorRT引擎
            # 示例代码，实际使用时需要替换为真实的引擎加载逻辑
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            
            # 尝试创建一个简单的引擎来测试可用性
            builder = trt.Builder(logger)
            network = builder.create_network()
            
            self.initialized = True
            return True
        except Exception as e:
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("TensorRT后端未初始化")
        
        # 实际处理逻辑需要实现
        # 这里使用CPU后备方案
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def release(self):
        self.engine = None

class ONNXBackend(AccelerationBackend):
    """ONNX Runtime加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "ONNX"
        self.session = None
    
    def initialize(self) -> bool:
        try:
            import onnxruntime as ort
            self.backend_info["onnx_version"] = ort.__version__
            
            # 测试ONNX Runtime是否可用
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                self.session_options = ort.SessionOptions()
                self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.initialized = True
                return True
            return False
        except:
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("ONNX后端未初始化")
        
        # 实际处理逻辑需要实现
        # 这里使用CPU后备方案
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def release(self):
        self.session = None

class OpenVINOBackend(AccelerationBackend):
    """OpenVINO加速后端"""
    def __init__(self):
        super().__init__()
        self.name = "OpenVINO"
        self.core = None
        self.model = None
    
    def initialize(self) -> bool:
        try:
            from openvino.runtime import Core
            self.core = Core()
            devices = self.core.available_devices
            self.backend_info["openvino_devices"] = devices
            self.initialized = True
            return True
        except:
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("OpenVINO后端未初始化")
        
        # 实际处理逻辑需要实现
        # 这里使用CPU后备方案
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def release(self):
        self.core = None
        self.model = None

class YOLOBackend(AccelerationBackend):
    """YOLO目标检测后端"""
    def __init__(self):
        super().__init__()
        self.name = "YOLO"
        self.model = None
    
    def initialize(self) -> bool:
        try:
            # 尝试导入ultralytics YOLO
            from ultralytics import YOLO
            self.backend_info["yolo_version"] = YOLO.__version__
            
            # 加载一个小型YOLO模型进行测试
            self.model = YOLO('yolov8n.pt')
            self.initialized = True
            return True
        except:
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.model is None:
            raise RuntimeError("YOLO后端未初始化")
        
        # 使用YOLO检测变化
        results1 = self.model(frame1)
        results2 = self.model(frame2)
        
        # 比较检测结果
        # 这里简化处理，实际应该比较检测框的变化
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def release(self):
        self.model = None

class MobileNetBackend(AccelerationBackend):
    """MobileNet后端"""
    def __init__(self):
        super().__init__()
        self.name = "MobileNet"
        self.model = None
    
    def initialize(self) -> bool:
        try:
            import tensorflow as tf
            self.backend_info["tf_version"] = tf.__version__
            
            # 加载MobileNet模型
            self.model = tf.keras.applications.MobileNetV2(weights='imagenet')
            self.initialized = True
            return True
        except:
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.model is None:
            raise RuntimeError("MobileNet后端未初始化")
        
        # 使用MobileNet提取特征并比较
        # 这里简化处理，实际应该比较特征差异
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def release(self):
        self.model = None

class CPUBackend(AccelerationBackend):
    """CPU后备方案"""
    def __init__(self):
        super().__init__()
        self.name = "CPU"
    
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

class AccelerationManager:
    """加速后端管理器"""
    def __init__(self):
        self.backends = [
            PyTorchBackend(),
            CUDABackend(),
            TensorRTBackend(),
            ONNXBackend(),
            OpenVINOBackend(),
            YOLOBackend(),
            MobileNetBackend(),
            CPUBackend()  # 最后一个是CPU后备
        ]
        self.current_backend = None
    
    def detect_best_backend(self) -> AccelerationBackend:
        """检测并返回最佳可用的加速后端"""
        for backend in self.backends:
            if backend.initialize():
                return backend
        return self.backends[-1]  # 返回CPU后端
    
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
    """算法管理器，提供多种处理算法"""
    def __init__(self):
        self.algorithms = {
            "原始差分": self.original_diff,
            "高斯模糊": self.gaussian_diff,
            "背景减除": self.background_subtraction,
            "光流法": self.optical_flow,
            "多尺度": self.multi_scale,
            "形态学": self.morphological,
            "深度学习": self.deep_learning
        }
        self.current_algorithm = "原始差分"
        
        # 背景减除器
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
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
    
    def set_algorithm(self, algorithm_name: str):
        """设置当前算法"""
        if algorithm_name in self.algorithms:
            self.current_algorithm = algorithm_name
            return True
        return False
    
    def process_frame(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int, algorithm: str = None) -> np.ndarray:
        """使用指定算法处理帧"""
        algo = algorithm if algorithm else self.current_algorithm
        return self.algorithms[algo](frame1, frame2, threshold)
    
    def smart_process(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """智能选择最佳算法"""
        # 根据帧特性自动选择算法
        if frame1.shape[0] < 480 or frame1.shape[1] < 640:  # 小分辨率
            return self.gaussian_diff(frame1, frame2, threshold)
        else:  # 大分辨率
            return self.multi_scale(frame1, frame2, threshold)
    
    # ===== 算法实现 =====
    def original_diff(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """原始帧差算法"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def gaussian_diff(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """高斯模糊预处理"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def background_subtraction(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """背景减除算法"""
        fg_mask = self.back_sub.apply(frame2)
        _, thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def optical_flow(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """光流法"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **self.flow_params)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def multi_scale(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """多尺度检测"""
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
    
    def morphological(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """形态学处理"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh
    
    def deep_learning(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """深度学习方式(简化版)"""
        # 实际应用中应该使用训练好的模型
        return self.multi_scale(frame1, frame2, threshold)

# ==================== 帧处理器 ====================
class FrameProcessor:
    """多线程帧处理器"""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.running = True
    
    def start_processing(self):
        """启动处理线程"""
        threading.Thread(target=self.process_frames_thread, daemon=True).start()
    
    def process_frames_thread(self):
        """处理线程主函数"""
        while self.running:
            try:
                frame1, frame2, threshold, backend = self.frame_queue.get(timeout=0.1)
                future = self.executor.submit(
                    self.process_frame_task, 
                    frame1, frame2, threshold, backend
                )
                self.result_queue.put(future)
            except queue.Empty:
                continue
    
    def process_frame_task(self, frame1, frame2, threshold, backend):
        """处理帧的任务函数"""
        try:
            return backend.process_frames(frame1, frame2, threshold)
        except Exception as e:
            print(f"处理帧出错: {str(e)}")
            return None
    
    def stop_processing(self):
        """停止处理线程"""
        self.running = False
        self.executor.shutdown(wait=False)

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = ChangeDetectionApp(root)
    
    # 设置关闭事件处理
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    
    root.mainloop()