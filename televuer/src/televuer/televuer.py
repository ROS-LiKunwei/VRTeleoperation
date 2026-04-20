from vuer.server import Vuer # 导入 Vuer 框架，用于 WebXR 渲染和交互
# 导入 Vuer 的各种场景元素组件
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane 
from multiprocessing import Value, Array, Process, shared_memory # 导入多进程共享内存相关模块
import numpy as np
import asyncio # 导入 asyncio 用于异步编程
import threading # 导入 threading 用于多线程处理
import cv2
import os
from pathlib import Path
from typing import Literal # 导入类型注解模块


class TeleVuer:
    def __init__(self, use_hand_tracking: bool, binocular: bool=True, img_shape: tuple=None, display_fps: float=30.0,
                       display_mode: Literal["immersive", "pass-through", "ego"]="immersive", zmq: bool=False, webrtc: bool=False, webrtc_url: str=None, 
                       cert_file: str=None, key_file: str=None, port: int=8012):
        """
        TeleVuer 类：基于 OpenXR 的 XR 遥操作应用
        
        功能：
            1. 与 Vuer 服务器通信，处理 VR 头显的数据流
            2. 管理图像和姿态数据的传输和同步
            3. 支持手部追踪和控制器追踪
            4. 支持单目/双目、不同显示模式

        :param use_hand_tracking: bool - 是否使用手部追踪(True)还是控制器追踪(False)
        :param binocular: bool - 是否为双目立体视觉(True)或单目(False)
        :param img_shape: tuple - 头部图像尺寸 (高度, 宽度)
        :param display_fps: float - 显示更新的目标帧率，默认 30.0
        :param display_mode: str - VR 观看模式
            * "immersive": 完全沉浸模式,虚拟现实(VR)显示机器人的第一人称视角(必须启用zmq或webrtc)。
            * "pass-through": 穿透模式,VR 显示真实世界（通过 VR 头显摄像头）;不显示来自zmq或webrtc的图像(即使已启用)。
            * "ego": 自我视角模式，中心小窗口显示机器人视角，周围显示真实世界
        :param zmq: bool - 是否使用 ZMQ 传输图像
        :param webrtc: bool - 是否使用 WebRTC 实时通信
        :param webrtc_url: str - WebRTC offer URL(如果使用 WebRTC 则必须提供)
        :param cert_file: str - SSL 证书文件路径
        :param key_file: str - SSL 私钥文件路径
        
        - 同一时间只能激活一种图像模式
        - 只有在 "immersive" 或 "ego" 模式下，且启用了 ZMQ 或 WebRTC,才会传输图像到 VR
        - 如果同时启用 ZMQ 和 WebRTC,优先使用 WebRTC

        --------------              -------------------           --------------       -----------------                     -------
         display_mode       |        display behavior         |    image to VR     |      image source        |               Notes
        --------------              -------------------           --------------       -----------------                     ------- 
           immersive        |   fully immersive view (robot)  |     Yes (full)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------
         pass-through       |       Real world view (VR)      |         No         |          N/A             |  even if image source enabled, don't display
        --------------              -------------------           --------------       -----------------                     -------
              ego           |      ego view (robot + VR)      |    Yes (small)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------

        """
        
        print(f"[DEBUG] use_hand_tracking: {use_hand_tracking}")
        print(f"[DEBUG] display_mode: {display_mode}")
        print(f"[DEBUG] webrtc: {webrtc}")
    
        # 保存基本配置
        self.use_hand_tracking = use_hand_tracking
        self.binocular = binocular
        
        # 图像尺寸验证和处理
        if img_shape is None:
            raise ValueError("[TeleVuer] img_shape must be provided.")
        # 图像形状：(高度, 宽度, 3 通道 RGB)
        self.img_shape = (img_shape[0], img_shape[1], 3)
        self.display_fps = display_fps
        # 计算单眼图像尺寸
        self.img_height = self.img_shape[0] # 图像高度
        if self.binocular:
            self.img_width  = self.img_shape[1] // 2 # 双目：宽度减半
        else:
            self.img_width  = self.img_shape[1] # 单目：使用全宽
        # 计算宽高比，用于正确的图像显示
        self.aspect_ratio = self.img_width / self.img_height
        
        # SSL 证书路径解析（优先级：环境变量 > 用户配置目录 > 包根目录）
        env_cert = os.getenv("XR_TELEOP_CERT") # 从环境变量获取证书路径
        env_key = os.getenv("XR_TELEOP_KEY")   # 从环境变量获取私钥路径
        if cert_file is None or key_file is None:
            # 1. 尝试使用环境变量
            if env_cert and env_key:
                cert_file = cert_file or env_cert
                key_file = key_file or env_key
            else:
                # 2. 尝试使用用户配置目录 ~/.config/xr_teleoperate/
                user_conf_dir = Path.home() / ".config" / "xr_teleoperate"
                cert_path_user = user_conf_dir / "cert.pem"
                key_path_user = user_conf_dir / "key.pem"

                if cert_path_user.exists() and key_path_user.exists():
                    cert_file = cert_file or str(cert_path_user)
                    key_file = key_file or str(key_path_user)
                else:
                    # 3. 回退到包根目录（当前逻辑）
                    current_module_dir = Path(__file__).resolve().parent.parent.parent
                    cert_file = cert_file or str(current_module_dir / "cert.pem")
                    key_file = key_file or str(current_module_dir / "key.pem")

            # 4.Last resort: current working directory
            if not os.path.exists(cert_file) or not os.path.exists(key_file):
                cert_file = cert_file or str(Path.cwd() / "cert.pem")
                key_file = key_file or str(Path.cwd() / "key.pem")

        # 初始化 Vuer 实例
        # host='0.0.0.0' 监听所有网络接口; queries=dict(grid=False) 禁用网格显示; queue_len=3 事件队列长度为 3
        self.vuer = Vuer(host='0.0.0.0', port=port, cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)
        
        # 注册摄像头移动事件处理器
        print(f"[DEBUG] Registering event handlers for hand tracking: {self.use_hand_tracking}")
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        
        # 根据追踪模式注册相应的事件处理器
        if self.use_hand_tracking:
            print(f"[DEBUG] Registering HAND_MOVE handler")
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            print(f"[DEBUG] Registering CONTROLLER_MOVE handler")
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move) # 控制器追踪

        # 保存显示模式配置
        self.display_mode = display_mode
        self.zmq = zmq
        self.webrtc = webrtc
        self.webrtc_url = webrtc_url

        # 根据显示模式设置主渲染函数
        if self.display_mode == "immersive":
            # 完全沉浸模式
            if self.webrtc:
                # 使用 WebRTC 传输：选择双目或单目的 WebRTC 渲染函数
                fn = self.main_image_binocular_webrtc if self.binocular else self.main_image_monocular_webrtc
            elif self.zmq:
                # 使用 ZMQ 传输：创建共享内存用于图像传输
                # 创建共享内存，大小为图像总像素数 × 每个像素的字节数
                self.img2display_shm = shared_memory.SharedMemory(
                    create=True, 
                    size=np.prod(self.img_shape) * np.uint8().itemsize
                )
                # 将共享内存映射为 NumPy 数组，用于快速图像数据写入
                self.img2display = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img2display_shm.buf)
                
                # 初始化图像传输相关的线程控制变量
                self.latest_frame = None  # 最新的图像帧
                self.new_frame_event = threading.Event()  # 新帧到达事件
                self.stop_writer_event = threading.Event()  # 停止写入线程事件
                
                # 创建并启动写入线程（守护线程，主程序退出时自动结束）
                self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
                self.writer_thread.start()
                
                # 选择双目或单目的 ZMQ 渲染函数
                fn = self.main_image_binocular_zmq if self.binocular else self.main_image_monocular_zmq
            else:
                raise ValueError("[TeleVuer] immersive mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "ego":
            # 自我视角模式（小窗口显示）
            if self.webrtc:
                # 使用 WebRTC 传输：选择双目或单目的 ego 模式 WebRTC 渲染函数
                fn = self.main_image_binocular_webrtc_ego if self.binocular else self.main_image_monocular_webrtc_ego
            elif self.zmq:
                # 使用 ZMQ 传输：创建共享内存（与 immersive 模式相同）
                self.img2display_shm = shared_memory.SharedMemory(
                    create=True, 
                    size=np.prod(self.img_shape) * np.uint8().itemsize
                )
                self.img2display = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img2display_shm.buf)
                self.latest_frame = None
                self.new_frame_event = threading.Event()
                self.stop_writer_event = threading.Event()
                self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
                self.writer_thread.start()
                
                # 选择双目或单目的 ego 模式 ZMQ 渲染函数
                fn = self.main_image_binocular_zmq_ego if self.binocular else self.main_image_monocular_zmq_ego
            else:
                raise ValueError("[TeleVuer] ego mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "pass-through":
            # 穿透模式：只显示真实世界，不需要图像传输
            fn = self.main_pass_through
        else:
            raise ValueError(f"[TeleVuer] Unknown display_mode: {self.display_mode}")
        
        # 生成异步渲染函数（但不立即启动）
        self.vuer.spawn(start=False)(fn)

        # ==================== 创建共享内存用于多进程数据同步 ====================
        # 因为 Vuer 运行在独立进程中，需要使用共享内存来传递数据到主进程
        
        self.head_pose_shared = Array('d', 16, lock=True) # 头部姿态矩阵（4x4 = 16 个元素）
        self.left_arm_pose_shared = Array('d', 16, lock=True) # 左手臂姿态矩阵（各 4x4 = 16 个元素）
        self.right_arm_pose_shared = Array('d', 16, lock=True) # 右手臂姿态矩阵（4x4 = 16 个元素）
        # 根据追踪模式创建不同的共享内存
        if self.use_hand_tracking:
            self.left_hand_position_shared = Array('d', 75, lock=True)
            self.right_hand_position_shared = Array('d', 75, lock=True)
            self.left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self.right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self.left_hand_pinch_shared = Value('b', False, lock=True)
            self.left_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.left_hand_squeeze_shared = Value('b', False, lock=True)
            self.left_hand_squeezeValue_shared = Value('d', 0.0, lock=True)

            self.right_hand_pinch_shared = Value('b', False, lock=True)
            self.right_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.right_hand_squeeze_shared = Value('b', False, lock=True)
            self.right_hand_squeezeValue_shared = Value('d', 0.0, lock=True)
        else:
            self.left_ctrl_trigger_shared = Value('b', False, lock=True)
            self.left_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.left_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.left_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.left_ctrl_aButton_shared = Value('b', False, lock=True)
            self.left_ctrl_bButton_shared = Value('b', False, lock=True)

            self.right_ctrl_trigger_shared = Value('b', False, lock=True)
            self.right_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.right_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.right_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.right_ctrl_aButton_shared = Value('b', False, lock=True)
            self.right_ctrl_bButton_shared = Value('b', False, lock=True)
        
        # 创建并启动 Vuer 进程（守护进程）
        self.process = Process(target=self._vuer_run)
        self.process.daemon = True  # 设置为守护进程，主程序退出时自动终止
        self.process.start()
    
    def _vuer_run(self):
        """
            在独立进程中运行 Vuer 服务器, 这个方法是 Vuer 进程的入口点，负责启动 WebXR 服务器
        """
        try:
            self.vuer.run() # 启动 Vuer 服务器（阻塞调用）
        except KeyboardInterrupt:
            pass # 捕获键盘中断，静默处理
        except Exception as e:
            print(f"Vuer encountered an error: {e}")
        finally:
            # 无论是否发生异常，都要设置停止写入事件
            if hasattr(self, "stop_writer_event"):
                self.stop_writer_event.set()

    def _xr_render_loop(self):
        """ XR 渲染循环（运行在独立线程中）
        功能：持续等待新图像帧，将其从 BGR 转换为 RGB 格式，然后写入共享内存
        这样 Vuer 进程可以直接从共享内存读取图像数据进行渲染
        """
        # 循环直到收到停止信号
        while not self.stop_writer_event.is_set():
            # 等待新帧事件，超时时间 0.1 秒
            if not self.new_frame_event.wait(timeout=0.1):
                continue # 超时则继续等待
            # 清除事件标志，准备接收下一帧
            self.new_frame_event.clear()
            # 如果没有最新帧，继续等待
            if self.latest_frame is None:
                continue
            latest_frame = self.latest_frame # 获取最新帧的副本（避免并发修改问题）
            latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB) # 转换颜色空间：OpenCV 默认使用 BGR，但 VR 渲染需要 RGB
            self.img2display[:] = latest_frame # 将转换后的图像写入共享内存
    
    def render_to_xr(self, image):
        """ 将图像渲染到 XR 设备（仅用于 ZMQ 模式）
        :param image: np.ndarray - 要渲染的图像(BGR 格式)
        注意：当使用 WebRTC 或穿透模式时，这个方法会被忽略
        """
        if self.webrtc or self.display_mode == "pass-through":
            # WebRTC 模式或穿透模式下，不需要通过这个方法传递图像
            print("[TeleVuer] Warning: render_to_xr is ignored when webrtc is enabled or pass_through is True.")
            return
        # 保存最新帧并通知写入线程
        self.latest_frame = image
        self.new_frame_event.set()

    def close(self):
        """ 关闭 TeleVuer,释放所有资源
        功能：
            1. 终止 Vuer 进程
            2. 停止图像写入线程
            3. 关闭并释放共享内存
        """
        # 终止 Vuer 进程
        self.process.terminate()
        self.process.join(timeout=0.5)
        # 如果使用 ZMQ 模式，需要清理图像传输相关资源
        if self.display_mode in ("immersive", "ego") and not self.webrtc:
            # 设置停止事件，通知写入线程退出
            self.stop_writer_event.set()
            self.new_frame_event.set() # 唤醒可能正在等待的线程
            self.writer_thread.join(timeout=0.5) # 等待写入线程结束
            try:
                # 关闭并释放共享内存
                self.img2display_shm.close() # 关闭共享内存句柄
                self.img2display_shm.unlink() # 删除共享内存对象
            except:
                pass

    # ==================== 事件处理器 ====================
    async def on_cam_move(self, event, session, fps=60):
        """ 摄像头移动事件处理器: 当 VR 头显移动时，这个函数会被调用，更新头部姿态矩阵
        :param event: 包含摄像头姿态数据的事件对象
        :param session: Vuer 会话对象
        :param fps: 事件更新频率
        """
        try:
            # 使用锁保护共享内存的写入操作
            with self.head_pose_shared.get_lock():
                # 从事件中提取摄像头变换矩阵并保存到共享内存
                # 矩阵是 4x4 的列主序格式（OpenXR 标准）
                self.head_pose_shared[:] = event.value["camera"]["matrix"]
        except:
            pass

    async def on_controller_move(self, event, session, fps=60):
        """ 控制器移动事件处理器
        :param event: 包含控制器姿态和状态的事件对象
        :param session: Vuer 会话对象
        :param fps: 事件更新频率
        
        参考文档:https://docs.vuer.ai/en/latest/examples/20_motion_controllers.html
        """

        try:
            # 更新左右控制器的姿态矩阵
            with self.left_arm_pose_shared.get_lock():
                self.left_arm_pose_shared[:] = event.value["left"]
            with self.right_arm_pose_shared.get_lock():
                self.right_arm_pose_shared[:] = event.value["right"]
            # 获取左右控制器的按钮状态
            left_controller = event.value["leftState"]
            right_controller = event.value["rightState"]
            
            # 定义内部函数：提取控制器的按钮状态
            def extract_controllers(controllerState, prefix):
                """ 从控制器状态字典中提取各个按钮的值
                :param controllerState: dict - 控制器状态字典
                :param prefix: str - 前缀（"left" 或 "right"），用于确定要更新哪个共享变量
                """
                # trigger
                with getattr(self, f"{prefix}_ctrl_trigger_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_trigger_shared").value = bool(controllerState.get("trigger", False))
                with getattr(self, f"{prefix}_ctrl_triggerValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_triggerValue_shared").value = float(controllerState.get("triggerValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_ctrl_squeeze_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_squeeze_shared").value = bool(controllerState.get("squeeze", False))
                with getattr(self, f"{prefix}_ctrl_squeezeValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_squeezeValue_shared").value = float(controllerState.get("squeezeValue", 0.0))
                # thumbstick
                with getattr(self, f"{prefix}_ctrl_thumbstick_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_thumbstick_shared").value = bool(controllerState.get("thumbstick", False))
                with getattr(self, f"{prefix}_ctrl_thumbstickValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_thumbstickValue_shared")[:] = controllerState.get("thumbstickValue", [0.0, 0.0])
                # buttons
                with getattr(self, f"{prefix}_ctrl_aButton_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_aButton_shared").value = bool(controllerState.get("aButton", False))
                with getattr(self, f"{prefix}_ctrl_bButton_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_bButton_shared").value = bool(controllerState.get("bButton", False))

            extract_controllers(left_controller, "left")
            extract_controllers(right_controller, "right")
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        """ 手部追踪事件处理器
        :param event: 包含手部追踪数据的事件对象
        :param session: Vuer 会话对象
        :param fps: 事件更新频率
        
        参考文档:https://docs.vuer.ai/en/latest/examples/19_hand_tracking.html
        """
        print(f"[DEBUG] Hand move event received: {event}")
        try:
            # 获取左右手的原始数据和状态
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            left_hand = event.value["leftState"]
            right_hand = event.value["rightState"]
            # 定义内部函数：提取手部姿态数据
            def extract_hand_poses(hand_data, arm_pose_shared, hand_position_shared, hand_orientation_shared):
                """ 从手部数据中提取手臂姿态和各个手指关节的位置/方向
                :param hand_data: list - 手部数据，包含 1 个手臂矩阵 + 25 个关节矩阵，每个 16 个元素
                :param arm_pose_shared: Array - 手臂姿态共享内存
                :param hand_position_shared: Array - 手部位置共享内存(25 个点 x 3 坐标)
                :param hand_orientation_shared: Array - 手部方向共享内存(25 个点 x 3x3 旋转矩阵)
                """
                # 提取手臂姿态（前 16 个元素）
                with arm_pose_shared.get_lock():
                    arm_pose_shared[:] = hand_data[0:16]

                # 提取 25 个关键点的位置
                with hand_position_shared.get_lock():
                    for i in range(25):
                        base = i * 16 # 每个关节矩阵的起始索引
                        # 位置是矩阵的平移分量（第 13、14、15 个元素，索引 12、13、14）
                        hand_position_shared[i * 3: i * 3 + 3] = [
                            hand_data[base + 12],  # X 坐标
                            hand_data[base + 13],  # Y 坐标
                            hand_data[base + 14]   # Z 坐标
                        ]
                        
                # 提取 25 个关键点的方向（旋转矩阵）
                with hand_orientation_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        # 提取 3x3 旋转矩阵（列主序格式，即 Fortran 顺序）
                        hand_orientation_shared[i * 9: i * 9 + 9] = [
                            hand_data[base + 0], hand_data[base + 1], hand_data[base + 2], # 第1列（X轴）
                            hand_data[base + 4], hand_data[base + 5], hand_data[base + 6], # 第2列（Y轴）
                            hand_data[base + 8], hand_data[base + 9], hand_data[base + 10], # 第3列（Z轴）
                        ]
            # 定义内部函数：提取手势状态（捏合、抓握等）
            def extract_hands(handState, prefix):
                """ 从手势状态字典中提取捏合和抓握的状态值
                :param handState: dict - 手势状态字典，包含 pinch, pinchValue, squeeze, squeezeValue
                :param prefix: str - 前缀（"left" 或 "right"），用于确定要更新哪个共享变量
                """
                # 捏合手势状态和强度
                with getattr(self, f"{prefix}_hand_pinch_shared").get_lock():
                    getattr(self, f"{prefix}_hand_pinch_shared").value = bool(handState.get("pinch", False))
                with getattr(self, f"{prefix}_hand_pinchValue_shared").get_lock():
                    getattr(self, f"{prefix}_hand_pinchValue_shared").value = float(handState.get("pinchValue", 0.0))
                # 抓握手势状态和强度
                with getattr(self, f"{prefix}_hand_squeeze_shared").get_lock():
                    getattr(self, f"{prefix}_hand_squeeze_shared").value = bool(handState.get("squeeze", False))
                with getattr(self, f"{prefix}_hand_squeezeValue_shared").get_lock():
                    getattr(self, f"{prefix}_hand_squeezeValue_shared").value = float(handState.get("squeezeValue", 0.0))
            
            # 提取左手的数据
            extract_hand_poses(left_hand_data, self.left_arm_pose_shared, self.left_hand_position_shared, self.left_hand_orientation_shared)
            extract_hand_poses(right_hand_data, self.right_arm_pose_shared, self.right_hand_position_shared, self.right_hand_orientation_shared)
            # 提取左右手的手势状态
            extract_hands(left_hand, "left")
            extract_hands(right_hand, "right")

        except:
            pass
        
    # ==================== 沉浸模式（immersive）渲染函数 ====================
    ## immersive MODE
    async def main_image_binocular_zmq(self, session):
        """ 沉浸模式下的双目 ZMQ 图像渲染函数
        功能：在 VR 中全屏显示左右眼的立体图像，通过 ZMQ 共享内存获取图像数据
        :param session: Vuer 会话对象
        """
        # 根据追踪模式添加相应的交互组件
        if self.use_hand_tracking:
            # 手部追踪模式：添加手部追踪组件
            session.upsert(
                Hands(
                    stream=True,    # 启用数据流
                    key="hands",    # 组件唯一标识
                    hideLeft=False, #是否隐藏
                    hideRight=False,
                    enable=True
                ),
                to="bgChildren",     # 添加到背景子元素中
            )
        else:
            # 控制器模式：添加运动控制器组件
            session.upsert(
                MotionControllers(
                    stream=True,    # 启用数据流
                    key="motionControllers",
                    left=True,      # 显示左手控制器模型
                    right=True,     # 显示右手控制器模型
                ),
                to="bgChildren",
            )
        # 主渲染循环
        while True:
            # 更新左右眼的图像背景
            session.upsert(
                [
                    ImageBackground(
                        self.img2display[:, :self.img_width], # 左眼图像（图像左半部分）
                        aspect=self.aspect_ratio,             # 宽高比
                        height=1,                             # 显示高度（归一化单位）
                        distanceToCamera=1,                   # 距离摄像头的距离
                        # 底层渲染引擎支持对象和相机的图层二进制位掩码
                        # 下面我们将左右两个图像平面分别设置为 layers=1 和 layers=2
                        # 这两个掩码分别对应左眼相机和右眼相机
                        layers=1,                             # 只在左眼显示
                        format="jpeg",
                        quality=80,                           # JPEG 压缩质量
                        key="background-left",                # 组件唯一标识
                        interpolate=True,                     # 启用插值平滑
                    ),
                    ImageBackground(
                        self.img2display[:, self.img_width:],   # 右眼图像（图像右半部分）
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        layers=2,                               # 只在右眼显示
                        format="jpeg",
                        quality=80,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 等待下一帧，控制帧率
            #  'jpeg' 编码配合 16ms 等待大约可以达到 30fps
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_zmq(self, session):
        """沉浸模式下的单目 ZMQ 图像渲染函数
        功能：在 VR 中全屏显示单目图像，双眼看到相同的画面
        :param session: Vuer 会话对象
        """
        # 根据追踪模式添加交互组件
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False,
                    enable=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        # 主渲染循环
        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display,           # 完整图像（双眼显示相同内容）
                        aspect=self.aspect_ratio,   # 宽高比
                        height=1,                   # 显示高度
                        distanceToCamera=1,         # 距离
                        format="jpeg",              # 编码格式
                        quality=80,                 # 压缩质量
                        key="background-mono",      # 组件标识
                        interpolate=True,           # 插值
                    ),
                ],
                to="bgChildren",
            )
            # 控制帧率
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_binocular_webrtc(self, session):
        """ 沉浸模式下的双目 WebRTC 视频流渲染函数
        功能：通过 WebRTC 协议接收立体视频流并在 VR 中全屏显示
        :param session: Vuer 会话对象
        """
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False,
                    enable=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,           # WebRTC 信令服务器 URL
                    iceServer=None,                 # ICE 服务器配置（可选）
                    iceServers=[],                  # ICE 服务器列表（可选）
                    key="video-quad",               # 组件唯一标识
                    aspect=self.aspect_ratio,       # 视频宽高比
                    height=7,                       # 显示高度（沉浸模式下更大）
                    layout="stereo-left-right"      # 立体视频布局：左右排列
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc(self, session):
        """ 沉浸模式下的单目 WebRTC 视频流渲染函数
        功能：通过 WebRTC 协议接收单目视频流并在 VR 中全屏显示
        :param session: Vuer 会话对象
        """
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False,
                    enable=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCVideoPlane(
                    src=self.webrtc_url,       # WebRTC 信令服务器 URL
                    iceServer=None,             # ICE 服务器配置
                    iceServers=[],              # ICE 服务器列表
                    key="video-quad",           # 组件标识
                    aspect=self.aspect_ratio,   # 视频宽高比
                    height=7,                   # 显示高度
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    # ==================== 自我视角模式（ego）渲染函数 ====================
    async def main_image_binocular_zmq_ego(self, session):
        """ 自我视角模式下的双目 ZMQ 图像渲染函数
        功能：在 VR 视野中心以小窗口显示立体图像，周围显示真实世界
        与沉浸模式的主要区别:height=0.75(更小)，distanceToCamera=2(更远)
        :param session: Vuer 会话对象
        """
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False,
                    enable=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display[:, :self.img_width], # 左眼图像
                        aspect=self.aspect_ratio,
                        height=0.75,                          # 显示高度减小，形成小窗口效果
                        distanceToCamera=2,                   # 距离增加，使窗口看起来更远
                        # 底层渲染引擎支持为对象和相机设置层二进制位掩码。 
                        # 下面我们将左右两个图像平面分别设置为layers=1和layers=2。 
                        # 注意，这两个遮罩分别与左眼摄像头和右眼摄像头相关联。
                        layers=1,                             # 只在左眼显示
                        format="jpeg",
                        quality=80,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        self.img2display[:, self.img_width:],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        layers=2,
                        format="jpeg",
                        quality=80,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_zmq_ego(self, session):
        """ 自我视角模式下的单目 ZMQ 图像渲染函数
        功能：在 VR 视野中心以小窗口显示单目图像
        :param session: Vuer 会话对象
        """
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False,
                    enable=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display,
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        format="jpeg",
                        quality=80,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_binocular_webrtc_ego(self, session):
        """ 自我视角模式下的双目 WebRTC 视频流渲染函数
        功能：通过 WebRTC 协议接收立体视频流并在 VR 视野中心以小窗口显示
        :param session: Vuer 会话对象
        """
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False,
                    enable=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[], 
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,                   # 小窗口高度（比沉浸模式的7小很多）
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc_ego(self, session):
        """ 自我视角模式下的单目 WebRTC 视频流渲染函数
        功能：通过 WebRTC 协议接收单目视频流并在 VR 视野中心以小窗口显示
        :param session: Vuer 会话对象
        """
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False,
                    enable=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[],
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,
                ),
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.display_fps)

    # ==================== 穿透模式（pass-through）渲染函数 ====================
    async def main_pass_through(self, session):
        """ 穿透模式渲染函数
        功能：完全不显示任何视频图像，只显示真实世界（通过 VR 头显的摄像头）,但仍然保持手部追踪或控制器的交互功能
        :param session: Vuer 会话对象
        """
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False,
                    enable=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
            
        # 主循环：什么都不渲染，只保持会话活跃
        while True:
            await asyncio.sleep(1.0 / self.display_fps)

    # ==================== 公共数据属性访问器 ====================
    # 这些属性提供了线程安全的共享数据访问接口
    @property
    def head_pose(self):
        """np.ndarray, shape (4, 4), head SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.head_pose_shared.get_lock():
            return np.array(self.head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        """np.ndarray, shape (4, 4), left arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.left_arm_pose_shared.get_lock():
            return np.array(self.left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        """np.ndarray, shape (4, 4), right arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.right_arm_pose_shared.get_lock():
            return np.array(self.right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== Hand Tracking Data ====================
    @property
    def left_hand_positions(self):
        """np.ndarray, shape (25, 3), left hand 25 landmarks' 3D positions."""
        with self.left_hand_position_shared.get_lock():
            return np.array(self.left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        """np.ndarray, shape (25, 3), right hand 25 landmarks' 3D positions."""
        with self.right_hand_position_shared.get_lock():
            return np.array(self.right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), left hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.left_hand_orientation_shared.get_lock():
            return np.array(self.left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def right_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), right hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.right_hand_orientation_shared.get_lock():
            return np.array(self.right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def left_hand_pinch(self):
        """bool, whether left hand is pinching."""
        with self.left_hand_pinch_shared.get_lock():
            return self.left_hand_pinch_shared.value

    @property
    def left_hand_pinchValue(self):
        """float, pinch strength of left hand."""
        with self.left_hand_pinchValue_shared.get_lock():
            return self.left_hand_pinchValue_shared.value

    @property
    def left_hand_squeeze(self):
        """bool, whether left hand is squeezing."""
        with self.left_hand_squeeze_shared.get_lock():
            return self.left_hand_squeeze_shared.value

    @property
    def left_hand_squeezeValue(self):
        """float, squeeze strength of left hand."""
        with self.left_hand_squeezeValue_shared.get_lock():
            return self.left_hand_squeezeValue_shared.value

    @property
    def right_hand_pinch(self):
        """bool, whether right hand is pinching."""
        with self.right_hand_pinch_shared.get_lock():
            return self.right_hand_pinch_shared.value

    @property
    def right_hand_pinchValue(self):
        """float, pinch strength of right hand."""
        with self.right_hand_pinchValue_shared.get_lock():
            return self.right_hand_pinchValue_shared.value

    @property
    def right_hand_squeeze(self):
        """bool, whether right hand is squeezing."""
        with self.right_hand_squeeze_shared.get_lock():
            return self.right_hand_squeeze_shared.value

    @property
    def right_hand_squeezeValue(self):
        """float, squeeze strength of right hand."""
        with self.right_hand_squeezeValue_shared.get_lock():
            return self.right_hand_squeezeValue_shared.value

    # ==================== Controller Data ====================
    @property
    def left_ctrl_trigger(self):
        """bool, left controller trigger pressed or not."""
        with self.left_ctrl_trigger_shared.get_lock():
            return self.left_ctrl_trigger_shared.value

    @property
    def left_ctrl_triggerValue(self):
        """float, left controller trigger analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_triggerValue_shared.get_lock():
            return self.left_ctrl_triggerValue_shared.value

    @property
    def left_ctrl_squeeze(self):
        """bool, left controller squeeze pressed or not."""
        with self.left_ctrl_squeeze_shared.get_lock():
            return self.left_ctrl_squeeze_shared.value

    @property
    def left_ctrl_squeezeValue(self):
        """float, left controller squeeze analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_squeezeValue_shared.get_lock():
            return self.left_ctrl_squeezeValue_shared.value

    @property
    def left_ctrl_thumbstick(self):
        """bool, whether left thumbstick is touched or clicked."""
        with self.left_ctrl_thumbstick_shared.get_lock():
            return self.left_ctrl_thumbstick_shared.value

    @property
    def left_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), left thumbstick 2D axis values (x, y)."""
        with self.left_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.left_ctrl_thumbstickValue_shared[:])

    @property
    def left_ctrl_aButton(self):
        """bool, left controller 'A' button pressed."""
        with self.left_ctrl_aButton_shared.get_lock():
            return self.left_ctrl_aButton_shared.value

    @property
    def left_ctrl_bButton(self):
        """bool, left controller 'B' button pressed."""
        with self.left_ctrl_bButton_shared.get_lock():
            return self.left_ctrl_bButton_shared.value

    @property
    def right_ctrl_trigger(self):
        """bool, right controller trigger pressed or not."""
        with self.right_ctrl_trigger_shared.get_lock():
            return self.right_ctrl_trigger_shared.value

    @property
    def right_ctrl_triggerValue(self):
        """float, right controller trigger analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_triggerValue_shared.get_lock():
            return self.right_ctrl_triggerValue_shared.value

    @property
    def right_ctrl_squeeze(self):
        """bool, right controller squeeze pressed or not."""
        with self.right_ctrl_squeeze_shared.get_lock():
            return self.right_ctrl_squeeze_shared.value

    @property
    def right_ctrl_squeezeValue(self):
        """float, right controller squeeze analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_squeezeValue_shared.get_lock():
            return self.right_ctrl_squeezeValue_shared.value

    @property
    def right_ctrl_thumbstick(self):
        """bool, whether right thumbstick is touched or clicked."""
        with self.right_ctrl_thumbstick_shared.get_lock():
            return self.right_ctrl_thumbstick_shared.value

    @property
    def right_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), right thumbstick 2D axis values (x, y)."""
        with self.right_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.right_ctrl_thumbstickValue_shared[:])

    @property
    def right_ctrl_aButton(self):
        """bool, right controller 'A' button pressed."""
        with self.right_ctrl_aButton_shared.get_lock():
            return self.right_ctrl_aButton_shared.value

    @property
    def right_ctrl_bButton(self):
        """bool, right controller 'B' button pressed."""
        with self.right_ctrl_bButton_shared.get_lock():
            return self.right_ctrl_bButton_shared.value