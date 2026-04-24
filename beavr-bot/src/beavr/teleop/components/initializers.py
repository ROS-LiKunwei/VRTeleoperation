import logging
import os
from abc import ABC, abstractmethod
from multiprocessing import Process

from beavr.teleop.common.factory.instantiator import instantiate_from_target

logger = logging.getLogger(__name__)


class ProcessInstantiator(ABC):
    """
    进程实例化器基类：
    - 统一保存配置与子进程列表
    - 子类通过实现 `_start_component()` 来定义“如何启动一个组件”
    """

    def __init__(self, configs):
        self.configs = configs
        self.processes = []

    @abstractmethod
    def _start_component(self, configs):
        """Abstract method that must be implemented by subclasses."""
        pass

    def get_processes(self):
        return self.processes


class TeleOperator(ProcessInstantiator):
    """
    Teleop 进程初始化器：根据配置组装 teleoperation 所需的全部子进程。

    Now uses only the structured MainConfig format.
    """

    def __init__(self, main_config):
        """
        Initialize TeleOperator with structured MainConfig.

        Args:
            main_config: MainConfig instance with teleop and robot sections
        """
        self.main_config = main_config
        self.teleop_config = main_config.teleop
        self.robot_config = main_config.robot
        self.processes = []

        logger.info("🔧 Initializing TeleOperator with structured configuration")

        # 启动仿真环境（如果配置中有environment组件）
        if self.teleop_config.flags.sim_env:
            self._init_sim_environment()
        elif hasattr(self.robot_config, "environment") and self.robot_config.environment:
            self._init_sim_environment()
            logger.info("🌐 Auto-starting simulation environment from robot config")
        # 启动手部/关键点检测器
        self._init_detector()
        # 启动关键点变换链路（坐标系/尺度/滤波等）
        self._init_keypoint_transform()
        # 启动可视化（包含可选的 XELA 可视化）
        self._init_visualizers()

        # 启用机器人接口时，启动机器人相关进程（连接/控制/状态等）
        if self.teleop_config.flags.robot_interface:
            self._init_robot_interface()

        # 启用操作器时，启动操作策略/控制器链路
        if self.teleop_config.flags.operate:
            self._init_operator()

    def _start_component(self, configs):
        """
        根据配置动态实例化组件并启动其 `stream()`。

        兼容两种返回形式：
        - 单个组件实例
        - 组件列表（逐个调用 `stream()`)
        """
        try:
            component = instantiate_from_target(configs)
            # Handle both single component and list of components
            if isinstance(component, list):
                for comp in component:
                    comp.stream()
            else:
                component.stream()
        except Exception as e:
            logger.error(f"Error starting component: {e}")
            raise

    def _init_detector(self):
        """初始化 detector 进程。"""
        self.processes.append(Process(target=self._start_component, args=(self.robot_config.detector,))) # 在 Python 中，args 必须接收一个元组（Tuple）

    def _init_sim_environment(self):
        """初始化仿真环境相关进程（可能有多个环境组件）。"""
        for env_config in self.robot_config.environment:
            self.processes.append(Process(target=self._start_component, args=(env_config,)))

    def _init_keypoint_transform(self):
        """初始化关键点变换相关进程（按 transforms 列表顺序启动）。"""
        for transform_config in self.robot_config.transforms:
            self.processes.append(Process(target=self._start_component, args=(transform_config,)))

    def _init_visualizers(self):
        """
        初始化可视化进程：
        - `robot_config.visualizers` 始终按配置启动
        - `run_xela=True` 时额外启动 `robot_config.xela_visualizers`（若存在）
        """
        for visualizer_config in self.robot_config.visualizers:
            self.processes.append(Process(target=self._start_component, args=(visualizer_config,)))
        # XELA visualizer
        if self.teleop_config.flags.run_xela:
            xela_visualizers = getattr(self.robot_config, "xela_visualizers", [])
            for visualizer_config in xela_visualizers:
                self.processes.append(Process(target=self._start_component, args=(visualizer_config,)))

    def _init_operator(self):
        """初始化 operator 进程（可能有多个操作器组件）。"""
        for operator_config in self.robot_config.operators:
            self.processes.append(Process(target=self._start_component, args=(operator_config,)))

    def _init_robot_interface(self):
        """
        初始化机器人接口进程（每个 `robot_config` 对应一个独立子进程）。

        约定：组件的构建/连接逻辑在 `_start_component()` 内部通过
        `instantiate_from_target(...)` 创建实例后，由其 `stream()` 驱动。
        """
        for robot_config in self.robot_config.robots:
            # Derive a human-readable robot name from the dataclass type.
            # Instantiate the robot config in a separate process.
            # This is where the ``build()`` method is called.
            # Create the process first
            process = Process(target=self._start_component, args=(robot_config,))
            self.processes.append(process)


# Data Collector Class
class Collector(ProcessInstantiator):
    """
    数据采集进程初始化器：组装 recorder 链路所需的全部子进程。

    主要职责：
    - 创建演示数据目录
    - 初始化相机录制器
    - 按 `sim_env` 分支初始化仿真/真实机器人记录器
    - `run_xela=True` 时可选初始化传感器记录器
    """

    def __init__(self, main_config, demo_num):
        """
        Initialize Collector with structured MainConfig.

        Args:
            main_config: MainConfig instance with teleop and robot sections
            demo_num: Demonstration number for storage path
        """
        self.main_config = main_config
        self.teleop_config = main_config.teleop
        self.robot_config = main_config.robot
        self.processes = []
        self.demo_num = demo_num

        # 存储路径优先来自配置；没有则回退到默认目录
        storage_path = getattr(main_config, "storage_path", "data/recordings")
        self._storage_path = os.path.join(storage_path, "demonstration_{}".format(self.demo_num))

        self._create_storage_dir()
        self._init_camera_recorders()
        # 记录器按仿真/真实环境分支初始化
        if self.teleop_config.flags.sim_env is True:
            self._init_sim_recorders()
        else:
            logger.info("Initialising robot recorders")
            self._init_robot_recorders()

        # 可选：XELA 传感器记录链路（若 flags 未包含该字段则默认为 False）
        is_xela = getattr(self.teleop_config.flags, "run_xela", False)
        if is_xela is True:
            self._init_sensor_recorders()

    def _create_storage_dir(self):
        """确保当前 demo 的存储目录存在（不存在则创建）。"""
        if os.path.exists(self._storage_path):
            return
        else:
            os.makedirs(self._storage_path)

    def _start_component(self, component):
        """
        启动 recorder 组件的 `stream()`。

        兼容两种输入形式：
        - 单个组件实例
        - 组件列表（逐个调用 `stream()`）
        """
        # Handle both single component and list of components
        if isinstance(component, list):
            for comp in component:
                comp.stream()
        else:
            component.stream()
