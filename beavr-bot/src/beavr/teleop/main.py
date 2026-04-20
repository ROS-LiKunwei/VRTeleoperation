#!/usr/bin/env python3
"""
Main entry point for Beavr Teleop system.

This module provides the CLI interface and main execution logic for the teleoperation system.
Uses the structured configuration system with automatic CLI flag generation via Draccus.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import draccus

from beavr.teleop.common.configs.loader import (
    Laterality,
    apply_yaml_preserving_cli,
    load_robot_config,
    load_yaml_config,
)
from beavr.teleop.common.logging.logger import setup_root_logger
from beavr.teleop.configs.constants.models import TeleopConfig

logger = logging.getLogger(__name__)


@dataclass
class MainConfig:
    """主配置：将结构化的遥操作配置与机器人选择结合在一起."""

    # 使用新的结构化配置作为基础(base)
    """ 
        语法拆解：
        1. field 是 Python dataclasses 模块里的一个专门的函数。当普通的默认值（比如 age = 18 或 name = "张三"）无法满足需求时，我们就用 field() 来对这个变量的默认行为进行更高级的定制。
        2. 既然是给个默认值，为什么不直接写成 teleop: TeleopConfig = TeleopConfig() 呢？ 为什么要搞个 default_factory 这么复杂？
            这是因为 Python 中有一个著名的**“可变默认参数陷阱”**:
                如果你写成 teleop = TeleopConfig(),Python 只会在程序刚加载这段代码时，实例化一次 TeleopConfig 对象。
                结果就是：如果你后来创建了 10 个主配置对象，这 10 个对象底下的 teleop 属性，其实是指向内存里同一个 TeleopConfig 的！ 也就是“一处修改，处处跟着变”，这在配置管理中是致命的灾难。
            为了解决这个问题,Python 引入了 default_factory(默认工厂):
                它要求你传入一个可调用的东西（比如一个函数名，或者一个类名）。
                当你每次创建一个新的主对象且没有传 teleop 的值时,系统就会立刻呼叫这个工厂:TeleopConfig()，为你现做一个全新且独立的配置对象。
    """
    teleop: TeleopConfig = field(default_factory=TeleopConfig) 


    # 机器人选择——支持用逗号分隔来指定多个机器人
    robot_name: str = ""

    # 机器人配置的侧别设置(Laterality setting)
    laterality: str = "right"  # Options: "right", "left", "bimanual"

    # 可选的配置文件覆盖项
    config_file: str = "configs/environment/dev.yaml"

    # 数据存储配置
    storage_path: str = "data/recordings"

    # 构建后的机器人结构（在运行时填充）
    robot: Any = field(init=False)

    def __post_init__(self):
        """根据 robot_name(可为多个)初始化机器人配置."""
        if not self.robot_name or self.robot_name == "":
            raise ValueError(
                "robot_name must be provided. Examples:\n"
                "  Single robot: --robot_name=leap\n"
                "  Multiple robots: --robot_name=leap,xarm7\n"
                "Available robots: leap, xarm7, etc."
            )

        # 转换为枚举类型(enum)供内部使用
        self.laterality_enum = Laterality(self.laterality)

        # Get simulation mode from teleop flags
        simulation_mode = self.teleop.flags.sim_env

        # 使用工具函数加载机器人配置（可多个）
        self.robot = load_robot_config(self.robot_name, self.laterality_enum, simulation_mode)

    # TODO: 当完全迁移到新的结构化配置后移除这部分
    # 为向后兼容提供便捷的属性委托(attribute delegation)
    def __getattr__(self, item):
        """将未知属性委托给 teleop 配置，以保持向后兼容."""
        try:
            return getattr(self.teleop, item)
        except AttributeError:
            # 尝试从 teleop.network 中获取（用于扁平化的网络参数访问）
            try:
                return getattr(self.teleop.network, item)
            except AttributeError as exc:
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'") from exc


def run_teleop(config: MainConfig):
    """按给定配置启动遥操作系统的多个进程，并在中断时安全关闭."""

    # 函数入口与日志初始化
    setup_root_logger(logging.DEBUG)

    logger.info("🚀 Starting Beavr Teleop System")
    logger.info(f"📡 Network host: {config.teleop.network.host_address}")
    logger.info(f"🎮 Operation mode: {'ENABLED' if config.teleop.flags.operate else 'DISABLED'}")
    logger.info(f"🎯 Simulation mode: {'ENABLED' if config.teleop.flags.sim_env else 'DISABLED'}")
    logger.info(f"🤖 Robot: {config.robot_name}")

    from beavr.teleop.components import (
        TeleOperator,
    )

    # Initialize the teleoperator with structured config
    teleop = TeleOperator(config)
    processes = teleop.get_processes()

    try:
        # Start all processes
        logger.info(f"🔄 Starting {len(processes)} teleop processes...")
        for process in processes:
            process.start()
            logger.debug(f"  ✅ Started process: {process.name}")

        # Wait for all processes to complete
        logger.info("✨ All processes started. Press Ctrl+C to stop.")
        while any(p.is_alive() for p in processes):
            for p in processes:
                p.join(timeout=0.1)  # 不是无限阻塞等待，而是短超时轮询，这样可以及时响应 KeyboardInterrupt（Ctrl+C）

    # 用户按 Ctrl+C 后，进入优雅停机流程：
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown requested...")

        # 先对仍活着的进程调用 terminate() 发送终止信号 
        for p in processes:
            if p.is_alive():
                p.terminate()

        # 再 join(timeout=2.0) 等它们自行退出
        for p in processes:
            p.join(timeout=2.0)

        # 若还有没退出的，调用 kill() 强制杀掉，并再 join(timeout=1.0)
        for p in processes:
            if p.is_alive():
                logger.warning(f"Process {p.name} did not terminate gracefully - force killing")
                p.kill()
                p.join(timeout=1.0)

    finally:
        # Final cleanup
        # 无论正常结束还是异常，都再检查一次存活进程：对残留进程再尝试 terminate + join(0.5)
        for p in processes:
            if p.is_alive():
                try:
                    p.terminate()
                    p.join(timeout=0.5)
                except Exception as e: # 清理失败会记录错误日志
                    logger.error(f"Error cleaning up process {p.name}: {e}")

        logger.info("🏁 Teleop shutdown complete")


@draccus.wrap()
def main(cfg: MainConfig):
    """
    Main entry point for Beavr Teleop system.

    This function is wrapped with Draccus to automatically generate CLI flags for all
    configuration parameters. Configuration precedence (highest to lowest):
    1. CLI flags (via Draccus)
    2. YAML config file overrides
    3. Default values

    draccus 是一个开源的 Python 配置项解析库:
        它的核心作用是将代码里的数据类(Dataclass)、命令行的输入(CLI)以及配置文件(YAML)无缝结合在一起。
        这样，用户可以通过命令行、配置文件或代码直接设置参数，而无需修改代码。
        配置项的优先级是: CLI > YAML > 代码。
    Examples:
        # Single robot usage
        python -m beavr.teleop.main --robot_name=leap --laterality=right
        python -m beavr.teleop.main --robot_name=xarm7 --laterality=left

        # Multiple robots (composite configuration)
        python -m beavr.teleop.main --robot_name=leap,xarm7 --laterality=right
        python -m beavr.teleop.main --robot_name=leap,xarm7 --laterality=bimanual

        # Use production config
        python -m beavr.teleop.main --robot_name=leap,xarm7 --config_file=config/prod.yaml

        # Override network settings via CLI (highest priority)
        python -m beavr.teleop.main --robot_name=leap --teleop.network.host_address=192.168.1.100

        # Enable simulation mode
        python -m beavr.teleop.main --robot_name=leap --teleop.flags.sim_env=True

        # Override control parameters
        python -m beavr.teleop.main --robot_name=leap,xarm7 --teleop.control.vr_freq=60

        # Override multiple port settings
        python -m beavr.teleop.main --robot_name=xarm7 \
            --teleop.ports.keypoint_stream_port=9000 \
            --teleop.ports.control_stream_port=9001
    """

    # Apply YAML configuration overrides
    # Note: CLI flags (from Draccus) already applied, YAML merges underneath
    yaml_overrides = load_yaml_config(cfg.config_file)
    if yaml_overrides:
        logger.info(f"🔧 Applying YAML overrides from {cfg.config_file}")

        # Apply YAML overrides while preserving CLI flag precedence
        apply_yaml_preserving_cli(cfg, yaml_overrides)

    run_teleop(cfg)


if __name__ == "__main__":
    main()
