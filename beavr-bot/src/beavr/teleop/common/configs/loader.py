"""Configuration utilities for robot teleoperation system.

This module provides comprehensive configuration management including:
- Laterality definitions and validation for robot arm setups
- Multi-robot configuration loading and composition
- YAML configuration file processing with CLI precedence
- Composite robot config merging for multiple robot scenarios

The utilities support both single robot configurations and comma-separated
multi-robot setups via CLI, enabling flexible system composition.
"""

import importlib
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List

import yaml

from beavr.teleop.configs.constants.models import TeleopConfig
from beavr.teleop.configs.robots import TeleopRobotConfig

logger = logging.getLogger(__name__)
_CONFIGS_PKG = "beavr.teleop.configs.robots"


class Laterality(Enum):
    """Enumeration for robot arm laterality options."""

    RIGHT = "right"
    LEFT = "left"
    BIMANUAL = "bimanual"


def validate_laterality(laterality: Laterality):
    """Validate laterality enum parameter."""
    valid_laterality = [Laterality.RIGHT, Laterality.LEFT, Laterality.BIMANUAL]
    if laterality not in valid_laterality:
        raise ValueError(f"Invalid laterality: {laterality}. Must be one of: {valid_laterality}")


def log_laterality_configuration(laterality: Laterality, robot_name: str):
    """Log the laterality configuration being used.

    Args:
        laterality: The laterality setting
        robot_name: Name of the robot being configured
    """
    logger.info(f"Configuring {robot_name} for {laterality.value} arm operation")


@dataclass
class CompositeRobotConfig:
    """
    Composite robot configuration that combines multiple individual robot configs.

    Provides the same interface as individual robot configs but merges components
    from multiple robots into unified lists for the TeleOperator.
    """

    robot_name: str
    robot_configs: List[Any] = field(default_factory=list)

    # Merged component lists (populated by _merge_components)
    detector: List[Any] = field(default_factory=list, init=False)
    transforms: List[Any] = field(default_factory=list, init=False)
    visualizers: List[Any] = field(default_factory=list, init=False)
    operators: List[Any] = field(default_factory=list, init=False)
    robots: List[Any] = field(default_factory=list, init=False)
    environment: List[Any] = field(default_factory=list, init=False)
    recorded_data: List[Any] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Merge components from all individual robot configs."""
        self._merge_components()

    def _merge_components(self):
        """
        Merge component configs from individual robot configs into unified lists.

        Since SharedComponentRegistry ensures singleton instances per hand side,
        we can simply concatenate all components without complex deduplication logic.
        """
        logger.info(f"🔄 Merging component configs from {len(self.robot_configs)} robot configs")

        for config in self.robot_configs:
            logger.debug(f"  📦 Processing config for: {getattr(config, 'robot_name', 'unknown')}")

            # Merge detector configs (shared components are automatically deduplicated by registry)
            if hasattr(config, "detector"):
                detector = config.detector
                if detector:
                    detectors = detector if isinstance(detector, list) else [detector]
                    self.detector.extend(detectors)

            # Merge transform configs (shared components are automatically deduplicated by registry)
            if hasattr(config, "transforms"):
                transforms = config.transforms or []
                self.transforms.extend(transforms)

            # Merge visualizer configs (shared components are automatically deduplicated by registry)
            if hasattr(config, "visualizers"):
                visualizers = config.visualizers or []
                self.visualizers.extend(visualizers)

            # Merge operator configs (always robot-specific, so keep all)
            if hasattr(config, "operators"):
                operators = config.operators or []
                self.operators.extend(operators)

            # Merge robot configs (always robot-specific, so keep all)
            if hasattr(config, "robots"):
                robots = config.robots or []
                self.robots.extend(robots)

            # Merge environment configs
            if hasattr(config, "environment"):
                environment = config.environment or []
                self.environment.extend(environment)

            # Merge recorded_data configs
            if hasattr(config, "recorded_data"):
                recorded_data = config.recorded_data or []
                self.recorded_data.extend(recorded_data)

        # Deduplicate shared components using object identity (registry ensures same instances for same hand side)
        def deduplicate_by_identity(items):
            seen = set()
            result = []
            for item in items:
                item_id = id(item)
                if item_id not in seen:
                    seen.add(item_id)
                    result.append(item)
            return result

        self.detector = deduplicate_by_identity(self.detector)
        self.transforms = deduplicate_by_identity(self.transforms)
        self.visualizers = deduplicate_by_identity(self.visualizers)

        logger.info(f"  📡 Final detector configs: {len(self.detector)}")
        logger.info(f"  🔄 Final transform configs: {len(self.transforms)}")
        logger.info(f"  👁️  Final visualizer configs: {len(self.visualizers)}")
        logger.info(f"  🎮 Robot-specific operator configs: {len(self.operators)}")
        logger.info(f"  🤖 Robot-specific robot configs: {len(self.robots)}")
        logger.info(f"  🌍 Environment configs: {len(self.environment)}")

    def build(self):
        """
        Build method for compatibility with existing robot config interface.

        Builds all merged config objects and returns components in the format
        expected by TeleOperator.
        """
        logger.info(f"🔨 Building composite robot configuration: {self.robot_name}")

        # Build detector configs
        built_detectors = []
        for detector_config in self.detector:
            if hasattr(detector_config, "build"):
                built_detectors.append(detector_config.build())
            else:
                built_detectors.append(detector_config)

        # Build transform configs
        built_transforms = []
        for transform_config in self.transforms:
            if hasattr(transform_config, "build"):
                built_transforms.append(transform_config.build())
            else:
                built_transforms.append(transform_config)

        # Build visualizer configs
        built_visualizers = []
        for visualizer_config in self.visualizers:
            if hasattr(visualizer_config, "build"):
                built_visualizers.append(visualizer_config.build())
            else:
                built_visualizers.append(visualizer_config)

        # Build operator configs
        built_operators = []
        for operator_config in self.operators:
            if hasattr(operator_config, "build"):
                built_operators.append(operator_config.build())
            else:
                built_operators.append(operator_config)

        # Build robot configs
        built_robots = []
        for robot_config in self.robots:
            if hasattr(robot_config, "build"):
                built_robots.append(robot_config.build())
            else:
                built_robots.append(robot_config)

        # Build environment configs
        built_environment = []
        for env_config in self.environment:
            if hasattr(env_config, "build"):
                built_environment.append(env_config.build())
            else:
                built_environment.append(env_config)

        logger.info(
            f"  🏗️  Built {len(built_detectors)} detectors, {len(built_transforms)} transforms, "
            f"{len(built_visualizers)} visualizers, {len(built_operators)} operators, "
            f"{len(built_robots)} robots, {len(built_environment)} environments"
        )

        return {
            "robot_name": self.robot_name,
            "detector": built_detectors,
            "transforms": built_transforms,
            "visualizers": built_visualizers,
            "operators": built_operators,
            "robots": built_robots,
            "environment": built_environment,
            "recorded_data": self.recorded_data,
        }


def load_robot_config(robot_name: str, laterality: Laterality, simulation_mode: bool = False) -> Any:
    """
    根据机器人名称（可多个）和侧别(laterality)加载机器人配置.
    同时支持单机器人以及用逗号分隔的多机器人.

    Args:
        robot_name: Single robot name or comma-separated list (e.g., "leap,xarm7")
        laterality: Laterality enum for robot configuration
        simulation_mode: Whether to run in simulation mode

    Returns:
        Single robot config or CompositeRobotConfig for multiple robots

    Raises:
        ValueError: If robot config module cannot be found
    """
    # 解析机器人名称（支持逗号分隔列表）
    robot_names = [name.strip() for name in robot_name.split(",")] # strip() 的作用是消除字符串开头和结尾的所有空白字符（包括空格、换行符、制表符等）
    logger.info(f"🤖 Loading robot config(s): {robot_names}")

    if len(robot_names) == 1:
        # Single robot - use existing logic
        return _load_single_robot(robot_names[0], laterality, simulation_mode)
    else:
        # Multiple robots - create composite config
        return _load_multiple_robots(robot_names, laterality, simulation_mode)


def _load_single_robot(robot_name: str, laterality: Laterality, simulation_mode: bool = False) -> Any:
    """Load configuration for a single robot."""
    logger.info(f"📦 Loading single robot config: {robot_name}")

    # Import the config module
    try:
        importlib.import_module(f"{_CONFIGS_PKG}.{robot_name}_config")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Could not find config module for robot '{robot_name}'. Available configs in {_CONFIGS_PKG}/"
        ) from exc

    # 检索并实例化已注册的机器人配置
    cfg_cls = TeleopRobotConfig.get_choice_class(robot_name) # 直接去花名册里查：“名字叫 xarm7 的是谁注册的？” 花名册直接返回 XArm7Config 这个类，然后系统就可以愉快地去实例化它了。

    # Build kwargs dynamically to ensure compatibility with robot config __init__ signature
    cfg_kwargs = {}
    if "laterality" in getattr(cfg_cls, "__dataclass_fields__", {}):
        """
            __dataclass_fields__: 这是 Python 中 @dataclass 装饰器自动给类生成的一份**“说明书”**（字典形式），里面罗列了这个类所有定义好的变量名字
            getattr(..., {})：安全地获取这本说明书。万一这个类不是用 @dataclass 写的，拿不到说明书，就返回一个空字典 {}，防止程序报错崩溃。
        """
        cfg_kwargs["laterality"] = laterality
    if "simulation_mode" in getattr(cfg_cls, "__dataclass_fields__", {}):
        cfg_kwargs["simulation_mode"] = simulation_mode

    robot_config = cfg_cls(**cfg_kwargs)

    logger.info(f"✅ Loaded robot configuration: {robot_name} with laterality: {laterality.value}, simulation_mode: {simulation_mode}")
    return robot_config


def _load_multiple_robots(robot_names: List[str], laterality: Laterality, simulation_mode: bool = False) -> CompositeRobotConfig:
    """为多个机器人加载并合并配置"""
    logger.info(f"📦 Loading composite robot config: {','.join(robot_names)}")

    individual_configs = []

    for robot_name in robot_names:
        # Load individual robot config
        robot_config = _load_single_robot(robot_name, laterality, simulation_mode)
        individual_configs.append(robot_config)
        logger.info(f"  ✅ Loaded {robot_name} config with laterality: {laterality.value}, simulation_mode: {simulation_mode}")

    # Create composite config
    composite_name = ",".join(robot_names)
    composite_config = CompositeRobotConfig(robot_name=composite_name, robot_configs=individual_configs)

    logger.info(f"✅ Created composite robot configuration: {composite_name}")
    return composite_config


def load_yaml_config(config_file: str) -> dict:
    """
    Load YAML configuration file with error handling.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Dictionary of configuration overrides
    """
    if not os.path.exists(config_file):
        logger.warning(f"⚠️  Config file not found: {config_file} - using defaults")
        return {}

    try:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            if config_data is None:
                return {}
            logger.info(f"📄 Loaded config overrides from: {config_file}")
            return config_data
    except yaml.YAMLError as e:
        logger.error(f"❌ Failed to parse YAML config {config_file}: {e}")
        return {}
    except Exception as e:
        logger.error(f"❌ Failed to load config {config_file}: {e}")
        return {}


def apply_section_override(target: Any, yaml_obj: dict, defaults: Any, section_name: str):
    """
    Apply YAML overrides to a config section while preserving CLI flag precedence.

    Args:
        target: The target config object to modify
        yaml_obj: The YAML overrides dictionary for this section
        defaults: The default config object for comparison
        section_name: Section name for logging purposes
    """
    overrides = yaml_obj or {}

    for key, yaml_value in overrides.items():
        try:
            current = getattr(target, key)
            default = getattr(defaults, key)

            # If current value equals default, it wasn't overridden by CLI
            if current == default:
                setattr(target, key, yaml_value)
                logger.debug(f"📝 Applied YAML override: {section_name}.{key} = {yaml_value}")
            else:
                logger.debug(f"🚫 Skipped YAML override (CLI precedence): {section_name}.{key}")

        except AttributeError as e:
            logger.warning(f"⚠️  Unknown config key in YAML: {section_name}.{key} - {e}")


def apply_yaml_preserving_cli(target_cfg: Any, yaml_overrides: dict):
    """
    Apply YAML overrides while preserving CLI flag precedence.

    Uses _apply_section_override for each config section to maintain clean separation.
    """

    # Safety check - ensure teleop section exists
    teleop_overrides = yaml_overrides.get("teleop", {})
    if not teleop_overrides:
        logger.debug("No 'teleop' section found in YAML config")
        return

    # Create defaults for comparison
    defaults = TeleopConfig()

    # Apply each config section systematically
    sections = [
        (target_cfg.teleop.network, teleop_overrides.get("network"), defaults.network, "teleop.network"),
        (target_cfg.teleop.ports, teleop_overrides.get("ports"), defaults.ports, "teleop.ports"),
        (target_cfg.teleop.flags, teleop_overrides.get("flags"), defaults.flags, "teleop.flags"),
        (target_cfg.teleop.control, teleop_overrides.get("control"), defaults.control, "teleop.control"),
        (target_cfg.teleop.camera, teleop_overrides.get("camera"), defaults.camera, "teleop.camera"),
    ]

    for target_section, yaml_section, default_section, section_name in sections:
        if yaml_section:  # Only process if section exists in YAML
            apply_section_override(target_section, yaml_section, default_section, section_name)

    # Handle robot-level overrides if any (rare, but possible)
    robot_overrides = yaml_overrides.get("robot", {})
    if robot_overrides and hasattr(target_cfg, "robot"):
        logger.debug("Found robot-level YAML overrides - applying directly")
        for key, value in robot_overrides.items():
            if hasattr(target_cfg.robot, key):
                setattr(target_cfg.robot, key, value)
                logger.debug(f"📝 Applied robot override: robot.{key} = {value}")
            else:
                logger.warning(f"⚠️  Unknown robot config key: robot.{key}")
