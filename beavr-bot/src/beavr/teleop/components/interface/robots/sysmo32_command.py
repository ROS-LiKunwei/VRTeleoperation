"""SYSMO-32 real-interface command helpers.

This module is intentionally sysmo32-specific.  It owns the exact 18-field arm
command contract used by ``/sysmo_left_arm_controller/commands`` and the fixed
hand action contract used by LinkerHand O6 gesture topics.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)

SYSMO32_ARM_JOINT_COUNT = 6
SYSMO32_COMMAND_LENGTH = 18
SYSMO32_LEFT_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
)
SYSMO32_RIGHT_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
)
SYSMO32_HAND_ACTION_RELEASE = 1
SYSMO32_HAND_ACTION_GRASP = 2


@dataclass(frozen=True)
class Sysmo32ArmCommand:
    """Exact 18-field command sent to the real SYSMO-32 arm controller."""

    timestamp_s: float
    values: Tuple[float, ...]

    def __post_init__(self):
        if len(self.values) != SYSMO32_COMMAND_LENGTH:
            raise ValueError(f"SYSMO-32 arm command length must be 18, got {len(self.values)}")
        if not np.all(np.isfinite(np.asarray(self.values, dtype=np.float64))):
            raise ValueError("SYSMO-32 arm command contains NaN/Inf")

    @property
    def left_arm(self) -> Tuple[float, ...]:
        return self.values[0:6]

    @property
    def right_arm(self) -> Tuple[float, ...]:
        return self.values[6:12]

    @property
    def speed_mode(self) -> float:
        return self.values[12]

    @property
    def reserved(self) -> Tuple[float, ...]:
        return self.values[13:17]

    @property
    def neck_joint(self) -> float:
        return self.values[17]

    def to_list(self) -> list[float]:
        return [float(v) for v in self.values]


@dataclass(frozen=True)
class Sysmo32HandAction:
    """Fixed LinkerHand action command.

    ``action_id`` must be 1 (release) or 2 (grasp bottle).
    """

    timestamp_s: float
    hand_side: str
    action_id: int
    reason: str = ""

    def __post_init__(self):
        if self.hand_side not in (robots.LEFT, robots.RIGHT):
            raise ValueError(f"hand_side must be left/right, got {self.hand_side}")
        if self.action_id not in (SYSMO32_HAND_ACTION_RELEASE, SYSMO32_HAND_ACTION_GRASP):
            raise ValueError(f"SYSMO-32 hand action must be 1 or 2, got {self.action_id}")


@dataclass
class Sysmo32ArmSafetyConfig:
    """Safety limits for building the real arm command."""

    speed_mode: float = 0.0
    reserved: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    neck_joint: float = 0.0
    joint_lower_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([-3.14] * 12))
    joint_upper_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([3.14] * 12))
    max_joint_velocity_rad_s: Tuple[float, ...] = field(default_factory=lambda: tuple([3.0] * 12))
    max_joint_jump_rad: float = 0.35  # 最大跳转角度，单个控制周期内的最大角度变化
    max_translation_step_m: float = 0.08
    max_rotation_step_rad: float = 0.24
    workspace_limits: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (-0.5, 1.5),
        }
    )

    def __post_init__(self):
        if len(self.reserved) != 4:
            raise ValueError("reserved must contain exactly 4 values")
        for name, values in (
            ("joint_lower_limits_rad", self.joint_lower_limits_rad),
            ("joint_upper_limits_rad", self.joint_upper_limits_rad),
            ("max_joint_velocity_rad_s", self.max_joint_velocity_rad_s),
        ):
            if len(values) != 12:
                raise ValueError(f"{name} must contain 12 values")

class Sysmo32CommandBuilder:
    """Build the exact 18-field ``Float64MultiArray.data`` payload."""

    def __init__(self, safety_config: Optional[Sysmo32ArmSafetyConfig] = None):
        self.config = safety_config or Sysmo32ArmSafetyConfig()

    def build(
        self,
        left_arm_joints_rad: Sequence[float],
        right_arm_joints_rad: Sequence[float],
        timestamp_s: Optional[float] = None,
    ) -> Sysmo32ArmCommand:
        left = _as_joint_vector(left_arm_joints_rad, SYSMO32_ARM_JOINT_COUNT, "left_arm")
        right = _as_joint_vector(right_arm_joints_rad, SYSMO32_ARM_JOINT_COUNT, "right_arm")
        values = (
            tuple(float(v) for v in left)
            + tuple(float(v) for v in right)
            + (float(self.config.speed_mode),)
            + tuple(float(v) for v in self.config.reserved)
            + (float(self.config.neck_joint),)
        )
        return Sysmo32ArmCommand(timestamp_s=timestamp_s or time.time(), values=values)


class Sysmo32CommandLimiter:
    """Validate, clip and rate-limit the 12 arm joints before publishing."""

    def __init__(self, safety_config: Optional[Sysmo32ArmSafetyConfig] = None):
        self.config = safety_config or Sysmo32ArmSafetyConfig()
        self._last_command: Optional[Sysmo32ArmCommand] = None
        self._last_publish_time_s: Optional[float] = None

    @property
    def last_command(self) -> Optional[Sysmo32ArmCommand]:
        return self._last_command

    def reset(self, current_joints_rad: Optional[Sequence[float]] = None) -> None:
        self._last_command = None
        self._last_publish_time_s = None
        # 如果传入了当前关节角度，用它初始化状态
        if current_joints_rad is not None:
            joints = _as_joint_vector(current_joints_rad, 12, "current_joints_rad")
            builder = Sysmo32CommandBuilder(self.config)
            self._last_command = builder.build(joints[:6], joints[6:], timestamp_s=time.time())
            self._last_publish_time_s = self._last_command.timestamp_s

    def limit(
        self, command: Sysmo32ArmCommand, now_s: Optional[float] = None
    ) -> tuple[Optional[Sysmo32ArmCommand], str]:
        now = now_s or time.time()
        raw = np.asarray(command.values[:12], dtype=np.float64)
        # NaN/Inf 检查
        if not np.all(np.isfinite(raw)):
            return None, "command contains NaN/Inf"

        # 关节限位裁剪
        lower = np.asarray(self.config.joint_lower_limits_rad, dtype=np.float64)
        upper = np.asarray(self.config.joint_upper_limits_rad, dtype=np.float64)
        clipped = np.clip(raw, lower, upper)
        reason_parts = []
        if not np.allclose(clipped, raw, atol=1e-12):
            reason_parts.append("joint position clipped")

        if self._last_command is not None:
            last = np.asarray(self._last_command.values[:12], dtype=np.float64)
            # 跳变限幅（与上一帧比较）。过大跳变不能直接透传，也不要整条拒绝，
            # 否则 MuJoCo/真实接口会在手腕快速移动后冻结在上一条命令。
            jump = np.abs(clipped - last)
            max_joint_jump = float(self.config.max_joint_jump_rad)
            if max_joint_jump > 0.0 and np.max(jump) > max_joint_jump:
                clipped = last + np.clip(clipped - last, -max_joint_jump, max_joint_jump)
                reason_parts.append(f"joint jump limited: {float(np.max(jump)):.3f} rad")

            dt = max(1e-3, now - (self._last_publish_time_s or now))
            # 速度限幅（基于时间差计算最大允许变化量）
            max_delta = np.asarray(self.config.max_joint_velocity_rad_s, dtype=np.float64) * dt
            limited = last + np.clip(clipped - last, -max_delta, max_delta)
            if not np.allclose(limited, clipped, atol=1e-12):
                reason_parts.append("joint velocity limited")
            clipped = limited

        builder = Sysmo32CommandBuilder(self.config)
        limited_command = builder.build(clipped[:6], clipped[6:], timestamp_s=now)
        self._last_command = limited_command
        self._last_publish_time_s = now
        return limited_command, ", ".join(reason_parts)


@dataclass
class Sysmo32JointStateSnapshot:
    timestamp_s: float
    left_arm: Tuple[float, ...]
    right_arm: Tuple[float, ...]

    @property
    def all_joints(self) -> np.ndarray:
        return np.asarray(self.left_arm + self.right_arm, dtype=np.float64)


class Sysmo32JointStateCache:
    """Parse and cache ``sensor_msgs/JointState`` arm positions."""

    def __init__(self, joint_state_timeout_s: float = 0.8):
        self.joint_state_timeout_s = joint_state_timeout_s
        self._snapshot: Optional[Sysmo32JointStateSnapshot] = None

    @property
    def snapshot(self) -> Optional[Sysmo32JointStateSnapshot]:
        return self._snapshot

    def update_from_joint_state_msg(
        self, msg, now_s: Optional[float] = None
    ) -> Optional[Sysmo32JointStateSnapshot]:
        # 构建关节名 -> 关节位置的映射字典
        name_to_position = dict(zip(list(msg.name), list(msg.position), strict=False))
        try:
            left = tuple(float(name_to_position[name]) for name in SYSMO32_LEFT_JOINT_NAMES)
            right = tuple(float(name_to_position[name]) for name in SYSMO32_RIGHT_JOINT_NAMES)
        except KeyError as exc:
            # 如果缺少某个关节，记录警告并返回上一次的快照
            logger.warning("JointState missing SYSMO-32 arm joint: %s", exc)
            return self._snapshot
        return self.update(left, right, now_s=now_s)

    def update(
        self,
        left_arm: Sequence[float],
        right_arm: Sequence[float],
        now_s: Optional[float] = None,
    ) -> Sysmo32JointStateSnapshot:
        left = tuple(float(v) for v in _as_joint_vector(left_arm, 6, "left_arm"))
        right = tuple(float(v) for v in _as_joint_vector(right_arm, 6, "right_arm"))
        all_values = np.asarray(left + right, dtype=np.float64)
        if not np.all(np.isfinite(all_values)):
            raise ValueError("Joint state contains NaN/Inf")
        self._snapshot = Sysmo32JointStateSnapshot(now_s or time.time(), left, right)
        return self._snapshot

    def is_fresh(self, now_s: Optional[float] = None) -> bool:
        if self._snapshot is None:
            return False
        return (now_s or time.time()) - self._snapshot.timestamp_s <= self.joint_state_timeout_s


class Sysmo32HandGestureMapper:
    """Map VR hand frames to fixed LinkerHand actions with hysteresis.

    The mapper derives grasp/release from the 26 hand keypoints.  The output is
    always the real SYSMO-32 hand contract: 1 = release, 2 = grasp.
    """

    def __init__(
        self,
        default_action: int = SYSMO32_HAND_ACTION_RELEASE,
        grasp_action: int = SYSMO32_HAND_ACTION_GRASP,
        grasp_enter_threshold_m: float = 0.035,
        grasp_exit_threshold_m: float = 0.055,
        confirm_frames: int = 3,
        hand_frame_timeout_s: float = 0.5,
    ):
        self.default_action = default_action
        self.grasp_action = grasp_action
        self.grasp_enter_threshold_m = grasp_enter_threshold_m
        self.grasp_exit_threshold_m = grasp_exit_threshold_m
        self.confirm_frames = max(1, confirm_frames)
        self.hand_frame_timeout_s = hand_frame_timeout_s
        self._actions = {robots.LEFT: default_action, robots.RIGHT: default_action}
        self._confirm_counts = {robots.LEFT: 0, robots.RIGHT: 0}
        self._last_frame_time = {robots.LEFT: 0.0, robots.RIGHT: 0.0}

    def update_from_keypoints(self, hand_side: str, keypoints, now_s: Optional[float] = None) -> int:
        now = now_s or time.time()
        if hand_side not in (robots.LEFT, robots.RIGHT):
            raise ValueError(f"hand_side must be left/right, got {hand_side}")

        arr = np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] < robots.OCULUS_NUM_KEYPOINTS or not np.all(np.isfinite(arr)):
            return self.force_release(hand_side)

        inferred_action, distance = self._infer_action_from_keypoints(arr)
        self._last_frame_time[hand_side] = now

        # 迟滞判断（防抖）
        if self._actions[hand_side] == self.grasp_action:
            if inferred_action == self.default_action and distance > self.grasp_exit_threshold_m:
                self._actions[hand_side] = self.default_action
                self._confirm_counts[hand_side] = 0
        else:
            if inferred_action == self.grasp_action:
                self._confirm_counts[hand_side] += 1
                if self._confirm_counts[hand_side] >= self.confirm_frames:
                    self._actions[hand_side] = self.grasp_action
            else:
                self._confirm_counts[hand_side] = 0

        return self._actions[hand_side]

    def _infer_action_from_keypoints(self, arr: np.ndarray) -> tuple[int, float]:
        joints = robots.OCULUS_JOINTS
        thumb_tip = arr[joints["thumb"][-1]]
        index_tip = arr[joints["index"][-1]]
        pinch_distance = float(np.linalg.norm(index_tip - thumb_tip))
        curl_action = self._action_from_finger_curl(arr)
        if pinch_distance < self.grasp_enter_threshold_m or curl_action == self.grasp_action:
            return self.grasp_action, pinch_distance
        if pinch_distance > self.grasp_exit_threshold_m:
            return self.default_action, pinch_distance
        return self.default_action, pinch_distance

    def _action_from_finger_curl(self, arr: np.ndarray) -> int:
        wrist = arr[0]
        curled = 0
        usable = 0
        for finger in ("index", "middle", "ring", "pinky"):
            chain = robots.OCULUS_JOINTS[finger]
            base = arr[chain[0]]
            tip = arr[chain[-1]]
            base_len = float(np.linalg.norm(base - wrist))
            tip_len = float(np.linalg.norm(tip - wrist))
            if base_len < 1e-6 or tip_len < 1e-6:
                continue
            usable += 1
            if tip_len < base_len * 1.25:
                curled += 1
        return self.grasp_action if usable >= 3 and curled >= 3 else self.default_action

    def action_for(self, hand_side: str, now_s: Optional[float] = None) -> int:
        now = now_s or time.time()
        if now - self._last_frame_time.get(hand_side, 0.0) > self.hand_frame_timeout_s:
            return self.force_release(hand_side)
        return self._actions[hand_side]

    def force_release(self, hand_side: Optional[str] = None) -> int:
        sides = (robots.LEFT, robots.RIGHT) if hand_side is None else (hand_side,)
        for side in sides:
            self._actions[side] = self.default_action
            self._confirm_counts[side] = 0
        return self.default_action

    def has_fresh_frame(self, hand_side: str, now_s: Optional[float] = None) -> bool:
        now = now_s or time.time()
        return now - self._last_frame_time.get(hand_side, 0.0) <= self.hand_frame_timeout_s


def _as_joint_vector(values: Sequence[float], expected_len: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (expected_len,):
        raise ValueError(f"{name} must have shape ({expected_len},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def quaternion_angle_delta_rad(q1_xyzw: Sequence[float], q2_xyzw: Sequence[float]) -> float:
    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    q2 = np.asarray(q2_xyzw, dtype=np.float64)
    if q1.shape != (4,) or q2.shape != (4,):
        return math.inf
    # 计算范数（长度）
    n1 = np.linalg.norm(q1)
    n2 = np.linalg.norm(q2)
    # 检查是否为零向量（无效四元数）
    if n1 < 1e-9 or n2 < 1e-9:
        return math.inf
    # 归一化四元数（确保单位长度）
    q1 = q1 / n1
    q2 = q2 / n2
    # 计算点积的绝对值
    dot = float(abs(np.dot(q1, q2)))
    # 数值稳定性保护：限制在 [-1, 1] 范围内
    dot = min(1.0, max(-1.0, dot))
    # 计算角度差：θ = 2 * arccos(|q1 · q2|)
    return 2.0 * math.acos(dot)


# __all__ 是 Python 的 模块公共接口声明 ，定义了当使用 from module import * 时会被导入的符号列表。
# __all__ 相当于模块的 公开接口文档 ，告诉使用者哪些是可以安全使用的公共 API，哪些是内部实现细节（如以下划线开头的 _as_joint_vector ）。
__all__ = [
    "SYSMO32_ARM_JOINT_COUNT",
    "SYSMO32_COMMAND_LENGTH",
    "SYSMO32_LEFT_JOINT_NAMES",
    "SYSMO32_RIGHT_JOINT_NAMES",
    "SYSMO32_HAND_ACTION_RELEASE",
    "SYSMO32_HAND_ACTION_GRASP",
    "Sysmo32ArmCommand",
    "Sysmo32HandAction",
    "Sysmo32ArmSafetyConfig",
    "Sysmo32CommandBuilder",
    "Sysmo32CommandLimiter",
    "Sysmo32JointStateCache",
    "Sysmo32JointStateSnapshot",
    "Sysmo32HandGestureMapper",
    "quaternion_angle_delta_rad",
]
