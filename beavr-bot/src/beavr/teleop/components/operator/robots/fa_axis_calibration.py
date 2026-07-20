"""FA 双手遥操作坐标轴标定。

该模块只负责根据 PICO 双手腕位置估计“VR 坐标系到机器人坐标系”的
旋转矩阵，不处理网络收发、语音播放和机器人控制。标定动作依次为：
记录起点、双手向前平移、回到起点、双手向左平移，最后回到起点附近
稳定后建立遥操作基准。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FaAxisCalibrationState(str, Enum):
    """标定会话状态；外部根据状态变化播放对应的 PICO 语音。"""

    DISABLED = "DISABLED"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
    # 等待并采集稳定的双手起始位置。
    WAITING_STABLE_ORIGIN = "WAITING_STABLE_ORIGIN"
    CAPTURING_ORIGIN = "CAPTURING_ORIGIN"
    # 采集向前动作；动作太短时必须先回起点再重试。
    CAPTURING_FORWARD = "CAPTURING_FORWARD"
    WAITING_RETURN_BEFORE_FORWARD_RETRY = "WAITING_RETURN_BEFORE_FORWARD_RETRY"
    # 向前动作完成后，先回起点，再采集向左动作。
    WAITING_RETURN_AFTER_FORWARD = "WAITING_RETURN_AFTER_FORWARD"
    CAPTURING_LEFT = "CAPTURING_LEFT"
    WAITING_RETURN_BEFORE_LEFT_RETRY = "WAITING_RETURN_BEFORE_LEFT_RETRY"
    VALIDATING = "VALIDATING"
    # 矩阵已解算成功；双手回到起点附近并稳定后开始遥操作。
    WAITING_RETURN_BEFORE_READY = "WAITING_RETURN_BEFORE_READY"
    READY = "READY"
    # READY 后检测到 PICO 追踪坐标原点跳变，当前结果立即失效。
    INVALIDATED = "INVALIDATED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class FaAxisCalibrationConfig:
    """标定阈值配置，位置和距离的单位均为米。"""

    enable_vr_axis_calibration: bool = True
    require_calibration_each_enable: bool = True
    # 每个姿态的采样窗口，以及判定“已稳定”的持续时间。
    calibration_sample_duration_s: float = 0.4
    calibration_stable_dwell_s: float = 0.3
    calibration_stable_position_epsilon_m: float = 0.02
    # 最终开始遥操作前的回位条件单独放宽，不影响原点采集和动作重试。
    calibration_ready_return_position_epsilon_m: float = 0.04
    calibration_ready_return_dwell_s: float = 0.20
    # 前向和左向动作的有效距离范围。
    calibration_min_motion_distance_m: float = 0.06
    calibration_max_motion_distance_m: float = 0.60
    # 两个动作方向的点积绝对值上限；越接近 0 表示越接近正交。
    calibration_max_axis_abs_dot: float = 0.35
    # 每只手相对双手整体移动方向最多偏差 20 度；较短一侧的移动距离
    # 必须至少达到较长一侧的 65%（距离比例误差不超过 0.35）。
    calibration_max_left_right_direction_error_deg: float = 20.0
    calibration_max_left_right_distance_ratio_error: float = 0.35
    # 最终旋转矩阵必须满足 R^T R = I 且 det(R) = 1。
    calibration_rotation_orthogonality_tolerance: float = 1e-4
    calibration_rotation_determinant_tolerance: float = 1e-4
    calibration_max_timestamp_skew_s: float = 0.15
    calibration_max_frame_age_s: float = 1.0
    # READY 后用于识别 PICO 追踪坐标系整体瞬移的阈值。
    tracking_origin_jump_detection_enabled: bool = True
    tracking_origin_jump_translation_m: float = 0.15
    tracking_origin_jump_rotation_deg: float = 15.0
    tracking_origin_jump_confirm_frames: int = 2
    tracking_origin_jump_interhand_change_m: float = 0.04


@dataclass(frozen=True)
class BimanualWristSample:
    """同一控制周期内配对的左右手腕位置样本。"""

    timestamp_s: float
    left: np.ndarray
    right: np.ndarray
    left_timestamp_s: float
    right_timestamp_s: float

    @property
    def center(self) -> np.ndarray:
        """双手中点，用它描述操作者双手的整体平移动作。"""
        return 0.5 * (self.left + self.right)

    @property
    def interhand_distance(self) -> float:
        return float(np.linalg.norm(self.left - self.right))


@dataclass(frozen=True)
class FaAxisCalibrationResult:
    """标定输出以及用于日志和诊断的中间质量指标。"""

    r_vr_to_robot: np.ndarray
    forward_vector_vr: np.ndarray
    left_vector_vr: np.ndarray
    up_vector_vr: np.ndarray
    forward_motion_distance_left: float
    forward_motion_distance_right: float
    left_motion_distance_left: float
    left_motion_distance_right: float
    forward_left_dot: float
    rotation_determinant: float
    rotation_orthogonality_error: float


def _safe_normalize(vector: np.ndarray, min_norm: float = 1e-9) -> np.ndarray | None:
    """安全地将三维向量单位化；非法值或近零向量返回 None。"""
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < min_norm:
        return None
    return vector / norm


def _mean_position(samples: list[BimanualWristSample], side: str) -> np.ndarray:
    return np.mean([getattr(sample, side) for sample in samples], axis=0)


def _mean_center(samples: list[BimanualWristSample]) -> np.ndarray:
    return np.mean([sample.center for sample in samples], axis=0)


def _window_duration(samples: list[BimanualWristSample]) -> float:
    if len(samples) < 2:
        return 0.0
    return float(samples[-1].timestamp_s - samples[0].timestamp_s)


def _direction_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_unit = _safe_normalize(a)
    b_unit = _safe_normalize(b)
    if a_unit is None or b_unit is None:
        return 180.0
    dot = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _distance_ratio_error(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def _hand_motion_consistent(
    left_delta: np.ndarray,
    right_delta: np.ndarray,
    center_delta: np.ndarray,
    *,
    max_direction_error_deg: float,
    max_distance_ratio_error: float,
) -> bool:
    """判断两只手是否都沿双手中点的整体运动方向平移。"""

    center_unit = _safe_normalize(center_delta)
    left_unit = _safe_normalize(left_delta)
    right_unit = _safe_normalize(right_delta)
    if center_unit is None or left_unit is None or right_unit is None:
        return False

    # 将每只手与中心运动进行比较，而不仅仅是与另一只手进行比较。这可以容忍正常的PICO左/右样本偏斜，同时仍然拒绝明显朝错误方向移动的手
    if _direction_error_deg(left_delta, center_delta) > max_direction_error_deg:
        return False
    if _direction_error_deg(right_delta, center_delta) > max_direction_error_deg:
        return False

    left_distance = float(np.linalg.norm(left_delta))
    right_distance = float(np.linalg.norm(right_delta))
    return _distance_ratio_error(left_distance, right_distance) <= max_distance_ratio_error


def _sampled_hand_motion_consistent(
    origin_samples: list[BimanualWristSample],
    motion_samples: list[BimanualWristSample],
    center_delta: np.ndarray,
    config: FaAxisCalibrationConfig,
) -> bool:
    """检查一段已采集动作中的左右手方向和移动距离是否一致。"""

    origin_left = _mean_position(origin_samples, "left")
    origin_right = _mean_position(origin_samples, "right")
    left_delta = _mean_position(motion_samples, "left") - origin_left
    right_delta = _mean_position(motion_samples, "right") - origin_right
    return _hand_motion_consistent(
        left_delta,
        right_delta,
        center_delta,
        max_direction_error_deg=config.calibration_max_left_right_direction_error_deg,
        max_distance_ratio_error=config.calibration_max_left_right_distance_ratio_error,
    )


def compute_fa_axis_calibration(
    *,
    r_vr_to_robot_nominal: np.ndarray,
    origin_samples: list[BimanualWristSample],
    forward_samples: list[BimanualWristSample],
    left_samples: list[BimanualWristSample],
    config: FaAxisCalibrationConfig,
) -> tuple[FaAxisCalibrationResult | None, str | None]:
    """由起点、向前、向左三组样本解算 VR 到机器人坐标系旋转矩阵。

    成功时返回 ``(result, None)``；失败时返回 ``(None, reason)``。
    ``reason`` 由状态机转换成重试流程或最终失败提示。
    """

    if not origin_samples or not forward_samples or not left_samples:
        return None, "TRACKING_INVALID"

    # 双手中点可以抵消一部分单手姿态和手臂展开程度的影响，更适合表示 “双手共同平移”。两个标定方向都相对于同一个起点计算。
    origin_center = _mean_center(origin_samples)
    forward_center = _mean_center(forward_samples)
    left_center = _mean_center(left_samples)

    forward_delta = forward_center - origin_center
    left_delta = left_center - origin_center
    forward_distance = float(np.linalg.norm(forward_delta))
    left_distance = float(np.linalg.norm(left_delta))

    if not np.isfinite(forward_distance) or not np.isfinite(left_distance):
        return None, "TRACKING_INVALID"
    if forward_distance < config.calibration_min_motion_distance_m:
        return None, "FORWARD_MOTION_TOO_SHORT"
    if left_distance < config.calibration_min_motion_distance_m:
        return None, "LEFT_MOTION_TOO_SHORT"
    if forward_distance > config.calibration_max_motion_distance_m:
        return None, "FORWARD_MOTION_TOO_LARGE"
    if left_distance > config.calibration_max_motion_distance_m:
        return None, "LEFT_MOTION_TOO_LARGE"

    f_raw = _safe_normalize(forward_delta)
    l_raw = _safe_normalize(left_delta)
    if f_raw is None or l_raw is None:
        return None, "TRACKING_INVALID"

    # 前向和左向必须提供两个可区分的方向。若二者过于平行，无法可靠构造
    # 三维正交基，通常表示用户尚未执行左移动作或移动方向错误。
    forward_left_dot = float(np.dot(f_raw, l_raw))
    if abs(forward_left_dot) > config.calibration_max_axis_abs_dot:
        return None, "AXES_NEARLY_COLLINEAR"

    origin_left = _mean_position(origin_samples, "left")
    origin_right = _mean_position(origin_samples, "right")
    forward_left_delta = _mean_position(forward_samples, "left") - origin_left
    forward_right_delta = _mean_position(forward_samples, "right") - origin_right
    left_left_delta = _mean_position(left_samples, "left") - origin_left
    left_right_delta = _mean_position(left_samples, "right") - origin_right

    if not _sampled_hand_motion_consistent(origin_samples, forward_samples, forward_delta, config):
        return None, "FORWARD_HANDS_MISMATCH"
    if not _sampled_hand_motion_consistent(origin_samples, left_samples, left_delta, config):
        return None, "LEFT_HANDS_MISMATCH"

    forward_left_distance = float(np.linalg.norm(forward_left_delta))
    forward_right_distance = float(np.linalg.norm(forward_right_delta))
    left_left_distance = float(np.linalg.norm(left_left_delta))
    left_right_distance = float(np.linalg.norm(left_right_delta))

    # Gram-Schmidt 正交化：保留实测前向 f，去掉左向中沿 f 的分量，
    # 再通过叉乘得到右手系的上方向 u，并重新计算左向消除累计误差。
    f = f_raw
    l_projected = l_raw - float(np.dot(l_raw, f)) * f
    l = _safe_normalize(l_projected)
    if l is None:
        return None, "AXES_NEARLY_COLLINEAR"
    u = _safe_normalize(np.cross(f, l))
    if u is None:
        return None, "AXES_NEARLY_COLLINEAR"
    l = _safe_normalize(np.cross(u, f))
    if l is None:
        return None, "AXES_NEARLY_COLLINEAR"

    # 每一列分别是在 VR 坐标中测得的机器人前、左、上方向。
    e_measured_vr = np.column_stack([f, l, u])
    if not np.all(np.isfinite(e_measured_vr)) or np.linalg.det(e_measured_vr) <= 0.0:
        return None, "TRACKING_INVALID"

    # nominal 矩阵给出系统默认轴映射。实测基相对 nominal 基的旋转用于<修正头显安装偏差>和<当前追踪坐标系偏差>。
    r_vr_to_robot_nominal = np.asarray(r_vr_to_robot_nominal, dtype=np.float64).reshape(3, 3)
    e_nominal_vr = r_vr_to_robot_nominal.T
    r_correction_vr = e_nominal_vr @ e_measured_vr.T
    r_calibrated_raw = r_vr_to_robot_nominal @ r_correction_vr

    # 数值误差可能使原始结果不再是严格旋转矩阵。使用 SVD 投影到最近的
    # SO(3) 矩阵，同时通过 d 避免产生 det=-1 的镜像变换。
    try:
        u_svd, _, vt_svd = np.linalg.svd(r_calibrated_raw)
    except np.linalg.LinAlgError:
        return None, "SVD_PROJECTION_FAILED"
    d = np.eye(3, dtype=np.float64)
    d[2, 2] = np.linalg.det(u_svd @ vt_svd)
    r_calibrated = u_svd @ d @ vt_svd

    orthogonality_error = float(np.linalg.norm(r_calibrated.T @ r_calibrated - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(r_calibrated))
    if orthogonality_error > config.calibration_rotation_orthogonality_tolerance:
        return None, "SVD_PROJECTION_FAILED"
    if abs(determinant - 1.0) > config.calibration_rotation_determinant_tolerance:
        return None, "SVD_PROJECTION_FAILED"

    return (
        FaAxisCalibrationResult(
            r_vr_to_robot=r_calibrated,
            forward_vector_vr=f,
            left_vector_vr=l,
            up_vector_vr=u,
            forward_motion_distance_left=forward_left_distance,
            forward_motion_distance_right=forward_right_distance,
            left_motion_distance_left=left_left_distance,
            left_motion_distance_right=left_right_distance,
            forward_left_dot=forward_left_dot,
            rotation_determinant=determinant,
            rotation_orthogonality_error=orthogonality_error,
        ),
        None,
    )


class FaAxisCalibrationSession:
    """由连续腕部样本自动推进的双手标定状态机。"""

    def __init__(self, r_vr_to_robot_nominal: np.ndarray, config: FaAxisCalibrationConfig):
        self.r_vr_to_robot_nominal = np.asarray(r_vr_to_robot_nominal, dtype=np.float64).reshape(3, 3)
        self.config = config
        self.state = (
            FaAxisCalibrationState.CALIBRATION_REQUIRED
            if config.enable_vr_axis_calibration
            else FaAxisCalibrationState.DISABLED
        )
        self.failure_reason: str | None = None
        self.result: FaAxisCalibrationResult | None = None
        self._origin_samples: list[BimanualWristSample] = []
        self._forward_samples: list[BimanualWristSample] = []
        self._left_samples: list[BimanualWristSample] = []
        self._active_window: list[BimanualWristSample] = []
        self._origin_mean: np.ndarray | None = None
        self._last_sample: BimanualWristSample | None = None
        self._jump_confirm_count = 0
        self._return_stable_since_s: float | None = None
        self._last_reported_state: FaAxisCalibrationState | None = None

    @property
    def ready(self) -> bool:
        return self.state == FaAxisCalibrationState.READY and self.result is not None

    def consume_state_change(self) -> FaAxisCalibrationState | None:
        """每个新状态只上报一次，供外部日志和语音提示去重。"""
        if self._last_reported_state == self.state:
            return None
        self._last_reported_state = self.state
        return self.state

    def require_recalibration(self, reason: str | None = None) -> None:
        """清空旧样本和结果，从标定入口重新开始。"""
        if not self.config.enable_vr_axis_calibration:
            self.state = FaAxisCalibrationState.DISABLED
            return
        self.failure_reason = reason
        self.result = None
        self._origin_samples = []
        self._forward_samples = []
        self._left_samples = []
        self._active_window = []
        self._origin_mean = None
        self._last_sample = None
        self._jump_confirm_count = 0
        self._return_stable_since_s = None
        self.state = FaAxisCalibrationState.CALIBRATION_REQUIRED

    def update(self, sample: BimanualWristSample) -> None:
        """消费一帧双手样本，并按当前状态推进标定流程。"""

        if self.state == FaAxisCalibrationState.DISABLED:
            return
        if self._last_sample is not None and sample.timestamp_s <= self._last_sample.timestamp_s:
            return
        if not self._valid_sample(sample):
            self._fail("TRACKING_INVALID")
            return
        if self.state == FaAxisCalibrationState.READY:
            self._check_tracking_origin_jump(sample)
            self._last_sample = sample
            return
        if self.state in (FaAxisCalibrationState.INVALIDATED, FaAxisCalibrationState.FAILED):
            return
        if self.state == FaAxisCalibrationState.CALIBRATION_REQUIRED:
            self.state = FaAxisCalibrationState.WAITING_STABLE_ORIGIN
            self._active_window = []

        if self.state in (FaAxisCalibrationState.WAITING_STABLE_ORIGIN, FaAxisCalibrationState.CAPTURING_ORIGIN):
            self.state = FaAxisCalibrationState.CAPTURING_ORIGIN
            self._capture_origin(sample)
        elif self.state == FaAxisCalibrationState.CAPTURING_FORWARD:
            self._capture_direction_window(sample, "forward")
        elif self.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_FORWARD_RETRY:
            self._wait_return_before_retry(sample, "forward")
        elif self.state == FaAxisCalibrationState.WAITING_RETURN_AFTER_FORWARD:
            self._wait_return_after_forward(sample)
        elif self.state == FaAxisCalibrationState.CAPTURING_LEFT:
            self._capture_direction_window(sample, "left")
        elif self.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_LEFT_RETRY:
            self._wait_return_before_retry(sample, "left")
        elif self.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY:
            self._wait_return_before_ready(sample)

        self._last_sample = sample

    def _valid_sample(self, sample: BimanualWristSample) -> bool:
        if not np.all(np.isfinite(sample.left)) or not np.all(np.isfinite(sample.right)):
            return False
        return True

    def _capture_origin(self, sample: BimanualWristSample) -> None:
        """持续采样起点；窗口内波动过大时从当前帧重新计时。"""
        self._active_window.append(sample)
        if _window_duration(self._active_window) < self.config.calibration_sample_duration_s:
            return

        centers = np.asarray([item.center for item in self._active_window], dtype=np.float64)
        center_mean = np.mean(centers, axis=0)
        max_deviation = float(np.max(np.linalg.norm(centers - center_mean, axis=1)))
        if max_deviation > self.config.calibration_stable_position_epsilon_m:
            self._active_window = [sample]
            self.state = FaAxisCalibrationState.WAITING_STABLE_ORIGIN
            return

        self._origin_samples = list(self._active_window)
        self._origin_mean = center_mean
        self._active_window = []
        self.state = FaAxisCalibrationState.CAPTURING_FORWARD

    def _capture_direction_window(self, sample: BimanualWristSample, direction: str) -> None:
        """采集一个方向动作末端的稳定窗口。"""

        if self._origin_mean is None:
            self._fail("TRACKING_INVALID")
            return
        displacement = float(np.linalg.norm(sample.center - self._origin_mean))
        min_motion_m = self.config.calibration_min_motion_distance_m
        # 小于 attempt_motion_m 视为用户尚未开始动作，不启动计时；已经明显离开起点但最终仍不足最小距离，才判定为一次“移动太短”的尝试。
        attempt_motion_m = max(self.config.calibration_stable_position_epsilon_m * 1.5, min_motion_m * 0.4)
        if displacement < min_motion_m:
            if displacement < attempt_motion_m:
                self._active_window = []
                return
            self._active_window.append(sample)
            if _window_duration(self._active_window) >= self.config.calibration_sample_duration_s:
                self._start_direction_retry(direction, f"{direction.upper()}_MOTION_TOO_SHORT")
            return
        if self._active_window:
            first_displacement = float(np.linalg.norm(self._active_window[0].center - self._origin_mean))
            if first_displacement < min_motion_m:
                self._active_window = []
        self._active_window.append(sample)
        if _window_duration(self._active_window) < self.config.calibration_sample_duration_s:
            return

        direction_samples = list(self._active_window)
        center_delta = _mean_center(direction_samples) - self._origin_mean
        motion_distance = float(np.linalg.norm(center_delta))
        if motion_distance < self.config.calibration_min_motion_distance_m:
            self._start_direction_retry(direction, f"{direction.upper()}_MOTION_TOO_SHORT")
            return
        if not _sampled_hand_motion_consistent(
            self._origin_samples,
            direction_samples,
            center_delta,
            self.config,
        ):
            self._start_direction_retry(direction, f"{direction.upper()}_HANDS_MISMATCH")
            return

        if direction == "forward":
            self._forward_samples = direction_samples
            self._active_window = []
            self._return_stable_since_s = None
            self.state = FaAxisCalibrationState.WAITING_RETURN_AFTER_FORWARD
        else:
            self._left_samples = direction_samples
            self._active_window = []
            self.state = FaAxisCalibrationState.VALIDATING
            self._validate()

    def _start_direction_retry(self, direction: str, reason: str) -> None:
        """保留起点，丢弃失败方向的数据，并要求用户先回到起点。"""
        self.failure_reason = reason
        self._active_window = []
        self._return_stable_since_s = None
        if direction == "forward":
            self._forward_samples = []
            self._left_samples = []
            self.state = FaAxisCalibrationState.WAITING_RETURN_BEFORE_FORWARD_RETRY
        else:
            self._left_samples = []
            self.state = FaAxisCalibrationState.WAITING_RETURN_BEFORE_LEFT_RETRY

    def _wait_return_before_retry(self, sample: BimanualWristSample, direction: str) -> None:
        """确认双手在起点附近稳定停留后，重新提示对应方向动作。"""
        if self._origin_mean is None:
            self._fail("TRACKING_INVALID")
            return
        displacement = float(np.linalg.norm(sample.center - self._origin_mean))
        if displacement > self.config.calibration_stable_position_epsilon_m:
            self._return_stable_since_s = None
            return
        if self._return_stable_since_s is None:
            self._return_stable_since_s = sample.timestamp_s
            return
        if sample.timestamp_s - self._return_stable_since_s < self.config.calibration_stable_dwell_s:
            return
        self.failure_reason = None
        self._active_window = []
        self._return_stable_since_s = None
        if direction == "forward":
            self.state = FaAxisCalibrationState.CAPTURING_FORWARD
        else:
            self.state = FaAxisCalibrationState.CAPTURING_LEFT

    def _wait_return_after_forward(self, sample: BimanualWristSample) -> None:
        """前向采集成功后，等待回到起点再开放左向采集。"""
        if self._origin_mean is None:
            self._fail("TRACKING_INVALID")
            return
        displacement = float(np.linalg.norm(sample.center - self._origin_mean))
        if displacement > self.config.calibration_stable_position_epsilon_m:
            self._return_stable_since_s = None
            return
        if self._return_stable_since_s is None:
            self._return_stable_since_s = sample.timestamp_s
            return
        if sample.timestamp_s - self._return_stable_since_s >= self.config.calibration_stable_dwell_s:
            self.state = FaAxisCalibrationState.CAPTURING_LEFT
            self._active_window = []

    def _wait_return_before_ready(self, sample: BimanualWristSample) -> None:
        """等待双手回到较宽松的起点范围，再允许建立遥操作手部基准。"""

        if self._origin_mean is None or self.result is None:
            self._fail("TRACKING_INVALID")
            return

        origin_left = _mean_position(self._origin_samples, "left")
        origin_right = _mean_position(self._origin_samples, "right")
        left_displacement = float(np.linalg.norm(sample.left - origin_left))
        right_displacement = float(np.linalg.norm(sample.right - origin_right))
        if (
            max(left_displacement, right_displacement)
            > self.config.calibration_ready_return_position_epsilon_m
        ):
            self._return_stable_since_s = None
            return
        if self._return_stable_since_s is None:
            self._return_stable_since_s = sample.timestamp_s
            return
        if sample.timestamp_s - self._return_stable_since_s >= self.config.calibration_ready_return_dwell_s:
            self._return_stable_since_s = None
            self.state = FaAxisCalibrationState.READY

    def _validate(self) -> None:
        """解算并校验矩阵；可恢复问题进入对应方向的重试流程。"""
        result, failure_reason = compute_fa_axis_calibration(
            r_vr_to_robot_nominal=self.r_vr_to_robot_nominal,
            origin_samples=self._origin_samples,
            forward_samples=self._forward_samples,
            left_samples=self._left_samples,
            config=self.config,
        )
        if result is None:
            if failure_reason in ("FORWARD_MOTION_TOO_SHORT", "FORWARD_HANDS_MISMATCH"):
                self._start_direction_retry("forward", failure_reason)
                return
            if failure_reason in ("LEFT_MOTION_TOO_SHORT", "LEFT_HANDS_MISMATCH"):
                self._start_direction_retry("left", failure_reason)
                return
            # 此时 forward 已单独采集完成。共线问题只需要重新采左向动作，
            # 无需让用户从头再做一遍前向标定。
            if failure_reason == "AXES_NEARLY_COLLINEAR":
                self._start_direction_retry("left", failure_reason)
                return
            self._fail(failure_reason or "SVD_PROJECTION_FAILED")
            return
        self.result = result
        self.failure_reason = None
        self._return_stable_since_s = None
        self.state = FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY

    def _check_tracking_origin_jump(self, sample: BimanualWristSample) -> None:
        """检测左右手同步瞬移，以识别 PICO 追踪坐标原点重定位。"""

        if not self.config.tracking_origin_jump_detection_enabled or self._last_sample is None:
            return
        delta_left = sample.left - self._last_sample.left
        delta_right = sample.right - self._last_sample.right
        common_delta = 0.5 * (delta_left + delta_right)
        common_distance = float(np.linalg.norm(common_delta))
        interhand_change = abs(sample.interhand_distance - self._last_sample.interhand_distance)
        same_direction = _direction_error_deg(delta_left, delta_right) <= self.config.calibration_max_left_right_direction_error_deg
        # 真正的手部动作通常会改变双手间距或两手方向；追踪原点跳变则表现为
        # 两手几乎同向、同距离地整体瞬移。连续多帧确认可过滤单帧噪声。
        if (
            common_distance >= self.config.tracking_origin_jump_translation_m
            and interhand_change <= self.config.tracking_origin_jump_interhand_change_m
            and same_direction
        ):
            self._jump_confirm_count += 1
        else:
            self._jump_confirm_count = 0

        if self._jump_confirm_count >= self.config.tracking_origin_jump_confirm_frames:
            self.result = None
            self.failure_reason = "TRACKING_ORIGIN_JUMP"
            self.state = FaAxisCalibrationState.INVALIDATED

    def _fail(self, reason: str) -> None:
        """进入不可自动恢复的失败状态，并丢弃可能不可信的结果。"""
        self.result = None
        self.failure_reason = reason
        self.state = FaAxisCalibrationState.FAILED
