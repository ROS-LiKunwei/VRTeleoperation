"""FA 机器人遥操作适配器。

该类复用 ``XArmOperator`` 已有的手腕位姿重定向链路，仅增加 FA 的坐标轴
标定、双手标定帧输入、PICO 语音提示以及平移倍率切换。标定未完成或失效
时，会在父类执行重置、坐标变换和指令发布之前阻断当前控制周期。
"""

from __future__ import annotations

import logging
import time

import numpy as np

from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.string_publisher import ZMQStringPublisherManager
from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.components.operator.robots.fa_axis_calibration import (
    BimanualWristSample,
    FaAxisCalibrationConfig,
    FaAxisCalibrationSession,
    FaAxisCalibrationState,
)
from beavr.teleop.components.operator.robots.xarm7_operator import XArmOperator
from beavr.teleop.configs.constants import ports, robots

logger = logging.getLogger(__name__)

# FA 的 IK 以 pelvis 为机器人基坐标系，其轴方向与 SYSMO-32 基坐标系一致。
# H_R_V_FA 遵循父类 h_r_v 的定义；实际把 VR 向量转换到机器人坐标系时，
# 使用其旋转部分的逆矩阵，见 ``_fa_r_vr_to_robot_nominal``。
H_R_V_FA = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


class FaOperator(XArmOperator):
    """在现有 XArm 重定向链路上增加 FA 专用行为。"""

    def __init__(
        self,
        operator_name: str = "fa_right_operator",
        host: str = "127.0.0.1",
        transformed_keypoints_port: int = 8092,
        stream_configs: dict | None = None,
        stream_oculus: bool = True,
        endeff_publish_port: int = 10011,
        endeff_subscribe_port: int = 10012,
        moving_average_limit: int = 3,
        use_filter: bool = False,
        arm_resolution_port: int | None = None,
        teleoperation_state_port: int | None = None,
        logging_config: dict | None = None,
        hand_side: str = "right",
        hand_frame_timeout_s: float = 0.5,
        rotation_delta_frame: str = "base",
        translation_scale: float = 1.3,
        high_resolution_translation_scale: float = 1.3,
        low_resolution_translation_scale: float = 1.3,
        post_resume_stable_position_epsilon_m: float = 0.008,
        post_resume_stable_orientation_epsilon_rad: float = 0.08,
        post_resume_stable_dwell_s: float = 1.0,
        h_r_v: np.ndarray | None = None,
        right_transformed_keypoints_port: int = ports.KEYPOINT_TRANSFORM_PORT,
        left_transformed_keypoints_port: int = ports.LEFT_KEYPOINT_TRANSFORM_PORT,
        enable_vr_axis_calibration: bool = True,
        require_calibration_each_enable: bool = True,
        calibration_sample_duration_s: float = 0.4,
        calibration_stable_dwell_s: float = 0.3,
        calibration_stable_position_epsilon_m: float = 0.02,
        calibration_ready_return_position_epsilon_m: float = 0.04,
        calibration_ready_return_dwell_s: float = 0.20,
        calibration_min_motion_distance_m: float = 0.06,
        calibration_max_motion_distance_m: float = 0.60,
        calibration_max_axis_abs_dot: float = 0.35,
        calibration_max_left_right_direction_error_deg: float = 20.0,
        calibration_max_left_right_distance_ratio_error: float = 0.35,
        calibration_rotation_orthogonality_tolerance: float = 1e-4,
        calibration_rotation_determinant_tolerance: float = 1e-4,
        calibration_max_timestamp_skew_s: float = 0.15,
        calibration_max_frame_age_s: float = 1.0,
        tracking_origin_jump_detection_enabled: bool = True,
        tracking_origin_jump_translation_m: float = 0.15,
        tracking_origin_jump_rotation_deg: float = 15.0,
        tracking_origin_jump_confirm_frames: int = 2,
        tracking_origin_jump_interhand_change_m: float = 0.04,
        calibration_audio_enabled: bool = False,
        calibration_prompt_port: int = ports.FA_CALIBRATION_PROMPT_PORT,
        calibration_prompt_topic: str = "fa_calibration_prompt",
        calibration_audio_cooldown_s: float = 2.5,
        **kwargs,
    ):
        # 保持与父类相同的默认网络和日志配置，避免 FA 形成平行控制链路。
        if stream_configs is None:
            stream_configs = {"host": host, "port": 8086}
        if logging_config is None:
            logging_config = {"enabled": False}

        super().__init__(
            operator_name=operator_name,
            host=host,
            transformed_keypoints_port=transformed_keypoints_port,
            stream_configs=stream_configs,
            stream_oculus=stream_oculus,
            endeff_publish_port=endeff_publish_port,
            endeff_subscribe_port=endeff_subscribe_port,
            moving_average_limit=moving_average_limit,
            h_r_v=H_R_V_FA if h_r_v is None else np.asarray(h_r_v, dtype=np.float64),
            use_filter=use_filter,
            arm_resolution_port=arm_resolution_port,
            teleoperation_state_port=teleoperation_state_port,
            logging_config=logging_config,
            hand_side=hand_side,
            hand_frame_timeout_s=hand_frame_timeout_s,
            rotation_delta_frame=rotation_delta_frame,
            post_resume_stable_position_epsilon_m=post_resume_stable_position_epsilon_m,
            post_resume_stable_orientation_epsilon_rad=post_resume_stable_orientation_epsilon_rad,
            post_resume_stable_dwell_s=post_resume_stable_dwell_s,
        )
        # 父类使用 resolution_scale 缩放 VR 位移；FA 可在运行时切换高、低
        # 分辨率倍率，但未收到切换消息时保持 translation_scale。
        self.resolution_scale = float(translation_scale)
        self._fa_high_resolution_translation_scale = float(high_resolution_translation_scale)
        self._fa_low_resolution_translation_scale = float(low_resolution_translation_scale)

        # nominal 是未标定时的固定轴映射。标定成功后只替换这个旋转映射，
        # 后续 IK、滤波和发布仍由父类原有流程执行。
        self._fa_r_vr_to_robot_nominal = np.linalg.inv(self.h_r_v[:3, :3])
        self._fa_axis_calibration_config = FaAxisCalibrationConfig(
            enable_vr_axis_calibration=enable_vr_axis_calibration,
            require_calibration_each_enable=require_calibration_each_enable,
            calibration_sample_duration_s=calibration_sample_duration_s,
            calibration_stable_dwell_s=calibration_stable_dwell_s,
            calibration_stable_position_epsilon_m=calibration_stable_position_epsilon_m,
            calibration_ready_return_position_epsilon_m=calibration_ready_return_position_epsilon_m,
            calibration_ready_return_dwell_s=calibration_ready_return_dwell_s,
            calibration_min_motion_distance_m=calibration_min_motion_distance_m,
            calibration_max_motion_distance_m=calibration_max_motion_distance_m,
            calibration_max_axis_abs_dot=calibration_max_axis_abs_dot,
            calibration_max_left_right_direction_error_deg=calibration_max_left_right_direction_error_deg,
            calibration_max_left_right_distance_ratio_error=calibration_max_left_right_distance_ratio_error,
            calibration_rotation_orthogonality_tolerance=calibration_rotation_orthogonality_tolerance,
            calibration_rotation_determinant_tolerance=calibration_rotation_determinant_tolerance,
            calibration_max_timestamp_skew_s=calibration_max_timestamp_skew_s,
            calibration_max_frame_age_s=calibration_max_frame_age_s,
            tracking_origin_jump_detection_enabled=tracking_origin_jump_detection_enabled,
            tracking_origin_jump_translation_m=tracking_origin_jump_translation_m,
            tracking_origin_jump_rotation_deg=tracking_origin_jump_rotation_deg,
            tracking_origin_jump_confirm_frames=tracking_origin_jump_confirm_frames,
            tracking_origin_jump_interhand_change_m=tracking_origin_jump_interhand_change_m,
        )
        self._fa_axis_calibration = FaAxisCalibrationSession(
            r_vr_to_robot_nominal=self._fa_r_vr_to_robot_nominal,
            config=self._fa_axis_calibration_config,
        )
        # 左右订阅者非阻塞读取，缓存各自最新的腕部位置和源时间戳；每个控制
        # 周期再从缓存构造一帧 BimanualWristSample。
        self._fa_latest_calibration_frames: dict[str, tuple[np.ndarray, float]] = {}
        self._fa_last_calibration_sample_unavailable_reason: str | None = None
        # 高频控制循环中的等待、失败和同步诊断分别限频，避免刷满运行日志。
        self._fa_last_calibration_wait_log_time = 0.0
        self._fa_last_calibration_failure_log_time = 0.0
        self._fa_last_calibration_sync_wait_log_time = 0.0
        self._fa_calibration_audio_enabled = bool(calibration_audio_enabled)
        self._fa_calibration_prompt_host = host
        self._fa_calibration_prompt_port = int(calibration_prompt_port)
        self._fa_calibration_prompt_topic = calibration_prompt_topic
        self._fa_calibration_audio_cooldown_s = float(calibration_audio_cooldown_s)
        # 同一提示键在 cooldown 内不重复发送；origin 额外限制为每轮标定只
        # 提示一次，防止状态抖动打断 PICO 正在播放的语音。
        self._fa_last_calibration_audio_key: str | None = None
        self._fa_last_calibration_audio_time = 0.0
        self._fa_calibration_origin_prompted = False
        self._fa_calibration_prompt_publisher = ZMQStringPublisherManager.get_instance()

        self._fa_calibration_right_subscriber = None
        self._fa_calibration_left_subscriber = None
        if enable_vr_axis_calibration:
            # 无论当前 operator 控制左臂还是右臂，坐标轴标定都需要同时观察
            # 双手的 transformed hand frame，确认用户执行的是整体平移动作。
            self._fa_calibration_right_subscriber = ZMQSubscriber(
                host=host,
                port=right_transformed_keypoints_port,
                topic=f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}",
                context=self._context,
                message_type=InputFrame,
            )
            self._fa_calibration_left_subscriber = ZMQSubscriber(
                host=host,
                port=left_transformed_keypoints_port,
                topic=f"{robots.LEFT}_{robots.TRANSFORMED_HAND_FRAME}",
                context=self._context,
                message_type=InputFrame,
            )
            self._subscribers["fa_calibration_right"] = self._fa_calibration_right_subscriber
            self._subscribers["fa_calibration_left"] = self._fa_calibration_left_subscriber

    def _get_resolution_scale_mode(self) -> float:
        """读取手臂分辨率模式，并返回当前生效的 FA 平移倍率。"""

        if not self._arm_resolution_subscriber:
            return self.resolution_scale

        data = self._arm_resolution_subscriber.recv_keypoints()
        if data is None:
            return self.resolution_scale

        # 分辨率消息不是遥操作的硬依赖；消息格式异常时继续沿用当前倍率。
        try:
            from beavr.teleop.configs.constants import robots

            scale_mode = data.value
            if scale_mode == robots.ARM_HIGH_RESOLUTION:
                self.resolution_scale = self._fa_high_resolution_translation_scale
            elif scale_mode == robots.ARM_LOW_RESOLUTION:
                self.resolution_scale = self._fa_low_resolution_translation_scale
            return self.resolution_scale
        except Exception:
            return self.resolution_scale

    def _get_r_vr_to_robot(self) -> np.ndarray:
        """优先返回标定矩阵，否则返回 FA 名义坐标轴映射。"""

        if self._fa_axis_calibration.ready:
            return self._fa_axis_calibration.result.r_vr_to_robot
        return self._fa_r_vr_to_robot_nominal

    def _before_retargeting_cycle(
        self,
        new_arm_teleop_state: int,
        resume_edge: bool,
        fresh_resume_command: bool,
    ) -> bool:
        """在父类重置和发布前推进标定，并决定当前周期是否放行。

        返回 ``False`` 时，父类只更新遥操作开关状态，不读取手部控制帧、
        不建立遥操作基准，也不发布新的机器人目标。
        """

        if not self._fa_axis_calibration_config.enable_vr_axis_calibration:
            return True

        # 每次启用遥操作都重新标定。fresh resume 是带时间戳的命令事件，必须
        # 立即标记为已消费，否则标定期间每个周期都会再次回到 REQUIRED。
        if (resume_edge or fresh_resume_command) and self._fa_axis_calibration_config.require_calibration_each_enable:
            self._fa_axis_calibration.require_recalibration("RESUME_REQUIRES_CALIBRATION")
            self._fa_calibration_origin_prompted = False
            if fresh_resume_command:
                self._last_processed_resume_command_timestamp_s = self._last_teleop_command_timestamp_s

        # READY 后仍持续输入双手样本，但只用于检测 PICO 追踪原点整体跳变。
        # 未检测到失效时直接放行父类原有重定向流程。
        if self._fa_axis_calibration.ready:
            sample = self._get_fa_calibration_sample()
            if sample is not None:
                self._fa_axis_calibration.update(sample)
                if self._fa_axis_calibration.state == FaAxisCalibrationState.INVALIDATED:
                    self._handle_fa_calibration_invalidated()
                    return False
            return True

        # FAILED/INVALIDATED 不会自动继续控制，等待下一次 resume 显式重标定。
        if self._fa_axis_calibration.state in (
            FaAxisCalibrationState.FAILED,
            FaAxisCalibrationState.INVALIDATED,
        ):
            self._log_fa_calibration_failure()
            return False

        sample = self._get_fa_calibration_sample()
        if sample is None:
            reason = self._fa_last_calibration_sample_unavailable_reason or "missing transformed hand frame"
            # 只有真正缺帧才提示“把双手放入视野”；帧过旧属于数据链路问题，
            # 仅记录日志，避免给用户错误动作指导。
            audio_key = "fa_calib_wait_hands" if reason == "missing transformed hand frame" else None
            self._log_fa_calibration_wait(reason, audio_key=audio_key)
            return False

        was_ready = self._fa_axis_calibration.ready
        self._fa_axis_calibration.update(sample)
        self._log_fa_calibration_state_if_needed()

        if self._fa_axis_calibration.state == FaAxisCalibrationState.FAILED:
            self._log_fa_calibration_failure()
            return False

        if self._fa_axis_calibration.ready and not was_ready:
            # 双手已回到较宽松的起点范围。清空标定期间积累的控制帧，并在
            # 下一周期让父类使用新鲜手帧重新建立 hand_init，避免机器人跳变。
            self.is_first_frame = True
            self._ignore_hand_frames_before_s = time.time()
            self._clear_hand_tracking_cache()
            self._log_fa_calibration_success()
            return False

        return self._fa_axis_calibration.ready

    def _get_fa_calibration_sample(self) -> BimanualWristSample | None:
        """合并左右手最新腕部位置，构造状态机需要的双手样本。"""

        # recv_keypoints() 为非阻塞读取；没有新帧时保留此前缓存的最新帧。
        self._consume_fa_calibration_frame(robots.RIGHT, self._fa_calibration_right_subscriber)
        self._consume_fa_calibration_frame(robots.LEFT, self._fa_calibration_left_subscriber)

        right = self._fa_latest_calibration_frames.get(robots.RIGHT)
        left = self._fa_latest_calibration_frames.get(robots.LEFT)
        if right is None or left is None:
            self._fa_last_calibration_sample_unavailable_reason = "missing transformed hand frame"
            return None

        # 缓存存在不代表仍可用。任何一侧超过最大帧龄都不能参与标定。
        now_s = time.time()
        max_frame_age_s = self._fa_axis_calibration_config.calibration_max_frame_age_s
        right_age_s = now_s - right[1]
        left_age_s = now_s - left[1]
        if right_age_s > max_frame_age_s or left_age_s > max_frame_age_s:
            self._fa_last_calibration_sample_unavailable_reason = "stale transformed hand frame"
            self._log_fa_calibration_stale_frame_wait(right_age_s, left_age_s)
            return None

        # 左右源时间戳偏差只记录诊断，不阻断样本。标定窗口会对连续样本求均值，
        # 空间一致性检查再负责拒绝由严重不同步造成的错误双手动作。
        timestamp_skew_s = abs(left[1] - right[1])
        if timestamp_skew_s > self._fa_axis_calibration_config.calibration_max_timestamp_skew_s:
            self._log_fa_calibration_sync_wait(timestamp_skew_s)

        self._fa_last_calibration_sample_unavailable_reason = None
        # 使用较新的时间戳驱动状态机，单侧原始时间戳同时保留用于诊断。
        return BimanualWristSample(
            timestamp_s=max(left[1], right[1]),
            left=left[0],
            right=right[0],
            left_timestamp_s=left[1],
            right_timestamp_s=right[1],
        )

    def _consume_fa_calibration_frame(self, hand_side: str, subscriber: ZMQSubscriber | None) -> None:
        """读取并校验一侧 hand frame，将腕部位置写入最新帧缓存。"""

        if subscriber is None:
            return
        data = subscriber.recv_keypoints()
        if data is None or data.frame_vectors is None:
            return
        try:
            frame = np.asarray(data.frame_vectors, dtype=np.float64).reshape(4, 3)
        except Exception:
            return
        # 复用父类清洗逻辑过滤 NaN、Inf 和非法形状。frame[0] 是腕部原点，
        # 标定只需要平移轨迹，不使用其余三个方向向量。
        frame = self._sanitize_hand_frame(frame)
        if frame is None:
            return
        # 兼容未携带源时间戳的旧消息，退化为本机接收时间。
        timestamp_s = float(getattr(data, "timestamp_s", 0.0) or time.time())
        self._fa_latest_calibration_frames[hand_side] = (frame[0].copy(), timestamp_s)

    def _log_fa_calibration_sync_wait(self, timestamp_skew_s: float) -> None:
        """限频记录左右手时间戳偏差，不阻塞当前标定样本。"""

        now_s = time.time()
        if now_s - self._fa_last_calibration_sync_wait_log_time < 1.0:
            return
        self._fa_last_calibration_sync_wait_log_time = now_s
        logger.info(
            "[Diag][FA_AXIS_CALIBRATION] operator=%s state=%s reason=waiting for synchronized frames "
            "timestamp_skew_s=%.3f max_timestamp_skew_s=%.3f",
            self.operator_name,
            self._fa_axis_calibration.state.value,
            timestamp_skew_s,
            self._fa_axis_calibration_config.calibration_max_timestamp_skew_s,
        )

    def _log_fa_calibration_stale_frame_wait(self, right_age_s: float, left_age_s: float) -> None:
        """限频记录左右手帧龄超限。"""

        now_s = time.time()
        if now_s - self._fa_last_calibration_sync_wait_log_time < 1.0:
            return
        self._fa_last_calibration_sync_wait_log_time = now_s
        logger.info(
            "[Diag][FA_AXIS_CALIBRATION] operator=%s state=%s reason=waiting for fresh transformed frames "
            "right_age_s=%.3f left_age_s=%.3f max_frame_age_s=%.3f",
            self.operator_name,
            self._fa_axis_calibration.state.value,
            right_age_s,
            left_age_s,
            self._fa_axis_calibration_config.calibration_max_frame_age_s,
        )

    def _log_fa_calibration_wait(self, reason: str, audio_key: str | None = "fa_calib_wait_hands") -> None:
        """限频记录标定等待原因，并按需发送等待语音。"""

        now_s = time.time()
        if now_s - self._fa_last_calibration_wait_log_time < 1.0:
            return
        self._fa_last_calibration_wait_log_time = now_s
        logger.info(
            "[Diag][FA_AXIS_CALIBRATION] operator=%s state=%s reason=%s",
            self.operator_name,
            self._fa_axis_calibration.state.value,
            reason,
        )
        if audio_key is not None:
            self._play_fa_calibration_audio(audio_key)

    def _log_fa_calibration_state_if_needed(self) -> None:
        """每个新状态只记录和播报一次。"""

        state = self._fa_axis_calibration.consume_state_change()
        if state is None:
            return
        logger.info(
            "[Diag][FA_AXIS_CALIBRATION] operator=%s state=%s instruction=%s",
            self.operator_name,
            state.value,
            self._fa_calibration_instruction(state),
        )
        audio_key = self._fa_calibration_audio_key_for_state(state)
        if audio_key is not None:
            self._play_fa_calibration_audio(audio_key)

    def _log_fa_calibration_success(self) -> None:
        """记录完整标定质量指标，并发送成功提示。"""

        result = self._fa_axis_calibration.result
        logger.info(
            "[Diag][FA_AXIS_CALIBRATION] operator=%s state=READY "
            "forward_vr=%s left_vr=%s up_vr=%s det=%.6f orth_err=%.6g dot=%.4f "
            "r_vr_to_robot=%s",
            self.operator_name,
            result.forward_vector_vr.tolist(),
            result.left_vector_vr.tolist(),
            result.up_vector_vr.tolist(),
            result.rotation_determinant,
            result.rotation_orthogonality_error,
            result.forward_left_dot,
            result.r_vr_to_robot.tolist(),
        )
        self._play_fa_calibration_audio("fa_calib_success")

    def _log_fa_calibration_failure(self) -> None:
        """限频记录不可继续的失败状态，并播放对应原因提示。"""

        now_s = time.time()
        if now_s - self._fa_last_calibration_failure_log_time < 1.0:
            return
        self._fa_last_calibration_failure_log_time = now_s
        logger.warning(
            "[Diag][FA_AXIS_CALIBRATION] operator=%s state=%s failure_reason=%s",
            self.operator_name,
            self._fa_axis_calibration.state.value,
            self._fa_axis_calibration.failure_reason,
        )
        self._play_fa_calibration_audio(
            self._fa_calibration_audio_key_for_failure(self._fa_axis_calibration.failure_reason)
        )

    def _handle_fa_calibration_invalidated(self) -> None:
        """标定失效后清除遥操作基准和旧手帧，立即停止继续映射。"""

        self.is_first_frame = True
        self._ignore_hand_frames_before_s = time.time()
        self._clear_hand_tracking_cache()
        self._log_fa_calibration_failure()

    def _fa_calibration_instruction(self, state: FaAxisCalibrationState) -> str:
        """将状态转换成便于日志排查的中文动作说明。"""

        if state in (FaAxisCalibrationState.WAITING_STABLE_ORIGIN, FaAxisCalibrationState.CAPTURING_ORIGIN):
            return "保持双手不动，采集标定原点"
        if state == FaAxisCalibrationState.CAPTURING_FORWARD:
            return "双手平行向前移动并保持"
        if state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_FORWARD_RETRY:
            if self._fa_axis_calibration.failure_reason == "FORWARD_HANDS_MISMATCH":
                return "前向动作中左右手不一致，双手先回到原点附近并保持，然后重新向前移动"
            return "移动距离太短，双手先回到原点附近并保持，然后重新向前移动"
        if state == FaAxisCalibrationState.WAITING_RETURN_AFTER_FORWARD:
            return "双手回到原点附近并保持"
        if state == FaAxisCalibrationState.CAPTURING_LEFT:
            return "双手平行向左移动并保持"
        if state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_LEFT_RETRY:
            if self._fa_axis_calibration.failure_reason == "LEFT_HANDS_MISMATCH":
                return "左向动作中左右手不一致，双手先回到原点附近并保持，然后重新向左移动"
            return "移动距离太短，双手先回到原点附近并保持，然后重新向左移动"
        if state == FaAxisCalibrationState.VALIDATING:
            return "验证标定矩阵"
        if state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY:
            return "标定已完成，双手回到起点附近并保持，准备开始遥操作"
        if state == FaAxisCalibrationState.READY:
            return "标定完成，重新建立遥操作基准"
        if state in (FaAxisCalibrationState.FAILED, FaAxisCalibrationState.INVALIDATED):
            return "标定失败或失效，暂停遥操作并等待重新标定"
        return "等待标定"

    def _fa_calibration_audio_key_for_state(self, state: FaAxisCalibrationState) -> str | None:
        """将正常流程状态映射到 Unity Resources 中的语音资源键。"""

        if state in (FaAxisCalibrationState.CALIBRATION_REQUIRED, FaAxisCalibrationState.WAITING_STABLE_ORIGIN):
            return "fa_calib_wait_hands"
        if state == FaAxisCalibrationState.CAPTURING_ORIGIN:
            return "fa_calib_origin"
        if state == FaAxisCalibrationState.CAPTURING_FORWARD:
            return "fa_calib_forward"
        if state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_FORWARD_RETRY:
            return "fa_calib_return_origin"
        if state == FaAxisCalibrationState.WAITING_RETURN_AFTER_FORWARD:
            return "fa_calib_return_origin"
        if state == FaAxisCalibrationState.CAPTURING_LEFT:
            return "fa_calib_left"
        if state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_LEFT_RETRY:
            return "fa_calib_return_origin"
        if state == FaAxisCalibrationState.VALIDATING:
            return "fa_calib_validating"
        if state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY:
            return "fa_calib_return_origin"
        if state == FaAxisCalibrationState.READY:
            return "fa_calib_success"
        return None

    def _fa_calibration_audio_key_for_failure(self, failure_reason: str | None) -> str:
        """将内部失败原因归并到尽量少的用户语音提示。"""

        reason = failure_reason or ""
        if reason == "TRACKING_INVALID":
            return "fa_calib_tracking_lost"
        if reason in ("FORWARD_MOTION_TOO_SHORT", "LEFT_MOTION_TOO_SHORT"):
            return "fa_calib_motion_too_short"
        if reason in ("FORWARD_MOTION_TOO_LARGE", "LEFT_MOTION_TOO_LARGE"):
            return "fa_calib_motion_too_large"
        if reason in ("FORWARD_HANDS_MISMATCH", "LEFT_HANDS_MISMATCH"):
            return "fa_calib_hands_mismatch"
        if reason in ("AXES_NEARLY_COLLINEAR", "SVD_PROJECTION_FAILED"):
            return "fa_calib_axes_bad"
        return "fa_calib_reset"

    def _play_fa_calibration_audio(self, audio_key: str) -> None:
        """向 PICO 发布语音资源键，并抑制短时间内的重复提示。"""

        if not self._fa_calibration_audio_enabled:
            return
        if audio_key == "fa_calib_origin" and self._fa_calibration_origin_prompted:
            return
        now_s = time.time()
        if (
            audio_key == self._fa_last_calibration_audio_key
            and now_s - self._fa_last_calibration_audio_time < self._fa_calibration_audio_cooldown_s
        ):
            return
        # 语音是辅助通道，发布失败只能记录告警，不能中断遥操作主循环。
        try:
            self._fa_calibration_prompt_publisher.publish(
                self._fa_calibration_prompt_host,
                self._fa_calibration_prompt_port,
                self._fa_calibration_prompt_topic,
                audio_key,
            )
            self._fa_last_calibration_audio_key = audio_key
            self._fa_last_calibration_audio_time = now_s
            logger.info(
                "[Diag][FA_AXIS_CALIBRATION_AUDIO] operator=%s prompt_key=%s port=%d",
                self.operator_name,
                audio_key,
                self._fa_calibration_prompt_port,
            )
            if audio_key == "fa_calib_origin":
                self._fa_calibration_origin_prompted = True
        except Exception as exc:
            logger.warning(
                "[Diag][FA_AXIS_CALIBRATION_AUDIO] operator=%s failed_prompt_key=%s error=%s",
                self.operator_name,
                audio_key,
                exc,
            )
