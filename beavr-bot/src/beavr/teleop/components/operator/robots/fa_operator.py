"""FA robot operator adapter."""

from __future__ import annotations

import numpy as np

from beavr.teleop.components.operator.robots.xarm7_operator import XArmOperator

# FA uses pelvis as the robot base frame for IK. Its coordinate directions match
# the SYSMO-32 base frame, so the same VR-to-robot mapping is used.
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
    """FA operator using the existing XArm retargeting logic."""

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
        h_r_v: np.ndarray | None = None,
        **kwargs,
    ):
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
        )
