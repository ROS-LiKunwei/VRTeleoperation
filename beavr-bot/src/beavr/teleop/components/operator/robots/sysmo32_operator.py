"""
SYSMO-32双臂机器人Operator模块

继承XArmOperator,使用SYSMO-32特定的坐标系变换矩阵。
SYSMO-32是6自由度双臂机器人,与XArm7(7自由度单臂)有以下区别：
    - 每臂6个关节(XArm7是7个)
    - 双臂固定在同一个base_link上
    - 坐标系变换矩阵不同(H_R_V)

核心遥操作逻辑(坐标变换、滤波、发布)与XArmOperator完全一致,只是变换矩阵参数不同,因此直接继承XArmOperator。

坐标系变换矩阵说明：
    H_R_V: 机器人基座坐标系 → VR坐标系的变换矩阵
        描述了SYSMO-32机器人基座相对于VR追踪空间的位置和朝向。

        PICO Unity Integration SDK 原始坐标是 Unity 左手系:
            X=右, Y=上, Z=前。
        keypoint_transform.py 在入口处将 Z 取反,统一成内部右手系:
            X=右, Y=上, Z=后。
        SYSMO-32机器人基座(ROS约定,机器人第一视角):
            X=前, Y=左, Z=上。
        如果用户站在机器人正前方,则 Robot +X 指向用户。

        当前遥操作映射按 PICO/Unity 原始手势方向定义:
            PICO +X(右) -> Robot -Y(右)
            PICO +Y(上) -> Robot +Z(上)
            PICO +Z(前) -> Robot +X(前)

        因为 keypoint_transform.py 会先将 PICO/Unity +Z 前转换为内部 VR -Z,
        所以内右手系 VR 到机器人应满足:
            内部 VR +X -> Robot -Y
            内部 VR +Y -> Robot +Z
            内部 VR +Z -> Robot -X

        因此 H_R_V(robotBase -> VR)的旋转部分:
            VR_X = -Robot_Y -> [0, -1, 0]
            VR_Y = Robot_Z  -> [0, 0, 1]
            VR_Z = -Robot_X -> [-1, 0, 0]

        注意: det(R)=+1, 为正常旋转(无反射), 保证共轭变换结果有效

数据流位置：
    keypoint_transform.py → [本模块: Sysmo32Operator] → sysmo32_robot.py / mujoco_sim.py

端口映射：
    右臂Operator发布到10011 (XARM_ENDEFF_SUBSCRIBE_PORT + 2)
    左臂Operator发布到10013 (XARM_ENDEFF_SUBSCRIBE_PORT + 4)
"""

import numpy as np

from beavr.teleop.components.operator.robots.xarm7_operator import XArmOperator

# SYSMO-32坐标系变换矩阵
# H_R_V: 机器人基座坐标系 → VR坐标系的变换矩阵
H_R_V_SYSMO32 = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
])



class Sysmo32Operator(XArmOperator):
    """
    SYSMO-32双臂机器人Operator。

    继承XArmOperator的所有遥操作逻辑,仅替换坐标系变换矩阵。

    与XArmOperator的区别：
        - 使用SYSMO-32特定的H_R_V变换矩阵
        - operator_name前缀为"sysmo32_"而非"xarm7_"

    遥操作流程（继承自XArmOperator）：
        1. 订阅keypoint_transform.py发布的手部方向帧
        2. 将4x3帧转换为4x4齐次变换矩阵
        3. 计算手部相对运动（当前帧相对于初始帧）
        4. 通过H_R_V变换矩阵和共轭变换映射到机器人坐标系
        5. 计算目标机器人位姿
        6. 互补滤波器平滑
        7. 输出CartesianTarget（位置+四元数姿态）
        8. 发布给下游的Sysmo32Robot或MuJoCo仿真器

    发布Topic：
        - 'endeff_coords': 笛卡尔空间目标命令
        - 'reset': 重置命令
    """

    def __init__(
        self,
        operator_name: str = "sysmo32_right_operator",
        host: str = "127.0.0.1",
        transformed_keypoints_port: int = 8092,
        stream_configs: dict = None,
        stream_oculus: bool = True,
        endeff_publish_port: int = 10011,
        endeff_subscribe_port: int = 10012,
        moving_average_limit: int = 3,
        use_filter: bool = False,
        arm_resolution_port: int = None,
        teleoperation_state_port: int = None,
        logging_config: dict = None,
        hand_side: str = "right",
        hand_frame_timeout_s: float = 0.3,
        rotation_delta_frame: str = "base",
        **kwargs,
    ):
        h_r_v = H_R_V_SYSMO32

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
            h_r_v=h_r_v,
            use_filter=use_filter,
            arm_resolution_port=arm_resolution_port,
            teleoperation_state_port=teleoperation_state_port,
            logging_config=logging_config,
            hand_side=hand_side,
            hand_frame_timeout_s=hand_frame_timeout_s,
            rotation_delta_frame=rotation_delta_frame,
        )
