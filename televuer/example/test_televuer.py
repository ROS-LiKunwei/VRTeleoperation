import os, sys
this_file = os.path.abspath(__file__) # 获取当前文件的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '..')) # 获取项目根目录（当前文件的上上级目录）
# 如果项目根目录不在系统路径中，则将其插入到系统路径的最前面
# 这样可以确保优先导入项目中的模块，而不是系统中同名的模块
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
# 导入自定义的 TeleVuer 类，用于 XR 遥操作
from televuer import TeleVuer
import logging_mp
logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

def run_test_TeleVuer():
    """
        TeleVuer 类的测试函数
        测试 XR 遥操作功能，包括图像渲染和姿态数据获取
    """
    use_hand_track = True # 是否使用手部追踪（False 表示使用控制器）
    # 不使用 teleimager 图像客户端
    # 初始化 TeleVuer 实例
    tv = TeleVuer(use_hand_tracking=use_hand_track, 
                  binocular=True, 
                  img_shape=(480, 1280), 
                  display_fps=30.0,
                  display_mode="pass-through",      # "ego" - 自我视角（小窗口显示机器人视角） or "immersive" or "pass-through"
                  zmq=False,
                  webrtc=False, 
                  webrtc_url=None
                  )
    
    # 使用 teleimager 图像客户端
    # teleimager + televuer (requires teleimager server)
    # from teleimager.image_client import ImageClient
    # img_client = ImageClient(host="192.168.123.164")
    # camera_config = img_client.get_cam_config()
    # tv = TeleVuer(use_hand_tracking=use_hand_track, 
    #               binocular=camera_config['head_camera']['binocular'],
    #               img_shape=camera_config['head_camera']['image_shape'],
    #               display_fps=camera_config['head_camera']['fps'],
    #               display_mode="immersive",   # "ego" or "immersive" or "pass-through"
    #               zmq=camera_config['head_camera']['enable_zmq'],
    #               webrtc=camera_config['head_camera']['enable_webrtc'],
    #               webrtc_url=f"https://192.168.123.164:{camera_config['head_camera']['webrtc_port']}/offer"
    #               )

    try:
        input("Press Enter to start TeleVuer test...")
        running = True
        while running:
            # 注释部分：从图像客户端获取头部帧
            # img, _= img_client.get_head_frame()
            
            # 纯 TeleVuer 模式：生成虚拟图像（全黑图像）
            import numpy as np
            img = np.zeros((480, 1280, 3), dtype=np.uint8) # 创建全黑图像
            tv.render_to_xr(img) # 将图像渲染到 XR 设备

            start_time = time.time()
            logger_mp.info("=" * 80)
            logger_mp.info("Common Data (always available):")
            # 打印头部、左臂、右臂姿态矩阵（4x4 变换矩阵）
            logger_mp.info(f"head_pose shape: {tv.head_pose.shape}\n{tv.head_pose}\n")
            logger_mp.info(f"left_arm_pose shape: {tv.left_arm_pose.shape}\n{tv.left_arm_pose}\n")
            logger_mp.info(f"right_arm_pose shape: {tv.right_arm_pose.shape}\n{tv.right_arm_pose}\n")
            logger_mp.info("=" * 80)

            if use_hand_track:
                # 手部追踪模式的数据
                logger_mp.info("Hand Tracking Data:")
                # 左手 25 个关键点的 3D 位置
                logger_mp.info(f"left_hand_positions shape: {tv.left_hand_positions.shape}\n{tv.left_hand_positions}\n")
                logger_mp.info(f"right_hand_positions shape: {tv.right_hand_positions.shape}\n{tv.right_hand_positions}\n")
                # 左手 25 个关键点的方向（3x3 旋转矩阵）
                logger_mp.info(f"left_hand_orientations shape: {tv.left_hand_orientations.shape}\n{tv.left_hand_orientations}\n")
                logger_mp.info(f"right_hand_orientations shape: {tv.right_hand_orientations.shape}\n{tv.right_hand_orientations}\n")
                # 左手捏合手势状态和强度
                logger_mp.info(f"left_hand_pinch: {tv.left_hand_pinch}")
                logger_mp.info(f"left_hand_pinchValue: {tv.left_hand_pinchValue}")
                # 左手抓握手势状态和强度
                logger_mp.info(f"left_hand_squeeze: {tv.left_hand_squeeze}")
                logger_mp.info(f"left_hand_squeezeValue: {tv.left_hand_squeezeValue}")
                logger_mp.info(f"right_hand_pinch: {tv.right_hand_pinch}")
                logger_mp.info(f"right_hand_pinchValue: {tv.right_hand_pinchValue}")
                logger_mp.info(f"right_hand_squeeze: {tv.right_hand_squeeze}")
                logger_mp.info(f"right_hand_squeezeValue: {tv.right_hand_squeezeValue}")
            else:
                # 控制器模式的数据
                logger_mp.info("Controller Data:")
                # 左手控制器的扳机键状态和值
                logger_mp.info(f"left_ctrl_trigger: {tv.left_ctrl_trigger}")
                logger_mp.info(f"left_ctrl_triggerValue: {tv.left_ctrl_triggerValue}")
                # 左手控制器的抓握键状态和值
                logger_mp.info(f"left_ctrl_squeeze: {tv.left_ctrl_squeeze}")
                logger_mp.info(f"left_ctrl_squeezeValue: {tv.left_ctrl_squeezeValue}")
                # 左手控制器的摇杆状态和值
                logger_mp.info(f"left_ctrl_thumbstick: {tv.left_ctrl_thumbstick}")
                logger_mp.info(f"left_ctrl_thumbstickValue: {tv.left_ctrl_thumbstickValue}")
                # 左手控制器的 A 和 B 按钮
                logger_mp.info(f"left_ctrl_aButton: {tv.left_ctrl_aButton}")
                logger_mp.info(f"left_ctrl_bButton: {tv.left_ctrl_bButton}")
                logger_mp.info(f"right_ctrl_trigger: {tv.right_ctrl_trigger}")
                logger_mp.info(f"right_ctrl_triggerValue: {tv.right_ctrl_triggerValue}")
                logger_mp.info(f"right_ctrl_squeeze: {tv.right_ctrl_squeeze}")
                logger_mp.info(f"right_ctrl_squeezeValue: {tv.right_ctrl_squeezeValue}")
                logger_mp.info(f"right_ctrl_thumbstick: {tv.right_ctrl_thumbstick}")
                logger_mp.info(f"right_ctrl_thumbstickValue: {tv.right_ctrl_thumbstickValue}")
                logger_mp.info(f"right_ctrl_aButton: {tv.right_ctrl_aButton}")
                logger_mp.info(f"right_ctrl_bButton: {tv.right_ctrl_bButton}")
            logger_mp.info("=" * 80)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, 0.016 - time_elapsed) 
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")
    except KeyboardInterrupt:
        running = False
        logger_mp.warning("KeyboardInterrupt, exiting program...")
    finally:
        tv.close()
        logger_mp.warning("Finally, exiting program...")
        exit(0)

if __name__ == '__main__':
    run_test_TeleVuer()