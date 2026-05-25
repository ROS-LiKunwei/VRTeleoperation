"""PICO4 VR手部追踪探测器测试脚本"""

import sys
import os

# 添加源码路径到Python路径
src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_root)

from beavr.teleop.components.detector.vr.pico4 import PICO4VRHandDetector
from beavr.teleop.configs.constants import network, robots



def test_pico4_detector():
    """测试PICO4 VR手部追踪探测器配置"""
    print("开始测试PICO4 VR手部追踪探测器配置...")
    
    # 测试右手单手配置
    print("\n测试右手单手配置...")
    try:
        hand_config = robots.RIGHT
        hand_ports = {}
        
        if hand_config in [robots.RIGHT, robots.BIMANUAL]:
            hand_ports[robots.RIGHT] = network.RIGHT_HAND_PICO4_PORT
        
        print("✓ 右手单手配置创建成功")
        print(f"  - 手部配置: {hand_config}")
        print(f"  - 手部端口: {hand_ports}")
    except Exception as e:
        print(f"✗ 右手单手配置创建失败: {e}")
        return False
    
    # 测试左手单手配置
    print("\n测试左手单手配置...")
    try:
        hand_config = robots.LEFT
        hand_ports = {}
        
        if hand_config in [robots.LEFT, robots.BIMANUAL]:
            hand_ports[robots.LEFT] = network.LEFT_HAND_PICO4_PORT
        
        print("✓ 左手单手配置创建成功")
        print(f"  - 手部配置: {hand_config}")
        print(f"  - 手部端口: {hand_ports}")
    except Exception as e:
        print(f"✗ 左手单手配置创建失败: {e}")
        return False
    
    # 测试双手配置
    print("\n测试双手配置...")
    try:
        hand_config = robots.BIMANUAL
        hand_ports = {}
        
        if hand_config in [robots.RIGHT, robots.BIMANUAL]:
            hand_ports[robots.RIGHT] = network.RIGHT_HAND_PICO4_PORT
        
        if hand_config in [robots.LEFT, robots.BIMANUAL]:
            hand_ports[robots.LEFT] = network.LEFT_HAND_PICO4_PORT
        
        print("✓ 双手配置创建成功")
        print(f"  - 手部配置: {hand_config}")
        print(f"  - 手部端口: {hand_ports}")
    except Exception as e:
        print(f"✗ 双手配置创建失败: {e}")
        return False
    
    print("\n所有测试通过！")
    return True


def _detector():
    return PICO4VRHandDetector(
        host="127.0.0.1",
        pico4_pub_port=8088,
        button_port=8095,
        teleop_reset_port=8100,
        hand_config=robots.RIGHT,
    )


def test_process_keypoints_accepts_unity_trailing_colon():
    detector = _detector()
    coords = "|".join(f"{i + 0.1},{i + 0.2},{i + 0.3}" for i in range(26))

    values, send_timestamp = detector._process_keypoints(f"absolute:{coords}:".encode())

    assert send_timestamp is None
    assert len(values) == 78
    assert values[-3:] == [25.1, 25.2, 25.3]


def test_is_relative_frame_with_timestamped_absolute_marker():
    detector = _detector()
    coords = "|".join("0.1,0.2,0.3" for _ in range(26))

    assert detector._is_relative_frame(f"19:31:32.123456:absolute:{coords}:".encode()) is False
    assert detector._is_relative_frame(f"19:31:32.123456:relative:{coords}:".encode()) is True


def test_invalid_all_zero_keypoints_are_rejected():
    detector = _detector()
    keypoints = [0.0] * (26 * 3)

    is_valid, reason = detector._is_valid_hand_keypoints(keypoints, robots.RIGHT)

    assert is_valid is False
    assert "重合" in reason or "占位帧" in reason


if __name__ == "__main__":
    success = test_pico4_detector()
    sys.exit(0 if success else 1)
