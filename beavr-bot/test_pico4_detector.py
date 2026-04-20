#!/usr/bin/env python3
"""
测试PICO4VRHandDetector的配置和基本功能。

此测试验证PICO4VRHandDetector的配置逻辑，包括端口分配、手部模式解析等，
而不实际绑定套接字，确保跨平台兼容性。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from beavr.teleop.configs.robots.shared_components import UnifiedPICO4VRHandDetectorCfg
from beavr.teleop.configs.constants import robots, network


def test_pico4_detector_config():
    """测试PICO4探测器配置的正确性。"""
    print("=== 测试PICO4VRHandDetector配置 ===")
    
    # 测试右手模式配置
    print("\n1. 测试右手模式配置:")
    right_config = UnifiedPICO4VRHandDetectorCfg(hand_config=robots.RIGHT)
    right_detector = right_config.build()
    assert robots.RIGHT in right_detector.hand_ports, "右手模式应包含右手端口"
    assert robots.LEFT not in right_detector.hand_ports, "右手模式不应包含左手端口"
    assert right_detector.hand_ports[robots.RIGHT] == network.RIGHT_HAND_PICO4_PORT, "右手端口应匹配PICO4配置"
    print("✓ 右手模式配置正确")
    
    # 测试左手模式配置
    print("\n2. 测试左手模式配置:")
    left_config = UnifiedPICO4VRHandDetectorCfg(hand_config=robots.LEFT)
    left_detector = left_config.build()
    assert robots.LEFT in left_detector.hand_ports, "左手模式应包含左手端口"
    assert robots.RIGHT not in left_detector.hand_ports, "左手模式不应包含右手端口"
    assert left_detector.hand_ports[robots.LEFT] == network.LEFT_HAND_PICO4_PORT, "左手端口应匹配PICO4配置"
    print("✓ 左手模式配置正确")
    
    # 测试双手模式配置
    print("\n3. 测试双手模式配置:")
    bimanual_config = UnifiedPICO4VRHandDetectorCfg(hand_config=robots.BIMANUAL)
    bimanual_detector = bimanual_config.build()
    assert robots.LEFT in bimanual_detector.hand_ports, "双手模式应包含左手端口"
    assert robots.RIGHT in bimanual_detector.hand_ports, "双手模式应包含右手端口"
    assert bimanual_detector.hand_ports[robots.LEFT] == network.LEFT_HAND_PICO4_PORT, "左手端口应匹配PICO4配置"
    assert bimanual_detector.hand_ports[robots.RIGHT] == network.RIGHT_HAND_PICO4_PORT, "右手端口应匹配PICO4配置"
    print("✓ 双手模式配置正确")
    
    # 测试端口验证
    print("\n4. 测试端口验证:")
    try:
        # 尝试创建具有重复端口的配置
        invalid_config = UnifiedPICO4VRHandDetectorCfg(
            pico4_pub_port=8088,
            button_port=8088  # 重复端口
        )
        assert False, "应拒绝重复端口配置"
    except ValueError as e:
        print(f"✓ 正确拒绝重复端口配置: {e}")
    
    try:
        # 尝试创建具有无效端口的配置
        invalid_config = UnifiedPICO4VRHandDetectorCfg(
            pico4_pub_port=99999  # 无效端口
        )
        assert False, "应拒绝无效端口配置"
    except ValueError as e:
        print(f"✓ 正确拒绝无效端口配置: {e}")
    
    print("\n=== 所有测试通过! ===")
    

if __name__ == "__main__":
    test_pico4_detector_config()
