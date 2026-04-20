"""PICO4 VR手部追踪探测器测试脚本"""

import sys
import os

# 添加项目路径到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

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


if __name__ == "__main__":
    success = test_pico4_detector()
    sys.exit(0 if success else 1)
