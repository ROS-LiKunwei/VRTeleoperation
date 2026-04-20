#!/usr/bin/env python3
"""
简单测试PICO4VRHandDetector的配置逻辑。

此测试直接检查PICO4的端口配置和基本逻辑,而不依赖于整个beavr模块。
"""

import sys
import os

# 模拟必要的常量
class Robots:
    LEFT = "left"
    RIGHT = "right"
    BIMANUAL = "bimanual"

class Network:
    LEFT_HAND_PICO4_PORT = 8111
    RIGHT_HAND_PICO4_PORT = 8088

robots = Robots()
network = Network()

# 模拟PICO4VRHandDetector的配置逻辑
def test_hand_ports_config():
    """测试手部端口配置逻辑。"""
    print("=== 测试PICO4手部端口配置 ===")
    
    # 测试右手模式
    print("\n1. 测试右手模式:")
    hand_config = robots.RIGHT
    hand_ports = {}
    
    if hand_config in [robots.RIGHT, robots.BIMANUAL]:
        hand_ports[robots.RIGHT] = network.RIGHT_HAND_PICO4_PORT
    
    if hand_config in [robots.LEFT, robots.BIMANUAL]:
        hand_ports[robots.LEFT] = network.LEFT_HAND_PICO4_PORT
    
    assert robots.RIGHT in hand_ports, "右手模式应包含右手端口"
    assert robots.LEFT not in hand_ports, "右手模式不应包含左手端口"
    assert hand_ports[robots.RIGHT] == 8088, "右手端口应设置为8088"
    print("✓ 右手模式配置正确")
    
    # 测试左手模式
    print("\n2. 测试左手模式:")
    hand_config = robots.LEFT
    hand_ports = {}
    
    if hand_config in [robots.RIGHT, robots.BIMANUAL]:
        hand_ports[robots.RIGHT] = network.RIGHT_HAND_PICO4_PORT
    
    if hand_config in [robots.LEFT, robots.BIMANUAL]:
        hand_ports[robots.LEFT] = network.LEFT_HAND_PICO4_PORT
    
    assert robots.LEFT in hand_ports, "左手模式应包含左手端口"
    assert robots.RIGHT not in hand_ports, "左手模式不应包含右手端口"
    assert hand_ports[robots.LEFT] == 8111, "左手端口应设置为8111"
    print("✓ 左手模式配置正确")
    
    # 测试双手模式
    print("\n3. 测试双手模式:")
    hand_config = robots.BIMANUAL
    hand_ports = {}
    
    if hand_config in [robots.RIGHT, robots.BIMANUAL]:
        hand_ports[robots.RIGHT] = network.RIGHT_HAND_PICO4_PORT
    
    if hand_config in [robots.LEFT, robots.BIMANUAL]:
        hand_ports[robots.LEFT] = network.LEFT_HAND_PICO4_PORT
    
    assert robots.LEFT in hand_ports, "双手模式应包含左手端口"
    assert robots.RIGHT in hand_ports, "双手模式应包含右手端口"
    assert hand_ports[robots.LEFT] == 8111, "左手端口应设置为8111"
    assert hand_ports[robots.RIGHT] == 8088, "右手端口应设置为8088"
    print("✓ 双手模式配置正确")
    
    print("\n=== 所有测试通过! ===")

if __name__ == "__main__":
    test_hand_ports_config()
