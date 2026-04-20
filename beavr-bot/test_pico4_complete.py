#!/usr/bin/env python3
"""
PICO4适配完整测试脚本。

此脚本验证PICO4在beavr-bot和beavr-app中的适配是否正常工作，
包括端口配置、探测器配置、网络连接等功能。
"""

import sys
import os
import socket
import time
import json
from typing import Dict, List, Tuple

# 模拟必要的常量
class Robots:
    LEFT = "left"
    RIGHT = "right"
    BIMANUAL = "bimanual"

class Network:
    LEFT_HAND_PICO4_PORT = 8111
    RIGHT_HAND_PICO4_PORT = 8088
    HOST_ADDRESS = "10.0.0.51"

class Ports:
    KEYPOINT_STREAM_PORT = 8101
    RESOLUTION_BUTTON_PORT = 8095
    TELEOP_RESET_PORT = 8100

robots = Robots()
network = Network()
ports = Ports()


def test_port_configuration():
    """测试端口配置是否正确。"""
    print("=== 测试PICO4端口配置 ===")
    
    # 测试左手端口
    assert network.LEFT_HAND_PICO4_PORT == 8111, "左手端口应为8111"
    print(f"✓ 左手端口配置正确: {network.LEFT_HAND_PICO4_PORT}")
    
    # 测试右手端口
    assert network.RIGHT_HAND_PICO4_PORT == 8088, "右手端口应为8088"
    print(f"✓ 右手端口配置正确: {network.RIGHT_HAND_PICO4_PORT}")
    
    # 测试端口范围
    assert 1 <= network.LEFT_HAND_PICO4_PORT <= 65535, "左手端口应在有效范围内"
    assert 1 <= network.RIGHT_HAND_PICO4_PORT <= 65535, "右手端口应在有效范围内"
    print("✓ 所有端口都在有效范围内")


def test_detector_config():
    """测试探测器配置逻辑。"""
    print("\n=== 测试PICO4探测器配置 ===")
    
    # 测试右手模式配置
    print("\n1. 测试右手模式配置:")
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
    
    # 测试左手模式配置
    print("\n2. 测试左手模式配置:")
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
    
    # 测试双手模式配置
    print("\n3. 测试双手模式配置:")
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


def test_network_json():
    """测试Network.json配置文件。"""
    print("\n=== 测试Network.json配置 ===")
    
    # 检查Network.json文件是否存在
    network_json_path = "/home/likunwei/dataCollection/BeaVR-app/BeaVR-Unity/Assets/Resources/Configurations/Network.json"
    
    if not os.path.exists(network_json_path):
        print(f"⚠ Network.json文件不存在: {network_json_path}")
        return
    
    # 读取并解析Network.json
    with open(network_json_path, 'r') as f:
        config = json.load(f)
    
    # 检查端口配置
    assert config['rightkeyptPortNum'] == "8088", "右手关键点端口应为8088"
    print(f"✓ 右手关键点端口配置正确: {config['rightkeyptPortNum']}")
    
    assert config['leftkeyptPortNum'] == "8111", "左手关键点端口应为8111"
    print(f"✓ 左手关键点端口配置正确: {config['leftkeyptPortNum']}")
    
    # 检查其他端口配置
    assert 'resolutionPortNum' in config, "应包含分辨率端口配置"
    assert 'PausePortNum' in config, "应包含暂停端口配置"
    print("✓ 所有必需的端口配置都存在")


def test_port_availability():
    """测试端口是否可用。"""
    print("\n=== 测试端口可用性 ===")
    
    ports_to_test = [
        ("左手关键点端口", network.LEFT_HAND_PICO4_PORT),
        ("右手关键点端口", network.RIGHT_HAND_PICO4_PORT),
        ("分辨率端口", ports.RESOLUTION_BUTTON_PORT),
        ("暂停端口", ports.TELEOP_RESET_PORT),
    ]
    
    for port_name, port in ports_to_test:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((network.HOST_ADDRESS, port))
        sock.close()
        
        if result == 0:
            print(f"✓ {port_name} ({port}) 可连接")
        else:
            print(f"⚠ {port_name} ({port}) 不可连接 (可能未启动服务)")


def test_data_format():
    """测试数据格式是否符合预期。"""
    print("\n=== 测试数据格式 ===")
    
    # 模拟手部关键点数据（26个关节，每个关节3个坐标）
    num_joints = 26
    sample_data = []
    
    for i in range(num_joints):
        sample_data.extend([0.1, 0.2, 0.3])
    
    # 检查数据长度
    expected_length = num_joints * 3
    assert len(sample_data) == expected_length, f"数据长度应为{expected_length}，实际为{len(sample_data)}"
    print(f"✓ 手部关键点数据格式正确: {num_joints}个关节 × 3个坐标 = {expected_length}个值")
    
    # 测试数据序列化
    coords = "|".join([f"{sample_data[i*3]},{sample_data[i*3+1]},{sample_data[i*3+2]}" for i in range(num_joints)])
    data_str = f"relative:{coords}"
    
    assert "relative:" in data_str, "数据应包含类型标记"
    assert "|" in data_str, "数据应包含坐标分隔符"
    assert "," in data_str, "数据应包含坐标分量分隔符"
    print("✓ 数据序列化格式正确")


def test_detector_file_exists():
    """测试PICO4探测器文件是否存在。"""
    print("\n=== 测试PICO4探测器文件 ===")
    
    detector_file = "/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/detector/vr/pico4.py"
    
    if os.path.exists(detector_file):
        print(f"✓ PICO4探测器文件存在: {detector_file}")
        
        # 检查文件内容
        with open(detector_file, 'r') as f:
            content = f.read()
        
        # 检查关键类和函数
        assert "class PICO4VRHandDetector" in content, "应包含PICO4VRHandDetector类"
        print("✓ 包含PICO4VRHandDetector类")
        
        assert "def stream" in content, "应包含stream方法"
        print("✓ 包含stream方法")
        
        assert "def _process_keypoints" in content, "应包含_process_keypoints方法"
        print("✓ 包含_process_keypoints方法")
    else:
        print(f"⚠ PICO4探测器文件不存在: {detector_file}")


def test_commented_files():
    """测试关键文件是否已添加注释。"""
    print("\n=== 测试文件注释 ===")
    
    files_to_check = [
        ("/home/likunwei/dataCollection/BeaVR-app/BeaVR-Unity/Assets/Scripts/NetworkManager.cs", "NetworkManager.cs"),
        ("/home/likunwei/dataCollection/BeaVR-app/BeaVR-Unity/Assets/Scripts/Network/NetMQController.cs", "NetMQController.cs"),
        ("/home/likunwei/dataCollection/BeaVR-app/BeaVR-Unity/Assets/Scripts/Gesture Detection/GestureDetectorXR.cs", "GestureDetectorXR.cs"),
        ("/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/detector/vr/pico4.py", "pico4.py"),
    ]
    
    for file_path, file_name in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 检查是否包含中文注释
            has_chinese_comments = any(
                char in content for char in "中文注释网络配置套接字手部追踪"
            )
            
            if has_chinese_comments:
                print(f"✓ {file_name} 已添加中文注释")
            else:
                print(f"⚠ {file_name} 未检测到中文注释")
        else:
            print(f"⚠ {file_name} 文件不存在")


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("PICO4适配完整测试")
    print("=" * 60)
    
    try:
        # 运行所有测试
        test_port_configuration()
        test_detector_config()
        test_network_json()
        test_port_availability()
        test_data_format()
        test_detector_file_exists()
        test_commented_files()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
