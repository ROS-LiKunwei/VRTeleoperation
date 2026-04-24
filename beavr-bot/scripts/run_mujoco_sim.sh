#!/bin/bash
# SYSMO-32 MuJoCo仿真启动脚本
#
# 用法：
#   bash run_mujoco_sim.sh              # 使用默认配置（带渲染）
#   bash run_mujoco_sim.sh --no-render  # 无渲染模式
#
# 该脚本启动MuJoCo仿真环境，接收来自PICO4手势遥操作系统的
# 笛卡尔空间目标命令，驱动SYSMO-32双臂机器人在仿真中运动。
#
# 数据流：
#   PICO4 → pico4.py → keypoint_transform.py → xarm7_operator.py → [MuJoCo仿真]
#
# 前置条件：
#   1. PICO4 VR头显已连接并运行Unity应用
#   2. BeaVR-bot的pico4.py和keypoint_transform.py已启动
#   3. xarm7_operator.py已启动（发布CartesianTarget命令）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# 默认参数
RENDER_FLAG="True"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-render)
            RENDER_FLAG="False"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--no-render]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  SYSMO-32 MuJoCo仿真启动"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "渲染模式: $RENDER_FLAG"
echo "=========================================="

PYTHONPATH=src python3.10 -c "
from beavr.teleop.configs.robots.sysmo_mujoco_config import MuJoCoSimConfig

config = MuJoCoSimConfig(render=$RENDER_FLAG)
simulator = config.build()
simulator.stream()
"
