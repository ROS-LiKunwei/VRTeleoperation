#!/bin/bash
# SYSMO-32双臂机器人遥操作启动脚本
#
# 用法：
#   bash run_sysmo32.sh                          # 双手模式（默认）
#   bash run_sysmo32.sh --laterality=right       # 右手模式
#   bash run_sysmo32.sh --laterality=left        # 左手模式
#   bash run_sysmo32.sh --mujoco                 # 启动MuJoCo仿真（替代物理机器人）
#
# 该脚本启动SYSMO-32双臂机器人的完整遥操作系统：
#   PICO4 → pico4.py → keypoint_transform.py → sysmo32_operator → sysmo32_robot/MuJoCo
#
# 前置条件：
#   1. PICO4 VR头显已连接并运行Unity应用
#   2. Unity应用通过ZMQ PUSH发送手部数据到8087/8110端口

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# 默认参数
LATERALITY="bimanual"
MUJOCO=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --laterality=*)
            LATERALITY="${1#*=}"
            shift
            ;;
        --mujoco)
            MUJOCO=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--laterality=right|left|bimanual] [--mujoco]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  SYSMO-32 双臂机器人遥操作启动"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "模式: $LATERALITY"
echo "MuJoCo仿真: $MUJOCO"
echo "=========================================="

# 启动主遥操作系统
PYTHONPATH=src python3.10 -c "
from beavr.teleop.main import main
import sys
sys.argv = ['teleop', '--robot_name=sysmo32', '--laterality=$LATERALITY']
main()
" &
TELEOP_PID=$!

# 如果启用MuJoCo仿真，额外启动仿真器
if [ "$MUJOCO" = true ]; then
    echo "启动MuJoCo仿真..."
    sleep 3  # 等待主系统启动
    PYTHONPATH=src python3.10 -c "
from beavr.teleop.configs.robots.sysmo_mujoco_config import MuJoCoSimConfig
config = MuJoCoSimConfig(render=True)
simulator = config.build()
simulator.stream()
" &
    MUJOCO_PID=$!
    echo "MuJoCo仿真PID: $MUJOCO_PID"
fi

echo "遥操作系统PID: $TELEOP_PID"
echo "按Ctrl+C停止所有进程..."

# 等待主进程
wait $TELEOP_PID
