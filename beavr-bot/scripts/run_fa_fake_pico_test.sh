#!/usr/bin/env bash
# Run FA teleop in MuJoCo with synthetic PICO4 hand data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DURATION_S=30
RATE_HZ=60
HOST="192.168.1.134"
LATERALITY="right"
CONTROL_BACKEND="mujoco"
START_RVIZ=false
KEEP_RUNNING=false

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --duration SEC        Fake PICO run duration, default: $DURATION_S
  --rate HZ             Fake PICO publish rate, default: $RATE_HZ
  --host IP             BeaVR bind host, default: $HOST
  --laterality MODE     right|left|bimanual, default: $LATERALITY
  --backend MODE        mujoco|real_with_mujoco|real, default: $CONTROL_BACKEND
  --rviz                Also launch fa_rviz_command_bridge
  --keep-running        Do not stop teleop/RViz after fake sender exits
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
            DURATION_S="$2"
            shift 2
            ;;
        --duration=*)
            DURATION_S="${1#*=}"
            shift
            ;;
        --rate)
            RATE_HZ="$2"
            shift 2
            ;;
        --rate=*)
            RATE_HZ="${1#*=}"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        --laterality)
            LATERALITY="$2"
            shift 2
            ;;
        --laterality=*)
            LATERALITY="${1#*=}"
            shift
            ;;
        --backend)
            CONTROL_BACKEND="$2"
            shift 2
            ;;
        --backend=*)
            CONTROL_BACKEND="${1#*=}"
            shift
            ;;
        --rviz)
            START_RVIZ=true
            shift
            ;;
        --keep-running)
            KEEP_RUNNING=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$LATERALITY" in
    right|left|bimanual) ;;
    *)
        echo "--laterality must be right, left, or bimanual" >&2
        exit 1
        ;;
esac

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

if [[ -f "/opt/ros/humble/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source /opt/ros/humble/setup.bash
fi
if [[ -f "/home/likunwei/humanoid_ws/install/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source /home/likunwei/humanoid_ws/install/setup.bash
fi
if [[ -f "install/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source install/setup.bash
fi

mkdir -p Log
RUN_ID="$(date +%Y-%m-%d-%H-%M-%S)"
TELEOP_STDOUT_LOG="Log/fa_fake_pico_test_${RUN_ID}_teleop_stdout.log"
FAKE_STDOUT_LOG="Log/fa_fake_pico_test_${RUN_ID}_fake_pico_stdout.log"
RVIZ_STDOUT_LOG="Log/fa_fake_pico_test_${RUN_ID}_rviz_stdout.log"

TELEOP_PID=""
RVIZ_PID=""

cleanup() {
    if [[ "$KEEP_RUNNING" == "true" ]]; then
        return
    fi
    if [[ -n "$RVIZ_PID" ]] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        kill -INT "$RVIZ_PID" 2>/dev/null || true
    fi
    if [[ -n "$TELEOP_PID" ]] && kill -0 "$TELEOP_PID" 2>/dev/null; then
        kill -INT "$TELEOP_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "=========================================="
echo "  FA fake PICO MuJoCo test"
echo "=========================================="
echo "project:      $PROJECT_DIR"
echo "python:       $PYTHON"
echo "host:         $HOST"
echo "laterality:   $LATERALITY"
echo "backend:      $CONTROL_BACKEND"
echo "duration:     ${DURATION_S}s"
echo "rate:         ${RATE_HZ}Hz"
echo "rviz:         $START_RVIZ"
echo "keep-running: $KEEP_RUNNING"
echo "teleop log:   $TELEOP_STDOUT_LOG"
echo "fake log:     $FAKE_STDOUT_LOG"
echo "=========================================="

"$PYTHON" -m beavr.teleop.main \
    --robot_name=fa \
    --laterality="$LATERALITY" \
    --teleop.flags.robot_interface=True \
    --teleop.flags.sim_env=True \
    --control_backend="$CONTROL_BACKEND" \
    >"$TELEOP_STDOUT_LOG" 2>&1 &
TELEOP_PID=$!
echo "Started FA teleop PID=$TELEOP_PID"

if [[ "$START_RVIZ" == "true" ]]; then
    ros2 launch fa_rviz_command_bridge fa_command_rviz.launch.py >"$RVIZ_STDOUT_LOG" 2>&1 &
    RVIZ_PID=$!
    echo "Started FA RViz bridge PID=$RVIZ_PID"
fi

echo "Waiting for teleop sockets..."
sleep 8

"$PYTHON" scripts/fa_fake_pico_sender.py \
    --host "$HOST" \
    --duration "$DURATION_S" \
    --rate "$RATE_HZ" \
    >"$FAKE_STDOUT_LOG" 2>&1

echo "Fake PICO sender finished."
echo "Recent BeaVR logs:"
ls -lt Log/beavr_run_*.log 2>/dev/null | head -3 || true
echo "Recent IK logs:"
ls -lt Log/ik/fa_ik_*.csv 2>/dev/null | head -3 || true

if [[ "$KEEP_RUNNING" == "true" ]]; then
    echo "Teleop/RViz left running:"
    echo "  teleop PID=$TELEOP_PID"
    if [[ -n "$RVIZ_PID" ]]; then
        echo "  rviz PID=$RVIZ_PID"
    fi
else
    echo "Stopping teleop/RViz..."
fi
