#!/usr/bin/env bash
set -euo pipefail

IK_INSTALL_PREFIX="${1:-/home/likunwei/humanoid_ws/install/ik_7dof}"
OUTPUT_TGZ="${2:-/tmp/ik_7dof_runtime.tgz}"

if [[ ! -d "${IK_INSTALL_PREFIX}" ]]; then
  echo "ik_7dof install prefix not found: ${IK_INSTALL_PREFIX}" >&2
  echo "Build it first, for example:" >&2
  echo "  cd /home/likunwei/humanoid_ws" >&2
  echo "  source /opt/ros/humble/setup.bash" >&2
  echo "  colcon build --packages-select ik_7dof" >&2
  exit 1
fi

if ! find "${IK_INSTALL_PREFIX}" -name 'ik_7dof_pybind*.so' -print -quit | grep -q .; then
  echo "ik_7dof_pybind*.so not found under ${IK_INSTALL_PREFIX}" >&2
  echo "Rebuild ik_7dof after enabling the pybind target." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_TGZ}")"
tar -C "$(dirname "${IK_INSTALL_PREFIX}")" -czf "${OUTPUT_TGZ}" "$(basename "${IK_INSTALL_PREFIX}")"
echo "Wrote ${OUTPUT_TGZ}"
echo "On the robot board:"
echo "  mkdir -p /opt/fa_runtime"
echo "  tar -C /opt/fa_runtime -xzf $(basename "${OUTPUT_TGZ}")"
echo "  export IK_7DOF_INSTALL_PREFIX=/opt/fa_runtime/$(basename "${IK_INSTALL_PREFIX}")"
