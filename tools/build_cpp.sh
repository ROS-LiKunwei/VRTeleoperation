#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
set -u
cd "$project_root/cpp_ws"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export MAKEFLAGS="${MAKEFLAGS:--j1}"
colcon build --packages-select fa_teleop --executor sequential --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
ctest --test-dir build/fa_teleop --output-on-failure
python3 src/fa_teleop/tools/create_receipt.py --prefix install/fa_teleop --output ../validation/BUILD_RECEIPT.json
python3 ../tools/export_versions.py --prefix install/fa_teleop --output ../provenance/COMPONENT_VERSIONS.json
