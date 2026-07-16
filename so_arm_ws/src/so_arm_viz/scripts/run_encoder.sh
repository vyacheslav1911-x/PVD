#!/usr/bin/env bash
# One-command launch for the live encoder node under the correct interpreter.
#
# The node imports BOTH rclpy (ROS Jazzy, cpython-3.12) and lerobot (conda
# lerobot_v6, Python 3.12.13). Conda 'base' is Python 3.13 and CANNOT load ROS's
# 3.12 rclpy C-extension -- so we force lerobot_v6 here regardless of which env
# the calling shell happens to be in.
#
# Usage:  ./run_encoder.sh [extra --ros-args ...]
# NB: no `set -u` -- ROS/conda setup scripts reference unbound vars.
set -eo pipefail

# Resolve workspace root (this script lives in <ws>/src/so_arm_viz/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 1) ROS Jazzy
source /opt/ros/jazzy/setup.bash

# 2) conda lerobot_v6 (Python 3.12) -- activate no matter the current env
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate lerobot_v6

# 3) workspace overlay (puts so_arm_viz on PYTHONPATH)
# shellcheck disable=SC1091
source "${WS_ROOT}/install/setup.bash"

echo "[run_encoder] python: $(python --version 2>&1) at $(which python)"
exec python -m so_arm_viz.encoder_node "$@"
