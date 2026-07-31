#!/usr/bin/env bash
# Interactive candidate browser: step through trajectories, see feasibility (Φ),
# play the chosen one as a ghost in RViz. Sets up conda lerobot_v6 (torch +
# pinocchio + lerobot) + ROS Jazzy + the workspace. Opens NO serial port.
#
#   ./run_interactive.sh                 # browse k_trajectories.pt
#   ./run_interactive.sh --threshold 50  # looser feasibility bar
#
# NB: no `set -u` -- ROS/conda setup scripts reference unbound vars.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source /opt/ros/jazzy/setup.bash

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate lerobot_v6

# shellcheck disable=SC1091
source "${WS_ROOT}/install/setup.bash"

echo "[run_interactive] python: $(python --version 2>&1) at $(which python)"
exec python "${SCRIPT_DIR}/play_interactive.py" "$@"
