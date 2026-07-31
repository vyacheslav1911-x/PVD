#!/usr/bin/env bash
# Play saved candidate trajectories as a ghost in RViz.
# Sets up conda lerobot_v6 (torch + lerobot) + ROS Jazzy + the workspace, then
# runs playback_trajectories.py, forwarding any args (e.g. --pause 1.5 --loop).
# NOTHING here opens a serial port or moves a motor.
#
#   ./run_playback.sh                       # play k_trajectories.pt once
#   ./run_playback.sh --loop --pause 1.0    # loop, 1s beat between candidates
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

echo "[run_playback] python: $(python --version 2>&1) at $(which python)"
exec python "${SCRIPT_DIR}/playback_trajectories.py" "$@"
