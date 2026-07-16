#!/usr/bin/env bash
# Drop-in replacement for `lerobot-rollout` that also drives RViz (live + ghost).
# Sets up the correct interpreter (conda lerobot_v6, Python 3.12) + ROS + workspace,
# then runs ghost_rollout.py, forwarding ALL your usual lerobot-rollout args.
#
# Usage:
#   ./run_ghost_rollout.sh --policy.path=... --robot.type=so101_follower \
#       --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41533793-if00 \
#       --task="..." --display_data=true
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

echo "[run_ghost_rollout] python: $(python --version 2>&1) at $(which python)"
exec python "${SCRIPT_DIR}/ghost_rollout.py" "$@"
