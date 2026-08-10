#!/usr/bin/env bash
# Log-only drop-in for `lerobot-rollout`: runs your rollout AND writes a per-step
# CSV of q / qdot / qddot / load / current (for RNEA torque validation).
# No ROS / RViz. Just conda lerobot_v6 + your usual rollout args.
#
#   ./run_log_dynamics_rollout.sh <your usual lerobot-rollout args> \
#       [--log.path=/path/to/dynamics.csv]
#
# NB: no `set -u` -- conda setup scripts reference unbound vars.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate lerobot_v6

echo "[run_log_dynamics] python: $(python --version 2>&1) at $(which python)"
exec python "${SCRIPT_DIR}/log_dynamics_rollout.py" "$@"
