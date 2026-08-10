#!/usr/bin/env bash
# Run the SmolVLA policy rollout AND record commanded+achieved positions the
# correct way (for tracking-error / scorer validation). No ROS/RViz.
#
#   ./run_record_rollout.sh <your usual lerobot-rollout args> [--rec.path=/path.csv]
#
# NB: no `set -u` -- conda setup scripts reference unbound vars.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate lerobot_v6

echo "[run_record_rollout] python: $(python --version 2>&1) at $(which python)"
exec python "${SCRIPT_DIR}/record_rollout.py" "$@"
