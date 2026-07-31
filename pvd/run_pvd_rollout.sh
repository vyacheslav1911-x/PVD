#!/usr/bin/env bash
# Drop-in for `lerobot-rollout` with PVD selection. Sets up conda lerobot_v6 and
# puts the repo on PYTHONPATH, then runs pvd_rollout.py forwarding ALL your args.
# Generation runs on cuda; the feasibility scorer runs on cpu (Pinocchio). No ROS.
#
#   ./run_pvd_rollout.sh <your usual lerobot-rollout args> \
#       --pvd.enabled=true --pvd.num_samples=32 --pvd.mode=selection --pvd.threshold=5.0
#
# NB: no `set -u` -- conda setup scripts reference unbound vars.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate lerobot_v6

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[run_pvd_rollout] python: $(python --version 2>&1) at $(which python)"
exec python "${SCRIPT_DIR}/pvd_rollout.py" "$@"
