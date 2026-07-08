#!/usr/bin/env bash
# DSI Slurm preemption-safe wrapper for Archive PyTorch training scripts.
#
# Forwards SIGUSR1 to the training process group, waits for checkpoint+exit 99,
# then exits 99 so Slurm requeues the job (--requeue required on sbatch).
#
# Usage (inside an sbatch script, from Archive/):
#   bash scripts/dsi_preempt_wrapper.sh pretrain_solution_torch.py run_torch.cfg
#   bash scripts/dsi_preempt_wrapper.sh train_torch.py run_torch.cfg
#
# Environment:
#   TORCH_ENV        Conda env name or prefix with PyTorch (default: pytorch)
#   ICE_DYNAMICS_ROOT  Repo root (default: parent of Archive/)

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <python_script.py> <config.cfg> [extra args...]" >&2
    exit 2
fi

SCRIPT_NAME="$1"
CONFIG_FILE="$2"
shift 2

ARCHIVE_DIR="${ARCHIVE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
TORCH_ENV="${TORCH_ENV:-pytorch}"

child_pid=""

signal_child_group() {
    local signal="$1"
    if [[ -n "${child_pid:-}" ]] && kill -0 "$child_pid" 2>/dev/null; then
        echo "[$(date)] Sending ${signal} to training process group..." >&2
        kill -s "${signal}" -- "-${child_pid}" 2>/dev/null || \
            kill -s "${signal}" "$child_pid" 2>/dev/null || true
    fi
}

on_preempt() {
    echo "[$(date)] USR1 received: checkpointing before preemption or time-limit warning." >&2
    signal_child_group USR1
    wait "$child_pid" || true
    echo "[$(date)] Exiting with code 99 for Slurm requeue policy." >&2
    exit 99
}

on_term() {
    echo "[$(date)] TERM received: terminating without intentional requeue." >&2
    signal_child_group TERM
    wait "$child_pid" || true
    exit 143
}

trap on_preempt USR1
trap on_term TERM

activate_torch_env() {
    if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
        # shellcheck disable=SC1091
        source /opt/conda/etc/profile.d/conda.sh
    elif [[ -f "${TORCH_ENV}/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "${TORCH_ENV}/bin/activate"
        return
    elif command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        echo "Could not activate TORCH_ENV=${TORCH_ENV}. Set TORCH_ENV to a conda env name or prefix." >&2
        exit 1
    fi
    conda activate "${TORCH_ENV}"
}

cd "${ARCHIVE_DIR}"
mkdir -p logs checkpoints outputs logs/slurm
export MPLBACKEND=Agg
activate_torch_env

echo "[$(date)] ROOT=${ROOT} ARCHIVE=${ARCHIVE_DIR} RESTART=${SLURM_RESTART_COUNT:-0}" >&2
echo "[$(date)] Running: ${SCRIPT_NAME} ${CONFIG_FILE} $*" >&2

setsid python "${SCRIPT_NAME}" "${CONFIG_FILE}" "$@" &
child_pid=$!

return_code=0
wait "$child_pid" || return_code=$?

echo "[$(date)] Training exited with code ${return_code}" >&2
if [[ "$return_code" -eq 99 ]]; then
    exit 99
fi
exit "$return_code"
