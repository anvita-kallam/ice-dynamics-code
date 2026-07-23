#!/bin/bash
#SBATCH --job-name=vi_pre_totten
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Totten Stage 1 — shared PINN pretrain (both sliding cases reuse this).
# Checkpoints: checkpoints/torch_pretrain/totten/
#
#   cd Archive
#   sbatch slurm/vi_pretrain_totten.sbatch

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints

DATA_NPZ="${ICE_DYNAMICS_ROOT}/data/real/totten/totten_archive_vi_2022.npz"
if [[ ! -f "${DATA_NPZ}" ]]; then
    echo "Missing Totten archive NPZ: ${DATA_NPZ}" >&2
    exit 1
fi

bash scripts/dsi_preempt_wrapper.sh \
    pretrain_solution_torch.py configs/totten/run_torch_pretrain_totten.cfg
