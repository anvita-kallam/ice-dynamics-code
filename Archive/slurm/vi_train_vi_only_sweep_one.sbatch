#!/bin/bash
#SBATCH --job-name=vi_only_sw
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Usage (from Archive/):
#   sbatch slurm/vi_train_vi_only_sweep_one.sbatch configs/vi_only_sweeps/run_....cfg

set -euo pipefail

CFG="${1:?Usage: sbatch vi_train_vi_only_sweep_one.sbatch <cfg>}"

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints outputs

PRETRAIN_CKPT="checkpoints/torch_pretrain/more_sliding_sequential/model_best.pt"
if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "Missing sequential pretrain checkpoint: ${PRETRAIN_CKPT}" >&2
    exit 1
fi

bash scripts/dsi_preempt_wrapper.sh train_vi_only_torch.py "${CFG}"
