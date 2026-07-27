#!/bin/bash
#SBATCH --job-name=totten_ub
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Totten η unit-bounds test (bounds only; SSA / optimizer / priors unchanged).
# Original bounds were Pa·yr (1e3, 1e10); convert → MPa·yr via / 1e6.
#   0: no_sliding  — eta_min=1e-3, eta_max=1e4
#   1: max_sliding — eta_min=1e-3, eta_max=1e4
#   2: no_sliding  — eta_min=1e-3, eta_max=1e6 (floor-only / extra headroom)
#   3: max_sliding — eta_min=1e-3, eta_max=1e6

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints outputs

CFGS=(
  configs/totten/unit_bounds/run_torch_vi_only_totten_no_sliding.cfg
  configs/totten/unit_bounds/run_torch_vi_only_totten_max_sliding.cfg
  configs/totten/unit_bounds/run_torch_vi_only_totten_no_sliding_floor_only.cfg
  configs/totten/unit_bounds/run_torch_vi_only_totten_max_sliding_floor_only.cfg
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CFG="${CFGS[$IDX]}"

PRETRAIN_CKPT="checkpoints/torch_pretrain/totten/model_best.pt"
if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "Missing Totten pretrain checkpoint: ${PRETRAIN_CKPT}" >&2
    exit 1
fi

echo "Running unit-bounds member ${IDX}: ${CFG}"
bash scripts/dsi_preempt_wrapper.sh train_vi_only_torch.py "${CFG}"
