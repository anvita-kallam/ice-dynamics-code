#!/bin/bash
#SBATCH --job-name=totten_eta_sw
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Totten η-collapse prior sweep (no_sliding C=100). Reuses frozen Totten PINN.
# Array maps to configs/totten/eta_collapse_sweep/*.cfg

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
  configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_15.cfg
  configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_20.cfg
  configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_25.cfg
  configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_30.cfg
  configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_15_softplus_floor.cfg
  configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_20_softplus_floor.cfg
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CFG="${CFGS[$IDX]}"

PRETRAIN_CKPT="checkpoints/torch_pretrain/totten/model_best.pt"
if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "Missing Totten pretrain checkpoint: ${PRETRAIN_CKPT}" >&2
    exit 1
fi

echo "Running sweep member ${IDX}: ${CFG}"
bash scripts/dsi_preempt_wrapper.sh train_vi_only_torch.py "${CFG}"
