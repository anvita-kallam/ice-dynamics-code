#!/bin/bash
#SBATCH --job-name=vi_pre_seq_ns
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Sequential Stage 1 — PINN pretrain on no_sliding.
# Checkpoints: checkpoints/torch_pretrain/no_sliding_sequential/

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

NPZ="../outputs/spinup/production/no_sliding/SteadyState_no_sliding_10500yr_ramp4000_1refine_grid.npz"
if [[ ! -f "${NPZ}" ]]; then
    echo "Missing no_sliding NPZ: ${NPZ}" >&2
    exit 1
fi

bash scripts/dsi_preempt_wrapper.sh \
    pretrain_solution_torch.py configs/mismip/run_torch_sequential_no_sliding.cfg
