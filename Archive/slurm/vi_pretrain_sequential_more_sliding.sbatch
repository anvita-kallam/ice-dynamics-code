#!/bin/bash
#SBATCH --job-name=vi_pre_seq_ms
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Sequential experiment — Stage 1: PINN pretrain (observation loss only).
# Checkpoints: checkpoints/torch_pretrain/more_sliding_sequential/
# Does not touch joint pretrain (more_sliding/) or joint training.

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

bash scripts/dsi_preempt_wrapper.sh pretrain_solution_torch.py run_torch_sequential.cfg
