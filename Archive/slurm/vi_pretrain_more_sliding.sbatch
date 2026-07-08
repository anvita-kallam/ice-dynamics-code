#!/bin/bash
#SBATCH --job-name=vi_pretrain_ms
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Stage 1 — PINN pretrain (MeanNetwork) for production more_sliding.
#
# Submit from login node:
#   cd /path/to/ice-dynamics/Archive
#   mkdir -p logs/slurm
#   sbatch slurm/vi_pretrain_more_sliding.sbatch
#
# Optional env overrides:
#   export TORCH_ENV=pytorch          # conda env with PyTorch + CUDA
#   export ICE_DYNAMICS_ROOT=/project/<account>/ice-dynamics

set -euo pipefail

# Slurm copies this script to /var/spool/... — do not use BASH_SOURCE for paths.
# Submit from Archive/:  cd ~/ice-dynamics/Archive && sbatch slurm/...
ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints

bash scripts/dsi_preempt_wrapper.sh pretrain_solution_torch.py run_torch.cfg
