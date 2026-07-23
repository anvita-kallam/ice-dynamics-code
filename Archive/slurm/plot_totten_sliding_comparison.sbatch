#!/bin/bash
#SBATCH --job-name=totten_cmp
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Plot Totten no_sliding vs max_sliding η mean/std comparison maps.
#
#   cd Archive
#   sbatch slurm/plot_totten_sliding_comparison.sbatch

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm outputs/figures/vi/totten_sliding_comparison

source /opt/conda/etc/profile.d/conda.sh
conda activate "${TORCH_ENV}"

python plot_totten_sliding_comparison.py \
  configs/totten/run_torch_vi_only_totten_no_sliding.cfg \
  configs/totten/run_torch_vi_only_totten_max_sliding.cfg \
  --checkpoint latest \
  --output-dir outputs/figures/vi/totten_sliding_comparison

echo "Done. Figures in outputs/figures/vi/totten_sliding_comparison/"
