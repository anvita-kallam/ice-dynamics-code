#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

CFG="${1:?Usage: sbatch vi_only_eta_bias_trial.sbatch CONFIG RUN_ID}"
RUN_ID="${2:?Usage: sbatch vi_only_eta_bias_trial.sbatch CONFIG RUN_ID}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints outputs
if [[ ! -f checkpoints/torch_pretrain/more_sliding_sequential/model_best.pt ]]; then
    echo "Missing sequential pretrain checkpoint." >&2
    exit 1
fi

bash scripts/dsi_preempt_wrapper.sh train_vi_only_torch.py "${CFG}"

# Rank only the best-correlation checkpoint, never the final early-stop state.
source /opt/conda/etc/profile.d/conda.sh
conda activate "${TORCH_ENV}"
python evaluate_vi_only_posterior.py "${CFG}" --checkpoint best

echo "VI-only eta-bias trial complete: ${RUN_ID}"
