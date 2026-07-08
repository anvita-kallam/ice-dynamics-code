#!/bin/bash
#SBATCH --job-name=vi_predict_ms
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Stage 3 — draw VI posterior samples (short job; no preemption wrapper needed).

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs outputs

if [[ ! -f checkpoints/torch_joint/more_sliding/model.pt ]]; then
    echo "Missing joint checkpoint: checkpoints/torch_joint/more_sliding/model.pt" >&2
    exit 1
fi

if [[ -f "${TORCH_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${TORCH_ENV}/bin/activate"
elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${TORCH_ENV}"
fi

python predict_torch.py run_torch.cfg
