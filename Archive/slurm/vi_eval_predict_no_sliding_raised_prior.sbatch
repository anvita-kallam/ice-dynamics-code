#!/bin/bash
#SBATCH --job-name=vi_eval_ns
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Evaluate + predict after no_sliding raised-prior VI.
#   sbatch --dependency=afterok:<VI> slurm/vi_eval_predict_no_sliding_raised_prior.sbatch

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

CFG="${1:-configs/mismip/run_torch_vi_only_no_sliding_raised_prior.cfg}"
TAG="${2:-no_sliding_raised_prior}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs outputs/vi_only/no_sliding_raised_prior outputs/figures/vi_only

CKPT="checkpoints/torch_vi_only/no_sliding_raised_prior/model_best.pt"
if [[ ! -f "${CKPT}" ]]; then
    # Fall back to latest if early-stop best was never written.
    CKPT="checkpoints/torch_vi_only/no_sliding_raised_prior/model.pt"
fi
if [[ ! -f "${CKPT}" ]]; then
    echo "Missing VI checkpoint under checkpoints/torch_vi_only/no_sliding_raised_prior/" >&2
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

echo "CFG=${CFG}"
echo "CKPT=${CKPT}"
echo "TAG=${TAG}"

CKPT_CHOICE="best"
if [[ ! -f checkpoints/torch_vi_only/no_sliding_raised_prior/model_best.pt ]]; then
    CKPT_CHOICE="latest"
fi

python evaluate_vi_only_posterior.py "${CFG}" \
    --tag "${TAG}" \
    --checkpoint "${CKPT_CHOICE}" \
    --output-dir "outputs/figures/vi_only/${TAG}"

python predict_vi_only_torch.py "${CFG}"

echo "Done."
echo "  figures: outputs/figures/vi_only/${TAG}/"
echo "  samples: outputs/vi_only/no_sliding_raised_prior/posterior_samples.h5"
