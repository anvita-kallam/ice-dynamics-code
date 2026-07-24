#!/bin/bash
#SBATCH --job-name=vi_xfer_ns
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Predict/eval only: more_sliding raised_prior_center (eta_init=15)
# checkpoint → no_sliding NPZ. Does NOT train.

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

CFG="${1:-configs/vi_only_eta_bias_suite/raised_prior_center_transfer_no_sliding.cfg}"
TAG="${2:-raised_prior15_on_no_sliding}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs outputs/vi_only_eta_bias_suite/eta_bias_v1/raised_prior_center_transfer_no_sliding
mkdir -p ../outputs/figures/vi_only

CKPT="checkpoints/torch_vi_only/eta_bias_suite/eta_bias_v1/raised_prior_center/model_best.pt"
NPZ="../outputs/spinup/production/no_sliding/SteadyState_no_sliding_10500yr_ramp4000_1refine_grid.npz"

if [[ ! -f "${CKPT}" ]]; then
    echo "Missing raised-prior checkpoint: ${CKPT}" >&2
    exit 1
fi
if [[ ! -f "${NPZ}" ]]; then
    echo "Missing no_sliding NPZ: ${NPZ}" >&2
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
echo "NPZ=${NPZ}"
echo "TAG=${TAG}"

python evaluate_vi_only_posterior.py "${CFG}" \
    --tag "${TAG}" \
    --checkpoint best \
    --output-dir "outputs/figures/vi_only/${TAG}"

python predict_vi_only_torch.py "${CFG}"

echo "Done. Summary under outputs/figures/vi_only/${TAG}/ and"
echo "  outputs/vi_only_eta_bias_suite/eta_bias_v1/raised_prior_center_transfer_no_sliding/"
