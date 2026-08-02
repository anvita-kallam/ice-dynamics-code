#!/bin/bash
#SBATCH --job-name=vi_xfer_a40
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Predict/eval only: more_sliding raised_prior_center (eta_init=15, A=20 train)
# checkpoint → more_sliding A=40 NPZ. Does NOT train.

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

CFG="${1:-configs/vi_only_eta_bias_suite/raised_prior_center_transfer_A40.cfg}"
TAG="${2:-raised_prior15_on_A40}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs outputs/vi_only_eta_bias_suite/eta_bias_v1/raised_prior_center_transfer_A40
mkdir -p outputs/figures/vi_only

CKPT="checkpoints/torch_vi_only/eta_bias_suite/eta_bias_v1/raised_prior_center/model_best.pt"
NPZ="../outputs/spinup/production/more_sliding_A40/SteadyState_more_sliding_A40_10500yr_ramp4000_1refine_grid.npz"

if [[ ! -f "${CKPT}" ]]; then
    echo "Missing raised-prior checkpoint: ${CKPT}" >&2
    exit 1
fi
if [[ ! -f "${NPZ}" ]]; then
    echo "Missing A40 NPZ: ${NPZ}" >&2
    exit 1
fi

# Match dsi_preempt_wrapper: non-interactive sbatch needs explicit conda.sh.
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${TORCH_ENV}"
elif [[ -f "${HOME}/.conda/envs/${TORCH_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.conda/envs/${TORCH_ENV}/bin/activate"
elif [[ -f "${TORCH_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${TORCH_ENV}/bin/activate"
else
    echo "Cannot activate TORCH_ENV=${TORCH_ENV}" >&2
    exit 1
fi

echo "python=$(command -v python)"
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
echo "  outputs/vi_only_eta_bias_suite/eta_bias_v1/raised_prior_center_transfer_A40/"
