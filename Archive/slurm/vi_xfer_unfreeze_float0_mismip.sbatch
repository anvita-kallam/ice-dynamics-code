#!/bin/bash
#SBATCH --job-name=vi_xfer_uf0
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Totten unfreeze_float0 max_sliding → MISMIP more_sliding predict/eval.
#   sbatch slurm/vi_xfer_unfreeze_float0_mismip.sbatch A20
#   sbatch slurm/vi_xfer_unfreeze_float0_mismip.sbatch A40

set -euo pipefail

CASE="${1:?Usage: sbatch vi_xfer_unfreeze_float0_mismip.sbatch A20|A40}"

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs outputs/figures/vi_only
mkdir -p outputs/totten/c_sensitivity/unfreeze_float0_xfer

CFG="configs/totten/c_sensitivity/unfreeze_float0_xfer_mismip/xfer_max_on_${CASE}.cfg"
TAG="unfreeze_float0_on_mismip_${CASE}"
CKPT_DIR="checkpoints/torch_vi_only/totten/c_sensitivity/unfreeze_float0/max_sliding"

if [[ "${CASE}" == "A20" ]]; then
    NPZ="../outputs/spinup/production/more_sliding/SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz"
elif [[ "${CASE}" == "A40" ]]; then
    NPZ="../outputs/spinup/production/more_sliding_A40/SteadyState_more_sliding_A40_10500yr_ramp4000_1refine_grid.npz"
else
    echo "CASE must be A20 or A40, got ${CASE}" >&2
    exit 1
fi

if [[ ! -f "${CFG}" ]]; then
    echo "Missing cfg: ${CFG}" >&2
    exit 1
fi
if [[ ! -f "${NPZ}" ]]; then
    echo "Missing MISMIP NPZ: ${NPZ}" >&2
    exit 1
fi

if [[ -f "${CKPT_DIR}/model_best.pt" ]]; then
    CKPT_CHOICE="best"
elif [[ -f "${CKPT_DIR}/model.pt" ]]; then
    CKPT_CHOICE="latest"
else
    echo "Missing Totten unfreeze_float0 checkpoint under ${CKPT_DIR}/" >&2
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
echo "CFG=${CFG} TAG=${TAG}"
echo "CKPT_DIR=${CKPT_DIR} choice=${CKPT_CHOICE}"
echo "NPZ=${NPZ}"

python evaluate_vi_only_posterior.py "${CFG}" \
    --tag "${TAG}" \
    --checkpoint "${CKPT_CHOICE}" \
    --output-dir "outputs/figures/vi_only/${TAG}"

python predict_vi_only_torch.py "${CFG}"

echo "Done."
echo "  figures: outputs/figures/vi_only/${TAG}/"
echo "  samples: outputs/totten/c_sensitivity/unfreeze_float0_xfer/mismip_${CASE}/posterior_samples.h5"
