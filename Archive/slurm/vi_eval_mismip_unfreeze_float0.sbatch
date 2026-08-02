#!/bin/bash
#SBATCH --job-name=vi_uf0_ev
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Evaluate + predict after MISMIP unfreeze_float0 VI train.
#   sbatch --dependency=afterok:<TRAIN> slurm/vi_eval_mismip_unfreeze_float0.sbatch A20

set -euo pipefail

CASE="${1:?Usage: sbatch vi_eval_mismip_unfreeze_float0.sbatch A20|A40}"

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

CFG="configs/mismip/unfreeze_float0/run_torch_vi_only_${CASE}.cfg"
TAG="mismip_unfreeze_float0_${CASE}"
CKPT_DIR="checkpoints/torch_vi_only/mismip/unfreeze_float0/more_sliding_${CASE}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs outputs/figures/vi_only outputs/mismip/unfreeze_float0

if [[ ! -f "${CFG}" ]]; then
    echo "Missing cfg: ${CFG}" >&2
    exit 1
fi

if [[ -f "${CKPT_DIR}/model_best.pt" ]]; then
    CKPT_CHOICE="best"
elif [[ -f "${CKPT_DIR}/model.pt" ]]; then
    CKPT_CHOICE="latest"
else
    echo "Missing VI checkpoint under ${CKPT_DIR}/" >&2
    exit 1
fi

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
echo "CFG=${CFG} TAG=${TAG} checkpoint=${CKPT_CHOICE}"

python evaluate_vi_only_posterior.py "${CFG}" \
    --tag "${TAG}" \
    --checkpoint "${CKPT_CHOICE}" \
    --output-dir "outputs/figures/vi_only/${TAG}"

python predict_vi_only_torch.py "${CFG}"

echo "Done."
echo "  figures: outputs/figures/vi_only/${TAG}/"
echo "  samples: outputs/mismip/unfreeze_float0/more_sliding_${CASE}/posterior_samples.h5"
