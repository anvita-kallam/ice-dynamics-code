#!/bin/bash
#SBATCH --job-name=vi_eval_ms
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Evaluate + predict after MISMIP more_sliding optimized VI.
#   sbatch --dependency=afterok:<VI> slurm/vi_eval_predict_more_sliding_optimized.sbatch

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

CFG="${1:-run_torch_vi_only_optimized.cfg}"
TAG="${2:-more_sliding_optimized}"
CKPT_DIR="checkpoints/torch_vi_only/more_sliding_optimized"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs outputs/figures/vi_only

if [[ -f "${CKPT_DIR}/model_best.pt" ]]; then
    CKPT_CHOICE="best"
elif [[ -f "${CKPT_DIR}/model.pt" ]]; then
    CKPT_CHOICE="latest"
else
    echo "Missing VI checkpoint under ${CKPT_DIR}/" >&2
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

echo "CFG=${CFG} TAG=${TAG} checkpoint=${CKPT_CHOICE}"

python evaluate_vi_only_posterior.py "${CFG}" \
    --tag "${TAG}" \
    --checkpoint "${CKPT_CHOICE}" \
    --output-dir "outputs/figures/vi_only/${TAG}"

python predict_vi_only_torch.py "${CFG}"

echo "Done."
echo "  figures: outputs/figures/vi_only/${TAG}/"
echo "  samples: (see predict.output_file in ${CFG})"
