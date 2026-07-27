#!/bin/bash
#SBATCH --job-name=totten_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Totten predict (no + max sliding) + comparison plots.
#   sbatch --dependency=afterok:<VI_NO>:<VI_MAX> slurm/vi_eval_predict_totten.sbatch

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm outputs/totten outputs/figures/vi/totten_sliding_comparison

for case in no_sliding max_sliding; do
    CKPT="checkpoints/torch_vi_only/totten/${case}/model.pt"
    if [[ ! -f "${CKPT}" ]]; then
        echo "Missing Totten VI checkpoint: ${CKPT}" >&2
        exit 1
    fi
done

if [[ -f "${TORCH_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${TORCH_ENV}/bin/activate"
elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${TORCH_ENV}"
fi

echo "Predicting Totten no_sliding..."
python predict_vi_only_torch.py configs/totten/run_torch_vi_only_totten_no_sliding.cfg

echo "Predicting Totten max_sliding..."
python predict_vi_only_torch.py configs/totten/run_torch_vi_only_totten_max_sliding.cfg

echo "Plotting no vs max sliding comparison..."
python plot_totten_sliding_comparison.py \
    configs/totten/run_torch_vi_only_totten_no_sliding.cfg \
    configs/totten/run_torch_vi_only_totten_max_sliding.cfg \
    --checkpoint latest \
    --output-dir outputs/figures/vi/totten_sliding_comparison

echo "Done."
echo "  samples: outputs/totten/{no,max}_sliding/posterior_samples.h5"
echo "  figures: outputs/figures/vi/totten_sliding_comparison/"
