#!/bin/bash
#SBATCH --job-name=vi_prior24_A40
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Rerun / resume raised prior eta_init=24 on more_sliding A=40.
#
#   cd ~/ice-dynamics/Archive
#   sbatch slurm/vi_rerun_raised_prior24_A40.sbatch            # resume if model.pt exists
#   FORCE_FRESH=1 sbatch slurm/vi_rerun_raised_prior24_A40.sbatch  # train from scratch
#   EVAL_ONLY=1 sbatch slurm/vi_rerun_raised_prior24_A40.sbatch    # skip train; eval+predict only
#
# After train: evaluate best checkpoint + write posterior samples.

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

CFG_FRESH="configs/vi_only_eta_bias_suite/raised_prior_center_24_A40.cfg"
CFG_RESUME="configs/vi_only_eta_bias_suite/raised_prior_center_24_A40_resume.cfg"
CKPT_DIR="checkpoints/torch_vi_only/eta_bias_suite/eta_bias_v1/raised_prior_center_24_A40"
PRETRAIN_CKPT="checkpoints/torch_pretrain/more_sliding_sequential/model_best.pt"
TAG="raised_prior_center_24_A40"
FORCE_FRESH="${FORCE_FRESH:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints outputs

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "Missing sequential pretrain checkpoint: ${PRETRAIN_CKPT}" >&2
    exit 1
fi

if [[ "${FORCE_FRESH}" == "1" ]]; then
    CFG="${CFG_FRESH}"
    MODE="fresh"
elif [[ -f "${CKPT_DIR}/model.pt" ]]; then
    CFG="${CFG_RESUME}"
    MODE="resume"
else
    CFG="${CFG_FRESH}"
    MODE="fresh (no checkpoint)"
fi

echo "[$(date)] MODE=${MODE} CFG=${CFG} EVAL_ONLY=${EVAL_ONLY}"
echo "[$(date)] CKPT_DIR=${CKPT_DIR}"
ls -la "${CKPT_DIR}" 2>/dev/null || true

if [[ "${EVAL_ONLY}" != "1" ]]; then
    bash scripts/dsi_preempt_wrapper.sh train_vi_only_torch.py "${CFG}"
fi

# Activate env for eval/predict (wrapper already used conda; re-activate for clarity).
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${TORCH_ENV}"
elif [[ -f "${HOME}/.conda/envs/${TORCH_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.conda/envs/${TORCH_ENV}/bin/activate"
fi

if [[ -f "${CKPT_DIR}/model_best.pt" ]]; then
    CKPT_CHOICE="best"
elif [[ -f "${CKPT_DIR}/model.pt" ]]; then
    CKPT_CHOICE="latest"
else
    echo "No checkpoint under ${CKPT_DIR}/ for eval" >&2
    exit 1
fi

echo "[$(date)] evaluate checkpoint=${CKPT_CHOICE}"
python evaluate_vi_only_posterior.py "${CFG_FRESH}" \
    --tag "${TAG}" \
    --checkpoint "${CKPT_CHOICE}" \
    --output-dir "outputs/figures/vi_only/${TAG}"

echo "[$(date)] predict posterior samples"
python predict_vi_only_torch.py "${CFG_FRESH}"

echo "Done ${TAG}"
echo "  figures: outputs/figures/vi_only/${TAG}/"
echo "  samples: outputs/vi_only_eta_bias_suite/eta_bias_v1/${TAG}/posterior_samples.h5"
