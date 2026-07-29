#!/bin/bash
#SBATCH --job-name=vi_c_sens_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Analyze one C-sensitivity experiment after no+max VI finish.
#   sbatch --dependency=afterok:NO:MAX slurm/vi_eval_totten_c_sensitivity.sbatch phys_w5

set -euo pipefail

TAG="${1:?Usage: sbatch vi_eval_totten_c_sensitivity.sbatch <experiment_tag>}"

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm

# Activate pytorch env (DSI: ~/.conda/envs/pytorch).
if [[ -f "${HOME}/.conda/envs/${TORCH_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.conda/envs/${TORCH_ENV}/bin/activate"
elif [[ -f "${TORCH_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${TORCH_ENV}/bin/activate"
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${TORCH_ENV}"
elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${TORCH_ENV}"
fi

NO_CFG="configs/totten/c_sensitivity/${TAG}/run_torch_vi_only_totten_no_sliding.cfg"
MAX_CFG="configs/totten/c_sensitivity/${TAG}/run_torch_vi_only_totten_max_sliding.cfg"
OUT_ABS="$(cd "${ARCHIVE_DIR}/.." && pwd)/outputs/figures/vi/totten_c_sensitivity/${TAG}"
ROOT_ABS="$(cd "${ARCHIVE_DIR}/.." && pwd)/outputs/figures/vi/totten_c_sensitivity"

echo "TAG=${TAG} python=$(command -v python || true) python3=$(command -v python3 || true)"
python -c "import torch; print('torch', torch.__version__)" 2>/dev/null \
  || python3 -c "import torch; print('torch', torch.__version__)"

PYTHON_BIN="$(command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]] || ! "${PYTHON_BIN}" -c "import torch" 2>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
fi

"${PYTHON_BIN}" ../scripts/totten_c_sensitivity_analyze.py \
    "${NO_CFG}" "${MAX_CFG}" \
    --tag "${TAG}" \
    --checkpoint latest \
    --device cpu \
    --output-dir "${OUT_ABS}"

"${PYTHON_BIN}" ../scripts/totten_c_sensitivity_report.py \
    --root "${ROOT_ABS}"

echo "Done ${TAG} → ${OUT_ABS}"
