#!/usr/bin/env bash
# Submit raised_prior eta_init=24 / A40 more_sliding train (resume by default).
#
# Usage (from anywhere):
#   ./scripts/rerun_raised_prior24_A40.sh
#   FORCE_FRESH=1 ./scripts/rerun_raised_prior24_A40.sh
#   EVAL_ONLY=1 ./scripts/rerun_raised_prior24_A40.sh
#   ./scripts/rerun_raised_prior24_A40.sh --dependency=afterany:1320798

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DSI_HOST:-login.ds}"
USER_HOST="${DSI_USER:-t-9akall}@${HOST}"
REMOTE_ARCHIVE="${DSI_REMOTE_ARCHIVE:-~/ice-dynamics/Archive}"

EXTRA_SBATCH_ARGS=("$@")
ENV_EXPORTS=()
[[ -n "${FORCE_FRESH:-}" ]] && ENV_EXPORTS+=("FORCE_FRESH=${FORCE_FRESH}")
[[ -n "${EVAL_ONLY:-}" ]] && ENV_EXPORTS+=("EVAL_ONLY=${EVAL_ONLY}")

echo "Syncing Archive -> ${USER_HOST}:${REMOTE_ARCHIVE}"
rsync -av \
  --exclude='.matplotlib_cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='checkpoints/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  "${ROOT}/Archive/" \
  "${USER_HOST}:${REMOTE_ARCHIVE}/"

REMOTE_CMD="cd ${REMOTE_ARCHIVE}"
if ((${#ENV_EXPORTS[@]})); then
  REMOTE_CMD+=" && export ${ENV_EXPORTS[*]}"
fi
REMOTE_CMD+=" && sbatch"
if ((${#EXTRA_SBATCH_ARGS[@]})); then
  REMOTE_CMD+=" ${EXTRA_SBATCH_ARGS[*]}"
fi
REMOTE_CMD+=" slurm/vi_rerun_raised_prior24_A40.sbatch"

echo "Submitting: ${REMOTE_CMD}"
ssh "${USER_HOST}" "${REMOTE_CMD}"
