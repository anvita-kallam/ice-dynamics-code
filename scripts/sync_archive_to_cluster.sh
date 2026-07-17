#!/usr/bin/env bash
# Sync local Archive training code/config/slurm to the DSI cluster.
#
# Usage (from repo root or anywhere):
#   ./scripts/sync_archive_to_cluster.sh
#   DSI_HOST=fe01 DSI_USER=t-9akall ./scripts/sync_archive_to_cluster.sh
#
# Then on the cluster:
#   cd ~/ice-dynamics/Archive
#   mv checkpoints/torch_joint/more_sliding checkpoints/torch_joint/more_sliding_flat_eta  # if needed
#   sbatch slurm/vi_train_more_sliding.sbatch

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DSI_HOST:-login.ds}"
USER_HOST="${DSI_USER:-t-9akall}@${HOST}"
REMOTE_ARCHIVE="${DSI_REMOTE_ARCHIVE:-${USER_HOST}:~/ice-dynamics/Archive}"

echo "Syncing Archive -> ${REMOTE_ARCHIVE}"
rsync -av \
  --exclude='.matplotlib_cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='checkpoints/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  --exclude='Untitled.ipynb' \
  "${ROOT}/Archive/" \
  "${REMOTE_ARCHIVE}/"

echo
echo "Done. On the cluster:"
echo "  ssh ${USER_HOST}"
echo "  cd ~/ice-dynamics/Archive"
echo "  sbatch slurm/vi_train_more_sliding.sbatch"
