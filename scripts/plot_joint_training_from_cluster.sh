#!/usr/bin/env bash
# Pull the current joint-train log from DSI and regenerate diagnostic plots locally.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE="${ROOT}/Archive"
HOST="${DSI_HOST:-login.ds}"
REMOTE="${DSI_USER:-t-9akall}@${HOST}:~/ice-dynamics/Archive"

mkdir -p "${ARCHIVE}/logs/metrics" "${ARCHIVE}/logs/figures"

echo "Pulling training log from ${REMOTE} ..."
rsync -av \
  "${REMOTE}/logs/log_train_torch_more_sliding" \
  "${ARCHIVE}/logs/log_train_torch_more_sliding"

# Optional metrics CSV directory (ignore if missing).
rsync -av --include='*.csv' --exclude='*' \
  "${REMOTE}/logs/metrics/" \
  "${ARCHIVE}/logs/metrics/" || true

cd "${ARCHIVE}"
export MPLBACKEND=Agg
python3 plot_training.py run_torch.cfg --stage joint

echo
echo "Plots written under:"
find logs -path '*figures*' -name '*.png' -newer logs/log_train_torch_more_sliding 2>/dev/null | sort
find logs -name 'recommended_losses.png' -o -name 'eta_vs_ref.png' -o -name 'grad_norms.png' 2>/dev/null | sort
