#!/usr/bin/env bash
# Resume the four eta-bias suite runs that hit the 12h wall clock.
# Flips train.restore / restore_optimizer to True in a temp cfg copy.
# n_epochs stays absolute (continues until the original target epoch).
# Usage (from Archive/):
#   bash scripts/resume_eta_bias_wallkilled.sh

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ARCHIVE_DIR}"

RUNS=(
  gp_capacity
  optimizer_ngd
  optimizer_fast_cosine
  combined_candidate
)

mkdir -p logs/slurm configs/vi_only_eta_bias_suite/resume
job_ids=()

for run_id in "${RUNS[@]}"; do
  cfg="configs/vi_only_eta_bias_suite/${run_id}.cfg"
  ckpt="checkpoints/torch_vi_only/eta_bias_suite/eta_bias_v1/${run_id}/model.pt"
  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}" >&2
    exit 1
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint to resume: ${ckpt}" >&2
    exit 1
  fi
  resume_cfg="configs/vi_only_eta_bias_suite/resume/${run_id}_restore.cfg"
  python3 - "${cfg}" "${resume_cfg}" <<'PY'
from pathlib import Path
import sys
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
lines, out, in_train = text.splitlines(keepends=True), [], False
for line in lines:
    stripped = line.strip()
    if stripped.startswith('[') and stripped.endswith(']'):
        in_train = stripped == '[train]'
        out.append(line)
        continue
    if in_train and stripped.startswith('restore ='):
        out.append('restore = True\n')
        continue
    if in_train and stripped.startswith('restore_optimizer ='):
        out.append('restore_optimizer = True\n')
        continue
    out.append(line)
dst.write_text(''.join(out))
PY
  job_id="$(sbatch --parsable \
    --job-name="eta_rs_${run_id:0:8}" \
    --time=12:00:00 \
    slurm/vi_only_eta_bias_trial.sbatch \
    "${resume_cfg}" "${run_id}")"
  job_id="${job_id%%;*}"
  job_ids+=("${job_id}")
  echo "${run_id}: ${job_id} (cfg=${resume_cfg})"
done

dependency="afterany:$(IFS=:; echo "${job_ids[*]}")"
collector="$(sbatch --parsable \
  --dependency="${dependency}" \
  --job-name=eta_rs_collect \
  slurm/vi_only_eta_bias_collect.sbatch \
  configs/vi_only_eta_bias_suite/manifest.json)"
collector="${collector%%;*}"
echo "collector: ${collector}"
echo "Monitor with: squeue -u \$USER"
