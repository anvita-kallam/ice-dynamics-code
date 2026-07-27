#!/bin/bash
# Submit Totten sequential pipeline on DSI Slurm with unit-corrected η bounds:
#   1) shared PINN pretrain (skipped if model_best exists)
#   2) VI no_sliding  (C=100)
#   3) VI max_sliding (C=0.001)  — parallel with no_sliding
#   4) predict both + comparison plots (after both VI jobs)
#
# Usage (from Archive/ on the cluster):
#   bash slurm/submit_totten_pipeline.sh
#   bash slurm/submit_totten_pipeline.sh --qos=protected

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints outputs

EXTRA_SBATCH_ARGS=("$@")
PRETRAIN_CKPT="checkpoints/torch_pretrain/totten/model_best.pt"

if [[ -f "${PRETRAIN_CKPT}" ]]; then
    echo "Found Totten pretrain ${PRETRAIN_CKPT} — skipping Stage 1."
    PRE_DEP=()
    PRE="(existing)"
else
    echo "Submitting Totten pretrain..."
    PRE=$(sbatch --parsable "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_pretrain_totten.sbatch)
    echo "  pretrain job: ${PRE}"
    PRE_DEP=(--dependency=afterok:"${PRE}")
fi

echo "Submitting Totten VI no_sliding..."
VI_NO=$(sbatch --parsable "${PRE_DEP[@]}" \
    "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_train_vi_only_totten_no_sliding.sbatch)
echo "  no_sliding job: ${VI_NO}"

echo "Submitting Totten VI max_sliding..."
VI_MAX=$(sbatch --parsable "${PRE_DEP[@]}" \
    "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_train_vi_only_totten_max_sliding.sbatch)
echo "  max_sliding job: ${VI_MAX}"

echo "Submitting Totten evaluate/predict/plot (afterok:${VI_NO},${VI_MAX})..."
EVAL=$(sbatch --parsable --dependency=afterok:"${VI_NO}":"${VI_MAX}" \
    "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_eval_predict_totten.sbatch)
echo "  eval job: ${EVAL}"

cat <<EOF

Totten pipeline queued:
  pretrain     ${PRE}
  no_sliding   ${VI_NO}  -> checkpoints/torch_vi_only/totten/no_sliding/
  max_sliding  ${VI_MAX} -> checkpoints/torch_vi_only/totten/max_sliding/
  eval/plot    ${EVAL}   -> outputs/figures/vi/totten_sliding_comparison/

eta bounds: eta_min=1e-3, eta_max=1e4 MPa·yr (converted from Pa·yr 1e3 / 1e10)

Monitor:
  squeue -u \$USER
EOF
