#!/bin/bash
# Submit the full Totten C-sensitivity suite (or a subset).
#
#   cd Archive
#   python scripts/generate_totten_c_sensitivity_cfgs.py
#   bash slurm/submit_totten_c_sensitivity_suite.sh
#   bash slurm/submit_totten_c_sensitivity_suite.sh phys_w5 freeze_shift

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ARCHIVE_DIR}"

python3 scripts/generate_totten_c_sensitivity_cfgs.py

if [[ $# -gt 0 ]]; then
    EXPS=("$@")
else
    EXPS=(baseline phys_w2 phys_w5 phys_w10 freeze_shift gp_short_ls gp_flex)
fi

echo "Submitting experiments: ${EXPS[*]}"
for EXP in "${EXPS[@]}"; do
    NO_CFG="configs/totten/c_sensitivity/${EXP}/run_torch_vi_only_totten_no_sliding.cfg"
    MAX_CFG="configs/totten/c_sensitivity/${EXP}/run_torch_vi_only_totten_max_sliding.cfg"
    [[ -f "${NO_CFG}" && -f "${MAX_CFG}" ]] || { echo "missing cfgs for ${EXP}"; exit 1; }
    NO=$(sbatch --parsable slurm/vi_train_vi_only_cfg.sbatch "${NO_CFG}")
    MAX=$(sbatch --parsable slurm/vi_train_vi_only_cfg.sbatch "${MAX_CFG}")
    EVAL=$(sbatch --parsable --dependency=afterok:${NO}:${MAX} \
        slurm/vi_eval_totten_c_sensitivity.sbatch "${EXP}")
    echo "${EXP}: NO=${NO} MAX=${MAX} EVAL=${EVAL}"
done
