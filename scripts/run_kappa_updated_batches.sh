#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/u/anon3/unlearn_diff"
SCRIPT="${REPO_DIR}/scripts/run_kappa_updated_prompts.py"
PROMPT_DIR="${REPO_DIR}/outputs/prompts"
OUT_DIR="${REPO_DIR}/outputs"
PARTIAL_DIR="${REPO_DIR}/outputs/kappa_partial"
LOG_DIR="${REPO_DIR}/outputs/kappa_logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${PARTIAL_DIR}"

TARGETS=(cat dog bear castle)

for t in "${TARGETS[@]}"; do
  echo "[$(date '+%F %T')] Running target=${t}"
  conda run -n py311 python "${SCRIPT}" \
    --prompt_dir "${PROMPT_DIR}" \
    --targets "${t}" \
    --output_dir "${PARTIAL_DIR}/${t}" \
    2>&1 | tee "${LOG_DIR}/kappa_${t}.log"
  echo "[$(date '+%F %T')] Completed target=${t}"
  echo

done

echo "[$(date '+%F %T')] Running final combined pass for all targets"
conda run -n py311 python "${SCRIPT}" \
  --prompt_dir "${PROMPT_DIR}" \
  --targets horse cat dog bear castle \
  --output_dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/kappa_all.log"

echo "All targets completed. Final outputs are in ${OUT_DIR}."
