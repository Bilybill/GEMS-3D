#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/experiments/configs/classification/default_suite.sh}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1090
source "${CONFIG_PATH}"

activate_conda_env_if_needed
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SUITE_LOG_DIR="${CLASSIFICATION_LOG_ROOT}/${TIMESTAMP}"
ensure_directory "${SUITE_LOG_DIR}"

echo "[classification-suite] project_root=${PROJECT_ROOT}"
echo "[classification-suite] config=${CONFIG_PATH}"
echo "[classification-suite] logs=${SUITE_LOG_DIR}"

for spec in "${CLASSIFICATION_RUN_SPECS[@]}"; do
  IFS='|' read -r run_name modalities_string batch_size <<< "${spec}"
  if ! matches_optional_filter "${run_name}" "${RUN_FILTER:-}"; then
    continue
  fi

  read -r -a modalities <<< "${modalities_string}"
  output_dir="${CLASSIFICATION_OUTPUT_ROOT}/$(IFS=_; echo "${modalities[*]}")"
  log_path="${SUITE_LOG_DIR}/${run_name}.log"
  eval_json="${output_dir}/heldout_eval.json"

  train_cmd=(
    python
    experiments/train_classification.py
    --root "${CLASSIFICATION_ROOT}"
    --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
    --modalities "${modalities[@]}"
    --device "${CLASSIFICATION_DEVICE}"
    --batch-size "${batch_size}"
    --epochs "${CLASSIFICATION_EPOCHS}"
    --lr "${CLASSIFICATION_LR}"
    --weight-decay "${CLASSIFICATION_WEIGHT_DECAY}"
    --num-workers "${CLASSIFICATION_NUM_WORKERS}"
    --val-fraction "${CLASSIFICATION_VAL_FRACTION}"
    --seed "${CLASSIFICATION_SEED}"
    --embedding-dim "${CLASSIFICATION_EMBEDDING_DIM}"
    --fusion-hidden-dim "${CLASSIFICATION_FUSION_HIDDEN_DIM}"
    --max-train-samples "${CLASSIFICATION_MAX_TRAIN_SAMPLES}"
    --max-val-samples "${CLASSIFICATION_MAX_VAL_SAMPLES}"
    --max-heldout-samples "${CLASSIFICATION_MAX_HELDOUT_SAMPLES}"
    --output-root "${CLASSIFICATION_OUTPUT_ROOT}"
  )
  run_logged_command "${log_path}" "${train_cmd[@]}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    continue
  fi

  eval_cmd=(
    python
    experiments/eval_classification.py
    --checkpoint "${output_dir}/best_model.pt"
    --root "${CLASSIFICATION_ROOT}"
    --device "${CLASSIFICATION_DEVICE}"
    --batch-size "${batch_size}"
    --num-workers "${CLASSIFICATION_NUM_WORKERS}"
    --val-fraction "${CLASSIFICATION_VAL_FRACTION}"
    --seed "${CLASSIFICATION_SEED}"
    --split heldout
  )
  "${eval_cmd[@]}" > "${eval_json}"
  echo "[classification-suite] wrote held-out metrics to ${eval_json}" | tee -a "${log_path}"
done
