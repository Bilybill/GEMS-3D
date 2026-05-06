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
read -r -a CLASSIFICATION_PARALLEL_DEVICES <<< "$(resolve_parallel_devices "${CLASSIFICATION_DEVICE}")"
CLASSIFICATION_MAX_PARALLEL="$(normalize_parallel_count "${CLASSIFICATION_MAX_PARALLEL:-0}" "${#CLASSIFICATION_PARALLEL_DEVICES[@]}")"

echo "[classification-suite] project_root=${PROJECT_ROOT}"
echo "[classification-suite] config=${CONFIG_PATH}"
echo "[classification-suite] parallel_devices=${CLASSIFICATION_PARALLEL_DEVICES[*]}"
echo "[classification-suite] max_parallel=${CLASSIFICATION_MAX_PARALLEL}"
echo "[classification-suite] num_workers=${CLASSIFICATION_NUM_WORKERS}"
echo "[classification-suite] oom_retry=${OOM_RETRY_ENABLED:-1}"
echo "[classification-suite] logs=${SUITE_LOG_DIR}"

job_index=0
failed=0

for spec in "${CLASSIFICATION_RUN_SPECS[@]}"; do
  IFS='|' read -r run_name modalities_string batch_size <<< "${spec}"
  if ! matches_optional_filter "${run_name}" "${RUN_FILTER:-}"; then
    continue
  fi

  read -r -a modalities <<< "${modalities_string}"
  output_dir="${CLASSIFICATION_OUTPUT_ROOT}/$(IFS=_; echo "${modalities[*]}")"
  log_path="${SUITE_LOG_DIR}/${run_name}.log"
  eval_json="${output_dir}/heldout_eval.json"
  job_device="${CLASSIFICATION_PARALLEL_DEVICES[$((job_index % ${#CLASSIFICATION_PARALLEL_DEVICES[@]}))]}"
  job_index=$((job_index + 1))

  train_cmd=(
    python
    experiments/train_classification.py
    --root "${CLASSIFICATION_ROOT}"
    --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
    --modalities "${modalities[@]}"
    --device "${job_device}"
    --batch-size "${batch_size}"
    --epochs "${CLASSIFICATION_EPOCHS}"
    --lr "${CLASSIFICATION_LR}"
    --weight-decay "${CLASSIFICATION_WEIGHT_DECAY}"
    --num-workers "${CLASSIFICATION_NUM_WORKERS}"
    --prefetch-factor "${CLASSIFICATION_PREFETCH_FACTOR}"
    --val-fraction "${CLASSIFICATION_VAL_FRACTION}"
    --seed "${CLASSIFICATION_SEED}"
    --embedding-dim "${CLASSIFICATION_EMBEDDING_DIM}"
    --fusion-hidden-dim "${CLASSIFICATION_FUSION_HIDDEN_DIM}"
    --max-train-samples "${CLASSIFICATION_MAX_TRAIN_SAMPLES}"
    --max-val-samples "${CLASSIFICATION_MAX_VAL_SAMPLES}"
    --max-heldout-samples "${CLASSIFICATION_MAX_HELDOUT_SAMPLES}"
    --output-root "${CLASSIFICATION_OUTPUT_ROOT}"
  )
  eval_cmd=(
    python
    experiments/eval_classification.py
    --checkpoint "${output_dir}/best_model.pt"
    --root "${CLASSIFICATION_ROOT}"
    --device "${job_device}"
    --batch-size "${batch_size}"
    --num-workers "${CLASSIFICATION_NUM_WORKERS}"
    --prefetch-factor "${CLASSIFICATION_PREFETCH_FACTOR}"
    --val-fraction "${CLASSIFICATION_VAL_FRACTION}"
    --seed "${CLASSIFICATION_SEED}"
    --split heldout
  )
  if ! wait_for_available_slot "${CLASSIFICATION_MAX_PARALLEL}"; then
    failed=1
  fi
  (
    run_logged_command_with_oom_retries \
      "${log_path}" \
      "${batch_size}" \
      "${CLASSIFICATION_OOM_RETRY_BATCH_SIZES:-}" \
      "${train_cmd[@]}"
    eval_batch="${RUN_LOGGED_COMMAND_LAST_BATCH:-${batch_size}}"
    run_json_command_with_oom_retries \
      "${log_path}" \
      "${eval_json}" \
      "${eval_batch}" \
      "${CLASSIFICATION_OOM_RETRY_BATCH_SIZES:-}" \
      "${eval_cmd[@]}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      echo "[classification-suite] wrote held-out metrics to ${eval_json}" | tee -a "${log_path}"
    fi
  ) &
done

if ! wait_for_all_jobs; then
  failed=1
fi
exit "${failed}"
