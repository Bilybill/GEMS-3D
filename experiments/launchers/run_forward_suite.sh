#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/experiments/configs/forward/default_suite.sh}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1090
source "${CONFIG_PATH}"

activate_conda_env_if_needed
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SUITE_LOG_DIR="${FORWARD_LOG_ROOT}/${TIMESTAMP}"
ensure_directory "${SUITE_LOG_DIR}"

echo "[forward-suite] project_root=${PROJECT_ROOT}"
echo "[forward-suite] config=${CONFIG_PATH}"
echo "[forward-suite] device=${FORWARD_DEVICE}"
echo "[forward-suite] logs=${SUITE_LOG_DIR}"

for task in "${FORWARD_TASKS[@]}"; do
  if ! matches_optional_filter "${task}" "${TASK_FILTER:-}"; then
    continue
  fi

  batch_size="${FORWARD_BATCH_SIZE[$task]}"
  epochs="${FORWARD_EPOCHS[$task]}"
  lr="${FORWARD_LR[$task]}"

  models_for_task=("${FORWARD_MODELS[@]}")
  if declare -p FORWARD_TASK_MODELS >/dev/null 2>&1; then
    task_specific_models="${FORWARD_TASK_MODELS[$task]:-}"
    if [[ -n "${task_specific_models}" ]]; then
      read -r -a models_for_task <<< "${task_specific_models}"
    fi
  fi

  for model in "${models_for_task[@]}"; do
    if ! matches_optional_filter "${model}" "${MODEL_FILTER:-}"; then
      continue
    fi

    run_name="${task}_${model}"
    log_path="${SUITE_LOG_DIR}/${run_name}.log"
    output_dir="${FORWARD_OUTPUT_ROOT}/${run_name}"
    eval_json="${output_dir}/heldout_eval.json"

    train_cmd=(
      python
      experiments/train_forward_surrogate.py
      --root "${FORWARD_ROOT}"
      --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
      --task "${task}"
      --model "${model}"
      --device "${FORWARD_DEVICE}"
      --batch-size "${batch_size}"
      --epochs "${epochs}"
      --lr "${lr}"
      --weight-decay "${FORWARD_WEIGHT_DECAY}"
      --num-workers "${FORWARD_NUM_WORKERS}"
      --val-fraction "${FORWARD_VAL_FRACTION}"
      --seed "${FORWARD_SEED}"
      --max-train-samples "${FORWARD_MAX_TRAIN_SAMPLES}"
      --max-val-samples "${FORWARD_MAX_VAL_SAMPLES}"
      --max-heldout-samples "${FORWARD_MAX_HELDOUT_SAMPLES}"
      --output-root "${FORWARD_OUTPUT_ROOT}"
    )
    if [[ "${#FORWARD_DEVELOPMENT_SOURCE_PREFIXES[@]}" -gt 0 ]]; then
      train_cmd+=(--development-source-prefixes "${FORWARD_DEVELOPMENT_SOURCE_PREFIXES[@]}")
    fi
    if [[ "${#FORWARD_HELDOUT_SOURCE_PREFIXES[@]}" -gt 0 ]]; then
      train_cmd+=(--heldout-source-prefixes "${FORWARD_HELDOUT_SOURCE_PREFIXES[@]}")
    fi
    run_logged_command "${log_path}" "${train_cmd[@]}"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      continue
    fi

    eval_cmd=(
      python
      experiments/eval_forward.py
      --checkpoint "${output_dir}/best_model.pt"
      --root "${FORWARD_ROOT}"
      --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
      --device "${FORWARD_DEVICE}"
      --batch-size "${batch_size}"
      --num-workers "${FORWARD_NUM_WORKERS}"
      --val-fraction "${FORWARD_VAL_FRACTION}"
      --seed "${FORWARD_SEED}"
      --split heldout
    )
    if [[ "${#FORWARD_DEVELOPMENT_SOURCE_PREFIXES[@]}" -gt 0 ]]; then
      eval_cmd+=(--development-source-prefixes "${FORWARD_DEVELOPMENT_SOURCE_PREFIXES[@]}")
    fi
    if [[ "${#FORWARD_HELDOUT_SOURCE_PREFIXES[@]}" -gt 0 ]]; then
      eval_cmd+=(--heldout-source-prefixes "${FORWARD_HELDOUT_SOURCE_PREFIXES[@]}")
    fi
    "${eval_cmd[@]}" > "${eval_json}"
    echo "[forward-suite] wrote held-out metrics to ${eval_json}" | tee -a "${log_path}"
  done
done
