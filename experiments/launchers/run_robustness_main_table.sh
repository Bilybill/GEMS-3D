#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

activate_conda_env_if_needed
cd "${PROJECT_ROOT}"

ROBUST_ROOT="${ROBUST_ROOT:-./outputs/PretrainDataset_forward}"
COMPLETION_OUTPUT_ROOT="${COMPLETION_OUTPUT_ROOT:-./outputs/ExperimentRuns/modality_completion}"
INVERSE_OUTPUT_ROOT="${INVERSE_OUTPUT_ROOT:-./outputs/ExperimentRuns/inverse_regression}"
ROBUST_OUTPUT_ROOT="${ROBUST_OUTPUT_ROOT:-./outputs/ExperimentRuns/robustness}"
ROBUST_LOG_ROOT="${ROBUST_LOG_ROOT:-${ROBUST_OUTPUT_ROOT}/logs}"
ROBUST_DEVICE="${ROBUST_DEVICE:-${DEVICE:-cuda}}"
ROBUST_BATCH_SIZE="${ROBUST_BATCH_SIZE:-4}"
ROBUST_COMPLETION_BATCH_SIZE="${ROBUST_COMPLETION_BATCH_SIZE:-${ROBUST_BATCH_SIZE}}"
ROBUST_INVERSION_EVAL_BATCH_SIZE="${ROBUST_INVERSION_EVAL_BATCH_SIZE:-2}"
ROBUST_EPOCHS="${ROBUST_EPOCHS:-30}"
ROBUST_LR="${ROBUST_LR:-1e-3}"
ROBUST_WEIGHT_DECAY="${ROBUST_WEIGHT_DECAY:-1e-4}"
ROBUST_NUM_WORKERS="${ROBUST_NUM_WORKERS:-12}"
ROBUST_PREFETCH_FACTOR="${ROBUST_PREFETCH_FACTOR:-2}"
ROBUST_VAL_FRACTION="${ROBUST_VAL_FRACTION:-0.15}"
ROBUST_SEED="${ROBUST_SEED:-42}"
ROBUST_MAX_TRAIN_SAMPLES="${ROBUST_MAX_TRAIN_SAMPLES:-0}"
ROBUST_MAX_VAL_SAMPLES="${ROBUST_MAX_VAL_SAMPLES:-0}"
ROBUST_MAX_HELDOUT_SAMPLES="${ROBUST_MAX_HELDOUT_SAMPLES:-0}"
ROBUST_MODELS=(${ROBUST_MODELS:-unet fno})
COMPLETION_TARGETS=(${COMPLETION_TARGETS:-gravity magnetic mt seismic})
INCLUDE_TOP_LEVELS=(${INCLUDE_TOP_LEVELS:-train-river tests-river})

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SUITE_LOG_DIR="${ROBUST_LOG_ROOT}/${TIMESTAMP}"
ensure_directory "${SUITE_LOG_DIR}"
ensure_directory "${ROBUST_OUTPUT_ROOT}"
read -r -a ROBUST_PARALLEL_DEVICES <<< "$(resolve_parallel_devices "${ROBUST_DEVICE}")"
ROBUST_MAX_PARALLEL="$(normalize_parallel_count "${ROBUST_MAX_PARALLEL:-0}" "${#ROBUST_PARALLEL_DEVICES[@]}")"

completion_inputs() {
  local target="$1"
  local modalities=(gravity magnetic mt seismic)
  local selected=()
  local modality
  for modality in "${modalities[@]}"; do
    if [[ "${modality}" != "${target}" ]]; then
      selected+=("${modality}")
    fi
  done
  echo "${selected[*]}"
}

echo "[robustness-suite] project_root=${PROJECT_ROOT}"
echo "[robustness-suite] root=${ROBUST_ROOT}"
echo "[robustness-suite] completion_output_root=${COMPLETION_OUTPUT_ROOT}"
echo "[robustness-suite] robustness_output_root=${ROBUST_OUTPUT_ROOT}"
echo "[robustness-suite] parallel_devices=${ROBUST_PARALLEL_DEVICES[*]}"
echo "[robustness-suite] max_parallel=${ROBUST_MAX_PARALLEL}"
echo "[robustness-suite] completion_batch_size=${ROBUST_COMPLETION_BATCH_SIZE}"
echo "[robustness-suite] inversion_eval_batch_size=${ROBUST_INVERSION_EVAL_BATCH_SIZE}"
echo "[robustness-suite] num_workers=${ROBUST_NUM_WORKERS}"
echo "[robustness-suite] skip_completed=${SKIP_COMPLETED:-1}"
echo "[robustness-suite] oom_retry=${OOM_RETRY_ENABLED:-1}"

job_index=0
failed=0

for target in "${COMPLETION_TARGETS[@]}"; do
  if ! matches_optional_filter "missing_${target}" "${TASK_FILTER:-}"; then
    continue
  fi
  read -r -a inputs <<< "$(completion_inputs "${target}")"
  for model in "${ROBUST_MODELS[@]}"; do
    if ! matches_optional_filter "${model}" "${MODEL_FILTER:-}"; then
      continue
    fi
    run_name="missing_${target}_${model}"
    log_path="${SUITE_LOG_DIR}/${run_name}.log"
    output_dir="${COMPLETION_OUTPUT_ROOT}/$(IFS=_; echo "${inputs[*]}")_to_${target}_${model}"
    checkpoint_path="${output_dir}/best_model.pt"
    eval_json="${output_dir}/heldout_eval.json"
    job_device="${ROBUST_PARALLEL_DEVICES[$((job_index % ${#ROBUST_PARALLEL_DEVICES[@]}))]}"
    job_index=$((job_index + 1))
    train_cmd=(
      python experiments/train_modality_completion.py
      --root "${ROBUST_ROOT}"
      --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
      --input-modalities "${inputs[@]}"
      --target-modality "${target}"
      --model "${model}"
      --device "${job_device}"
      --batch-size "${ROBUST_COMPLETION_BATCH_SIZE}"
      --epochs "${ROBUST_EPOCHS}"
      --lr "${ROBUST_LR}"
      --weight-decay "${ROBUST_WEIGHT_DECAY}"
      --num-workers "${ROBUST_NUM_WORKERS}"
      --prefetch-factor "${ROBUST_PREFETCH_FACTOR}"
      --val-fraction "${ROBUST_VAL_FRACTION}"
      --seed "${ROBUST_SEED}"
      --max-train-samples "${ROBUST_MAX_TRAIN_SAMPLES}"
      --max-val-samples "${ROBUST_MAX_VAL_SAMPLES}"
      --max-heldout-samples "${ROBUST_MAX_HELDOUT_SAMPLES}"
      --output-root "${COMPLETION_OUTPUT_ROOT}"
    )
    eval_cmd=(
      python experiments/eval_modality_completion.py
      --checkpoint "${checkpoint_path}"
      --root "${ROBUST_ROOT}"
      --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
      --device "${job_device}"
      --batch-size "${ROBUST_COMPLETION_BATCH_SIZE}"
      --num-workers "${ROBUST_NUM_WORKERS}"
      --prefetch-factor "${ROBUST_PREFETCH_FACTOR}"
      --val-fraction "${ROBUST_VAL_FRACTION}"
      --seed "${ROBUST_SEED}"
      --split heldout
    )
    if skip_completed_enabled && completed_eval_exists "${eval_json}"; then
      log_launcher_message "${log_path}" "[robustness-suite] skip ${run_name}; found ${eval_json}"
      continue
    fi

    run_training=1
    if skip_completed_enabled && completed_file_exists "${checkpoint_path}"; then
      run_training=0
    fi

    if ! wait_for_available_slot "${ROBUST_MAX_PARALLEL}"; then
      failed=1
    fi
    (
      eval_batch="${ROBUST_COMPLETION_BATCH_SIZE}"
      if [[ "${run_training}" == "1" ]]; then
        run_logged_command_with_oom_retries \
          "${log_path}" \
          "${ROBUST_COMPLETION_BATCH_SIZE}" \
          "${ROBUST_OOM_RETRY_BATCH_SIZES:-}" \
          "${train_cmd[@]}"
        eval_batch="${RUN_LOGGED_COMMAND_LAST_BATCH:-${ROBUST_COMPLETION_BATCH_SIZE}}"
      else
        log_launcher_message "${log_path}" "[robustness-suite] reuse ${checkpoint_path}; only running completion eval"
      fi
      run_json_command_with_oom_retries \
        "${log_path}" \
        "${eval_json}" \
        "${eval_batch}" \
        "${ROBUST_OOM_RETRY_BATCH_SIZES:-}" \
        "${eval_cmd[@]}"
      if [[ "${DRY_RUN:-0}" != "1" ]]; then
        echo "[robustness-suite] wrote completion metrics to ${eval_json}" | tee -a "${log_path}"
      fi
    ) &
  done
done

if ! wait_for_all_jobs; then
  failed=1
fi

for model in "${ROBUST_MODELS[@]}"; do
  if ! matches_optional_filter "degraded_${model}" "${TASK_FILTER:-}"; then
    continue
  fi
  if ! matches_optional_filter "${model}" "${MODEL_FILTER:-}"; then
    continue
  fi
  checkpoint="${INVERSE_OUTPUT_ROOT}/gravity_magnetic_mt_seismic_to_vp_rho_phi_chi_${model}/best_model.pt"
  output_json="${ROBUST_OUTPUT_ROOT}/degraded_inversion_${model}.json"
  log_path="${SUITE_LOG_DIR}/degraded_inversion_${model}.log"
  job_device="${ROBUST_PARALLEL_DEVICES[$((job_index % ${#ROBUST_PARALLEL_DEVICES[@]}))]}"
  job_index=$((job_index + 1))
  eval_cmd=(
    python experiments/eval_degraded_inversion.py
    --checkpoint "${checkpoint}"
    --root "${ROBUST_ROOT}"
    --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
    --device "${job_device}"
    --batch-size "${ROBUST_INVERSION_EVAL_BATCH_SIZE}"
    --num-workers "${ROBUST_NUM_WORKERS}"
    --prefetch-factor "${ROBUST_PREFETCH_FACTOR}"
    --val-fraction "${ROBUST_VAL_FRACTION}"
    --seed "${ROBUST_SEED}"
    --split heldout
  )
  if skip_completed_enabled && completed_eval_exists "${output_json}"; then
    log_launcher_message "${log_path}" "[robustness-suite] skip degraded inversion for ${model}; found ${output_json}"
    continue
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    if ! wait_for_available_slot "${ROBUST_MAX_PARALLEL}"; then
      failed=1
    fi
    run_json_command_with_oom_retries \
      "${log_path}" \
      "${output_json}" \
      "${ROBUST_INVERSION_EVAL_BATCH_SIZE}" \
      "${ROBUST_OOM_RETRY_BATCH_SIZES:-}" \
      "${eval_cmd[@]}" &
    continue
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[robustness-suite] skip degraded inversion for ${model}; missing checkpoint ${checkpoint}" | tee -a "${log_path}"
    continue
  fi
  if ! wait_for_available_slot "${ROBUST_MAX_PARALLEL}"; then
    failed=1
  fi
  (
    run_json_command_with_oom_retries \
      "${log_path}" \
      "${output_json}" \
      "${ROBUST_INVERSION_EVAL_BATCH_SIZE}" \
      "${ROBUST_OOM_RETRY_BATCH_SIZES:-}" \
      "${eval_cmd[@]}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      echo "[robustness-suite] wrote degraded metrics to ${output_json}" | tee -a "${log_path}"
    fi
  ) &
done

if ! wait_for_all_jobs; then
  failed=1
fi
exit "${failed}"
