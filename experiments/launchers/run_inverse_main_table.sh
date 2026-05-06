#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

activate_conda_env_if_needed
cd "${PROJECT_ROOT}"

INVERSE_ROOT="${INVERSE_ROOT:-./outputs/PretrainDataset_forward}"
INVERSE_OUTPUT_ROOT="${INVERSE_OUTPUT_ROOT:-./outputs/ExperimentRuns/inverse_regression}"
INVERSE_LOG_ROOT="${INVERSE_LOG_ROOT:-${INVERSE_OUTPUT_ROOT}/logs}"
INVERSE_DEVICE="${INVERSE_DEVICE:-${DEVICE:-cuda}}"
INVERSE_BATCH_SIZE="${INVERSE_BATCH_SIZE:-2}"
INVERSE_EPOCHS="${INVERSE_EPOCHS:-30}"
INVERSE_LR="${INVERSE_LR:-1e-3}"
INVERSE_WEIGHT_DECAY="${INVERSE_WEIGHT_DECAY:-1e-4}"
INVERSE_NUM_WORKERS="${INVERSE_NUM_WORKERS:-12}"
INVERSE_PREFETCH_FACTOR="${INVERSE_PREFETCH_FACTOR:-2}"
INVERSE_VAL_FRACTION="${INVERSE_VAL_FRACTION:-0.15}"
INVERSE_SEED="${INVERSE_SEED:-42}"
INVERSE_TARGET_SHAPE="${INVERSE_TARGET_SHAPE:-64 64 64}"
INVERSE_MAX_TRAIN_SAMPLES="${INVERSE_MAX_TRAIN_SAMPLES:-0}"
INVERSE_MAX_VAL_SAMPLES="${INVERSE_MAX_VAL_SAMPLES:-0}"
INVERSE_MAX_HELDOUT_SAMPLES="${INVERSE_MAX_HELDOUT_SAMPLES:-0}"
INCLUDE_TOP_LEVELS=(${INCLUDE_TOP_LEVELS:-train-river tests-river})
INVERSE_MODELS=(${INVERSE_MODELS:-unet pinn deeponet fno gnot})
INVERSE_TASKS=(${INVERSE_TASKS:-dg_to_rho dm_to_chi dmt_to_phi ds_to_vp dg_dm_to_rho_chi dg_dm_dmt_to_rho_chi_phi all_to_all})

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SUITE_LOG_DIR="${INVERSE_LOG_ROOT}/${TIMESTAMP}"
ensure_directory "${SUITE_LOG_DIR}"
read -r -a INVERSE_PARALLEL_DEVICES <<< "$(resolve_parallel_devices "${INVERSE_DEVICE}")"
INVERSE_MAX_PARALLEL="$(normalize_parallel_count "${INVERSE_MAX_PARALLEL:-0}" "${#INVERSE_PARALLEL_DEVICES[@]}")"

task_modalities() {
  case "$1" in
    dg_to_rho) echo "gravity" ;;
    dm_to_chi) echo "magnetic" ;;
    dmt_to_phi) echo "mt" ;;
    ds_to_vp) echo "seismic" ;;
    dg_dm_to_rho_chi) echo "gravity magnetic" ;;
    dg_dm_dmt_to_rho_chi_phi) echo "gravity magnetic mt" ;;
    all_to_all) echo "gravity magnetic mt seismic" ;;
    *) echo "Unsupported inverse task '$1'" >&2; return 1 ;;
  esac
}

task_targets() {
  case "$1" in
    dg_to_rho) echo "rho" ;;
    dm_to_chi) echo "chi" ;;
    dmt_to_phi) echo "phi" ;;
    ds_to_vp) echo "vp" ;;
    dg_dm_to_rho_chi) echo "rho chi" ;;
    dg_dm_dmt_to_rho_chi_phi) echo "rho chi phi" ;;
    all_to_all) echo "vp rho phi chi" ;;
    *) echo "Unsupported inverse task '$1'" >&2; return 1 ;;
  esac
}

echo "[inverse-suite] project_root=${PROJECT_ROOT}"
echo "[inverse-suite] root=${INVERSE_ROOT}"
echo "[inverse-suite] output_root=${INVERSE_OUTPUT_ROOT}"
echo "[inverse-suite] parallel_devices=${INVERSE_PARALLEL_DEVICES[*]}"
echo "[inverse-suite] max_parallel=${INVERSE_MAX_PARALLEL}"
echo "[inverse-suite] batch_size=${INVERSE_BATCH_SIZE}"
echo "[inverse-suite] num_workers=${INVERSE_NUM_WORKERS}"
echo "[inverse-suite] skip_completed=${SKIP_COMPLETED:-1}"
echo "[inverse-suite] oom_retry=${OOM_RETRY_ENABLED:-1}"
echo "[inverse-suite] logs=${SUITE_LOG_DIR}"

job_index=0
failed=0

for task in "${INVERSE_TASKS[@]}"; do
  if ! matches_optional_filter "${task}" "${TASK_FILTER:-}"; then
    continue
  fi
  read -r -a modalities <<< "$(task_modalities "${task}")"
  read -r -a targets <<< "$(task_targets "${task}")"

  for model in "${INVERSE_MODELS[@]}"; do
    if ! matches_optional_filter "${model}" "${MODEL_FILTER:-}"; then
      continue
    fi
    run_name="${task}_${model}"
    log_path="${SUITE_LOG_DIR}/${run_name}.log"
    output_dir="${INVERSE_OUTPUT_ROOT}/$(IFS=_; echo "${modalities[*]}")_to_$(IFS=_; echo "${targets[*]}")_${model}"
    checkpoint_path="${output_dir}/best_model.pt"
    eval_json="${output_dir}/heldout_eval.json"
    job_device="${INVERSE_PARALLEL_DEVICES[$((job_index % ${#INVERSE_PARALLEL_DEVICES[@]}))]}"
    job_index=$((job_index + 1))
    train_cmd=(
      python experiments/train_inverse_regression.py
      --root "${INVERSE_ROOT}"
      --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
      --modalities "${modalities[@]}"
      --target-properties "${targets[@]}"
      --model "${model}"
      --device "${job_device}"
      --batch-size "${INVERSE_BATCH_SIZE}"
      --epochs "${INVERSE_EPOCHS}"
      --lr "${INVERSE_LR}"
      --weight-decay "${INVERSE_WEIGHT_DECAY}"
      --num-workers "${INVERSE_NUM_WORKERS}"
      --prefetch-factor "${INVERSE_PREFETCH_FACTOR}"
      --val-fraction "${INVERSE_VAL_FRACTION}"
      --seed "${INVERSE_SEED}"
      --target-shape ${INVERSE_TARGET_SHAPE}
      --max-train-samples "${INVERSE_MAX_TRAIN_SAMPLES}"
      --max-val-samples "${INVERSE_MAX_VAL_SAMPLES}"
      --max-heldout-samples "${INVERSE_MAX_HELDOUT_SAMPLES}"
      --output-root "${INVERSE_OUTPUT_ROOT}"
    )
    eval_cmd=(
      python experiments/eval_inverse_regression.py
      --checkpoint "${checkpoint_path}"
      --root "${INVERSE_ROOT}"
      --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
      --device "${job_device}"
      --batch-size "${INVERSE_BATCH_SIZE}"
      --num-workers "${INVERSE_NUM_WORKERS}"
      --prefetch-factor "${INVERSE_PREFETCH_FACTOR}"
      --val-fraction "${INVERSE_VAL_FRACTION}"
      --seed "${INVERSE_SEED}"
      --split heldout
    )
    if skip_completed_enabled && completed_eval_exists "${eval_json}"; then
      log_launcher_message "${log_path}" "[inverse-suite] skip ${run_name}; found ${eval_json}"
      continue
    fi

    run_training=1
    if skip_completed_enabled && completed_file_exists "${checkpoint_path}"; then
      run_training=0
    fi

    if ! wait_for_available_slot "${INVERSE_MAX_PARALLEL}"; then
      failed=1
    fi
    (
      eval_batch="${INVERSE_BATCH_SIZE}"
      if [[ "${run_training}" == "1" ]]; then
        run_logged_command_with_oom_retries \
          "${log_path}" \
          "${INVERSE_BATCH_SIZE}" \
          "${INVERSE_OOM_RETRY_BATCH_SIZES:-}" \
          "${train_cmd[@]}"
        eval_batch="${RUN_LOGGED_COMMAND_LAST_BATCH:-${INVERSE_BATCH_SIZE}}"
      else
        log_launcher_message "${log_path}" "[inverse-suite] reuse ${checkpoint_path}; only running held-out eval"
      fi
      run_json_command_with_oom_retries \
        "${log_path}" \
        "${eval_json}" \
        "${eval_batch}" \
        "${INVERSE_OOM_RETRY_BATCH_SIZES:-}" \
        "${eval_cmd[@]}"
      if [[ "${DRY_RUN:-0}" != "1" ]]; then
        echo "[inverse-suite] wrote held-out metrics to ${eval_json}" | tee -a "${log_path}"
      fi
    ) &
  done
done

if ! wait_for_all_jobs; then
  failed=1
fi
exit "${failed}"
