#!/usr/bin/env bash

set -euo pipefail

launcher_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

project_root() {
  cd "$(launcher_dir)/../.." && pwd
}

activate_conda_env_if_needed() {
  local conda_env="${CONDA_ENV:-torch}"
  local conda_sh="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"

  if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
    return 0
  fi

  if [[ ! -f "${conda_sh}" ]]; then
    echo "Error: conda activation script not found at ${conda_sh}"
    return 1
  fi

  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${conda_env}"
}

matches_optional_filter() {
  local value="$1"
  local filter_csv="${2:-}"
  local candidate

  if [[ -z "${filter_csv}" ]]; then
    return 0
  fi

  IFS=',' read -r -a filter_items <<< "${filter_csv}"
  for candidate in "${filter_items[@]}"; do
    candidate="${candidate//[[:space:]]/}"
    if [[ -n "${candidate}" && "${value}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

ensure_directory() {
  mkdir -p "$1"
}

skip_completed_enabled() {
  [[ "${SKIP_COMPLETED:-1}" == "1" ]]
}

completed_file_exists() {
  [[ -s "$1" ]]
}

completed_eval_exists() {
  [[ -s "$1" ]] && grep -q '"metric_stats"' "$1"
}

log_launcher_message() {
  local log_path="$1"
  shift
  ensure_directory "$(dirname "${log_path}")"
  echo "$*" | tee -a "${log_path}"
}

batch_retry_values() {
  local initial_batch="$1"
  local configured="${2:-}"
  local values=()
  local seen=" "
  local candidate

  values+=("${initial_batch}")
  seen+=" ${initial_batch} "

  configured="${configured//,/ }"
  if [[ -n "${configured//[[:space:]]/}" ]]; then
    for candidate in ${configured}; do
      if [[ "${candidate}" =~ ^[0-9]+$ ]] && (( candidate >= 1 && candidate <= initial_batch )); then
        if [[ "${seen}" != *" ${candidate} "* ]]; then
          values+=("${candidate}")
          seen+="${candidate} "
        fi
      fi
    done
  else
    candidate="${initial_batch}"
    while (( candidate > 1 )); do
      candidate=$(( (candidate + 1) / 2 ))
      if [[ "${seen}" != *" ${candidate} "* ]]; then
        values+=("${candidate}")
        seen+="${candidate} "
      fi
      if (( candidate == 1 )); then
        break
      fi
    done
  fi

  if [[ "${seen}" != *" 1 "* ]]; then
    values+=("1")
    seen+="1 "
  fi

  echo "${values[*]}"
}

set_batch_arg() {
  local -n cmd_ref="$1"
  local batch_size="$2"
  local idx
  for idx in "${!cmd_ref[@]}"; do
    if [[ "${cmd_ref[$idx]}" == "--batch-size" ]]; then
      cmd_ref[$((idx + 1))]="${batch_size}"
      return 0
    fi
  done
  cmd_ref+=(--batch-size "${batch_size}")
}

attempt_has_cuda_oom() {
  local attempt_log="$1"
  grep -qiE "CUDA out of memory|OutOfMemoryError|out of memory" "${attempt_log}"
}

run_logged_command_capture() {
  local log_path="$1"
  local attempt_log="$2"
  shift 2

  ensure_directory "$(dirname "${log_path}")"
  echo "[launcher] $(date '+%F %T') :: $*" | tee -a "${log_path}" "${attempt_log}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  set +e
  "$@" 2>&1 | tee -a "${log_path}" "${attempt_log}"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

run_logged_command_with_oom_retries() {
  local log_path="$1"
  local initial_batch="$2"
  local retry_schedule="$3"
  shift 3
  local base_cmd=("$@")
  local retry_values=()
  local batch_size
  local attempt_log
  local status=0
  local attempt=0

  RUN_LOGGED_COMMAND_LAST_BATCH="${initial_batch}"
  if [[ "${OOM_RETRY_ENABLED:-1}" != "1" ]]; then
    retry_values=("${initial_batch}")
  else
    read -r -a retry_values <<< "$(batch_retry_values "${initial_batch}" "${retry_schedule}")"
  fi
  ensure_directory "$(dirname "${log_path}")"

  for batch_size in "${retry_values[@]}"; do
    attempt=$((attempt + 1))
    attempt_log="$(mktemp "$(dirname "${log_path}")/.${attempt}.oom_retry.XXXXXX.log")"
    local cmd=("${base_cmd[@]}")
    set_batch_arg cmd "${batch_size}"
    log_launcher_message "${log_path}" "[launcher] attempt ${attempt}/${#retry_values[@]} with batch_size=${batch_size}"
    if run_logged_command_capture "${log_path}" "${attempt_log}" "${cmd[@]}"; then
      RUN_LOGGED_COMMAND_LAST_BATCH="${batch_size}"
      rm -f "${attempt_log}"
      return 0
    fi
    status=$?
    if attempt_has_cuda_oom "${attempt_log}" && (( attempt < ${#retry_values[@]} )); then
      log_launcher_message "${log_path}" "[launcher] CUDA OOM detected; retrying with smaller batch size"
      rm -f "${attempt_log}"
      continue
    fi
    rm -f "${attempt_log}"
    return "${status}"
  done
  return "${status}"
}

run_json_command_with_oom_retries() {
  local log_path="$1"
  local output_path="$2"
  local initial_batch="$3"
  local retry_schedule="$4"
  shift 4
  local base_cmd=("$@")
  local retry_values=()
  local batch_size
  local attempt_log
  local tmp_output
  local status=0
  local attempt=0

  RUN_JSON_COMMAND_LAST_BATCH="${initial_batch}"
  if [[ "${OOM_RETRY_ENABLED:-1}" != "1" ]]; then
    retry_values=("${initial_batch}")
  else
    read -r -a retry_values <<< "$(batch_retry_values "${initial_batch}" "${retry_schedule}")"
  fi
  ensure_directory "$(dirname "${log_path}")"

  for batch_size in "${retry_values[@]}"; do
    attempt=$((attempt + 1))
    attempt_log="$(mktemp "$(dirname "${log_path}")/.${attempt}.json_oom_retry.XXXXXX.log")"
    tmp_output="${output_path}.tmp.${attempt}.$$"
    local cmd=("${base_cmd[@]}")
    set_batch_arg cmd "${batch_size}"
    echo "[launcher] $(date '+%F %T') :: attempt ${attempt}/${#retry_values[@]} batch_size=${batch_size} :: ${cmd[*]} > ${output_path}" | tee -a "${log_path}" "${attempt_log}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      rm -f "${attempt_log}" "${tmp_output}"
      RUN_JSON_COMMAND_LAST_BATCH="${batch_size}"
      return 0
    fi

    set +e
    "${cmd[@]}" > "${tmp_output}" 2> >(tee -a "${log_path}" "${attempt_log}" >&2)
    status=$?
    set -e

    if [[ "${status}" == "0" ]]; then
      mv "${tmp_output}" "${output_path}"
      RUN_JSON_COMMAND_LAST_BATCH="${batch_size}"
      rm -f "${attempt_log}"
      return 0
    fi
    rm -f "${tmp_output}"
    if attempt_has_cuda_oom "${attempt_log}" && (( attempt < ${#retry_values[@]} )); then
      log_launcher_message "${log_path}" "[launcher] CUDA OOM detected during eval; retrying with smaller batch size"
      rm -f "${attempt_log}"
      continue
    fi
    rm -f "${attempt_log}"
    return "${status}"
  done
  return "${status}"
}

count_visible_cuda_devices() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "1"
    return 0
  fi
  local count
  count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
  if [[ -z "${count}" || "${count}" == "0" ]]; then
    echo "1"
    return 0
  fi
  echo "${count}"
}

resolve_parallel_devices() {
  local base_device="${1:-cuda}"
  local configured="${GMES_CUDA_DEVICES:-${CUDA_DEVICES:-}}"
  if [[ -n "${configured}" ]]; then
    configured="${configured//,/ }"
    echo "${configured}"
    return 0
  fi

  if [[ "${base_device}" == "cuda" || "${base_device}" == "auto" ]]; then
    local gpu_count
    gpu_count="$(count_visible_cuda_devices)"
    if (( gpu_count > 1 )); then
      local devices=()
      local idx
      for ((idx = 0; idx < gpu_count; idx++)); do
        devices+=("cuda:${idx}")
      done
      echo "${devices[*]}"
      return 0
    fi
  fi
  echo "${base_device}"
}

normalize_parallel_count() {
  local requested="$1"
  local fallback="$2"
  if [[ -z "${requested}" || "${requested}" == "0" ]]; then
    echo "${fallback}"
    return 0
  fi
  if (( requested < 1 )); then
    echo "1"
    return 0
  fi
  echo "${requested}"
}

running_job_count() {
  jobs -pr | wc -l | tr -d '[:space:]'
}

wait_for_available_slot() {
  local max_parallel="$1"
  local status=0
  while (( "$(running_job_count)" >= max_parallel )); do
    if ! wait -n; then
      status=1
    fi
  done
  return "${status}"
}

wait_for_all_jobs() {
  local status=0
  while (( "$(running_job_count)" > 0 )); do
    if ! wait -n; then
      status=1
    fi
  done
  return "${status}"
}

run_logged_command() {
  local log_path="$1"
  shift

  ensure_directory "$(dirname "${log_path}")"
  echo "[launcher] $(date '+%F %T') :: $*" | tee -a "${log_path}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  "$@" 2>&1 | tee -a "${log_path}"
  local status=${PIPESTATUS[0]}
  return "${status}"
}
