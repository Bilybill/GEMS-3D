#!/usr/bin/env bash

set -euo pipefail

launcher_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

project_root() {
  cd "$(launcher_dir)/../.." && pwd
}

activate_conda_env_if_needed() {
  local conda_env="${CONDA_ENV:-}"
  local conda_sh="${CONDA_SH:-}"

  if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" || -z "${conda_env}" ]]; then
    return 0
  fi

  if [[ -z "${conda_sh}" ]]; then
    conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
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
