#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

activate_conda_env_if_needed
cd "${PROJECT_ROOT}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-./outputs/ExperimentRuns}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/neurips_table_exports}"

python experiments/export_neurips_tables.py \
  --forward-root "${FORWARD_ROOT:-${EXPERIMENT_ROOT}/forward_exp1_braided_crossed}" \
  --inverse-root "${INVERSE_ROOT:-${EXPERIMENT_ROOT}/inverse_regression}" \
  --completion-root "${COMPLETION_ROOT:-${EXPERIMENT_ROOT}/modality_completion}" \
  --robustness-root "${ROBUSTNESS_ROOT:-${EXPERIMENT_ROOT}/robustness}" \
  --output-dir "${OUTPUT_DIR}"
