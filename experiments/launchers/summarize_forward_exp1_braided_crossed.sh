#!/usr/bin/env bash
#
# Summarize Experiment 1 forward-modeling results into JSON / CSV / Markdown.
#
# Usage:
#   bash experiments/launchers/summarize_forward_exp1_braided_crossed.sh
#   FORWARD_EXP1_ROOT=/path/to/forward_exp1_braided_crossed \
#   bash experiments/launchers/summarize_forward_exp1_braided_crossed.sh
#
# Outputs:
#   ${FORWARD_EXP1_ROOT}/forward_results_summary.json
#   ${FORWARD_EXP1_ROOT}/forward_results_summary.csv
#   ${FORWARD_EXP1_ROOT}/forward_results_summary.md
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FORWARD_EXP1_ROOT="${FORWARD_EXP1_ROOT:-./outputs/ExperimentRuns/forward_exp1_braided_crossed}"

cd "${PROJECT_ROOT}"

python experiments/summarize_forward_results.py \
  --root "${FORWARD_EXP1_ROOT}" \
  --output-prefix forward_results_summary
