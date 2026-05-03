#!/usr/bin/env bash
#
# Experiment 1 forward-modeling launcher.
#
# Purpose:
#   Run the forward surrogate benchmark on the development subset formed by
#   train-river/braided + train-river/crossed, while keeping tests-river as
#   the held-out evaluation set.
#
# Default behavior:
#   If EXP1_PHASE is not set, the script runs phase1:
#     - rho_to_gravity
#     - chi_to_magnetic
#   with:
#     - unet
#     - fno
#
# Phase options:
#   phase1 : rho_to_gravity + chi_to_magnetic
#   phase2 : res_to_mt
#   phase3 : vp_source_to_seismic_shot
#   phase4 : joint_multiphysics
#   full   : all tasks above
#
# Common usage:
#   bash experiments/launchers/run_forward_exp1_braided_crossed.sh
#   DEVICE=cuda bash experiments/launchers/run_forward_exp1_braided_crossed.sh
#   EXP1_PHASE=phase2 bash experiments/launchers/run_forward_exp1_braided_crossed.sh
#   EXP1_PHASE=phase1 MODEL_FILTER=unet bash experiments/launchers/run_forward_exp1_braided_crossed.sh
#   TASK_FILTER=joint_multiphysics MODEL_FILTER=fno bash experiments/launchers/run_forward_exp1_braided_crossed.sh
#   FORWARD_OUTPUT_ROOT=/tmp/forward_exp1 EXP1_PHASE=phase3 bash experiments/launchers/run_forward_exp1_braided_crossed.sh
#   DEVICE=cuda:1 EXP1_PHASE=phase3 SEISMIC_SHOT_BATCH_SIZE=3 bash experiments/launchers/run_forward_exp1_braided_crossed.sh
#
# Dry run:
#   SKIP_CONDA_ACTIVATE=1 DRY_RUN=1 bash experiments/launchers/run_forward_exp1_braided_crossed.sh
#
# Results:
#   Outputs are written under:
#     ./outputs/ExperimentRuns/forward_exp1_braided_crossed
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/experiments/configs/forward/exp1_braided_crossed.sh"
EXP1_PHASE="${EXP1_PHASE:-phase1}"
DEFAULT_RESULTS_ROOT="./outputs/ExperimentRuns/forward_exp1_braided_crossed"

# Accept a short DEVICE alias for convenience.
if [[ -n "${DEVICE:-}" && -z "${FORWARD_DEVICE:-}" ]]; then
  export FORWARD_DEVICE="${DEVICE}"
fi

case "${EXP1_PHASE}" in
  phase1)
    DEFAULT_TASK_FILTER="rho_to_gravity,chi_to_magnetic"
    DEFAULT_MODEL_FILTER="unet,fno"
    ;;
  phase2)
    DEFAULT_TASK_FILTER="res_to_mt"
    DEFAULT_MODEL_FILTER="unet,fno"
    ;;
  phase3)
    DEFAULT_TASK_FILTER="vp_source_to_seismic_shot"
    DEFAULT_MODEL_FILTER="shot_film"
    ;;
  phase4)
    DEFAULT_TASK_FILTER="joint_multiphysics"
    DEFAULT_MODEL_FILTER="unet,fno"
    ;;
  full)
    DEFAULT_TASK_FILTER="rho_to_gravity,chi_to_magnetic,res_to_mt,vp_source_to_seismic_shot,joint_multiphysics"
    DEFAULT_MODEL_FILTER="unet,fno,shot_film"
    ;;
  *)
    echo "Error: unsupported EXP1_PHASE='${EXP1_PHASE}'. Expected one of: phase1, phase2, phase3, phase4, full."
    exit 1
    ;;
esac

export TASK_FILTER="${TASK_FILTER:-${DEFAULT_TASK_FILTER}}"
export MODEL_FILTER="${MODEL_FILTER:-${DEFAULT_MODEL_FILTER}}"

echo "[exp1-forward] phase=${EXP1_PHASE}"
echo "[exp1-forward] task_filter=${TASK_FILTER}"
echo "[exp1-forward] model_filter=${MODEL_FILTER}"
echo "[exp1-forward] device=${FORWARD_DEVICE:-cuda}"
echo "[exp1-forward] config=${CONFIG_PATH}"
echo "[exp1-forward] results_root=${FORWARD_OUTPUT_ROOT:-${DEFAULT_RESULTS_ROOT}}"
echo "[exp1-forward] per-run outputs: results_root/<task>_<model>/"
echo "[exp1-forward] logs: results_root/logs/<timestamp>/"

exec bash "${SCRIPT_DIR}/run_forward_suite.sh" "${CONFIG_PATH}"
