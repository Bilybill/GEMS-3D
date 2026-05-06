#!/usr/bin/env bash

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_FORWARD_OUTPUT_ROOT="${FORWARD_OUTPUT_ROOT:-}"
USER_FORWARD_LOG_ROOT="${FORWARD_LOG_ROOT:-}"
# shellcheck disable=SC1091
source "${CONFIG_DIR}/default_suite.sh"

# Experiment 1 output directory.
# Each task/model pair will be written to:
#   ${FORWARD_OUTPUT_ROOT}/${task}_${model}/
# and logs will be written to:
#   ${FORWARD_LOG_ROOT}/YYYYMMDD_HHMMSS/
FORWARD_OUTPUT_ROOT="${USER_FORWARD_OUTPUT_ROOT:-./outputs/ExperimentRuns/forward_exp1_braided_crossed}"
FORWARD_LOG_ROOT="${USER_FORWARD_LOG_ROOT:-${FORWARD_OUTPUT_ROOT}/logs}"

# Use only braided + crossed from train-river as the development pool.
# With val_fraction=0.15, this gives:
#   train = 680, val = 120, held-out = 120
FORWARD_VAL_FRACTION="0.15"

INCLUDE_TOP_LEVELS=(
  "train-river"
  "tests-river"
)

FORWARD_DEVELOPMENT_SOURCE_PREFIXES=(
  "train-river/braided"
  "train-river/crossed"
)

FORWARD_HELDOUT_SOURCE_PREFIXES=()

FORWARD_TASKS=(
  "rho_to_gravity"
  "chi_to_magnetic"
  "res_to_mt"
  "vp_to_seismic"
  "vp_source_to_seismic_shot"
  "joint_multiphysics"
)

FORWARD_MODELS=(
  "unet"
  "pinn"
  "deeponet"
  "fno"
  "gnot"
  "shot_film"
)

declare -A FORWARD_TASK_MODELS=(
  ["rho_to_gravity"]="unet pinn deeponet fno gnot"
  ["chi_to_magnetic"]="unet pinn deeponet fno gnot"
  ["res_to_mt"]="unet pinn deeponet fno gnot"
  ["vp_to_seismic"]="unet pinn deeponet fno gnot"
  ["vp_source_to_seismic_shot"]="shot_film"
  ["joint_multiphysics"]="unet pinn deeponet fno gnot"
)

# Defaults are tuned for one AMD EPYC 9354 host with two 96 GiB Blackwell GPUs.
# Override any of these from the environment if a specific model family needs a
# more conservative setting.
FORWARD_BATCH_SIZE["rho_to_gravity"]="${RHO_GRAVITY_BATCH_SIZE:-8}"
FORWARD_BATCH_SIZE["chi_to_magnetic"]="${CHI_MAGNETIC_BATCH_SIZE:-8}"
FORWARD_BATCH_SIZE["res_to_mt"]="${RES_MT_BATCH_SIZE:-4}"
FORWARD_BATCH_SIZE["vp_to_seismic"]="${VP_SEISMIC_BATCH_SIZE:-4}"
FORWARD_BATCH_SIZE["joint_multiphysics"]="${JOINT_MULTIPHYSICS_BATCH_SIZE:-4}"
FORWARD_BATCH_SIZE["vp_source_to_seismic_shot"]="${SEISMIC_SHOT_BATCH_SIZE:-4}"
