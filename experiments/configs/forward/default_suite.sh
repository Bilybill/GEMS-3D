#!/usr/bin/env bash

FORWARD_ROOT="${FORWARD_ROOT:-./outputs/PretrainDataset_forward}"
FORWARD_OUTPUT_ROOT="${FORWARD_OUTPUT_ROOT:-./outputs/ExperimentRuns/forward_surrogates}"
FORWARD_LOG_ROOT="${FORWARD_LOG_ROOT:-${FORWARD_OUTPUT_ROOT}/logs}"

INCLUDE_TOP_LEVELS=(
  "train-river"
  "tests-river"
)

FORWARD_DEVELOPMENT_SOURCE_PREFIXES=()
FORWARD_HELDOUT_SOURCE_PREFIXES=()

FORWARD_MODELS=(
  "unet"
  "pinn"
  "deeponet"
  "fno"
  "gnot"
  "shot_film"
)

FORWARD_TASKS=(
  "rho_to_gravity"
  "chi_to_magnetic"
  "res_to_mt"
  "vp_to_seismic"
  "vp_source_to_seismic_shot"
  "joint_multiphysics"
)

declare -A FORWARD_TASK_MODELS=(
  ["rho_to_gravity"]="unet pinn deeponet fno gnot"
  ["chi_to_magnetic"]="unet pinn deeponet fno gnot"
  ["res_to_mt"]="unet pinn deeponet fno gnot"
  ["vp_to_seismic"]="unet pinn deeponet fno gnot"
  ["vp_source_to_seismic_shot"]="shot_film"
  ["joint_multiphysics"]="unet pinn deeponet fno gnot"
)

declare -A FORWARD_BATCH_SIZE=(
  ["rho_to_gravity"]="8"
  ["chi_to_magnetic"]="8"
  ["res_to_mt"]="4"
  ["vp_to_seismic"]="4"
  ["vp_source_to_seismic_shot"]="4"
  ["joint_multiphysics"]="4"
)

declare -A FORWARD_EPOCHS=(
  ["rho_to_gravity"]="30"
  ["chi_to_magnetic"]="30"
  ["res_to_mt"]="30"
  ["vp_to_seismic"]="30"
  ["vp_source_to_seismic_shot"]="30"
  ["joint_multiphysics"]="30"
)

declare -A FORWARD_LR=(
  ["rho_to_gravity"]="1e-3"
  ["chi_to_magnetic"]="1e-3"
  ["res_to_mt"]="1e-3"
  ["vp_to_seismic"]="1e-3"
  ["vp_source_to_seismic_shot"]="1e-3"
  ["joint_multiphysics"]="5e-4"
)

FORWARD_WEIGHT_DECAY="${FORWARD_WEIGHT_DECAY:-1e-4}"
FORWARD_NUM_WORKERS="${FORWARD_NUM_WORKERS:-12}"
FORWARD_PREFETCH_FACTOR="${FORWARD_PREFETCH_FACTOR:-2}"
FORWARD_VAL_FRACTION="${FORWARD_VAL_FRACTION:-0.1}"
FORWARD_SEED="${FORWARD_SEED:-42}"
FORWARD_DEVICE="${FORWARD_DEVICE:-cuda}"
FORWARD_MAX_TRAIN_SAMPLES="${FORWARD_MAX_TRAIN_SAMPLES:-0}"
FORWARD_MAX_VAL_SAMPLES="${FORWARD_MAX_VAL_SAMPLES:-0}"
FORWARD_MAX_HELDOUT_SAMPLES="${FORWARD_MAX_HELDOUT_SAMPLES:-0}"
