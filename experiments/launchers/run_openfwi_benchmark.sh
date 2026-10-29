#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/external/OpenFWI/CurveFault_B}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
MODEL="${MODEL:-fno}"
PYTHON_BIN="${PYTHON_BIN:-/home/wangyh/anaconda3/envs/torch/bin/python}"
TRAIN_SIZE="${TRAIN_SIZE:-680}"
VAL_SIZE="${VAL_SIZE:-120}"
TEST_SIZE="${TEST_SIZE:-120}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TIME_STRIDE="${TIME_STRIDE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/ExperimentRuns/openfwi/CurveFault-B}"

"${PYTHON_BIN}" "${ROOT}/experiments/openfwi_benchmark.py" \
  --train-velocity "${DATA_ROOT}/vel6_1_0.npy" "${DATA_ROOT}/vel6_1_1.npy" \
  --train-seismic "${DATA_ROOT}/seis6_1_0.npy" "${DATA_ROOT}/seis6_1_1.npy" \
  --test-velocity "${DATA_ROOT}/vel6_1_32.npy" \
  --test-seismic "${DATA_ROOT}/seis6_1_32.npy" \
  --train-size "${TRAIN_SIZE}" --val-size "${VAL_SIZE}" --test-size "${TEST_SIZE}" \
  --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --time-stride "${TIME_STRIDE}" \
  --seed "${SEED}" --device "${DEVICE}" --model "${MODEL}" \
  --output "${OUTPUT_ROOT}/${MODEL}/seed_${SEED}.json"
