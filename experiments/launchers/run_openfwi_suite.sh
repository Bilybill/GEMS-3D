#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/ExperimentRuns/openfwi/CurveFault-B}"
PYTHON_BIN="${PYTHON_BIN:-/home/wangyh/anaconda3/envs/torch/bin/python}"
MODELS="${MODELS:-unet pinn deeponet fno gnot}"
SEEDS="${SEEDS:-42 43 44}"

mkdir -p "${OUTPUT_ROOT}/logs"

run_index=0
declare -a pids=()
for model in ${MODELS}; do
  for seed in ${SEEDS}; do
    output="${OUTPUT_ROOT}/${model}/seed_${seed}.json"
    if [[ -s "${output}" ]]; then
      echo "skip existing ${output}"
      continue
    fi
    device="cuda:$((run_index % 2))"
    log="${OUTPUT_ROOT}/logs/${model}_seed_${seed}.log"
    echo "launch model=${model} seed=${seed} device=${device}"
    MODEL="${model}" SEED="${seed}" DEVICE="${device}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" bash "${ROOT}/experiments/launchers/run_openfwi_benchmark.sh" \
      >"${log}" 2>&1 &
    pids+=("$!")
    run_index=$((run_index + 1))
    if (( ${#pids[@]} >= 2 )); then
      wait "${pids[0]}"
      wait "${pids[1]}"
      pids=()
    fi
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

mapfile -t results < <(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -type f -name 'seed_*.json' | sort)
"${PYTHON_BIN}" "${ROOT}/experiments/summarize_openfwi_benchmark.py" \
  "${results[@]}" --output "${OUTPUT_ROOT}/all_models_summary.json"
