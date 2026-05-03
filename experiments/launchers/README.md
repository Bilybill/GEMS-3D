# GMES-3D Experiment Launchers

These launchers are intended for direct server-side submission of the current forward-modeling and anomaly-classification benchmarks.

## Forward suite

Default run:

```bash
bash experiments/launchers/run_forward_suite.sh
```

Run only selected tasks or models:

```bash
TASK_FILTER="joint_multiphysics,vp_to_seismic" MODEL_FILTER="fno,gnot" \
bash experiments/launchers/run_forward_suite.sh
```

Experiment-1 launcher for `train-river/braided + train-river/crossed`:

```bash
bash experiments/launchers/run_forward_exp1_braided_crossed.sh
```

Phase-based execution:

```bash
EXP1_PHASE=phase1 bash experiments/launchers/run_forward_exp1_braided_crossed.sh
EXP1_PHASE=phase2 bash experiments/launchers/run_forward_exp1_braided_crossed.sh
EXP1_PHASE=phase3 bash experiments/launchers/run_forward_exp1_braided_crossed.sh
EXP1_PHASE=phase4 bash experiments/launchers/run_forward_exp1_braided_crossed.sh
EXP1_PHASE=full bash experiments/launchers/run_forward_exp1_braided_crossed.sh
```

Here `phase3` runs the shot-conditioned seismic baseline
`vp_source_to_seismic_shot` with the `shot_film` model.

Summarize Experiment 1 results after training:

```bash
bash experiments/launchers/summarize_forward_exp1_braided_crossed.sh
```

## Classification suite

Default run:

```bash
bash experiments/launchers/run_classification_suite.sh
```

Run only selected modality settings:

```bash
RUN_FILTER="mt,seismic,all_modalities" \
bash experiments/launchers/run_classification_suite.sh
```

## Useful environment overrides

- `CONDA_ENV=torch`: conda environment to activate
- `SKIP_CONDA_ACTIVATE=1`: skip conda activation if the environment is already active
- `DRY_RUN=1`: print commands without executing training
- `TASK_FILTER=...`: comma-separated forward-task filter
- `MODEL_FILTER=...`: comma-separated forward-model filter
- `RUN_FILTER=...`: comma-separated classification-run filter
- `EXP1_PHASE=phase1|phase2|phase3|phase4|full`: staged execution for the forward Experiment 1 launcher

## Config files

- `experiments/configs/forward/default_suite.sh`
- `experiments/configs/forward/exp1_braided_crossed.sh`
- `experiments/configs/classification/default_suite.sh`

To use a custom config, pass its path as the first argument:

```bash
bash experiments/launchers/run_forward_suite.sh /path/to/custom_forward_config.sh
bash experiments/launchers/run_classification_suite.sh /path/to/custom_classification_config.sh
```

## Result summary

Forward result aggregation script:

```bash
python experiments/summarize_forward_results.py \
  --root ./outputs/ExperimentRuns/forward_exp1_braided_crossed
```

This writes:

- `forward_results_summary.json`
- `forward_results_summary.csv`
- `forward_results_summary.md`
