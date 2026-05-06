# GMES-3D Experiment Launchers

These launchers are intended for direct server-side submission of the current forward-modeling and anomaly-classification benchmarks.

By default, the suite is tuned for one AMD EPYC 9354 node with two 96 GiB
NVIDIA RTX PRO 6000 Blackwell GPUs. Launchers auto-expand `DEVICE=cuda` to
`cuda:0 cuda:1`, run one job per GPU, and use 12 DataLoader workers per job
with pinned host memory and prefetching.

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
EXP1_PHASE=table bash experiments/launchers/run_forward_exp1_braided_crossed.sh
EXP1_PHASE=full bash experiments/launchers/run_forward_exp1_braided_crossed.sh
```

Here `phase3` runs the shot-conditioned seismic baseline
`vp_source_to_seismic_shot` with the `shot_film` model.
The `table` phase runs the main forward-modeling table matrix.

## Inverse and robustness suites

Run inverse property-reconstruction jobs for the main manuscript table:

```bash
bash experiments/launchers/run_inverse_main_table.sh
```

Run modality-completion and degraded-inversion jobs for the robustness table:

```bash
bash experiments/launchers/run_robustness_main_table.sh
```

Summarize all manuscript table outputs:

```bash
bash experiments/launchers/summarize_main_tables.sh
```

Summarize Experiment 1 results after training:

```bash
bash experiments/launchers/summarize_forward_exp1_braided_crossed.sh
```

## Monitoring

The persistent monitor sends periodic OpenClaw updates and immediate Feishu
alerts for new errors. Non-OOM errors automatically start a Codex repair
session with `workspace-write` permissions.

```bash
OPENCLAW_TARGET="<feishu-user-or-chat-id>" \
GMES_RUN_ROOT="<experiment-root>/tmux_runs/<run_id>" \
GMES_TMUX_SESSION="gmes_neurips_tables" \
GMES_MONITOR_REPORT_INTERVAL_SECONDS=43200 \
GMES_MONITOR_ERROR_POLL_SECONDS=60 \
python tools/gmes_experiment_monitor.py
```

Runtime artifacts are written under
`<run_root>/monitor_state.json`, `<run_root>/events/`, and
`<run_root>/codex_repair/`.

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
- `SKIP_COMPLETED=1`: default; skip runs with an existing `heldout_eval.json` containing `metric_stats`, or reuse `best_model.pt` and run eval only
- `SKIP_COMPLETED=0`: force retraining and overwrite existing run outputs
- `OOM_RETRY_ENABLED=1`: default; retry the current training/evaluation command after CUDA OOM with smaller batch sizes
- `FORWARD_OOM_RETRY_BATCH_SIZES="8 4 2 1"`: optional explicit retry schedule; without it, the launcher halves the configured batch size down to 1
- `INVERSE_OOM_RETRY_BATCH_SIZES="4 2 1"` / `ROBUST_OOM_RETRY_BATCH_SIZES="4 2 1"` / `CLASSIFICATION_OOM_RETRY_BATCH_SIZES="4 2 1"`: optional suite-specific retry schedules
- `GMES_CUDA_DEVICES="cuda:0 cuda:1"`: explicit device list for round-robin job assignment
- `*_MAX_PARALLEL=2`: cap concurrent jobs for a suite, e.g. `FORWARD_MAX_PARALLEL=1`
- `*_NUM_WORKERS=12`: DataLoader worker count per training/evaluation process
- `*_PREFETCH_FACTOR=2`: DataLoader prefetch factor when workers are enabled
- `TASK_FILTER=...`: comma-separated forward-task filter
- `MODEL_FILTER=...`: comma-separated forward-model filter
- `RUN_FILTER=...`: comma-separated classification-run filter
- `EXP1_PHASE=phase1|phase2|phase3|phase4|table|full`: staged execution for the forward Experiment 1 launcher

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

Main-table export script:

```bash
python experiments/export_neurips_tables.py
```

This writes `neurips_table_export.json` and `neurips_table_export.md`.
