# GMES-3D Experiment Code Roadmap

This roadmap translates the current paper design into an implementation plan for benchmark code. The goal is to get a reproducible `v1` benchmark running quickly, then expand toward stronger baselines and additional tasks without rewriting the whole stack.

## 1. Paper benchmark scope and implementation staging

The manuscript should describe the full benchmark scope rather than only the first implemented scripts. Engineering rollout can be staged, but the paper-level Tier-1 design should already include the joint multimodal surrogate task.

- Tier 1: forward modeling surrogates
  - `rho -> gravity`
  - `chi -> magnetic`
  - `res -> MT`
  - `vp -> seismic`
  - `[vp, rho, res, chi] -> [gravity, magnetic, MT, seismic]`
- Tier 2: inverse interpretation
  - 9-way anomaly-family classification
  - coarse anomaly localization / segmentation
- Tier 3: robustness and representation learning
  - missing-modality evaluation
  - modality-shuffled consistency control
  - model-side self-supervised pretraining

Recommended data protocol:

- Supervised forward-complete set:
  - public development partition
  - public held-out evaluation partition
- Model-side pretraining set:
  - released model packages from the public development partition
  - optional future scale-up on the full internal 9,833-model corpus
- Split rule:
  - background-disjoint
  - validation split drawn from development-partition backgrounds

Tier-1 tensorized benchmark definitions:

- `rho -> gravity`
  - input: standardized `1 x 256 x 256 x 256`
  - target: `1 x 256 x 256`
- `chi -> magnetic`
  - input: standardized `1 x 256 x 256 x 256`
  - target: `1 x 256 x 256`
- `res -> MT`
  - input: log-standardized `1 x 256 x 256 x 256`
  - target: stacked `app_res + phase`, shape `76 x 50 x 50`
- `vp -> seismic`
  - input: standardized `1 x 256 x 256 x 256`
  - target: downsampled shot gathers, shape `25 x 256 x 188`
- `[vp, rho, res, chi] -> [gravity, magnetic, MT, seismic]`
  - input: aligned four-channel property tensor, shape `4 x 256 x 256 x 256`
  - targets: the four native modality tensors above
  - optimization: mean of modality-specific losses
  - reporting: per-modality `RL2 / R / MAE` plus macro-averaged joint score

## 2. Recommended code layout

Create a new benchmark package under `GMESDataset/experiments/`:

```text
experiments/
  configs/
    classification/
    forward/
    pretraining/
    segmentation/
  datasets/
    benchmark_index.py
    gmes_inverse_dataset.py
    gmes_forward_dataset.py
    gmes_pretrain_dataset.py
    modality_transforms.py
    collate.py
  models/
    encoders_2d.py
    encoders_mt.py
    encoders_seismic.py
    fusion.py
    heads_classification.py
    heads_segmentation.py
    joint_forward.py
    operator_unet.py
    operator_fno.py
    pretraining_mae.py
  utils/
    io.py
    splits.py
    metrics_classification.py
    metrics_segmentation.py
    metrics_forward.py
    logging.py
    seed.py
  audit_dataset.py
  train_classification.py
  eval_classification.py
  train_segmentation.py
  eval_segmentation.py
  train_forward_surrogate.py
  eval_forward.py
  train_pretraining.py
  eval_missing_modalities.py
  eval_consistency_shuffle.py
```

The design principle is to share one dataset/index layer and one modality-encoder layer across tasks.

## 3. Data assumptions to encode once

From the current bundle format, the main input shapes are:

- property volumes: `256 x 256 x 256`
- gravity: `256 x 256`
- magnetic: `256 x 256`
- MT:
  - `mt_app_res`: `19 x 50 x 50 x 2`
  - `mt_phase`: `19 x 50 x 50 x 2`
- seismic:
  - `25 x 4096 x 750`

These assumptions should live in dataset/transforms code, not be hardcoded separately in every training script.

## 4. Build order

The implementation order matters. Do not start from segmentation or pretraining first.

### Stage 0: index, splits, and audit

Write these files first:

- `experiments/datasets/benchmark_index.py`
- `experiments/utils/splits.py`
- `experiments/utils/io.py`
- `experiments/audit_dataset.py`

Responsibilities:

- scan `model_bundle.npz` and `forward_bundle.npz`
- build one canonical sample index
- record split, background id, anomaly family, available modalities, and file paths
- enforce background-disjoint train/val splitting
- emit CSV or JSON summaries for the paper tables

Minimum deliverable:

- one saved index file
- one audit script that prints split counts, family counts, modality availability, and QC-ready summary stats

This stage unblocks everything else.

### Stage 1: shared transforms and metrics

Write next:

- `experiments/datasets/modality_transforms.py`
- `experiments/datasets/collate.py`
- `experiments/utils/metrics_classification.py`
- `experiments/utils/metrics_segmentation.py`
- `experiments/utils/metrics_forward.py`

Recommended preprocessing:

- gravity / magnetic:
  - robust standardization per sample
- MT:
  - use `log10(app_res)`
  - normalize phase into a fixed range
  - stack frequency and polarization as channels
- seismic:
  - clip extreme amplitudes
  - normalize per shot or per sample
  - add optional receiver/time downsampling switches
- segmentation label:
  - convert `anomaly_label > 0` to binary mask
  - provide downsample targets such as `64^3`

Minimum deliverable:

- deterministic preprocessing for every modality
- reusable metric functions for all later training scripts

### Stage 2: classification baseline first

Write next:

- `experiments/datasets/gmes_inverse_dataset.py`
- `experiments/models/encoders_2d.py`
- `experiments/models/encoders_mt.py`
- `experiments/models/encoders_seismic.py`
- `experiments/models/fusion.py`
- `experiments/models/heads_classification.py`
- `experiments/train_classification.py`
- `experiments/eval_classification.py`

Recommended baseline design:

- gravity / magnetic:
  - small 2D CNN or ResNet-style encoder
- MT:
  - 2D encoder over frequency-stacked channels
- seismic:
  - shot-wise 2D encoder plus shot pooling
- fusion:
  - late fusion MLP

Minimum deliverable:

- single-modality baselines for `G`, `M`, `MT`, `S`
- multimodal late-fusion baseline for `G+M+MT+S`
- results on validation and held-out evaluation

This should be the first fully trainable benchmark.

### Stage 3: missing-modality and shuffled-control evaluation

Write next:

- `experiments/eval_missing_modalities.py`
- `experiments/eval_consistency_shuffle.py`

These scripts should reuse the classification code rather than creating a second pipeline.

Responsibilities:

- evaluate trained classifiers with one or more missing modalities
- compare:
  - standard training
  - modality-dropout training
  - modality-dropout plus consistency regularization
- implement modality-shuffled negative controls

Minimum deliverable:

- one table for missing-modality robustness
- one table for aligned vs shuffled controls

This stage is high paper value and low engineering cost.

### Stage 4: forward modeling baselines

Write next:

- `experiments/datasets/gmes_forward_dataset.py`
- `experiments/models/operator_unet.py`
- `experiments/models/operator_fno.py`
- `experiments/train_forward_surrogate.py`
- `experiments/eval_forward.py`

Recommended order inside forward modeling:

1. `rho -> gravity`
2. `chi -> magnetic`
3. `res -> MT`
4. `vp -> seismic`
5. `[vp, rho, res, chi] -> [gravity, magnetic, MT, seismic]`

This is an implementation order only, not a statement about benchmark scope. Start from the simplest per-physics tasks so the evaluation protocol stabilizes early, then add the joint multimodal surrogate once task-wise preprocessing and metrics are stable.

Joint-task implementation modules:

- `experiments/datasets/gmes_forward_dataset.py`
  - add a `joint_multiphysics` task that packs four aligned property volumes and returns four modality targets
- `experiments/models/joint_forward.py`
  - build a jointly optimized surrogate with modality-specific output heads
- `experiments/utils/metrics_forward.py`
  - support per-modality metrics and macro-averaged joint reporting
- `experiments/train_forward_surrogate.py`
  - support averaged multi-output losses and nested metric logging
- `experiments/eval_forward.py`
  - emit both aggregate and per-modality held-out results for the joint task

Recommended outputs and metrics:

- gravity / magnetic:
  - `RL2`, `MAE`, `R`
- MT:
  - `RL2`, `MAE`, `R` on apparent resistivity and phase
- seismic:
  - `RL2`, `trace correlation`, `inference time`

Minimum deliverable:

- one baseline per forward task
- one evaluation script shared across tasks

### Stage 5: coarse anomaly segmentation

Write next:

- `experiments/models/heads_segmentation.py`
- `experiments/train_segmentation.py`
- `experiments/eval_segmentation.py`

Use the inverse dataset and modality encoders from classification. Only the task head should change.

Recommended design:

- encode each modality
- fuse embeddings
- decode to `64^3` anomaly mask

Minimum deliverable:

- seismic-only baseline
- MT+S baseline
- full multimodal baseline

This is enough for the first paper.

### Stage 6: model-side self-supervised pretraining

Write last:

- `experiments/datasets/gmes_pretrain_dataset.py`
- `experiments/models/pretraining_mae.py`
- `experiments/train_pretraining.py`

Then extend:

- `experiments/train_classification.py`
- `experiments/train_segmentation.py`

to support pretrained checkpoints.

Recommended first pretraining task:

- masked reconstruction of aligned property volumes
- optional auxiliary reconstruction of anomaly mask or facies labels

Minimum deliverable:

- pretrained encoder checkpoint
- label-efficiency curves for classification and segmentation

## 5. Config files to add early

Write config files as soon as Stage 2 starts:

- `experiments/configs/classification/base.yaml`
- `experiments/configs/classification/full_fusion.yaml`
- `experiments/configs/forward/rho_gravity.yaml`
- `experiments/configs/forward/chi_magnetic.yaml`
- `experiments/configs/forward/res_mt.yaml`
- `experiments/configs/forward/vp_seismic.yaml`
- `experiments/configs/segmentation/base.yaml`
- `experiments/configs/pretraining/mae_property.yaml`

Each config should define:

- split file or index path
- modality list
- preprocessing switches
- model name
- optimizer and scheduler
- batch size
- evaluation metrics
- output directory

## 6. Recommended first milestone

The fastest path to a real paper result is:

1. `benchmark_index.py`
2. `audit_dataset.py`
3. `gmes_inverse_dataset.py`
4. `metrics_classification.py`
5. `encoders_2d.py`, `encoders_mt.py`, `encoders_seismic.py`
6. `fusion.py`
7. `train_classification.py`
8. `eval_missing_modalities.py`
9. `eval_consistency_shuffle.py`

This milestone already supports:

- the main classification table
- missing-modality ablations
- physical-consistency control

That is enough to anchor the first strong results section.

## 7. Recommended second milestone

After classification is stable:

1. `gmes_forward_dataset.py`
2. `metrics_forward.py`
3. `train_forward_surrogate.py`
4. `eval_forward.py`
5. `train_segmentation.py`
6. `eval_segmentation.py`

This milestone fills the forward-modeling and localization tiers in the paper.

## 8. Recommended third milestone

After the benchmark is stable:

1. `gmes_pretrain_dataset.py`
2. `pretraining_mae.py`
3. `train_pretraining.py`
4. fine-tuning support in classification and segmentation training

This milestone fills the label-efficiency and foundation-model part of the paper.

## 9. What not to do first

Avoid these early traps:

- do not begin with full-resolution `256^3` inversion
- do not begin engineering from the joint four-modality surrogate
- do not write separate dataset code for every task
- do not hardcode split logic in every `train_*.py`
- do not mix model-side and observation-side pretraining before the basic benchmark is stable

## 10. Practical rule

If only one experiment can be implemented first, make it:

- late-fusion anomaly-family classification
- with ID/OOD evaluation
- plus missing-modality and shuffled-control ablations

This gives the highest paper value per unit engineering effort and directly validates the core claim of sample-level multimodal physical consistency.
