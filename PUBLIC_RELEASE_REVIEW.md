# Public Release Review

This file records the review decisions made while extracting `GEMS-3D` from the working `GMESDataset` tree.

## Kept

- Core anomaly, petrophysics, multiphysics, and forward-wrapper code.
- Gravity, magnetic, seismic, and MT forward-modeling implementations needed by the public pipeline.
- Dataset generation and forward-bundle scripts.
- Benchmark loaders, baseline models, training, evaluation, metrics, and launcher scripts.
- Configuration files for anomaly presets and benchmark runs.
- Public-facing documentation and a static project homepage.

## Removed

- `__pycache__`, compiled extensions, object files, logs, debug outputs, local cache directories, and generated `.npz/.bin` artifacts.
- Exploratory sample directories and prototype branches.
- Machine-specific shell wrappers and test scripts with absolute local paths.
- Temporary benchmark indexes and empty internal utility/exporter placeholders.

## Remaining Release Tasks

- Replace placeholder GitHub and documentation URLs.
- Add a small downloadable example bundle if the paper release plan permits it.
- Run a full smoke test on a clean machine with the intended CUDA/PyTorch/deepwave stack.
