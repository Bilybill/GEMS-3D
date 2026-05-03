# Data Directory

This directory is intentionally empty in the public code release.

Recommended local layout:

```text
data/
  background_velocity/        # raw .bin velocity volumes for dataset generation
  samples/                    # optional mirrored .npz files with gtime volumes
  examples/                   # small local examples for solver smoke tests
```

Large generated artifacts should be written under `outputs/` and should not be committed.
