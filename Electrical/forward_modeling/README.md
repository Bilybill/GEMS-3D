# CUDA 3D MT Forward Modeling

This module solves the frequency-domain 3D MT curl-curl system with first-order
edge elements and a CUDA left-Jacobi BiCGStab solver. The Python API returns
apparent resistivity and phase for `Zxy` and `Zyx`.

## Build

Build from this directory with a CUDA-compatible host compiler:

```bash
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 python setup.py build_ext --inplace
```

## Recommended Use

Use an explicit air layer for physical surface responses:

```python
from mt_forward import MTForward3D

solver = MTForward3D(
    freqs=[0.1, 4.0, 100.0],
    dx=500.0,
    dy=500.0,
    dz=125.0,
    boundary_mode="explicit_air",
    air_layers=8,
)
app_res, phase = solver(rho)
```

`explicit_air` defaults to a `1e-8` true-and-scaled residual tolerance and
independent frequency solves. `legacy` preserves the earth-surface tangential-E
boundary, uses `1e-7`, and enables high-to-low frequency continuation. Override
these choices with `solver_tolerance`, `solver_max_iterations`, and
`frequency_continuation` when benchmarking a specific model. The expanding
remote mesh is configurable with `padding_layers` and `padding_growth`.

The multiphysics and pretraining runners use `explicit_air` by default and
expose the same controls as `--mt-boundary-mode`, `--mt-air-layers`,
`--mt-solver-tolerance`, `--mt-padding-layers`, `--mt-padding-growth`, and
`--[no-]mt-frequency-continuation`.

Run the bundled DATAFOLDER demonstration and generate response figures with:

```bash
python examples/run_datfolder_mt_demo.py --frequencies 1000,400,200
```

To solve the NeurIPS benchmark's fixed 19-frequency grid from 10 kHz to 30 Hz:

```bash
python examples/run_datfolder_mt_demo.py \
  --frequencies nips19 --solver-tolerance 1e-6 \
  --frequency-continuation \
  --output-dir outputs/mt_datfolder_demo/serpentinized_nips19
```

The fixed grid is the loader's 76-channel learning representation. Dataset
production may instead store a shorter, sample-adaptive Phoenix schedule.

For the recommended refined version of the 6.4 km-deep example, split every
128 m vertical cell in two and extend the remote boundary to 12 padding cells:

```bash
python examples/run_datfolder_mt_demo.py \
  --frequencies nips19 --solver-tolerance 1e-6 \
  --frequency-continuation --vertical-refinement 2 \
  --air-layers 16 \
  --padding-layers 12 --padding-growth 1.4 \
  --output-dir outputs/mt_datfolder_demo/serpentinized_nips19_refined
```

This changes both vertical discretization and remote lateral extent, so compare
it with the original as a two-level grid-and-boundary sensitivity study, not as
a formal grid-convergence rate.

Render the saved model and its `x-y-frequency` MT response volumes with cigvis:

```bash
xvfb-run -a python examples/visualize_mt_cigvis.py
```

For the refined result, pass
`--results outputs/mt_datfolder_demo/serpentinized_nips19_refined/mt_demo_results.npz`.

To inspect one fixed-X Y-Z model section together with the MT responses along
the same X line, use:

```bash
xvfb-run -a python examples/visualize_yz_mt_cigvis.py \
  --results outputs/mt_datfolder_demo/serpentinized_nips19_refined/mt_demo_results.npz \
  --x-index 25 --y-index 25
```

The script writes a static Y-Z/response overview, a station sounding, a cigvis
Y-Z model plane, and cigvis Y-frequency response planes under `cigvis_yz/`.
`--x-index` selects the model plane; `--y-index` marks the corresponding
surface station on the maps and sounding.

Use `python examples/visualize_mt_cigvis.py --run-app` on a desktop for
interactive windows. The script also writes a frequency-depth table using
`delta = 503 sqrt(rho/f)`. These values are uniform-half-space skin-depth scales;
apparent-resistivity-derived values are heuristic proxies, not DOI estimates.

The default case uses the 50-cubed serpentinized-zone MT model and writes its
arrays, summary, model slices, response maps, and sounding curves under
`GEMS-3D/outputs/mt_datfolder_demo/`.

## Validation

Run the Python and CUDA regression suite:

```bash
python -m pytest tests -q
```

Cross-check against existing MT3D_MFEM cases:

```bash
python examples/cross_validate_mt3d_mfem.py \
  --mfem-results-dir ../../outputs/mt3d_mfem_cross_validation_padded_multifreq \
  --gems-boundary-mode explicit_air --frequencies 0.1,4,100
```

The comparison script records full-field metrics, arrays, and figures. Interpret
absolute differences with care: GEMS currently applies a remote 1D electric
Dirichlet condition above the explicit air, while MT3D_MFEM uses a remote
natural magnetic boundary. The anomaly-over-uniform response is therefore the
more directly comparable diagnostic.
