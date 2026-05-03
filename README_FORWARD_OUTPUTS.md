# Forward Output Guide

`run_multiphysics_forward.py` now saves the generated 3D models, acquisition metadata, and forward responses into one unified bundle under `--save_dir`:

```text
DATAFOLDER/Cache/ForwardOutput/forward_bundle.npz
```

This design makes it easier to keep the geological model and all forward responses aligned during loading.

## Main bundle contents

### 1. 3D property models
- `vp_model`: 3D P-wave velocity volume
- `rho_model`: 3D density volume, unit `g/cm^3`
- `res_model`: 3D resistivity volume, unit `Ohm-m`
- `chi_model`: 3D susceptibility volume, unit SI
- `rho_bg_model`: background density model
- `chi_bg_model`: background susceptibility model
- `anomaly_label`: anomaly label volume
- `facies_bg`: background facies labels when available

### 2. Global metadata
- `dx`, `dy`, `dz`: grid spacing
- `rho_unit`: density unit string
- `anomaly_type`: registry key such as `igneous_swarm`
- `anomaly_name_en`, `anomaly_name_zh`

### 3. MT forward outputs
- `mt_status`: `ok`, `failed`, or `not_run`
- `mt_app_res`: apparent resistivity tensor, shape `(n_freq, nx, ny, 2)` with last dimension `(Zxy, Zyx)`
- `mt_phase`: phase tensor, shape `(n_freq, nx, ny, 2)` with last dimension `(Zxy, Zyx)`
- `mt_res_model`: downsampled MT resistivity model actually used for MT forward modeling
- `mt_dx`, `mt_dy`, `mt_dz`: MT grid spacing after downsampling
- `mt_freqs_hz`: frequency list when available
- `mt_error`: present only if MT failed

### 4. Gravity forward outputs
- `gravity_status`: `ok`, `failed`, or `not_run`
- `gravity_data`: 2D gravity anomaly grid, unit `mGal`

### 5. Magnetic forward outputs
- `magnetic_status`: `ok`, `failed`, or `not_run`
- `magnetic_data`: 2D magnetic anomaly grid, unit `nT`

### 6. 3D seismic forward outputs
- `seismic_status`: `ok`, `failed`, or `not_run`
- `seismic_data`: seismic shot gathers, shape `(n_shots, n_receivers, n_time)`
- `seismic_source_locations`: source geometry
- `seismic_receiver_locations`: receiver geometry
- `seismic_wavelet`: source wavelet
- `seismic_dt`, `seismic_nt`, `seismic_freq_hz`
- `seismic_mode`: currently `3D`
- `seismic_preset`: `full` or `light`
- `seismic_batch_size`: resolved number of shots per Deepwave batch
- `seismic_error`: present only if seismic forward failed

## Preview figures saved alongside the bundle

The script also saves several quick-look figures to the same directory:
- `forward_gravity.png`
- `forward_magnetic.png`
- `forward_seismic.png`
- `mt_res_downsampled_xz_slice.png`
- `mt_res_downsampled_xz_slice_linear.png`

## Loading example

```python
import numpy as np

bundle = np.load("DATAFOLDER/Cache/ForwardOutput/forward_bundle.npz", allow_pickle=True)

vp = bundle["vp_model"]
rho = bundle["rho_model"]
res = bundle["res_model"]
chi = bundle["chi_model"]

gravity = bundle["gravity_data"] if bundle["gravity_status"].item() == "ok" else None
magnetic = bundle["magnetic_data"] if bundle["magnetic_status"].item() == "ok" else None
mt_app_res = bundle["mt_app_res"] if bundle["mt_status"].item() == "ok" else None
seismic = bundle["seismic_data"] if bundle["seismic_status"].item() == "ok" else None

print(bundle["anomaly_type"].item(), vp.shape)
```

## Notes

- If a forward module is skipped, its status is stored as `not_run`.
- If a forward module fails, the corresponding `*_status` becomes `failed`, and an error string may be saved as `*_error`.
- The seismic bundle is only written on successful completion of the seismic stage; the current pipeline treats seismic failure as critical and re-raises the exception.
