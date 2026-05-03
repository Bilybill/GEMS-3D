# Seismic Forward Modeling Framework

This is a forward modeling framework based on `deepwave`.

## Structure

- `config.yaml`: Configuration file for simulation parameters.
- `main.py`: Main script to run the simulation.
- `utils.py`: Helper functions.

## Setup

Ensure `deepwave`, `torch`, `numpy`, `pyyaml`, and `matplotlib` are installed.

## Configuration (`config.yaml`)

- **simulation**:
  - `mode`: "2D" or "3D".
  - `device`: "cuda" or "cpu".

- **model**:
  - `file_path`: Path to velocity model file (`.npy`, `.npz`, `.bin`).
    - **Convention**:
      - 2D: Shape `(mx, mz)` or `(width, depth)`. The first dimension is usually the horizontal acquisition direction.
      - 3D: Shape `(mx, my, mz)`. The last dimension is depth.
  - `shape`: Shape of the model `[nx, nz]` or `[nx, ny, nz]`.
  - `dx`: Grid spacing.
  - `anomalies` (optional): Inject anomaly bodies into the base velocity model.
    - `enabled`: true/false
    - `items`: List of anomaly definitions (see config example)
    - Supported types: `sphere`, `ellipsoid`, `box`, `cylinder` (3D), `layer`, `gaussian`, `mask`, `random_spheres`
    - Supported modes: `add`, `replace`, `multiply`

- **source**:
  - `n_shots`: Number of shots.
  - `d_source`: Shot spacing (in grid cells).
  - `source_depth`: Depth index of source (index in the LAST dimension, e.g., Z).
  - Wavelet settings (`ricker`, `freq`).

- **receiver**:
  - `n_receivers_per_shot`: Number of channels.
  - `d_receiver`: Receiver spacing.
  - `receiver_depth`: Receiver depth index.

## Usage

1. Edit `config.yaml`.
2. Run config:
   ```bash
   python main.py --config config.yaml
   ```
3. Output is saved to `output.save_path`.
