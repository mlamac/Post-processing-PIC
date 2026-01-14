# EPOCH Post-Processing Script

A standalone Python script for visualizing **EPOCH quasi-3D** laser wakefield acceleration (LWFA) simulations. Generates animated 2-panel visualizations showing plasma density with laser field overlay and density-colored phase space.

## What It Does

Takes raw EPOCH SDF output files and creates an interactive HTML animation:

| Panel | Shows |
|-------|-------|
| Top | Electron density (grayscale) with laser field overlay (pink), dual colorbars |
| Bottom | Longitudinal phase space colored by local density, with momentum lineout |

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

You also need `sdf_helper` from EPOCH:
```bash
cd /path/to/epoch/epoch2d
make sdfutils
export PYTHONPATH="/path/to/epoch/epoch2d/SDF/utilities:$PYTHONPATH"
```

### 2. Run

Copy the script to your EPOCH output directory and run:

```bash
cd /path/to/your/epoch/output
cp /path/to/Post-processing-PIC/post-process-lwfa.py .
python post-process-lwfa.py
```

### 3. View Results

Open `post-process-output/lwfa_q3d_animation.html` in a web browser.

## Configuration

Edit the `CONFIG` dictionary at the top of `post-process-lwfa.py`:

```python
CONFIG = {
    # Your simulation parameters
    'lambda0_um': 1.0,          # Laser wavelength [μm]
    'a0': 3.0,                  # Normalized vector potential
    'n_over_nc': 0.0025,        # Plasma density / critical density

    # File patterns (EPOCH default naming)
    'efield_pattern': 'E_field{:04d}.sdf',
    'ener_pattern': 'ener{:04d}.sdf',
    'dens_pattern': 'dens*.sdf',

    # Particle species name in your simulation
    'species': 'He_electron',

    # Visualization settings
    'downsample_x': 10,         # Reduce data for faster plotting
    'density_vmin': 1e-4,       # Density color range (log scale)
    'density_vmax': 1.0,
    'field_vmax': 5.0,          # Field overlay range
    'ps_density_bins': (100, 100),  # Phase space density resolution
}
```

## How It Works

1. **Find frames**: Scans directory for SDF files matching the pattern
2. **Load data**: For each frame, loads 3 files:
   - `E_field*.sdf` → Electric field modes and grid
   - `dens*.sdf` → Electron density
   - `ener*.sdf` → Particle data (positions, momenta, weights)
3. **Normalize**: Converts to physical units (lengths in λ₀, density in n_c, etc.)
4. **Visualize**: Creates animated 2-panel plot with colorbars and lineouts
5. **Save**: Outputs HTML animation + NumPy arrays for reuse

## Output Files

In `post-process-output/`:
- `lwfa_q3d_animation.html` - Interactive animation (open in browser)
- `lwfa_q3d_*.npy` - Processed data arrays for custom plotting

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib
- `sdf_helper` (compiled from EPOCH)

## License

MIT License
