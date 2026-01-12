# EPOCH Post-Processing Pipeline

A Python pipeline for visualizing **EPOCH quasi-3D** laser wakefield acceleration (LWFA) simulations. Generates animated 3-panel visualizations showing electron density, phase space, and energy spectra.

## What It Does

Takes raw EPOCH SDF output files and creates an interactive HTML animation:

| Panel | Shows |
|-------|-------|
| Top | Electron density (grayscale) with laser field overlay (pink) |
| Middle | Longitudinal phase space (x vs momentum) |
| Bottom | Momentum distribution |

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

Copy the processing script to your EPOCH output directory and run:

```bash
cd /path/to/your/epoch/output
cp /path/to/Post-processing-PIC/examples/scripts/process_epoch_lwfa.py .
python process_epoch_lwfa.py
```

### 3. View Results

Open `post-process-output/lwfa_q3d_animation.html` in a web browser.

## Configuration

Edit the `CONFIG` dictionary in `process_epoch_lwfa.py`:

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
}
```

## Project Structure

```
Post-processing-PIC/
├── src/pipeline/
│   ├── loaders.py      # Load and normalize EPOCH SDF data
│   ├── processors.py   # Compute histograms and statistics
│   └── visualizers.py  # Create 3-panel animations
├── src/utils/
│   ├── physics.py      # Physical constants and derived quantities
│   └── hpc.py          # Progress bars and file utilities
├── examples/scripts/
│   └── process_epoch_lwfa.py   # Main script (copy this to use)
└── working-script/
    └── post-process-lwfa.py    # Reference implementation
```

## How It Works

1. **Find frames**: Scans directory for SDF files matching the pattern
2. **Load data**: For each frame, loads 3 files:
   - `E_field*.sdf` → Electric field modes and grid
   - `dens*.sdf` → Electron density
   - `ener*.sdf` → Particle data (positions, momenta, weights)
3. **Normalize**: Converts to physical units (lengths in λ₀, density in n_c, etc.)
4. **Visualize**: Creates animated 3-panel plot
5. **Save**: Outputs HTML animation + NumPy arrays for reuse

## Output Files

In `post-process-output/`:
- `lwfa_q3d_animation.html` - Interactive animation (open in browser)
- `lwfa_q3d_*.npy` - Processed data arrays for custom plotting

## Using the Pipeline in Your Own Code

```python
from pipeline.loaders import load_frame_data, find_sdf_files
from pipeline.processors import compute_momentum_histogram
from pipeline.visualizers import create_animation, save_animation_html
from utils.physics import compute_derived_quantities

# Setup
params = compute_derived_quantities(CONFIG)
indices = find_sdf_files('dens*.sdf')

# Load a single frame
frame = load_frame_data(indices[0], params, CONFIG)
# Returns: E_x, E_tot, n_e, x, r, x_he, px_he, w_he

# Compute histogram
bins = np.logspace(0, 4, 2001)
hist = compute_momentum_histogram(frame['px_he'], frame['w_he'], bins)
```

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib, h5py
- `sdf_helper` (compiled from EPOCH)

## License

MIT License
