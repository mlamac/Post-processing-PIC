# PIC Post-Processing Pipeline

Comprehensive data processing and visualization pipeline for **EPOCH quasi-3D** Particle-in-Cell (PIC) simulations, with focus on Laser Wakefield Acceleration (LWFA).

## Features

- **EPOCH Quasi-3D Support**: Native support for modal field decomposition and cylindrical geometry
- **3-Panel Animated Visualizations**: Density + field overlay, phase space, and momentum distribution
- **HPC-Compatible**: Agg backend, progress bars, batch processing optimized for cluster environments
- **Physical Normalization**: Proper normalization to laser wavelength, plasma frequency, and critical density
- **Publication-Quality Styling**: STIX fonts, logarithmic density plots, transparent field overlays

## Visualization Style

Matching the proven EPOCH post-processing workflow:

- **Density**: Grayscale with logarithmic normalization
- **E-field Overlay**: Semi-transparent PuRd colormap (pink-red) at 20% opacity
- **3-Panel Layout**:
  1. 2D density + transverse field (moving window supported)
  2. Longitudinal phase space (x-px scatter)
  3. Momentum distribution (dN/dpx line plot)

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/Post-processing-PIC.git
cd Post-processing-PIC
pip install -r requirements.txt
```

#### Dependencies
- Python 3.9+
- NumPy, SciPy, Matplotlib
- h5py (for synthetic data)
- **Optional**: `sdf_helper` (compiled from EPOCH for real SDF files)

To compile `sdf_helper`:
```bash
cd /path/to/epoch/epoch2d
make sdfutils
# Add to PYTHONPATH or copy to your Python site-packages
```

### 2. Generate Example Data

```bash
python examples/scripts/generate_epoch_data.py
```

This creates synthetic EPOCH quasi-3D LWFA data in `examples/data/epoch_lwfa_example.h5`.

### 3. Run the Pipeline

```bash
python examples/scripts/process_epoch_lwfa.py
```

**Outputs** (in `post-process-output/`):
- `lwfa_q3d_animation.html` - Interactive 3-panel animation
- `lwfa_q3d_*.npy` - Processed data arrays for reuse

## Usage

### Processing EPOCH Simulations

The main processing script mirrors the working `post-process-lwfa.py` structure:

```python
# Configure your simulation parameters
CONFIG = {
    # Laser parameters
    'lambda0_um': 1.0,              # Laser wavelength [μm]
    'a0': 3.0,                      # Normalized vector potential

    # Plasma parameters
    'n_over_nc': 0.0025,            # n_e / n_c

    # Visualization
    'density_vmin': 1e-4,
    'density_vmax': 1.0,
    'field_vmax': 5.0,
    'r_max_lambda0': 130,

    # Output
    'output_dir': 'post-process-output',
}
```

### Loading EPOCH Data

```python
from pipeline.loaders import EPOCHLoader
from utils.physics import compute_derived_quantities

# Compute physical parameters
params = compute_derived_quantities(CONFIG)

# Load a single frame
loader = EPOCHLoader('E_field0060.sdf', use_sdf_helper=True)
frame = loader.load_frame(params=params, downsample_x=10, downsample_r=1)

# Access data (all normalized)
x = frame['x']              # Grid in λ0 units
r = frame['r']
n_e = frame['n_e']          # Density / n_c
E_tot = frame['E_tot']      # Transverse field (normalized)
E_x = frame['E_x']          # Longitudinal field (normalized)
px = frame['px_particles']  # Momentum in m_e*c units
```

### Creating Visualizations

```python
from pipeline.visualizers import EPOCHVisualizer
from pipeline.processors import DataProcessor

# Process data
processor = DataProcessor()
px_bins, px_hist = processor.compute_momentum_histogram(
    px_particles, weights=weights, log_bins=True
)

# Create animation
visualizer = EPOCHVisualizer(output_dir='output')
fig, anim = visualizer.create_3panel_animation(data, CONFIG, params)
visualizer.save_animation_html(anim, 'my_lwfa_simulation')
```

## Project Structure

```
Post-processing-PIC/
├── src/
│   ├── pipeline/
│   │   ├── loaders.py          # EPOCH SDF/HDF5 data loading with modal reconstruction
│   │   ├── processors.py       # Data processing (histograms, beam stats)
│   │   └── visualizers.py      # 3-panel animation visualization
│   └── utils/
│       ├── physics.py          # Physical constants and normalization
│       ├── hpc.py              # HPC utilities (progress bars, file finding)
│       └── synthetic_data.py   # EPOCH quasi-3D synthetic data generator
├── examples/
│   ├── scripts/
│   │   ├── generate_epoch_data.py     # Generate synthetic test data
│   │   └── process_epoch_lwfa.py      # Complete processing pipeline
│   └── data/                          # Example datasets
├── working-script/
│   └── post-process-lwfa.py    # Reference implementation
├── requirements.txt
└── README.md
```

## Advanced Usage

### Batch Processing Multiple Files

```python
from utils.hpc import find_sdf_files, progress_bar

# Find all matching SDF files
indices = find_sdf_files('E_field*.sdf', directory='./sims/run01')

for i, idx in enumerate(indices):
    progress_bar(i+1, len(indices), 'Processing')
    filepath = f'E_field{idx:04d}.sdf'
    # Process frame...
```

### Custom Momentum Histograms

```python
# Log-spaced bins for wide dynamic range
px_bins = np.logspace(np.log10(1), np.log10(10000), 2000)
bin_centers, hist = processor.compute_momentum_histogram(
    px, weights=weights, bins=px_bins
)
```

### Beam Statistics

```python
stats = processor.compute_beam_statistics(px, pr, weights)
print(f"Mean energy: {stats['mean_energy_MeV']:.2f} MeV")
print(f"Energy spread: {stats['energy_spread']*100:.1f}%")
print(f"Total charge: {stats['total_charge_pC']:.2f} pC")
```

## Physical Normalization

The pipeline uses consistent normalization matching EPOCH conventions:

- **Lengths**: Normalized to laser wavelength λ₀
- **Density**: Normalized to critical density n_c = ε₀ m_e ω₀² / e²
- **Transverse E-fields**: Normalized to m_e c ω₀ / e
- **Longitudinal E-fields**: Normalized to m_e c ω_pe / e
- **Momentum**: Normalized to m_e c
- **Time**: Normalized to ω_pe⁻¹

All derived quantities are computed automatically from your laser and plasma parameters.

## Configuration Options

### Visualization Parameters

```python
CONFIG = {
    # Density plot (log scale)
    'density_vmin': 1e-4,       # Minimum density (n_e/n_c)
    'density_vmax': 1.0,        # Maximum density

    # Field overlay
    'field_vmax': 5.0,          # Max transverse field (normalized)

    # Spatial extent
    'r_max_lambda0': 130,       # Max radial extent [λ₀]

    # Phase space
    'px_ylim': (1, 10000),      # Momentum range [m_e c]

    # Momentum histogram
    'n_px_bins': 2000,          # Number of bins
    'px_min': 1,                # Min momentum
    'px_max': 10000,            # Max momentum

    # Animation
    'animation_interval': 50,   # ms between frames
}
```

### Downsampling for Performance

```python
# Reduce data size for faster plotting
CONFIG['downsample_x'] = 10   # Keep every 10th point in x
CONFIG['downsample_r'] = 1    # Keep all points in r
```

## Output Files

### Processed Data (.npy files)

The pipeline saves intermediate results for quick replotting:

- `lwfa_q3d_E_x.npy` - Longitudinal electric field
- `lwfa_q3d_E_tot.npy` - Transverse electric field magnitude
- `lwfa_q3d_n_e.npy` - Electron density
- `lwfa_q3d_x.npy`, `lwfa_q3d_r.npy` - Grid coordinates
- `lwfa_q3d_px_centers.npy` - Momentum bin centers
- `lwfa_q3d_px_dist.npy` - Momentum distributions

### Animation (HTML)

Self-contained HTML file with embedded JavaScript animation. Can be:
- Viewed in any web browser
- Shared via email/cloud
- Embedded in websites or presentations

## Working with Real EPOCH Data

When processing actual EPOCH simulations:

1. **Set up file patterns** in CONFIG:
   ```python
   CONFIG['dens_pattern'] = 'dens*.sdf'
   CONFIG['efield_pattern'] = 'E_field{:04d}.sdf'
   CONFIG['ener_pattern'] = 'ener{:04d}.sdf'
   ```

2. **Run in simulation directory**:
   ```bash
   cd /path/to/epoch/output
   cp /path/to/process_epoch_lwfa.py .
   # Edit CONFIG in script
   python process_epoch_lwfa.py
   ```

3. **Enable sdf_helper**:
   ```python
   loader = EPOCHLoader('E_field0060.sdf', use_sdf_helper=True)
   ```

## Quasi-3D Modal Field Reconstruction

EPOCH quasi-3D stores fields as modal coefficients:

F(x,r,θ) = F₀(x,r) + F₁(x,r)cos(θ) + F₁ᵢ(x,r)sin(θ)

The pipeline automatically reconstructs physical fields at θ=0 (y-direction) and θ=π/2 (z-direction):

```python
from pipeline.loaders import reconstruct_field_from_modes

E_y = reconstruct_field_from_modes(Erm_real, Erm_imag, theta=0)
E_z = reconstruct_field_from_modes(Erm_real, Erm_imag, theta=np.pi/2)
```

## Troubleshooting

### Import Errors

If you see import errors, ensure `src/` is in your Python path:
```python
import sys
sys.path.insert(0, '/path/to/Post-processing-PIC/src')
```

### sdf_helper Not Found

For synthetic data testing, the pipeline falls back to HDF5:
```python
loader = EPOCHLoader(filepath, use_sdf_helper=False)
```

For real EPOCH data, compile sdf_helper from EPOCH source.

### Memory Issues with Large Simulations

- Increase downsampling factors
- Process frames individually instead of loading all at once
- Use lower resolution for quick previews

## Contributing

Contributions welcome! Priority areas:
- Support for additional PIC codes (Smilei, WarpX, OSIRIS)
- 3D visualization capabilities
- Advanced beam diagnostics (emittance, betatron radiation)
- Parameter scan automation tools

## License

MIT License - see LICENSE file for details

## Citation

If you use this pipeline in your research:

```bibtex
@software{pic_postprocessing_epoch,
  title = {EPOCH PIC Post-Processing Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Post-processing-PIC}
}
```

## Acknowledgments

Built for processing EPOCH quasi-3D laser wakefield acceleration simulations.
Based on proven HPC post-processing workflows.
