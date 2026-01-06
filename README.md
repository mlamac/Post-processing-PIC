# PIC Post-Processing Pipeline

A comprehensive data processing and visualization pipeline for Particle-in-Cell (PIC) simulation data, with focus on Laser Wakefield Acceleration (LWFA) simulations.

## Features

- **Multi-format Support**: Load data from EPOCH (SDF) and Smilei (HDF5) simulations
- **Rich Visualizations**: Generate both interactive HTML plots and publication-quality PNG figures
- **Custom Colormaps**: Specialized colormaps for plasma physics (grayscale density, transparent divergent fields)
- **Comprehensive Analysis**: Energy spectra, phase space plots, field distributions, beam diagnostics
- **Synthetic Data**: Built-in LWFA data generator for testing and examples
- **Easy to Use**: Simple API with sensible defaults

## Installation

### Requirements

- Python 3.9 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/yourusername/Post-processing-PIC.git
cd Post-processing-PIC
pip install -e .
```

### Dependencies

Core dependencies are automatically installed:
- numpy, scipy, pandas
- matplotlib, plotly
- h5py (for HDF5 files)
- sdf (for EPOCH SDF files)

For Jupyter notebook support:
```bash
pip install jupyter ipywidgets
```

## Quick Start

### 1. Generate Example Data

```bash
python examples/scripts/generate_example_data.py
```

This creates synthetic LWFA simulation data in `examples/data/example_lwfa.h5`.

### 2. Run the Processing Pipeline

```bash
python examples/scripts/process_lwfa_data.py
```

This generates visualizations in the `output/` directory:
- Interactive HTML plots for exploration
- High-quality PNG images for publications

### 3. Explore with Jupyter

```bash
jupyter notebook examples/notebooks/lwfa_analysis_example.ipynb
```

## Usage Examples

### Loading Data

```python
from pipeline.loaders import load_data

# Auto-detect format (SDF or HDF5)
loader = load_data('simulation_output.h5')

# Load fields
density_data = loader.load_field('Rho_electron')
n_e = density_data['data']
x = density_data['grid_x']
y = density_data['grid_y']

# Load particles
particles = loader.load_particles('electron')
x_particles = particles['x']
px_particles = particles['px']
```

### Processing Data

```python
from pipeline.processors import DataProcessor

processor = DataProcessor()

# Compute particle energies
energy_MeV = processor.compute_particle_energy(
    particles['px'], particles['py'], particles['pz']
)

# Compute energy spectrum
energy_bins, spectrum = processor.compute_energy_spectrum(
    particles['px'], particles['py'], particles['pz'],
    weight=particles['weight']
)

# Calculate total charge
charge_pC = processor.compute_charge(particles['weight'])
```

### Creating Visualizations

```python
from pipeline.visualizers import Visualizer

visualizer = Visualizer(output_dir='output')

# Density with E-field overlay (grayscale + transparent seismic)
visualizer.plot_density_with_field_overlay(
    x=x, y=y,
    density=n_e,
    field=E_y,
    output_name='density_field',
    save_png=True,
    save_html=True
)

# Phase space plot
visualizer.plot_phase_space(
    x=particles['x'],
    px=particles['px'],
    weight=particles['weight'],
    output_name='phase_x_px'
)

# Energy spectrum
visualizer.plot_energy_spectrum(
    energy=energy_bins,
    spectrum=spectrum,
    output_name='spectrum'
)
```

## Supported File Formats

### EPOCH (SDF Format)

EPOCH is a PIC code that outputs data in SDF format. Fields and particles are accessed by their standard names:

```python
loader = load_data('output.sdf', format='sdf')

# Fields: 'Electric_Field_Ex', 'Electric_Field_Ey', 'Derived_Number_Density_electron', etc.
field = loader.load_field('Electric_Field_Ex')

# Particles: 'electron', 'proton', etc.
particles = loader.load_particles('electron')
```

### Smilei (HDF5 Format)

Smilei outputs HDF5 files with a structured hierarchy:

```python
loader = load_data('output.h5', format='hdf5')

# Fields stored in /Fields/ or /Rho/
field = loader.load_field('Ex')
density = loader.load_field('Rho_electron')

# Particles stored in /Species/
particles = loader.load_particles('electron')
```

## Visualization Outputs

### Interactive HTML Plots

- Zoomable and pannable
- Hover information
- Export to PNG from browser
- Shareable and embeddable

### Publication-Quality PNGs

- Customizable DPI (default: 150)
- Vector-like quality
- Ready for manuscript submission
- Configurable figure size

### Custom Colormaps

#### Grayscale Density
Standard grayscale colormap for electron density visualization, providing clear contrast for plasma structures.

#### Transparent Seismic Field Overlay
Divergent colormap (blue-to-red) with transparency near zero values. Perfect for overlaying transverse electric fields on density maps:
- Blue: Negative field values
- Transparent: Near-zero values (doesn't obscure density)
- Red: Positive field values

## Project Structure

```
Post-processing-PIC/
├── src/
│   ├── pipeline/
│   │   ├── loaders.py          # Data loading (SDF, HDF5)
│   │   ├── processors.py       # Data processing & analysis
│   │   └── visualizers.py      # Visualization generation
│   └── utils/
│       └── synthetic_data.py   # Synthetic LWFA data generator
├── examples/
│   ├── scripts/
│   │   ├── generate_example_data.py
│   │   └── process_lwfa_data.py
│   ├── notebooks/
│   │   └── lwfa_analysis_example.ipynb
│   └── data/                   # Example datasets
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── output/                     # Default output directory
├── requirements.txt
├── setup.py
└── README.md
```

## Advanced Usage

### Batch Processing Multiple Files

```python
from pathlib import Path
from pipeline.loaders import load_data
from pipeline.visualizers import Visualizer

data_dir = Path('simulations')
visualizer = Visualizer(output_dir='batch_output')

for data_file in data_dir.glob('*.h5'):
    loader = load_data(data_file)
    density = loader.load_field('Rho_electron')

    output_name = data_file.stem
    visualizer.plot_density_with_field_overlay(
        x=density['grid_x'],
        y=density['grid_y'],
        density=density['data'],
        field=loader.load_field('Ey')['data'],
        output_name=output_name
    )
```

### Custom Colormap Settings

```python
# Adjust field overlay transparency and smoothing
visualizer.plot_density_with_field_overlay(
    x=x, y=y,
    density=n_e,
    field=E_y,
    smooth_density=1.0,      # Gaussian smoothing sigma
    smooth_field=0.5,         # Less smoothing for fields
    field_symmetric=True,     # Symmetric colormap around zero
    figsize=(16, 8),          # Larger figure
    dpi=300                   # Higher resolution
)
```

### Energy Spectrum Analysis

```python
# Compute spectrum with custom bins
energy_bins, spectrum = processor.compute_energy_spectrum(
    particles['px'], particles['py'], particles['pz'],
    weight=particles['weight'],
    bins=100,
    energy_range=(0, 500)  # MeV
)

# Calculate statistics
peak_energy = energy_bins[np.argmax(spectrum)]
mean_energy = np.average(energy_bins, weights=spectrum)
print(f"Peak energy: {peak_energy:.2f} MeV")
print(f"Mean energy: {mean_energy:.2f} MeV")
```

## Synthetic Data Generator

Generate realistic LWFA simulation data for testing:

```python
from utils.synthetic_data import generate_lwfa_data

data = generate_lwfa_data(
    output_path='test_data.h5',
    nx=400,                    # Grid points in x
    ny=200,                    # Grid points in y
    n_particles=10000,         # Number of particles
    plasma_density=1e24,       # Background density (m^-3)
    laser_a0=2.0,              # Normalized laser amplitude
    seed=42                    # Random seed
)
```

The generator creates:
- Electron density with wakefield structure (bubble regime)
- Transverse electric field (laser + wakefield)
- Longitudinal electric field (acceleration field)
- Accelerated electron beam with realistic energy distribution

## Contributing

Contributions are welcome! Areas for improvement:
- Additional PIC code support (WarpX, OSIRIS, FBPIC)
- More analysis tools (emittance, beam quality)
- 3D visualization support
- Animation generation
- Performance optimization

## License

MIT License - see LICENSE file for details

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{pic_postprocessing,
  title = {PIC Post-Processing Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Post-processing-PIC}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: your.email@example.com

## Acknowledgments

Developed for processing laser wakefield acceleration simulation data from EPOCH and Smilei PIC codes.
