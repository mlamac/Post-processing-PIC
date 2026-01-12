# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Post-processing pipeline for **EPOCH quasi-3D** Particle-in-Cell (PIC) simulations, focused on Laser Wakefield Acceleration (LWFA). Transforms raw SDF simulation data into publication-quality 3-panel animated visualizations showing plasma density, phase space, and momentum distributions.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the processing pipeline (in directory with SDF files)
python examples/scripts/process_epoch_lwfa.py

# Development install with test/lint tools
pip install -e ".[dev]"
```

**Usage:** Copy `process_epoch_lwfa.py` to the directory containing your EPOCH SDF files, edit CONFIG to match your simulation parameters, and run. Requires `sdf_helper` module compiled from EPOCH.

## Architecture

```
Raw EPOCH SDF → loaders.py → processors.py → visualizers.py → HTML animation + .npy files
```

### Core Modules (src/pipeline/)

- **loaders.py**: Function-based SDF loading. Key functions:
  - `load_frame_data(index, params, config)`: Loads 3 SDF files (E_field, dens, ener), returns normalized, transposed, downsampled data
  - `find_sdf_files(pattern)`: Returns sorted frame indices from filename pattern
  - `reconstruct_field_from_modes()`: Quasi-3D modal field reconstruction

- **processors.py**: Data analysis functions:
  - `compute_momentum_histogram(px, weights, bins)`: Weighted histogram
  - `compute_beam_statistics(px, pr, weights)`: Energy spread, divergence, charge

- **visualizers.py**: Animation functions:
  - `create_animation(data, params, config)`: 3-panel animation (density+field, phase space, momentum)
  - `save_animation_html(anim, output_path)`: Save as standalone HTML

### Utilities (src/utils/)

- **physics.py**: Physical constants (SI), `compute_derived_quantities(config)` for laser/plasma parameters
- **hpc.py**: `progress_bar()`, `find_sdf_files()`, output directory helpers

## Key Concepts

### Critical: Normalization Order

The pipeline normalizes transverse fields (E_y, E_z) **before** computing E_tot:
```python
E_y = reconstruct_field_from_modes(...) / params['E_norm_laser']  # Normalize first
E_z = reconstruct_field_from_modes(...) / params['E_norm_laser']  # Normalize first
E_tot = np.sqrt(E_y**2 + E_z**2)  # Then compute magnitude
```

### Quasi-3D Modal Reconstruction

EPOCH quasi-3D stores fields as Fourier modes: `F(x,r,θ) = F₀(x,r) + F₁(x,r)cos(θ) + F₁ᵢ(x,r)sin(θ)`

### Normalization (EPOCH conventions)

| Quantity | Normalization | Notes |
|----------|--------------|-------|
| Length | λ₀ | Laser wavelength |
| Density | n_c | Critical density |
| Transverse E-field | m_e c ω₀ / e | `E_norm_laser` |
| Longitudinal E-field | m_e c ωₚₑ / e | `E_norm_plasma` |
| Momentum | m_e c | Relativistic units |

### SDF Field Names

- Electric fields: `Electric_Field_Modes_Erm_real/imag`, `Electric_Field_Modes_Exm_real/imag`
- Density: `Derived_Number_Density_Subset_total_e`
- Grid: `Grid_Grid_mid.data[0]` (x), `Grid_Grid_mid.data[1]` (r)
- Particles: `Grid_Particles_{species}`, `Particles_Px_{species}`, `Particles_Weight_{species}`

## Configuration

All config is dict-based in `process_epoch_lwfa.py`:
```python
CONFIG = {
    'lambda0_um': 1.0,              # Laser wavelength [μm]
    'a0': 3.0,                      # Normalized vector potential
    'n_over_nc': 0.0025,            # Density ratio n_e/n_c
    'efield_pattern': 'E_field{:04d}.sdf',
    'ener_pattern': 'ener{:04d}.sdf',
    'downsample_x': 10,
    'species': 'He_electron',
    # ...
}
```

## Working with This Codebase

- Reference implementation: `working-script/post-process-lwfa.py` (proven working, use as template)
- Output: `post-process-output/` with `.html` animation and `.npy` cache files
- HPC-ready: Matplotlib Agg backend, text progress bars
