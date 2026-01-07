#!/usr/bin/env python3
"""
EPOCH Quasi-3D LWFA Post-Processing Script

Process EPOCH quasi-3D LWFA simulation data and generate 3-panel animated visualization.
Based on the working post-process-lwfa.py script structure.

Usage:
    1. Edit CONFIG below to match your simulation parameters
    2. Run: python process_epoch_lwfa.py

Output:
    - post-process-output/*.npy: Processed data arrays
    - post-process-output/lwfa_q3d_animation.html: Interactive visualization
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pipeline.loaders import EPOCHLoader
from pipeline.processors import DataProcessor
from pipeline.visualizers import EPOCHVisualizer
from utils.physics import compute_derived_quantities
from utils.hpc import progress_bar, print_section, print_parameter_summary, create_output_directory

# =============================================================================
# USER CONFIGURATION
# =============================================================================

CONFIG = {
    # Laser parameters
    'lambda0_um': 1.0,              # Laser wavelength in microns
    'a0': 3.0,                      # Normalized vector potential

    # Plasma parameters
    'n_over_nc': 0.0025,            # Plasma density as fraction of critical density

    # Data file (for synthetic data testing)
    'data_file': 'examples/data/epoch_lwfa_example.h5',

    # Processing options
    'downsample_x': 10,             # Spatial downsampling factor (x)
    'downsample_r': 1,              # Spatial downsampling factor (r)

    # Momentum histogram bins
    'n_px_bins': 2000,
    'px_min': 1,
    'px_max': 10000,

    # Visualization limits
    'density_vmin': 1e-4,           # Min for log-scale density plot
    'density_vmax': 1.0,            # Max for log-scale density plot
    'field_vmax': 5.0,              # Max for transverse field plot
    'r_max_lambda0': 100,           # Max r for plotting (in lambda0 units)
    'px_ylim': (1, 10000),          # Phase space y-axis limits

    # Animation settings
    'dumpstep_fs': 160,             # Time between dumps in femtoseconds
    'animation_interval': 50,       # ms between animation frames

    # Output settings
    'output_dir': 'post-process-output',
    'output_prefix': 'lwfa_q3d',

    # Particle species
    'species': 'He_electron',
}

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_single_frame(filepath, params, config):
    """
    Process a single frame.

    Args:
        filepath: Path to data file
        params: Dict from compute_derived_quantities()
        config: CONFIG dict

    Returns:
        Dict with processed frame data
    """
    loader = EPOCHLoader(filepath, use_sdf_helper=False)  # Use HDF5 for synthetic data

    frame = loader.load_frame(
        params=params,
        downsample_x=config['downsample_x'],
        downsample_r=config['downsample_r'],
        species=config['species']
    )

    loader.close()

    return frame


def process_all_frames(data_files, params, config):
    """
    Process all simulation frames.

    Args:
        data_files: List of file paths
        params: Physical parameters dict
        config: Configuration dict

    Returns:
        Dict with all processed arrays for animation
    """
    # Storage containers
    all_E_x = []
    all_E_tot = []
    all_n_e = []
    all_x = []
    all_r = []
    particle_data = []
    px_dist_mat = []

    # Momentum bins (log-spaced)
    px_bins = np.logspace(
        np.log10(config['px_min']),
        np.log10(config['px_max']),
        config['n_px_bins'] + 1
    )
    px_centers = 0.5 * (px_bins[:-1] + px_bins[1:])

    # Process frames
    processor = DataProcessor()

    print(f"Processing {len(data_files)} frame(s)...")
    for i, filepath in enumerate(data_files):
        progress_bar(i + 1, len(data_files), 'Loading')

        frame = process_single_frame(filepath, params, config)

        all_E_x.append(frame['E_x'])
        all_E_tot.append(frame['E_tot'])
        all_n_e.append(frame['n_e'])
        all_x.append(frame['x'])
        all_r.append(frame['r'])

        # Store particle data
        particle_data.append({
            'x_he': frame['x_particles'],
            'px_he': frame['px_particles'],
        })

        # Compute momentum histogram
        px_centers_hist, hist = processor.compute_momentum_histogram(
            frame['px_particles'],
            weights=frame['weight'],
            bins=px_bins
        )
        px_dist_mat.append(hist)

    return {
        'E_x': all_E_x,
        'E_tot': all_E_tot,
        'n_e': all_n_e,
        'x': all_x,
        'r': all_r,
        'particles': particle_data,
        'px_dist': px_dist_mat,
        'px_bins': px_bins,
        'px_centers': px_centers,
    }


def main():
    """Main entry point"""

    print_section("EPOCH Quasi-3D LWFA Post-Processing")

    # Compute derived quantities
    params = compute_derived_quantities(CONFIG)
    print("Physical Parameters:")
    print(f"  Laser wavelength: {params['lambda0']*1e6:.2f} μm")
    print(f"  Plasma density: {CONFIG['n_over_nc']:.4f} n_c")
    print(f"  Plasma wavelength: {params['lambda_pe']*1e6:.2f} μm")
    print(f"  Critical density: {params['n_c']:.2e} m^-3")
    print()

    # Find data files
    data_file = Path(CONFIG['data_file'])
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        print("Run generate_epoch_data.py first to create synthetic data.")
        sys.exit(1)

    data_files = [data_file]  # Single file for now
    print(f"Found {len(data_files)} frame(s) to process")
    print()

    # Create output directory
    output_dir = create_output_directory(CONFIG['output_dir'], verbose=True)

    # Process all frames
    data = process_all_frames(data_files, params, CONFIG)

    if len(data['x']) == 0:
        print("ERROR: No frames were successfully processed.")
        sys.exit(1)

    # Save processed data
    print("\nSaving processed data...")
    prefix = f"{CONFIG['output_dir']}/{CONFIG['output_prefix']}"
    np.save(f"{prefix}_E_x.npy", np.stack(data['E_x']))
    np.save(f"{prefix}_E_tot.npy", np.stack(data['E_tot']))
    np.save(f"{prefix}_n_e.npy", np.stack(data['n_e']))
    np.save(f"{prefix}_x.npy", np.stack(data['x']))
    np.save(f"{prefix}_r.npy", np.stack(data['r']))
    np.save(f"{prefix}_px_centers.npy", data['px_centers'])
    np.save(f"{prefix}_px_dist.npy", np.stack(data['px_dist']))
    print(f"  Saved .npy data files to {CONFIG['output_dir']}/")

    # Create and save animation
    print("\nCreating 3-panel animation...")
    visualizer = EPOCHVisualizer(output_dir=CONFIG['output_dir'])
    fig, anim = visualizer.create_3panel_animation(data, CONFIG, params)

    visualizer.save_animation_html(anim, CONFIG['output_prefix'] + '_animation')

    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)

    print_section("Processing Complete!")
    print()
    print(f"Outputs in: {CONFIG['output_dir']}/")
    print(f"  - {CONFIG['output_prefix']}_animation.html")
    print(f"  - {CONFIG['output_prefix']}_*.npy")
    print()


if __name__ == '__main__':
    main()
