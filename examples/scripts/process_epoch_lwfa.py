#!/usr/bin/env python3
"""
EPOCH Quasi-3D LWFA Post-Processing Script

Process EPOCH quasi-3D LWFA simulation data and generate 3-panel animated visualization.
Based on the working post-process-lwfa.py script.

Usage:
    1. Copy this script to the directory containing your SDF output files
    2. Edit CONFIG below to match your simulation parameters
    3. Run: python process_epoch_lwfa.py

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

from pipeline.loaders import load_frame_data, find_sdf_files
from pipeline.processors import compute_momentum_histogram
from pipeline.visualizers import create_animation, save_animation_html
from utils.physics import compute_derived_quantities
from utils.hpc import progress_bar

# =============================================================================
# USER CONFIGURATION
# =============================================================================

CONFIG = {
    # Laser parameters
    'lambda0_um': 1.0,              # Laser wavelength in microns
    'a0': 3.0,                      # Normalized vector potential

    # Plasma parameters
    'n_over_nc': 0.0025,            # Plasma density as fraction of critical density

    # File patterns (script runs in directory with SDF files)
    'dens_pattern': 'dens*.sdf',
    'efield_pattern': 'E_field{:04d}.sdf',
    'ener_pattern': 'ener{:04d}.sdf',

    # Processing options
    'downsample_x': 10,             # Spatial downsampling factor (x)
    'downsample_r': 1,              # Spatial downsampling factor (r)

    # Particle species
    'species': 'He_electron',       # Adjust for your simulation

    # Momentum histogram bins
    'n_px_bins': 2000,
    'px_min': 1,
    'px_max': 10000,

    # Visualization limits
    'density_vmin': 1e-4,           # Min for log-scale density plot
    'density_vmax': 1.0,            # Max for log-scale density plot
    'field_vmax': 5.0,              # Max for transverse field plot
    'r_max_lambda0': 130,           # Max r for plotting (in lambda0 units)
    'px_ylim': (1, 10000),          # Phase space y-axis limits

    # Animation settings
    'dumpstep_fs': 160,             # Time between dumps in femtoseconds
    'animation_interval': 50,       # ms between animation frames

    # Output settings
    'output_dir': 'post-process-output',
    'output_prefix': 'lwfa_q3d',
}


def process_all_frames(indices, params, config):
    """
    Process all simulation frames and collect data.

    Returns dict containing all processed arrays for animation.
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

    # Process each frame
    print(f"Processing {len(indices)} frames...")
    for i, idx in enumerate(indices):
        progress_bar(i + 1, len(indices), 'Loading')

        frame = load_frame_data(idx, params, config)
        if frame is None:
            continue

        all_E_x.append(frame['E_x'])
        all_E_tot.append(frame['E_tot'])
        all_n_e.append(frame['n_e'])
        all_x.append(frame['x'])
        all_r.append(frame['r'])

        particle_data.append({
            'x_he': frame['x_he'],
            'px_he': frame['px_he'],
        })

        # Compute momentum histogram
        hist = compute_momentum_histogram(frame['px_he'], frame['w_he'], px_bins)
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
    """Main entry point."""
    print("=" * 60)
    print("EPOCH Quasi-3D LWFA Post-Processing")
    print("=" * 60)

    # Compute derived quantities
    params = compute_derived_quantities(CONFIG)
    print(f"Laser wavelength: {params['lambda0']*1e6:.2f} um")
    print(f"Plasma density: {CONFIG['n_over_nc']:.4f} n_c")
    print(f"Plasma wavelength: {params['lambda_pe']*1e6:.2f} um")
    print(f"Plasma frequency: {params['omega_pe']:.2e} rad/s")
    print()

    # Find SDF files
    indices = find_sdf_files(CONFIG['dens_pattern'])
    if not indices:
        print(f"ERROR: No SDF files found matching pattern: {CONFIG['dens_pattern']}")
        print("Make sure you are running this script in the directory with SDF files.")
        sys.exit(1)
    print(f"Found {len(indices)} frames: {indices[0]:04d} to {indices[-1]:04d}")
    print()

    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Process all frames
    data = process_all_frames(indices, params, CONFIG)

    if len(data['x']) == 0:
        print("ERROR: No frames were successfully processed.")
        sys.exit(1)

    # Save processed data as .npy files
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
    print("\nCreating animation...")
    fig, anim = create_animation(data, params, CONFIG)

    html_path = f"{prefix}_animation.html"
    save_animation_html(anim, html_path)

    import matplotlib.pyplot as plt
    plt.close(fig)

    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
