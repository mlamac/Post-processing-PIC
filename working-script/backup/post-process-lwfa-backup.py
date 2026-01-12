#!/usr/bin/env python3
"""
EPOCH Quasi-3D LWFA Post-Processing Script

HPC-compatible script for processing laser wakefield acceleration simulations
from EPOCH PIC code in quasi-3D geometry. Generates interactive HTML visualizations
showing the moving window, longitudinal phase space, and electron energy spectrum.

Usage:
    1. Copy this script to the directory containing your SDF output files
    2. Edit the CONFIG section below to match your simulation parameters
    3. Run: python post-process-lwfa.py

Output:
    - post-process-output/*.npy: Processed data arrays
    - post-process-output/lwfa_q3d_animation.html: Interactive visualization
"""

# =============================================================================
# USER CONFIGURATION - Modify these parameters for your simulation
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
    'dumpstep_fs': 160,              # Time between dumps in femtoseconds
    'animation_interval': 50,       # ms between animation frames

    # Output settings
    'output_dir': 'post-process-output',
    'output_prefix': 'lwfa_q3d',
}

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import matplotlib as mpl
mpl.use('Agg')  # HPC-compatible non-GUI backend - must be before pyplot import

import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import scipy.ndimage

# Import sdf_helper with error handling
try:
    import sdf_helper as sh
except ImportError:
    print("ERROR: sdf_helper module not found.")
    print("Please compile with: cd /path/to/epoch/epoch2d && make sdfutils")
    sys.exit(1)

# Configure matplotlib for publication-quality figures
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams["mathtext.fontset"] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['animation.embed_limit'] = 2**31 - 1  # Large animation limit for HPC

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# SI units
C_LIGHT = 2.99792458e8              # Speed of light [m/s]
M_E = 9.1093837e-31                 # Electron mass [kg]
Q_E = 1.60217663e-19                # Elementary charge [C]
EPS_0 = 8.8541878188e-12            # Vacuum permittivity [F/m]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_derived_quantities(config):
    """
    Compute derived plasma and laser parameters from configuration.

    Returns dict with:
        lambda0, omega_0: Laser wavelength and frequency
        n_c, n_0: Critical and plasma densities
        omega_pe, lambda_pe: Plasma frequency and wavelength
        E_norm_laser, E_norm_plasma: Field normalization factors
        dumpstep: Time step in normalized units
    """
    lambda0 = config['lambda0_um'] * 1e-6
    omega_0 = 2 * np.pi * C_LIGHT / lambda0

    # Critical density
    n_c = EPS_0 * M_E * omega_0**2 / Q_E**2
    n_0 = config['n_over_nc'] * n_c

    # Plasma parameters
    omega_pe = np.sqrt(Q_E**2 * n_0 / (EPS_0 * M_E))
    lambda_pe = 2 * np.pi * C_LIGHT / omega_pe

    # Normalization factors
    E_norm_laser = M_E * C_LIGHT * omega_0 / Q_E    # For transverse fields
    E_norm_plasma = M_E * C_LIGHT * omega_pe / Q_E  # For longitudinal fields

    # Time step
    dumpstep = config['dumpstep_fs'] * 1e-15 * omega_pe

    return {
        'lambda0': lambda0,
        'omega_0': omega_0,
        'n_c': n_c,
        'n_0': n_0,
        'omega_pe': omega_pe,
        'lambda_pe': lambda_pe,
        'E_norm_laser': E_norm_laser,
        'E_norm_plasma': E_norm_plasma,
        'dumpstep': dumpstep,
    }


def find_sdf_files(pattern):
    """
    Find SDF files matching pattern and return sorted list of indices.

    Args:
        pattern: Glob pattern like 'dens*.sdf'

    Returns:
        Sorted list of integer indices (e.g., [60, 120, 180])
    """
    files = glob.glob(pattern)
    indices = []

    # Extract frame number from filename (e.g., dens0060.sdf -> 60)
    for f in files:
        match = re.search(r'(\d{4})\.sdf$', os.path.basename(f))
        if match:
            indices.append(int(match.group(1)))

    return sorted(indices)


def reconstruct_field_from_modes(mode_real, mode_imag, theta=0):
    """
    Reconstruct field from quasi-3D modal data at given azimuthal angle.

    For quasi-3D simulations, fields are stored as modal coefficients:
        F(x,r,theta) = F_0(x,r) + F_1(x,r)*cos(theta) + F_1i(x,r)*sin(theta)

    Args:
        mode_real: Array with shape (nx, nr, n_modes) - real parts
        mode_imag: Array with shape (nx, nr, n_modes) - imaginary parts
        theta: Azimuthal angle in radians (0 for y-direction, pi/2 for z)

    Returns:
        Reconstructed 2D field at the specified theta
    """
    # Mode 0 (axisymmetric) + Mode 1 with azimuthal dependence
    field = (mode_real[:, :, 0] +
             mode_real[:, :, 1] * np.cos(theta) +
             mode_imag[:, :, 1] * np.sin(theta))
    return field


def load_frame_data(index, params, config):
    """
    Load all data for a single simulation frame.

    Args:
        index: Frame index (e.g., 60 for E_field0060.sdf)
        params: Dict from compute_derived_quantities()
        config: CONFIG dict

    Returns:
        Dict with field, density, and particle data, or None if failed
    """
    try:
        # Load SDF files
        data_E = sh.getdata(config['efield_pattern'].format(index))
        data_dens = sh.getdata('dens{:04d}.sdf'.format(index))
        data_ener = sh.getdata(config['ener_pattern'].format(index))

        # Reconstruct transverse fields from modes
        # E_y at theta=0 (y-direction)
        E_y = reconstruct_field_from_modes(
            data_E.Electric_Field_Modes_Erm_real.data,
            data_E.Electric_Field_Modes_Erm_imag.data,
            theta=0
        ) / params['E_norm_laser']

        # E_z at theta=pi/2 (z-direction)
        E_z = reconstruct_field_from_modes(
            data_E.Electric_Field_Modes_Erm_real.data,
            data_E.Electric_Field_Modes_Erm_imag.data,
            theta=np.pi/2
        ) / params['E_norm_laser']

        # Longitudinal field E_x
        E_x = reconstruct_field_from_modes(
            data_E.Electric_Field_Modes_Exm_real.data,
            data_E.Electric_Field_Modes_Exm_imag.data,
            theta=0
        ) / params['E_norm_plasma']

        # Total transverse field magnitude
        E_tot = np.sqrt(E_y**2 + E_z**2)

        # Electron density normalized to critical density
        n_e = data_dens.Derived_Number_Density_Subset_total_e.data / params['n_c']

        # Grid coordinates normalized to laser wavelength
        x = data_E.Grid_Grid_mid.data[0] / params['lambda0']
        r = data_E.Grid_Grid_mid.data[1] / params['lambda0']

        # Apply downsampling
        ds_x = config['downsample_x']
        ds_r = config['downsample_r']

        # Particle data (He_electron species)
        x_he = np.squeeze(data_ener.Grid_Particles_He_electron.data[0]) / params['lambda0']
        px_he = np.squeeze(data_ener.Particles_Px_He_electron.data) / (M_E * C_LIGHT)
        w_he = np.squeeze(data_ener.Particles_Weight_He_electron.data)

        return {
            'E_x': E_x[::ds_x, ::ds_r].T,
            'E_y': E_y[::ds_x, ::ds_r].T,
            'E_tot': E_tot[::ds_x, ::ds_r].T,
            'n_e': n_e[::ds_x, ::ds_r].T,
            'x': x[::ds_x],
            'r': r[::ds_r],
            'x_he': x_he.flatten() if x_he.size else np.array([]),
            'px_he': px_he.flatten() if px_he.size else np.array([]),
            'w_he': w_he.flatten() if w_he.size else np.array([]),
        }

    except Exception as e:
        print(f"WARNING: Failed to load frame {index}: {e}")
        return None


def compute_momentum_histogram(px, weights, bins):
    """Compute weighted momentum histogram."""
    hist, _ = np.histogram(px, bins=bins, weights=weights)
    return hist


def progress_bar(current, total, prefix='Progress'):
    """Simple text progress bar for HPC jobs."""
    pct = 100 * current / total
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = '=' * filled + '-' * (bar_len - filled)
    print(f'\r{prefix}: [{bar}] {pct:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()  # Newline at end


# =============================================================================
# MAIN PROCESSING
# =============================================================================

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


def create_animation(data, params, config):
    """
    Create 3-panel animation figure.

    Panel 1: 2D density and field (pcolormesh)
    Panel 2: Longitudinal phase space (scatter)
    Panel 3: Momentum distribution (line plot)
    """
    n_frames = len(data['x'])
    dumpstep = params['dumpstep']
    a0 = config['a0']

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(3.54, 2*3.54), dpi=200,
        gridspec_kw={'height_ratios': [1.75, 1, 1]}
    )

    # Initial data
    x0, r0 = data['x'][0], data['r'][0]

    # Panel 1: 2D density and field
    plot_dens = ax1.pcolormesh(
        x0, r0, data['n_e'][0],
        cmap='Greys',
        norm=colors.LogNorm(vmin=config['density_vmin'], vmax=config['density_vmax']),
        shading='auto'
    )
    plot_field = ax1.pcolormesh(
        x0, r0, data['E_tot'][0],
        cmap='PuRd', vmin=0, vmax=config['field_vmax'],
        alpha=0.2, shading='auto'
    )
    ax1.set_ylabel(r"$r\: /\: \lambda_{0}$")
    ax1.set_xlabel(r"$x\: /\: \lambda_{0}$")
    ax1.set_ylim(r0[0], config['r_max_lambda0'])
    ax1.set_title(
        f"$a_{{0}} = {a0:.1f},\\ \\omega_{{pe}} t = {0:.0f}$",
        loc='left', fontsize=12
    )

    # Panel 2: Phase space scatter
    pdata = data['particles'][0]
    scatter_he = ax2.scatter(
        pdata['x_he'], pdata['px_he'],
        s=2, c='r', alpha=0.5, marker='.', linewidths=0
    )
    ax2.set_ylabel(r"$p_x\: /\: m_{e}c$")
    ax2.set_xlabel(r"$x\: /\: \lambda_{0}$")
    ax2.set_ylim(config['px_ylim'])
    ax2.set_xlim(x0[0], x0[-1])

    legend_handles = [Line2D([0], [0], color='red', lw=1, label='Plasma e-')]
    ax2.legend(handles=legend_handles, frameon=True, framealpha=0, loc="upper left")

    # Panel 3: Momentum distribution
    px_centers = data['px_centers']
    line_dist, = ax3.plot(px_centers, data['px_dist'][0], color='red', label='Plasma e-')
    ax3.set_xlabel(r"$p_x\: /\: m_e c$")
    ax3.set_ylabel(r"$dN\:/\:dp_x$")
    ax3.set_xlim(px_centers[0], px_centers[-1])

    # Auto-scale y-axis based on max histogram value
    max_hist = max(np.max(h) for h in data['px_dist'] if len(h) > 0)
    ax3.set_ylim(0, max_hist * 0.5)

    ax3.legend(frameon=True, framealpha=0, loc="upper right")

    fig.tight_layout()

    # Store plot objects that need to be recreated each frame
    plot_objects = {'dens': plot_dens, 'field': plot_field}

    def animate(frame):
        x = data['x'][frame]
        r = data['r'][frame]

        # Update Panel 1: density and field
        # Remove old pcolormesh and create new ones (needed for moving window)
        plot_objects['dens'].remove()
        plot_objects['field'].remove()

        plot_objects['dens'] = ax1.pcolormesh(
            x, r, data['n_e'][frame],
            cmap='Greys',
            norm=colors.LogNorm(vmin=config['density_vmin'], vmax=config['density_vmax']),
            shading='auto'
        )
        plot_objects['field'] = ax1.pcolormesh(
            x, r, data['E_tot'][frame],
            cmap='PuRd', vmin=0, vmax=config['field_vmax'],
            alpha=0.2, shading='auto'
        )

        ax1.set_xlim(x[0], x[-1])
        ax1.set_title(
            f"$a_{{0}} = {a0:.1f},\\ \\omega_{{pe}} t = {frame * dumpstep:.0f}$",
            loc='left', fontsize=12
        )

        # Update Panel 2: phase space
        pdata = data['particles'][frame]
        if pdata['x_he'].size and pdata['px_he'].size:
            he_data = np.column_stack((pdata['x_he'], pdata['px_he']))
            scatter_he.set_offsets(he_data)
        else:
            scatter_he.set_offsets(np.empty((0, 2)))
        ax2.set_xlim(x[0], x[-1])

        # Update Panel 3: momentum distribution
        line_dist.set_ydata(data['px_dist'][frame])

        return plot_objects['dens'], plot_objects['field'], scatter_he, line_dist

    anim = FuncAnimation(
        fig, animate, frames=n_frames,
        interval=config['animation_interval'], blit=False, repeat=True
    )

    return fig, anim


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

    html_output = anim.to_jshtml()
    html_path = f"{prefix}_animation.html"
    with open(html_path, 'w') as f:
        f.write(html_output)
    print(f"  Saved animation: {html_path}")

    plt.close(fig)

    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
