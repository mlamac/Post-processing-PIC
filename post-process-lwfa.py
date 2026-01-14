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

    # Particle species name in SDF files
    'species': 'He_electron',

    # Processing options
    'downsample_x': 10,             # Spatial downsampling factor (x)
    'downsample_r': 1,              # Spatial downsampling factor (r)
    'frame_interval': 1,            # Process every Nth frame (1 = all frames)

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

    # Phase space scatter density coloring
    'ps_density_bins': (100, 100),  # Bins for density computation (x, px)

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
import scipy.ndimage
import time
import contextlib
import io

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
        # Load SDF files (silence verbose sdf_helper output)
        with contextlib.redirect_stdout(io.StringIO()):
            data_E = sh.getdata(config['efield_pattern'].format(index))
            data_dens = sh.getdata('dens{:04d}.sdf'.format(index))
            data_ener = sh.getdata(config['ener_pattern'].format(index))

        # Validate required fields exist
        validate_sdf_fields(data_E, [
            'Electric_Field_Modes_Erm_real', 'Electric_Field_Modes_Erm_imag',
            'Electric_Field_Modes_Exm_real', 'Electric_Field_Modes_Exm_imag',
            'Grid_Grid_mid'
        ], f'E_field{index:04d}.sdf')

        validate_sdf_fields(data_dens, ['Derived_Number_Density_Subset_total_e'],
                            f'dens{index:04d}.sdf')

        species = config['species']
        validate_sdf_fields(data_ener, [
            f'Grid_Particles_{species}',
            f'Particles_Px_{species}',
            f'Particles_Weight_{species}'
        ], f'ener{index:04d}.sdf')

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

        # Particle data (configurable species)
        x_he = np.squeeze(getattr(data_ener, f'Grid_Particles_{species}').data[0]) / params['lambda0']
        px_he = np.squeeze(getattr(data_ener, f'Particles_Px_{species}').data) / (M_E * C_LIGHT)
        w_he = np.squeeze(getattr(data_ener, f'Particles_Weight_{species}').data)

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


def compute_particle_density(x, px, weights, n_bins=(100, 100)):
    """
    Compute local phase space density at each particle position.

    Uses 2D weighted histogram with fast bin lookup.

    Args:
        x: Particle x positions
        px: Particle momenta
        weights: Particle weights
        n_bins: Tuple of (n_x_bins, n_px_bins) for histogram

    Returns:
        Array of density values at each particle position
    """
    if len(x) == 0:
        return np.array([])

    # Create 2D weighted histogram
    hist, x_edges, px_edges = np.histogram2d(
        x, px, bins=n_bins, weights=weights
    )

    # Find bin indices for each particle (clip to valid range)
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins[0] - 1)
    px_idx = np.clip(np.digitize(px, px_edges) - 1, 0, n_bins[1] - 1)

    # Look up density from histogram
    density = hist[x_idx, px_idx]

    return density


def progress_bar(current, total, prefix='Progress', suffix=''):
    """Simple text progress bar for HPC jobs with optional suffix."""
    pct = 100 * current / total
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = '=' * filled + '-' * (bar_len - filled)
    msg = f'\r{prefix}: [{bar}] {pct:.1f}% ({current}/{total})'
    if suffix:
        msg += f' {suffix}'
    print(msg, end='', flush=True)
    if current == total:
        print()  # Newline at end


def validate_config(config):
    """
    Validate that all required CONFIG keys exist.

    Returns:
        Tuple of (is_valid, error_message)
    """
    required = [
        'lambda0_um', 'a0', 'n_over_nc',
        'dens_pattern', 'efield_pattern', 'ener_pattern',
        'species', 'downsample_x', 'downsample_r',
        'n_px_bins', 'px_min', 'px_max',
        'output_dir', 'output_prefix'
    ]
    missing = [k for k in required if k not in config]
    if missing:
        return False, f"Missing required config keys: {', '.join(missing)}"
    return True, ""


def validate_sdf_fields(data, required_fields, file_desc):
    """
    Check that required SDF field names exist in loaded data.

    Args:
        data: Loaded SDF data object
        required_fields: List of field names that must exist
        file_desc: Description of file for error messages

    Raises:
        ValueError: If any required field is missing
    """
    missing = [f for f in required_fields if not hasattr(data, f)]
    if missing:
        raise ValueError(f"{file_desc} missing fields: {', '.join(missing)}")


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
            'w_he': frame['w_he'],
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
    Create 2-panel animation figure.

    Panel 1: 2D density and laser field (pcolormesh) with stacked colorbars
    Panel 2: Longitudinal phase space (density-colored scatter) with momentum lineout
    """
    n_frames = len(data['x'])
    dumpstep = params['dumpstep']
    a0 = config['a0']
    omega_ratio = 1.0 / np.sqrt(config['n_over_nc'])  # omega_0 / omega_pe

    # Pre-compute particle densities for coloring
    print("  Computing phase space densities...")
    particle_densities = []
    n_frames = len(data['particles'])
    for i, pdata in enumerate(data['particles']):
        progress_bar(i + 1, n_frames, 'Densities')
        if pdata['x_he'].size and pdata['px_he'].size:
            dens = compute_particle_density(
                pdata['x_he'], pdata['px_he'], pdata['w_he'],
                n_bins=config.get('ps_density_bins', (100, 100))
            )
            particle_densities.append(dens)
        else:
            particle_densities.append(np.array([]))

    # Find global max for consistent color scaling
    max_dens = max((d.max() for d in particle_densities if d.size > 0), default=1.0)

    # Max histogram value for lineout scaling
    max_hist = max(np.max(h) for h in data['px_dist'] if len(h) > 0)

    # Uniform font size (larger for readability)
    fontsize = MEDIUM_SIZE

    # Create figure with GridSpec - 4 rows for stacked colorbars
    # Rows 0-1: Panel 1 with density colorbar (row 0) and field colorbar (row 1)
    # Rows 2-3: Panel 2 with phase space colorbar
    fig = plt.figure(figsize=(6.0, 7.0), dpi=200)
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 0.04],
                          height_ratios=[1, 1, 1, 1],
                          wspace=0.03, hspace=0.1)

    # Panel 1 spans rows 0-1
    ax1 = fig.add_subplot(gs[0:2, 0])
    cax_dens = fig.add_subplot(gs[0, 1])    # Density colorbar (top half)
    cax_field = fig.add_subplot(gs[1, 1])   # Field colorbar (bottom half)

    # Panel 2 spans rows 2-3
    ax2 = fig.add_subplot(gs[2:4, 0])
    cax_ps = fig.add_subplot(gs[2:4, 1])    # Phase space colorbar

    # Initial data
    x0, r0 = data['x'][0], data['r'][0]

    # Panel 1: 2D density and laser field
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
    ax1.set_ylabel(r"$r\:/\:\lambda_{0}$", fontsize=fontsize)
    ax1.set_xlabel(r"$x\:/\:\lambda_{0}$", fontsize=fontsize)
    ax1.tick_params(axis='both', labelsize=fontsize)
    ax1.set_ylim(r0[0], config['r_max_lambda0'])
    ax1.set_title(
        f"$a_0 = {a0:.1f},\\quad \\omega_0\\:/\\:\\omega_{{pe}} = {omega_ratio:.0f},\\quad \\omega_{{pe}} t = {0:.0f}$",
        loc='left', fontsize=fontsize
    )

    # Colorbars for Panel 1 (stacked vertically)
    cbar_dens = fig.colorbar(plot_dens, cax=cax_dens)
    cbar_dens.set_label(r'$n_e\:/\:n_c$', fontsize=fontsize)
    cbar_dens.ax.tick_params(labelsize=fontsize)

    cbar_field = fig.colorbar(plot_field, cax=cax_field)
    cbar_field.set_label(r'$|E_\perp|\:/\:E_0$', fontsize=fontsize)
    cbar_field.ax.tick_params(labelsize=fontsize)

    # Panel 2: Phase space scatter with density coloring
    pdata = data['particles'][0]
    scatter_he = ax2.scatter(
        pdata['x_he'], pdata['px_he'],
        s=2, c=particle_densities[0], cmap='jet',
        norm=colors.LogNorm(vmin=max_dens*1e-4, vmax=max_dens),
        marker='.', linewidths=0
    )
    ax2.set_xlabel(r"$x\:/\:\lambda_{0}$", fontsize=fontsize)
    ax2.set_ylabel(r"$p_x\:/\:m_{e}c$", fontsize=fontsize)
    ax2.tick_params(axis='both', labelsize=fontsize)
    ax2.set_ylim(config['px_ylim'])
    ax2.set_xlim(x0[0], x0[-1])

    # Colorbar for phase space density
    cbar_ps = fig.colorbar(scatter_he, cax=cax_ps)
    cbar_ps.set_label(r'$f(x,\:p_x)$', fontsize=fontsize)
    cbar_ps.ax.tick_params(labelsize=fontsize)

    # Create twin axis for momentum distribution lineout
    ax2_twin = ax2.twiny()  # Share y-axis

    # Plot momentum distribution as vertical profile (red, no labels/ticks)
    px_centers = data['px_centers']
    line_dist, = ax2_twin.plot(data['px_dist'][0], px_centers, color='red', lw=1.5, alpha=0.8)
    ax2_twin.set_xlim(0, max_hist * 0.5)

    # Hide the twin axis labels and ticks
    ax2_twin.set_xticklabels([])
    ax2_twin.set_xticks([])
    ax2_twin.spines['top'].set_visible(False)

    # Adjust layout with right margin for colorbar labels
    fig.subplots_adjust(left=0.12, right=0.82, top=0.95, bottom=0.08, hspace=0.25)

    # Store plot objects that need to be recreated each frame
    plot_objects = {'dens': plot_dens, 'field': plot_field}

    def animate(frame):
        x = data['x'][frame]
        r = data['r'][frame]

        # Update Panel 1: density and field
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
            f"$a_0 = {a0:.1f},\\quad \\omega_0\\:/\\:\\omega_{{pe}} = {omega_ratio:.0f},\\quad \\omega_{{pe}} t = {frame * dumpstep:.0f}$",
            loc='left', fontsize=fontsize
        )

        # Update Panel 2: phase space with density colors
        pdata = data['particles'][frame]
        if pdata['x_he'].size and pdata['px_he'].size:
            he_data = np.column_stack((pdata['x_he'], pdata['px_he']))
            scatter_he.set_offsets(he_data)
            scatter_he.set_array(particle_densities[frame])
        else:
            scatter_he.set_offsets(np.empty((0, 2)))
            scatter_he.set_array(np.array([]))
        ax2.set_xlim(x[0], x[-1])

        # Update momentum lineout
        line_dist.set_xdata(data['px_dist'][frame])

        return plot_objects['dens'], plot_objects['field'], scatter_he, line_dist

    anim = FuncAnimation(
        fig, animate, frames=n_frames,
        interval=config['animation_interval'], blit=False, repeat=True
    )

    return fig, anim


def main():
    """Main entry point."""
    start_time = time.time()

    print("=" * 60)
    print("EPOCH Quasi-3D LWFA Post-Processing")
    print("=" * 60)

    # Validate configuration
    valid, msg = validate_config(CONFIG)
    if not valid:
        print(f"ERROR: {msg}")
        sys.exit(1)

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

    # Apply frame interval
    interval = CONFIG.get('frame_interval', 1)
    if interval > 1:
        indices = indices[::interval]
        print(f"Using every {interval}th frame: {len(indices)} frames selected")
    print()

    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Process all frames with timing
    load_start = time.time()
    data = process_all_frames(indices, params, CONFIG)
    load_time = time.time() - load_start

    if len(data['x']) == 0:
        print("ERROR: No frames were successfully processed.")
        sys.exit(1)

    # Show frame statistics
    success_count = len(data['x'])
    total_count = len(indices)
    print(f"  Frame loading completed in {load_time:.1f}s")
    if success_count < total_count:
        print(f"  WARNING: {total_count - success_count} of {total_count} frames failed to load")

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

    # Create animation with timing
    print("\nCreating animation...")
    anim_start = time.time()
    fig, anim = create_animation(data, params, CONFIG)
    anim_time = time.time() - anim_start
    print(f"  Animation created in {anim_time:.1f}s")

    # Render HTML with timing
    print("  Rendering HTML...")
    render_start = time.time()
    html_output = anim.to_jshtml()
    render_time = time.time() - render_start
    html_path = f"{prefix}_animation.html"
    with open(html_path, 'w') as f:
        f.write(html_output)
    print(f"  HTML rendered in {render_time:.1f}s")
    print(f"  Saved animation: {html_path}")

    plt.close(fig)

    # Total processing time
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.1f}s")

    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
