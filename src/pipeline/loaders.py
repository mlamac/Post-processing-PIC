"""
EPOCH SDF Data Loaders

Functions for loading and processing EPOCH quasi-3D simulation data.
Based on the working post-process-lwfa.py script.
"""

import os
import glob
import re
import numpy as np

# Import sdf_helper with error handling
try:
    import sdf_helper as sh
    SDF_AVAILABLE = True
except ImportError:
    SDF_AVAILABLE = False

# Physical constants (SI units)
C_LIGHT = 2.99792458e8              # Speed of light [m/s]
M_E = 9.1093837e-31                 # Electron mass [kg]


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


def find_sdf_files(pattern, directory=None):
    """
    Find SDF files matching pattern and return sorted list of indices.

    Args:
        pattern: Glob pattern like 'dens*.sdf'
        directory: Optional directory to search in (default: current directory)

    Returns:
        Sorted list of integer indices (e.g., [60, 120, 180])
    """
    if directory:
        search_pattern = os.path.join(directory, pattern)
    else:
        search_pattern = pattern

    files = glob.glob(search_pattern)
    indices = []

    # Extract frame number from filename (e.g., dens0060.sdf -> 60)
    for f in files:
        match = re.search(r'(\d{4})\.sdf$', os.path.basename(f))
        if match:
            indices.append(int(match.group(1)))

    return sorted(indices)


def load_frame_data(index, params, config, directory=None):
    """
    Load all data for a single simulation frame.

    Loads three SDF files (E_field, dens, ener) and returns normalized,
    transposed, and downsampled data ready for visualization.

    Args:
        index: Frame index (e.g., 60 for E_field0060.sdf)
        params: Dict from compute_derived_quantities() containing:
            - lambda0: Laser wavelength [m]
            - n_c: Critical density [m^-3]
            - E_norm_laser: Transverse field normalization
            - E_norm_plasma: Longitudinal field normalization
        config: CONFIG dict containing:
            - efield_pattern: Format string for E-field files
            - ener_pattern: Format string for energy files
            - downsample_x, downsample_r: Downsampling factors
            - species: Particle species name (default: 'He_electron')
        directory: Optional directory containing SDF files

    Returns:
        Dict with keys: E_x, E_y, E_tot, n_e, x, r, x_he, px_he, w_he
        All fields are normalized, transposed (.T), and downsampled.
        Returns None if loading fails.
    """
    if not SDF_AVAILABLE:
        raise ImportError(
            "sdf_helper module not found. "
            "Please compile with: cd /path/to/epoch/epoch2d && make sdfutils"
        )

    # Build file paths
    def get_path(pattern, idx):
        filename = pattern.format(idx)
        if directory:
            return os.path.join(directory, filename)
        return filename

    try:
        # Load SDF files
        data_E = sh.getdata(get_path(config['efield_pattern'], index))
        data_dens = sh.getdata(get_path('dens{:04d}.sdf', index))
        data_ener = sh.getdata(get_path(config['ener_pattern'], index))

        # Reconstruct transverse fields from modes and normalize FIRST
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

        # Longitudinal field E_x (different normalization!)
        E_x = reconstruct_field_from_modes(
            data_E.Electric_Field_Modes_Exm_real.data,
            data_E.Electric_Field_Modes_Exm_imag.data,
            theta=0
        ) / params['E_norm_plasma']

        # Total transverse field magnitude (computed AFTER normalization)
        E_tot = np.sqrt(E_y**2 + E_z**2)

        # Electron density normalized to critical density
        n_e = data_dens.Derived_Number_Density_Subset_total_e.data / params['n_c']

        # Grid coordinates normalized to laser wavelength
        x = data_E.Grid_Grid_mid.data[0] / params['lambda0']
        r = data_E.Grid_Grid_mid.data[1] / params['lambda0']

        # Apply downsampling
        ds_x = config.get('downsample_x', 1)
        ds_r = config.get('downsample_r', 1)

        # Particle data
        species = config.get('species', 'He_electron')
        grid_attr = f'Grid_Particles_{species}'
        px_attr = f'Particles_Px_{species}'
        weight_attr = f'Particles_Weight_{species}'

        x_particles = np.squeeze(getattr(data_ener, grid_attr).data[0]) / params['lambda0']
        px_particles = np.squeeze(getattr(data_ener, px_attr).data) / (M_E * C_LIGHT)
        w_particles = np.squeeze(getattr(data_ener, weight_attr).data)

        return {
            'E_x': E_x[::ds_x, ::ds_r].T,
            'E_y': E_y[::ds_x, ::ds_r].T,
            'E_tot': E_tot[::ds_x, ::ds_r].T,
            'n_e': n_e[::ds_x, ::ds_r].T,
            'x': x[::ds_x],
            'r': r[::ds_r],
            'x_he': x_particles.flatten() if x_particles.size else np.array([]),
            'px_he': px_particles.flatten() if px_particles.size else np.array([]),
            'w_he': w_particles.flatten() if w_particles.size else np.array([]),
        }

    except Exception as e:
        print(f"WARNING: Failed to load frame {index}: {e}")
        return None
