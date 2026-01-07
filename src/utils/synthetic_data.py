"""
Synthetic EPOCH quasi-3D LWFA data generator.
Creates realistic-looking laser wakefield acceleration data in EPOCH format
with modal field decomposition for testing the pipeline.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import h5py
from .physics import (
    C_LIGHT, M_E, Q_E, EPS_0,
    compute_derived_quantities,
    normalize_momentum
)


def reconstruct_field_from_modes(mode_real: np.ndarray, mode_imag: np.ndarray,
                                  theta: float = 0) -> np.ndarray:
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
    field = (mode_real[:, :, 0] +
             mode_real[:, :, 1] * np.cos(theta) +
             mode_imag[:, :, 1] * np.sin(theta))
    return field


def generate_modal_field(field_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate modal representation from a 2D field.
    For synthetic data, we approximate the modal decomposition.

    Args:
        field_2d: 2D field array (nx, nr)

    Returns:
        (mode_real, mode_imag) with shape (nx, nr, 2) for modes 0 and 1
    """
    nx, nr = field_2d.shape

    # Mode 0: axisymmetric part (average)
    # Mode 1: asymmetric part (the field itself for theta=0)
    mode_real = np.zeros((nx, nr, 2))
    mode_imag = np.zeros((nx, nr, 2))

    # For simplicity: put most of the field in mode 1 real part
    # This reconstructs to the original field at theta=0
    mode_real[:, :, 0] = 0  # No axisymmetric component
    mode_real[:, :, 1] = field_2d  # Asymmetric component

    return mode_real, mode_imag


def generate_epoch_lwfa_data(
    output_path: str,
    nx: int = 400,
    nr: int = 200,
    x_range: Tuple[float, float] = (0, 200),  # in lambda0 units
    r_max: float = 100,  # in lambda0 units
    n_particles: int = 10000,
    config: Optional[Dict] = None,
    seed: Optional[int] = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic EPOCH quasi-3D LWFA simulation data.

    Args:
        output_path: Path to save HDF5 file (mimics EPOCH structure)
        nx: Number of grid points in x (longitudinal)
        nr: Number of grid points in r (radial)
        x_range: (x_min, x_max) in laser wavelength units
        r_max: Maximum r in laser wavelength units
        n_particles: Number of accelerated electron particles
        config: Optional configuration dict with laser/plasma parameters
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing generated data
    """
    if seed is not None:
        np.random.seed(seed)

    # Default configuration
    if config is None:
        config = {
            'lambda0_um': 1.0,
            'a0': 3.0,
            'n_over_nc': 0.0025,
            'dumpstep_fs': 160,
        }

    # Compute physical parameters
    params = compute_derived_quantities(config)

    # Create grids (normalized to lambda0)
    x = np.linspace(x_range[0], x_range[1], nx)
    r = np.linspace(0, r_max, nr)
    X, R = np.meshgrid(x, r, indexing='ij')

    # Physical quantities
    lambda0 = params['lambda0']
    lambda_pe = params['lambda_pe']
    n_c = params['n_c']
    n_0 = params['n_0']
    a0 = config['a0']

    # Convert normalized coordinates to physical units
    X_m = X * lambda0
    R_m = R * lambda0

    # Plasma wavelength in lambda0 units
    lambda_pe_norm = lambda_pe / lambda0

    # =========================================================================
    # Generate electron density (normalized to n_c)
    # =========================================================================

    # Background plasma density
    n_e = np.ones_like(X) * config['n_over_nc']

    # Wakefield structure (bubble regime)
    # Focus position
    x_focus = x_range[0] + 0.3 * (x_range[1] - x_range[0])

    # Bubble radius (depends on a0)
    r_bubble = 0.5 * lambda_pe_norm * np.sqrt(a0)

    # Create wakefield oscillations
    wakefield_phase = 2 * np.pi * (X - x_focus) / lambda_pe_norm

    # Bubble shape
    bubble_shape = np.exp(-R**2 / (2 * (0.8 * r_bubble)**2))

    # Add multiple wake periods with density depression
    for i in range(3):
        phase_shift = wakefield_phase - i * 2 * np.pi
        # Density depression in bubble
        depression = 0.7 * bubble_shape * np.exp(-((phase_shift % (2*np.pi)) - np.pi)**2 / 1.0)
        depression *= (X > x_focus) * (X < x_focus + 3 * lambda_pe_norm)
        n_e -= depression * config['n_over_nc']

    # Ensure density is non-negative
    n_e = np.maximum(n_e, 0.01 * config['n_over_nc'])

    # Convert to physical units for output
    n_e_physical = n_e * n_c

    # =========================================================================
    # Generate E-fields (normalized)
    # =========================================================================

    # Laser pulse (transverse field)
    laser_center = x_range[0] + 0.25 * (x_range[1] - x_range[0])
    laser_sigma_x = 3.0  # in lambda0 units
    laser_w0 = 5.0  # waist in lambda0 units
    k_laser = 2 * np.pi  # in 1/lambda0 units

    laser_envelope = a0 * np.exp(-(X - laser_center)**2 / (2 * laser_sigma_x**2))
    laser_envelope *= np.exp(-R**2 / (2 * laser_w0**2))
    laser_oscillation = np.cos(k_laser * X)

    # Transverse field (E_r mode, will be reconstructed to E_y at theta=0)
    E_r_laser = laser_envelope * laser_oscillation

    # Wakefield transverse field
    E_r_wake = np.zeros_like(X)
    for i in range(3):
        phase_shift = wakefield_phase - i * 2 * np.pi
        wake_amp = 0.3 * a0
        wake_amp *= np.exp(-R**2 / (2 * r_bubble**2))
        E_r_wake += wake_amp * R / (r_bubble + 1e-9) * np.sin(phase_shift)
        E_r_wake *= (X > x_focus) * (X < x_focus + 3 * lambda_pe_norm)

    E_r_total = E_r_laser + E_r_wake

    # Longitudinal field (E_x)
    E_x = np.zeros_like(X)
    for i in range(3):
        phase_shift = wakefield_phase - i * 2 * np.pi
        wake_ex = a0**2 * 0.5
        wake_ex *= np.exp(-R**2 / (2 * r_bubble**2))
        E_x += wake_ex * np.cos(phase_shift)
        E_x *= (X > x_focus) * (X < x_focus + 3 * lambda_pe_norm)

    # =========================================================================
    # Generate modal field representations
    # =========================================================================

    # E_r modes (transverse)
    Erm_real, Erm_imag = generate_modal_field(E_r_total)

    # E_x modes (longitudinal)
    Exm_real, Exm_imag = generate_modal_field(E_x)

    # Convert to physical units
    Erm_real_physical = Erm_real * params['E_norm_laser']
    Erm_imag_physical = Erm_imag * params['E_norm_laser']
    Exm_real_physical = Exm_real * params['E_norm_plasma']
    Exm_imag_physical = Exm_imag * params['E_norm_plasma']

    # =========================================================================
    # Generate particles (He_electron species)
    # =========================================================================

    # Injection region (back of first bubble)
    x_inject = x_focus + 0.7 * lambda_pe_norm
    x_particles_norm = x_inject + np.random.exponential(0.3 * lambda_pe_norm, n_particles)
    x_particles_norm = x_particles_norm[x_particles_norm < x_range[1]]

    n_particles = len(x_particles_norm)
    x_particles = x_particles_norm * lambda0  # Convert to physical units

    # Radial position
    r_particles_norm = np.abs(np.random.normal(0, 0.3 * r_bubble, n_particles))
    r_particles = r_particles_norm * lambda0

    # Momentum distribution
    # Accelerated electrons with energy gain
    mean_energy_MeV = 100 + 50 * (x_particles_norm - x_inject) / lambda_pe_norm
    energy_spread = 0.15
    energy_MeV = np.abs(np.random.normal(mean_energy_MeV, energy_spread * mean_energy_MeV))

    # Convert to momentum (normalized to m_e * c)
    gamma = energy_MeV * 1e6 * Q_E / (M_E * C_LIGHT**2) + 1
    p_total = np.sqrt(gamma**2 - 1)

    # Mostly longitudinal
    px_particles_norm = p_total * 0.98
    pr_particles_norm = np.random.normal(0, 0.05 * p_total, n_particles)

    # Convert to physical units (SI)
    px_particles_physical = px_particles_norm * M_E * C_LIGHT
    pr_particles_physical = pr_particles_norm * M_E * C_LIGHT

    # Weight
    weight = np.ones(n_particles) * Q_E * 1e-15

    # =========================================================================
    # Save to HDF5 file (mimicking EPOCH structure)
    # =========================================================================

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Grid (in physical units) - store as separate datasets
        grid_group = f.create_group('Grid')
        grid_mid = grid_group.create_group('Grid_mid')
        grid_mid_x = grid_mid.create_dataset('x', data=x * lambda0)
        grid_mid_r = grid_mid.create_dataset('r', data=r * lambda0)
        grid_mid_x.attrs['description'] = 'Grid x-coordinates in meters'
        grid_mid_r.attrs['description'] = 'Grid r-coordinates in meters'

        # Particle grid
        grid_particles = grid_group.create_group('Particles_He_electron')
        grid_particles.create_dataset('x', data=x_particles)
        grid_particles.create_dataset('r', data=r_particles)

        # Electric field modes
        efield_group = f.create_group('Electric_Field_Modes')

        # E_r modes
        erm_real_ds = efield_group.create_dataset('Erm_real', data=Erm_real_physical)
        erm_real_ds.attrs['description'] = 'Transverse E-field mode (real part) [V/m]'

        erm_imag_ds = efield_group.create_dataset('Erm_imag', data=Erm_imag_physical)
        erm_imag_ds.attrs['description'] = 'Transverse E-field mode (imaginary part) [V/m]'

        # E_x modes
        exm_real_ds = efield_group.create_dataset('Exm_real', data=Exm_real_physical)
        exm_real_ds.attrs['description'] = 'Longitudinal E-field mode (real part) [V/m]'

        exm_imag_ds = efield_group.create_dataset('Exm_imag', data=Exm_imag_physical)
        exm_imag_ds.attrs['description'] = 'Longitudinal E-field mode (imaginary part) [V/m]'

        # Electron density
        density_group = f.create_group('Derived_Number_Density')
        dens_ds = density_group.create_dataset('Subset_total_e', data=n_e_physical)
        dens_ds.attrs['description'] = 'Electron number density [m^-3]'

        # Particles (He_electron)
        particles_group = f.create_group('Particles_He_electron')

        px_ds = particles_group.create_dataset('Px', data=px_particles_physical)
        px_ds.attrs['description'] = 'Particle momentum x [kg*m/s]'

        pr_ds = particles_group.create_dataset('Pr', data=pr_particles_physical)
        pr_ds.attrs['description'] = 'Particle momentum r [kg*m/s]'

        weight_ds = particles_group.create_dataset('Weight', data=weight)
        weight_ds.attrs['description'] = 'Particle weight [C]'

        # Metadata
        f.attrs['lambda0_um'] = config['lambda0_um']
        f.attrs['a0'] = config['a0']
        f.attrs['n_over_nc'] = config['n_over_nc']
        f.attrs['plasma_density'] = n_0
        f.attrs['lambda_pe'] = lambda_pe
        f.attrs['n_c'] = n_c

    print(f"Generated synthetic EPOCH quasi-3D LWFA data: {output_path}")
    print(f"  Grid: {nx} x {nr}")
    print(f"  Particles: {n_particles}")
    print(f"  Plasma wavelength: {lambda_pe*1e6:.2f} μm ({lambda_pe_norm:.2f} λ0)")
    print(f"  Bubble radius: {r_bubble:.2f} λ0")

    return {
        'x': x,
        'r': r,
        'n_e': n_e,
        'E_x': E_x,
        'E_r': E_r_total,
        'particles': {
            'x': x_particles_norm,
            'r': r_particles_norm,
            'px': px_particles_norm,
            'pr': pr_particles_norm,
            'weight': weight
        },
        'params': params
    }
