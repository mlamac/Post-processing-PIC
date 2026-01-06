"""
Synthetic LWFA data generator for testing and examples.
Creates realistic-looking laser wakefield acceleration data.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import h5py


def generate_lwfa_data(
    output_path: str,
    nx: int = 400,
    ny: int = 200,
    x_range: Tuple[float, float] = (0, 40e-6),  # 40 microns
    y_range: Tuple[float, float] = (-10e-6, 10e-6),  # 20 microns
    n_particles: int = 10000,
    plasma_density: float = 1e24,  # m^-3
    laser_a0: float = 2.0,
    seed: Optional[int] = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic LWFA simulation data.

    Args:
        output_path: Path to save HDF5 file
        nx: Number of grid points in x (longitudinal)
        ny: Number of grid points in y (transverse)
        x_range: (x_min, x_max) in meters
        y_range: (y_min, y_max) in meters
        n_particles: Number of accelerated electron particles
        plasma_density: Background plasma density in m^-3
        laser_a0: Normalized laser amplitude
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing generated data
    """
    if seed is not None:
        np.random.seed(seed)

    # Create grids
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Physical constants
    c = 3e8  # speed of light
    m_e = 9.11e-31  # electron mass
    e = 1.6e-19  # elementary charge
    epsilon_0 = 8.85e-12  # permittivity

    # Plasma wavelength
    omega_p = np.sqrt(plasma_density * e**2 / (m_e * epsilon_0))
    lambda_p = 2 * np.pi * c / omega_p

    # Laser parameters
    laser_wavelength = 800e-9  # 800 nm
    k_laser = 2 * np.pi / laser_wavelength
    w0 = 5e-6  # laser waist
    x_focus = 15e-6  # focus position

    # Generate electron density with wakefield structure
    # Background plasma
    n_e = np.ones_like(X) * plasma_density

    # Add wakefield oscillations (blowout regime)
    wakefield_phase = 2 * np.pi * (X - x_focus) / lambda_p
    bubble_radius = 0.5 * lambda_p * np.sqrt(laser_a0)

    # Create bubble structure
    r_transverse = np.abs(Y)
    bubble_shape = np.exp(-r_transverse**2 / (2 * (0.8 * bubble_radius)**2))

    # Density depression in wake
    for i in range(3):  # Multiple wake periods
        phase_shift = wakefield_phase - i * 2 * np.pi
        depression = 0.8 * bubble_shape * np.exp(-((phase_shift) % (2*np.pi) - np.pi)**2 / 1.0)
        depression *= (X > x_focus) * (X < x_focus + 3 * lambda_p)
        n_e -= depression * plasma_density

    # Ensure density is non-negative
    n_e = np.maximum(n_e, 0.01 * plasma_density)

    # Generate transverse electric field (Ey)
    # Laser pulse component
    laser_envelope = laser_a0 * np.exp(-(X - x_focus)**2 / (2 * (3e-6)**2))
    laser_envelope *= np.exp(-Y**2 / (2 * w0**2))
    laser_oscillation = np.cos(k_laser * X)

    # Transverse wakefield component
    wakefield_ey = np.zeros_like(X)
    for i in range(3):
        phase_shift = wakefield_phase - i * 2 * np.pi
        wake_amplitude = 0.5 * laser_a0 * m_e * c * omega_p / e
        wake_amplitude *= np.exp(-r_transverse**2 / (2 * bubble_radius**2))
        wakefield_ey += wake_amplitude * Y / (bubble_radius + 1e-9) * \
                       np.sin(phase_shift) * (X > x_focus) * (X < x_focus + 3 * lambda_p)

    # Total transverse E-field (in V/m)
    E_y = laser_envelope * laser_oscillation * m_e * c * omega_p / e + wakefield_ey

    # Generate longitudinal electric field (Ex) - acceleration field
    E_x = np.zeros_like(X)
    for i in range(3):
        phase_shift = wakefield_phase - i * 2 * np.pi
        wake_ex = laser_a0**2 * m_e * c * omega_p / (2 * e)
        wake_ex *= np.exp(-r_transverse**2 / (2 * bubble_radius**2))
        E_x += wake_ex * np.cos(phase_shift) * (X > x_focus) * (X < x_focus + 3 * lambda_p)

    # Generate accelerated electron particles
    # Particles injected near the back of the first bubble
    x_inject = x_focus + 0.7 * lambda_p
    x_particles = x_inject + np.random.exponential(0.3 * lambda_p, n_particles)
    x_particles = x_particles[x_particles < x_range[1]]

    n_particles = len(x_particles)
    y_particles = np.random.normal(0, 0.3 * bubble_radius, n_particles)

    # Momentum distribution - accelerated beam
    # Longitudinal momentum (accelerated electrons)
    mean_energy_MeV = 100 + 50 * (x_particles - x_inject) / lambda_p  # Energy gain
    energy_spread = 0.1  # 10% energy spread
    energy_MeV = np.abs(np.random.normal(mean_energy_MeV, energy_spread * mean_energy_MeV))

    # Convert to momentum (in units of m_e * c)
    gamma = energy_MeV * 1e6 * e / (m_e * c**2) + 1
    p_total = np.sqrt(gamma**2 - 1)

    # Mostly longitudinal momentum
    px_particles = p_total * 0.98
    py_particles = np.random.normal(0, 0.05 * p_total, n_particles)
    pz_particles = np.random.normal(0, 0.05 * p_total, n_particles)

    # Particle weights (charge per macroparticle)
    weight = np.ones(n_particles) * e * 1e-15  # Each represents ~6000 electrons

    # Save to HDF5 file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Grid
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)

        # Fields
        fields_group = f.create_group('Fields')
        fields_group.create_dataset('Ex', data=E_x.T)  # Transpose for consistency
        fields_group.create_dataset('Ey', data=E_y.T)

        # Density
        rho_group = f.create_group('Rho')
        rho_group.create_dataset('Rho_electron', data=n_e.T)

        # Particles
        species_group = f.create_group('Species')
        electron_group = species_group.create_group('electron')
        electron_group.create_dataset('x', data=x_particles)
        electron_group.create_dataset('y', data=y_particles)
        electron_group.create_dataset('px', data=px_particles)
        electron_group.create_dataset('py', data=py_particles)
        electron_group.create_dataset('pz', data=pz_particles)
        electron_group.create_dataset('weight', data=weight)

        # Metadata
        f.attrs['time'] = 50e-15  # 50 fs
        f.attrs['iteration'] = 1000
        f.attrs['plasma_density'] = plasma_density
        f.attrs['laser_a0'] = laser_a0
        f.attrs['lambda_p'] = lambda_p

    print(f"Generated synthetic LWFA data at {output_path}")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Particles: {n_particles}")
    print(f"  Plasma wavelength: {lambda_p*1e6:.2f} μm")

    return {
        'x': x,
        'y': y,
        'n_e': n_e,
        'E_x': E_x,
        'E_y': E_y,
        'particles': {
            'x': x_particles,
            'y': y_particles,
            'px': px_particles,
            'py': py_particles,
            'pz': pz_particles,
            'weight': weight
        }
    }
