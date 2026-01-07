"""
Physical constants and parameter calculations for PIC simulations.
Handles normalization, unit conversions, and derived quantities.
"""

import numpy as np
from typing import Dict, Any


# =============================================================================
# PHYSICAL CONSTANTS (SI units)
# =============================================================================

C_LIGHT = 2.99792458e8          # Speed of light [m/s]
M_E = 9.1093837e-31             # Electron mass [kg]
Q_E = 1.60217663e-19            # Elementary charge [C]
EPS_0 = 8.8541878188e-12        # Vacuum permittivity [F/m]
M_P = 1.67262192e-27            # Proton mass [kg]


# =============================================================================
# PARAMETER CALCULATIONS
# =============================================================================

def compute_laser_parameters(lambda0_um: float) -> Dict[str, float]:
    """
    Compute laser-related parameters.

    Args:
        lambda0_um: Laser wavelength in microns

    Returns:
        Dict with lambda0 [m], omega_0 [rad/s], k_0 [1/m]
    """
    lambda0 = lambda0_um * 1e-6
    omega_0 = 2 * np.pi * C_LIGHT / lambda0
    k_0 = 2 * np.pi / lambda0

    return {
        'lambda0': lambda0,
        'omega_0': omega_0,
        'k_0': k_0,
    }


def compute_plasma_parameters(n_0: float) -> Dict[str, float]:
    """
    Compute plasma-related parameters.

    Args:
        n_0: Plasma electron density [m^-3]

    Returns:
        Dict with omega_pe [rad/s], lambda_pe [m], k_pe [1/m], v_ph [m/s]
    """
    omega_pe = np.sqrt(Q_E**2 * n_0 / (EPS_0 * M_E))
    lambda_pe = 2 * np.pi * C_LIGHT / omega_pe
    k_pe = 2 * np.pi / lambda_pe
    v_ph = omega_pe / k_pe  # Phase velocity (equals c for plasma waves)

    return {
        'omega_pe': omega_pe,
        'lambda_pe': lambda_pe,
        'k_pe': k_pe,
        'v_ph': v_ph,
    }


def compute_critical_density(lambda0_um: float) -> float:
    """
    Compute critical plasma density for given laser wavelength.

    Args:
        lambda0_um: Laser wavelength in microns

    Returns:
        Critical density n_c in m^-3
    """
    laser_params = compute_laser_parameters(lambda0_um)
    omega_0 = laser_params['omega_0']
    n_c = EPS_0 * M_E * omega_0**2 / Q_E**2
    return n_c


def compute_derived_quantities(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute all derived plasma and laser parameters from configuration.
    This function mirrors the working script's compute_derived_quantities.

    Args:
        config: Configuration dict with keys:
            - lambda0_um: Laser wavelength in microns
            - n_over_nc: Plasma density as fraction of critical density
            - dumpstep_fs: Time between dumps in femtoseconds (optional)
            - a0: Normalized vector potential (optional, for reference)

    Returns:
        Dict with derived parameters:
            - lambda0, omega_0, k_0: Laser parameters
            - n_c, n_0: Critical and plasma densities
            - omega_pe, lambda_pe, k_pe: Plasma parameters
            - E_norm_laser: Normalization for transverse E-fields [V/m]
            - E_norm_plasma: Normalization for longitudinal E-fields [V/m]
            - dumpstep: Time step in normalized units (if dumpstep_fs provided)
    """
    # Laser parameters
    laser_params = compute_laser_parameters(config['lambda0_um'])

    # Critical density
    n_c = compute_critical_density(config['lambda0_um'])

    # Plasma density
    n_0 = config['n_over_nc'] * n_c

    # Plasma parameters
    plasma_params = compute_plasma_parameters(n_0)

    # Field normalization factors
    # For transverse fields (laser): E_norm = m_e * c * omega_0 / e
    E_norm_laser = M_E * C_LIGHT * laser_params['omega_0'] / Q_E

    # For longitudinal fields (wakefield): E_norm = m_e * c * omega_pe / e
    E_norm_plasma = M_E * C_LIGHT * plasma_params['omega_pe'] / Q_E

    result = {
        **laser_params,
        'n_c': n_c,
        'n_0': n_0,
        **plasma_params,
        'E_norm_laser': E_norm_laser,
        'E_norm_plasma': E_norm_plasma,
    }

    # Optional: time step normalization
    if 'dumpstep_fs' in config:
        dumpstep = config['dumpstep_fs'] * 1e-15 * plasma_params['omega_pe']
        result['dumpstep'] = dumpstep

    return result


# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================

def normalize_length(length_m: float, lambda0_m: float) -> float:
    """Normalize length to laser wavelength units."""
    return length_m / lambda0_m


def normalize_density(n_e: float, n_c: float) -> float:
    """Normalize electron density to critical density."""
    return n_e / n_c


def normalize_field_laser(E_field: np.ndarray, params: Dict) -> np.ndarray:
    """Normalize transverse E-field to laser field units."""
    return E_field / params['E_norm_laser']


def normalize_field_plasma(E_field: np.ndarray, params: Dict) -> np.ndarray:
    """Normalize longitudinal E-field to plasma field units."""
    return E_field / params['E_norm_plasma']


def normalize_momentum(p: np.ndarray, mass: float = M_E) -> np.ndarray:
    """
    Normalize momentum to m*c units.

    Args:
        p: Momentum in SI units [kg*m/s]
        mass: Particle mass in kg (default: electron)

    Returns:
        Momentum in units of m*c
    """
    return p / (mass * C_LIGHT)


def normalize_time(t_seconds: float, omega_pe: float) -> float:
    """Normalize time to plasma frequency units."""
    return t_seconds * omega_pe


# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

def compute_gamma(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """
    Compute Lorentz factor from normalized momentum (p in units of m*c).

    Args:
        px, py, pz: Momentum components in units of m*c

    Returns:
        Lorentz factor gamma
    """
    p_squared = px**2 + py**2 + pz**2
    gamma = np.sqrt(1 + p_squared)
    return gamma


def compute_kinetic_energy_MeV(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                                mass: float = M_E) -> np.ndarray:
    """
    Compute kinetic energy in MeV from normalized momentum.

    Args:
        px, py, pz: Momentum components in units of m*c
        mass: Particle mass in kg (default: electron)

    Returns:
        Kinetic energy in MeV
    """
    gamma = compute_gamma(px, py, pz)
    rest_energy_MeV = mass * C_LIGHT**2 / (1e6 * Q_E)  # Rest energy in MeV
    kinetic_energy = (gamma - 1) * rest_energy_MeV
    return kinetic_energy


def compute_debye_length(n_0: float, T_eV: float) -> float:
    """
    Compute Debye length.

    Args:
        n_0: Plasma density in m^-3
        T_eV: Electron temperature in eV

    Returns:
        Debye length in meters
    """
    T_joules = T_eV * Q_E
    lambda_D = np.sqrt(EPS_0 * T_joules / (n_0 * Q_E**2))
    return lambda_D


# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

def eV_to_joules(energy_eV: float) -> float:
    """Convert energy from eV to Joules."""
    return energy_eV * Q_E


def joules_to_eV(energy_J: float) -> float:
    """Convert energy from Joules to eV."""
    return energy_J / Q_E


def microns_to_meters(length_um: float) -> float:
    """Convert length from microns to meters."""
    return length_um * 1e-6


def meters_to_microns(length_m: float) -> float:
    """Convert length from meters to microns."""
    return length_m * 1e6


def femtoseconds_to_seconds(time_fs: float) -> float:
    """Convert time from femtoseconds to seconds."""
    return time_fs * 1e-15


def seconds_to_femtoseconds(time_s: float) -> float:
    """Convert time from seconds to femtoseconds."""
    return time_s * 1e15
