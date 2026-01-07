"""Utility functions and helpers"""

from .synthetic_data import generate_epoch_lwfa_data
from .physics import (
    compute_derived_quantities,
    compute_laser_parameters,
    compute_plasma_parameters,
    compute_critical_density,
    normalize_momentum,
    compute_kinetic_energy_MeV,
    C_LIGHT, M_E, Q_E, EPS_0
)
from .hpc import (
    progress_bar,
    find_sdf_files,
    print_section,
    print_parameter_summary,
    create_output_directory
)

__all__ = [
    'generate_epoch_lwfa_data',
    'compute_derived_quantities',
    'compute_laser_parameters',
    'compute_plasma_parameters',
    'compute_critical_density',
    'normalize_momentum',
    'compute_kinetic_energy_MeV',
    'progress_bar',
    'find_sdf_files',
    'print_section',
    'print_parameter_summary',
    'create_output_directory',
    'C_LIGHT', 'M_E', 'Q_E', 'EPS_0'
]
