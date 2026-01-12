"""
Data processing utilities for EPOCH quasi-3D PIC simulations.
"""

import numpy as np
from utils.physics import compute_kinetic_energy_MeV


def compute_momentum_histogram(px, weights, bins):
    """
    Compute weighted momentum histogram.

    Args:
        px: Momentum array (normalized to m_e*c)
        weights: Particle weights
        bins: Bin edges array

    Returns:
        Histogram counts array
    """
    hist, _ = np.histogram(px, bins=bins, weights=weights)
    return hist


def compute_energy_histogram(px, pr=None, weights=None, n_bins=100, energy_range=None):
    """
    Compute particle energy histogram.

    Args:
        px: Longitudinal momentum (m_e*c units)
        pr: Radial momentum (m_e*c units, optional)
        weights: Particle weights
        n_bins: Number of bins
        energy_range: (E_min, E_max) in MeV

    Returns:
        (energy_bins, histogram) tuple
    """
    if pr is None:
        pr = np.zeros_like(px)

    pz = np.zeros_like(px)  # Assume no z-momentum in quasi-3D
    energy_MeV = compute_kinetic_energy_MeV(px, pr, pz)

    if weights is None:
        weights = np.ones_like(px)

    if energy_range is None:
        energy_range = (0, np.max(energy_MeV) * 1.1)

    hist, bin_edges = np.histogram(energy_MeV, bins=n_bins, range=energy_range, weights=weights)
    energy_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return energy_bins, hist


def compute_charge(weights):
    """
    Compute total charge from particle weights.

    Args:
        weights: Particle weights in Coulombs

    Returns:
        Total charge in pC (picoCoulombs)
    """
    return np.sum(weights) * 1e12


def filter_particles_by_energy(px, pr, weights, x, r, energy_min=0, energy_max=np.inf):
    """
    Filter particles by energy range.

    Args:
        px, pr: Momentum arrays (m_e*c units)
        weights: Particle weights
        x, r: Position arrays
        energy_min, energy_max: Energy range in MeV

    Returns:
        Dict with filtered arrays
    """
    pz = np.zeros_like(px)
    energy = compute_kinetic_energy_MeV(px, pr, pz)

    mask = (energy >= energy_min) & (energy <= energy_max)

    return {
        'px': px[mask],
        'pr': pr[mask],
        'x': x[mask],
        'r': r[mask],
        'weight': weights[mask],
        'energy': energy[mask]
    }


def compute_beam_statistics(px, pr, weights=None):
    """
    Compute beam quality statistics.

    Args:
        px, pr: Momentum arrays (m_e*c units)
        weights: Particle weights

    Returns:
        Dict with statistical quantities
    """
    if weights is None:
        weights = np.ones_like(px)

    pz = np.zeros_like(px)
    energy = compute_kinetic_energy_MeV(px, pr, pz)

    # Weighted statistics
    total_weight = np.sum(weights)
    mean_energy = np.average(energy, weights=weights)
    std_energy = np.sqrt(np.average((energy - mean_energy)**2, weights=weights))

    mean_px = np.average(px, weights=weights)
    std_px = np.sqrt(np.average((px - mean_px)**2, weights=weights))

    # Divergence (transverse momentum over longitudinal)
    divergence = pr / (np.abs(px) + 1e-9)
    mean_divergence = np.average(np.abs(divergence), weights=weights)

    return {
        'mean_energy_MeV': mean_energy,
        'std_energy_MeV': std_energy,
        'energy_spread': std_energy / (mean_energy + 1e-9),
        'mean_px': mean_px,
        'std_px': std_px,
        'mean_divergence': mean_divergence,
        'total_charge_pC': total_weight * 1e12,
    }
