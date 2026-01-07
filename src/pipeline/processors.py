"""
Data processing utilities for EPOCH quasi-3D PIC simulations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.physics import M_E, C_LIGHT, compute_kinetic_energy_MeV


class DataProcessor:
    """Process and analyze EPOCH quasi-3D simulation data"""

    @staticmethod
    def compute_momentum_histogram(px: np.ndarray, weights: Optional[np.ndarray] = None,
                                   bins: np.ndarray = None, n_bins: int = 2000,
                                   px_min: float = 1, px_max: float = 10000,
                                   log_bins: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute weighted momentum histogram.

        Args:
            px: Momentum array (normalized to m_e*c)
            weights: Particle weights
            bins: Explicit bin edges (if provided, overrides n_bins/px_min/px_max)
            n_bins: Number of bins (if bins not provided)
            px_min: Minimum momentum for binning
            px_max: Maximum momentum for binning
            log_bins: Use log-spaced bins

        Returns:
            (bin_centers, histogram) tuple
        """
        if weights is None:
            weights = np.ones_like(px)

        if bins is None:
            if log_bins:
                bins = np.logspace(np.log10(px_min), np.log10(px_max), n_bins + 1)
            else:
                bins = np.linspace(px_min, px_max, n_bins + 1)

        hist, _ = np.histogram(px, bins=bins, weights=weights)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        return bin_centers, hist

    @staticmethod
    def compute_energy_histogram(px: np.ndarray, pr: np.ndarray = None,
                                 weights: Optional[np.ndarray] = None,
                                 n_bins: int = 100,
                                 energy_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def compute_charge(weights: np.ndarray) -> float:
        """
        Compute total charge from particle weights.

        Args:
            weights: Particle weights in Coulombs

        Returns:
            Total charge in pC (picoCoulombs)
        """
        return np.sum(weights) * 1e12

    @staticmethod
    def downsample_field(data: np.ndarray, factor_x: int = 1, factor_r: int = 1) -> np.ndarray:
        """
        Downsample 2D field data.

        Args:
            data: Input array (nx, nr)
            factor_x: Downsampling factor in x
            factor_r: Downsampling factor in r

        Returns:
            Downsampled array
        """
        return data[::factor_x, ::factor_r]

    @staticmethod
    def filter_particles_by_energy(px: np.ndarray, pr: np.ndarray,
                                   weights: np.ndarray,
                                   x: np.ndarray, r: np.ndarray,
                                   energy_min: float = 0,
                                   energy_max: float = np.inf) -> Dict[str, np.ndarray]:
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

    @staticmethod
    def compute_beam_statistics(px: np.ndarray, pr: np.ndarray,
                                weights: Optional[np.ndarray] = None) -> Dict[str, float]:
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
