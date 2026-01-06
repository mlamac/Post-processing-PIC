"""
Data processing utilities for PIC simulation data.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import ndimage


class DataProcessor:
    """Process and analyze PIC simulation data"""

    @staticmethod
    def normalize_field(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize field data.

        Args:
            data: Input array
            method: Normalization method ('minmax', 'std', 'none')

        Returns:
            Normalized array
        """
        if method == 'minmax':
            min_val, max_val = np.min(data), np.max(data)
            if max_val - min_val > 0:
                return (data - min_val) / (max_val - min_val)
            return data
        elif method == 'std':
            mean, std = np.mean(data), np.std(data)
            if std > 0:
                return (data - mean) / std
            return data - mean
        else:
            return data

    @staticmethod
    def compute_particle_energy(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                               rest_energy_MeV: float = 0.511) -> np.ndarray:
        """
        Compute particle energy from momentum.

        Args:
            px, py, pz: Momentum components (in units of m*c)
            rest_energy_MeV: Rest mass energy in MeV (default: electron)

        Returns:
            Energy in MeV
        """
        p_squared = px**2 + py**2 + pz**2
        gamma = np.sqrt(1 + p_squared)
        energy_MeV = (gamma - 1) * rest_energy_MeV
        return energy_MeV

    @staticmethod
    def compute_energy_spectrum(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                                weight: Optional[np.ndarray] = None,
                                bins: int = 50,
                                energy_range: Optional[Tuple[float, float]] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute particle energy spectrum.

        Args:
            px, py, pz: Momentum components
            weight: Particle weights
            bins: Number of histogram bins
            energy_range: (E_min, E_max) in MeV

        Returns:
            (energy_bins, spectrum) tuple
        """
        energy = DataProcessor.compute_particle_energy(px, py, pz)

        if weight is None:
            weight = np.ones_like(energy)

        if energy_range is None:
            energy_range = (0, np.max(energy) * 1.1)

        spectrum, bin_edges = np.histogram(
            energy, bins=bins, range=energy_range, weights=weight
        )
        energy_bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        return energy_bins, spectrum

    @staticmethod
    def smooth_field(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Smooth field data using Gaussian filter.

        Args:
            data: Input array
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Smoothed array
        """
        return ndimage.gaussian_filter(data, sigma=sigma)

    @staticmethod
    def compute_divergence(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
        """
        Compute beam divergence angles.

        Args:
            px, py, pz: Momentum components

        Returns:
            Divergence angles in radians
        """
        p_total = np.sqrt(px**2 + py**2 + pz**2)
        p_trans = np.sqrt(py**2 + pz**2)
        divergence = np.arctan2(p_trans, np.abs(px))
        return divergence

    @staticmethod
    def downsample_field(data: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        Downsample field data for faster plotting.

        Args:
            data: Input array
            factor: Downsampling factor

        Returns:
            Downsampled array
        """
        if factor <= 1:
            return data
        return data[::factor, ::factor]

    @staticmethod
    def compute_charge(weight: np.ndarray) -> float:
        """
        Compute total charge from particle weights.

        Args:
            weight: Particle weights (in Coulombs)

        Returns:
            Total charge in pC (picoCoulombs)
        """
        total_charge_C = np.sum(weight)
        return total_charge_C * 1e12  # Convert to pC
