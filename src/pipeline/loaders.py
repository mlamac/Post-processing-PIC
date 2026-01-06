"""
Data loaders for PIC simulation outputs.
Supports EPOCH (SDF) and Smilei (HDF5) formats.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, List, Union
import numpy as np


class BaseLoader(ABC):
    """Base class for data loaders"""

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

    @abstractmethod
    def load_field(self, field_name: str) -> Dict[str, np.ndarray]:
        """Load a field from the file"""
        pass

    @abstractmethod
    def load_particles(self, species: str) -> Dict[str, np.ndarray]:
        """Load particle data for a given species"""
        pass

    @abstractmethod
    def list_fields(self) -> List[str]:
        """List available fields in the file"""
        pass

    @abstractmethod
    def list_species(self) -> List[str]:
        """List available particle species"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict:
        """Get simulation metadata (time, grid info, etc.)"""
        pass


class SDFLoader(BaseLoader):
    """Loader for EPOCH SDF files"""

    def __init__(self, filepath: Union[str, Path]):
        super().__init__(filepath)
        try:
            import sdf
            self._sdf = sdf
            self.data = sdf.read(str(self.filepath))
        except ImportError:
            raise ImportError(
                "SDF module not found. Install with: pip install sdf"
            )

    def load_field(self, field_name: str) -> Dict[str, np.ndarray]:
        """
        Load a field from SDF file.

        Args:
            field_name: Name of the field (e.g., 'Electric Field/Ex', 'Derived/Number_Density/electron')

        Returns:
            Dictionary containing 'data', 'grid_x', 'grid_y', 'grid_z' (if applicable)
        """
        if not hasattr(self.data, field_name.replace('/', '_')):
            raise ValueError(f"Field '{field_name}' not found in SDF file")

        # SDF stores fields with underscores instead of slashes in attribute names
        field_attr = field_name.replace('/', '_')
        field_data = getattr(self.data, field_attr)

        result = {'data': field_data.data}

        # Get grid information
        if hasattr(field_data, 'grid'):
            grid = field_data.grid
            if hasattr(grid, 'data'):
                # For structured grids
                grid_data = grid.data
                if len(grid_data) >= 1:
                    result['grid_x'] = grid_data[0]
                if len(grid_data) >= 2:
                    result['grid_y'] = grid_data[1]
                if len(grid_data) >= 3:
                    result['grid_z'] = grid_data[2]

        return result

    def load_particles(self, species: str) -> Dict[str, np.ndarray]:
        """
        Load particle data for a given species.

        Args:
            species: Name of species (e.g., 'electron', 'proton')

        Returns:
            Dictionary with particle arrays: 'x', 'y', 'z', 'px', 'py', 'pz', 'weight'
        """
        result = {}

        # Try to load particle data
        for coord in ['x', 'y', 'z']:
            attr_name = f"Particles_{species}_{coord.upper()}"
            if hasattr(self.data, attr_name):
                result[coord] = getattr(self.data, attr_name).data

        for momentum in ['px', 'py', 'pz']:
            attr_name = f"Particles_{species}_P{momentum[1].upper()}"
            if hasattr(self.data, attr_name):
                result[momentum] = getattr(self.data, attr_name).data

        # Weight
        weight_attr = f"Particles_{species}_Weight"
        if hasattr(self.data, weight_attr):
            result['weight'] = getattr(self.data, weight_attr).data

        return result

    def list_fields(self) -> List[str]:
        """List available fields"""
        fields = []
        for key in dir(self.data):
            if not key.startswith('_'):
                attr = getattr(self.data, key)
                if hasattr(attr, 'data') and hasattr(attr, 'grid'):
                    fields.append(key.replace('_', '/'))
        return fields

    def list_species(self) -> List[str]:
        """List available particle species"""
        species = set()
        for key in dir(self.data):
            if key.startswith('Particles_'):
                parts = key.split('_')
                if len(parts) >= 2:
                    species.add(parts[1])
        return list(species)

    def get_metadata(self) -> Dict:
        """Get simulation metadata"""
        metadata = {}
        if hasattr(self.data, 'Header'):
            header = self.data.Header
            metadata['time'] = getattr(header, 'time', None)
            metadata['step'] = getattr(header, 'step', None)
        return metadata


class HDF5Loader(BaseLoader):
    """Loader for Smilei HDF5 files"""

    def __init__(self, filepath: Union[str, Path]):
        super().__init__(filepath)
        try:
            import h5py
            self._h5py = h5py
            self.file = h5py.File(str(self.filepath), 'r')
        except ImportError:
            raise ImportError(
                "h5py module not found. Install with: pip install h5py"
            )

    def __del__(self):
        """Close HDF5 file on deletion"""
        if hasattr(self, 'file'):
            self.file.close()

    def load_field(self, field_name: str) -> Dict[str, np.ndarray]:
        """
        Load a field from HDF5 file.

        Args:
            field_name: Name of the field (e.g., 'Ex', 'Ey', 'Rho_electron')

        Returns:
            Dictionary containing 'data', 'grid_x', 'grid_y', 'grid_z' (if applicable)
        """
        # Smilei typically stores fields under '/Fields/' or '/Rho/'
        possible_paths = [
            f'/Fields/{field_name}',
            f'/Rho/{field_name}',
            f'/{field_name}',
        ]

        field_data = None
        for path in possible_paths:
            if path in self.file:
                field_data = self.file[path][:]
                break

        if field_data is None:
            raise ValueError(f"Field '{field_name}' not found in HDF5 file")

        result = {'data': field_data}

        # Try to load grid information
        if '/x' in self.file:
            result['grid_x'] = self.file['/x'][:]
        if '/y' in self.file:
            result['grid_y'] = self.file['/y'][:]
        if '/z' in self.file:
            result['grid_z'] = self.file['/z'][:]

        return result

    def load_particles(self, species: str) -> Dict[str, np.ndarray]:
        """
        Load particle data for a given species.

        Args:
            species: Name of species (e.g., 'electron', 'ion')

        Returns:
            Dictionary with particle arrays: 'x', 'y', 'z', 'px', 'py', 'pz', 'weight'
        """
        result = {}

        # Smilei stores particles under /Species/{species}/
        species_path = f'/Species/{species}'

        if species_path not in self.file:
            raise ValueError(f"Species '{species}' not found in HDF5 file")

        species_group = self.file[species_path]

        # Load position and momentum data
        for key in ['x', 'y', 'z', 'px', 'py', 'pz', 'weight']:
            if key in species_group:
                result[key] = species_group[key][:]

        return result

    def list_fields(self) -> List[str]:
        """List available fields"""
        fields = []

        # Check common field locations
        for group_name in ['/Fields', '/Rho']:
            if group_name in self.file:
                group = self.file[group_name]
                fields.extend([name for name in group.keys()])

        return fields

    def list_species(self) -> List[str]:
        """List available particle species"""
        if '/Species' in self.file:
            return list(self.file['/Species'].keys())
        return []

    def get_metadata(self) -> Dict:
        """Get simulation metadata"""
        metadata = {}

        # Try to read common metadata
        if '/time' in self.file:
            metadata['time'] = self.file['/time'][()]
        if '/iteration' in self.file:
            metadata['step'] = self.file['/iteration'][()]

        # Read attributes
        for key in self.file.attrs.keys():
            metadata[key] = self.file.attrs[key]

        return metadata


def load_data(filepath: Union[str, Path], format: Optional[str] = None) -> BaseLoader:
    """
    Automatically load data based on file format.

    Args:
        filepath: Path to the data file
        format: Optional format specification ('sdf' or 'hdf5'). If None, auto-detect.

    Returns:
        Appropriate loader instance

    Examples:
        >>> loader = load_data('output.sdf')
        >>> field = loader.load_field('Electric_Field_Ex')
        >>> particles = loader.load_particles('electron')
    """
    filepath = Path(filepath)

    if format is None:
        # Auto-detect based on file extension
        ext = filepath.suffix.lower()
        if ext == '.sdf':
            format = 'sdf'
        elif ext in ['.h5', '.hdf5']:
            format = 'hdf5'
        else:
            # Try to detect by attempting to open
            try:
                import h5py
                with h5py.File(str(filepath), 'r'):
                    format = 'hdf5'
            except:
                try:
                    import sdf
                    sdf.read(str(filepath))
                    format = 'sdf'
                except:
                    raise ValueError(
                        f"Could not determine format for {filepath}. "
                        "Please specify format='sdf' or format='hdf5'"
                    )

    if format.lower() == 'sdf':
        return SDFLoader(filepath)
    elif format.lower() in ['hdf5', 'h5']:
        return HDF5Loader(filepath)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'sdf' or 'hdf5'")
