"""
Data loaders for EPOCH quasi-3D PIC simulations.
Supports loading SDF files using sdf_helper module and HDF5 for synthetic data.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import h5py

# Import utilities
from utils.hpc import get_filename_from_pattern

# Try to import sdf_helper (compiled EPOCH utility)
try:
    import sdf_helper as sh
    HAS_SDF_HELPER = True
except ImportError:
    HAS_SDF_HELPER = False
    sh = None


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
    # Handle case with only 1 mode (axisymmetric)
    if mode_real.shape[-1] == 1:
        return mode_real[:, :, 0]

    # Mode 0 (axisymmetric) + Mode 1 (asymmetric)
    field = (mode_real[:, :, 0] +
             mode_real[:, :, 1] * np.cos(theta) +
             mode_imag[:, :, 1] * np.sin(theta))
    return field


class EPOCHLoader:
    """
    Loader for EPOCH quasi-3D SDF files.

    Usage:
        loader = EPOCHLoader('E_field0060.sdf')
        data = loader.load_frame()
    """

    def __init__(self, filepath: Union[str, Path], use_sdf_helper: bool = True):
        """
        Initialize EPOCH loader.

        Args:
            filepath: Path to SDF file
            use_sdf_helper: Try to use sdf_helper module (True), or HDF5 fallback (False)
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self.use_sdf_helper = use_sdf_helper and HAS_SDF_HELPER
        self.data = None

        if self.use_sdf_helper:
            if not HAS_SDF_HELPER:
                raise ImportError(
                    "sdf_helper module not found. "
                    "Compile with: cd /path/to/epoch/epoch2d && make sdfutils"
                )
            # Load SDF file
            self.data = sh.getdata(str(self.filepath))
        else:
            # Use HDF5 fallback for synthetic data
            self.file = h5py.File(str(self.filepath), 'r')

    def load_electric_fields(self, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Load electric field data from quasi-3D SDF file.

        Args:
            params: Optional dict with E_norm_laser and E_norm_plasma for normalization

        Returns:
            Dict with:
                - E_x: Longitudinal field (nx, nr)
                - E_y: Transverse field at theta=0 (nx, nr)
                - E_z: Transverse field at theta=pi/2 (nx, nr)
                - E_tot: Total transverse field magnitude (nx, nr)
                - E_x_norm, E_tot_norm: Normalized fields (if params provided)
        """
        if self.use_sdf_helper:
            return self._load_fields_sdf(params)
        else:
            return self._load_fields_h5(params)

    def _load_fields_sdf(self, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Load fields from EPOCH SDF file using sdf_helper."""
        # Get modal field data
        Erm_real = self.data.Electric_Field_Modes_Erm_real.data
        Erm_imag = self.data.Electric_Field_Modes_Erm_imag.data
        Exm_real = self.data.Electric_Field_Modes_Exm_real.data
        Exm_imag = self.data.Electric_Field_Modes_Exm_imag.data

        # Reconstruct fields at different azimuthal angles
        E_y = reconstruct_field_from_modes(Erm_real, Erm_imag, theta=0)
        E_z = reconstruct_field_from_modes(Erm_real, Erm_imag, theta=np.pi/2)
        E_x = reconstruct_field_from_modes(Exm_real, Exm_imag, theta=0)

        # Total transverse field
        E_tot = np.sqrt(E_y**2 + E_z**2)

        result = {
            'E_x': E_x,
            'E_y': E_y,
            'E_z': E_z,
            'E_tot': E_tot,
        }

        # Apply normalization if parameters provided
        if params is not None:
            if 'E_norm_plasma' in params:
                result['E_x_norm'] = E_x / params['E_norm_plasma']
            if 'E_norm_laser' in params:
                result['E_tot_norm'] = E_tot / params['E_norm_laser']
                result['E_y_norm'] = E_y / params['E_norm_laser']
                result['E_z_norm'] = E_z / params['E_norm_laser']

        return result

    def _load_fields_h5(self, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Load fields from HDF5 file (synthetic data)."""
        efield_group = self.file['Electric_Field_Modes']

        Erm_real = efield_group['Erm_real'][:]
        Erm_imag = efield_group['Erm_imag'][:]
        Exm_real = efield_group['Exm_real'][:]
        Exm_imag = efield_group['Exm_imag'][:]

        # Reconstruct fields
        E_y = reconstruct_field_from_modes(Erm_real, Erm_imag, theta=0)
        E_z = reconstruct_field_from_modes(Erm_real, Erm_imag, theta=np.pi/2)
        E_x = reconstruct_field_from_modes(Exm_real, Exm_imag, theta=0)

        E_tot = np.sqrt(E_y**2 + E_z**2)

        result = {
            'E_x': E_x,
            'E_y': E_y,
            'E_z': E_z,
            'E_tot': E_tot,
        }

        # Apply normalization
        if params is not None:
            if 'E_norm_plasma' in params:
                result['E_x_norm'] = E_x / params['E_norm_plasma']
            if 'E_norm_laser' in params:
                result['E_tot_norm'] = E_tot / params['E_norm_laser']
                result['E_y_norm'] = E_y / params['E_norm_laser']
                result['E_z_norm'] = E_z / params['E_norm_laser']

        return result

    def load_density(self, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Load electron density data.

        Args:
            params: Optional dict with n_c for normalization

        Returns:
            Dict with:
                - n_e: Electron density (nx, nr) in m^-3
                - n_e_norm: Normalized to critical density (if params provided)
        """
        if self.use_sdf_helper:
            return self._load_density_sdf(params)
        else:
            return self._load_density_h5(params)

    def _load_density_sdf(self, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Load density from EPOCH SDF file."""
        # Try common density field names in EPOCH
        possible_names = [
            'Derived_Number_Density_Subset_total_e',  # Total electrons
            'Derived_Number_Density_electron',         # Electron species
            'Derived_Number_Density_Electron',         # Capital E variant
            'Number_Density_electron',                 # Without Derived prefix
            'Derived_Number_Density_Subset_electron',  # Alternative subset name
        ]
        
        n_e = None
        found_name = None
        
        for name in possible_names:
            attr_name = name.replace('/', '_')
            if hasattr(self.data, attr_name):
                n_e = getattr(self.data, attr_name).data
                found_name = name
                break
        
        if n_e is None:
            # List available fields to help user
            available = [attr for attr in dir(self.data) if not attr.startswith('_')]
            density_fields = [f for f in available if 'density' in f.lower() or 'number' in f.lower()]
            raise ValueError(
                f"Could not find electron density field in SDF file.\n"
                f"Tried: {possible_names}\n"
                f"Available density-like fields: {density_fields}\n"
                f"All fields: {available[:20]}... (showing first 20)"
            )

        result = {'n_e': n_e}

        if params is not None and 'n_c' in params:
            result['n_e_norm'] = n_e / params['n_c']

        return result

    def _load_density_h5(self, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Load density from HDF5 file."""
        n_e = self.file['Derived_Number_Density/Subset_total_e'][:]

        result = {'n_e': n_e}

        if params is not None and 'n_c' in params:
            result['n_e_norm'] = n_e / params['n_c']

        return result

    def load_grid(self, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Load grid coordinates.

        Args:
            params: Optional dict with lambda0 for normalization

        Returns:
            Dict with:
                - x: X-coordinates in meters (1D array)
                - r: R-coordinates in meters (1D array)
                - x_norm, r_norm: Normalized to lambda0 (if params provided)
        """
        if self.use_sdf_helper:
            return self._load_grid_sdf(params)
        else:
            return self._load_grid_h5(params)

    def _load_grid_sdf(self, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Load grid from EPOCH SDF file."""
        # EPOCH stores grid as separate attributes or in Grid_Grid_mid
        if hasattr(self.data, 'Grid_Grid_mid'):
            grid_data = self.data.Grid_Grid_mid.data
            x = grid_data[0]
            r = grid_data[1]
        elif hasattr(self.data, 'Grid_Grid'):
            grid_data = self.data.Grid_Grid.data
            x = grid_data[0]
            r = grid_data[1]
        else:
            # Try individual grid arrays
            if hasattr(self.data, 'Grid_grid_mid'):
                grid_mid = self.data.Grid_grid_mid.data
                x = grid_mid[0]
                r = grid_mid[1]
            else:
                raise ValueError(
                    f"Could not find grid data in SDF file. "
                    f"Available Grid attributes: {[a for a in dir(self.data) if 'Grid' in a or 'grid' in a]}"
                )

        result = {'x': x, 'r': r}

        if params is not None and 'lambda0' in params:
            result['x_norm'] = x / params['lambda0']
            result['r_norm'] = r / params['lambda0']

        return result

    def _load_grid_h5(self, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Load grid from HDF5 file."""
        x = self.file['Grid/Grid_mid/x'][:]
        r = self.file['Grid/Grid_mid/r'][:]

        result = {'x': x, 'r': r}

        if params is not None and 'lambda0' in params:
            result['x_norm'] = x / params['lambda0']
            result['r_norm'] = r / params['lambda0']

        return result

    def load_particles(self, species: str = 'He_electron',
                       params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Load particle data for a given species.

        Args:
            species: Species name (e.g., 'He_electron', 'H_ion')
            params: Optional dict with lambda0 and M_E, C_LIGHT for normalization

        Returns:
            Dict with:
                - x, r: Position arrays in meters
                - px, pr: Momentum arrays in kg*m/s
                - weight: Particle weights in Coulombs
                - x_norm, r_norm: Normalized positions (if params provided)
                - px_norm, pr_norm: Normalized momentum (if params provided)
        """
        if self.use_sdf_helper:
            return self._load_particles_sdf(species, params)
        else:
            return self._load_particles_h5(species, params)

    def _load_particles_sdf(self, species: str, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Load particles from EPOCH SDF file."""
        # Get particle positions from grid
        grid_attr = f"Grid_Particles_{species}"
        if not hasattr(self.data, grid_attr):
            return {'x': np.array([]), 'r': np.array([]), 'px': np.array([]),
                    'pr': np.array([]), 'weight': np.array([])}

        grid_data = getattr(self.data, grid_attr).data
        x = np.squeeze(grid_data[0])
        r = np.squeeze(grid_data[1])

        # Get momentum
        px = np.squeeze(getattr(self.data, f"Particles_Px_{species}").data)
        pr = np.squeeze(getattr(self.data, f"Particles_Pr_{species}").data)

        # Get weight
        weight = np.squeeze(getattr(self.data, f"Particles_Weight_{species}").data)

        # Flatten arrays
        result = {
            'x': x.flatten() if x.size else np.array([]),
            'r': r.flatten() if r.size else np.array([]),
            'px': px.flatten() if px.size else np.array([]),
            'pr': pr.flatten() if pr.size else np.array([]),
            'weight': weight.flatten() if weight.size else np.array([]),
        }

        # Apply normalization
        if params is not None:
            if 'lambda0' in params:
                result['x_norm'] = result['x'] / params['lambda0']
                result['r_norm'] = result['r'] / params['lambda0']
            if 'M_E' in params and 'C_LIGHT' in params:
                from ..utils.physics import M_E, C_LIGHT
                result['px_norm'] = result['px'] / (M_E * C_LIGHT)
                result['pr_norm'] = result['pr'] / (M_E * C_LIGHT)

        return result

    def _load_particles_h5(self, species: str, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Load particles from HDF5 file."""
        particles_group = f'Particles_{species}'
        if particles_group not in self.file:
            return {'x': np.array([]), 'r': np.array([]), 'px': np.array([]),
                    'pr': np.array([]), 'weight': np.array([])}

        x = self.file[f'Grid/{particles_group}/x'][:]
        r = self.file[f'Grid/{particles_group}/r'][:]

        px = self.file[f'{particles_group}/Px'][:]
        pr = self.file[f'{particles_group}/Pr'][:]
        weight = self.file[f'{particles_group}/Weight'][:]

        result = {
            'x': x.flatten(),
            'r': r.flatten(),
            'px': px.flatten(),
            'pr': pr.flatten(),
            'weight': weight.flatten(),
        }

        # Apply normalization
        if params is not None:
            if 'lambda0' in params:
                result['x_norm'] = result['x'] / params['lambda0']
                result['r_norm'] = result['r'] / params['lambda0']
            if 'M_E' in params and 'C_LIGHT' in params:
                from ..utils.physics import M_E, C_LIGHT
                result['px_norm'] = result['px'] / (M_E * C_LIGHT)
                result['pr_norm'] = result['pr'] / (M_E * C_LIGHT)

        return result


    def list_available_fields(self) -> Dict[str, List[str]]:
        """
        List all available fields in the SDF file for debugging.
        
        Returns:
            Dict with categorized field names
        """
        if not self.use_sdf_helper:
            return {'error': ['Only works with sdf_helper mode']}
        
        all_attrs = [attr for attr in dir(self.data) if not attr.startswith('_')]
        
        # Categorize fields
        fields = {
            'electric': [f for f in all_attrs if 'Electric' in f or 'electric' in f],
            'magnetic': [f for f in all_attrs if 'Magnetic' in f or 'magnetic' in f],
            'density': [f for f in all_attrs if 'Density' in f or 'density' in f or 'Number' in f],
            'particles': [f for f in all_attrs if 'Particles' in f or 'particles' in f],
            'grid': [f for f in all_attrs if 'Grid' in f or 'grid' in f],
            'other': [f for f in all_attrs if f not in 
                     [item for sublist in [
                         [x for x in all_attrs if 'Electric' in x or 'electric' in x],
                         [x for x in all_attrs if 'Magnetic' in x or 'magnetic' in x],
                         [x for x in all_attrs if 'Density' in x or 'density' in x or 'Number' in x],
                         [x for x in all_attrs if 'Particles' in x or 'particles' in x],
                         [x for x in all_attrs if 'Grid' in x or 'grid' in x],
                     ] for item in sublist]]
        }
        
        return fields

    def load_frame(self, params: Optional[Dict] = None,
                   downsample_x: int = 1, downsample_r: int = 1,
                   species: str = 'He_electron') -> Dict:
        """
        Load all data for a single frame (convenience method).
        
        NOTE: For real EPOCH data, you should load from separate files:
          - E_field*.sdf contains fields and grid
          - dens*.sdf contains density (optional, can try from E_field)
          - ener*.sdf contains particles
        
        This method assumes all data is in the current file (for synthetic or single-file data).

        Args:
            params: Physical parameters dict from compute_derived_quantities()
            downsample_x: Downsampling factor for x-direction
            downsample_r: Downsampling factor for r-direction
            species: Particle species to load

        Returns:
            Dict with all loaded and normalized data
        """
        grid = self.load_grid(params)
        fields = self.load_electric_fields(params)
        
        # Try to load density (may not be in E_field file)
        try:
            density = self.load_density(params)
        except (AttributeError, ValueError) as e:
            print(f"Warning: Could not load density from this file: {e}")
            print("For EPOCH data, density is typically in dens*.sdf files")
            # Create dummy density
            density = {'n_e': None, 'n_e_norm': None}
        
        # Try to load particles (may not be in E_field file)
        try:
            particles = self.load_particles(species, params)
        except (AttributeError, ValueError) as e:
            print(f"Warning: Could not load particles from this file: {e}")
            print("For EPOCH data, particles are typically in ener*.sdf files")
            particles = {
                'x': np.array([]), 'r': np.array([]),
                'px': np.array([]), 'pr': np.array([]),
                'weight': np.array([])
            }

        # Apply downsampling to field data
        ds_x = downsample_x
        ds_r = downsample_r

        result = {
            'x': grid.get('x_norm', grid['x'])[::ds_x],
            'r': grid.get('r_norm', grid['r'])[::ds_r],
        }

        # Add downsampled fields (transposed for correct shape)
        if 'E_x_norm' in fields:
            result['E_x'] = fields['E_x_norm'][::ds_x, ::ds_r].T
            result['E_tot'] = fields['E_tot_norm'][::ds_x, ::ds_r].T
        else:
            result['E_x'] = fields['E_x'][::ds_x, ::ds_r].T
            result['E_tot'] = fields['E_tot'][::ds_x, ::ds_r].T

        # Add density
        if 'n_e_norm' in density:
            result['n_e'] = density['n_e_norm'][::ds_x, ::ds_r].T
        else:
            result['n_e'] = density['n_e'][::ds_x, ::ds_r].T

        # Add particles (use normalized if available)
        result['x_particles'] = particles.get('x_norm', particles['x'])
        result['r_particles'] = particles.get('r_norm', particles['r'])
        result['px_particles'] = particles.get('px_norm', particles['px'])
        result['pr_particles'] = particles.get('pr_norm', particles['pr'])
        result['weight'] = particles['weight']

        return result

    def close(self):
        """Close file if using HDF5."""
        if not self.use_sdf_helper and hasattr(self, 'file'):
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_epoch_frame(frame_index: int, config: Dict, params: Dict) -> Dict:
    """
    Load a complete EPOCH frame from multiple SDF files (matching working script convention).
    
    EPOCH typically outputs separate files per frame:
      - dens{:04d}.sdf: Density data
      - E_field{:04d}.sdf: Electric field modes and grid
      - ener{:04d}.sdf: Particle/energy data
    
    Args:
        frame_index: Frame number (e.g., 60 for dens0060.sdf)
        config: CONFIG dict with file patterns
        params: Physical parameters from compute_derived_quantities()
    
    Returns:
        Dict with all frame data (fields, density, grid, particles)
    
    Example:
        >>> frame = load_epoch_frame(60, CONFIG, params)
        >>> n_e = frame['n_e']
        >>> E_tot = frame['E_tot']
    """
    # Get filenames for this frame
    dens_file = get_filename_from_pattern(config['dens_pattern'], frame_index)
    efield_file = get_filename_from_pattern(config['efield_pattern'], frame_index)
    ener_file = get_filename_from_pattern(config['ener_pattern'], frame_index)
    
    # Load E-field file (contains fields and grid)
    loader_efield = EPOCHLoader(efield_file, use_sdf_helper=True)
    fields = loader_efield.load_electric_fields(params)
    grid = loader_efield.load_grid(params)
    loader_efield.close()
    
    # Load density file
    loader_dens = EPOCHLoader(dens_file, use_sdf_helper=True)
    density = loader_dens.load_density(params)
    loader_dens.close()
    
    # Load particle file
    loader_ener = EPOCHLoader(ener_file, use_sdf_helper=True)
    particles = loader_ener.load_particles(config.get('species', 'He_electron'), params)
    loader_ener.close()
    
    # Apply downsampling
    ds_x = config.get('downsample_x', 1)
    ds_r = config.get('downsample_r', 1)
    
    result = {
        'x': grid.get('x_norm', grid['x'])[::ds_x],
        'r': grid.get('r_norm', grid['r'])[::ds_r],
    }
    
    # Add downsampled fields (transposed for correct shape)
    if 'E_x_norm' in fields:
        result['E_x'] = fields['E_x_norm'][::ds_x, ::ds_r].T
        result['E_tot'] = fields['E_tot_norm'][::ds_x, ::ds_r].T
    else:
        result['E_x'] = fields['E_x'][::ds_x, ::ds_r].T
        result['E_tot'] = fields['E_tot'][::ds_x, ::ds_r].T
    
    # Add density
    if 'n_e_norm' in density:
        result['n_e'] = density['n_e_norm'][::ds_x, ::ds_r].T
    else:
        result['n_e'] = density['n_e'][::ds_x, ::ds_r].T
    
    # Add particles (use normalized if available)
    result['x_particles'] = particles.get('x_norm', particles['x'])
    result['r_particles'] = particles.get('r_norm', particles['r'])
    result['px_particles'] = particles.get('px_norm', particles['px'])
    result['pr_particles'] = particles.get('pr_norm', particles['pr'])
    result['weight'] = particles['weight']
    
    return result
