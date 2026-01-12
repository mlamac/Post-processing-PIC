"""Pipeline module for EPOCH data loading, processing, and visualization"""

from .loaders import load_frame_data, find_sdf_files, reconstruct_field_from_modes
from .processors import compute_momentum_histogram, compute_beam_statistics
from .visualizers import create_animation, save_animation_html

__all__ = [
    'load_frame_data',
    'find_sdf_files',
    'reconstruct_field_from_modes',
    'compute_momentum_histogram',
    'compute_beam_statistics',
    'create_animation',
    'save_animation_html',
]
