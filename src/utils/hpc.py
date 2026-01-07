"""
HPC-compatible utilities for batch processing.
Progress bars, file finding, and I/O helpers.
"""

import os
import sys
import glob
import re
from pathlib import Path
from typing import List, Optional, Tuple


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

def progress_bar(current: int, total: int, prefix: str = 'Progress', bar_length: int = 40):
    """
    Display a simple text-based progress bar for HPC jobs.
    Compatible with non-interactive environments (no cursor control libraries).

    Args:
        current: Current progress (1-indexed)
        total: Total number of items
        prefix: Description text before the bar
        bar_length: Length of the progress bar in characters

    Example:
        >>> for i in range(100):
        >>>     progress_bar(i+1, 100, 'Processing')
        Processing: [========================================] 100.0% (100/100)
    """
    percent = 100.0 * current / total
    filled_length = int(bar_length * current / total)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)

    print(f'\r{prefix}: [{bar}] {percent:.1f}% ({current}/{total})', end='', flush=True)

    if current == total:
        print()  # Newline at end


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def find_sdf_files(pattern: str, directory: str = '.') -> List[int]:
    """
    Find SDF files matching pattern and return sorted list of frame indices.

    Args:
        pattern: Glob pattern like 'dens*.sdf' or 'E_field*.sdf'
        directory: Directory to search (default: current directory)

    Returns:
        Sorted list of integer frame indices (e.g., [60, 120, 180, ...])

    Example:
        >>> indices = find_sdf_files('dens*.sdf')
        >>> print(indices)
        [60, 120, 180, 240, 300]
    """
    # Change to directory if specified
    original_dir = os.getcwd()
    if directory != '.':
        os.chdir(directory)

    try:
        files = glob.glob(pattern)
        indices = []

        # Extract frame number from filename
        # Matches patterns like: dens0060.sdf, E_field0120.sdf, etc.
        for f in files:
            basename = os.path.basename(f)
            # Look for 4-digit number before .sdf extension
            match = re.search(r'(\d{4})\.sdf$', basename)
            if match:
                indices.append(int(match.group(1)))

        return sorted(indices)

    finally:
        # Always return to original directory
        os.chdir(original_dir)


def find_all_sdf_patterns(directory: str = '.') -> dict:
    """
    Scan directory for common SDF file patterns and return available types.

    Args:
        directory: Directory to search

    Returns:
        Dict mapping pattern names to lists of frame indices
        Example: {'dens': [60, 120], 'E_field': [60, 120], 'particles': [60, 120]}
    """
    common_patterns = {
        'dens': 'dens*.sdf',
        'E_field': 'E_field*.sdf',
        'efield': 'efield*.sdf',
        'particles': 'particles*.sdf',
        'ener': 'ener*.sdf',
        'fields': 'fields*.sdf',
    }

    found = {}
    for name, pattern in common_patterns.items():
        indices = find_sdf_files(pattern, directory)
        if indices:
            found[name] = indices

    return found


def get_filename_from_pattern(pattern: str, index: int) -> str:
    """
    Generate filename from pattern and frame index.

    Args:
        pattern: Format string like 'dens{:04d}.sdf' or 'E_field{:04d}.sdf'
        index: Frame index (e.g., 60)

    Returns:
        Formatted filename (e.g., 'dens0060.sdf')

    Example:
        >>> get_filename_from_pattern('dens{:04d}.sdf', 60)
        'dens0060.sdf'
    """
    if '{' in pattern and '}' in pattern:
        # Pattern has format specifier
        return pattern.format(index)
    else:
        # Pattern is glob-style, try to infer format
        # Replace * with {:04d}
        pattern_fmt = pattern.replace('*', '{:04d}')
        return pattern_fmt.format(index)


# =============================================================================
# DATA CACHING
# =============================================================================

def create_output_directory(path: str, verbose: bool = True) -> Path:
    """
    Create output directory if it doesn't exist.

    Args:
        path: Path to directory
        verbose: Print creation message

    Returns:
        Path object for the directory
    """
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose and not output_path.exists():
        print(f"Created output directory: {output_path}")

    return output_path


def check_cache_exists(cache_dir: str, prefix: str, required_files: List[str]) -> bool:
    """
    Check if all required cache files exist.

    Args:
        cache_dir: Directory containing cache files
        prefix: Filename prefix (e.g., 'lwfa_q3d')
        required_files: List of suffixes to check (e.g., ['_n_e.npy', '_E_x.npy'])

    Returns:
        True if all files exist, False otherwise

    Example:
        >>> exists = check_cache_exists('output', 'lwfa', ['_n_e.npy', '_x.npy'])
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return False

    for suffix in required_files:
        filepath = cache_path / f"{prefix}{suffix}"
        if not filepath.exists():
            return False

    return True


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config(config: dict, required_keys: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that configuration dict contains all required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required key names

    Returns:
        (is_valid, error_message) tuple

    Example:
        >>> config = {'lambda0_um': 1.0, 'a0': 3.0}
        >>> valid, msg = validate_config(config, ['lambda0_um', 'a0', 'n_over_nc'])
        >>> if not valid:
        >>>     print(msg)
    """
    missing = [key for key in required_keys if key not in config]

    if missing:
        error_msg = f"Missing required config keys: {', '.join(missing)}"
        return False, error_msg

    return True, None


def print_section(title: str, width: int = 60, char: str = '='):
    """
    Print a formatted section header for console output.

    Args:
        title: Section title
        width: Total width of the header
        char: Character to use for borders

    Example:
        >>> print_section("Processing Data")
        ============================================================
        Processing Data
        ============================================================
    """
    print(char * width)
    if title:
        print(title)
        print(char * width)
    else:
        # Just a separator line
        pass


def print_parameter_summary(params: dict, title: str = "Parameters"):
    """
    Print a formatted summary of parameters.

    Args:
        params: Dictionary of parameters to display
        title: Section title

    Example:
        >>> params = {'lambda0': 1e-6, 'n_c': 1.7e27}
        >>> print_parameter_summary(params, "Laser Parameters")
    """
    print(f"\n{title}:")
    for key, value in params.items():
        if isinstance(value, float):
            if abs(value) > 1e4 or abs(value) < 1e-3:
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
