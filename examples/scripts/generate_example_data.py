#!/usr/bin/env python3
"""
Generate synthetic LWFA data for examples and testing.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.synthetic_data import generate_lwfa_data


def main():
    """Generate example LWFA data"""

    output_path = Path(__file__).parent.parent / 'data' / 'example_lwfa.h5'

    print("Generating synthetic LWFA data...")
    print("=" * 60)

    data = generate_lwfa_data(
        output_path=str(output_path),
        nx=400,
        ny=200,
        n_particles=10000,
        plasma_density=1e24,
        laser_a0=2.0,
        seed=42
    )

    print("=" * 60)
    print("Done! Example data saved to:")
    print(f"  {output_path}")
    print("\nYou can now run the visualization examples.")


if __name__ == '__main__':
    main()
