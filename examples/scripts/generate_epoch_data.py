#!/usr/bin/env python3
"""
Generate synthetic EPOCH quasi-3D LWFA data for testing.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.synthetic_data import generate_epoch_lwfa_data


def main():
    """Generate example EPOCH quasi-3D data"""

    output_path = Path(__file__).parent.parent / 'data' / 'epoch_lwfa_example.h5'

    print("=" * 60)
    print("Generating Synthetic EPOCH Quasi-3D LWFA Data")
    print("=" * 60)
    print()

    config = {
        'lambda0_um': 1.0,
        'a0': 3.0,
        'n_over_nc': 0.0025,
        'dumpstep_fs': 160,
    }

    data = generate_epoch_lwfa_data(
        output_path=str(output_path),
        nx=400,
        nr=200,
        x_range=(0, 200),
        r_max=100,
        n_particles=5000,
        config=config,
        seed=42
    )

    print()
    print("=" * 60)
    print("Done! Synthetic data saved to:")
    print(f"  {output_path}")
    print()
    print("You can now run process_epoch_lwfa.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
