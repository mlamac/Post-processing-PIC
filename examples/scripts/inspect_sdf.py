#!/usr/bin/env python3
"""
Diagnostic script to inspect EPOCH SDF file contents.
Run this to see what fields are available in your SDF files.

Usage:
    python inspect_sdf.py E_field0060.sdf
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    import sdf_helper as sh
    HAS_SDF = True
except ImportError:
    HAS_SDF = False
    print("ERROR: sdf_helper not found. Compile EPOCH utilities first.")
    print("cd /path/to/epoch/epoch2d && make sdfutils")
    sys.exit(1)


def inspect_sdf_file(filepath):
    """Inspect an SDF file and print all available fields."""

    print("=" * 70)
    print(f"Inspecting: {filepath}")
    print("=" * 70)

    try:
        data = sh.getdata(filepath)
    except Exception as e:
        print(f"ERROR loading file: {e}")
        return

    # Get all attributes
    all_attrs = [attr for attr in dir(data) if not attr.startswith('_')]

    # Categorize
    electric = [f for f in all_attrs if 'Electric' in f or 'electric' in f]
    magnetic = [f for f in all_attrs if 'Magnetic' in f or 'magnetic' in f]
    density = [f for f in all_attrs if 'Density' in f or 'density' in f or 'Number' in f]
    particles = [f for f in all_attrs if 'Particles' in f or 'particles' in f]
    grid = [f for f in all_attrs if 'Grid' in f or 'grid' in f]
    other = [f for f in all_attrs if f not in
             electric + magnetic + density + particles + grid]

    print("\n📊 ELECTRIC FIELDS:")
    for f in electric:
        print(f"  - {f}")

    print("\n📊 MAGNETIC FIELDS:")
    for f in magnetic:
        print(f"  - {f}")

    print("\n📊 DENSITY FIELDS:")
    for f in density:
        attr = getattr(data, f)
        shape = attr.data.shape if hasattr(attr, 'data') else "?"
        print(f"  - {f}  (shape: {shape})")

    print("\n📊 PARTICLE FIELDS:")
    for f in particles:
        print(f"  - {f}")

    print("\n📊 GRID FIELDS:")
    for f in grid:
        attr = getattr(data, f)
        shape = attr.data.shape if hasattr(attr, 'data') else "?"
        print(f"  - {f}  (shape: {shape})")

    if other:
        print("\n📊 OTHER FIELDS:")
        for f in other[:10]:  # Show first 10
            print(f"  - {f}")
        if len(other) > 10:
            print(f"  ... and {len(other)-10} more")

    print("\n" + "=" * 70)
    print(f"Total fields: {len(all_attrs)}")
    print("=" * 70)

    # Suggest correct field names
    print("\n💡 SUGGESTED CONFIG SETTINGS:")

    if density:
        print(f"\nFor density, use one of:")
        for f in density:
            print(f"  CONFIG['density_field'] = '{f}'")

    if electric:
        print(f"\nElectric field modes detected:")
        for f in electric:
            if 'mode' in f.lower() or 'Mode' in f:
                print(f"  {f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_sdf.py <sdf_file>")
        print("Example: python inspect_sdf.py E_field0060.sdf")
        sys.exit(1)

    filepath = sys.argv[1]
    inspect_sdf_file(filepath)
