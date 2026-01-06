#!/usr/bin/env python3
"""
Complete example: Load and visualize LWFA simulation data.
Generates both interactive HTML and publication-quality PNG outputs.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pipeline.loaders import load_data
from pipeline.processors import DataProcessor
from pipeline.visualizers import Visualizer


def main():
    """Main processing pipeline"""

    # Configuration
    data_file = Path(__file__).parent.parent / 'data' / 'example_lwfa.h5'
    output_dir = Path(__file__).parent.parent.parent / 'output'

    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please run generate_example_data.py first.")
        return

    print("=" * 60)
    print("PIC Post-Processing Pipeline - LWFA Example")
    print("=" * 60)

    # Initialize
    print("\n1. Loading data...")
    loader = load_data(data_file)
    visualizer = Visualizer(output_dir=output_dir)
    processor = DataProcessor()

    # Load fields
    print("   - Loading electron density...")
    density_data = loader.load_field('Rho_electron')
    n_e = density_data['data']
    x = density_data['grid_x']
    y = density_data['grid_y']

    print("   - Loading electric fields...")
    ex_data = loader.load_field('Ex')
    E_x = ex_data['data']

    ey_data = loader.load_field('Ey')
    E_y = ey_data['data']

    # Load particles
    print("   - Loading particle data...")
    particles = loader.load_particles('electron')

    # Get metadata
    metadata = loader.get_metadata()
    print(f"\n   Simulation time: {metadata.get('time', 'N/A')}")
    print(f"   Grid size: {n_e.shape}")
    print(f"   Number of particles: {len(particles['x'])}")

    # Process data
    print("\n2. Processing data...")

    # Compute particle energies
    energy_MeV = processor.compute_particle_energy(
        particles['px'], particles['py'], particles['pz']
    )
    print(f"   - Mean particle energy: {np.mean(energy_MeV):.2f} MeV")
    print(f"   - Max particle energy: {np.max(energy_MeV):.2f} MeV")

    # Compute energy spectrum
    energy_bins, spectrum = processor.compute_energy_spectrum(
        particles['px'], particles['py'], particles['pz'],
        weight=particles['weight'],
        bins=50
    )

    # Compute charge
    total_charge_pC = processor.compute_charge(particles['weight'])
    print(f"   - Total accelerated charge: {total_charge_pC:.2f} pC")

    # Create visualizations
    print("\n3. Creating visualizations...")

    # 1. Density with E-field overlay
    print("   - Density with transverse E-field overlay...")
    visualizer.plot_density_with_field_overlay(
        x=x,
        y=y,
        density=n_e,
        field=E_y,
        output_name='lwfa_density_ey',
        title='LWFA: Electron Density with Transverse E-Field',
        density_label='n_e [m⁻³]',
        field_label='E_y [V/m]',
        save_png=True,
        save_html=True,
        figsize=(14, 6),
        smooth_density=0.5,
        smooth_field=0.5
    )

    # 2. Phase space plots
    print("   - Phase space: x-px...")
    visualizer.plot_phase_space(
        x=particles['x'],
        px=particles['px'],
        weight=particles['weight'],
        output_name='phase_space_x_px',
        title='Phase Space: x-px',
        xlabel='x [μm]',
        ylabel='p_x [m_e c]',
        save_png=True,
        save_html=True,
        bins=(100, 100)
    )

    print("   - Phase space: y-py...")
    visualizer.plot_phase_space(
        x=particles['y'] * 1e6,  # Already convert to microns
        px=particles['py'],
        weight=particles['weight'],
        output_name='phase_space_y_py',
        title='Phase Space: y-py',
        xlabel='y [μm]',
        ylabel='p_y [m_e c]',
        save_png=True,
        save_html=True,
        bins=(100, 100)
    )

    # 3. Energy spectrum
    print("   - Energy spectrum...")
    visualizer.plot_energy_spectrum(
        energy=energy_bins,
        spectrum=spectrum,
        output_name='energy_spectrum',
        title=f'Electron Energy Spectrum (Q = {total_charge_pC:.2f} pC)',
        save_png=True,
        save_html=True,
        log_scale=False
    )

    # 4. Particle spatial distribution colored by energy
    print("   - Particle distribution...")
    visualizer.plot_particle_distribution(
        x=particles['x'],
        y=particles['y'],
        color_by=energy_MeV,
        output_name='particle_distribution',
        title='Particle Spatial Distribution',
        color_label='Energy [MeV]',
        save_png=True,
        save_html=True,
        alpha=0.6,
        marker_size=2.0
    )

    # 5. Longitudinal E-field (acceleration field)
    print("   - Longitudinal E-field...")
    visualizer.plot_density_with_field_overlay(
        x=x,
        y=y,
        density=n_e,
        field=E_x,
        output_name='lwfa_density_ex',
        title='LWFA: Electron Density with Longitudinal E-Field',
        density_label='n_e [m⁻³]',
        field_label='E_x [V/m]',
        save_png=True,
        save_html=True,
        figsize=(14, 6),
        smooth_density=0.5,
        smooth_field=0.5
    )

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 60)
    print("\nGenerated files:")
    print("  PNG files (publication-quality):")
    for png_file in sorted(output_dir.glob('*.png')):
        print(f"    - {png_file.name}")
    print("\n  HTML files (interactive):")
    for html_file in sorted(output_dir.glob('*.html')):
        print(f"    - {html_file.name}")


if __name__ == '__main__':
    main()
