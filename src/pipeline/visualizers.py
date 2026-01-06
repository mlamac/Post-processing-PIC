"""
Visualization module for PIC simulation data.
Creates interactive HTML plots and publication-quality PNG figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from .processors import DataProcessor


class Visualizer:
    """Create visualizations for PIC simulation data"""

    def __init__(self, output_dir: Union[str, Path] = 'output'):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for saving output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create custom colormaps
        self._setup_colormaps()

    def _setup_colormaps(self):
        """Setup custom colormaps for visualization"""

        # Transparent seismic colormap (divergent, transparent near zero)
        # Blue (negative) -> Transparent (zero) -> Red (positive)
        colors_seismic = [
            (0.0, (0.0, 0.0, 0.6, 1.0)),   # Dark blue, opaque
            (0.25, (0.3, 0.3, 1.0, 0.7)),  # Blue, semi-transparent
            (0.45, (0.7, 0.7, 1.0, 0.2)),  # Light blue, very transparent
            (0.5, (1.0, 1.0, 1.0, 0.0)),   # White, fully transparent
            (0.55, (1.0, 0.7, 0.7, 0.2)),  # Light red, very transparent
            (0.75, (1.0, 0.3, 0.3, 0.7)),  # Red, semi-transparent
            (1.0, (0.6, 0.0, 0.0, 1.0))    # Dark red, opaque
        ]

        cmap_data = [(pos, color) for pos, color in colors_seismic]
        positions = [pos for pos, _ in cmap_data]
        colors = [color for _, color in cmap_data]

        self.cmap_seismic_transparent = LinearSegmentedColormap.from_list(
            'seismic_transparent',
            list(zip(positions, colors))
        )

        # Standard grayscale for density
        self.cmap_density = plt.cm.gray

    def plot_density_with_field_overlay(
        self,
        x: np.ndarray,
        y: np.ndarray,
        density: np.ndarray,
        field: np.ndarray,
        output_name: str = 'density_field',
        title: str = 'Electron Density with Transverse E-Field',
        density_label: str = 'Electron Density',
        field_label: str = 'Transverse E-Field',
        save_png: bool = True,
        save_html: bool = True,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 150,
        field_symmetric: bool = True,
        smooth_density: float = 0.0,
        smooth_field: float = 0.0
    ) -> None:
        """
        Create density plot with field overlay.

        Args:
            x: X-axis coordinates (m)
            y: Y-axis coordinates (m)
            density: 2D electron density array
            field: 2D field array (e.g., E_y)
            output_name: Base name for output files
            title: Plot title
            density_label: Label for density colorbar
            field_label: Label for field colorbar
            save_png: Save static PNG
            save_html: Save interactive HTML
            figsize: Figure size for PNG
            dpi: DPI for PNG
            field_symmetric: Use symmetric colormap for field
            smooth_density: Gaussian smoothing sigma for density
            smooth_field: Gaussian smoothing sigma for field
        """

        # Apply smoothing if requested
        if smooth_density > 0:
            density = DataProcessor.smooth_field(density, smooth_density)
        if smooth_field > 0:
            field = DataProcessor.smooth_field(field, smooth_field)

        # Convert to microns for display
        x_um = x * 1e6
        y_um = y * 1e6

        # Static PNG plot with matplotlib
        if save_png:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

            # Plot density in grayscale
            im1 = ax.imshow(
                density.T,
                extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
                aspect='auto',
                cmap=self.cmap_density,
                origin='lower',
                interpolation='bilinear'
            )

            # Overlay field with transparent divergent colormap
            if field_symmetric:
                vmax_field = np.max(np.abs(field))
                vmin_field = -vmax_field
            else:
                vmin_field = np.min(field)
                vmax_field = np.max(field)

            im2 = ax.imshow(
                field.T,
                extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
                aspect='auto',
                cmap=self.cmap_seismic_transparent,
                origin='lower',
                vmin=vmin_field,
                vmax=vmax_field,
                interpolation='bilinear'
            )

            # Colorbars
            cbar1 = plt.colorbar(im1, ax=ax, pad=0.02, fraction=0.046)
            cbar1.set_label(density_label, rotation=270, labelpad=20)

            cbar2 = plt.colorbar(im2, ax=ax, pad=0.12, fraction=0.046)
            cbar2.set_label(field_label, rotation=270, labelpad=20)

            ax.set_xlabel('x [μm]')
            ax.set_ylabel('y [μm]')
            ax.set_title(title)

            plt.tight_layout()
            png_path = self.output_dir / f'{output_name}.png'
            plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved PNG: {png_path}")

        # Interactive HTML plot with plotly
        if save_html:
            # Create subplots with shared axes
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(density_label, field_label),
                horizontal_spacing=0.12
            )

            # Density plot
            fig.add_trace(
                go.Heatmap(
                    x=x_um,
                    y=y_um,
                    z=density.T,
                    colorscale='Greys',
                    colorbar=dict(x=0.42, len=0.9),
                    name=density_label
                ),
                row=1, col=1
            )

            # Field plot
            if field_symmetric:
                vmax_field = np.max(np.abs(field))
                zmin, zmax = -vmax_field, vmax_field
            else:
                zmin, zmax = np.min(field), np.max(field)

            fig.add_trace(
                go.Heatmap(
                    x=x_um,
                    y=y_um,
                    z=field.T,
                    colorscale='RdBu_r',
                    zmid=0,
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=dict(x=1.02, len=0.9),
                    name=field_label
                ),
                row=1, col=2
            )

            # Update layout
            fig.update_xaxes(title_text='x [μm]', row=1, col=1)
            fig.update_xaxes(title_text='x [μm]', row=1, col=2)
            fig.update_yaxes(title_text='y [μm]', row=1, col=1)
            fig.update_yaxes(title_text='y [μm]', row=1, col=2)

            fig.update_layout(
                title_text=title,
                height=500,
                width=1200,
                showlegend=False
            )

            html_path = self.output_dir / f'{output_name}.html'
            fig.write_html(str(html_path))
            print(f"Saved HTML: {html_path}")

    def plot_phase_space(
        self,
        x: np.ndarray,
        px: np.ndarray,
        weight: Optional[np.ndarray] = None,
        output_name: str = 'phase_space_x_px',
        title: str = 'Phase Space: x-px',
        xlabel: str = 'x [μm]',
        ylabel: str = 'px [m_e c]',
        save_png: bool = True,
        save_html: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        bins: Tuple[int, int] = (100, 100),
        log_scale: bool = True
    ) -> None:
        """
        Create phase space plot.

        Args:
            x: Position array (m)
            px: Momentum array (m_e c units)
            weight: Particle weights
            output_name: Base name for output files
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_png: Save static PNG
            save_html: Save interactive HTML
            figsize: Figure size for PNG
            dpi: DPI for PNG
            bins: Number of bins (nx, ny)
            log_scale: Use log scale for colormap
        """

        # Convert x to microns
        x_um = x * 1e6

        if weight is None:
            weight = np.ones_like(x)

        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(
            x_um, px, bins=bins, weights=weight
        )

        if log_scale:
            H = np.log10(H + 1)  # +1 to avoid log(0)

        # Static PNG plot
        if save_png:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

            im = ax.imshow(
                H.T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect='auto',
                origin='lower',
                cmap='viridis',
                interpolation='bilinear'
            )

            cbar = plt.colorbar(im, ax=ax)
            cbar_label = 'log10(Charge) [a.u.]' if log_scale else 'Charge [a.u.]'
            cbar.set_label(cbar_label, rotation=270, labelpad=20)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

            plt.tight_layout()
            png_path = self.output_dir / f'{output_name}.png'
            plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved PNG: {png_path}")

        # Interactive HTML plot
        if save_html:
            x_centers = 0.5 * (xedges[1:] + xedges[:-1])
            y_centers = 0.5 * (yedges[1:] + yedges[:-1])

            fig = go.Figure(data=go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=H.T,
                colorscale='Viridis',
                colorbar=dict(
                    title='log10(Charge)' if log_scale else 'Charge'
                )
            ))

            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                height=600,
                width=900
            )

            html_path = self.output_dir / f'{output_name}.html'
            fig.write_html(str(html_path))
            print(f"Saved HTML: {html_path}")

    def plot_energy_spectrum(
        self,
        energy: np.ndarray,
        spectrum: np.ndarray,
        output_name: str = 'energy_spectrum',
        title: str = 'Electron Energy Spectrum',
        save_png: bool = True,
        save_html: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        log_scale: bool = False
    ) -> None:
        """
        Plot energy spectrum.

        Args:
            energy: Energy bins (MeV)
            spectrum: Spectrum values
            output_name: Base name for output files
            title: Plot title
            save_png: Save static PNG
            save_html: Save interactive HTML
            figsize: Figure size for PNG
            dpi: DPI for PNG
            log_scale: Use log scale for y-axis
        """

        # Static PNG plot
        if save_png:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

            ax.plot(energy, spectrum, 'b-', linewidth=2)
            ax.fill_between(energy, spectrum, alpha=0.3)

            ax.set_xlabel('Energy [MeV]')
            ax.set_ylabel('dQ/dE [a.u.]')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            if log_scale:
                ax.set_yscale('log')

            plt.tight_layout()
            png_path = self.output_dir / f'{output_name}.png'
            plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved PNG: {png_path}")

        # Interactive HTML plot
        if save_html:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=energy,
                y=spectrum,
                mode='lines',
                fill='tozeroy',
                line=dict(color='blue', width=2),
                name='Spectrum'
            ))

            fig.update_layout(
                title=title,
                xaxis_title='Energy [MeV]',
                yaxis_title='dQ/dE [a.u.]',
                height=600,
                width=900,
                hovermode='x unified'
            )

            if log_scale:
                fig.update_yaxes(type='log')

            html_path = self.output_dir / f'{output_name}.html'
            fig.write_html(str(html_path))
            print(f"Saved HTML: {html_path}")

    def plot_particle_distribution(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color_by: Optional[np.ndarray] = None,
        output_name: str = 'particle_distribution',
        title: str = 'Particle Distribution',
        color_label: str = 'Energy [MeV]',
        save_png: bool = True,
        save_html: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        alpha: float = 0.5,
        marker_size: float = 1.0
    ) -> None:
        """
        Plot particle spatial distribution.

        Args:
            x: X positions (m)
            y: Y positions (m)
            color_by: Optional array to color particles by (e.g., energy)
            output_name: Base name for output files
            title: Plot title
            color_label: Label for color bar
            save_png: Save static PNG
            save_html: Save interactive HTML
            figsize: Figure size for PNG
            dpi: DPI for PNG
            alpha: Transparency of markers
            marker_size: Size of markers
        """

        x_um = x * 1e6
        y_um = y * 1e6

        # Static PNG plot
        if save_png:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

            if color_by is not None:
                scatter = ax.scatter(
                    x_um, y_um, c=color_by, cmap='viridis',
                    s=marker_size, alpha=alpha
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_label, rotation=270, labelpad=20)
            else:
                ax.scatter(x_um, y_um, s=marker_size, alpha=alpha, c='blue')

            ax.set_xlabel('x [μm]')
            ax.set_ylabel('y [μm]')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            png_path = self.output_dir / f'{output_name}.png'
            plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved PNG: {png_path}")

        # Interactive HTML plot
        if save_html:
            fig = go.Figure()

            if color_by is not None:
                fig.add_trace(go.Scatter(
                    x=x_um,
                    y=y_um,
                    mode='markers',
                    marker=dict(
                        size=marker_size * 2,
                        color=color_by,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=color_label),
                        opacity=alpha
                    ),
                    name='Particles'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=x_um,
                    y=y_um,
                    mode='markers',
                    marker=dict(size=marker_size * 2, opacity=alpha),
                    name='Particles'
                ))

            fig.update_layout(
                title=title,
                xaxis_title='x [μm]',
                yaxis_title='y [μm]',
                height=600,
                width=900
            )

            html_path = self.output_dir / f'{output_name}.html'
            fig.write_html(str(html_path))
            print(f"Saved HTML: {html_path}")
