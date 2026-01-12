"""
Visualization module for EPOCH quasi-3D LWFA simulations.
Creates 3-panel animated visualizations matching the working script style.
"""

import matplotlib as mpl
# Use Agg backend for HPC compatibility (must be before pyplot import)
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np

# Configure matplotlib for publication-quality figures
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams["mathtext.fontset"] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['animation.embed_limit'] = 2**31 - 1  # Large animation limit

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def create_animation(data, params, config):
    """
    Create 3-panel animation figure.

    Panel 1: 2D density and field (pcolormesh)
    Panel 2: Longitudinal phase space (scatter)
    Panel 3: Momentum distribution (line plot)

    Args:
        data: Dict containing:
            - E_x: List of E_x arrays (one per frame)
            - E_tot: List of E_tot arrays
            - n_e: List of density arrays
            - x, r: List of grid arrays
            - particles: List of particle dicts with 'x_he', 'px_he'
            - px_dist: List of momentum histograms
            - px_centers: Momentum bin centers
        params: Physical parameters dict (for dumpstep)
        config: Configuration dict with visualization parameters

    Returns:
        (fig, anim) tuple
    """
    n_frames = len(data['x'])
    dumpstep = params.get('dumpstep', 1.0)
    a0 = config.get('a0', 3.0)

    # Get visualization parameters
    density_vmin = config.get('density_vmin', 1e-4)
    density_vmax = config.get('density_vmax', 1.0)
    field_vmax = config.get('field_vmax', 5.0)
    r_max = config.get('r_max_lambda0', 130)
    px_ylim = config.get('px_ylim', (1, 10000))

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(3.54, 2*3.54), dpi=200,
        gridspec_kw={'height_ratios': [1.75, 1, 1]}
    )

    # Initial data
    x0, r0 = data['x'][0], data['r'][0]

    # Panel 1: 2D density and field
    plot_dens = ax1.pcolormesh(
        x0, r0, data['n_e'][0],
        cmap='Greys',
        norm=colors.LogNorm(vmin=density_vmin, vmax=density_vmax),
        shading='auto'
    )
    plot_field = ax1.pcolormesh(
        x0, r0, data['E_tot'][0],
        cmap='PuRd', vmin=0, vmax=field_vmax,
        alpha=0.2, shading='auto'
    )
    ax1.set_ylabel(r"$r\: /\: \lambda_{0}$")
    ax1.set_xlabel(r"$x\: /\: \lambda_{0}$")
    ax1.set_ylim(r0[0], r_max)
    ax1.set_title(
        f"$a_{{0}} = {a0:.1f},\\ \\omega_{{pe}} t = {0:.0f}$",
        loc='left', fontsize=12
    )

    # Panel 2: Phase space scatter
    pdata = data['particles'][0]
    scatter_he = ax2.scatter(
        pdata['x_he'], pdata['px_he'],
        s=2, c='r', alpha=0.5, marker='.', linewidths=0
    )
    ax2.set_ylabel(r"$p_x\: /\: m_{e}c$")
    ax2.set_xlabel(r"$x\: /\: \lambda_{0}$")
    ax2.set_ylim(px_ylim)
    ax2.set_xlim(x0[0], x0[-1])

    legend_handles = [Line2D([0], [0], color='red', lw=1, label='Plasma e-')]
    ax2.legend(handles=legend_handles, frameon=True, framealpha=0, loc="upper left")

    # Panel 3: Momentum distribution
    px_centers = data['px_centers']
    line_dist, = ax3.plot(px_centers, data['px_dist'][0], color='red', label='Plasma e-')
    ax3.set_xlabel(r"$p_x\: /\: m_e c$")
    ax3.set_ylabel(r"$dN\:/\:dp_x$")
    ax3.set_xlim(px_centers[0], px_centers[-1])

    # Auto-scale y-axis based on max histogram value
    valid_hists = [np.max(h) for h in data['px_dist'] if len(h) > 0 and np.max(h) > 0]
    max_hist = max(valid_hists) if valid_hists else 1.0
    ax3.set_ylim(0, max_hist * 0.5)

    ax3.legend(frameon=True, framealpha=0, loc="upper right")

    fig.tight_layout()

    # Store plot objects that need to be recreated each frame
    plot_objects = {'dens': plot_dens, 'field': plot_field}

    def animate(frame):
        x = data['x'][frame]
        r = data['r'][frame]

        # Update Panel 1: density and field
        # Remove old pcolormesh and create new ones (needed for moving window)
        plot_objects['dens'].remove()
        plot_objects['field'].remove()

        plot_objects['dens'] = ax1.pcolormesh(
            x, r, data['n_e'][frame],
            cmap='Greys',
            norm=colors.LogNorm(vmin=density_vmin, vmax=density_vmax),
            shading='auto'
        )
        plot_objects['field'] = ax1.pcolormesh(
            x, r, data['E_tot'][frame],
            cmap='PuRd', vmin=0, vmax=field_vmax,
            alpha=0.2, shading='auto'
        )

        ax1.set_xlim(x[0], x[-1])
        ax1.set_title(
            f"$a_{{0}} = {a0:.1f},\\ \\omega_{{pe}} t = {frame * dumpstep:.0f}$",
            loc='left', fontsize=12
        )

        # Update Panel 2: phase space
        pdata = data['particles'][frame]
        if pdata['x_he'].size and pdata['px_he'].size:
            he_data = np.column_stack((pdata['x_he'], pdata['px_he']))
            scatter_he.set_offsets(he_data)
        else:
            scatter_he.set_offsets(np.empty((0, 2)))
        ax2.set_xlim(x[0], x[-1])

        # Update Panel 3: momentum distribution
        line_dist.set_ydata(data['px_dist'][frame])

        return plot_objects['dens'], plot_objects['field'], scatter_he, line_dist

    # Create animation
    interval = config.get('animation_interval', 50)
    anim = FuncAnimation(
        fig, animate, frames=n_frames,
        interval=interval, blit=False, repeat=True
    )

    return fig, anim


def save_animation_html(anim, output_path):
    """
    Save animation as standalone HTML file.

    Args:
        anim: FuncAnimation object
        output_path: Full path for output HTML file
    """
    html_output = anim.to_jshtml()
    with open(output_path, 'w') as f:
        f.write(html_output)
    print(f"Saved animation: {output_path}")
