import os

import matplotlib.pyplot as plt
import numpy as np

from . import constants as c
from . import eos


def save_plot(fig, name):
    """
    Saves a matplotlib figure to the standardized figures directory.

    Ensures that the output directory exists before attempting to save.
    The figure is saved in PDF format with a tight bounding box.

    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        name (str): The base filename (without extension) for the saved plot.
    """
    folder = "../figures/"
    
    # Ensure the target directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    path = os.path.join(folder, f"{name}.pdf")
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {path}")


import os
import matplotlib.pyplot as plt
import numpy as np
from . import constants as c
from . import eos

def save_plot(fig, name):
    # Assumes your save_plot function remains exactly as is
    folder = "../figures/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, f"{name}.pdf")
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {path}")


def plot_diagnostics(results, save_name="structure_diagnostics"):
    """
    Generates a 6-panel diagnostic suite for Rock-Water-Gas architectures.
    """

    if not results:
        print("No results to plot.")
        return

    # ==========================
    # Extract dimensions
    # ==========================
    R_total = results['R'][-1]
    R_norm = results['R'] / R_total

    M = results['M']
    P = results['P']
    T = results['T']
    Z = results['Z']
    Rho = results['Rho']
    S = results['S']

    # ==========================
    # Zone detection
    # ==========================
    R_int = results.get('R_int')
    R_rock = results.get('R_rock')
    has_water = results.get('M_water', 0.0) > 0.0

    R_int_norm = (R_int / R_total) if R_int is not None else None
    R_rock_norm = (R_rock / R_total) if R_rock is not None else None

    # ==========================
    # Mask profiles inside core
    # ==========================
    Z_plot = Z.copy()
    S_plot = S.copy()

    if R_rock_norm is not None:
        core_mask = R_norm < R_rock_norm
        Z_plot[core_mask] = np.nan
        S_plot[core_mask] = np.nan

    # ==========================
    # Figure setup
    # ==========================
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    # ==========================
    # Scientific color palette
    # ==========================
    colors = {
        "density": "#264653",      # deep teal
        "mass": "#1d3557",         # navy
        "temperature": "#e76f51", # warm orange-red
        "composition": "#2a9d8f", # sea green
        "pt": "#7b2cbf",           # rich purple
        "entropy": "#c1121f"       # crimson-red
    }

    # ==========================
    # Styling
    # ==========================
    label_style = {'fontsize': 15, 'labelpad': 6}
    tick_size = 13
    title_size = 16

    def shade(ax):
        """
        Shade rock/water zones.
        """
        rock_boundary = (
            R_rock_norm if R_rock_norm is not None
            else R_int_norm
        )

        ymin, ymax = ax.get_ylim()

        # ROCK
        if rock_boundary is not None:
            ax.axvspan(
                0,
                rock_boundary,
                color='saddlebrown',
                alpha=0.22
            )

            ax.text(
                rock_boundary / 2,
                ymin + 0.05 * (ymax - ymin),
                "ROCK",
                color='saddlebrown',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

        # WATER
        if (
            has_water
            and R_rock_norm is not None
            and R_int_norm is not None
        ):
            ax.axvspan(
                R_rock_norm,
                R_int_norm,
                color='dodgerblue',
                alpha=0.15
            )

            ax.text(
                (R_rock_norm + R_int_norm) / 2,
                ymin + 0.05 * (ymax - ymin),
                "WATER",
                color='dodgerblue',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

    # ==========================
    # Panel 0: Density
    # ==========================
    axes[0].plot(
        R_norm,
        Rho / 1000.0,
        lw=2.8,
        color=colors["density"]
    )
    axes[0].set_ylabel(
        r"Density [g/cm$^3$]",
        **label_style
    )
    shade(axes[0])

    # ==========================
    # Panel 1: Mass
    # ==========================
    axes[1].plot(
        R_norm,
        M / c.M_EARTH,
        lw=2.8,
        color=colors["mass"]
    )

    axes[1].set_ylabel(
        r"Mass [$M_\oplus$]",
        **label_style
    )

    axes[1].set_title(
        f"Total Mass: {M[-1] / c.M_EARTH:.2f} $M_\\oplus$",
        fontsize=title_size
    )

    shade(axes[1])

    # ==========================
    # Panel 2: Temperature
    # ==========================
    axes[2].plot(
        R_norm,
        T,
        lw=2.8,
        color=colors["temperature"]
    )

    axes[2].set_yscale('log')
    axes[2].set_ylabel(
        "Temperature [K]",
        **label_style
    )

    shade(axes[2])

    # ==========================
    # Panel 3: Composition (Z)
    # ==========================
    axes[3].plot(
        R_norm,
        Z_plot,
        lw=2.8,
        color=colors["composition"]
    )

    axes[3].set_ylabel(
        "Heavy Element Fraction (Z)",
        **label_style
    )

    shade(axes[3])

    # ==========================
    # Panel 4: P-T profile
    # ==========================
    axes[4].plot(
        P,
        np.log10(T),
        lw=2.8,
        color=colors["pt"]
    )

    axes[4].invert_xaxis()

    axes[4].set_ylabel(
        r"$\log_{10} T$ [K]",
        **label_style
    )

    axes[4].set_xlabel(
        r"$\log_{10} P$ [bar]",
        **label_style
    )

    # ==========================
    # Panel 5: Entropy
    # ==========================
    axes[5].plot(
        R_norm,
        S_plot,
        lw=2.8,
        color=colors["entropy"]
    )

    axes[5].set_ylabel(
        "Entropy [J/kg/K]",
        **label_style
    )

    shade(axes[5])

    # ==========================
    # Common radial x-labels
    # ==========================
    for ax in [axes[0], axes[1], axes[2], axes[3], axes[5]]:
        ax.set_xlabel(
            r"Relative Radius ($r/R_{total}$)",
            **label_style
        )

    # Tick styling
    for ax in axes:
        ax.tick_params(
            axis='both',
            labelsize=tick_size
        )
        ax.grid(alpha=0.25, ls='--')

    plt.tight_layout()

    save_plot(fig, save_name)
    plt.show()


def plot_trajectory_on_eos(
    results, params, max_plots=9, save_name="eos_trajectory"
):
    """
    Overlay the planetary P-T trajectory on the compositional phase space.

    This function generates scatter plots of the local Equation of State (EOS)
    grid points, colored by the base-10 logarithm of entropy to filter out
    extreme grid-squaring artifacts. The actual trajectory is drawn as a ghost
    path, highlighting the active segment corresponding to the local
    composition (Z). The layout is constrained for publication-quality spacing.

    Args:
        results (dict): Integration results containing 'P', 'T', and 'Z'
            arrays representing the planetary trajectory.
        params (dict): Planetary parameters, specifically requiring the
            'z_profile' key to generate fluid interpolators.
        max_plots (int, optional): Maximum number of subplots to generate.
            Defaults to 9.
        save_name (str, optional): Filename for the output plot. Defaults to
            "eos_trajectory".

    Returns:
        None
    """
    if not results:
        return

    # Extract trajectory paths
    path_lp = results['P']
    path_lt = np.log10(results['T'])
    path_z = results['Z']

    # Generate local interpolators for the fluid stack
    # (Assumes 'eos' module is imported globally)
    fluid_stack = eos.generate_fluid_interpolators(params['z_profile'])
    available_z = sorted(fluid_stack.keys())

    # Subsample plots if there are too many unique Z fractions
    if len(available_z) > max_plots:
        indices = np.linspace(0, len(available_z) - 1, max_plots, dtype=int)
        display_z = [available_z[i] for i in indices]
    else:
        display_z = available_z

    # --- DYNAMIC GRID SETUP ---
    num_plots = len(display_z)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols

    # Tighter, more standard dimensions for papers (~3.5 inches per col/row)
    fig_width = 3.5 * cols
    fig_height = 3.5 * rows 
    
    # layout='constrained' is the modern, robust way to eliminate dead space
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        layout='constrained' 
    )

    # Ensure axes is always a 1D iterable (handles 1x1 grid properly)
    axes = np.atleast_1d(axes).flatten()

    # --- LOG-ENTROPY PERCENTILE FILTERING ---
    all_log_s = [
        np.log10(np.abs(d['S_values']) + 1e-12)
        for d in fluid_stack.values()
    ]
    flat_log_s = np.concatenate(all_log_s)

    # Tight percentile to ensure colorbar focuses on the physical manifold
    vmin, vmax = np.percentile(flat_log_s, [0, 90])

    for i, z_val in enumerate(display_z):
        ax = axes[i]
        data = fluid_stack[z_val]

        # Apply the log transformation locally
        log_s_current = np.log10(np.abs(data['S_values']) + 1e-12)

        # Local mask to remove grid-squaring extremums from the scatter
        mask = (log_s_current >= vmin) & (log_s_current <= vmax)

        # Plot the EOS background phase space
        sc = ax.scatter(
            data['points'][mask, 0],
            data['points'][mask, 1],
            c=log_s_current[mask],
            cmap='RdYlBu_r',
            s=8,           # Slightly smaller scatter points for a cleaner look
            vmin=vmin,
            vmax=vmax,
            alpha=0.3,
            rasterized=True, # Crucial to prevent massive PDF file sizes
            edgecolors='none'
        )

        # Draw the ghost path (Full Planet Trajectory)
        ax.plot(path_lp, path_lt, 'k--', lw=1, alpha=0.8)

        # Determine which segments of the trajectory match the current Z
        dist_current = np.abs(path_z - z_val)
        is_active = np.ones_like(dist_current, dtype=bool)

        for other_z in available_z:
            if other_z == z_val:
                continue
            # Deactivate if another available Z is closer
            is_active[np.abs(path_z - other_z) < dist_current] = False

        if np.any(is_active):
            # Plot the active segment for this specific composition
            ax.plot(path_lp[is_active], path_lt[is_active], 'r-', lw=2.5)

            # Highlight the starting point of the active segment
            ax.scatter(
                path_lp[is_active][0],
                path_lt[is_active][0],
                c='white',
                edgecolors='k',
                s=40,
                linewidths=1,
                zorder=10
            )

        # Cleaner title formatting for papers
        ax.set_title(rf"$Z = {z_val:.4f}$", fontsize=11)
        
        # Gridlines that guide the eye but don't overpower the data
        ax.grid(True, linestyle=':', alpha=0.6)

    # Clean up any unused axes in the grid
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # --- DYNAMIC HORIZONTAL COLORBAR AND LAYOUT ---
    active_axes = axes[:num_plots]

    # Constrained layout handles the padding automatically. 
    # 'shrink' makes it slightly narrower than the full figure width for elegance.
    cbar = fig.colorbar(
        sc,
        ax=active_axes.tolist(),
        orientation='horizontal',
        label=r'$\log_{10}(\text{Entropy})$',
        shrink=0.8,
        aspect=45
    )
    
    # Optional: ensure colorbar ticks don't look cluttered
    cbar.ax.tick_params(labelsize=10)

    # Apply global axis labels using sup-labels (hugs the axes tightly)
    fig.supxlabel(r'$\log_{10}(\text{Pressure [Pa]})$', fontsize=12)
    fig.supylabel(r'$\log_{10}(\text{Temperature [K]})$', fontsize=12)

    # (Assumes 'save_plot' is a globally defined helper function)
    # Using bbox_inches='tight' in your save function is highly recommended!
    save_plot(fig, save_name)
    plt.show()