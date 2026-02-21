import matplotlib.pyplot as plt
import numpy as np
import os
from . import constants as c
from . import eos

def save_plot(fig, name):
    """Helper to ensure the figures directory exists and save the plot."""
    folder = "../figures/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, f"{name}.pdf")
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {path}")

def plot_diagnostics(results, save_name="structure_diagnostics"):
    """
    Enhanced 6-panel diagnostic suite supporting Rock-Water-Gas architectures.
    Radial plots now shade both the rock core and water mantle zones.
    """
    if not results: 
        print("No results to plot.")
        return

    # 1. Extract Dimensions and Normalization
    R_total = results['R'][-1]
    R_norm = results['R'] / R_total
    M, P = results['M'], results['P']
    T, Z, Rho = results['T'], results['Z'], results['Rho']
    S = results['S'] 
    
    # 2. Zone Detection
    # R_int is the water-envelope boundary; R_rock is the rock-water boundary
    R_int = results.get('R_int')
    R_rock = results.get('R_rock')
    
    R_int_norm = R_int / R_total
    R_rock_norm = (R_rock / R_total) if R_rock is not None else None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    def shade(ax):
        """Dynamic shading based on detected planetary zones."""
        # Rock Zone (0 to R_rock or 0 to R_int for gas giants)
        rock_boundary = R_rock_norm if R_rock_norm is not None else R_int_norm
        ax.axvspan(0, rock_boundary, color='saddlebrown', alpha=0.3)
        ax.text(rock_boundary/2, ax.get_ylim()[0], "ROCK", 
                color='saddlebrown', ha='center', fontweight='bold', fontsize=9)
        
        # Water Zone (Only if a mantle exists)
        if R_rock_norm is not None:
            ax.axvspan(R_rock_norm, R_int_norm, color='dodgerblue', alpha=0.2)
            ax.text((R_rock_norm + R_int_norm)/2, ax.get_ylim()[0], "WATER", 
                    color='dodgerblue', ha='center', fontweight='bold', fontsize=9)

    # Panel 0: Density (with rock/water plateaus)
    axes[0].plot(R_norm, Rho/1000, 'k-', lw=2)
    axes[0].set_title("Density [g/cm3]")
    shade(axes[0])

    # Panel 1: Mass Distribution
    axes[1].plot(R_norm, M/c.M_EARTH, 'b-', lw=2)
    axes[1].set_title(f"Total Mass: {M[-1]/c.M_EARTH:.2f} M_E")
    shade(axes[1])

    # Panel 2: Temperature (Log Scale)
    axes[2].plot(R_norm, T, 'orange', lw=2)
    axes[2].set_yscale('log')
    axes[2].set_title("Temperature [K]")
    shade(axes[2])

    # Panel 3: Compositional Profile
    axes[3].plot(R_norm, Z, 'c-', lw=2)
    axes[3].set_title("Water Mass Fraction Z")
    shade(axes[3])
    
    # Panel 4: P-T Profile (Internal Adiabat)
    axes[4].plot(P, np.log10(T), 'purple', lw=2)
    axes[4].invert_xaxis()
    axes[4].set_title("Internal P-T Profile")
    axes[4].set_xlabel("log10 P [bar]")
    
    # Panel 5: Entropy (Layered Jumps)
    axes[5].plot(R_norm, S, 'm-', lw=2)
    axes[5].set_title("Entropy Profile")
    shade(axes[5])
    
    for ax in [axes[0], axes[1], axes[2], axes[3], axes[5]]:
        ax.set_xlabel(r"Relative Radius ($r/R_{total}$)")
    
    plt.tight_layout()
    save_plot(fig, save_name)
    plt.show()


def plot_trajectory_on_eos(results, params, max_plots=9, save_name="eos_trajectory"):
    """
    Overlays the planet's trajectory on the compositional phase space.
    Plots the log10 of Entropy on the color axis and filters out 
    extreme grid-squaring artifacts.
    """
    if not results: return

    path_lp = results['P']
    path_lt = np.log10(results['T'])
    path_z  = results['Z']
    
    fluid_stack = eos.generate_fluid_interpolators(params['z_profile'])
    available_z = sorted(fluid_stack.keys())
    
    if len(available_z) > max_plots:
        indices = np.linspace(0, len(available_z)-1, max_plots, dtype=int)
        display_z = [available_z[i] for i in indices]
    else:
        display_z = available_z

    cols = 3
    rows = (len(display_z) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    # --- NEW: LOG-ENTROPY PERCENTILE FILTERING ---
    # We calculate the log10 of the entropy values and filter extremums
    # to avoid unphysical outliers used for grid squaring.
    all_log_S = [np.log10(np.abs(d['S_values']) + 1e-12) for d in fluid_stack.values()]
    flat_log_S = np.concatenate(all_log_S)
    
    # Tight percentile to ensure the colorbar focuses on the physical manifold
    vmin, vmax = np.percentile(flat_log_S, [0, 90]) 
    # ----------------------------------------------

    for i, z_val in enumerate(display_z):
        ax = axes[i]
        data = fluid_stack[z_val]
        
        # Apply the log transformation locally
        log_S_current = np.log10(np.abs(data['S_values']) + 1e-12)
        
        # Local mask to remove grid-squaring extremums from the scatter
        mask = (log_S_current >= vmin) & (log_S_current <= vmax)
        
        sc = ax.scatter(data['points'][mask, 0], data['points'][mask, 1], 
                        c=log_S_current[mask], 
                        cmap='RdYlBu_r', s=10, vmin=vmin, vmax=vmax, alpha=0.3, 
                        rasterized=True, edgecolors='none')

        # Ghost Path (Full Planet Trajectory)
        ax.plot(path_lp, path_lt, 'k--', lw=1, alpha=0.8)
        
        dist_current = np.abs(path_z - z_val)
        is_active = np.ones_like(dist_current, dtype=bool)
        for other_z in available_z:
            if other_z == z_val: continue
            is_active[np.abs(path_z - other_z) < dist_current] = False
            
        if np.any(is_active):
            # Active segment for this composition
            ax.plot(path_lp[is_active], path_lt[is_active], 'r-', lw=3)
            ax.scatter(path_lp[is_active][0], path_lt[is_active][0], 
                       c='white', edgecolors='k', s=50, zorder=10)

        ax.set_title(f"Z = {z_val:.4f}")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label=r'$\log_{10}(\text{Entropy})$')
    
    fig.text(0.5, 0.01, r'$\log_{10}(\text{Pressure [Pa]})$', ha='center')
    fig.text(0.01, 0.5, r'$\log_{10}(\text{Temperature [K]})$', va='center', rotation='vertical')
    #plt.suptitle("Planet Trajectory: Log-Entropy Phase Space", y=1.02, fontsize=16)
    
    save_plot(fig, save_name)
    plt.show()