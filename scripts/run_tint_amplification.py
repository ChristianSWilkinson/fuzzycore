"""
Radius Amplification via Thermal Blanketing (Fuzzy Cores)

This script demonstrates the non-linear "amplification" effect that dilute 
compositional gradients (fuzzy cores) have on planetary radii. It evaluates 
a theoretical 8.0 Earth-mass planet under four distinct configurations:
a 2x2 matrix of Sharp vs. Fuzzy cores, and Cold (500 K) vs. Hot (1200 K) 
internal boundary temperatures.



Because composition gradients suppress convection, they force a steeper 
radiative/advective temperature gradient. This script proves that a fuzzy 
core inflates a planet significantly more than a fully convective sharp 
core when subjected to the exact same internal heat.
"""

import os
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Clean package imports natively utilizing the installed fuzzycore package
import fuzzycore.constants as c
import fuzzycore.solver as solver


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Fixed total mass for our theoretical sub-Neptune
M_TOTAL: float = 8.0 * c.M_EARTH

# Configurations for the 2x2 Matrix: [Name, M_core, sigma, T_int, Type]
CONFIGS: list[dict] = [
    {
        "name": "Sharp Core (Cold)", 
        "m_core": 7.0, 
        "sigma": 0.01, 
        "t_int": 500.0, 
        "type": "Sharp"
    },
    {
        "name": "Sharp Core (Hot)",  
        "m_core": 7.0, 
        "sigma": 0.01, 
        "t_int": 1200.0, 
        "type": "Sharp"
    },
    {
        "name": "Fuzzy Core (Cold)", 
        "m_core": 2.0, 
        "sigma": 0.30, 
        "t_int": 500.0, 
        "type": "Fuzzy"
    },
    {
        "name": "Fuzzy Core (Hot)",  
        "m_core": 2.0, 
        "sigma": 0.30, 
        "t_int": 1200.0, 
        "type": "Fuzzy"
    }
]

CSV_OUT: str = "tint_amplification_temp.csv"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    results = []
    dummy_lock = threading.Lock()
    
    # Clean up any leftover temporary files from previous runs
    if os.path.exists(CSV_OUT):
        os.remove(CSV_OUT)

    print("===============================================================")
    print(" Starting T_int Amplification Analysis (10 bar boundary)")
    print("===============================================================")

    # 1. Run the Integrations
    for cfg in CONFIGS:
        print(f"\n[*] Evaluating: {cfg['name']} | T_int = {cfg['t_int']} K")
        
        # Build the structural recipe
        params = {
            'M_core': cfg['m_core'] * c.M_EARTH,   
            'M_rock': cfg['m_core'] * c.M_EARTH,   
            'M_water': 0.0,                 
            'P_surf': 10.0,             # Deep convective boundary proxy
            'T_surf': cfg['t_int'],     # This acts as our effective T_int
            'z_base': 0.05,                 
            'z_profile': np.linspace(0.05, 1.0, 20), 
            'sigma_val': cfg['sigma'],              
            'iron_fraction': 0.33,          
            'debug': False
        }
        
        trial_name = cfg['name'].replace(" ", "_")
        
        try:
            # Execute the boundary-shooting solver
            res = solver.solve_structure(
                target_val=M_TOTAL,
                params=params,
                mode='mass',
                trial_id=trial_name,
                csv_file=CSV_OUT,
                write_lock=dummy_lock
            )
            
            # Extract and store successful results
            if res is not None:
                r_tot_re = res['R'][-1] / c.R_EARTH
                results.append({
                    'Config': cfg['name'],
                    'Type': cfg['type'],
                    'T_int': cfg['t_int'],
                    'Radius_Re': r_tot_re
                })
                print(f"   -> Success! Radius = {r_tot_re:.2f} R_E")
            else:
                print("   -> Failed to converge on a physical solution.")
                
        except Exception as e:
            print(f"   -> Solver crashed: {e}")

    # Convert results to a pandas DataFrame for easy filtering
    df = pd.DataFrame(results)

    # =========================================================================
    # 2. Plotting the Amplification Effect
    # =========================================================================
    
    # Proceed to plot only if all 4 configurations successfully converged
    if len(df) == 4:
        # Extract specific radii
        sharp_cold = df[(df['Type'] == 'Sharp') & (df['T_int'] == 500)]['Radius_Re'].values[0]
        sharp_hot  = df[(df['Type'] == 'Sharp') & (df['T_int'] == 1200)]['Radius_Re'].values[0]
        
        fuzzy_cold = df[(df['Type'] == 'Fuzzy') & (df['T_int'] == 500)]['Radius_Re'].values[0]
        fuzzy_hot  = df[(df['Type'] == 'Fuzzy') & (df['T_int'] == 1200)]['Radius_Re'].values[0]
        
        # Calculate the absolute radial inflation (Delta R)
        delta_sharp = sharp_hot - sharp_cold
        delta_fuzzy = fuzzy_hot - fuzzy_cold
        
        print("\n--- RESULTS ---")
        print(f"Sharp Core Inflation (500K -> 1200K): +{delta_sharp:.2f} R_E")
        print(f"Fuzzy Core Inflation (500K -> 1200K): +{delta_fuzzy:.2f} R_E")
        
        # Set up the stacked bar chart
        labels = [
            'Sharp Core\n($7.0 M_\\oplus$ solid)', 
            'Fuzzy Core\n($2.0 M_\\oplus$ solid)'
        ]
        cold_radii = [sharp_cold, fuzzy_cold]
        inflations = [delta_sharp, delta_fuzzy]
        
        x = np.arange(len(labels))
        width = 0.5
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Base radius (Cold/Old state)
        ax.bar(
            x, cold_radii, width, 
            label='$T_{int} = 500$ K (Cold/Old)', 
            color='#3498db', 
            edgecolor='black'
        )
        
        # Additional radius from internal heating (Hot/Young state)
        ax.bar(
            x, inflations, width, bottom=cold_radii, 
            label='Inflation from $T_{int} = 1200$ K (Hot/Young)', 
            color='#e74c3c', 
            hatch='//', 
            edgecolor='black'
        )
        
        # Formatting
        ax.set_ylabel('Total Planetary Radius ($R_\\oplus$)', fontsize=14)
        ax.set_title(
            'Radius Amplification via Fuzzy Core Thermal Blanketing\n'
            '($P_{surf} = 10$ bar, $M_{tot} = 8.0 M_\\oplus$)', 
            fontsize=14
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.legend(fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Annotate the specific Delta R values directly on the bars
        ax.text(
            0, sharp_hot + 0.1, 
            f"+{delta_sharp:.2f} $R_\\oplus$", 
            ha='center', fontsize=12, fontweight='bold', color='#c0392b'
        )
        ax.text(
            1, fuzzy_hot + 0.1, 
            f"+{delta_fuzzy:.2f} $R_\\oplus$", 
            ha='center', fontsize=12, fontweight='bold', color='#c0392b'
        )
        
        plt.tight_layout()
        
        # Save Output
        plot_filename = "tint_amplification_bar.pdf"
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.show()
        
        print(f"\n[*] Plot successfully saved to '{plot_filename}'")
        
    else:
        print("\n[!] Could not generate plot: One or more configurations failed to converge.")


if __name__ == '__main__':
    main()