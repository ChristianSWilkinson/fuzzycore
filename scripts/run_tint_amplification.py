import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import os
import sys

sys.path.append('./..')

# Import your framework modules
from src import solver
from src import constants as c

# Define fixed total mass for our theoretical planet
M_TOTAL = 8.0 * c.M_EARTH

# Configurations for the 2x2 Matrix
# [Name, M_core, sigma, T_int]
configs = [
    {"name": "Sharp Core (Cold)", "m_core": 7.0, "sigma": 0.01, "t_int": 500.0, "type": "Sharp"},
    {"name": "Sharp Core (Hot)",  "m_core": 7.0, "sigma": 0.01, "t_int": 1200.0, "type": "Sharp"},
    {"name": "Fuzzy Core (Cold)", "m_core": 2.0, "sigma": 0.30, "t_int": 500.0, "type": "Fuzzy"},
    {"name": "Fuzzy Core (Hot)",  "m_core": 2.0, "sigma": 0.30, "t_int": 1200.0, "type": "Fuzzy"}
]

results = []
dummy_lock = threading.Lock()
csv_out = "tint_amplification_temp.csv"

if os.path.exists(csv_out):
    os.remove(csv_out)

print("Starting T_int Amplification Analysis (10 bar boundary)...")

for cfg in configs:
    print(f"\nEvaluating: {cfg['name']} | T_int = {cfg['t_int']} K")
    
    params = {
        'M_core': cfg['m_core'] * c.M_EARTH,   
        'M_rock': cfg['m_core'] * c.M_EARTH,   
        'M_water': 0.0,                 
        'P_surf': 10.0,                         # Top of the convective envelope!
        'T_surf': cfg['t_int'],                 # This now acts as our T_int
        'z_base': 0.05,                 
        'z_profile': np.linspace(0.05, 1.0, 20), 
        'sigma_val': cfg['sigma'],              
        'iron_fraction': 0.33,          
        'debug': False
    }
    
    trial_name = cfg['name'].replace(" ", "_")
    
    try:
        res = solver.solve_structure(
            target_val=M_TOTAL,
            params=params,
            mode='mass',
            trial_id=trial_name,
            csv_file=csv_out,
            write_lock=dummy_lock
        )
        
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
            print("   -> Failed to converge.")
    except Exception as e:
        print(f"   -> Solver crashed: {e}")

df = pd.DataFrame(results)

# ---------------------------------------------------------
# Plotting the Amplification Effect
# ---------------------------------------------------------
if len(df) == 4:
    sharp_cold = df[(df['Type'] == 'Sharp') & (df['T_int'] == 500)]['Radius_Re'].values[0]
    sharp_hot  = df[(df['Type'] == 'Sharp') & (df['T_int'] == 1200)]['Radius_Re'].values[0]
    fuzzy_cold = df[(df['Type'] == 'Fuzzy') & (df['T_int'] == 500)]['Radius_Re'].values[0]
    fuzzy_hot  = df[(df['Type'] == 'Fuzzy') & (df['T_int'] == 1200)]['Radius_Re'].values[0]
    
    delta_sharp = sharp_hot - sharp_cold
    delta_fuzzy = fuzzy_hot - fuzzy_cold
    
    print("\n--- RESULTS ---")
    print(f"Sharp Core Inflation (500K -> 1200K): +{delta_sharp:.2f} R_E")
    print(f"Fuzzy Core Inflation (500K -> 1200K): +{delta_fuzzy:.2f} R_E")
    
    # Bar Chart
    labels = ['Sharp Core\n($7.0 M_\oplus$ solid)', 'Fuzzy Core\n($2.0 M_\oplus$ solid)']
    cold_radii = [sharp_cold, fuzzy_cold]
    inflations = [delta_sharp, delta_fuzzy]
    
    x = np.arange(len(labels))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Base radius (Cold)
    ax.bar(x, cold_radii, width, label='$T_{int} = 500$ K (Cold/Old)', color='#3498db', edgecolor='black')
    
    # Additional radius from heating (Hot)
    ax.bar(x, inflations, width, bottom=cold_radii, label='Inflation from $T_{int} = 1200$ K (Hot/Young)', color='#e74c3c', hatch='//', edgecolor='black')
    
    ax.set_ylabel('Total Planetary Radius ($R_\oplus$)', fontsize=14)
    ax.set_title('Radius Amplification via Fuzzy Core Thermal Blanketing\n($P_{surf} = 10$ bar, $M_{tot} = 8.0 M_\oplus$)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate the Delta R
    ax.text(0, sharp_hot + 0.1, f"+{delta_sharp:.2f} $R_\oplus$", ha='center', fontsize=12, fontweight='bold', color='#c0392b')
    ax.text(1, fuzzy_hot + 0.1, f"+{delta_fuzzy:.2f} $R_\oplus$", ha='center', fontsize=12, fontweight='bold', color='#c0392b')
    
    plt.tight_layout()
    plt.savefig("tint_amplification_bar.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    
    print("Plot saved to 'tint_amplification_bar.pdf'")
else:
    print("\n[!] Could not plot: One or more configurations failed to converge.")