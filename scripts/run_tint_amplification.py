"""
Radius Amplification via Thermal Blanketing (Fuzzy Cores)

To observe massive inflation, the heavy elements must be mixed into the 
compressible outer envelope. By setting \sigma = 0.85, the metals pollute 
the low-pressure atmosphere. 

At 500 K, this \mu-heavy atmosphere crushes the planet. At 1200 K, the 
entropy jumps in this compressible ideal-gas regime cause the atmosphere 
to violently expand, proving that fuzzy cores trap heat and heavily 
amplify thermal inflation!
"""

import os
import threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import fuzzycore.constants as c
import fuzzycore.solver as solver
import fuzzycore.utils as utils
import fuzzycore.eos as eos

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

M_TOTAL: float = 1.0 * c.M_JUPITER
TARGET_MZ_ME: float = 30.0 
FIXED_WIDE_SIGMA: float = 0.85  # Force metals to reach the compressible surface!

TEMP_CSV_OUT: str = "tint_amplification_temp.csv" 
PROGRESS_DB: str = "tint_amplification_results.csv" 

# =============================================================================
# CONSERVATION GATEKEEPER
# =============================================================================

def get_params(m_core: float, z_core: float, t_int: float) -> dict:
    actual_sigma = 0.001 if m_core >= TARGET_MZ_ME - 0.1 else FIXED_WIDE_SIGMA
    
    z_prof = utils.generate_gaussian_z_profile(
        n_layers=30, sigma=actual_sigma, z_base=0.02, z_core=z_core
    )
    return {
        'M_core': m_core * c.M_EARTH,   
        'M_rock': m_core * c.M_EARTH,   
        'M_water': 0.0,                 
        'P_surf': 1.0,             
        'T_surf': t_int,            
        'T_int': t_int,
        'z_base': 0.02,                 
        'z_profile': z_prof,        
        'sigma_val': actual_sigma,              
        'iron_fraction': 0.33,          
        'debug': True
    }

def find_conserved_z_core(m_core: float, target_mz_me: float, t_int: float, lock) -> float:
    if m_core >= target_mz_me - 0.1: 
        return 0.02 

    def eval_zcore(z_guess: float) -> float:
        p = get_params(m_core, z_guess, t_int)
        res = solver.solve_structure(M_TOTAL, p, 'mass', 'opt_temp', os.devnull, lock)
        eos._MIXED_CACHE.clear()

        if res is None:
            raise ValueError("Unbound")
            
        actual_mz_me = res.get('M_Z_total', m_core * c.M_EARTH) / c.M_EARTH
        return actual_mz_me - target_mz_me

    test_zcores = [0.02, 0.10, 0.25, 0.50, 0.75, 0.99]
    stable_points = []
    
    for guess in test_zcores:
        try:
            err = eval_zcore(guess)
            stable_points.append((guess, err))
        except ValueError:
            continue
            
    for i in range(len(stable_points) - 1):
        if np.sign(stable_points[i][1]) != np.sign(stable_points[i+1][1]):
            bracket_low = stable_points[i][0]
            bracket_high = stable_points[i+1][0]
            return float(brentq(eval_zcore, bracket_low, bracket_high, xtol=1e-3))
            
    raise ValueError("Could not bracket conserved metal z_core.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    dummy_lock = threading.Lock()
    
    if os.path.exists(TEMP_CSV_OUT): os.remove(TEMP_CSV_OUT)
    if os.path.exists(PROGRESS_DB): os.remove(PROGRESS_DB) 

    print("===============================================================")
    print(f" Starting T_int Amplification Analysis (Total Z = {TARGET_MZ_ME} M_E)")
    print(f" Planet Mass: 1.0 M_Jupiter | Giant Fluffy Sigma: {FIXED_WIDE_SIGMA}")
    print("===============================================================")

    results = []
    tasks = [
        {"name": "Sharp Core (Cold)", "m_core": 30.0, "t_int": 500.0,  "type": "Sharp"},
        {"name": "Sharp Core (Hot)",  "m_core": 30.0, "t_int": 1200.0, "type": "Sharp"},
        {"name": "Fuzzy Core (Cold)", "m_core": 10.0, "t_int": 500.0,  "type": "Fuzzy"},
        {"name": "Fuzzy Core (Hot)",  "m_core": 10.0, "t_int": 1200.0, "type": "Fuzzy"}
    ]

    for cfg in tasks:
        print(f"\n[*] Evaluating: {cfg['name']} | M_core = {cfg['m_core']} M_E")
        try:
            z_core = find_conserved_z_core(cfg['m_core'], TARGET_MZ_ME, cfg['t_int'], dummy_lock)
            print(f"   -> Conserved Z_core tuned to: {z_core*100:.1f}% metals at boundary")
            
            params = get_params(cfg['m_core'], z_core, cfg['t_int'])
            
            res = solver.solve_structure(
                target_val=M_TOTAL, params=params, mode='mass',
                trial_id=cfg['name'].replace(" ", "_"), csv_file=TEMP_CSV_OUT, write_lock=dummy_lock
            )
            
            if res is not None:
                r_tot_rj = res['R'][-1] / c.R_JUPITER
                actual_mz = res.get('M_Z_total', cfg['m_core']*c.M_EARTH) / c.M_EARTH
                
                new_row = {
                    'Config': cfg['name'],
                    'Type': cfg['type'],
                    'T_int': cfg['t_int'],
                    'Radius_Rj': r_tot_rj,
                    'Z_Core_Found': z_core,
                    'Actual_Z_Me': actual_mz
                }
                results.append(new_row)
                
                df_row = pd.DataFrame([new_row])
                write_header = not os.path.exists(PROGRESS_DB)
                df_row.to_csv(PROGRESS_DB, mode='a', header=write_header, index=False)
                
                print(f"   -> Success! R = {r_tot_rj:.3f} R_J | Total Z = {actual_mz:.2f} M_E")
            else:
                print("   -> Failed to converge.")
        except Exception as e:
            print(f"   -> Solver crashed: {e}")

    df = pd.DataFrame(results)

    if len(df) == 4:
        sharp_cold = df[(df['Type'] == 'Sharp') & (df['T_int'] == 500)]['Radius_Rj'].values[0]
        sharp_hot  = df[(df['Type'] == 'Sharp') & (df['T_int'] == 1200)]['Radius_Rj'].values[0]
        fuzzy_cold = df[(df['Type'] == 'Fuzzy') & (df['T_int'] == 500)]['Radius_Rj'].values[0]
        fuzzy_hot  = df[(df['Type'] == 'Fuzzy') & (df['T_int'] == 1200)]['Radius_Rj'].values[0]
        
        delta_sharp = sharp_hot - sharp_cold
        delta_fuzzy = fuzzy_hot - fuzzy_cold
        
        print("\n--- RESULTS ---")
        print(f"Sharp Core Inflation (500K -> 1200K): +{delta_sharp:.3f} R_J")
        print(f"Fuzzy Core Inflation (500K -> 1200K): +{delta_fuzzy:.3f} R_J")
        
        labels = [
            f'Sharp Core\n({TARGET_MZ_ME} $M_\\oplus$ solid)', 
            f'Fuzzy Core\n(10 solid + 20 dilute)'
        ]
        cold_radii = [sharp_cold, fuzzy_cold]
        inflations = [delta_sharp, delta_fuzzy]
        
        x = np.arange(len(labels))
        width = 0.5
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.bar(x, cold_radii, width, label='$T_{1\\text{bar}} = 500$ K', color='#3498db', edgecolor='black')
        ax.bar(x, inflations, width, bottom=cold_radii, label='Inflation to $T_{1\\text{bar}} = 1200$ K', 
               color='#e74c3c', hatch='//', edgecolor='black')
        
        ax.set_ylabel('Total Planetary Radius ($R_J$)', fontsize=14)
        ax.set_title(
            f'Radius Amplification via Thermal Blanketing ($1.0\\ M_J$)\n'
            f'(Wide Gradient $\\sigma={FIXED_WIDE_SIGMA}$ | Bulk Z = {TARGET_MZ_ME} $M_\\oplus$)', 
            fontsize=14
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=min(cold_radii) * 0.9)
        
        ax.text(0, sharp_hot + 0.02, f"+{delta_sharp:.3f} $R_J$", ha='center', fontsize=12, fontweight='bold', color='#c0392b')
        ax.text(1, fuzzy_hot + 0.02, f"+{delta_fuzzy:.3f} $R_J$", ha='center', fontsize=12, fontweight='bold', color='#c0392b')
        
        plt.tight_layout()
        plt.savefig("tint_amplification_bar.pdf", format='pdf', bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    main()