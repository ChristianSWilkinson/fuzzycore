"""
Kepler-11e 1D Parameter Sweep: Distribution Analysis

This script performs a parallelized 1D parameter sweep for the highly inflated 
super-puff exoplanet Kepler-11e. It runs two distinct physical tracks to isolate 
the radius inflation mechanisms:

1. Bulk Envelope Transfer: Mass shifts from the solid core into the envelope using a 
   fixed compositional shape (sigma = 0.25), adding both Metals and H/He to the gas.
2. Direct Metal Transfer: Dynamically tunes the diluteness parameter (sigma) 
   to ensure the total heavy element mass is strictly conserved at 7.5 M_Earth. 
   Mass lost from the solid core is converted 100% into suspended metals.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Clean package imports natively utilizing the installed fuzzycore package
import fuzzycore.constants as c
import fuzzycore.solver as solver
import fuzzycore.utils as utils


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

M_KEPLER11E: float = 7.95 * c.M_EARTH
TARGET_MZ_TOTAL: float = 7.9 * c.M_EARTH  # The conserved total heavy element mass

RESULTS_FILE: str = "../data/kepler11e_sweep_results.csv"
TEMP_FILE: str = "../data/kepler11e_sweep_temp.csv"


# =============================================================================
# PARALLEL WORKER FUNCTIONS
# =============================================================================

def get_params(m_core: float, sigma: float) -> dict:
    """Helper to generate the standard parameter dictionary."""
    z_prof = utils.generate_gaussian_z_profile(
        n_layers=100, sigma=sigma, z_base=0.05, z_core=0.99
    )
    return {
        'M_core': m_core * c.M_EARTH,   
        'M_rock': m_core * c.M_EARTH,   
        'M_water': 0.0,                 
        'P_surf': 1.0,
        'T_surf': 900.0,                
        'z_base': 0.05,                 
        'z_profile': np.round(z_prof, 3), 
        'sigma_val': sigma,              
        'iron_fraction': 0.33,          
        'debug': False
    }


def find_conserved_sigma(m_core: float, lock) -> float:
    """
    Optimizes the sigma parameter so the total heavy element mass (Core + Dilute)
    is strictly conserved at TARGET_MZ_TOTAL.
    """
    # =========================================================================
    # THE PHYSICAL BYPASS:
    # m_core is in Earth masses. If we are at our max target mass (7.5 M_E), 
    # we bypass the optimizer entirely to prevent integration crashes and
    # strictly force the sigma=0.0 adiabatic (zero-dilute-mass) fallback.
    # =========================================================================
    if m_core >= 7.49: 
        return 0.0  

    def mz_error(sig: float) -> float:
        p = get_params(m_core, sig)
        # Suppress CSV writing during root-finding using os.devnull
        res = solver.solve_structure(
            M_KEPLER11E, p, 'mass', 'opt_temp', os.devnull, lock
        )
        if res is None:
            return 1e6  # Heavy penalty for unphysical regions
            
        actual_mz = res.get('M_Z_total', m_core * c.M_EARTH)
        return abs(actual_mz - TARGET_MZ_TOTAL)

    # Search for the optimal sigma, allowing it to get extremely sharp
    res = minimize_scalar(
        mz_error, bounds=(1e-4, 0.90), method='bounded', options={'xatol': 0.005}
    )
    return float(res.x)


def run_single_model(args: tuple, lock) -> tuple:
    """
    Executes the integration for a specific core mass and transfer mode, 
    enforcing strict conservation checks for the Direct Metal track.
    """
    m_core, mode = args
    trial_id = f"K11e_{mode.replace(' ', '')}_{m_core:.3f}"
    
    try:
        # Determine the correct sigma based on the physical mode
        if mode == 'Bulk Envelope Transfer':
            # Original physics: Fixed sigma shape
            sigma = 0.25
        else:
            # Direct Metal Transfer: Tune sigma to conserve exactly 7.5 M_E of metals
            sigma = find_conserved_sigma(m_core, lock)
            
        params = get_params(m_core, sigma)
        
        # Final integration with the determined parameters
        res = solver.solve_structure(
            target_val=M_KEPLER11E,
            params=params,
            mode='mass',
            trial_id=trial_id,
            csv_file=TEMP_FILE,
            write_lock=lock
        )
        
        if res is not None:
            r_tot_re = res['R'][-1] / c.R_EARTH
            m_z_tot_kg = res.get('M_Z_total', m_core * c.M_EARTH)
            m_z_tot_me = m_z_tot_kg / c.M_EARTH
            
            # =================================================================
            # STRICT GATEKEEPER:
            # If we are in Direct Metal mode, the total heavy elements MUST be
            # extremely close to 7.5 Earth masses. If the optimizer failed and 
            # gave us a bad sigma, we discard the result so it doesn't pollute 
            # our plot! (Tolerance set to 0.05 M_earth)
            # =================================================================
            if mode == 'Direct Metal Transfer':
                target_me = TARGET_MZ_TOTAL / c.M_EARTH
                if abs(m_z_tot_me - target_me) > 0.05:
                    error_msg = f"Discarded: Conservation failed (M_z = {m_z_tot_me:.2f} M_E)"
                    return False, m_core, mode, error_msg

            # Dilute mass is the total heavy elements minus the solid core
            m_dilute = max(m_z_tot_me - m_core, 0.0)
            
            result_dict = {
                'M_core_Me': m_core,
                'Transfer_Mode': mode,
                'M_dilute_Me': m_dilute,
                'M_Z_total_Me': m_z_tot_me,
                'R_total_Re': r_tot_re,
                'Sigma_Used': sigma
            }
            return True, m_core, mode, result_dict
        else:
            return False, m_core, mode, "Solver failed to converge."
            
    except Exception as e:
        return False, m_core, mode, str(e)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    print("================================================================")
    print(" Kepler-11e 1D Sweep: Bulk vs Direct Metal Transfer ")
    print("================================================================")
    
    modes = ['Bulk Envelope Transfer', 'Direct Metal Transfer']
    
    # 1. Bulk Transfer Grid: Core can grow up to ~7.90 Me
    core_masses_bulk = np.concatenate([
        np.linspace(7.90, 7.5, 8),   # Very dense near zero dilute mass
        np.linspace(7.0, 2.0, 10)    # Standard spacing elsewhere
    ])
    
    # 2. Direct Metal Grid: Core strictly capped at 7.50 Me
    core_masses_direct = np.concatenate([
        np.linspace(7.90, 7.0, 8),   # Very dense near zero dilute mass
        np.linspace(6.5, 2.0, 10)    # Standard spacing elsewhere
    ])
    
    # Checkpoint Logic
    completed_tasks = set()
    # Ensure data directory exists
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    if os.path.exists(RESULTS_FILE):
        try:
            df_existing = pd.read_csv(RESULTS_FILE)
            for _, row in df_existing.iterrows():
                completed_tasks.add((round(row['M_core_Me'], 3), row['Transfer_Mode']))
            print(f"[*] Found {len(completed_tasks)} completed runs.")
        except Exception as e:
            print(f"[*] Could not read {RESULTS_FILE}: {e}")
            
    # Assign the correct grid to the correct mode!
    tasks_to_run = []
    for mode in modes:
        mass_grid = core_masses_bulk if mode == 'Bulk Envelope Transfer' else core_masses_direct
        for m in mass_grid:
            if (round(m, 3), mode) not in completed_tasks:
                tasks_to_run.append((m, mode))
            
    print(f"[*] Tasks remaining to compute: {len(tasks_to_run)}")
    
    # Parallel Execution
    if tasks_to_run:
        n_workers = min(mp.cpu_count() - 1, 4)
        n_workers = max(1, n_workers) 
        n_workers = 1
        
        print(f"[*] Starting parallel execution with {n_workers} workers...")
        manager = mp.Manager()
        shared_lock = manager.Lock()
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_single_model, task, shared_lock): task 
                for task in tasks_to_run
            }
            
            for future in as_completed(futures):
                success, m_core, mode, data = future.result()
                
                if success:
                    print(f"  [+] {mode} | Core: {m_core:.2f} -> R = {data['R_total_Re']:.2f} R_E")
                    df_new = pd.DataFrame([data])
                    file_exists = os.path.exists(RESULTS_FILE)
                    df_new.to_csv(RESULTS_FILE, mode='a', header=not file_exists, index=False)
                else:
                    print(f"  [-] Failed: {mode} | Core: {m_core:.2f} | Err: {data}")
                    
        print("[*] All parallel tasks finished.")

    # =========================================================================
    # GENERATE OUTPUT PLOT
    # =========================================================================
    print("[*] Generating comparative plot...")
    if os.path.exists(RESULTS_FILE):
        df_plot = pd.read_csv(RESULTS_FILE)
        
        plt.figure(figsize=(9, 6))
        
        colors = {'Bulk Envelope Transfer': '#3498db', 'Direct Metal Transfer': '#e74c3c'}
        markers = {'Bulk Envelope Transfer': 's', 'Direct Metal Transfer': 'o'}

        for mode in modes:
            df_mode = df_plot[df_plot['Transfer_Mode'] == mode].sort_values(by='M_core_Me', ascending=False)
            if len(df_mode) > 1:
                plt.plot(
                    df_mode['M_dilute_Me'], df_mode['R_total_Re'], 
                    marker=markers[mode], markersize=7, linestyle='-', 
                    color=colors[mode], linewidth=2.5, label=mode
                )

        plt.axhline(y=4.67, color='black', linestyle='--', alpha=0.6, label='Kepler-11e Observed Radius')

        plt.xlabel(r"Dilute Metal Mass in Envelope ($M_\oplus$)", fontsize=14)
        plt.ylabel(r"Total Planetary Radius ($R_\oplus$)", fontsize=14)
        plt.title(
            r"Radius Inflation Mechanisms for Kepler-11e ($M_{tot} = 7.95 \ M_\oplus$)", 
            fontsize=14
        )
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12, loc='lower right')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        # Ensure figures directory exists
        plot_filename = "../figures/kepler11e_transfer_comparison.pdf"
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        print(f"[*] Plot successfully saved as '{plot_filename}'.")
    else:
        print("[!] No results file found to plot.")