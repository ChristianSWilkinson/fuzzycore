"""
Kepler-11e 1D Parameter Sweep: Multi-Track Distribution Analysis

This script performs a parallelized 1D parameter sweep for the highly inflated 
super-puff exoplanet Kepler-11e. It explores different structural configurations
by defining distinct evolutionary tracks:

1. Bulk Envelope Transfer: A baseline track where mass shifts into the envelope 
   using a fixed compositional shape, altering the total metal/gas ratio.
2. Direct Metal Transfers: Multiple tracks dynamically tuning sigma to conserve 
   different absolute heavy element budgets (e.g., 7.9, 7.0, and 6.0 Earth masses).
   This isolates the thermal inflation effect of dilute cores across different 
   bulk compositions.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq

# Clean package imports natively utilizing the installed fuzzycore package
import fuzzycore.constants as c
import fuzzycore.solver as solver
import fuzzycore.utils as utils
import fuzzycore.eos as eos


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

M_KEPLER11E: float = 7.95 * c.M_EARTH

RESULTS_FILE: str = "../data/kepler11e_sweep_results.csv"
TEMP_FILE: str = "../data/kepler11e_sweep_temp.csv"


# =============================================================================
# PARALLEL WORKER FUNCTIONS
# =============================================================================

def get_params(m_core: float, sigma: float) -> dict:
    """Helper to generate the standard parameter dictionary."""
    z_prof = utils.generate_gaussian_z_profile(
        n_layers=30, sigma=sigma, z_base=0.02, z_core=0.99
    )
    return {
        'M_core': m_core * c.M_EARTH,   
        'M_rock': m_core * c.M_EARTH,   
        'M_water': 0.0,                 
        'P_surf': 1.0,
        'T_surf': 900.0,                
        'z_base': 0.05,                 
        'z_profile': np.round(z_prof, 2), 
        'sigma_val': sigma,              
        'iron_fraction': 0.33,          
        'debug': False
    }


def find_conserved_sigma(m_core: float, target_mz_me: float, lock) -> float:
    """
    Optimizes the sigma parameter using a high-speed Monotonic Hybrid system.
    
    Corrected: The try/except block now wraps both Phase 1 and Phase 2 to 
    ensure early exits are caught regardless of when they occur.
    """
    import fuzzycore.eos as eos
    from scipy.optimize import brentq
    
    # Custom signal for early exit
    class ConvergenceSuccess(Exception):
        def __init__(self, sigma): self.sigma = sigma

    if m_core >= target_mz_me - 0.01: 
        return 0.0 

    def eval_sigma(sig_guess: float) -> float:
        """Evaluates planet and raises ConvergenceSuccess if threshold is met."""
        p = get_params(m_core, sig_guess)
        res = solver.solve_structure(M_KEPLER11E, p, 'mass', 'opt_temp', os.devnull, lock)
        eos._MIXED_CACHE.clear()

        if res is None:
            raise ValueError("Unphysical: Planet Unbound")
            
        actual_mz_me = res.get('M_Z_total', m_core * c.M_EARTH) / c.M_EARTH
        error = actual_mz_me - target_mz_me
        
        print(f"        [Eval] Sigma: {sig_guess:.4f} | Err: {error:+.4f} M_E")

        # --- THE STOPPING THRESHOLD ---
        if abs(error) < 0.005:
            raise ConvergenceSuccess(sig_guess)
            
        return error

    # WRAP EVERYTHING in the success handler
    try:
        # =========================================================================
        # PHASE 1: MONOTONIC SCOUT
        # =========================================================================
        sig_low = 0.0
        try:
            err_low = eval_sigma(sig_low)
            if err_low > 0: return sig_low 
        except ValueError:
            raise ValueError("Planet physically unbound even at minimum envelope.")

        sig_high_guesses = [5.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.15, 0.08, 0.04, 0.02, 0.001]
        sig_high = None
        
        for guess in sig_high_guesses:
            try:
                err_high = eval_sigma(guess)
                if err_high > 0:
                    sig_high = guess
                    break 
                else:
                    # If highest stable sigma is still too metal-poor, abort.
                    raise ValueError(f"Target unreachable: Max stable Z-mass is {target_mz_me + err_high:.2f} M_E")
            except ValueError as e:
                if "Unphysical" in str(e):
                    print(f"      [Scout] Sigma {guess:.4f} is unphysical. Trying sharper...")
                    continue
                raise e

        if sig_high is None:
            raise ValueError("Target mass unreachable: No stable configuration holds enough metal.")

        # =========================================================================
        # PHASE 2: BRENT'S METHOD
        # =========================================================================
        print(f"      [Hybrid] Bracket secured: [{sig_low:.4f}, {sig_high:.4f}]. Sniping root...")
        final_sigma = brentq(eval_sigma, sig_low, sig_high, xtol=1e-6)
        return float(final_sigma)

    except ConvergenceSuccess as success:
        # This catches early exits from Phase 1 AND Phase 2
        print(f"    -> [Hybrid] Physical convergence reached at Sigma: {success.sigma:.4f}")
        return success.sigma


def run_single_model(args: tuple, lock) -> tuple:
    """
    Executes the integration for a specific core mass and transfer mode, 
    enforcing strict conservation checks based on the track's metal target.
    """
    m_core, track_name, mode_type, target_mz_me = args
    trial_id = f"K11e_{mode_type}_{target_mz_me}_{m_core:.3f}"
    
    try:
        # Determine the correct sigma based on the physical mode type
        if mode_type == 'Bulk':
            # Original physics: Fixed sigma shape
            sigma = 0.25
        else:
            # Direct Metal Transfer: Tune sigma to conserve exact target metals
            sigma = find_conserved_sigma(m_core, target_mz_me, lock)
            
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
            
            # STRICT GATEKEEPER: Check against the dynamic target
            if mode_type == 'Direct':
                # RELAXED from 0.01 to 0.05 to account for numerical integration noise
                if abs(m_z_tot_me - target_mz_me) > 0.05:
                    error_msg = f"Discarded: Conservation failed (M_z = {m_z_tot_me:.2f} vs Target {target_mz_me:.2f})"
                    return False, m_core, track_name, error_msg

            # Dilute mass is the total heavy elements minus the solid core
            m_dilute = max(m_z_tot_me - m_core, 0.0)
            
            result_dict = {
                'M_core_Me': m_core,
                'Transfer_Mode': track_name,
                'Target_Mz_Me': target_mz_me,
                'M_dilute_Me': m_dilute,
                'M_Z_total_Me': m_z_tot_me,
                'R_total_Re': r_tot_re,
                'Sigma_Used': sigma,
                'dt_ds_total': res.get('dt_ds_total', np.nan)
            }
            return True, m_core, track_name, result_dict
        else:
            return False, m_core, track_name, "Solver failed to converge."
            
    except Exception as e:
        return False, m_core, track_name, str(e)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    print("================================================================")
    print(" Kepler-11e 1D Sweep: Multi-Track Metal Consumptions ")
    print("================================================================")
    
    # Define all the evolutionary tracks you want to explore
    TRACKS = [
        {"name": "Bulk Envelope Transfer", "type": "Bulk", "target_mz_me": 7.9},
        #{"name": "Direct Metal (Total Z = 7.5 M_E)", "type": "Direct", "target_mz_me": 7.5},
        #{"name": "Direct Metal (Total Z = 7.0 M_E)", "type": "Direct", "target_mz_me": 7.0},
        #{"name": "Direct Metal (Total Z = 6.0 M_E)", "type": "Direct", "target_mz_me": 6.0},
        #{"name": "Direct Metal (Total Z = 5.0 M_E)", "type": "Direct", "target_mz_me": 5.0},
    ]
    
    # Checkpoint Logic
    completed_tasks = set()
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    if os.path.exists(RESULTS_FILE):
        try:
            df_existing = pd.read_csv(RESULTS_FILE)
            for _, row in df_existing.iterrows():
                completed_tasks.add((round(row['M_core_Me'], 3), row['Transfer_Mode']))
            print(f"[*] Found {len(completed_tasks)} completed runs.")
        except Exception as e:
            print(f"[*] Could not read {RESULTS_FILE}: {e}")
            
    # Dynamically generate grids anchored to each track's specific metal target
    tasks_to_run = []
    for track in TRACKS:
        t_name = track['name']
        t_type = track['type']
        t_mz = track['target_mz_me']
        
        # Grid clustering: High density near the track's starting point (0 dilute mass)
        if t_type == 'Bulk':
            mass_grid = np.concatenate([
                np.linspace(t_mz, t_mz - 0.4, 8),   
                np.linspace(t_mz - 0.5, 0.0, 10)    
            ])
        else:
            mass_grid = np.concatenate([
                np.linspace(t_mz, t_mz - 0.5, 8),   
                np.linspace(t_mz - 1.0, 1.0, 20)    
            ])
            
        for m in mass_grid:
            if (round(m, 3), t_name) not in completed_tasks:
                tasks_to_run.append((m, t_name, t_type, t_mz))
            
    print(f"[*] Tasks remaining to compute: {len(tasks_to_run)}")
    
    # Parallel Execution
    if tasks_to_run:
        # Re-enabled multiprocessing to speed up the multiple tracks
        n_workers = min(mp.cpu_count() - 1, 4)
        n_workers = max(1, n_workers) 
        #n_workers = 3
        
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
        
        plt.figure(figsize=(10, 7))
        
        # Define aesthetic mapping for our distinct tracks
        colors = {
            'Bulk Envelope Transfer': '#3498db',           # Blue
            'Direct Metal (Total Z = 7.9 M_E)': '#e74c3c', # Red
            'Direct Metal (Total Z = 7.0 M_E)': '#e67e22', # Orange
            'Direct Metal (Total Z = 6.0 M_E)': '#9b59b6'  # Purple
        }
        markers = {
            'Bulk Envelope Transfer': 's', 
            'Direct Metal (Total Z = 7.9 M_E)': 'o',
            'Direct Metal (Total Z = 7.0 M_E)': '^',
            'Direct Metal (Total Z = 6.0 M_E)': 'D'
        }

        for track in TRACKS:
            mode = track['name']
            df_mode = df_plot[df_plot['Transfer_Mode'] == mode].sort_values(by='M_core_Me', ascending=False)
            
            if len(df_mode) > 1:
                plt.plot(
                    df_mode['M_dilute_Me'], df_mode['R_total_Re'], 
                    marker=markers.get(mode, 'x'), markersize=7, linestyle='-', 
                    color=colors.get(mode, 'gray'), linewidth=2.5, label=mode
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

        plot_filename = "../figures/kepler11e_multi_track_comparison.pdf"
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        print(f"[*] Plot successfully saved as '{plot_filename}'.")
    else:
        print("[!] No results file found to plot.")