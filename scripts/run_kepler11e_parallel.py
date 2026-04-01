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

# =============================================================================
# PARALLEL WORKER FUNCTIONS
# =============================================================================

def get_params(m_core: float, sigma: float) -> dict:
    """Helper to generate the standard parameter dictionary."""
    z_prof = utils.generate_gaussian_z_profile(
        n_layers=50, sigma=sigma, z_base=0.02, z_core=0.99
    )
    return {
        'M_core': m_core * c.M_EARTH,   
        'M_rock': m_core * c.M_EARTH,   
        'M_water': 0.0,                 
        'P_surf': 1.0,
        'T_surf': 900.0,  
        'T_int': 300.0,              
        'z_base': 0.02,                 
        'z_profile': z_prof,             # Unrounded, continuous profile as requested
        'sigma_val': sigma,              
        'iron_fraction': 0.33,          
        'debug': True,
        'initial_log_pc': 7
    }


def find_conserved_sigma(m_core: float, target_mz_me: float, lock, hint_sigma: float = None) -> float:
    """
    Optimizes the sigma parameter using a robust array-based scouting system.
    Safely bypasses discontinuous runaway-gas cliffs at low sigma values.
    """
    import fuzzycore.eos as eos
    from scipy.optimize import brentq
    
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
            raise ValueError("Unphysical: Planet Unbound or on Runaway Cliff")
            
        actual_mz_me = res.get('M_Z_total', m_core * c.M_EARTH) / c.M_EARTH
        error = actual_mz_me - target_mz_me
        
        print(f"        [Eval] Sigma: {sig_guess:.4f} | Err: {error:+.4f} M_E")

        if abs(error) < 0.005:
            raise ConvergenceSuccess(sig_guess)
            
        return error

    try:
        # =========================================================================
        # PHASE 0: FAST-TRACK WARM START
        # =========================================================================
        if hint_sigma is not None and hint_sigma > 0.0:
            print(f"      [Fast-Track] Testing bracket around hint sigma: {hint_sigma:.4f}")
            try:
                sl = max(0.001, hint_sigma * 0.4)
                sh = min(3.0, hint_sigma * 2.5)
                
                err_l = eval_sigma(sl)
                err_h = eval_sigma(sh)
                
                if np.sign(err_l) != np.sign(err_h):
                    print(f"      [Fast-Track] Bracket secured: [{sl:.4f}, {sh:.4f}].")
                    return float(brentq(eval_sigma, sl, sh, xtol=1e-6))
            except Exception:
                print("      [Fast-Track] Hint unstable. Falling back to Phase 1.")

        # =========================================================================
        # PHASE 1: ROBUST ARRAY SCOUTING
        # =========================================================================
        test_sigmas = [0.0, 0.001, 0.01, 0.05, 0.15, 0.25, 0.5, 1.0, 2.0, 3.0]
        stable_points = []
        
        for guess in test_sigmas:
            try:
                err = eval_sigma(guess)
                stable_points.append((guess, err))
            except ValueError:
                if get_params(m_core, guess).get('debug'):
                    print(f"      [Scout] Sigma {guess:.4f} physically unstable (Skipping).")
                continue
                
        if not stable_points:
            raise ValueError("All tested Sigma values hit runaway cliffs for this core mass.")
            
        # =========================================================================
        # PHASE 2: BRACKET EXTRACTION & BRENTQ
        # =========================================================================
        bracket_low = None
        bracket_high = None
        
        for i in range(len(stable_points) - 1):
            if np.sign(stable_points[i][1]) != np.sign(stable_points[i+1][1]):
                bracket_low = stable_points[i][0]
                bracket_high = stable_points[i+1][0]
                break
                
        if bracket_low is not None and bracket_high is not None:
            print(f"      [Hybrid] Bracket secured: [{bracket_low:.4f}, {bracket_high:.4f}]. Sniping root...")
            final_sigma = brentq(eval_sigma, bracket_low, bracket_high, xtol=1e-6)
            return float(final_sigma)
        else:
            # Handle edge cases where curve doesn't cross zero within the test domain
            if all(err < 0 for s, err in stable_points):
                raise ValueError(f"Target unreachable: Max stable Z-mass is too low even at sigma={stable_points[-1][0]}")
            elif all(err > 0 for s, err in stable_points):
                return stable_points[0][0] 

    except ConvergenceSuccess as success:
        print(f"    -> [Hybrid] Physical convergence reached at Sigma: {success.sigma:.4f}")
        return success.sigma
    

def run_single_model(args: tuple, lock) -> tuple:
    """ Executes the integration and validates against DDC scaling laws. """
    m_core, track_name, mode_type, target_mz_me, hint_sigma = args
    trial_id = f"K11e_{mode_type}_{target_mz_me}_{m_core:.3f}"
    
    worker_temp_file = f"../data/temp_{trial_id}.csv"
    
    try:
        if mode_type == 'Bulk':
            sigma = 0.10
        else:
            sigma = find_conserved_sigma(m_core, target_mz_me, lock, hint_sigma=hint_sigma)
            
        params = get_params(m_core, sigma)
        
        res = solver.solve_structure(
            target_val=M_KEPLER11E,
            params=params,
            mode='mass',
            trial_id=trial_id,
            csv_file=worker_temp_file,
            write_lock=lock
        )
        
        if os.path.exists(worker_temp_file):
            try: os.remove(worker_temp_file)
            except: pass
        
        if res is not None:
            r_tot_re = res['R'][-1] / c.R_EARTH
            m_z_tot_kg = res.get('M_Z_total', m_core * c.M_EARTH)
            m_z_tot_me = m_z_tot_kg / c.M_EARTH
            
            if mode_type == 'Direct':
                if abs(m_z_tot_me - target_mz_me) > 0.05:
                    return False, m_core, track_name, "Conservation failed"

            # 🛑 THE FIX: Removed 'if mode_type == Direct' restriction!
            # This now runs for BOTH Bulk and Direct tracks.
            match_ratio, grad_fuzzy, grad_ddc = np.nan, np.nan, np.nan
            try:
                ddc_proof = utils.verify_ddc_macroscopic_gradient(
                    results=res, 
                    t_int=params.get('T_int', params['T_surf']), 
                    lambda_cd=10.0,   
                    Ra_T=1e8,         
                    l_H=0.1           
                )
                if ddc_proof.get('valid', False):
                    match_ratio = ddc_proof['match_ratio']
                    grad_fuzzy = ddc_proof['grad_fuzzy']
                    grad_ddc = ddc_proof['grad_ddc']
            except Exception:
                pass

            m_dilute = max(m_z_tot_me - m_core, 0.0)
            
            result_dict = {
                'M_core_Me': m_core,
                'Transfer_Mode': track_name,
                'Target_Mz_Me': target_mz_me,
                'M_dilute_Me': m_dilute,
                'M_Z_total_Me': m_z_tot_me,
                'R_total_Re': r_tot_re,
                'Sigma_Used': sigma,
                'dt_ds_total': res.get('dt_ds_total', np.nan),
                'DDC_Match_Ratio': match_ratio, # Now populated for Bulk!
                'Grad_Fuzzy': grad_fuzzy,
                'Grad_DDC': grad_ddc
            }
            
            proof_str = f" | DDC Proof: {match_ratio:.3f}" if not np.isnan(match_ratio) else ""
            print(f"  [+] {track_name} | Core: {m_core:.2f} -> R = {r_tot_re:.2f} R_E{proof_str}")
            
            return True, m_core, track_name, result_dict
        else:
            return False, m_core, track_name, "Solver failed to converge."
            
    except Exception as e:
        if os.path.exists(worker_temp_file):
            try: os.remove(worker_temp_file)
            except: pass
        return False, m_core, track_name, str(e)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    print("================================================================")
    print(" Kepler-11e 1D Sweep: Multi-Track Metal Consumptions ")
    print("================================================================")
    
    TRACKS = [
        #{"name": "Bulk Envelope Transfer", "type": "Bulk", "target_mz_me": 7.9},
        #{"name": "Direct Metal (Total Z = 7.5 M_E)", "type": "Direct", "target_mz_me": 7.5},
        {"name": "Direct Metal (Total Z = 7.0 M_E)", "type": "Direct", "target_mz_me": 7.0},
        #{"name": "Direct Metal (Total Z = 6.0 M_E)", "type": "Direct", "target_mz_me": 6.0},
        #{"name": "Direct Metal (Total Z = 5.0 M_E)", "type": "Direct", "target_mz_me": 5.0},
    ]
    
    completed_tasks = set()
    completed_sigmas = {} 
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    if os.path.exists(RESULTS_FILE):
        try:
            df_existing = pd.read_csv(RESULTS_FILE)
            for _, row in df_existing.iterrows():
                m_rnd = round(row['M_core_Me'], 3)
                t_mode = row['Transfer_Mode']
                completed_tasks.add((m_rnd, t_mode))
                
                if t_mode != 'Bulk Envelope Transfer':
                    completed_sigmas[(t_mode, m_rnd)] = row['Sigma_Used']
                    
            print(f"[*] Found {len(completed_tasks)} completed runs.")
        except Exception as e:
            print(f"[*] Could not read {RESULTS_FILE}: {e}")
            
    tasks_to_run = []
    for track in TRACKS:
        t_name = track['name']
        t_type = track['type']
        t_mz = track['target_mz_me']
        
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
            mass_grid = mass_grid [::-1]
            
        for m in mass_grid:
            if (round(m, 3), t_name) not in completed_tasks:
                hint_sigma = None
                if t_type == 'Direct' and completed_sigmas:
                    track_keys = [k for k in completed_sigmas.keys() if k[0] == t_name]
                    if track_keys:
                        closest_key = min(track_keys, key=lambda k: abs(k[1] - m))
                        hint_sigma = completed_sigmas[closest_key]

                tasks_to_run.append((m, t_name, t_type, t_mz, hint_sigma))
            
    print(f"[*] Tasks remaining to compute: {len(tasks_to_run)}")
    
    if tasks_to_run:
        n_workers = min(mp.cpu_count() - 1, 1)
        n_workers = max(1, n_workers) 
        
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
        
        colors = {
            'Bulk Envelope Transfer': '#3498db',           
            'Direct Metal (Total Z = 7.5 M_E)': '#e74c3c', 
            'Direct Metal (Total Z = 7.0 M_E)': '#e67e22', 
            'Direct Metal (Total Z = 6.0 M_E)': '#9b59b6'  
        }
        markers = {
            'Bulk Envelope Transfer': 's', 
            'Direct Metal (Total Z = 7.5 M_E)': 'o',
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