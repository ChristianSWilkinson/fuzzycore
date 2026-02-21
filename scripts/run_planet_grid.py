import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
import time
import os
import sys
import itertools
import gc 
import random # NEW: For shuffling tasks

# Ensure we can import our local package
sys.path.append(os.getcwd())
sys.path.append('./..')

from python_interior_clean import solver, constants as c, utils, eos

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_CBWR'] = 'COMPATIBLE'

write_lock = mp.Lock()

def find_initial_guess(p_surf, t_surf, m_core, sigma, target_val, mode, csv_file):
    """Scans the CSV for the closest successful trial to provide a starting logPc."""
    target_mj = target_val / c.M_JUPITER
    default_guess = 10.5 if target_mj <= 1.0 else 11.5
    
    if not os.path.exists(csv_file):
        return default_guess 
    
    try:
        cols = ['P_surf_bar', 'T_surf_K', 'M_core_input_Me', 'Sigma_dilute', 
                'target_value', 'target_mode', 'P_center_bar', 'status']
        df_history = pd.read_csv(csv_file, usecols=lambda x: x in cols)
        
        df_success = df_history[
            (df_history['status'] == 'success') & 
            (df_history['target_mode'] == mode)
        ]
        
        if df_success.empty:
            return default_guess 

        dist = (
            ((df_success['P_surf_bar'] - p_surf) / 10.0)**2 +
            ((df_success['T_surf_K'] - t_surf) / 500.0)**2 +
            ((df_success['M_core_input_Me'] - m_core) / 10.0)**2 +
            ((df_success['Sigma_dilute'] - sigma) / 0.1)**2 +
            ((df_success['target_value'] - target_val) / target_val)**2
        )
        
        closest_idx = dist.idxmin()
        return np.log10(df_success.loc[closest_idx, 'P_center_bar'])
    except Exception:
        return default_guess

def run_single_grid_point(args, csv_file, p_surf_val, mode, z_base, iron_frac):
    """Worker function for a single coordinate in the 3x3 multi-parameter grid."""
    trial_id, target_val, t_surf, m_core_val, sigma = args
    
    output = {
        'trial_id': trial_id,
        'target_mode': mode,
        'target_value': target_val,
        'P_surf_bar': p_surf_val,
        'T_surf_K': t_surf,
        'M_total_Mj': np.nan,
        'R_total_Rj': np.nan,
        'M_Z_total_Me': np.nan,
        'M_core_input_Me': m_core_val,
        'Sigma_dilute': sigma,
        'Z_base': z_base,
        'Iron_Fraction': iron_frac,
        'P_center_bar': np.nan,
        'status': 'pending'
    }
    
    try:
        z_core = 0.99  
        z_profile = utils.generate_gaussian_z_profile(n_layers=100, sigma=sigma, z_base=z_base, z_core=z_core)
        z_profile = np.round(z_profile, 3)
        
        log_pc_guess = find_initial_guess(p_surf_val, t_surf, m_core_val, sigma, target_val, mode, csv_file)
        
        current_params = {
            'P_surf': p_surf_val,
            'T_surf': t_surf,
            'M_core': m_core_val * c.M_EARTH,
            'z_profile': z_profile,
            'sigma_val': sigma,
            'z_base': z_base,
            'initial_log_pc': log_pc_guess,
            'iron_fraction': iron_frac, 
            'debug': False
        }
        
        # --- EXECUTE SOLVER ---
        result = solver.solve_structure(target_val, current_params, mode, trial_id, csv_file, write_lock)
        
        # --- MEMORY CLEANUP ---
        eos.clear_mixed_cache() 
        gc.collect() 
        
        if result is not None:
            output['M_total_Mj'] = result['M'][-1] / c.M_JUPITER
            output['R_total_Rj'] = result['R'][-1] / c.R_JUPITER
            output['M_Z_total_Me'] = result['M_Z_total'] / c.M_EARTH
            output['P_center_bar'] = 10**result['P'][0]
            
            total_r = output['R_total_Rj']
            if not np.isnan(total_r) and 0.4 < total_r < 2.5: 
                output['status'] = 'success'
            else:
                output['status'] = 'unphysical_radius'
        else:
            output['status'] = 'solver_returned_none'
            
    except Exception as e:
        output['status'] = f'error: {str(e)}'
        eos.clear_mixed_cache()
        gc.collect()

    try:
        with write_lock:
            pd.DataFrame([output]).to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
    except Exception as write_err:
        print(f"  [Fatal] Could not write trial {trial_id} to CSV: {write_err}")

    return output

def main():
    # --- GLOBAL CONFIGURATION ---
    MODE = 'mass'
    P_SURF = 1.0
    Z_BASE = 0.02   
    IRON_FRAC = 0.33 
    
    OUTPUT_FILE = "equivalence_grid_3x3.csv"

    # --- 1. RESUME LOGIC ---
    completed_trials = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE, usecols=['trial_id'])
            # Convert to string to avoid type mismatches
            completed_trials = set(df_existing['trial_id'].dropna().astype(str).tolist())
            print(f"Found {len(completed_trials)} already completed trials in {OUTPUT_FILE}.")
        except Exception as e:
            print(f"Warning: Could not read existing trials. Starting fresh. Error: {e}")

    # --- 2. 3x3 PARAMETER SPACE ---
    MASSES_MJ = [0.3, 1.0, 5.0]
    TEMPS_K = [200.0, 500.0, 1000.0]
    
    n_points = 20 
    
    tasks = []
    trial_idx = 0
    total_models_in_grid = 0
    
    print("Building task list...")
    for m_mj in MASSES_MJ:
        target_mass = m_mj * c.M_JUPITER
        
        if m_mj <= 0.5:
            max_sigma = 0.40
        elif m_mj <= 2.0:
            max_sigma = 0.20
        else:
            max_sigma = 0.05
            
        m_core_array = np.linspace(0.0, 100.0, n_points) 
        sigma_array = np.linspace(0.001, max_sigma, n_points)
        
        grid_points = list(itertools.product(m_core_array, sigma_array))
        
        for t_k in TEMPS_K:
            for m_c, sig in grid_points:
                # Deterministic ID based on loop counter
                trial_id = f"M{m_mj}_T{t_k}_{trial_idx}"
                total_models_in_grid += 1
                
                # Only add if we haven't solved it yet
                if trial_id not in completed_trials:
                    tasks.append((trial_id, target_mass, t_k, m_c, sig))
                
                trial_idx += 1
                
    N_CORES = max(1, mp.cpu_count() - 2) 
    
    # --- 3. RANDOMIZE EXECUTION ORDER ---
    print(f"Shuffling {len(tasks)} remaining tasks (out of {total_models_in_grid} total grid points)...")
    random.shuffle(tasks)
    
    if len(tasks) == 0:
        print("All grid points are already completed! Exiting.")
        return

    worker_func = partial(
        run_single_grid_point, 
        csv_file=OUTPUT_FILE,
        p_surf_val=P_SURF,
        mode=MODE,
        z_base=Z_BASE,
        iron_frac=IRON_FRAC
    )

    print(f"Starting execution pool with {N_CORES} CPU cores.")
    print(f"Results appending to {OUTPUT_FILE}...")
    
    # maxtasksperchild=5 prevents memory leaks
    with mp.Pool(processes=N_CORES, maxtasksperchild=5) as pool:
        pool.map(worker_func, tasks)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nManual stop detected. Cleaning up processes...")
    finally:
        time.sleep(1)