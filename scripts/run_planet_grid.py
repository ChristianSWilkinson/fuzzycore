"""
Mass-Temperature Equivalence Grid Sweep

This script executes a heavy-duty, parallelized 3x3 parameter sweep mapping the 
structural degeneracy of giant planets across varying total masses (0.3, 1.0, 5.0 M_J) 
and surface temperatures (200, 500, 1000 K). 

It systematically explores the trade-off between solid core mass and dilute 
envelope width (sigma) to find models that produce physically consistent radii.
The execution order is randomized to efficiently sample the parameter space, 
and results are checkpointed to a CSV to allow seamless pausing and resuming.
"""

import gc
import itertools
import multiprocessing as mp
import os
import random
import time
from functools import partial

import numpy as np
import pandas as pd

# Clean package imports natively utilizing the installed fuzzycore package
import fuzzycore.constants as c
import fuzzycore.eos as eos
import fuzzycore.solver as solver
import fuzzycore.utils as utils


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
# Intel MKL optimizations for stability during heavy multiprocessing
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_CBWR'] = 'COMPATIBLE'

# Global multiprocessing lock to prevent race conditions during CSV writing
write_lock = mp.Lock()


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================

def find_initial_guess(p_surf: float, t_surf: float, m_core: float, 
                       sigma: float, target_val: float, mode: str, 
                       csv_file: str) -> float:
    """
    Scans the historical results CSV to find the closest successful trial 
    and uses its central pressure as a warm-start guess for the root-finder.
    
    Args:
        p_surf (float): Surface pressure in bar.
        t_surf (float): Surface temperature in K.
        m_core (float): Input core mass in Earth masses.
        sigma (float): Diluteness parameter.
        target_val (float): Target total mass in kg.
        mode (str): Convergence mode ('mass').
        csv_file (str): Path to the results database.
        
    Returns:
        float: A log10(Pc) initial guess.
    """
    target_mj = target_val / c.M_JUPITER
    default_guess = 10.5 if target_mj <= 1.0 else 11.5
    
    if not os.path.exists(csv_file):
        return default_guess 
    
    try:
        cols = ['P_surf_bar', 'T_surf_K', 'M_core_input_Me', 'Sigma_dilute', 
                'target_value', 'target_mode', 'P_center_bar', 'status']
        
        # Load only the required columns to save memory
        df_history = pd.read_csv(csv_file, usecols=lambda x: x in cols)
        
        # Filter for successful runs matching the current target mode
        df_success = df_history[
            (df_history['status'] == 'success') & 
            (df_history['target_mode'] == mode)
        ]
        
        if df_success.empty:
            return default_guess 

        # Calculate a normalized Euclidean distance metric in parameter space
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


def run_single_grid_point(args: tuple, csv_file: str, p_surf_val: float, 
                          mode: str, z_base: float, iron_frac: float) -> dict:
    """
    Worker function executed by the multiprocessing pool for a single grid coordinate.
    
    Builds the structural parameters, fetches a warm-start guess, invokes the 
    hydrostatic solver, and writes the output to the shared CSV.
    """
    trial_id, target_val, t_surf, m_core_val, sigma = args
    
    # Initialize the output dictionary with NaNs
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
        # Generate the high-resolution compositional profile
        z_core = 0.99  
        z_profile = utils.generate_gaussian_z_profile(
            n_layers=100, sigma=sigma, z_base=z_base, z_core=z_core
        )
        z_profile = np.round(z_profile, 3)
        
        # Fetch an intelligent initial guess for central pressure
        log_pc_guess = find_initial_guess(
            p_surf_val, t_surf, m_core_val, sigma, target_val, mode, csv_file
        )
        
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
        result = solver.solve_structure(
            target_val, current_params, mode, trial_id, csv_file, write_lock
        )
        
        # --- MEMORY CLEANUP ---
        # Crucial for preventing massive RAM leaks during multi-day runs
        eos.clear_mixed_cache() 
        gc.collect() 
        
        # Process results
        if result is not None:
            output['M_total_Mj'] = result['M'][-1] / c.M_JUPITER
            output['R_total_Rj'] = result['R'][-1] / c.R_JUPITER
            output['M_Z_total_Me'] = result['M_Z_total'] / c.M_EARTH
            output['P_center_bar'] = 10 ** result['P'][0]
            
            # Filter out completely unphysical radius blowouts
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

    # Safely write the final row to the database
    try:
        with write_lock:
            df_out = pd.DataFrame([output])
            file_exists = os.path.exists(csv_file)
            df_out.to_csv(csv_file, mode='a', header=not file_exists, index=False)
    except Exception as write_err:
        print(f"  [Fatal] Could not write trial {trial_id} to CSV: {write_err}")

    return output


# =============================================================================
# MAIN EXECUTION SCRIPT
# =============================================================================

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
            print(f"[*] Found {len(completed_trials)} already completed trials in {OUTPUT_FILE}.")
        except Exception as e:
            print(f"[!] Warning: Could not read existing trials. Starting fresh. Error: {e}")

    # --- 2. BUILD THE 3x3 PARAMETER SPACE ---
    MASSES_MJ = [0.3, 1.0, 5.0]
    TEMPS_K = [200.0, 500.0, 1000.0]
    
    n_points = 20 
    
    tasks = []
    trial_idx = 0
    total_models_in_grid = 0
    
    print("[*] Building task list...")
    for m_mj in MASSES_MJ:
        target_mass = m_mj * c.M_JUPITER
        
        # Adjust maximum diluteness allowed based on the planet's total mass
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
                # Deterministic ID based on exact coordinate
                trial_id = f"M{m_mj}_T{t_k}_{trial_idx}"
                total_models_in_grid += 1
                
                # Only append the task if it hasn't been solved in a previous run
                if trial_id not in completed_trials:
                    tasks.append((trial_id, target_mass, t_k, m_c, sig))
                
                trial_idx += 1
                
    # Reserve two cores for OS stability
    N_CORES = max(1, mp.cpu_count() - 2) 
    
    # --- 3. RANDOMIZE EXECUTION ORDER ---
    # Shuffling prevents the grid from getting "stuck" in a slow region of phase space
    print(f"[*] Shuffling {len(tasks)} remaining tasks (out of {total_models_in_grid} total grid points)...")
    random.shuffle(tasks)
    
    if len(tasks) == 0:
        print("[*] All grid points are already completed! Exiting.")
        return

    # Package the static arguments into the worker function
    worker_func = partial(
        run_single_grid_point, 
        csv_file=OUTPUT_FILE,
        p_surf_val=P_SURF,
        mode=MODE,
        z_base=Z_BASE,
        iron_frac=IRON_FRAC
    )

    print(f"[*] Starting execution pool with {N_CORES} CPU cores.")
    print(f"[*] Results appending to {OUTPUT_FILE}...")
    
    # Use maxtasksperchild=5 to periodically kill and restart worker processes,
    # preventing memory fragmentation leaks in C-level Scipy interpolators.
    with mp.Pool(processes=N_CORES, maxtasksperchild=5) as pool:
        pool.map(worker_func, tasks)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Manual stop detected. Cleaning up processes...")
    finally:
        time.sleep(1)