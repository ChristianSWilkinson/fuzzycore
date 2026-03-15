"""
Planetary Structure Solver

This module handles the root-finding and parameter-sweeping logic required
to converge on a physically consistent planetary interior. It dynamically 
switches between different integration architectures (e.g., Gas Giants vs. 
Water Worlds) and tracks intermediate solutions during the solving process.
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import constants as c
from . import eos
from . import physics


def solve_structure(target_val: float, params: dict, mode: str, 
                    trial_id: str, csv_file: str, write_lock) -> dict:
    """
    Solves for the planetary structure by finding the central pressure 
    required to match a target total mass or surface gravity.

    This function dynamically detects if the requested model is a "Water World" 
    (featuring a condensed mantle) or a "Simple Model" (gas giant with a 
    direct core-envelope transition) based on the presence of `M_water`.

    Args:
        target_val (float): The target physical value to converge on 
            (e.g., Total Mass in kg, or Surface Gravity).
        params (dict): Dictionary of planetary boundary conditions and 
            structural mass parameters.
        mode (str): The convergence target mode ('mass' or 'gravity').
        trial_id (str): A unique identifier string for logging this specific run.
        csv_file (str): Path to the temporary CSV file for tracking solver steps.
        write_lock (threading.Lock): A lock object to prevent race conditions 
            when writing intermediate results to the CSV in parallel.

    Returns:
        dict: The final converged planetary profile dictionary. Returns `None` 
            if the solver fails to find a physically valid root.
    """

    # =========================================================================
    # 0. DEBUG: ABSOLUTE INPUT INTERCEPTION (TRIPWIRE)
    # =========================================================================
    if params.get('debug', False):
        print("\n" + "="*60, flush=True)
        print(f"🛑 [FUZZYCORE TRIPWIRE: EXACT INPUTS RECEIVED] 🛑", flush=True)
        print(f"Trial ID: {trial_id} | Mode: {mode}", flush=True)
        
        # Check the Target Value (Mass or Gravity)
        unit_guess = "kg" if mode == 'mass' else "m/s^2"
        print(f"Target Value: {target_val:.5e} [{unit_guess}]", flush=True)
        if mode == 'mass':
            print(f"   -> Equivalent to: {target_val / c.M_EARTH:.2f} Earth Masses", flush=True)
            print(f"   -> Equivalent to: {target_val / c.M_JUPITER:.5f} Jupiter Masses", flush=True)
            
        print("-" * 60, flush=True)
        print("Raw Parameters Dictionary:", flush=True)
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                print(f"  - {k}: ndarray (shape: {v.shape}, mean: {np.mean(v):.3f})", flush=True)
            elif isinstance(v, float) and v > 1e20: # Flag suspiciously huge numbers
                print(f"  - {k}: {v:.3e} ⚠️ (MASSIVE NUMBER)", flush=True)
            else:
                print(f"  - {k}: {v}", flush=True)
        print("="*60 + "\n", flush=True)

    # =========================================================================
    # 1. Setup Equation of State (EOS) Data
    # =========================================================================
    
    # Initialize the base interpolators
    rock = eos.get_rock_interpolator()  # Now guaranteed to exist

    # Extract the dynamic H/He ratio from params (defaults to Solar 0.26)
    y_ratio = params.get('Y_ratio', 0.26)
    
    # Default to a 10-step gradient if z_profile is missing
    default_z_profile = np.linspace(0.01, 1.0, 10)
    fluid = eos.generate_fluid_interpolators(params.get('z_profile', default_z_profile))
    
    eos_data = {'rock': rock, 'fluid': fluid}
    
    # Detect Mode: Does this planet have a distinct water mantle?
    is_water_world = 'M_water' in params and params['M_water'] > 0
    
    # =========================================================================
    # 2. Parameter Aliasing & Fallbacks
    # =========================================================================
    
    # If the user provides M_rock but no M_water, seamlessly map it to M_core 
    # to prevent KeyErrors in the simple gas giant integrator.
    if 'M_core' not in params and 'M_rock' in params:
        params['M_core'] = params['M_rock']
    elif 'M_rock' not in params and 'M_core' in params:
        params['M_rock'] = params['M_core']  # Fallback for intermediate outputs

    # If it is a water world, load the specific ab-initio water tables
    if is_water_world:
        water = eos.get_water_interpolators_complete()
        eos_data['water'] = water

    params['target_m'] = target_val
    
    # =========================================================================
    # 3. Objective Function for Root Finding (WITH MEMORY CACHE)
    # =========================================================================
    
    # 🛑 THE EFFICIENCY FIX: Cache evaluations so we never integrate the same guess twice!
    eval_cache = {}

    def objective(log_pc: float) -> float:
        """
        Integrates the planet for a guessed central pressure (log_pc) 
        and returns the error relative to the target mass/gravity.
        """
        # Truncate slightly to prevent floating point cache misses
        log_pc_rounded = round(float(log_pc), 5)
        if log_pc_rounded in eval_cache:
            return eval_cache[log_pc_rounded]

        try:
            if is_water_world:
                res = physics.integrate_water_world(log_pc, params, eos_data)
                interior_mass = params['M_rock'] + params['M_water']
            else:
                res = physics.integrate_planet(log_pc, params, eos_data)
                interior_mass = params['M_core']
            
            # --- FAIL CRITERIA ---
            if res is None or np.isnan(res['M'][-1]):
                # 🛑 INSTEAD of a violent -1e30 crash, return the bare interior mass.
                # This tells the root-finder: "At this pressure, you get exactly 0 envelope."
                # It creates a perfectly smooth, physical slope for Brentq to follow!
                if mode == 'mass':
                    error = interior_mass - target_val 
                else:
                    # Approximate gravity of a bare rock to keep the gradient smooth
                    approx_r = (interior_mass / ( (4/3)*np.pi * 5000 ))**(1/3)
                    error = (c.G_CONST * interior_mass / approx_r**2) - target_val
                    
            elif res['M'][-1] < (interior_mass * 0.99):
                # Same fallback if the integration stopped prematurely
                if mode == 'mass':
                    error = interior_mass - target_val
                else:
                    approx_r = (interior_mass / ( (4/3)*np.pi * 5000 ))**(1/3)
                    error = (c.G_CONST * interior_mass / approx_r**2) - target_val
                    
            else:
                actual_m = res['M'][-1]
                actual_r = res['R'][-1]
                
                # Save intermediate solver steps for tracking
                intermediate_output = {
                    'trial_id': f"{trial_id}_step",
                    'target_mode': mode,
                    'target_value': target_val,
                    'P_surf_bar': params['P_surf'],
                    'T_surf_K': params['T_surf'],
                    'M_total_Mj': actual_m / c.M_JUPITER,
                    'R_total_Rj': actual_r / c.R_JUPITER,
                    'M_Z_total_Me': res.get('M_Z_total', params.get('M_rock', 0)) / c.M_EARTH,
                    'P_center_bar': 10 ** log_pc,
                    'Iron_Fraction': params.get('iron_fraction', 0.0),
                    'status': 'success_intermediate' 
                }
                
                with write_lock:
                    df_out = pd.DataFrame([intermediate_output])
                    file_exists = os.path.exists(csv_file)
                    df_out.to_csv(csv_file, mode='a', header=not file_exists, index=False)
                
                # Calculate final error and set current_val for logging
                if mode == 'gravity':
                    g_surf = (c.G_CONST * actual_m) / (actual_r ** 2)
                    error = g_surf - target_val
                    current_val = g_surf
                    unit = "m/s^2"
                elif mode == 'mass':
                    error = actual_m - target_val
                    current_val = actual_m
                    unit = "kg"
            
            # --- 🛑 NEW VERBOSITY INJECTION ---
            if params.get('debug', False):
                # Check if it was a bare-rock fallback
                if res is None or np.isnan(res['M'][-1]) or res['M'][-1] < (interior_mass * 0.99):
                    print(f"  [Solver Eval] logPc: {log_pc:.4f} | ⚠️ FALLBACK (Env Failed) | Error: {error:.4e}", flush=True)
                else:
                    # Print exact tracking metrics
                    print(f"  [Solver Eval] logPc: {log_pc:.4f} | Current: {current_val:.4e} {unit} | Target: {target_val:.4e} {unit} | Delta: {error:+.4e}", flush=True)
            # ----------------------------------

            eval_cache[log_pc_rounded] = error
            return error
                
        except Exception as e:
            if params.get('debug'):
                print(f"  [Solver] Error at logPc {log_pc:.2f}: {e}")
            eval_cache[log_pc_rounded] = 1e30
            return 1e30

    # =========================================================================
    # 4. Dynamic Bounds & Concentric Bracketing Search
    # =========================================================================
    
    # Vastly relaxed bounds. Thin envelopes require wildly varied central pressures.
    m_core_earth = params.get('M_rock', params.get('M_core', 5.0)) / c.M_EARTH
    
    if m_core_earth < 2.0: min_pc, max_pc = 4.5, 11.0
    elif m_core_earth < 10.0: min_pc, max_pc = 5.5, 13.0
    elif m_core_earth < 50.0: min_pc, max_pc = 6.5, 14.5
    else: min_pc, max_pc = 7.5, 15.5

    guess = params.get('initial_log_pc', None)
    bracket = None

    # --- A. Mine the Smart Prior ---
    if os.path.exists(csv_file):
        try:
            with write_lock:
                df_history = pd.read_csv(csv_file)
            
            df_history = df_history[df_history['status'] == 'success_intermediate']
            if not df_history.empty:
                if mode == 'mass':
                    achieved_mass_kg = df_history['M_total_Mj'] * c.M_JUPITER
                    best_idx = np.argmin(np.abs(achieved_mass_kg - target_val))
                elif mode == 'gravity':
                    achieved_mass_kg = df_history['M_total_Mj'] * c.M_JUPITER
                    achieved_radius_m = df_history['R_total_Rj'] * c.R_JUPITER
                    achieved_g = (c.G_CONST * achieved_mass_kg) / (achieved_radius_m**2)
                    best_idx = np.argmin(np.abs(achieved_g - target_val))
                    
                best_pc = df_history.iloc[best_idx]['P_center_bar']
                guess = np.log10(best_pc)
                if params.get('debug'):
                    print(f"  [Smart Prior] Mined historical model. Prior guess set to logPc={guess:.3f}")
        except Exception:
            pass

    # --- B. The Concentric Search Algorithm ---
    # We map the "Valley of Death" by testing tight offsets first (±0.05), 
    # ensuring we never accidentally step over the target mass!
    center = guess if guess is not None else (min_pc + max_pc) / 2.0
    center = max(min_pc, min(max_pc, center))

    offsets = [
        0.0, 
        -0.05, 0.05, 
        -0.15, 0.15, 
        -0.3, 0.3, 
        -0.6, 0.6, 
        -1.0, 1.0, 
        -1.5, 1.5, 
        -2.0, 2.0,
        -3.0, 3.0
    ]
    
    valid_evals = []

    if params.get('debug'):
        print(f"  [Solver] Launching concentric bracket search around logPc={center:.2f}...")

    for offset in offsets:
        p_test = center + offset
        if min_pc <= p_test <= max_pc:
            err = objective(p_test)
            
            if abs(err) < 1e29:
                valid_evals.append((p_test, err))
                valid_evals.sort(key=lambda x: x[0]) # Always sort by pressure
                
                # Check for a zero-crossing bracket anywhere in the mapped space
                for i in range(len(valid_evals) - 1):
                    if np.sign(valid_evals[i][1]) != np.sign(valid_evals[i+1][1]):
                        bracket = (valid_evals[i][0], valid_evals[i+1][0])
                        break
        if bracket:
            if params.get('debug'):
                print(f"  [Solver] ✅ Root securely bracketed between {bracket[0]:.3f} and {bracket[1]:.3f}!")
            break

    # --- C. Fallback Global Grid ---
    if not bracket:
        if params.get('debug'):
            print("  [Solver] Concentric search failed. Launching ultra-wide global fallback grid...")
        global_pts = np.linspace(min_pc, max_pc, 25)
        for p_test in global_pts:
            err = objective(p_test)
            if abs(err) < 1e29:
                valid_evals.append((p_test, err))
                valid_evals.sort(key=lambda x: x[0])
                for i in range(len(valid_evals) - 1):
                    if np.sign(valid_evals[i][1]) != np.sign(valid_evals[i+1][1]):
                        bracket = (valid_evals[i][0], valid_evals[i+1][0])
                        break
            if bracket:
                if params.get('debug'):
                    print(f"  [Solver] ✅ Root globally bracketed between {bracket[0]:.3f} and {bracket[1]:.3f}!")
                break

    # =========================================================================
    # 5. Final Convergence (Brent's Method)
    # =========================================================================
    try:
        if bracket:
            # Brentq is super fast because the bracket bounds are loaded from the eval_cache!
            root = brentq(objective, bracket[0], bracket[1], xtol=1e-4)
            
            if is_water_world:
                return physics.integrate_water_world(root, params, eos_data)
            return physics.integrate_planet(root, params, eos_data)
        else:
            if params.get('debug'):
                print("  [Solver] FATAL: Could not bracket the root. Planet may be physically impossible.")
            return None
            
    except Exception as e:
        if params.get('debug'):
            print(f"  [Solver] Root finding failed: {e}")
        return None