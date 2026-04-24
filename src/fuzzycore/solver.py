"""
Planetary Structure Solver

This module handles the root-finding and parameter-sweeping logic required
to converge on a physically consistent planetary interior. It dynamically 
switches between different integration architectures (e.g., Gas Giants vs. 
Water Worlds) and tracks intermediate solutions during the solving process.
"""

import os

import numpy as np
from scipy.optimize import brentq

from . import constants as c
from . import eos
from . import physics


def solve_structure(target_val: float, params: dict, mode: str, 
                    trial_id: str) -> dict:
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
    
    rock = eos.get_rock_interpolator()  
    y_ratio = params.get('Y_ratio', 0.26)
    
    default_z_profile = np.linspace(0.01, 1.0, 10)
    fluid = eos.generate_fluid_interpolators(params.get('z_profile', default_z_profile))
    
    eos_data = {'rock': rock, 'fluid': fluid}
    
    is_water_world = 'M_water' in params and params['M_water'] > 0
    
    # =========================================================================
    # 2. Parameter Aliasing & Fallbacks
    # =========================================================================
    
    if 'M_core' not in params and 'M_rock' in params:
        params['M_core'] = params['M_rock']
    elif 'M_rock' not in params and 'M_core' in params:
        params['M_rock'] = params['M_core']  

    if is_water_world:
        water = eos.get_water_interpolators_complete()
        eos_data['water'] = water

    params['target_m'] = target_val
    
    # =========================================================================
    # 3. Objective Function for Root Finding (WITH MEMORY CACHE, NO DISK I/O)
    # =========================================================================
    
    eval_cache = {}

    def objective(log_pc: float) -> float:
        """
        Integrates the planet for a guessed central pressure (log_pc) 
        and returns the error relative to the target mass/gravity.
        """
        log_pc_rounded = round(float(log_pc), 12)
        if log_pc_rounded in eval_cache:
            return eval_cache[log_pc_rounded]

        # 🛑 TRIPWIRE: Announce the attempt
        if params.get('debug'):
            print(f"\n    [Objective Attempt] logPc: {log_pc:.4f} (Pc: {10**log_pc:.2e} bar)")

        try:
            if is_water_world:
                res = physics.integrate_water_world(log_pc, params, eos_data)
                interior_mass = params['M_rock'] + params['M_water']
            else:
                res = physics.integrate_planet(log_pc, params, eos_data)
                interior_mass = params['M_core']
            
            # --- THE FIX: SMART FALLBACK ERROR ---
            if res is None or np.isnan(res['M'][-1]):
                # If physics integration fails completely (usually because pressure is too weak),
                # we force a strongly negative synthetic error to push brentq to hunt higher pressures.
                error = -1e20 + (log_pc * 1e18) 
                
                if params.get('debug'):
                    print(f"      ❌ FAILURE: Integration returned None (Unbound) | Synthetic Error: {error:.2e}")
                    
            elif res['M'][-1] < (interior_mass * 0.99):
                # If it stalled prematurely, also push higher
                error = -1e19 + (log_pc * 1e17)

                if params.get('debug'):
                    print(f"      ❌ FAILURE: Integration Prematurely Stalled | Synthetic Error: {error:.2e}")
                    
            else:
                actual_m = res['M'][-1]
                actual_r = res['R'][-1]
                
                # Calculate final error (NO CSV WRITING HAPPENS HERE ANYMORE = MASSIVE SPEEDUP)
                if mode == 'gravity':
                    g_surf = (c.G_CONST * actual_m) / (actual_r ** 2)
                    error = g_surf - target_val
                elif mode == 'mass':
                    error = actual_m - target_val
                
                if params.get('debug'):
                    print(f"      ✅ SUCCESS: Mass Achieved: {actual_m/c.M_EARTH:.3f} Me | Error: {error/c.M_EARTH:+.3f} Me")
            
            eval_cache[log_pc_rounded] = error
            return error
                
        except Exception as e:
            if params.get('debug'):
                print(f"      💥 CRASH in Objective: {str(e)}")
            eval_cache[log_pc_rounded] = -1e20
            return -1e20

    # =========================================================================
    # 4. Dynamic Bounds & Concentric Bracketing Search
    # =========================================================================
    
    m_core_earth = params.get('M_rock', params.get('M_core', 5.0)) / c.M_EARTH
    
    if m_core_earth < 2.0: min_pc, max_pc = 4.5, 9.0
    elif m_core_earth < 10.0: min_pc, max_pc = 5.5, 11.0
    elif m_core_earth < 50.0: min_pc, max_pc = 6.5, 14.5
    else: min_pc, max_pc = 7.5, 15.5

    guess = params.get('initial_log_pc', None)
    bracket = None

    # We map the "Valley of Death" by testing tight offsets first
    center = guess if guess is not None else (min_pc + max_pc) / 2.0
    center = max(min_pc, min(max_pc, center))

    # SPEEDUP: Streamlined concentric search offsets
    offsets = [0.0, -0.1, 0.1, -0.3, 0.3, -0.8, 0.8, -1.5, 1.5, -2.5, 2.5]
    
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
                        if params.get('debug'):
                            print(f"    🌟 BRACKET SECURED: [{bracket[0]:.4f}, {bracket[1]:.4f}]")
                        break
        if bracket:
            if params.get('debug'):
                print(f"  [Solver] ✅ Root securely bracketed between {bracket[0]:.3f} and {bracket[1]:.3f}!")
            break

    # --- C. Fallback Global Grid ---
    if not bracket:
        if params.get('debug'):
            print("  [Solver] Concentric search failed. Launching ultra-wide global fallback grid...")
        
        # SPEEDUP: Reduced from 25 points to 15 points
        global_pts = np.linspace(min_pc, max_pc, 15)
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

    if not bracket:
        print(f"  ❌ [Solver] FATAL: Could not bracket the root! Planet is physically impossible.")
        print(f"  ❌ Evaluated [logPc, Error] pairs: {[(round(p, 2), f'{err:.2e}') for p, err in valid_evals]}")
        return None

    # =========================================================================
    # 5. Final Convergence (Brent's Method)
    # =========================================================================
    try:
        if bracket:
            # Brentq is super fast because the bracket bounds are loaded from the eval_cache!
            root = brentq(objective, bracket[0], bracket[1], xtol=1e-7)
            
            if is_water_world:
                return physics.integrate_water_world(root, params, eos_data)
            return physics.integrate_planet(root, params, eos_data)
            
    except Exception as e:
        if params.get('debug'):
            print(f"  [Solver] Root finding failed: {e}")
        return None