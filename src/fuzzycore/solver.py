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
    if not is_water_world:
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
    # 3. Objective Function for Root Finding
    # =========================================================================
    
    def objective(log_pc: float) -> float:
        """
        The objective function evaluated by the Brent root-finder.
        Integrates the planet for a guessed central pressure (log_pc) 
        and returns the error relative to the target mass/gravity.
        """
        try:
            # Route to the correct physical integrator
            if is_water_world:
                res = physics.integrate_water_world(log_pc, params, eos_data)
                interior_mass = params['M_rock'] + params['M_water']
            else:
                res = physics.integrate_planet(log_pc, params, eos_data)
                interior_mass = params['M_core']
            
            # --- FAIL CRITERIA ---
            # Reject runs that failed to integrate or returned NaNs
            if res is None or np.isnan(res['M'][-1]):
                if params.get('debug'):
                    print(f"  [Solver] logPc {log_pc:.2f}: Integration returned None")
                return -1e30 
            
            # Reject physically invalid runs (e.g., planet mass is less than core mass)
            if res['M'][-1] < (interior_mass * 0.99):
                if params.get('debug'):
                    actual_m_earth = res['M'][-1] / c.M_EARTH
                    print(f"  [Solver] logPc {log_pc:.2f}: Mass too low ({actual_m_earth:.2f} Me)")
                return -1e30

            actual_m = res['M'][-1]
            actual_r = res['R'][-1]
            
            # Save intermediate solver steps for tracking / debugging
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
            
            # Use lock to safely write to the shared CSV in parallel
            with write_lock:
                df_out = pd.DataFrame([intermediate_output])
                file_exists = os.path.exists(csv_file)
                df_out.to_csv(csv_file, mode='a', header=not file_exists, index=False)
            
            # Calculate the final error metric to return to the root-finder
            if mode == 'gravity':
                g_surf = (c.G_CONST * actual_m) / (actual_r ** 2)
                return g_surf - target_val
            elif mode == 'mass':
                if params.get('debug', False):
                    actual_m_jup = actual_m / c.M_JUPITER
                    print(f"  [Debug] {trial_id} logPc: {log_pc:.2f} -> Mass: {actual_m_jup:.3f} Mj")  
                return actual_m - target_val
                
        except Exception as e:
            if params.get('debug'):
                print(f"  [Solver] Error: {e}")
            return 1e30

    # =========================================================================
    # 4. Warm Start & Bracketing Logic
    # =========================================================================
    
    guess = params.get('initial_log_pc', None)
    bracket = None

    # If an initial guess is provided, attempt to build a bracket around it
    if guess is not None:
        step = 0.5 
        test_points = [guess - step, guess + step]
        vals = [objective(tp) for tp in test_points]
        
        loop_iter = 0
        
        # Expand the bracket until the errors have opposite signs (root is bounded)
        while np.sign(vals[0]) == np.sign(vals[1]):
            loop_iter += 1
            
            if vals[1] < 0:
                test_points = [test_points[1], test_points[1] + step]
            else:
                test_points = [test_points[0] - step, test_points[0]]
            
            # Enforce physical bounds for central pressure [10^6 to 10^14 bar]
            test_points = [max(6.0, test_points[0]), min(14.0, test_points[1])]
            vals = [objective(tp) for tp in test_points]
            
            # Abort expansion if it gets stuck or hits the absolute physical limits
            if loop_iter > 15 or (test_points[0] <= 6.0 and test_points[1] >= 14.0):
                break
        else:
            # The 'else' block executes if the 'while' condition becomes False naturally
            bracket = (test_points[0], test_points[1])

    # Fallback: If no warm start or the bracket expansion failed, run a global scan
    if bracket is None:
        print(f"Trial {trial_id}: Global scan initiated")
        grid = np.linspace(6.5, 13.5, 15)
        vals = [objective(p) for p in grid]
        
        # Search the grid for a sign change indicating a root
        for i in range(len(vals) - 1):
            if np.sign(vals[i]) != np.sign(vals[i + 1]):
                bracket = (grid[i], grid[i + 1])
                break
    
    # =========================================================================
    # 5. Final Convergence (Brent's Method)
    # =========================================================================
    
    try:
        if bracket:
            # Solve for the exact log(Pc) that zeroes the objective function
            root = brentq(objective, bracket[0], bracket[1], xtol=1e-4)
            
            # Run one final, clean integration using the converged root
            if is_water_world:
                return physics.integrate_water_world(root, params, eos_data)
            return physics.integrate_planet(root, params, eos_data)
        else:
            return None
            
    except Exception as e:
        if params.get('debug'):
            print(f"  [Solver] Root finding failed: {e}")
        return None