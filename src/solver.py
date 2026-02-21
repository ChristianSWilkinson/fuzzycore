import numpy as np
import pandas as pd
import os
from scipy.optimize import brentq
from . import physics, eos
from . import constants as c

def solve_structure(target_val, params, mode, trial_id, csv_file, write_lock):
    """
    Solves for planetary structure.
    Dynamically switches between Gas Giant and Water World models.
    """
    # 1. Setup EOS data
    rock = eos.get_rock_interpolator() # Now guaranteed to exist
    fluid = eos.generate_fluid_interpolators(params['z_profile'])
    
    eos_data = {'rock': rock, 'fluid': fluid}
    
    # If using Direct Water EOS
    if 'M_water' in params and params['M_water'] > 0:
        water = eos.get_water_interpolators_complete()
        eos_data['water'] = water
    
    params['target_m'] = target_val
    idx = trial_id 
    
    # Detect Mode
    is_water_world = 'M_water' in params and params['M_water'] > 0
    
    def objective(logPc):
        try:
            if is_water_world:
                res = physics.integrate_water_world(logPc, params, eos_data)
                interior_mass = params['M_rock'] + params['M_water']
            else:
                res = physics.integrate_planet(logPc, params, eos_data)
                interior_mass = params['M_core']
            
            # --- FAIL CRITERIA ---
            if res is None or np.isnan(res['M'][-1]):
                if params.get('debug'): print(f"  [Solver] logPc {logPc:.2f}: Integration returned None")
                return -1e30 
            
            # Check if total mass is at least the interior mass (physics validity)
            if res['M'][-1] < (interior_mass * 0.99):
                 if params.get('debug'): print(f"  [Solver] logPc {logPc:.2f}: Mass too low ({res['M'][-1]/c.M_EARTH:.2f} Me)")
                 return -1e30

            actual_m = res['M'][-1]
            actual_r = res['R'][-1]
            
            # 3. SAVE INTERMEDIATE STEPS
            intermediate_output = {
                'trial_id': f"{idx}_step",
                'target_mode': mode,
                'target_value': target_val,
                'P_surf_bar': params['P_surf'],
                'T_surf_K': params['T_surf'],
                'M_total_Mj': actual_m / c.M_JUPITER,
                'R_total_Rj': actual_r / c.R_JUPITER,
                'M_Z_total_Me': res['M_Z_total'] / c.M_EARTH,
                'P_center_bar': 10**logPc,
                'Iron_Fraction': params.get('iron_fraction', 0.0), # Log Iron Fraction
                'status': 'success_intermediate' 
            }
            
            with write_lock:
                pd.DataFrame([intermediate_output]).to_csv(
                    csv_file, mode='a', header=not os.path.exists(csv_file), index=False
                )
            
            # 4. TARGET SELECTION
            if mode == 'gravity':
                g_surf = (c.G_CONST * actual_m) / actual_r**2
                return g_surf - target_val
            elif mode == 'mass':
                if params.get('debug', False):
                    diff = actual_m - target_val
                    print(f"  [Debug] {idx} logPc: {logPc:.2f} -> Mass: {actual_m/c.M_JUPITER:.3f} Mj")  
                return actual_m - target_val
        except Exception as e:
            if params.get('debug'): print(f"  [Solver] Error: {e}")
            return 1e30

    # --- WARM START & SCANNING LOGIC ---
    guess = params.get('initial_log_pc', None)
    bracket = None

    if guess is not None:
        step = 0.5 
        test_points = [guess - step, guess + step]
        vals = [objective(tp) for tp in test_points]
        
        loop_iter = 0
        while np.sign(vals[0]) == np.sign(vals[1]):
            loop_iter += 1
            if vals[1] < 0:
                test_points = [test_points[1], test_points[1] + step]
            else:
                test_points = [test_points[0] - step, test_points[0]]
            
            test_points = [max(6.0, test_points[0]), min(14.0, test_points[1])]
            vals = [objective(tp) for tp in test_points]
            
            if loop_iter > 15 or (test_points[0] <= 6.0 and test_points[1] >= 14.0):
                break
        else:
            bracket = (test_points[0], test_points[1])

    if bracket is None:
        print(f"Trial {idx}: Global scan initiated")
        grid = np.linspace(6.5, 13.5, 15)
        vals = [objective(p) for p in grid]
        for i in range(len(vals)-1):
            if np.sign(vals[i]) != np.sign(vals[i+1]):
                bracket = (grid[i], grid[i+1])
                break
    
    # 5. FINAL CONVERGENCE
    try:
        if bracket:
            root = brentq(objective, bracket[0], bracket[1], xtol=1e-4)
            if is_water_world:
                return physics.integrate_water_world(root, params, eos_data)
            return physics.integrate_planet(root, params, eos_data)
        else:
            return None
    except Exception:
        return None