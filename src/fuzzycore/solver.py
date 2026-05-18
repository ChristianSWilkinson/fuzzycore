"""
Planetary Structure Solver

This module handles the root-finding and parameter-sweeping logic required
to converge on a physically consistent planetary interior. It dynamically 
switches between different integration architectures (e.g., Gas Giants vs. 
Water Worlds) and tracks intermediate solutions during the solving process.
"""

import os
import logging

import numpy as np
from scipy.optimize import brentq

from . import constants as c
from . import eos
from . import physics
from .utils import time_it

# Module-level cache that persists across solve_structure calls within one process.
# Key: (m_core_kg_rounded, logPc_rounded, sigma_bin, target_val_rounded)
# Value: error (float)
_OBJECTIVE_CACHE: dict = {}

def _objective_cache_key(params: dict, log_pc: float, target_val: float):
    m_core = params.get('M_core', params.get('M_rock', 0.0))
    sigma  = params.get('sigma_val', 0.0)
    p_surf = params.get('P_surf', 1.0)
    t_surf = params.get('T_surf', 200.0)
    y_rat  = params.get('Y_ratio', 0.26)

    # Profile fingerprint: bytes-hash of the rounded array. Two profiles that
    # round identically to 4 decimals will share interpolators in the EOS
    # stack anyway, so they're cache-equivalent. Anything else collides only
    # if it's genuinely the same physics.
    z_profile = params.get('z_profile')
    if z_profile is not None:
        z_fp = hash(np.round(np.asarray(z_profile, dtype=float), 4).tobytes())
    else:
        z_fp = 0

    return (
        round(float(m_core),     6),
        round(float(log_pc),     4),
        round(float(sigma),      3),
        round(float(target_val), 6),
        round(float(p_surf),     6),
        round(float(t_surf),     3),
        round(float(y_rat),      4),
        z_fp,
    )

def clear_objective_cache() -> None:
    """Call between independent planet targets (e.g. new M_KEPLER11E, new track)."""
    _OBJECTIVE_CACHE.clear()

@time_it
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

    @time_it
    def objective(log_pc: float) -> float:
        log_pc_rounded = round(float(log_pc), 12)
        if log_pc_rounded in eval_cache:
            return eval_cache[log_pc_rounded]

        # Cross-call cache: hit when the SAME (m_core, logPc, sigma_bin, target) was seen
        # in a previous solve_structure call (typically: failure points from prior sigma probes).
        module_key = _objective_cache_key(params, log_pc, target_val)
        if module_key in _OBJECTIVE_CACHE:
            cached = _OBJECTIVE_CACHE[module_key]
            eval_cache[log_pc_rounded] = cached
            return cached

        if params.get('debug'):
            print(f"\n    [Objective Attempt] logPc: {log_pc:.4f} (Pc: {10**log_pc:.2e} bar)")

        try:
            if is_water_world:
                res = physics.integrate_water_world(log_pc, params, eos_data)
                interior_mass = params['M_rock'] + params['M_water']
            else:
                res = physics.integrate_planet(log_pc, params, eos_data)
                interior_mass = params['M_core']

            if res is None or np.isnan(res['M'][-1]):
                error = 1e30
                if params.get('debug'):
                    print(f"      ❌ FAILURE: Integration returned None | Synthetic Error: {error:.2e}")
            elif res['M'][-1] < (interior_mass * 0.99):
                error = -1e30
                if params.get('debug'):
                    print(f"      ❌ FAILURE: Integration Prematurely Stalled | Synthetic Error: {error:.2e}")
            else:
                actual_m = res['M'][-1]
                actual_r = res['R'][-1]
                if mode == 'gravity':
                    g_surf = (c.G_CONST * actual_m) / (actual_r ** 2)
                    error = g_surf - target_val
                    if params.get('debug'):
                        print(f"      ✅ SUCCESS: Mass: {actual_m/c.M_EARTH:.3f} Me | g_surf: {g_surf:.2f} m/s² | Err: {error:+.3f} m/s²")
                elif mode == 'mass':
                    error = actual_m - target_val
                    if params.get('debug'):
                        print(f"      ✅ SUCCESS: Mass: {actual_m/c.M_EARTH:.3f} Me | Err: {error/c.M_EARTH:+.3f} Me")

            eval_cache[log_pc_rounded] = error
            _OBJECTIVE_CACHE[module_key] = error            # <-- write-through
            return error

        except Exception as e:
            if params.get('debug'):
                print(f"      💥 CRASH in Objective: {str(e)}")
            eval_cache[log_pc_rounded] = -1e20
            _OBJECTIVE_CACHE[module_key] = -1e20            # <-- write-through
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
    # ─────────────────────────────────────────────────────────────────────────
    # Cliff bisection: zoom in on the fail/success transition, then check
    # whether the cliff itself brackets the target.
    # ─────────────────────────────────────────────────────────────────────────
    if not bracket:
        fail_pc = max((p for p, e in eval_cache.items() if e > 1e29),  default=None)
        succ_pc = min((p for p, e in eval_cache.items() if abs(e) < 1e29), default=None)

        if fail_pc is not None and succ_pc is not None and succ_pc > fail_pc:
            if params.get('debug'):
                print(f"  [Solver] Bisecting cliff between fail={fail_pc:.3f} and succ={succ_pc:.3f}...")

            while succ_pc - fail_pc > 0.001:
                mid = 0.5 * (fail_pc + succ_pc)
                err = objective(mid)
                if abs(err) < 1e29:
                    succ_pc = mid
                else:
                    fail_pc = mid

            # Rebuild valid_evals from the cache so we see ALL successful points,
            # including everything bisection just added.
            valid_evals = sorted(
                (p, e) for p, e in eval_cache.items() if abs(e) < 1e29
            )

            # Now look for a sign change anywhere in the enriched evaluation set
            for i in range(len(valid_evals) - 1):
                if np.sign(valid_evals[i][1]) != np.sign(valid_evals[i + 1][1]):
                    bracket = (valid_evals[i][0], valid_evals[i + 1][0])
                    if params.get('debug'):
                        print(f"  [Solver] ✅ Bracket found post-bisection: "
                            f"[{bracket[0]:.4f}, {bracket[1]:.4f}]")
                    break

            # If still no bracket and the lowest successful Pc has positive error,
            # the target is genuinely below the integrator's reachable mass floor.
            if not bracket:
                min_p, min_err = valid_evals[0]
                if min_err > 0:
                    print(f"  [Solver] Target unreachable: minimum achievable mass at "
                        f"this composition is {(target_val + min_err)/c.M_EARTH:.2f} Me, "
                        f"target was {target_val/c.M_EARTH:.2f} Me.")
                    return None

    if not bracket:
        print(f"  ❌ [Solver] FATAL: Could not bracket the root! Planet is physically impossible.")
        print(f"  ❌ Evaluated [logPc, Error] pairs: {[(round(p, 2), f'{err:.2e}') for p, err in valid_evals]}")
        return None

    # =========================================================================
    # 5. Final Convergence (Brent's Method)
    # =========================================================================
    try:
        root = brentq(objective, bracket[0], bracket[1], xtol=1e-5)

        if is_water_world:
            final_result = physics.integrate_water_world(root, params, eos_data)
        else:
            final_result = physics.integrate_planet(root, params, eos_data)

        if final_result is None:
            return None

        achieved_mass = final_result['M'][-1]
        achieved_r = final_result['R'][-1]

        # --- BUG FIX: Check convergence against the CORRECT target mode ---
        if mode == 'mass':
            rel_err = abs(achieved_mass - target_val) / target_val

            if rel_err > 0.05:
                logging.warning(
                    f"Solver converged to wrong mass: {achieved_mass/c.M_EARTH:.2f} "
                    f"vs {target_val/c.M_EARTH:.2f} Mₑ"
                )
                return None

            if rel_err > 1e-3:
                logging.warning(
                    f"Soft mass disagreement: brentq root gave {achieved_mass/c.M_EARTH:.3f} Mₑ "
                    f"vs target {target_val/c.M_EARTH:.3f} Mₑ (rel_err={rel_err:.2e}). "
                    f"Consider clear_objective_cache() if profile changed."
                )

        elif mode == 'gravity':
            achieved_g = (c.G_CONST * achieved_mass) / (achieved_r ** 2)
            rel_err = abs(achieved_g - target_val) / target_val

            if rel_err > 0.05:
                logging.warning(
                    f"Solver converged to wrong gravity: {achieved_g:.2f} "
                    f"vs {target_val:.2f} m/s²"
                )
                return None

            if rel_err > 1e-3:
                logging.warning(
                    f"Soft gravity disagreement: brentq root gave {achieved_g:.2f} m/s² "
                    f"vs target {target_val:.2f} m/s² (rel_err={rel_err:.2e})."
                )

        return final_result

    except Exception as e:
        if params.get('debug'):
            print(f"  [Solver] Root finding failed: {e}")
        return None