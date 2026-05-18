"""
Core Temperature Gradient Sensitivity — Gas Giants
==================================================
Sweeps the dimensionless core temperature gradient across three planetary masses.
The envelope composition is forced to a uniform Z_base (equivalent to a Gaussian
with sigma=0) to guarantee a purely adiabatic envelope, perfectly isolating the
structural impact of the core.
"""

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import fuzzycore.constants as c
import fuzzycore.solver as solver

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIGURATIONS = [
    dict(
        name           = '1.0_Mjup',
        M_total_Me     = 317.8,
        M_core_Me      = 10.0,
        T_surf_K       = 165.0,
        P_surf_bar     = 1.0,
        T_int_K        = 100.0,
        Z_base         = 0.0816,
        iron_fraction  = 0.33,
        Y_ratio        = 0.26,
        z_profile_path = None,  # Falls back to uniform/adiabatic
        initial_log_pc = 7.5,
    ),
    dict(
        name           = '0.5_Mjup',
        M_total_Me     = 158.9,      
        M_core_Me      = 10.0,
        T_surf_K       = 165.0,
        P_surf_bar     = 1.0,
        T_int_K        = 100.0,
        Z_base         = 0.0816,
        iron_fraction  = 0.33,
        Y_ratio        = 0.26,
        z_profile_path = None,  
        initial_log_pc = 7.5,
    ),
    dict(
        name           = '0.1_Mjup',
        M_total_Me     = 31.78,      
        M_core_Me      = 10.0,
        T_surf_K       = 165.0,
        P_surf_bar     = 1.0,
        T_int_K        = 100.0,
        Z_base         = 0.0816,
        iron_fraction  = 0.33,
        Y_ratio        = 0.26,
        z_profile_path = None,  
        initial_log_pc = 7.5,
    ),
]

CORE_NABLA_VALUES = np.array([
    0.000, 0.025, 0.050, 0.075, 0.100, 
    0.150, 0.200, 0.250, 0.300, 0.350, 0.400,
])

RESULTS_FILE      = "../data/core_nabla_giants.csv"
N_WORKERS         = max(1, mp.cpu_count() - 1)
SOLVE_TIMEOUT_SEC = 600
N_LAYERS          = 50

ROW_COLUMNS = [
    'config_name', 'core_nabla', 'M_total_Me_target', 'M_core_Me',
    'T_surf_K', 'R_total_Re', 'R_rock_Re', 'M_total_Me_achieved',
    'P_center_GPa', 'T_center_K', 'Status',
]

def _empty_row(cfg, nabla, status):
    row = {col: np.nan for col in ROW_COLUMNS}
    row['config_name']       = cfg['name']
    row['core_nabla']        = nabla
    row['M_total_Me_target'] = cfg['M_total_Me']
    row['M_core_Me']         = cfg['M_core_Me']
    row['T_surf_K']          = cfg['T_surf_K']
    row['Status']            = status
    return row

def _load_z_profile(cfg):
    """
    Returns a uniform composition array representing a fully mixed, 
    purely adiabatic envelope (equivalent to sigma = 0).
    """
    return np.full(N_LAYERS, cfg['Z_base'], dtype=float)

def run_one(task):
    cfg, nabla = task
    trial_id = f"NABLA_{cfg['name']}_{nabla:.4f}"

    try:
        solver.clear_objective_cache()
    except AttributeError: pass

    try:
        z_prof = _load_z_profile(cfg)

        params = {
            'M_core':         cfg['M_core_Me'] * c.M_EARTH,
            'M_rock':         cfg['M_core_Me'] * c.M_EARTH,
            'M_water':        0.0,                    
            'P_surf':         cfg['P_surf_bar'],
            'T_surf':         cfg['T_surf_K'],
            'T_int':          cfg['T_int_K'],
            'z_base':         cfg['Z_base'],
            'z_profile':      z_prof,
            'sigma_val':      0.0,
            'iron_fraction':  cfg['iron_fraction'],
            'Y_ratio':        cfg['Y_ratio'],
            'core_nabla':     float(nabla),           
            'debug':          False,
            'initial_log_pc': cfg['initial_log_pc'],
        }

        target_mass = cfg['M_total_Me'] * c.M_EARTH
        res = solver.solve_structure(target_mass, params, 'mass', trial_id)

        if res is None: return False, _empty_row(cfg, nabla, 'solver_returned_none')

        try:
            r_tot_re = float(res['R'][-1]) / c.R_EARTH
            m_tot_me = float(res['M'][-1]) / c.M_EARTH
        except Exception as e:
            return False, _empty_row(cfg, nabla, f'malformed_result_{type(e).__name__}')

        mass_err = (m_tot_me - cfg['M_total_Me']) / cfg['M_total_Me']
        if not np.isfinite(mass_err) or abs(mass_err) > 0.05:
            row = _empty_row(cfg, nabla, f'mass_mismatch_{mass_err:+.3f}')
            row['R_total_Re']          = r_tot_re
            row['M_total_Me_achieved'] = m_tot_me
            return False, row

        row = _empty_row(cfg, nabla, 'ok')
        row['R_total_Re']          = r_tot_re
        row['M_total_Me_achieved'] = m_tot_me

        if res.get('R_rock', None) is not None and np.isfinite(res['R_rock']):
            row['R_rock_Re'] = float(res['R_rock']) / c.R_EARTH

        try:
            row['P_center_GPa'] = 10 ** float(res['P'][0]) * 1e-4
            row['T_center_K']   = float(res['T'][0])
        except Exception: pass

        return True, row

    except Exception as e:
        return False, _empty_row(cfg, nabla, f'exception_{type(e).__name__}')

def _append_row(row, file_path, write_header):
    pd.DataFrame([row], columns=ROW_COLUMNS).to_csv(file_path, mode='a', header=write_header, index=False)

def build_tasks():
    return [(cfg, float(n)) for cfg in CONFIGURATIONS for n in CORE_NABLA_VALUES]

def main():
    print("=" * 72)
    print(" Core Temperature Gradient Sensitivity Test (Adiabatic Envelope)")
    print("=" * 72)
    out_dir = os.path.dirname(RESULTS_FILE)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    todo = build_tasks()
    # Ignoring the resume logic so it forces a fresh sweep with the new adiabatic profiles
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        print("[*] Cleared old results file for fresh sweep.")

    write_header = True
    n_done, n_failed = 0, 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = {exe.submit(run_one, t): t for t in todo}
        for fut in as_completed(futures):
            cfg, nabla = futures[fut]
            try:
                success, row = fut.result(timeout=SOLVE_TIMEOUT_SEC)
            except Exception as e:
                success, row = False, _empty_row(cfg, nabla, f'driver_{type(e).__name__}')

            _append_row(row, RESULTS_FILE, write_header)
            write_header = False
            n_done += 1
            if not success: n_failed += 1

            tag = f"R = {row['R_total_Re']:.4f} R_E, T_c = {row['T_center_K']:.0f} K" if success else f"FAIL [{row['Status']}]"
            print(f"  [{n_done:3d}/{len(todo)}]  {cfg['name']:>10s}  nabla = {nabla:.3f}  ->  {tag}")

if __name__ == '__main__':
    main()