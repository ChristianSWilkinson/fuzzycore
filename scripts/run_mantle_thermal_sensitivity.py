"""
Mantle Thermal Profile Sensitivity — Water Worlds
==================================================

Addresses referee comment #7 for the volatile mantle: water is much less
degenerate than rock at sub-Neptune mantle pressures, so the thermal profile
of the mantle could plausibly move the converged radius by more than the
rocky-core temperature gradient does.

We test three thermal modes (controlled by `params['mantle_thermal_mode']`):

    'adiabatic'   — entropy-matched to envelope (historical default).
                    Mantle T(P) follows the water-EOS isentrope from the
                    envelope-mantle interface.
    'isothermal'  — T_mantle(P) = T_int throughout the mantle.
                    Extreme cold-mantle limit. Models a mantle thermally
                    pinned to the interface but with no internal heating.
    'polytropic'  — T_mantle(P) = T_int * (P / P_int) ** mantle_nabla.
                    Single-parameter sweep. mantle_nabla = 0 reproduces
                    isothermal; values approaching the local adiabatic
                    index of water recover the convective limit.

Run plan:
   (1) one adiabatic baseline
   (2) one isothermal endpoint
   (3) seven polytropic points sweeping mantle_nabla in [0, 0.30]
       (the polytropic family brackets isothermal at 0 and a strong
       super-adiabatic at 0.30)

PREREQUISITE: physics.py must read `mantle_thermal_mode` and `mantle_nabla`
from `params` (see the corresponding patch).

OUTPUT: CSV with one row per (configuration, thermal_mode, mantle_nabla).
Final console block prints headline statistics:
  - relative R variation across all modes
  - mantle-radius (R_water - R_rock) variation
  - whether the result is conclusive (<1% R variation) or borderline.
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
        name           = 'gj1214b_like',
        M_total_Me     = 8.17,
        M_rock_Me      = 6.90,
        M_water_Me     = 1.00,
        T_surf_K       = 550.0,
        P_surf_bar     = 1.0,
        T_int_K        = 200.0,
        Z_base         = 0.02,
        Z_core         = 0.99,
        sigma_gauss    = 0.25,
        iron_fraction  = 0.33,
        Y_ratio        = 0.26,
        initial_log_pc = 7.0,
    ),
    dict(
        name           = 'gj1214b_like_more_water',
        M_total_Me     = 8.17,
        M_rock_Me      = 5.90,
        M_water_Me     = 2.0,
        T_surf_K       = 550.0,
        P_surf_bar     = 1.0,
        T_int_K        = 200.0,
        Z_base         = 0.02,
        Z_core         = 0.99,
        sigma_gauss    = 0.20,
        iron_fraction  = 0.33,
        Y_ratio        = 0.26,
        initial_log_pc = 7.0,
    ),
]

# Thermal-mode sweep. Each entry is (mode_label, mantle_thermal_mode, mantle_nabla).
# Polytropic spans the physical range from cold (n=0, ~isothermal) through
# typical adiabatic-like values (n~0.2-0.3).
THERMAL_SWEEP = [
    ('adiabatic',     'adiabatic',  0.00),    # default; mantle_nabla irrelevant
    ('isothermal',    'isothermal', 0.00),    # cold endpoint; mantle_nabla irrelevant
    ('polytropic_00', 'polytropic', 0.00),    # sanity: should match isothermal
    ('polytropic_05', 'polytropic', 0.05),
    ('polytropic_10', 'polytropic', 0.10),
    ('polytropic_15', 'polytropic', 0.15),
    ('polytropic_20', 'polytropic', 0.20),
    ('polytropic_25', 'polytropic', 0.25),
    ('polytropic_30', 'polytropic', 0.30),
]

# Fix core_nabla so the comparison isolates the mantle treatment.
# 0.1 is the historical water-world default; you can sweep this in a second pass.
FIXED_CORE_NABLA = 0.1

RESULTS_FILE      = "../data/mantle_thermal_sensitivity.csv"
N_WORKERS         = max(1, mp.cpu_count() - 1)
SOLVE_TIMEOUT_SEC = 600
N_LAYERS          = 50


# =============================================================================
# ROW SCHEMA
# =============================================================================

ROW_COLUMNS = [
    'config_name',
    'mode_label',
    'mantle_thermal_mode',
    'mantle_nabla',
    'core_nabla',
    'M_total_Me_target',
    'M_rock_Me',
    'M_water_Me',
    'T_surf_K',
    'R_total_Re',
    'R_rock_Re',
    'R_water_Re',                 # outer edge of the water mantle (if reported)
    'R_mantle_thickness_Re',      # R_water - R_rock
    'M_total_Me_achieved',
    'P_center_GPa',
    'T_center_K',
    'T_cmb_K',                    # temperature at top of rocky core
    'Status',
]


def _empty_row(cfg, mode_label, mode, nabla, status):
    row = {col: np.nan for col in ROW_COLUMNS}
    row['config_name']         = cfg['name']
    row['mode_label']          = mode_label
    row['mantle_thermal_mode'] = mode
    row['mantle_nabla']        = nabla
    row['core_nabla']          = FIXED_CORE_NABLA
    row['M_total_Me_target']   = cfg['M_total_Me']
    row['M_rock_Me']           = cfg['M_rock_Me']
    row['M_water_Me']          = cfg['M_water_Me']
    row['T_surf_K']            = cfg['T_surf_K']
    row['Status']              = status
    return row


# =============================================================================
# Z PROFILE
# =============================================================================

def _build_gaussian_profile(cfg):
    x = np.linspace(0, 1, N_LAYERS)
    sigma  = cfg['sigma_gauss']
    z_base = cfg['Z_base']
    z_core = cfg['Z_core']
    dx = 1.0 / max(1, N_LAYERS - 1)
    gauss_area = sigma * np.sqrt(np.pi / 2.0)
    amp = min(1.0, gauss_area / dx)
    dyn_core = z_base + (z_core - z_base) * amp
    raw = np.exp(-((x - 1.0) ** 2) / (2 * sigma ** 2))
    return np.clip(z_base + (dyn_core - z_base) * raw, 0.0, 0.99)


# =============================================================================
# WORKER
# =============================================================================

def run_one(task):
    cfg, mode_label, mode, mantle_nabla = task
    trial_id = f"MANTLE_{cfg['name']}_{mode_label}"

    try:
        solver.clear_objective_cache()
    except AttributeError:
        pass

    try:
        z_prof = _build_gaussian_profile(cfg)

        params = {
            'M_core':              cfg['M_rock_Me']  * c.M_EARTH,
            'M_rock':              cfg['M_rock_Me']  * c.M_EARTH,
            'M_water':             cfg['M_water_Me'] * c.M_EARTH,
            'P_surf':              cfg['P_surf_bar'],
            'T_surf':              cfg['T_surf_K'],
            'T_int':               cfg['T_int_K'],
            'z_base':              cfg['Z_base'],
            'z_profile':           z_prof,
            'sigma_val':           cfg['sigma_gauss'],
            'iron_fraction':       cfg['iron_fraction'],
            'Y_ratio':             cfg['Y_ratio'],
            'core_nabla':          FIXED_CORE_NABLA,
            'mantle_thermal_mode': mode,            # *** test variable 1 ***
            'mantle_nabla':        mantle_nabla,    # *** test variable 2 ***
            'debug':               False,
            'initial_log_pc':      cfg['initial_log_pc'],
        }

        target_mass = cfg['M_total_Me'] * c.M_EARTH
        res = solver.solve_structure(target_mass, params, 'mass', trial_id)

        if res is None:
            return False, _empty_row(cfg, mode_label, mode, mantle_nabla, 'solver_returned_none')

        try:
            r_tot_re = float(res['R'][-1]) / c.R_EARTH
            m_tot_me = float(res['M'][-1]) / c.M_EARTH
        except Exception as e:
            return False, _empty_row(cfg, mode_label, mode, mantle_nabla, f'malformed_result_{type(e).__name__}')

        mass_err = (m_tot_me - cfg['M_total_Me']) / cfg['M_total_Me']
        if not np.isfinite(mass_err) or abs(mass_err) > 0.05:
            row = _empty_row(cfg, mode_label, mode, mantle_nabla, f'mass_mismatch_{mass_err:+.3f}')
            row['R_total_Re']          = r_tot_re
            row['M_total_Me_achieved'] = m_tot_me
            return False, row

        row = _empty_row(cfg, mode_label, mode, mantle_nabla, 'ok')
        row['R_total_Re']          = r_tot_re
        row['M_total_Me_achieved'] = m_tot_me

        r_rock_m  = res.get('R_rock', None)
        r_water_m = res.get('R_int',  None)
        r_rock_re  = (float(r_rock_m)  / c.R_EARTH) if r_rock_m  is not None and np.isfinite(r_rock_m)  else np.nan
        r_water_re = (float(r_water_m) / c.R_EARTH) if r_water_m is not None and np.isfinite(r_water_m) else np.nan
        row['R_rock_Re']            = r_rock_re
        row['R_water_Re']           = r_water_re
        if np.isfinite(r_rock_re) and np.isfinite(r_water_re):
            row['R_mantle_thickness_Re'] = r_water_re - r_rock_re

        try:
            row['P_center_GPa'] = 10 ** float(res['P'][0]) * 1e-4
            row['T_center_K']   = float(res['T'][0])
        except Exception:
            pass

        # T at top of rocky core: read from the integrated profile if possible.
        # Look for the index just above r_rock.
        try:
            R_arr = np.asarray(res['R']) / c.R_EARTH
            T_arr = np.asarray(res['T'])
            if np.isfinite(r_rock_re):
                # The first T value above R_rock is T_CMB
                mask = R_arr > r_rock_re * 0.999
                if mask.any():
                    idx = int(np.argmax(mask))
                    row['T_cmb_K'] = float(T_arr[idx])
        except Exception:
            pass

        return True, row

    except Exception as e:
        return False, _empty_row(cfg, mode_label, mode, mantle_nabla, f'exception_{type(e).__name__}')


# =============================================================================
# DRIVER
# =============================================================================

def _append_row(row, file_path, write_header):
    pd.DataFrame([row], columns=ROW_COLUMNS).to_csv(
        file_path, mode='a', header=write_header, index=False)


def _resume_keys(file_path):
    if not os.path.exists(file_path):
        return set()
    try:
        df = pd.read_csv(file_path)
        return {(r['config_name'], str(r['mode_label']))
                for _, r in df.iterrows()}
    except Exception:
        return set()


def build_tasks():
    return [(cfg, label, mode, nabla)
            for cfg in CONFIGURATIONS
            for (label, mode, nabla) in THERMAL_SWEEP]


def print_conclusive_summary(file_path):
    if not os.path.exists(file_path):
        return
    df = pd.read_csv(file_path)
    df_ok = df[df['Status'] == 'ok']

    print("\n" + "=" * 78)
    print(" CONCLUSIVE SUMMARY — mantle thermal profile sensitivity (water worlds)")
    print("=" * 78)

    for cfg_name in df_ok['config_name'].unique():
        sub = df_ok[df_ok['config_name'] == cfg_name].copy()
        if sub.empty:
            continue
        # Sort with adiabatic first, then isothermal, then polytropic by nabla
        order = {'adiabatic': -1, 'isothermal': 0}
        sub['_sort'] = [order.get(m, 1) + 0.001 * n
                        for m, n in zip(sub['mantle_thermal_mode'], sub['mantle_nabla'])]
        sub = sub.sort_values('_sort')

        R = sub['R_total_Re'].values
        Rrock = sub['R_rock_Re'].values
        Rmantle = sub['R_mantle_thickness_Re'].values
        T_cmb = sub['T_cmb_K'].values
        T_c   = sub['T_center_K'].values

        R_rel_max     = 100 * (np.nanmax(R) - np.nanmin(R)) / np.nanmean(R)
        mant_rel_max  = (100 * (np.nanmax(Rmantle) - np.nanmin(Rmantle)) / np.nanmean(Rmantle)
                         if np.isfinite(Rmantle).any() else np.nan)
        T_cmb_contrast = (np.nanmax(T_cmb) / np.nanmin(T_cmb)
                          if np.isfinite(T_cmb).all() and np.nanmin(T_cmb) > 0 else np.nan)

        print(f"\n  [{cfg_name}]   modes tested: {len(sub)}")
        print(f"  R_total range:                     "
              f"[{np.nanmin(R):.4f}, {np.nanmax(R):.4f}] R_E  "
              f"-->  Delta R / R = {R_rel_max:.3f} %")
        if np.isfinite(mant_rel_max):
            print(f"  Mantle thickness range:            "
                  f"[{np.nanmin(Rmantle):.4f}, {np.nanmax(Rmantle):.4f}] R_E  "
                  f"-->  Delta / mean = {mant_rel_max:.2f} %")
        if np.isfinite(T_cmb_contrast):
            print(f"  T_cmb (top of rocky core):         "
                  f"factor of {T_cmb_contrast:.2f} contrast across modes")

        # Per-mode table
        print(f"\n  {'mode':>14s}  {'n_mantle':>9s}  {'R [R_E]':>10s}  "
              f"{'R_rock [R_E]':>13s}  {'R_mantle [R_E]':>14s}  {'T_cmb [K]':>10s}")
        for _, row in sub.iterrows():
            print(f"  {row['mode_label']:>14s}  {row['mantle_nabla']:>9.3f}  "
                  f"{row['R_total_Re']:>10.4f}  {row['R_rock_Re']:>13.4f}  "
                  f"{row['R_mantle_thickness_Re']:>14.4f}  "
                  f"{(row['T_cmb_K'] if np.isfinite(row['T_cmb_K']) else 0):>10.0f}")

        # Verdict
        if R_rel_max < 1.0:
            verdict = "CONCLUSIVE: radius is robust against mantle thermal treatment (<1%)"
        elif R_rel_max < 3.0:
            verdict = ("MILD SENSITIVITY: radius varies by 1-3% across mantle thermal modes; "
                       "worth reporting honestly")
        else:
            verdict = ("SIGNIFICANT SENSITIVITY: radius varies by >3% across modes; "
                       "this is a genuine result and a known limitation")
        print(f"\n  Verdict: {verdict}")
    print("=" * 78)


def main():
    print("=" * 78)
    print(" Mantle Thermal Profile Sensitivity Test")
    print("=" * 78)
    print(f"   Configurations:   {[c['name'] for c in CONFIGURATIONS]}")
    print(f"   Thermal modes:    {[t[0] for t in THERMAL_SWEEP]}")
    print(f"   Fixed core_nabla: {FIXED_CORE_NABLA}")
    print(f"   Workers:          {N_WORKERS}")
    print(f"   Output:           {RESULTS_FILE}")
    print("=" * 78)

    out_dir = os.path.dirname(RESULTS_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    all_tasks = build_tasks()
    done = _resume_keys(RESULTS_FILE)
    todo = [t for t in all_tasks if (t[0]['name'], t[1]) not in done]

    print(f"[*] Total tasks:        {len(all_tasks)}")
    print(f"[*] Already completed:  {len(done)}")
    print(f"[*] Remaining to run:   {len(todo)}")

    if not todo:
        print("[*] Nothing to do.")
        print_conclusive_summary(RESULTS_FILE)
        return

    write_header = not os.path.exists(RESULTS_FILE)
    n_done = n_failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = {exe.submit(run_one, t): t for t in todo}

        for fut in as_completed(futures):
            cfg, mode_label, mode, mantle_nabla = futures[fut]
            try:
                success, row = fut.result(timeout=SOLVE_TIMEOUT_SEC)
            except Exception as e:
                success = False
                row = _empty_row(cfg, mode_label, mode, mantle_nabla, f'driver_{type(e).__name__}')

            _append_row(row, RESULTS_FILE, write_header)
            write_header = False

            n_done += 1
            if not success:
                n_failed += 1

            tag = (f"R = {row['R_total_Re']:.4f} R_E,  T_c = {row['T_center_K']:.0f} K"
                   if success else f"FAIL [{row['Status']}]")
            print(f"  [{n_done:3d}/{len(todo)}]  "
                  f"{cfg['name']:>14s}  mode = {mode_label:>14s}  ->  {tag}")

    elapsed = (time.time() - t_start) / 60.0
    print("=" * 78)
    print(f"[*] Finished: {n_done - n_failed}/{n_done} succeeded")
    print(f"[*] Wall time: {elapsed:.1f} min")
    print(f"[*] Results:   {RESULTS_FILE}")

    print_conclusive_summary(RESULTS_FILE)


if __name__ == '__main__':
    main()