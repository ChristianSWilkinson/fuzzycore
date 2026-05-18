"""
Core Temperature Gradient Sensitivity — Water Worlds (Sub-Neptunes)
====================================================================

Addresses referee comment #7 for the water-rich sub-Neptune regime: vary the
dimensionless core temperature gradient

    core_nabla = d ln T / d ln P

across the range spanned by the three physical limits flagged by the referee:

    core_nabla = 0.00      isothermal core
    core_nabla = 0.07      radiative / conductive (electron-degenerate transport)
    core_nabla = 0.30      adiabatic (Schwarzschild-unstable, well-mixed)

Reference configuration: a GJ 1214 b-like water world (M_p = 7.81 M_E,
M_rock = 6.90 M_E, M_water = 1.00 M_E, T_surf = 550 K). The rock core is the
piece whose temperature gradient we're testing; the overlying condensed
water mantle sits on its own EOS-controlled adiabat (treated separately).

Output: CSV with one row per (configuration, core_nabla) point. The final
console block prints the headline numbers for the paper text.

Expected result: the converged total radius varies by <1% across the full
sweep, while the central temperature varies by a factor of 2-3. Together
with the gas-giant test, this demonstrates that fuzzycore's predictions are
robust against the assumed core thermal profile across both sub-Neptune and
giant-planet regimes.

PREREQUISITE: physics.py must read `core_nabla` from `params` (the patch
also updates the water-world path, which used a hardcoded value of 0.1).
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

# --- Canonical water-world configuration ---
CONFIGURATIONS = [
    dict(
        name           = 'gj1214b_like',
        M_total_Me     = 8.17,
        M_rock_Me      = 6.90,         # rocky inner core
        M_water_Me     = 1.00,         # condensed water mantle
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
    # Uncomment to also sweep a more mantle-dominated configuration:
    # dict(
    #     name           = 'water_dominated',
    #     M_total_Me     = 10.0,
    #     M_rock_Me      = 3.0,
    #     M_water_Me     = 6.0,
    #     T_surf_K       = 400.0,
    #     P_surf_bar     = 1.0,
    #     T_int_K        = 200.0,
    #     Z_base         = 0.02,
    #     Z_core         = 0.99,
    #     sigma_gauss    = 0.20,
    #     iron_fraction  = 0.33,
    #     Y_ratio        = 0.26,
    #     initial_log_pc = 7.0,
    # ),
]

# --- core_nabla sweep (matches the gas-giant test for direct comparison) ---
CORE_NABLA_VALUES = np.array([
    0.000,   # isothermal
    0.025,
    0.050,
    0.075,
    0.100,   # radiative/conductive (existing default for water worlds)
    0.150,
    0.200,
    0.250,
    0.300,   # adiabatic
    0.350,
    0.400,
])

# --- Execution ---
RESULTS_FILE      = "../data/core_nabla_waterworlds.csv"
N_WORKERS         = max(1, mp.cpu_count() - 1)
SOLVE_TIMEOUT_SEC = 600
N_LAYERS          = 50


# =============================================================================
# ROW SCHEMA
# =============================================================================

ROW_COLUMNS = [
    'config_name',
    'core_nabla',
    'M_total_Me_target',
    'M_rock_Me',
    'M_water_Me',
    'T_surf_K',
    'R_total_Re',
    'R_rock_Re',
    'R_water_Re',          # outer edge of the water mantle (if reported)
    'M_total_Me_achieved',
    'P_center_GPa',
    'T_center_K',
    'Status',
]


def _empty_row(cfg, nabla, status):
    row = {col: np.nan for col in ROW_COLUMNS}
    row['config_name']       = cfg['name']
    row['core_nabla']        = nabla
    row['M_total_Me_target'] = cfg['M_total_Me']
    row['M_rock_Me']         = cfg['M_rock_Me']
    row['M_water_Me']        = cfg['M_water_Me']
    row['T_surf_K']          = cfg['T_surf_K']
    row['Status']            = status
    return row


# =============================================================================
# Z-PROFILE
# =============================================================================

def _build_gaussian_profile(cfg):
    """Standard Gaussian dilution profile for the gaseous envelope."""
    x = np.linspace(0, 1, N_LAYERS)
    sigma  = cfg['sigma_gauss']
    z_base = cfg['Z_base']
    z_core = cfg['Z_core']
    # Sub-grid conservation: downscale amplitude when sigma < dx
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
    cfg, nabla = task
    trial_id = f"NABLA_WW_{cfg['name']}_{nabla:.4f}"

    try:
        solver.clear_objective_cache()
    except AttributeError:
        pass

    try:
        z_prof = _build_gaussian_profile(cfg)

        params = {
            'M_core':         cfg['M_rock_Me']  * c.M_EARTH,
            'M_rock':         cfg['M_rock_Me']  * c.M_EARTH,
            'M_water':        cfg['M_water_Me'] * c.M_EARTH,   # > 0 triggers water-world path
            'P_surf':         cfg['P_surf_bar'],
            'T_surf':         cfg['T_surf_K'],
            'T_int':          cfg['T_int_K'],
            'z_base':         cfg['Z_base'],
            'z_profile':      z_prof,
            'sigma_val':      cfg['sigma_gauss'],
            'iron_fraction':  cfg['iron_fraction'],
            'Y_ratio':        cfg['Y_ratio'],
            'core_nabla':     float(nabla),       # *** the test variable ***
            'debug':          False,
            'initial_log_pc': cfg['initial_log_pc'],
        }

        target_mass = cfg['M_total_Me'] * c.M_EARTH
        res = solver.solve_structure(target_mass, params, 'mass', trial_id)

        if res is None:
            return False, _empty_row(cfg, nabla, 'solver_returned_none')

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

        # Inner-rock-core radius and water-mantle outer radius (if present)
        r_rock = res.get('R_rock', None)
        if r_rock is not None and np.isfinite(r_rock):
            row['R_rock_Re'] = float(r_rock) / c.R_EARTH

        r_int = res.get('R_int', None)
        if r_int is not None and np.isfinite(r_int):
            row['R_water_Re'] = float(r_int) / c.R_EARTH

        # Central conditions
        try:
            row['P_center_GPa'] = 10 ** float(res['P'][0]) * 1e-4
            row['T_center_K']   = float(res['T'][0])
        except Exception:
            pass

        return True, row

    except Exception as e:
        return False, _empty_row(cfg, nabla, f'exception_{type(e).__name__}')


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
        return {(r['config_name'], round(float(r['core_nabla']), 5))
                for _, r in df.iterrows()}
    except Exception:
        return set()


def build_tasks():
    return [(cfg, float(n)) for cfg in CONFIGURATIONS for n in CORE_NABLA_VALUES]


def print_conclusive_summary(file_path):
    if not os.path.exists(file_path):
        return
    df = pd.read_csv(file_path)
    df_ok = df[df['Status'] == 'ok']

    print("\n" + "=" * 72)
    print(" CONCLUSIVE SUMMARY — core_nabla sensitivity (water worlds)")
    print("=" * 72)
    for cfg_name in df_ok['config_name'].unique():
        sub = df_ok[df_ok['config_name'] == cfg_name].sort_values('core_nabla')
        if sub.empty:
            continue
        R = sub['R_total_Re'].values
        T = sub['T_center_K'].values
        n = sub['core_nabla'].values
        R_rel = 100.0 * (R.max() - R.min()) / R.mean()
        T_rel = T.max() / T.min() if T.min() > 0 else np.nan
        print(f"\n  [{cfg_name}]  swept {len(sub)} values of core_nabla")
        print(f"  core_nabla range:     [{n.min():.3f}, {n.max():.3f}]")
        print(f"  R_total range:        [{R.min():.4f}, {R.max():.4f}] R_E")
        print(f"  --> Relative R variation:   {R_rel:.3f} %")
        print(f"  T_center range:       [{T.min():.0f}, {T.max():.0f}] K")
        print(f"  --> T_center contrast:      factor of {T_rel:.2f}")

        def at(n_target):
            i = int(np.argmin(np.abs(n - n_target)))
            return n[i], R[i], T[i]
        print(f"  Regime anchors:")
        for n_target, label in [(0.00, 'isothermal'),
                                 (0.10, 'radiative/conductive (default)'),
                                 (0.30, 'adiabatic')]:
            n_a, R_a, T_a = at(n_target)
            print(f"    {label:>32s} (n={n_a:.3f}):  "
                  f"R = {R_a:.4f} R_E,  T_center = {T_a:.0f} K")
    print("=" * 72)


def main():
    print("=" * 72)
    print(" Core Temperature Gradient Sensitivity Test — Water Worlds")
    print("=" * 72)
    print(f"   Configurations:   {[c['name'] for c in CONFIGURATIONS]}")
    print(f"   core_nabla sweep: {list(np.round(CORE_NABLA_VALUES, 3))}")
    print(f"   Workers:          {N_WORKERS}")
    print(f"   Output:           {RESULTS_FILE}")
    print("=" * 72)

    out_dir = os.path.dirname(RESULTS_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    all_tasks = build_tasks()
    done = _resume_keys(RESULTS_FILE)
    todo = [t for t in all_tasks
            if (t[0]['name'], round(t[1], 5)) not in done]

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
            cfg, nabla = futures[fut]
            try:
                success, row = fut.result(timeout=SOLVE_TIMEOUT_SEC)
            except Exception as e:
                success = False
                row = _empty_row(cfg, nabla, f'driver_{type(e).__name__}')

            _append_row(row, RESULTS_FILE, write_header)
            write_header = False

            n_done += 1
            if not success:
                n_failed += 1

            tag = (f"R = {row['R_total_Re']:.4f} R_E,  T_c = {row['T_center_K']:.0f} K"
                   if success else f"FAIL [{row['Status']}]")
            print(f"  [{n_done:3d}/{len(todo)}]  "
                  f"{cfg['name']:>14s}  nabla = {nabla:.3f}  ->  {tag}")

    elapsed = (time.time() - t_start) / 60.0
    print("=" * 72)
    print(f"[*] Finished: {n_done - n_failed}/{n_done} succeeded")
    print(f"[*] Wall time: {elapsed:.1f} min")
    print(f"[*] Results:   {RESULTS_FILE}")

    print_conclusive_summary(RESULTS_FILE)


if __name__ == '__main__':
    main()