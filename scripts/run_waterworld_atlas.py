"""
Water-World Degeneracy Atlas
============================

Forward-model exploration of the static interior degeneracy for sub-Neptune
water worlds in the (M_water, M_env, sigma) parameter space at fixed total
planetary mass.

For each grid point, the solver searches for the central pressure that
delivers the target total mass and reports the resulting planetary radius
along with a small set of structural diagnostics. The output is a CSV atlas
that can be sliced into iso-radius surfaces, projected to 2D, or used as a
lookup table for interpreting observed sub-Neptune (M, R) measurements.

The script:
- Resumes from a partially-completed run by skipping grid points already in
  the output CSV.
- Records *every* attempted point (success or failure) so the resume logic
  doesn't re-attempt the same failure twice.
- Isolates each grid point in its own worker process; a crash in one point
  cannot corrupt the others.
- Applies a hard timeout per point so pathological corners can't stall the
  whole sweep.

Edit the CONFIGURATION block below to set the target planet and grid
resolution. The defaults are tuned for a GJ 1214 b-like target (~8 M_E,
warm sub-Neptune).
"""

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import random

import numpy as np
import pandas as pd

import fuzzycore.constants as c
import fuzzycore.solver as solver
import fuzzycore.utils as utils
import fuzzycore.eos as eos


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Target planet (representative warm sub-Neptune) ---
M_TOTAL_ME   = 8.0      # Total planetary mass (Earth masses)
T_SURF       = 550.0    # Upper-atmosphere temperature (K)
P_SURF       = 1.0      # Upper-atmosphere pressure (bar)
T_INT        = 200.0    # Internal heat-flux temperature (K)
Z_BASE       = 0.02     # Atmospheric baseline metallicity
IRON_FRAC    = 0.33     # Iron mass fraction within the rocky core
Y_RATIO      = 0.26     # Helium-to-(H+He) mass ratio in the gas phase

# --- Grid resolution ---
N_MWATER     = 15
N_MENV       = 15
N_SIGMA      = 12

# --- Grid ranges (M in Earth masses, sigma dimensionless) ---
MWATER_RANGE = (0.0, 4.0)
MENV_RANGE   = (0.05, 3.0)         # M_env > 0; a zero envelope is unphysical for this study
SIGMA_RANGE  = (1e-3, 0.6)         # Log-spaced

# --- Minimum core mass to attempt (skip otherwise-unphysical corners) ---
M_CORE_MIN_ME = 0.5

# --- Execution ---
RESULTS_FILE        = "../data/waterworld_atlas.csv"
N_WORKERS           = max(1, mp.cpu_count() - 1)
SOLVE_TIMEOUT_SEC   = 600                # Hard cap per grid point
LOG_EVERY           = 1                  # Print every Nth completion


# =============================================================================
# STANDARDIZED ROW SCHEMA
# =============================================================================
# Every row in the output CSV has the same columns regardless of
# success/failure. Missing diagnostics are NaN. The 'Status' column carries
# success or a short failure tag.

ROW_COLUMNS = [
    'M_total_Me_target',
    'M_core_Me',
    'M_water_Me',
    'M_env_Me',
    'Sigma',
    'R_total_Re',
    'R_rock_Re',
    'M_total_Me_achieved',
    'M_Z_total_Me',
    'T_deep_K',
    'T_top_K',
    'Delta_T_K',
    'Status',
]


def _empty_row(task, status):
    """Build a row dict pre-filled with NaN for all diagnostics."""
    m_core, m_water, m_env, sigma = task
    row = {col: np.nan for col in ROW_COLUMNS}
    row['M_total_Me_target'] = M_TOTAL_ME
    row['M_core_Me']         = m_core
    row['M_water_Me']        = m_water
    row['M_env_Me']          = m_env
    row['Sigma']             = sigma
    row['Status']            = status
    return row


# =============================================================================
# GRID GENERATION & RESUME LOGIC
# =============================================================================

def build_grid():
    """Yield the full (M_core, M_water, M_env, sigma) parameter grid."""
    m_water_grid = np.linspace(*MWATER_RANGE, N_MWATER)
    m_env_grid   = np.linspace(*MENV_RANGE,   N_MENV)
    sigma_grid   = np.logspace(np.log10(SIGMA_RANGE[0]),
                               np.log10(SIGMA_RANGE[1]),
                               N_SIGMA)

    tasks = []
    for m_w, m_e, sig in product(m_water_grid, m_env_grid, sigma_grid):
        m_core = M_TOTAL_ME - m_w - m_e
        if m_core < M_CORE_MIN_ME:
            continue
        tasks.append((m_core, m_w, m_e, sig))

    random.shuffle(tasks)
    return tasks


def _key(task):
    """Round-trip-stable hash for a grid point so the resume set works."""
    m_core, m_water, m_env, sigma = task
    return (round(m_core, 4),
            round(m_water, 4),
            round(m_env,  4),
            round(sigma,  5))


def load_completed(results_file):
    """Return the set of grid-point keys already present in the output CSV."""
    if not os.path.exists(results_file):
        return set()
    try:
        df = pd.read_csv(results_file)
        return {(round(r['M_core_Me'], 4),
                 round(r['M_water_Me'], 4),
                 round(r['M_env_Me'],   4),
                 round(r['Sigma'],      5)) for _, r in df.iterrows()}
    except Exception as e:
        print(f"  [!] Could not parse {results_file}: {e}. Starting fresh.")
        return set()


# =============================================================================
# WORKER
# =============================================================================

def run_one_point(task):
    """
    Run a single grid point. Always returns (success_bool, row_dict).

    Any exception raised inside the solver is caught and converted into a
    failure row, so the driver never sees a crash from this function.
    """
    m_core, m_water, m_env, sigma = task

    # Belt-and-braces: if the user has the cross-call objective cache from
    # the solver patch, clear it. The cache key doesn't include M_water /
    # M_env, so it would give wrong answers if reused across grid points.
    try:
        solver.clear_objective_cache()
    except AttributeError:
        pass

    trial_id = (f"WW_Mc{m_core:.2f}_Mw{m_water:.2f}"
                f"_Me{m_env:.2f}_s{sigma:.4f}")

    try:
        z_prof = utils.generate_gaussian_z_profile(
            n_layers=50,
            sigma=sigma,
            z_base=Z_BASE,
            z_core=0.99,
        )

        params = {
            'M_core':         m_core  * c.M_EARTH,
            'M_rock':         m_core  * c.M_EARTH,
            'M_water':        m_water * c.M_EARTH,
            'P_surf':         P_SURF,
            'T_surf':         T_SURF,
            'T_int':          T_INT,
            'z_base':         Z_BASE,
            'z_profile':      z_prof,
            'sigma_val':      sigma,
            'iron_fraction':  IRON_FRAC,
            'Y_ratio':        Y_RATIO,
            'debug':          False,
            'initial_log_pc': 7.0,
        }

        target_mass_kg = M_TOTAL_ME * c.M_EARTH

        res = solver.solve_structure(
            target_val=target_mass_kg,
            params=params,
            mode='mass',
            trial_id=trial_id,
        )

        if res is None:
            return False, _empty_row(task, 'solver_returned_none')

        # Pull the headline scalars defensively.
        try:
            r_tot_re = float(res['R'][-1]) / c.R_EARTH
            m_tot_me = float(res['M'][-1]) / c.M_EARTH
        except Exception as e:
            row = _empty_row(task, f'malformed_result_{type(e).__name__}')
            return False, row

        # Was the converged mass actually close to the target?
        mass_err = (m_tot_me - M_TOTAL_ME) / M_TOTAL_ME
        if not np.isfinite(mass_err) or abs(mass_err) > 0.05:
            row = _empty_row(task, f'mass_mismatch_{mass_err:+.3f}')
            row['M_total_Me_achieved'] = m_tot_me
            row['R_total_Re']          = r_tot_re
            return False, row

        # --- Success: collect diagnostics ---
        row = _empty_row(task, 'ok')
        row['R_total_Re']          = r_tot_re
        row['M_total_Me_achieved'] = m_tot_me

        # Inner-rock-core radius
        r_rock = res.get('R_rock', None)
        if r_rock is not None and np.isfinite(r_rock):
            row['R_rock_Re'] = float(r_rock) / c.R_EARTH

        # Heavy-element budget (envelope + condensed) if the solver reports it
        m_z = res.get('M_Z_total', None)
        if m_z is not None and np.isfinite(m_z):
            row['M_Z_total_Me'] = float(m_z) / c.M_EARTH

        # Thermal contrast across the fluid envelope (best-effort)
        try:
            thermal = utils.calculate_thermal_contrast(res)
            if isinstance(thermal, dict):
                row['T_deep_K']  = float(thermal.get('T_deep', np.nan))
                row['T_top_K']   = float(thermal.get('T_top',  np.nan))
                row['Delta_T_K'] = float(thermal.get('Delta_T', np.nan))
        except Exception:
            pass

        return True, row

    except Exception as e:
        # Catch-all so a single bad grid point cannot crash the worker.
        return False, _empty_row(task, f'exception_{type(e).__name__}')


# =============================================================================
# DRIVER
# =============================================================================

def _append_row(row, file_path, write_header):
    """Append a single row to the output CSV."""
    pd.DataFrame([row], columns=ROW_COLUMNS).to_csv(
        file_path, mode='a', header=write_header, index=False,
    )


def main():
    print("=" * 72)
    print(" Water-World Degeneracy Atlas")
    print("=" * 72)
    print(f"   Target mass:    {M_TOTAL_ME} M_E")
    print(f"   T_surf / P_surf: {T_SURF} K  /  {P_SURF} bar")
    print(f"   T_int:          {T_INT} K")
    print(f"   Iron fraction:  {IRON_FRAC}")
    print(f"   Grid:           {N_MWATER} M_water  x  {N_MENV} M_env  x  {N_SIGMA} sigma")
    print(f"   Workers:        {N_WORKERS}")
    print(f"   Output:         {RESULTS_FILE}")
    print("=" * 72)

    out_dir = os.path.dirname(RESULTS_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    all_tasks = build_grid()
    completed_keys = load_completed(RESULTS_FILE)
    tasks_to_run = [t for t in all_tasks if _key(t) not in completed_keys]

    print(f"[*] Total grid points:   {len(all_tasks)}")
    print(f"[*] Already completed:   {len(completed_keys)}")
    print(f"[*] Remaining to run:    {len(tasks_to_run)}")

    if not tasks_to_run:
        print("[*] Nothing to do.")
        return

    write_header = not os.path.exists(RESULTS_FILE)
    n_done = 0
    n_failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_one_point, t): t for t in tasks_to_run}

        for future in as_completed(futures):
            task = futures[future]

            try:
                success, row = future.result(timeout=SOLVE_TIMEOUT_SEC)
            except Exception as e:
                # Hard timeout, process crash, or any other failure in the
                # future itself. Record as failed and keep going.
                success = False
                row = _empty_row(task, f'driver_{type(e).__name__}')

            _append_row(row, RESULTS_FILE, write_header)
            write_header = False  # only the first append writes the header

            n_done += 1
            if not success:
                n_failed += 1

            if n_done % LOG_EVERY == 0 or not success:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                remaining = len(tasks_to_run) - n_done
                eta_min = (remaining / rate / 60.0) if rate > 0 else float('inf')

                m_core, m_w, m_e, sig = task
                if success:
                    summary = f"R = {row['R_total_Re']:.3f} R_E"
                else:
                    summary = f"FAIL [{row['Status']}]"

                print(f"  [{n_done:5d}/{len(tasks_to_run)}] "
                      f"Mc={m_core:.2f} Mw={m_w:.2f} Me={m_e:.2f} "
                      f"sig={sig:.4f}  ->  {summary}  "
                      f"(rate {rate:.2f}/s, ETA {eta_min:.1f} min)")

    elapsed_min = (time.time() - t_start) / 60.0
    n_ok = n_done - n_failed
    print("=" * 72)
    print(f"[*] Finished: {n_ok}/{n_done} succeeded ({100.0*n_ok/n_done:.1f}%)")
    print(f"[*] Failures: {n_failed}/{n_done} ({100.0*n_failed/n_done:.1f}%)")
    print(f"[*] Wall time: {elapsed_min:.1f} min")
    print(f"[*] Atlas written to: {RESULTS_FILE}")
    print("=" * 72)


if __name__ == '__main__':
    main()