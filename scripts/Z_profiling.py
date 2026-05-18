"""
Z-Profile Functional-Form Robustness Test
==========================================

Tests whether the converged planetary radius is primarily sensitive to the
integrated heavy-element budget and the characteristic gradient width, or to
the specific functional form of Z(x).

For a fixed reference planet (M_total, M_core, M_water, T_surf, P_surf) we
sweep six functional families of Z(x) across their natural "width-like"
parameters. For each (family, width) the solver finds the central pressure
that delivers the target mass; we record the resulting radius and the
integrated envelope heavy-element excess. The atlas can then be plotted as
R vs integrated-Z with families overplotted.

If the families collapse onto a single (R, integrated-Z) curve, the Gaussian
parameterization is a representative -- not restrictive -- choice and the
functional form is observationally degenerate. That's the result you need
for referee comment #8 about the Gaussian justification.

Patterns: same robust failure handling, resume logic, and parallel structure
as the water-world atlas script.
"""

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import fuzzycore.constants as c
import fuzzycore.solver as solver
import fuzzycore.utils as utils

import z_profile_zoo as zoo


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Reference planet (Kepler-11e-like, gas-rich sub-Neptune) ---
# This is the same target used in the main paper's Sec 3.5 sweeps, so the
# functional-form robustness conclusion drops directly into that section.
M_TOTAL_ME     = 7.95
M_CORE_ME      = 6.60
M_WATER_ME     = 0.0
T_SURF         = 900.0
P_SURF         = 1.0
T_INT          = 200.0
Z_BASE         = 0.02
Z_CORE         = 0.99
IRON_FRAC      = 0.33
Y_RATIO        = 0.26
N_LAYERS       = 50

# --- Width-parameter sweep per family ---
# Each tuple is (family_name, parameter_name, values_array). The parameter
# name is purely a label that gets written to the CSV; the script uses it
# to dispatch to the right generator kwarg.
FAMILY_SWEEPS = [
    ('gaussian',    'sigma',      np.linspace(0.05, 0.40, 12)),
    ('step',        'frac_inner', np.linspace(0.05, 0.50, 12)),
    ('sigmoid',     'sharpness',  np.array([5, 10, 15, 20, 30, 50, 80, 150], dtype=float)),
    ('exponential', 'scale',      np.linspace(0.03, 0.40, 12)),
    ('bilinear',    'breakpoint', np.linspace(0.20, 0.85, 12)),
    ('vazan',       'index',      np.array([0.0])),   # Single canonical profile
]

# --- Execution ---
RESULTS_FILE      = "../data/profile_robustness.csv"
N_WORKERS         = max(1, mp.cpu_count() - 1)
SOLVE_TIMEOUT_SEC = 600
LOG_EVERY         = 1


# =============================================================================
# STANDARDIZED ROW SCHEMA
# =============================================================================

ROW_COLUMNS = [
    'Family',
    'Param_Name',
    'Param_Value',
    'M_total_Me_target',
    'M_core_Me',
    'M_water_Me',
    'R_total_Re',
    'M_total_Me_achieved',
    'Integrated_Z_excess',     # int (Z - z_base) dx over normalized envelope
    'Z_max',
    'Z_mean',
    'T_deep_K',
    'T_top_K',
    'Delta_T_K',
    'Status',
]


def _empty_row(family, pname, pval, status):
    row = {col: np.nan for col in ROW_COLUMNS}
    row['Family']             = family
    row['Param_Name']         = pname
    row['Param_Value']        = pval
    row['M_total_Me_target']  = M_TOTAL_ME
    row['M_core_Me']          = M_CORE_ME
    row['M_water_Me']         = M_WATER_ME
    row['Status']             = status
    return row


# =============================================================================
# PROFILE DISPATCH
# =============================================================================

def build_profile(family: str, pname: str, pval: float) -> np.ndarray:
    """Map (family, param_name, param_value) to a Z(x) array."""
    common = dict(n_layers=N_LAYERS, z_base=Z_BASE, z_core=Z_CORE)
    if family == 'gaussian':
        return zoo.gaussian_profile(sigma=pval, **common)
    if family == 'step':
        return zoo.step_profile(frac_inner=pval, **common)
    if family == 'sigmoid':
        # sweep sharpness with center fixed near the inner third
        return zoo.sigmoid_profile(sharpness=pval, center=0.75, **common)
    if family == 'exponential':
        return zoo.exponential_profile(scale=pval, **common)
    if family == 'bilinear':
        return zoo.bilinear_profile(breakpoint=pval, **common)
    if family == 'vazan':
        return zoo.vazan_staircase_profile(n_layers=N_LAYERS, z_base=Z_BASE)
    raise ValueError(f"Unknown family: {family}")


# =============================================================================
# TASK GENERATION & RESUME
# =============================================================================

def build_tasks():
    tasks = []
    for family, pname, pvalues in FAMILY_SWEEPS:
        for pv in pvalues:
            tasks.append((family, pname, float(pv)))
    return tasks


def _task_key(task):
    family, pname, pval = task
    return (family, pname, round(pval, 6))


def load_completed(results_file):
    if not os.path.exists(results_file):
        return set()
    try:
        df = pd.read_csv(results_file)
        return {(r['Family'], r['Param_Name'], round(r['Param_Value'], 6))
                for _, r in df.iterrows()}
    except Exception as e:
        print(f"  [!] Could not parse {results_file}: {e}. Starting fresh.")
        return set()


# =============================================================================
# WORKER
# =============================================================================

def run_one(task):
    """Run a single (family, parameter) point. Always returns (ok_bool, row)."""
    family, pname, pval = task
    trial_id = f"PROF_{family}_{pname}_{pval:.4f}"

    # The objective cache uses a profile fingerprint, so cross-task hits
    # should be safe — but clearing keeps each task fully isolated, which is
    # the right behaviour here since each task uses a genuinely different
    # profile shape.
    try:
        solver.clear_objective_cache()
    except AttributeError:
        pass

    try:
        z_prof = build_profile(family, pname, pval)

        # Pre-compute integrated Z so we have it even if the solver fails.
        integrated_Z = zoo.compute_integrated_Z(z_prof, z_base=Z_BASE)
        z_max  = float(np.max(z_prof))
        z_mean = float(np.mean(z_prof))

        params = {
            'M_core':         M_CORE_ME  * c.M_EARTH,
            'M_rock':         M_CORE_ME  * c.M_EARTH,
            'M_water':        M_WATER_ME * c.M_EARTH,
            'P_surf':         P_SURF,
            'T_surf':         T_SURF,
            'T_int':          T_INT,
            'z_base':         Z_BASE,
            'z_profile':      z_prof,
            # sigma_val is only used by the solver cache key, where the
            # z-profile fingerprint already disambiguates. Set to 0 for
            # non-Gaussian families so non-finite values can't poison the
            # cache hash.
            'sigma_val':      pval if family == 'gaussian' else 0.0,
            'iron_fraction':  IRON_FRAC,
            'Y_ratio':        Y_RATIO,
            'debug':          False,
            'initial_log_pc': 7.0,
        }

        target_mass = M_TOTAL_ME * c.M_EARTH
        res = solver.solve_structure(target_mass, params, 'mass', trial_id)

        if res is None:
            row = _empty_row(family, pname, pval, 'solver_returned_none')
            row['Integrated_Z_excess'] = integrated_Z
            row['Z_max']  = z_max
            row['Z_mean'] = z_mean
            return False, row

        try:
            r_tot_re = float(res['R'][-1]) / c.R_EARTH
            m_tot_me = float(res['M'][-1]) / c.M_EARTH
        except Exception as e:
            row = _empty_row(family, pname, pval,
                             f'malformed_result_{type(e).__name__}')
            row['Integrated_Z_excess'] = integrated_Z
            return False, row

        mass_err = (m_tot_me - M_TOTAL_ME) / M_TOTAL_ME
        if not np.isfinite(mass_err) or abs(mass_err) > 0.05:
            row = _empty_row(family, pname, pval,
                             f'mass_mismatch_{mass_err:+.3f}')
            row['R_total_Re']           = r_tot_re
            row['M_total_Me_achieved']  = m_tot_me
            row['Integrated_Z_excess']  = integrated_Z
            row['Z_max']                = z_max
            row['Z_mean']               = z_mean
            return False, row

        # --- Success ---
        row = _empty_row(family, pname, pval, 'ok')
        row['R_total_Re']          = r_tot_re
        row['M_total_Me_achieved'] = m_tot_me
        row['Integrated_Z_excess'] = integrated_Z
        row['Z_max']               = z_max
        row['Z_mean']              = z_mean

        try:
            thermal = utils.calculate_thermal_contrast(res)
            row['T_deep_K']  = float(thermal['T_deep'])
            row['T_top_K']   = float(thermal['T_top'])
            row['Delta_T_K'] = float(thermal['Delta_T'])
        except Exception:
            pass

        return True, row

    except Exception as e:
        return False, _empty_row(family, pname, pval,
                                 f'exception_{type(e).__name__}')


# =============================================================================
# DRIVER
# =============================================================================

def _append_row(row, file_path, write_header):
    pd.DataFrame([row], columns=ROW_COLUMNS).to_csv(
        file_path, mode='a', header=write_header, index=False)


def main():
    print("=" * 72)
    print(" Z-Profile Functional-Form Robustness Test")
    print("=" * 72)
    print(f"   Reference target:    M = {M_TOTAL_ME} M_E,  R = ? (solver finds)")
    print(f"   Solid core mass:     {M_CORE_ME} M_E")
    print(f"   Water mantle mass:   {M_WATER_ME} M_E")
    print(f"   Boundary conditions: T_surf = {T_SURF} K,  P_surf = {P_SURF} bar")
    print(f"   Families:            {[f for f, _, _ in FAMILY_SWEEPS]}")
    print(f"   Total grid points:   {sum(len(v) for _, _, v in FAMILY_SWEEPS)}")
    print(f"   Workers:             {N_WORKERS}")
    print(f"   Output:              {RESULTS_FILE}")
    print("=" * 72)

    out_dir = os.path.dirname(RESULTS_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    all_tasks = build_tasks()
    done_keys = load_completed(RESULTS_FILE)
    todo = [t for t in all_tasks if _task_key(t) not in done_keys]

    print(f"[*] Total tasks:        {len(all_tasks)}")
    print(f"[*] Already completed:  {len(done_keys)}")
    print(f"[*] Remaining to run:   {len(todo)}")

    if not todo:
        print("[*] Nothing to do.")
        return

    write_header = not os.path.exists(RESULTS_FILE)
    n_done = 0
    n_failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = {exe.submit(run_one, t): t for t in todo}

        for fut in as_completed(futures):
            task = futures[fut]
            try:
                success, row = fut.result(timeout=SOLVE_TIMEOUT_SEC)
            except Exception as e:
                success = False
                row = _empty_row(task[0], task[1], task[2],
                                 f'driver_{type(e).__name__}')

            _append_row(row, RESULTS_FILE, write_header)
            write_header = False

            n_done += 1
            if not success:
                n_failed += 1

            if n_done % LOG_EVERY == 0 or not success:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta_min = (len(todo) - n_done) / rate / 60.0 if rate > 0 else float('inf')

                if success:
                    tag = (f"R = {row['R_total_Re']:.3f} R_E"
                           f" (intZ={row['Integrated_Z_excess']:.4f})")
                else:
                    tag = f"FAIL [{row['Status']}]"

                print(f"  [{n_done:4d}/{len(todo)}]  "
                      f"{task[0]:>12s}  {task[1]}={task[2]:.4f}  ->  {tag}  "
                      f"(rate {rate:.2f}/s, ETA {eta_min:.1f} min)")

    elapsed_min = (time.time() - t_start) / 60.0
    n_ok = n_done - n_failed
    print("=" * 72)
    print(f"[*] Finished. {n_ok}/{n_done} succeeded ({100.0*n_ok/n_done:.1f}%)")
    print(f"[*] Failed:    {n_failed}/{n_done} ({100.0*n_failed/n_done:.1f}%)")
    print(f"[*] Wall time: {elapsed_min:.1f} min")
    print(f"[*] Atlas written to: {RESULTS_FILE}")
    print("=" * 72)


if __name__ == '__main__':
    main()