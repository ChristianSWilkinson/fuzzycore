import os
import numpy as np
import pandas as pd
import scipy.spatial
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree, Delaunay
from scipy.optimize import brentq
from pathlib import Path
from .utils import time_it

from . import constants as const

# =============================================================================
# ABSOLUTE PATH RESOLUTION
# =============================================================================
# __file__ gets the absolute path to this exact eos.py file.
# .resolve() resolves any symlinks.
# .parents[2] navigates up 3 levels: fuzzycore/src/fuzzycore/eos.py -> fuzzycore/
FUZZYCORE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = FUZZYCORE_ROOT / "data" / "EOS"


# =============================================================================
# GLOBAL CACHE
# =============================================================================
# Memory stores to prevent redundant loading and interpolation of heavy EOS tables.
_RAW_TABLES = {}
_MIXED_CACHE = {}
_CORE_CACHE = {}
_WATER_INTERP = None
_ROCK_BOUNDS = {'t_min': 300.0, 't_max': 100000.0}


def clear_mixed_cache() -> None:
    """
    Clears all in-memory caches containing interpolated and raw Equation of 
    State (EOS) data. Useful for resetting memory during bulk integrations.
    """
    global _MIXED_CACHE, _RAW_TABLES, _CORE_CACHE, _WATER_INTERP
    _MIXED_CACHE.clear()
    _RAW_TABLES.clear()
    _CORE_CACHE.clear()
    _WATER_INTERP = None


def get_mgo_eos_b2(rho_kg_m3, T_K):
    """Analytical B2 MgO EOS (Musella et al. 2018). Returns Pressure in bar."""
    M_mgo = 40.3044
    n_atoms = 2.0
    Z_total = 20.0
    R_gas = 8.31446

    V_0 = 10.970
    K_0 = 120.76
    K_0_prime = 4.803
    Theta_0 = 447.906
    gamma_0 = 1.755
    beta = -0.530
    gamma_inf = -7.579e-2
    a_0 = 1.341e-4
    m_param = 0.660

    V = M_mgo / (rho_kg_m3 / 1000.0)
    v_ratio = V / V_0
    X = v_ratio**(1.0/3.0) 

    P_FG0 = 1003.6 * (Z_total / V_0)**(5.0/3.0)  
    c_0 = -np.log(3.0 * K_0 / P_FG0)             
    c_2 = 1.5 * (K_0_prime - 3.0) - c_0          
    
    P_cold_GPa = 3.0 * K_0 * (X**-5) * (1.0 - X) * np.exp(c_0 * (1.0 - X)) * (1.0 + c_2 * X * (1.0 - X))

    gamma = gamma_inf + (gamma_0 - gamma_inf) * (v_ratio**beta)
    Theta = Theta_0 * (v_ratio**-gamma_inf) * np.exp(((gamma_0 - gamma_inf) / beta) * (1.0 - v_ratio**beta))
    
    safe_T = np.where(T_K > 0, T_K, 1e-10)
    # Guard the Debye integral: exp(x) overflows float64 for x > ~709
    debye_arg = np.clip(Theta / safe_T, 0.0, 500.0)
    E_harm = 3.0 * n_atoms * R_gas * (
        Theta / 2.0 + Theta / (np.exp(debye_arg) - 1.0 + 1e-300)
        )
        
    P_harm_GPa = (gamma * E_harm / V) / 1000.0
    P_ae_GPa = ((3.0 * R_gas / (2.0 * V)) * m_param * a_0 * (v_ratio**m_param) * (safe_T**2)) / 1000.0

    return (P_cold_GPa + P_harm_GPa + P_ae_GPa) * 10000.0

def build_mgo_grid(logp_eval, logt_eval):
    """
    Invert get_mgo_eos_b2 on the (logT, logP) grid using np.interp per T row.
    One vectorised EOS call per temperature (400 total) instead of one brentq
    per (T, P) cell (~114 000 total). Guarantees dρ/dP > 0 by construction.
    """
    rho_grid = np.full((len(logt_eval), len(logp_eval)), np.nan)

    # Dense ρ grid for inversion accuracy (~0.14% relative spacing)
    rho_arr  = np.geomspace(4.0e3, 1.0e6, 2000)   # kg/m³
    P_targets = 10.0**logp_eval                     # all target pressures, shape (n_P,)

    for j, logt in enumerate(logt_eval):
        T = 10.0**logt
        if T < 1000.0:
            continue

        # ONE vectorised call: P(ρ) at this T for all 2000 densities
        P_arr = get_mgo_eos_b2(rho_arr, np.full(len(rho_arr), T))   # bar

        # P should be monotonically increasing with ρ — skip row if not
        if not np.all(np.diff(P_arr) > 0):
            continue

        # Invert P → ρ for all target pressures simultaneously
        rho_row = np.interp(P_targets, P_arr, rho_arr, left=np.nan, right=np.nan)

        # Only keep high-pressure regime; low-P cells stay NaN (filled by sanitisation)
        rho_row[P_targets <= 1e7] = np.nan

        rho_grid[j, :] = rho_row

    return rho_grid

# =============================================================================
# DATA LOADING
# =============================================================================
@time_it
def _load_raw_table(name: str, path: str, cols: list, log_cols: bool = False) -> np.ndarray:
    """
    A generic file loader specifically tailored for extracting data from 
    ab-initio and DirEOS formatting conventions.

    Parameters
    ----------
    name : str
        Human-readable name of the component (e.g., 'Hydrogen', 'He') for logging.
    path : str
        Absolute or relative file path to the EOS data table.
    cols : list of int
        List of column indices to extract from the raw text file.
    log_cols : bool, optional
        If True, assumes the data columns are provided in base-10 logarithmic 
        format and automatically converts them to linear space (default is False).

    Returns
    -------
    np.ndarray or None
        A parsed numpy array containing the specific columns, or None if the 
        file is missing/corrupted.
    """
    try:
        if not os.path.exists(path):
            print(f"  [Error] File not found: {path}")
            return None

        # Load raw data skipping comment lines
        data = np.genfromtxt(path, delimiter='', comments='#', usecols=cols)

        if log_cols:
            # Filter out extreme values and exponentiate back to linear
            mask = np.all(data < 100, axis=1)
            data = data[mask]
            data = 10 ** data

        return data

    except Exception as e:
        print(f"Error loading {name} from {path}: {e}")
        return None

@time_it
def load_all_raw_data(base_dir: str = str(DATA_DIR)) -> dict:
    """
    Central function to load all fundamental EOS tables (H, He, H2O, Rock, Iron) 
    into memory. Only loads from disk on the first invocation.

    Parameters
    ----------
    base_dir : str, optional
        The root directory containing the EOS data files.

    Returns
    -------
    dict
        A dictionary mapping component names to their raw numpy arrays.
    """
    global _RAW_TABLES
    
    # Return immediately if cache is already populated
    if _RAW_TABLES:
        return _RAW_TABLES

    print("--- Loading Raw EOS Tables (From Disk) ---")

    # Define absolute/relative paths based on the base_dir
    h_path = os.path.join(base_dir, "DirEOS2021", "TABLE_H_TP_v1")
    he_path = os.path.join(base_dir, "DirEOS2021", "TABLE_HE_TP_v1")
    h2o_path = os.path.join(base_dir, "h2o-abinitio.dat")
    rock_path_old = os.path.join(base_dir, "aneosRock.dat")
    rock_path = os.path.join(base_dir, "eos_mgsio3.dat")
    # UPDATED: Pointing to the new SESAME file
    iron_path = os.path.join(base_dir, "nouvelle_sesame_fe.dat")

    # ---------------------------------------------------------
    # 1. Load Hydrogen (DirEOS)
    # ---------------------------------------------------------
    print("  > Loading Hydrogen...")
    # Extract: logT, logP, logRho, logS
    h_data = _load_raw_table("H", h_path, [0, 1, 2, 4], log_cols=True)
    if h_data is not None:
        # Unit conversions to match internal SI framework
        h_data[:, 1] *= 1e4             # Pressure: GPa -> Bar
        h_data[:, 2] *= 1000.0          # Density: g/cm^3 -> kg/m^3
        h_data[:, 3] *= const.MJ_TO_J   # Entropy: MJ/kg/K -> J/kg/K
        # Swap columns 0 and 1 to enforce internal standard: [Pressure, Temperature, ...]
        h_data[:, [0, 1]] = h_data[:, [1, 0]]
        _RAW_TABLES['H'] = h_data

    # ---------------------------------------------------------
    # 2. Load Helium (DirEOS)
    # ---------------------------------------------------------
    print("  > Loading Helium...")
    he_data = _load_raw_table("He", he_path, [0, 1, 2, 4], log_cols=True)
    if he_data is not None:
        he_data[:, 1] *= 1e4            
        he_data[:, 2] *= 1000.0         
        he_data[:, 3] *= const.MJ_TO_J  
        he_data[:, [0, 1]] = he_data[:, [1, 0]]
        _RAW_TABLES['He'] = he_data

    # ---------------------------------------------------------
    # 3. Load Water (Ab-initio)
    # ---------------------------------------------------------
    print("  > Loading Water...")
    if os.path.exists(h2o_path):
        try:
            cols = ['T', 'Rho', 'P', 'U', 'S_erg']
            df = pd.read_csv(h2o_path, sep=r'\s+', comment='#', header=None, names=cols)
            df = df[df['P'] > 0]  # Filter non-physical negative pressures
            h2o_arr = df[['P', 'T', 'Rho', 'S_erg']].to_numpy()
            
            # Unit conversions
            h2o_arr[:, 2] *= 1000.0         # Density: g/cm^3 -> kg/m^3
            h2o_arr[:, 3] *= 1e-4           # Entropy: erg/g/K -> J/kg/K
            _RAW_TABLES['H2O'] = h2o_arr
        except Exception as e:
            print(f"  [Warning] H2O failed: {e}")

    '''
    # ---------------------------------------------------------
    # 4. Load Silicate Rock (ANEOS)
    # ---------------------------------------------------------
    print("  > Loading Rock...")
    if os.path.exists(rock_path):
        try:
            cols = ['T', 'Rho', 'P', 'U', 'S_erg']
            df = pd.read_csv(rock_path, sep=r'\s+', comment='#', header=None, names=cols)
            df = df[df['P'] > 0]
            rock_arr = df[['P', 'T', 'Rho', 'S_erg']].to_numpy()
            
            rock_arr[:, 2] *= 1000.0
            rock_arr[:, 3] *= 1e-4
            _RAW_TABLES['Rock'] = rock_arr
        except Exception as e:
            print(f"  [Warning] Rock failed: {e}")
    '''

    # ---------------------------------------------------------
    # 4. Load Silicate Rock (MgSiO3 + MgO Synthetic Extension)
    # ---------------------------------------------------------
    print("  > Loading Rock (MgSiO3 + MgO)...")
    if os.path.exists(rock_path):
        try:
            # 1. REVERTED to the correct original columns!
            column_names = [
                'Density_g_cm3', 'Temperature_K', 'Pressure_bar', 
                'Energy_erg_g', 'FreeEnergy_erg_g'
            ]
            
            df_mgsio3 = pd.read_csv(
                rock_path, sep=r'\s+', header=None, 
                names=column_names, comment='#'
            )
            
            # Calculate Entropy in erg/(g*K) safely
            if not np.allclose(df_mgsio3['FreeEnergy_erg_g'].dropna(), 0):
                df_mgsio3['Entropy_erg_g_K'] = np.where(
                    df_mgsio3['Temperature_K'] > 0,
                    (df_mgsio3['Energy_erg_g'] - df_mgsio3['FreeEnergy_erg_g']) / df_mgsio3['Temperature_K'],
                    np.nan 
                )
            else:
                df_mgsio3['Entropy_erg_g_K'] = np.nan
                
            # Filter non-physical pressures and bound MgSiO3 to P <= 10 Mbar
            df_mgsio3 = df_mgsio3[(df_mgsio3['Pressure_bar'] > 0) & (df_mgsio3['Pressure_bar'] <= 1e7)]
            
            # Convert to fuzzycore internal units
            df_mgsio3['Rho_kg_m3'] = df_mgsio3['Density_g_cm3'] * 1000.0
            df_mgsio3['S_J_kg_K'] = df_mgsio3['Entropy_erg_g_K'] * 1e-4  
            
            # 2. Map dynamically to constants to prevent KD-Tree axis flips
            mgsio3_arr = np.zeros((len(df_mgsio3), 4))
            mgsio3_arr[:, const.P_COL] = df_mgsio3['Pressure_bar']
            mgsio3_arr[:, const.T_COL] = df_mgsio3['Temperature_K']
            mgsio3_arr[:, const.RHO_COL] = df_mgsio3['Rho_kg_m3']
            mgsio3_arr[:, const.S_COL] = df_mgsio3['S_J_kg_K'].fillna(0.0)
            
            # 3. Generate Dense MgO synthetic data for P > 10 Mbar
            rho_mgo = np.geomspace(4.0, 1000.0, 400) * 1000.0 
            T_mgo = np.geomspace(1000.0, 100000.0, 400) 
            Rho_mesh, T_mesh = np.meshgrid(rho_mgo, T_mgo)
            
            Rho_flat = Rho_mesh.flatten()
            T_flat = T_mesh.flatten()
            P_flat = get_mgo_eos_b2(Rho_flat, T_flat) 
            
            # Filter strictly > 10 Mbar
            mask = P_flat > 1e7
            
            # Safely align MgO to fuzzycore columns
            mgo_arr = np.zeros((np.sum(mask), 4))
            mgo_arr[:, const.P_COL] = P_flat[mask]
            mgo_arr[:, const.T_COL] = T_flat[mask]
            mgo_arr[:, const.RHO_COL] = Rho_flat[mask]
            mgo_arr[:, const.S_COL] = 0.0
            
            # 4. Store them strictly separated!
            _RAW_TABLES['MgSiO3'] = mgsio3_arr
            _RAW_TABLES['MgO'] = mgo_arr
            
        except Exception as e:
            print(f"  [Warning] Rock failed: {e}")

    # ---------------------------------------------------------
    # 5. Load Iron (SESAME Replacement)
    # ---------------------------------------------------------
    print("  > Loading Iron (SESAME)...")
    if os.path.exists(iron_path):
        try:
            # Columns in SESAME: T(eV) Rho(g/cm3) P(Mbar) Energy(Mj/kg) FreeEnergy(Mj/kg)
            cols = ['T_eV', 'Rho_g_cm3', 'P_Mbar', 'Energy_MJ_kg', 'FreeEnergy_MJ_kg']
            df = pd.read_csv(iron_path, sep=r'\s+', comment='#', names=cols)

            # Clean SESAME sentinels (unphysical placeholder values)
            df.replace(-0.99999996e+25, np.nan, inplace=True)
            df.dropna(inplace=True)

            # Convert Temperature: eV -> K
            df['T_K'] = df['T_eV'] * 11604.525

            # Filter non-physical negative pressures and zero temperatures (prevent div by zero)
            df = df[(df['P_Mbar'] > 0) & (df['T_K'] > 0)]

            # Convert Pressure: Mbar -> Bar
            df['P_bar'] = df['P_Mbar'] * 1e6

            # Convert Density: g/cm^3 -> kg/m^3
            df['Rho_kg_m3'] = df['Rho_g_cm3'] * 1000.0

            # Calculate Entropy in MJ/(kg*K), then convert to J/(kg*K) to match downstream
            # S = (E - F) / T
            df['S_J_kg_K'] = ((df['Energy_MJ_kg'] - df['FreeEnergy_MJ_kg']) / df['T_K']) * 1e6

            # Select and order columns precisely as internal framework demands: [P, T, Rho, S]
            iron_arr = df[['P_bar', 'T_K', 'Rho_kg_m3', 'S_J_kg_K']].to_numpy()
            
            _RAW_TABLES['Iron'] = iron_arr
        except Exception as e:
            print(f"  [Warning] Iron (SESAME) failed: {e}")

    return _RAW_TABLES


# =============================================================================
# INTERPOLATORS
# =============================================================================

@time_it
def interpolate_table(grid_points: np.ndarray, values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """
    Interpolates scattered 2D EOS data (P, T) using a Delaunay triangulation 
    (LinearNDInterpolator). Implements a robust fallback to Nearest-Neighbor 
    interpolation for points queried just outside the defined convex hull.
    """
    lin_interp = LinearNDInterpolator(grid_points, values, rescale=True)
    result = lin_interp(query_points)
    
    # Handle Out-of-Bounds Queries (NaNs)
    if np.any(np.isnan(result)):
        nan_mask = np.isnan(result)
        near_interp = NearestNDInterpolator(grid_points, values, rescale=True)
        # Overwrite NaN values using the nearest available boundary point
        result[nan_mask] = near_interp(query_points[nan_mask])
        
    return result


# =============================================================================
# CORE MIXER (Iron + Silicate Rock)
# =============================================================================
@time_it
def get_core_interpolator(iron_fraction: float = 0.33):
    """
    Builds a combined Iron-Rock interpolator. 
    Automatically handles the dissociation of MgSiO3 -> MgO at 10 Mbar 
    by rescaling the local mass fractions of the retained solid.
    """
    cache_key = f"core_{iron_fraction:.3f}"
    if cache_key in _CORE_CACHE:
        return _CORE_CACHE[cache_key]

    raw_tables = load_all_raw_data()
    iron_table = raw_tables['Iron']
    mgsio3_table = raw_tables['MgSiO3']
    mgo_table = raw_tables['MgO']

    print(f"--- Building Core Interpolator (iron_fraction={iron_fraction}) ---")

    # 0. FInd points
    iron_pts = np.column_stack((np.log10(iron_table[:, 0]), np.log10(iron_table[:, 1])))
    rock_pts = np.column_stack((np.log10(mgsio3_table[:, 0]), np.log10(mgsio3_table[:, 1])))
    mgo_pts  = np.column_stack((np.log10(mgo_table[:, 0]), np.log10(mgo_table[:, 1])))

    # 1. Create separate interpolators to avoid boundary bleed-over
    iron_lin  = LinearNDInterpolator(iron_pts, iron_table[:, 2], rescale=True)
    mgsio3_lin  = LinearNDInterpolator(rock_pts, mgsio3_table[:, 2], rescale=True)
    iron_near = NearestNDInterpolator(iron_pts, iron_table[:, 2], rescale=True)
    mgsio3_near = NearestNDInterpolator(rock_pts, mgsio3_table[:, 2], rescale=True)
    mgo_lin  = LinearNDInterpolator(mgo_pts, mgo_table[:, 2], rescale=True)
    mgo_near = NearestNDInterpolator(mgo_pts, mgo_table[:, 2], rescale=True)

    def _eval_smooth(lin_interp, near_interp, query_pts):
        vals = lin_interp(query_pts)
        bad  = np.isnan(vals)
        if np.any(bad):
            vals[bad] = near_interp(query_pts[bad])
        return vals

    # 2. Grid for the mixed core (Extended to 100,000 K and 1e12 bar)
    logp_eval = np.linspace(np.log10(1e5), np.log10(1e12), 400) 
    logt_eval = np.linspace(np.log10(300), np.log10(100000), 400)
    P_mesh, T_mesh = np.meshgrid(logp_eval, logt_eval)
    pts = np.column_stack((P_mesh.flatten(), T_mesh.flatten()))

    # Build MgO via physical EOS inversion (guarantees dρ/dP > 0)
    rho_mgo_grid_2d = build_mgo_grid(logp_eval, logt_eval)
    rho_mgo_eval    = rho_mgo_grid_2d.flatten()
    # Fill any cells where brentq failed (extreme T/P boundaries)
    mgo_nan = np.isnan(rho_mgo_eval)
    if np.any(mgo_nan):
        rho_mgo_eval[mgo_nan] = _eval_smooth(mgo_lin, mgo_near, pts[mgo_nan])

    # Iron and MgSiO3 via scatter interpolation (unchanged)
    rho_fe_eval     = _eval_smooth(iron_lin,   iron_near,   pts)
    rho_mgsio3_eval = _eval_smooth(mgsio3_lin, mgsio3_near, pts)


    # 3. Perform Volume Mixing
    p_lin = 10**pts[:, 0]
    z_rock = 1.0 - iron_fraction
    rho_mix = np.zeros_like(rho_fe_eval)

    # --- Regime A: Low Pressure (P <= 10 Mbar) ---
    mask_low = p_lin <= 1e7
    vol_low = iron_fraction / rho_fe_eval[mask_low] + z_rock / rho_mgsio3_eval[mask_low]
    rho_mix[mask_low] = 1.0 / vol_low

    # --- Regime B: High Pressure (P > 10 Mbar) ---
    mask_high = p_lin > 1e7
    mgo_mass_ratio = 40.3044 / 100.389  # Mass of MgO / Mass of MgSiO3
    
    # Calculate effective mass fractions for the remaining solid mixture
    retained_mass = iron_fraction + (z_rock * mgo_mass_ratio)
    z_fe_eff = iron_fraction / retained_mass
    z_mgo_eff = (z_rock * mgo_mass_ratio) / retained_mass
    
    vol_high = z_fe_eff / rho_fe_eval[mask_high] + z_mgo_eff / rho_mgo_eval[mask_high]
    rho_mix[mask_high] = 1.0 / vol_high

    # 4.A. Sanitise: NaN/inf/zero entries cause silent errors in bilinear interp.
    #    Identify bad cells and fill them from nearest valid neighbours before
    #    building the smooth interpolator.
    bad = ~np.isfinite(rho_mix) | (rho_mix <= 0)
    if np.any(bad):
        good_mask = ~bad
        filler = NearestNDInterpolator(pts[good_mask], rho_mix[good_mask])
        rho_mix[bad] = filler(pts[bad])

    # 4.B. Reshape to 2-D: meshgrid row = T, column = P  →  (n_T, n_P)
    rho_grid_2d = rho_mix.reshape(len(logt_eval), len(logp_eval))

    # 4.C. Build smooth bilinear interpolator on the regular grid.
    #    bounds_error=False + fill_value=None  →  linear extrapolation at edges
    #    (query_core_eos will catch non-physical results anyway).
    mixed_interp = RegularGridInterpolator(
        (logt_eval, logp_eval), rho_grid_2d,
        method='cubic', bounds_error=False, fill_value=None
    )

    def interpolator_wrapper(log_p, t):
        """
        t  : linear temperature in K  (query_core_eos passes t_clamped, not log_t)
        log_p : log10(P / bar)
        Returns linear density in kg/m³. Handles both scalars and arrays.
        """
        # 1. Clip and log the temperature
        log_t = np.log10(np.clip(t, 300.0, 100000.0))
        
        # 2. Ensure inputs are arrays so we can stack them as (N, 2) coordinates
        log_t_arr = np.atleast_1d(log_t)
        log_p_arr = np.atleast_1d(log_p)
        pts = np.column_stack((log_t_arr, log_p_arr))
        
        # 3. Evaluate the scipy interpolator
        vals = mixed_interp(pts)
        
        # 4. Clean up non-physical values
        bad = ~np.isfinite(vals) | (vals <= 0)
        vals[bad] = np.nan
        
        # 5. Return a standard float if a scalar was passed (for the integrator),
        #    otherwise return the numpy array (for vectorized plotting)
        if np.isscalar(log_p) or np.ndim(log_p) == 0:
            return float(vals[0]) if not bad[0] else np.nan
        
        return vals

    _CORE_CACHE[cache_key] = interpolator_wrapper
    return interpolator_wrapper

@time_it
def get_rock_interpolator(base_dir: str = str(DATA_DIR), debug: bool = False) -> LinearNDInterpolator:
    """Legacy wrapper function. Returns a core interpolator with 0% Iron (Pure Rock)."""
    return get_core_interpolator(iron_fraction=0.0)

@time_it
def query_core_eos(log_p: float, log_t: float, iron_fraction: float = 0.33) -> float:
    interp = get_core_interpolator(iron_fraction)
    if interp is None:
        return np.log10(4000.0)

    t_val = 10 ** log_t
    t_clamped = np.clip(t_val, _ROCK_BOUNDS['t_min'] + 1.0, _ROCK_BOUNDS['t_max'] - 1.0)

    # Pass LINEAR T — interpolator_wrapper applies log10 internally
    res = interp(log_p, t_clamped)

    if res is None or not np.isfinite(res) or res <= 0:
        p_bar = 10 ** log_p
        extrap_rho = 12000.0 * ((p_bar / 1e6) ** 0.3) if p_bar > 1e6 else 12000.0
        return np.log10(extrap_rho)

    # Return log10(density) — integrate_core does 10**rho_log
    return np.log10(float(res))


def query_rock_eos(log_p: float, log_t: float) -> float:
    """Legacy helper for backwards compatibility. Queries pure rock EOS."""
    return query_core_eos(log_p, log_t, iron_fraction=0.0)


# =============================================================================
# DIRECT WATER EOS
# =============================================================================
@time_it
def get_water_interpolators_complete(base_dir: str = str(DATA_DIR)) -> dict:
    """
    Builds direct 2D interpolators for high-pressure water/ice phases.
    """
    global _WATER_INTERP
    if _WATER_INTERP:
        return _WATER_INTERP
        
    raw = load_all_raw_data(base_dir)
    if 'H2O' not in raw:
        return None
        
    data = raw['H2O']
    
    # Filter physical domains
    mask = (data[:, 0] > 1e-10) & (data[:, 1] > 0) & (data[:, 2] > 0)
    clean = data[mask]
    
    points_log = np.log10(clean[:, :2])
    rho_log = np.log10(clean[:, 2])
    s_val = clean[:, 3]
    
    _WATER_INTERP = {
        'rho': LinearNDInterpolator(points_log, rho_log, rescale=True),
        'S': LinearNDInterpolator(points_log, s_val, rescale=True),
        
        'rho_near': NearestNDInterpolator(points_log, rho_log, rescale=True),
        'S_near': NearestNDInterpolator(points_log, s_val, rescale=True),
        
        'points': points_log,
        'Rho_values': rho_log,
        'S_values': s_val
    }
    
    return _WATER_INTERP


# =============================================================================
# FLUID MIXING (Hydrogen + Helium + Heavy Elements)
# =============================================================================

_ALIGNED_ENDMEMBERS = {}

@time_it
def get_mix_table(z_val: float, y_ratio: float = 0.26, base_dir: str = str(DATA_DIR)) -> np.ndarray:
    """
    Generates a blended (H/He/Z) fluid table.
    The H/He mass ratio is dynamically piloted by the y_ratio parameter.
    """
    global _MIXED_CACHE, _ALIGNED_ENDMEMBERS
    
    cache_key = (round(z_val, 4), round(y_ratio, 4))
    if cache_key in _MIXED_CACHE:
        return _MIXED_CACHE[cache_key]

    raw = load_all_raw_data(base_dir)
    if not raw or 'H' not in raw:
        return None

    x_frac = (1.0 - y_ratio) * (1.0 - z_val)
    y_frac = y_ratio * (1.0 - z_val)
    z_frac = z_val

    base_grid_lin = raw['H'][:, :2]
    mask = (base_grid_lin[:, 0] > 0) & (base_grid_lin[:, 1] > 0)
    base_grid_log = np.log10(base_grid_lin[mask])

    def get_component_props(comp_key):
        # 1. Check if we already aligned this pure table
        if comp_key in _ALIGNED_ENDMEMBERS:
            return _ALIGNED_ENDMEMBERS[comp_key]
            
        if comp_key not in raw:
            return None, None
            
        data = raw[comp_key]
        pts_log = np.log10(data[:, :2])
        rho_val = np.log10(data[:, 2])
        s_val = data[:, 3]
        
        # 2. Fast Path: If the grid is naturally aligned
        if pts_log.shape == base_grid_log.shape and np.allclose(pts_log, base_grid_log, atol=1e-5):
            res = (rho_val, s_val)
        else:
            # 3. Slow Path: Interpolate, but only ONCE globally!
            print(f"     [Optimizing] Aligning {comp_key} to master grid. This only happens ONCE!")
            r = interpolate_table(pts_log, rho_val, base_grid_log)
            s = interpolate_table(pts_log, s_val, base_grid_log)
            res = (r, s)
            
        _ALIGNED_ENDMEMBERS[comp_key] = res
        return res

    rho_h, s_h = get_component_props('H')
    rho_he, s_he = get_component_props('He')
    if rho_he is None:
        rho_he, s_he = rho_h, s_h
        
    rho_z, s_z = get_component_props('H2O')
    if rho_z is None:
        rho_z, s_z = rho_h + 0.7, s_h

    # Additive mixing rules
    vol_mix = (x_frac / 10**rho_h) + (y_frac / 10**rho_he) + (z_frac / 10**rho_z)
    rho_mix = 1.0 / vol_mix
    s_mix = x_frac * s_h + y_frac * s_he + z_frac * s_z
    
    mixed_data = np.column_stack((base_grid_log[:, 0], base_grid_log[:, 1], np.log10(rho_mix), s_mix))
    _MIXED_CACHE[cache_key] = mixed_data 
    
    return mixed_data

@time_it
def generate_fluid_interpolators(z_profile: np.ndarray, y_ratio: float = 0.26, base_dir: str = str(DATA_DIR), debug: bool = False) -> dict:
    """Pre-computes 2D interpolators for every Z-step, using the dynamic Y ratio."""
    unique_z = np.unique(z_profile)
    stack = {}
    
    if debug:
        print(f"\n[DEBUG] Fluid Stack: Generating interpolators for {len(unique_z)} unique Z layers...")
    
    # 1. Grab the first table just to extract the grid coordinates
    first_table = get_mix_table(unique_z[0], y_ratio, base_dir)
    if first_table is None:
        return stack
        
    points = first_table[:, :2]
    
    # 2. THE MEGA-OPTIMIZATION: Compute the Delaunay mesh ONCE.
    if debug: print("     [Optimizing] Building Master Delaunay Triangulation...")
    master_triangulation = Delaunay(points)
    
    for i, z in enumerate(unique_z):
        table = get_mix_table(z, y_ratio, base_dir) 
        if table is None:
            continue
            
        rho_vals = table[:, 2]
        s_vals = table[:, 3]
        
        # 3. Pass the pre-computed mesh directly to the interpolator
        # REMOVED: rescale=True
        stack[z] = {
            'rho': LinearNDInterpolator(master_triangulation, rho_vals),
            'S': LinearNDInterpolator(master_triangulation, s_vals),
            'points': points,
            'Rho_values': rho_vals,
            'S_values': s_vals
        }
            
    if debug:
        print("[DEBUG] Fluid Stack: ALL LAYERS SUCCESSFUL.")    
    return stack


# =============================================================================
# KD-TREE ADIABAT STEPPER
# =============================================================================

class RobustAdiabatStepper:
    """
    A multidimensional root-finding algorithm designed to step along planetary adiabats.
    
    Rather than relying on inverted 1D interpolations (which frequently fail near 
    jagged phase transitions), this algorithm queries a 3D KD-Tree of the local 
    phase space (P, T, S) and performs a local linear least-squares regression 
    to precisely locate the temperature required to maintain constant entropy 
    at a target pressure.
    """
    
    def __init__(self, layer_data: dict):
        """
        Initializes the stepper with the thermodynamic data of a specific Z-layer.
        """
        self.points = layer_data['points']
        self.rho_interp = layer_data['rho']
        self.s_vals = layer_data['S_values']
        self.rho_vals = layer_data['Rho_values']
        
        # Build spatial tree for rapid nearest-neighbor lookup
        self.tree = cKDTree(self.points)

    def get_state(self, p_log_target: float, t_log_guess: float, s_target: float) -> tuple[float, float]:
        """
        Calculates the thermodynamic state (T, Rho) at a new pressure step 
        while enforcing an adiabatic (constant entropy) constraint.
        """
        # 1. Fetch the 10 closest physical points in the local (P, T) phase space
        d, idxs = self.tree.query([p_log_target, t_log_guess], k=10)
        
        if np.any(np.isinf(d)):
            return self.points[idxs[0], 1], self.rho_vals[idxs[0]]

        # Extract the local manifold geometry
        nb_p = self.points[idxs, 0]
        nb_t = self.points[idxs, 1]
        nb_s = self.s_vals[idxs]

        # 2. Local Linear Regression via Least-Squares (OPTIMIZED)
        # Model: S(P, T) ≈ a*P + b*T + c
        p_m, t_m, s_m = np.mean(nb_p), np.mean(nb_t), np.mean(nb_s)
        
        # Explicitly build X^T * X for a 2-variable regression (P, T)
        nb_p_c = nb_p - p_m
        nb_t_c = nb_t - t_m
        nb_s_c = nb_s - s_m
        
        S_pp = np.dot(nb_p_c, nb_p_c)
        S_tt = np.dot(nb_t_c, nb_t_c)
        S_pt = np.dot(nb_p_c, nb_t_c)
        
        S_ps = np.dot(nb_p_c, nb_s_c)
        S_ts = np.dot(nb_t_c, nb_s_c)
        
        det = S_pp * S_tt - S_pt * S_pt
        
        if abs(det) < 1e-12:
            a, b = 0.0, 0.0
        else:
            a = (S_tt * S_ps - S_pt * S_ts) / det
            b = (S_pp * S_ts - S_pt * S_ps) / det

        # 3. Root Finding
        if abs(b) < 1e-5:
            t_pred = t_log_guess
        else:
            t_pred = t_m + (s_target - s_m - a * (p_log_target - p_m)) / b

        # 4. Enforce Physical Clamping
        t_min, t_max = np.min(nb_t), np.max(nb_t)
        t_pred = np.clip(t_pred, t_min - 0.5, t_max + 0.5)

        if np.isnan(t_pred):
            print("Temperature nan guard uses in stepper") 
            t_pred = t_log_guess

        # 5. Extract final density
        rho_pred = float(self.rho_interp(p_log_target, t_pred))
        
        if np.isnan(rho_pred):
            rho_pred = self.rho_vals[idxs[0]]
            
        return t_pred, rho_pred