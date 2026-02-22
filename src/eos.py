import os
import numpy as np
import pandas as pd
import scipy.spatial
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from . import constants as const

# =============================================================================
# GLOBAL CACHE
# =============================================================================
# Memory stores to prevent redundant loading and interpolation of heavy EOS tables.
_RAW_TABLES = {}
_MIXED_CACHE = {}
_CORE_CACHE = {}
_WATER_INTERP = None
_ROCK_BOUNDS = {'t_min': 300.0, 't_max': 50000.0}


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


# =============================================================================
# DATA LOADING
# =============================================================================

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


def load_all_raw_data(base_dir: str = "../data/EOS") -> dict:
    """
    Central function to load all fundamental EOS tables (H, He, H2O, Rock, Iron) 
    into memory. Only loads from disk on the first invocation.

    Parameters
    ----------
    base_dir : str, optional
        The root directory containing the EOS data files (default is "../data/EOS").

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
    h_path = os.path.join(base_dir, "DirEOS2021/TABLE_H_TP_v1")
    he_path = os.path.join(base_dir, "DirEOS2021/TABLE_HE_TP_v1")
    h2o_path = os.path.join(base_dir, "h2o-abinitio.dat")
    rock_path = os.path.join(base_dir, "aneosRock.dat")
    iron_path = os.path.join(base_dir, "aneosIron.dat")

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

    # ---------------------------------------------------------
    # 5. Load Iron (ANEOS)
    # ---------------------------------------------------------
    print("  > Loading Iron...")
    if os.path.exists(iron_path):
        try:
            cols = ['T', 'Rho', 'P', 'U', 'S_erg']
            df = pd.read_csv(iron_path, sep=r'\s+', comment='#', header=None, names=cols)
            df = df[df['P'] > 0]
            iron_arr = df[['P', 'T', 'Rho', 'S_erg']].to_numpy()
            
            iron_arr[:, 2] *= 1000.0
            iron_arr[:, 3] *= 1e-4
            _RAW_TABLES['Iron'] = iron_arr
        except Exception as e:
            print(f"  [Warning] Iron failed: {e}")

    return _RAW_TABLES


# =============================================================================
# INTERPOLATORS
# =============================================================================

def interpolate_table(grid_points: np.ndarray, values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """
    Interpolates scattered 2D EOS data (P, T) using a Delaunay triangulation 
    (LinearNDInterpolator). Implements a robust fallback to Nearest-Neighbor 
    interpolation for points queried just outside the defined convex hull.

    Parameters
    ----------
    grid_points : np.ndarray
        Array of shape (N, 2) containing the known (logP, logT) coordinates.
    values : np.ndarray
        Array of shape (N,) containing the scalar values to interpolate (e.g., Density).
    query_points : np.ndarray
        Array of shape (M, 2) containing the target (logP, logT) coordinates.

    Returns
    -------
    np.ndarray
        Interpolated values at the requested query_points.
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

def get_core_interpolator(iron_fraction: float = 0.33, base_dir: str = "../../data/EOS") -> LinearNDInterpolator:
    """
    Constructs an additive volume mixing interpolator for a condensed solid core.

    Parameters
    ----------
    iron_fraction : float, optional
        Mass fraction of iron in the core. Must be between 0.0 and 1.0. 
        (default is 0.33 for Earth-like bulk composition).
    base_dir : str, optional
        Path to the root directory containing EOS data.

    Returns
    -------
    LinearNDInterpolator
        A callable 2D interpolator object predicting log(Density) given 
        (log_Pressure, log_Temperature). Returns None if base tables are missing.
    """
    global _CORE_CACHE, _ROCK_BOUNDS
    
    # Check cache to avoid redundant, expensive Delaunay generation
    cache_key = round(iron_fraction, 3)
    if cache_key in _CORE_CACHE:
        return _CORE_CACHE[cache_key]

    raw = load_all_raw_data(base_dir)
    if 'Rock' not in raw:
        return None

    rock_data = raw['Rock']
    # Fallback entirely to Rock EOS if Iron EOS is unexpectedly missing
    iron_data = raw.get('Iron', rock_data)

    # Establish safety bounds for thermal clamping during integration
    _ROCK_BOUNDS['t_min'] = rock_data[:, 1].min()
    _ROCK_BOUNDS['t_max'] = rock_data[:, 1].max()

    # Convert coordinates and values to log10 space for interpolation stability
    rock_pts_log = np.log10(rock_data[:, :2])
    rock_rho_log = np.log10(rock_data[:, 2])

    iron_pts_log = np.log10(iron_data[:, :2])
    iron_rho_log = np.log10(iron_data[:, 2])

    # 1. Standardize the grid: Project the Iron EOS onto the exact Rock (P, T) nodes
    iron_rho_at_rock_grid = interpolate_table(iron_pts_log, iron_rho_log, rock_pts_log)

    # 2. Compute Additive Volume Mixing
    # Volume = Mass / Density. Assuming total mass = 1 for mixing fractions.
    x_fe = iron_fraction
    x_rock = 1.0 - x_fe

    vol_mix = (x_fe / 10**iron_rho_at_rock_grid) + (x_rock / 10**rock_rho_log)
    rho_mix = 1.0 / vol_mix

    # 3. Generate and cache the final interpolator
    interp = LinearNDInterpolator(rock_pts_log, np.log10(rho_mix), rescale=True)
    _CORE_CACHE[cache_key] = interp
    
    return interp


def get_rock_interpolator(base_dir: str = "../data/EOS") -> LinearNDInterpolator:
    """Legacy wrapper function. Returns a core interpolator with 0% Iron (Pure Rock)."""
    return get_core_interpolator(iron_fraction=0.0, base_dir=base_dir)


def query_core_eos(log_p: float, log_t: float, iron_fraction: float = 0.33) -> float:
    """
    Safely queries the mixed core interpolator with boundary guards.

    Parameters
    ----------
    log_p : float
        Base-10 logarithm of pressure (in bar).
    log_t : float
        Base-10 logarithm of temperature (in K).
    iron_fraction : float, optional
        Mass fraction of iron (default 0.33).

    Returns
    -------
    float
        Base-10 logarithm of the resultant core density (kg/m^3).
    """
    interp = get_core_interpolator(iron_fraction)
    # Absolute failure fallback
    if interp is None:
        return np.log10(4000.0)

    # Convert out of log space to apply safety clamps
    t_val = 10 ** log_t
    t_clamped = np.clip(t_val, _ROCK_BOUNDS['t_min'] + 1.0, _ROCK_BOUNDS['t_max'] - 1.0)
    log_t_clamped = np.log10(t_clamped)

    res = interp(log_p, log_t_clamped)
    
    # If out-of-bounds yields NaN, return a standard uncompressed rock proxy
    if np.isnan(res):
        return np.log10(12000.0)
        
    return res


def query_rock_eos(log_p: float, log_t: float) -> float:
    """Legacy helper for backwards compatibility. Queries pure rock EOS."""
    return query_core_eos(log_p, log_t, iron_fraction=0.0)


# =============================================================================
# DIRECT WATER EOS
# =============================================================================

def get_water_interpolators_complete(base_dir: str = "../data/EOS") -> dict:
    """
    Builds direct 2D interpolators for high-pressure water/ice phases.

    Returns
    -------
    dict
        A dictionary containing interpolators for Density ('rho') and Entropy ('S'),
        along with the raw underlying data points.
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
        'points': points_log,
        'Rho_values': rho_log,
        'S_values': s_val
    }
    
    return _WATER_INTERP


# =============================================================================
# FLUID MIXING (Hydrogen + Helium + Heavy Elements)
# =============================================================================

def get_mix_table(z_val: float, base_dir: str = "../data/EOS") -> np.ndarray:
    """
    Generates a blended (H/He/Z) fluid table based on a specific metallicity fraction.
    The H/He ratio is maintained at the solar proto-stellar value (Y / X ≈ 0.26 / 0.74).

    Parameters
    ----------
    z_val : float
        Mass fraction of heavy elements (metals/volatiles) in the fluid layer.
    base_dir : str, optional
        Directory containing the source EOS tables.

    Returns
    -------
    np.ndarray
        A structured array containing [logP, logT, logRho, Entropy] for the mixed fluid.
    """
    if z_val in _MIXED_CACHE:
        return _MIXED_CACHE[z_val]

    raw = load_all_raw_data(base_dir)
    if not raw or 'H' not in raw:
        return None

    # Determine mass fractions
    x_frac = 0.74 * (1.0 - z_val)
    y_frac = 0.26 * (1.0 - z_val)
    z_frac = z_val

    # Establish the master thermodynamic grid using Hydrogen as the base
    base_grid_lin = raw['H'][:, :2]
    mask = (base_grid_lin[:, 0] > 0) & (base_grid_lin[:, 1] > 0)
    base_grid_log = np.log10(base_grid_lin[mask])

    def get_component_props(comp_key):
        """Helper to safely extract and interpolate component properties."""
        if comp_key not in raw:
            return None, None
            
        data = raw[comp_key]
        pts_log = np.log10(data[:, :2])
        rho_val = np.log10(data[:, 2])
        s_val = data[:, 3]
        
        return interpolate_table(pts_log, rho_val, base_grid_log), \
               interpolate_table(pts_log, s_val, base_grid_log)

    # 1. Gather all component properties onto the master grid
    rho_h, s_h = get_component_props('H')
    
    rho_he, s_he = get_component_props('He')
    if rho_he is None:
        # Fallback if Helium is missing
        rho_he, s_he = rho_h, s_h
        
    rho_z, s_z = get_component_props('H2O')
    if rho_z is None:
        # Fallback if Heavy proxy (Water) is missing: simulate dense metal proxy
        rho_z, s_z = rho_h + 0.7, s_h

    # 2. Additive Volume Law for Density
    vol_mix = (x_frac / 10**rho_h) + (y_frac / 10**rho_he) + (z_frac / 10**rho_z)
    rho_mix = 1.0 / vol_mix
    
    # 3. Additive Entropy Mixing Law
    s_mix = x_frac * s_h + y_frac * s_he + z_frac * s_z
    
    # Bundle into final cached array
    mixed_data = np.column_stack((base_grid_log[:, 0], base_grid_log[:, 1], np.log10(rho_mix), s_mix))
    _MIXED_CACHE[z_val] = mixed_data
    
    return mixed_data


def generate_fluid_interpolators(z_profile: np.ndarray, base_dir: str = "../../data/EOS") -> dict:
    """
    Pre-computes and caches 2D interpolators for every unique metallicity step 
    in the planetary envelope's 'staircase' gradient.

    Parameters
    ----------
    z_profile : np.ndarray
        An array containing the discrete heavy element mass fractions ($Z$) for each layer.
    base_dir : str, optional
        Data directory location.

    Returns
    -------
    dict
        A dictionary mapping each unique $Z$ value to its corresponding 
        density and entropy interpolators.
    """
    unique_z = np.unique(z_profile)
    stack = {}
    
    for z in unique_z:
        table = get_mix_table(z, base_dir)
        if table is None:
            continue
            
        points = table[:, :2]
        rho_vals = table[:, 2]
        s_vals = table[:, 3]
        
        stack[z] = {
            'rho': LinearNDInterpolator(points, rho_vals, rescale=True),
            'S': LinearNDInterpolator(points, s_vals, rescale=True),
            'points': points,
            'Rho_values': rho_vals,
            'S_values': s_vals
        }
        
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
        
        Parameters
        ----------
        layer_data : dict
            The output dictionary generated by `generate_fluid_interpolators` 
            for a specific metallicity.
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

        Parameters
        ----------
        p_log_target : float
            The destination pressure in log10(bar).
        t_log_guess : float
            An initial guess for the temperature in log10(K).
        s_target : float
            The required entropy value to maintain the adiabat.

        Returns
        -------
        tuple of (float, float)
            The converged (log10(Temperature), log10(Density)).
        """
        # 1. Fetch the 10 closest physical points in the local (P, T) phase space
        d, idxs = self.tree.query([p_log_target, t_log_guess], k=10)
        
        if np.any(np.isinf(d)):
            return self.points[idxs[0], 1], self.rho_vals[idxs[0]]

        # Extract the local manifold geometry
        nb_p = self.points[idxs, 0]
        nb_t = self.points[idxs, 1]
        nb_s = self.s_vals[idxs]

        # 2. Local Linear Regression via Least-Squares
        # Model: S(P, T) ≈ a*P + b*T + c
        p_m, t_m, s_m = np.mean(nb_p), np.mean(nb_t), np.mean(nb_s)
        
        # Center the data to improve matrix condition number
        matrix_a = np.column_stack((nb_p - p_m, nb_t - t_m, np.ones_like(nb_p)))
        
        try:
            sol = np.linalg.lstsq(matrix_a, nb_s - s_m, rcond=None)[0]
            a, b, c_val = sol
        except Exception:
            # Fallback if matrix is singular (all points lie on a line)
            b = 0

        # 3. Root Finding
        # If the derivative wrt Temperature (b) is virtually flat, the local 
        # phase space is unstable (e.g., inside a phase transition zone). 
        # We invoke an isothermal guard.
        if abs(b) < 1e-5:
            t_pred = t_log_guess
        else:
            # Algebraically solve the regression plane for T given P and S
            t_pred = t_m + (s_target - s_m - a * (p_log_target - p_m)) / b

        # 4. Enforce Physical Clamping
        # Prevent the algorithm from wildly over-extrapolating out of the local cell
        t_min, t_max = np.min(nb_t), np.max(nb_t)
        t_pred = np.clip(t_pred, t_min - 0.5, t_max + 0.5)

        # 5. Extract final density
        rho_pred = float(self.rho_interp(p_log_target, t_pred))
        
        if np.isnan(rho_pred):
            rho_pred = self.rho_vals[idxs[0]]
            
        return t_pred, rho_pred