import os
import numpy as np
import pandas as pd
import scipy.spatial
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from . import constants as const

# --- GLOBAL CACHE ---
_RAW_TABLES = {}
_MIXED_CACHE = {}
_CORE_CACHE = {} 
_WATER_INTERP = None
_ROCK_BOUNDS = {'t_min': 300.0, 't_max': 50000.0}

def clear_mixed_cache():
    global _MIXED_CACHE, _RAW_TABLES, _CORE_CACHE, _WATER_INTERP
    _MIXED_CACHE.clear()
    _RAW_TABLES.clear()
    _CORE_CACHE.clear()
    _WATER_INTERP = None

# --- DATA LOADING ---

def _load_raw_table(name, path, cols, log_cols=False):
    """Generic loader for DirEOS (Hydrogen/Helium) tables."""
    try:
        if not os.path.exists(path):
            print(f"  [Error] File not found: {path}")
            return None
            
        data = np.genfromtxt(path, delimiter='', comments='#', usecols=cols)
        
        if log_cols:
            mask = np.all(data < 100, axis=1) 
            data = data[mask]
            data = 10**data
            
        return data
    except Exception as e:
        print(f"Error loading {name} from {path}: {e}")
        return None

def load_all_raw_data(base_dir="../../data/EOS"):
    global _RAW_TABLES
    if _RAW_TABLES: return _RAW_TABLES
    
    print("--- Loading Raw EOS Tables (From Disk) ---")
    
    h_path = os.path.join(base_dir, "DirEOS2021/TABLE_H_TP_v1")
    he_path = os.path.join(base_dir, "DirEOS2021/TABLE_HE_TP_v1")
    h2o_path = os.path.join(base_dir, "h2o-abinitio.dat")
    rock_path = os.path.join(base_dir, "aneosRock.dat")
    iron_path = os.path.join(base_dir, "aneosIron.dat")
    
    # 1. Hydrogen
    print(f"  > Loading Hydrogen...")
    H = _load_raw_table("H", h_path, [0,1,2,4], log_cols=True)
    if H is not None:
        H[:, 1] *= 1e4             
        H[:, 2] *= 1000.0          
        H[:, 3] *= const.MJ_TO_J   
        H[:, [0, 1]] = H[:, [1, 0]]
        _RAW_TABLES['H'] = H

    # 2. Helium
    print(f"  > Loading Helium...")
    He = _load_raw_table("He", he_path, [0,1,2,4], log_cols=True)
    if He is not None:
        He[:, 1] *= 1e4            
        He[:, 2] *= 1000.0         
        He[:, 3] *= const.MJ_TO_J  
        He[:, [0, 1]] = He[:, [1, 0]]
        _RAW_TABLES['He'] = He
        
    # 3. Water
    print(f"  > Loading Water...")
    if os.path.exists(h2o_path):
        try:
            cols = ['T', 'Rho', 'P', 'U', 'S_erg']
            df = pd.read_csv(h2o_path, sep=r'\s+', comment='#', header=None, names=cols)
            df = df[df['P'] > 0] 
            h2o_arr = df[['P', 'T', 'Rho', 'S_erg']].to_numpy()
            h2o_arr[:, 2] *= 1000.0 
            h2o_arr[:, 3] *= (1e-4) 
            _RAW_TABLES['H2O'] = h2o_arr
        except Exception as e: print(f"  [Warning] H2O failed: {e}")

    # 4. Rock
    print(f"  > Loading Rock...")
    if os.path.exists(rock_path):
        try:
            cols = ['T', 'Rho', 'P', 'U', 'S_erg']
            df = pd.read_csv(rock_path, sep=r'\s+', comment='#', header=None, names=cols)
            df = df[df['P'] > 0]
            rock_arr = df[['P', 'T', 'Rho', 'S_erg']].to_numpy()
            rock_arr[:, 2] *= 1000.0
            rock_arr[:, 3] *= (1e-4) 
            _RAW_TABLES['Rock'] = rock_arr
        except Exception as e: print(f"  [Warning] Rock failed: {e}")

    # 5. Iron
    print(f"  > Loading Iron...")
    if os.path.exists(iron_path):
        try:
            cols = ['T', 'Rho', 'P', 'U', 'S_erg']
            df = pd.read_csv(iron_path, sep=r'\s+', comment='#', header=None, names=cols)
            df = df[df['P'] > 0]
            iron_arr = df[['P', 'T', 'Rho', 'S_erg']].to_numpy()
            iron_arr[:, 2] *= 1000.0
            iron_arr[:, 3] *= (1e-4)
            _RAW_TABLES['Iron'] = iron_arr
        except Exception as e: print(f"  [Warning] Iron failed: {e}")

    return _RAW_TABLES

# --- INTERPOLATORS ---

def interpolate_table(grid_points, values, query_points):
    lin_interp = LinearNDInterpolator(grid_points, values, rescale=True)
    result = lin_interp(query_points)
    if np.any(np.isnan(result)):
        nan_mask = np.isnan(result)
        near_interp = NearestNDInterpolator(grid_points, values, rescale=True)
        result[nan_mask] = near_interp(query_points[nan_mask])
    return result

# --- CORE MIXER (Iron + Rock) ---
def get_core_interpolator(iron_fraction=0.33, base_dir="../../data/EOS"):
    """
    Returns an interpolator for a mixed Rock/Iron core.
    """
    global _CORE_CACHE, _ROCK_BOUNDS
    cache_key = round(iron_fraction, 3)
    if cache_key in _CORE_CACHE: return _CORE_CACHE[cache_key]
    
    raw = load_all_raw_data(base_dir)
    if 'Rock' not in raw: return None
    
    # Base: Rock
    rock_data = raw['Rock']
    # Fallback to Rock if Iron missing
    iron_data = raw.get('Iron', rock_data) 
    
    # Set Bounds
    _ROCK_BOUNDS['t_min'] = rock_data[:, 1].min()
    _ROCK_BOUNDS['t_max'] = rock_data[:, 1].max()

    rock_pts_log = np.log10(rock_data[:, :2])
    rock_rho_log = np.log10(rock_data[:, 2])
    
    iron_pts_log = np.log10(iron_data[:, :2])
    iron_rho_log = np.log10(iron_data[:, 2])
    
    # Interpolate Iron onto Rock Grid
    iron_rho_at_rock_grid = interpolate_table(iron_pts_log, iron_rho_log, rock_pts_log)
    
    # Additive Volume Mixing
    X_fe = iron_fraction
    X_rock = 1.0 - X_fe
    
    vol_mix = (X_fe / 10**iron_rho_at_rock_grid) + (X_rock / 10**rock_rho_log)
    rho_mix = 1.0 / vol_mix
    
    interp = LinearNDInterpolator(rock_pts_log, np.log10(rho_mix), rescale=True)
    _CORE_CACHE[cache_key] = interp
    return interp

# --- RESTORED FUNCTION ---
def get_rock_interpolator(base_dir="../../data/EOS"):
    """Legacy wrapper: returns core interpolator with 0% Iron (Pure Rock)."""
    return get_core_interpolator(iron_fraction=0.0, base_dir=base_dir)

def query_core_eos(log_p, log_t, iron_fraction=0.33):
    interp = get_core_interpolator(iron_fraction)
    if interp is None: return np.log10(4000.0) 
    
    t_val = 10**log_t
    t_clamped = np.clip(t_val, _ROCK_BOUNDS['t_min'] + 1.0, _ROCK_BOUNDS['t_max'] - 1.0)
    log_t_clamped = np.log10(t_clamped)
    
    res = interp(log_p, log_t_clamped)
    if np.isnan(res): return np.log10(12000.0)
    return res

# Helper for pure rock backward compatibility
def query_rock_eos(log_p, log_t):
    return query_core_eos(log_p, log_t, iron_fraction=0.0)

# --- DIRECT WATER ---
def get_water_interpolators_complete(base_dir="../../data/EOS"):
    global _WATER_INTERP
    if _WATER_INTERP: return _WATER_INTERP
    raw = load_all_raw_data(base_dir)
    if 'H2O' not in raw: return None
    data = raw['H2O']
    mask = (data[:, 0] > 1e-10) & (data[:, 1] > 0) & (data[:, 2] > 0)
    clean = data[mask]
    points_log = np.log10(clean[:, :2])
    rho_log = np.log10(clean[:, 2])
    s_val = clean[:, 3]
    _WATER_INTERP = {'rho': LinearNDInterpolator(points_log, rho_log, rescale=True),
                     'S': LinearNDInterpolator(points_log, s_val, rescale=True),
                     'points': points_log, 'Rho_values': rho_log, 'S_values': s_val}
    return _WATER_INTERP

# --- FLUID MIXING ---
def get_mix_table(z_val, base_dir="../../data/EOS"):
    if z_val in _MIXED_CACHE: return _MIXED_CACHE[z_val]
    raw = load_all_raw_data(base_dir)
    if not raw or 'H' not in raw: return None
    X = 0.74 * (1.0 - z_val); Y = 0.26 * (1.0 - z_val); Z = z_val
    base_grid_lin = raw['H'][:, :2]
    mask = (base_grid_lin[:,0] > 0) & (base_grid_lin[:,1] > 0)
    base_grid_log = np.log10(base_grid_lin[mask])
    
    def get_component_props(comp_key):
        if comp_key not in raw: return None, None
        data = raw[comp_key]
        pts_log = np.log10(data[:, :2])
        rho_val = np.log10(data[:, 2]); s_val = data[:, 3] 
        return interpolate_table(pts_log, rho_val, base_grid_log), interpolate_table(pts_log, s_val, base_grid_log)

    rho_H, s_H = get_component_props('H')
    rho_He, s_He = get_component_props('He')
    if rho_He is None: rho_He, s_He = rho_H, s_H
    rho_Z, s_Z = get_component_props('H2O')
    if rho_Z is None: rho_Z, s_Z = rho_H + 0.7, s_H

    vol_mix = (X / 10**rho_H) + (Y / 10**rho_He) + (Z / 10**rho_Z)
    rho_mix = 1.0 / vol_mix
    s_mix = X * s_H + Y * s_He + Z * s_Z
    mixed_data = np.column_stack((base_grid_log[:,0], base_grid_log[:,1], np.log10(rho_mix), s_mix))
    _MIXED_CACHE[z_val] = mixed_data
    return mixed_data

def generate_fluid_interpolators(z_profile, base_dir="../../data/EOS"):
    unique_z = np.unique(z_profile)
    stack = {}
    for z in unique_z:
        table = get_mix_table(z, base_dir)
        if table is None: continue
        points = table[:, :2]; rho_vals = table[:, 2]; s_vals = table[:, 3]
        stack[z] = {'rho': LinearNDInterpolator(points, rho_vals, rescale=True),
                    'S': LinearNDInterpolator(points, s_vals, rescale=True),
                    'points': points, 'Rho_values': rho_vals, 'S_values': s_vals}
    return stack

# --- STEPPER ---
class RobustAdiabatStepper:
    def __init__(self, layer_data):
        self.points = layer_data['points']
        self.rho_interp = layer_data['rho']
        self.s_vals = layer_data['S_values'] 
        self.rho_vals = layer_data['Rho_values']
        self.tree = cKDTree(self.points)
    def get_state(self, p_log_target, t_log_guess, s_target):
        d, idxs = self.tree.query([p_log_target, t_log_guess], k=10)
        if np.any(np.isinf(d)): return self.points[idxs[0], 1], self.rho_vals[idxs[0]]
        nb_p = self.points[idxs, 0]; nb_t = self.points[idxs, 1]; nb_s = self.s_vals[idxs]
        p_m, t_m, s_m = np.mean(nb_p), np.mean(nb_t), np.mean(nb_s)
        A = np.column_stack((nb_p - p_m, nb_t - t_m, np.ones_like(nb_p)))
        try: sol = np.linalg.lstsq(A, nb_s - s_m, rcond=None)[0]; a, b, c = sol
        except: b = 0
        if abs(b) < 1e-5: t_pred = t_log_guess
        else: t_pred = t_m + (s_target - s_m - a*(p_log_target - p_m)) / b
        t_min, t_max = np.min(nb_t), np.max(nb_t)
        t_pred = np.clip(t_pred, t_min - 0.5, t_max + 0.5)
        rho_pred = float(self.rho_interp(p_log_target, t_pred))
        if np.isnan(rho_pred): rho_pred = self.rho_vals[idxs[0]]
        return t_pred, rho_pred