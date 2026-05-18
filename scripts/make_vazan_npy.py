"""
Vazan Profile Loader & Exporter
===============================

Loads the digitized Jupiter profiles from Vazan, Helled & Guillot (2018)
and converts them from r/R coordinates to the normalized envelope mass
coordinate x in [0, 1] used by fuzzycore's z_profile arrays.

Once processed, it exports the 'vazan_z_x.npy' array so it can be 
consumed by the robustness sweep.
"""

import os
import numpy as np
import pandas as pd

# Jupiter equatorial radius (Vazan use this), in metres
R_JUPITER_M = 7.1492e7

# Kg per Earth mass
M_EARTH_KG  = 5.972e24

# FIXED: Replaced underscores with dots to match your exact file names (e.g. '4.55_Gyr_rho.csv')
AVAILABLE_AGES = {
    '0.050_Gyr': '0.050_Gyr',
    '4.55_Gyr':  '4.55_Gyr',
}


# =============================================================================
# RAW FILE LOADING
# =============================================================================

def _load_euro_csv(path: str) -> pd.DataFrame:
    """Load a semicolon-separated, comma-decimal CSV as a sorted 2-column frame."""
    df = pd.read_csv(path, sep=';', decimal=',', header=None, names=['rR', 'val'])
    df = df.sort_values('rR').drop_duplicates('rR').reset_index(drop=True)
    return df


def load_vazan_profile(data_dir: str, age: str) -> dict:
    if age not in AVAILABLE_AGES:
        raise ValueError(f"age must be one of {list(AVAILABLE_AGES.keys())}")

    tag = AVAILABLE_AGES[age]
    z_path   = os.path.join(data_dir, f"{tag}_z.csv")
    rho_path = os.path.join(data_dir, f"{tag}_rho.csv")
    T_path   = os.path.join(data_dir, f"{tag}_T.csv")

    df_z   = _load_euro_csv(z_path)
    df_rho = _load_euro_csv(rho_path)
    df_T   = _load_euro_csv(T_path)

    # Define a common grid over the intersection of valid ranges.
    rR_min = max(df_z['rR'].min(), df_rho['rR'].min(), df_T['rR'].min())
    rR_max = min(df_z['rR'].max(), df_rho['rR'].max(), df_T['rR'].max())
    if rR_max <= rR_min:
        raise RuntimeError(
            f"Profiles do not overlap in r/R: z={df_z['rR'].min():.3f}-{df_z['rR'].max():.3f}, "
            f"rho={df_rho['rR'].min():.3f}-{df_rho['rR'].max():.3f}, "
            f"T={df_T['rR'].min():.3f}-{df_T['rR'].max():.3f}"
        )

    rR = np.linspace(rR_min, rR_max, 400)

    # Linear interp of each on the common grid
    Z   = np.interp(rR, df_z['rR'],   df_z['val'])
    rho = np.interp(rR, df_rho['rR'], df_rho['val'])
    T   = np.interp(rR, df_T['rR'],   df_T['val'])

    # Clip obvious digitization artefacts
    T = np.clip(T, 1.0, None)            
    rho = np.clip(rho, 1e-4, None)       
    Z = np.clip(Z, 0.0, 1.0)

    return {
        'r_over_R': rR,
        'Z':        Z,
        'rho':      rho,
        'T':        T,
        'age':      age,
    }


# =============================================================================
# r/R  ->  MASS COORDINATE CONVERSION
# =============================================================================

def cumulative_mass_from_rho(r_over_R: np.ndarray,
                             rho_gcc: np.ndarray,
                             R_planet_m: float = R_JUPITER_M) -> np.ndarray:
    r_m   = r_over_R * R_planet_m
    rho_si = rho_gcc * 1.0e3                   
    integrand = 4.0 * np.pi * r_m**2 * rho_si  

    dm = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_m)
    m_kg = np.concatenate(([0.0], np.cumsum(dm)))
    return m_kg


def vazan_z_in_fuzzycore_coords(profile: dict,
                                M_core_Me: float = 0.3,
                                R_planet_m: float = R_JUPITER_M,
                                n_layers_out: int = 50,
                                z_base: float = 0.02) -> dict:
    rR  = profile['r_over_R']
    rho = profile['rho']
    Z   = profile['Z']

    m_kg = cumulative_mass_from_rho(rR, rho, R_planet_m=R_planet_m)
    M_total_kg = m_kg[-1]
    M_total_Me = M_total_kg / M_EARTH_KG

    M_core_kg = M_core_Me * M_EARTH_KG
    if M_core_kg >= M_total_kg:
        raise ValueError(f"Requested M_core ({M_core_Me} M_E) >= total integrated mass ({M_total_Me:.2f} M_E).")
        
    r_core_idx = np.searchsorted(m_kg, M_core_kg)
    if r_core_idx == 0:
        r_core_over_R = rR[0]
    else:
        m_lo, m_hi = m_kg[r_core_idx - 1], m_kg[r_core_idx]
        rR_lo, rR_hi = rR[r_core_idx - 1], rR[r_core_idx]
        frac = (M_core_kg - m_lo) / max(m_hi - m_lo, 1e-30)
        r_core_over_R = rR_lo + frac * (rR_hi - rR_lo)

    env_mask = rR >= r_core_over_R
    m_env_kg = m_kg[env_mask] - M_core_kg
    M_env_kg = m_env_kg[-1]
    Z_env    = Z[env_mask]

    x_env_raw = 1.0 - m_env_kg / max(M_env_kg, 1e-30)

    order = np.argsort(x_env_raw)
    x_sorted = x_env_raw[order]
    z_sorted = Z_env[order]
    
    _, uniq = np.unique(x_sorted, return_index=True)
    x_sorted = x_sorted[uniq]
    z_sorted = z_sorted[uniq]

    x_uniform = np.linspace(0.0, 1.0, n_layers_out)
    z_uniform = np.interp(x_uniform, x_sorted, z_sorted)
    z_uniform = np.clip(z_uniform, 0.0, 0.99)

    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    integrated_Z = float(_trapz(np.maximum(z_uniform - z_base, 0.0), x_uniform))

    return {
        'x_env':              x_uniform,
        'z_profile':          z_uniform,
        'M_total_Me':         M_total_Me,
        'M_env_Me':           M_env_kg / M_EARTH_KG,
        'r_core_over_R':      r_core_over_R,
        'integrated_Z_excess': integrated_Z,
        'm_of_r_kg':          m_kg,
    }

# =============================================================================
# EXECUTION BLOCK (Run to save the .npy file)
# =============================================================================

if __name__ == '__main__':
    # Try the explicit user path first, then gracefully fallback if ran from a different directory
    if os.path.exists('./data/Vazan/'):
        DATA_DIR = './data/Vazan/'
    elif os.path.exists('../data/Vazan/'):
        DATA_DIR = '../data/Vazan/'
    else:
        print("[!] Could not find the Vazan data folder. Please make sure the path is correct.")
        exit(1)

    print(f"Loading Vazan 4.55 Gyr profile from {DATA_DIR}...")
    
    try:
        # 1. Load the raw profiles
        raw_profile = load_vazan_profile(DATA_DIR, '4.55_Gyr')
        
        # 2. Convert to fuzzycore x-coordinates (50 layers)
        converted = vazan_z_in_fuzzycore_coords(raw_profile, n_layers_out=50)
        
        # 3. Extract the z_profile array and save it as .npy
        z_array = converted['z_profile']
        out_file = 'vazan_z_x.npy'
        np.save(out_file, z_array)
        
        print(f"[✓] Success! Processed Jupiter mass: {converted['M_total_Me']:.2f} M_Earth")
        print(f"[✓] Integrated Z excess: {converted['integrated_Z_excess']:.4f}")
        print(f"[✓] Saved shape {z_array.shape} array to {os.path.abspath(out_file)}")
        print("\nYou can now safely run your Jupiter Robustness SLURM job!")
        
    except Exception as e:
        print(f"[X] Error processing profiles: {e}")