"""
Direct Mass-Coordinate Vazan Loader
===================================
Bypasses density integration by directly loading a digitized plot of 
Heavy Element Fraction (Z) vs. Normalized Planetary Mass (m/M).
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to your new digitized CSV containing m/M and Z
CSV_PATH = '../data/Vazan/4.55_Gyr_z_mass.csv' 
OUT_NPY = 'vazan_z_x.npy'

# Planet definition to establish where the envelope ends and the core begins
M_TOTAL_ME = 317.8
M_CORE_ME  = 10.0   # Matches your Z_profiling_jupiter.py configuration

def load_euro_csv(path: str) -> pd.DataFrame:
    """Load a semicolon-separated, comma-decimal CSV (or standard CSV)."""
    # Try standard comma-separated first
    try:
        df = pd.read_csv(path, header=None, names=['m_M', 'Z'])
        # If it read as one column due to semicolons, it will fail the float conversion
        df['m_M'] = df['m_M'].astype(float) 
    except:
        # Fall back to European format
        df = pd.read_csv(path, sep=';', decimal=',', header=None, names=['m_M', 'Z'])
    
    df = df.sort_values('m_M').drop_duplicates('m_M').reset_index(drop=True)
    return df

def convert():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Could not find CSV at {CSV_PATH}")
        return

    print(f"Loading direct mass-coordinate profile from {CSV_PATH}...")
    df = load_euro_csv(CSV_PATH)
    
    m_frac_data = df['m_M'].values
    z_data = df['Z'].values

    # Determine the mass fraction where the rock/iron core ends
    m_core_frac = M_CORE_ME / M_TOTAL_ME

    # Filter out the solid core (we only want the envelope)
    # Vazan's m/M usually goes from 0 (center) to 1 (surface)
    env_mask = m_frac_data >= m_core_frac
    m_frac_env = m_frac_data[env_mask]
    z_env = z_data[env_mask]

    # Map to fuzzycore's x coordinate:
    # Surface (m/M = 1.0) -> x = 0.0
    # Core Boundary (m/M = m_core_frac) -> x = 1.0
    x_env_raw = (1.0 - m_frac_env) / (1.0 - m_core_frac)

    # Sort to ensure x is strictly increasing for interpolation
    order = np.argsort(x_env_raw)
    x_sorted = x_env_raw[order]
    z_sorted = z_env[order]

    # Interpolate onto fuzzycore's strict 50-layer grid
    x_fuzzy = np.linspace(0.0, 1.0, 50)
    
    # bounds_error=False handles tiny extrapolation differences at the very edges
    f_z = interp1d(x_sorted, z_sorted, bounds_error=False, fill_value=(z_sorted[0], z_sorted[-1]))
    z_fuzzy = f_z(x_fuzzy)
    z_fuzzy = np.clip(z_fuzzy, 0.0, 0.99)

    # Calculate Integrated Z Excess for diagnostic proof
    z_base = 0.02
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    int_Z = float(_trapz(np.maximum(z_fuzzy - z_base, 0.0), x_fuzzy))

    # Save the array
    np.save(OUT_NPY, z_fuzzy)
    print(f"[✓] Array mapped successfully without density integration!")
    print(f"    New Integrated Z Excess: {int_Z:.4f}")
    print(f"[✓] Saved to {os.path.abspath(OUT_NPY)}")

if __name__ == '__main__':
    convert()