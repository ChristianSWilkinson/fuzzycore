"""
Z-Profile Zoo
=============

Alternative heavy-element distribution profiles Z(x), where x in [0, 1] is
the normalized envelope mass coordinate with the same convention as
fuzzycore.utils.generate_gaussian_z_profile:

    x = 0  ->  outer atmosphere (z_base)
    x = 1  ->  inner core-envelope boundary (peak Z)

Drop this file next to your run script and `import z_profile_zoo as zoo`.
"""

import numpy as np
import os

def _grid(n_layers: int) -> np.ndarray:
    """Standard mass coordinate: x=0 outer, x=1 inner."""
    return np.linspace(0.0, 1.0, n_layers)

# =============================================================================
# 1. Gaussian
# =============================================================================
def gaussian_profile(n_layers: int = 50, sigma: float = 0.15,
                     z_base: float = 0.02, z_core: float = 0.99) -> np.ndarray:
    if sigma is None or sigma <= 0.0:
        return np.array([z_base])
    x = _grid(n_layers)
    dx = 1.0 / max(1, n_layers - 1)
    gaussian_area = sigma * np.sqrt(np.pi / 2.0)
    amplitude_scaler = min(1.0, gaussian_area / dx)
    dynamic_z_core = z_base + (z_core - z_base) * amplitude_scaler
    raw_z = np.exp(-((x - 1.0) ** 2) / (2.0 * sigma ** 2))
    z = z_base + (dynamic_z_core - z_base) * raw_z
    return np.clip(z, 0.0, 0.99)

# =============================================================================
# 2. Step function (sharp-core limit)
# =============================================================================
def step_profile(n_layers: int = 50, frac_inner: float = 0.30,
                 z_base: float = 0.02, z_core: float = 0.99) -> np.ndarray:
    x = _grid(n_layers)
    z = np.where(x >= (1.0 - frac_inner), z_core, z_base)
    return np.clip(z, 0.0, 0.99)

# =============================================================================
# 3. Sigmoid (smoothed step)
# =============================================================================
def sigmoid_profile(n_layers: int = 50, center: float = 0.75,
                    sharpness: float = 20.0,
                    z_base: float = 0.02, z_core: float = 0.99) -> np.ndarray:
    x = _grid(n_layers)
    z = z_base + (z_core - z_base) / (1.0 + np.exp(-sharpness * (x - center)))
    return np.clip(z, 0.0, 0.99)

# =============================================================================
# 4. Exponential decay
# =============================================================================
def exponential_profile(n_layers: int = 50, scale: float = 0.15,
                        z_base: float = 0.02, z_core: float = 0.99) -> np.ndarray:
    x = _grid(n_layers)
    z = z_base + (z_core - z_base) * np.exp(-(1.0 - x) / max(scale, 1e-6))
    return np.clip(z, 0.0, 0.99)

# =============================================================================
# 5. Bi-linear
# =============================================================================
def bilinear_profile(n_layers: int = 50, breakpoint: float = 0.60,
                     z_base: float = 0.02, z_core: float = 0.99) -> np.ndarray:
    x = _grid(n_layers)
    z = np.where(
        x < breakpoint,
        z_base,
        z_base + (z_core - z_base) * (x - breakpoint) / max(1.0 - breakpoint, 1e-6),
    )
    return np.clip(z, 0.0, 0.99)

# =============================================================================
# 6. Vazan-style discrete staircase
# =============================================================================
def vazan_staircase_profile(n_layers: int = 50, z_base: float = 0.02, steps=None) -> np.ndarray:
    if steps is None:
        steps = [
            (0.00, 0.25, 0.08),
            (0.25, 0.50, 0.20),
            (0.50, 0.75, 0.35),
            (0.75, 0.90, 0.55),
            (0.90, 1.00, 0.95),
        ]
    x = _grid(n_layers)
    z = np.full_like(x, z_base)
    for x_start, x_end, z_val in steps:
        mask = (x >= x_start) & (x < x_end + 1e-9)
        z[mask] = z_val
    return np.clip(z, 0.0, 0.99)

# =============================================================================
# 7. True Digitized Vazan Profile
# =============================================================================
def vazan_digitized_profile(z_array_path='vazan_z_x.npy', n_layers=50, z_base=0.02, **kwargs) -> np.ndarray:
    search_paths = [
        z_array_path,
        os.path.join('..', 'scripts', z_array_path),
        os.path.join('..', 'data', z_array_path),
        os.path.join(os.path.dirname(__file__), z_array_path)
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            z = np.load(path)
            if len(z) != n_layers:
                x_old = np.linspace(0, 1, len(z))
                x_new = np.linspace(0, 1, n_layers)
                z = np.interp(x_new, x_old, z)
            return np.clip(z, 0.0, 0.99)
            
    raise FileNotFoundError(f"Could not locate '{z_array_path}' in any of the search paths. Please ensure the array was saved.")

# =============================================================================
# Shared diagnostic
# =============================================================================
def compute_integrated_Z(z_profile, z_base: float = 0.02) -> float:
    z = np.asarray(z_profile, dtype=float)
    if z.size <= 1: return 0.0
    x = np.linspace(0.0, 1.0, z.size)
    return float(np.trapz(np.maximum(z - z_base, 0.0), x))

# =============================================================================
# Convenience dispatcher
# =============================================================================
PROFILE_GENERATORS = {
    'gaussian':        gaussian_profile,
    'step':            step_profile,
    'sigmoid':         sigmoid_profile,
    'exponential':     exponential_profile,
    'bilinear':        bilinear_profile,
    'vazan':           vazan_staircase_profile,
    'vazan_digitized': vazan_digitized_profile,
}

def build(family: str, **kwargs) -> np.ndarray:
    if family not in PROFILE_GENERATORS:
        raise ValueError(f"Unknown family '{family}'. Available: {sorted(PROFILE_GENERATORS.keys())}")
    return PROFILE_GENERATORS[family](**kwargs)