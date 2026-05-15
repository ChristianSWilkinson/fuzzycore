"""
Utility Functions for Planetary Interior Modeling.

This module provides helper functions and classes for generating compositional 
gradients, integrating exact mass profiles, calculating thermal properties, 
and handling execution locks during parallelized parameter sweeps.
"""

import numpy as np
import time
import logging
from functools import wraps

from . import constants as c

def time_it(func):
    """A decorator that prints the execution time of the function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        # Calculate time
        elapsed_ms = (end - start) * 1000
        if elapsed_ms > 1.0: # Only log things that take more than 1 millisecond
            #print(f"⏱️ [TIMER] {func.__name__} executed in {elapsed_ms:.2f} ms")
            pass
            
        return result
    return wrapper

def generate_linear_z_profile(
    n_layers: int = 25,
    z_base: float = 0.02,
    z_core: float = 0.98,
    **_kwargs,
) -> np.ndarray:
    """
    Linear Z gradient from atmosphere (index 0) to core boundary (index -1).

    Same index convention as `generate_gaussian_z_profile`. Monotonic.

    Args:
        n_layers: Number of discrete layers. <=1 returns a single-element
            array `[z_base]` (well-mixed adiabatic envelope).
        z_base: Z at the atmospheric surface.
        z_core: Z at the inner core-envelope interface.
        **_kwargs: Ignored; accepted for `generate_z_profile` compatibility.
    """
    if n_layers is None or n_layers <= 1:
        return np.array([z_base])
    return np.clip(np.linspace(z_base, z_core, n_layers), 0.0, 0.99)


def generate_inverted_z_profile(
    n_layers: int = 100,
    z_base: float = 0.02,
    z_core: float = 0.98,
    z_atm: float = 0.08,
    atm_depth: float = 0.15,
    transition_width: float = 0.03,
    sigma: float = 0.15,
    **_kwargs,
) -> np.ndarray:
    """
    Non-monotonic Z profile: enriched outer atmosphere -> depleted shell ->
    rising to a dilute core in the deep interior.

    Models a Jupiter-style Z inversion (e.g. molecular envelope enriched in
    heavies above a depleted radiative/He-rain layer, then dilute core):

        index 0  (surface)        Z ≈ z_atm   (enriched)
        index ~atm_depth*n        Z ≈ z_base  (depleted shell)
        index -1 (core boundary)  Z ≈ z_core  (dilute core)

    Requires the spatial-order layer builder in `build_staircase_envelope`
    (see physics.py patch) — the old sorted-unique logic does not handle
    non-monotonic profiles correctly.

    Args:
        n_layers: Number of discrete layers.
        z_base: Depleted-shell baseline.
        z_core: Peak Z at the core boundary.
        z_atm: Enriched Z at the very top of the atmosphere. Set equal to
            z_base to recover the pure Gaussian shape.
        atm_depth: Fraction (0–1) of the envelope occupied by the enriched
            atmosphere.
        transition_width: Tanh transition width between z_atm and z_base, in
            normalised depth units. Smaller = sharper step.
        sigma: Width of the inner Gaussian rise toward z_core (same meaning
            as in `generate_gaussian_z_profile`).
    """
    if n_layers is None or n_layers <= 1:
        return np.array([z_atm])

    x = np.linspace(0.0, 1.0, n_layers)  # 0 = surface, 1 = core

    # Inner Gaussian rise (same sub-grid amplitude correction as the
    # canonical Gaussian generator, so narrow-sigma cases stay mass-conserved)
    gaussian_area = sigma * np.sqrt(np.pi / 2.0)
    dx = 1.0 / max(1, n_layers - 1)
    amplitude_scaler = min(1.0, gaussian_area / dx)
    dynamic_z_core = z_base + (z_core - z_base) * amplitude_scaler
    inner = z_base + (dynamic_z_core - z_base) * np.exp(
        -((x - 1.0) ** 2) / (2.0 * sigma ** 2)
    )

    # Outer enriched atmosphere: smooth tanh step from z_atm down to z_base
    tw = max(transition_width, 1e-6)
    atm_weight = 0.5 * (1.0 - np.tanh((x - atm_depth) / tw))

    return np.clip(inner + (z_atm - z_base) * atm_weight, 0.0, 0.99)


def generate_z_profile(
    profile_type: str = "gaussian",
    n_layers: int = 25,
    **kwargs,
) -> np.ndarray:
    """
    Unified factory for Z profiles. Use this as the single entry point
    going forward — the older `generate_gaussian_z_profile` is kept as a
    backward-compatible alias.

    Profile types:
        'gaussian'  Classic falloff from core. Uses sigma, z_base, z_core.
        'linear'    Linear gradient. Uses z_base, z_core.
        'inverted'  Enriched atmosphere over depleted shell, rising to
                    dilute core. Uses z_atm, atm_depth, transition_width,
                    sigma, z_base, z_core. Non-monotonic.
        'custom'    User-supplied array via the `profile=...` kwarg.
                    Pass anything — posterior samples, MCMC chains, hand-
                    drawn shapes. Clipped to [0, 0.99] but otherwise used
                    as-is. Length >= 1, ordered surface -> core.

    Args:
        profile_type: One of the above.
        n_layers: Number of layers (ignored for 'custom').
        **kwargs: Profile-specific parameters; unused ones are silently
            ignored so retrieval samplers can pass a flat parameter dict.

    Returns:
        1D np.ndarray of Z values, ordered surface (idx 0) -> core (idx -1).
    """
    if profile_type == "custom":
        if "profile" not in kwargs:
            raise ValueError(
                "profile_type='custom' requires a 'profile' kwarg "
                "(1D array, surface -> core)."
            )
        profile = np.asarray(kwargs["profile"], dtype=float).ravel()
        if profile.size < 1:
            raise ValueError("Custom 'profile' must be non-empty.")
        return np.clip(profile, 0.0, 0.99)

    dispatch = {
        "gaussian": generate_gaussian_z_profile,
        "linear":   generate_linear_z_profile,
        "inverted": generate_inverted_z_profile,
    }
    if profile_type not in dispatch:
        raise ValueError(
            f"Unknown profile_type='{profile_type}'. "
            f"Choose from {list(dispatch.keys()) + ['custom']}."
        )
    return dispatch[profile_type](n_layers=n_layers, **kwargs)

def generate_gaussian_z_profile(
    n_layers: int = 25,
    sigma: float = 0.15,
    z_base: float = 0.02,
    z_core: float = 0.98,
) -> np.ndarray:
    """
    Generate a heavy element (Z) mass fraction profile following a Gaussian decay.

    This creates a discrete 1D array representing the transition from a 
    metal-rich core boundary out to a metal-poor atmospheric baseline. If 
    `sigma` is set to 0.0 or None, it returns a single-layer profile, triggering 
    the solver to model a purely adiabatic (fully convective) envelope.

    Args:
        n_layers (int, optional): Number of discrete steps/layers in the 
            envelope. Defaults to 25.
        sigma (float, optional): 'Width' of the compositional transition. 
            Smaller values create a sharp boundary; larger values create a 
            highly diffuse "fuzzy" core. If <= 0.0 or None, triggers a purely 
            adiabatic well-mixed envelope. Defaults to 0.15.
        z_base (float, optional): The baseline Z mass fraction at the 
            atmospheric surface. Defaults to 0.02.
        z_core (float, optional): The maximum Z mass fraction at the inner 
            core-envelope interface. Defaults to 0.98.

    Returns:
        np.ndarray: A 1D array containing the Z fraction for each layer, 
        safely clipped between [0.0, 0.99]. If `sigma` <= 0, returns a 
        single-element array `[z_base]`.
    """    
    if sigma is None or sigma <= 0.0:
        return np.array([z_base])

    x = np.linspace(0, 1, n_layers)
    
    # 1. Calculate the spatial grid resolution
    dx = 1.0 / max(1, n_layers - 1)
    
    # 2. Sub-grid analytical averaging
    # The exact area of the right-half of a Gaussian is sigma * sqrt(pi / 2).
    # If this area is smaller than our grid cell (dx), we calculate the 
    # mathematically conserved *average* metallicity of that cell.
    gaussian_area = sigma * np.sqrt(np.pi / 2)
    amplitude_scaler = min(1.0, gaussian_area / dx)
    
    dynamic_z_core = z_base + (z_core - z_base) * amplitude_scaler

    # 3. Generate and scale the profile
    raw_z = np.exp(-((x - 1.0) ** 2) / (2 * sigma ** 2))
    z_profile = z_base + (dynamic_z_core - z_base) * raw_z

    return np.clip(z_profile, 0.0, 0.99)


def evaluate_heavy_element_mass(
    results: dict, 
    z_top_atmosphere: float
) -> float:
    """
    Calculate the total mass of heavy elements (metals/rock) within the planet.

    This evaluates the mass by taking the central pure condensed core and 
    adding the integrated "excess" heavy element mass suspended strictly 
    within the gaseous envelope. It uses the exact integrated mass array 
    to prevent Riemann sum overestimation across downsampled radial steps.

    Args:
        results (dict): The converged planetary structure dictionary returned 
            by the solver. Must contain 'R', 'M', 'Z', 'M_core_actual', 
            and 'R_rock'.
        z_top_atmosphere (float): The baseline heavy element mass fraction 
            of the pristine upper atmosphere.

    Returns:
        float: The absolute total mass of heavy elements in the planet (in kg).
    """
    radius_array = results['R']
    mass_array = results['M']
    z_array = results['Z']

    # Establish the mass and boundary of the condensed solid core
    m_z_core = results.get('M_core_actual', 0.0)
    r_core = results.get('R_rock', 0.0)

    # Exact mass of each discrete spherical shell
    dm = np.diff(mass_array)

    # Create a mask to ONLY integrate the gaseous envelope.
    # Using radius_array[1:] perfectly slices shells ending outside the core.
    env_mask = radius_array[1:] > r_core

    # Extract the exact mass of the envelope shells
    shell_masses = dm[env_mask]

    # Determine the "excess" heavy elements suspended in the envelope
    z_envelope = z_array[1:][env_mask]
    z_excess = np.maximum(0, z_envelope - z_top_atmosphere)

    # Multiply the true mass of each shell by its excess heavy element fraction
    m_z_envelope = np.sum(shell_masses * z_excess)

    return m_z_core + m_z_envelope


class DummyLock:
    """
    A no-operation context manager mimicking a multiprocessing Lock.

    This is utilized during single-threaded executions (e.g., inside Jupyter 
    Notebooks) to satisfy the solver's requirement for a `write_lock` when 
    saving intermediate CSV steps, preventing the need to rewrite the solver 
    logic for different execution environments.
    """

    def __enter__(self):
        """
        Enter the runtime context.
        
        Returns:
            DummyLock: The current instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context without acquiring or releasing any locks.
        """
        pass


def calculate_staircase_dt_ds(results: dict, t_int: float) -> dict:
    """
    Calculate the effective inverse cooling rate (dt/dS) for the planet.
    This calculates the bulk thermal inertia of the fluid envelope, and 
    adds the parameterized thermal inertia of the solid core.
    """
    import numpy as np
    from . import constants as c

    radius_array = results['R']
    temp_array = results['T']
    mass_array = results['M']
    z_array = results['Z']
    s_array = results['S'] 

    radius_planet = radius_array[-1]

    # Calculate the Stefan-Boltzmann denominator (Luminosity / 4*pi)
    denominator = c.SIGMA_SB * (radius_planet ** 2) * (t_int ** 4)

    if denominator <= 0:
        return {'total_dt_ds': np.inf, 'layer_contributions': {}}

    # -------------------------------------------------------------------------
    # 1. ENVELOPE CONTRIBUTION (Fluid thermal inertia)
    # -------------------------------------------------------------------------
    dm = np.diff(mass_array)
    t_shell = (temp_array[:-1] + temp_array[1:]) / 2.0
    z_shell = z_array[1:]
    s_shell = s_array[1:]

    # Mask to isolate the fluid envelope
    env_mask = s_shell > 0.0

    dm_env = dm[env_mask]
    t_shell_env = t_shell[env_mask]
    z_shell_env = z_shell[env_mask]

    integrand = t_shell_env * (dm_env / (4 * np.pi))
    unique_z = np.unique(z_shell_env)

    layer_contributions = {}
    total_dt_ds = 0.0

    for z_val in unique_z:
        mask = np.isclose(z_shell_env, z_val, atol=1e-4)
        layer_integral = np.sum(integrand[mask])
        layer_dt_ds = -(layer_integral / denominator)

        layer_contributions[z_val] = layer_dt_ds
        total_dt_ds += layer_dt_ds

    # -------------------------------------------------------------------------
    # 2. SOLID CORE CONTRIBUTION (Lumped Heat Capacity)
    # -------------------------------------------------------------------------
    m_core = results.get('M_core_actual', 0.0)
    
    if m_core > 0:
        # 🛑 THE FIX: Slice temp_array[1:] to match the N-1 shell-based mask!
        # This correctly retrieves the temperature at the outer edge of the core.
        t_core_surface = temp_array[1:][~env_mask][-1] if not np.all(env_mask) else temp_array[0]
        
        # Specific heat capacity of Rock/Iron mix (approx 800 J/kg/K)
        cv_core = 800.0 
        
        # Scale core thermal inertia relative to gas c_p (~14000 J/kg/K)
        effective_core_weight = cv_core / 14000.0
        
        core_integral = t_core_surface * effective_core_weight * (m_core / (4 * np.pi))
        core_dt_ds = -(core_integral / denominator)
        
        layer_contributions['Core'] = core_dt_ds
        total_dt_ds += core_dt_ds

    return {
        'total_dt_ds': total_dt_ds,
        'layer_contributions': layer_contributions
    }


def verify_ddc_macroscopic_gradient(
    results: dict, 
    t_int: float, 
    lambda_cd: float = 10.0,  # Thermal conductivity (W / m K)
    Ra_T: float = 1e8,        # Modified Thermal Rayleigh Number
    l_H: float = 0.1          # Characteristic mixing length ratio
) -> dict:
    """
    Validates the fuzzycore artificial staircase gradient against the 
    theoretical Double-Diffusive Convection (DDC) scaling laws (Wood et al. 2013).
    """
    from . import constants as c
    
    R = results['R']
    T = results['T']
    Z = results['Z']
    
    # -------------------------------------------------------------------------
    # 1. Extract the Model's Macroscopic Gradient
    # -------------------------------------------------------------------------
    # Isolate the exact radial boundaries of the "Fuzzy Core" 
    # (Where the heavy element mass fraction Z is actively changing)
    z_diff = np.abs(np.diff(Z))
    changing_indices = np.where(z_diff > 1e-5)[0]
    
    if len(changing_indices) < 2:
        return {'valid': False, 'error': "No macroscopic compositional gradient detected."}
        
    idx_bot = changing_indices[0]
    idx_top = changing_indices[-1]
    
    delta_r_macro = R[idx_top] - R[idx_bot]
    delta_T_macro = T[idx_top] - T[idx_bot]
    
    # <dT/dr>_fuzzy (Note: R goes outward, so T drops, yielding a negative gradient)
    grad_fuzzy = delta_T_macro / delta_r_macro
    
    # -------------------------------------------------------------------------
    # 2. Calculate Theoretical DDC Nusselt Number
    # -------------------------------------------------------------------------
    # Empirical relation from Wood et al. (2013)
    Nu_T = 0.02 * Ra_T * (l_H ** 0.34) + 1.0
    
    # -------------------------------------------------------------------------
    # 3. Define the Physical Flux Equation & Local Adiabat
    # -------------------------------------------------------------------------
    # Extract total intrinsic luminosity (L) and local heat flux (F_tot)
    R_surf = R[-1]
    L_tot = c.SIGMA_SB * (R_surf ** 2) * (t_int ** 4) * 4 * np.pi
    r_mean = (R[idx_top] + R[idx_bot]) / 2.0
    F_tot = L_tot / (4 * np.pi * (r_mean ** 2))
    
    # Extract the pure adiabatic gradient <dT/dr>_ad natively from the EOS!
    # We do this by evaluating a spatial step INSIDE a single constant-Z layer,
    # where the fuzzycore integrator is strictly adiabatic by definition.
    grad_ad_list = []
    for i in range(idx_bot, idx_top):
        if np.isclose(Z[i], Z[i+1], atol=1e-5):  # Inside a single layer
            dr = R[i+1] - R[i]
            if dr > 1e-3:  # Prevent divide-by-zero
                grad_ad_list.append((T[i+1] - T[i]) / dr)
                
    if not grad_ad_list:
        raise ValueError("Grid resolution too low to extract a pure adiabatic segment.")
        
    grad_ad = np.mean(grad_ad_list)
        
    # -------------------------------------------------------------------------
    # 4. Solve for the Theoretical DDC Gradient
    # -------------------------------------------------------------------------
    # Inverting the total flux equation: 
    # F_tot = -lambda_cd * ( <dT/dr>_DDC - <dT/dr>_ad ) * Nu_T + ...
    grad_ddc = ( -(F_tot / lambda_cd) + (Nu_T - 1) * grad_ad ) / Nu_T
    
    # -------------------------------------------------------------------------
    # 5. The Verification Proof
    # -------------------------------------------------------------------------
    # A perfect physical match yields a ratio of exactly 1.0
    match_ratio = grad_fuzzy / grad_ddc
    
    return {
        'valid': True,
        'grad_fuzzy': grad_fuzzy,
        'grad_ddc': grad_ddc,
        'grad_ad': grad_ad,
        'Nu_T': Nu_T,
        'F_tot': F_tot,
        'match_ratio': match_ratio
    }

def calculate_thermal_contrast(results: dict) -> dict:
    """
    Evaluates the static thermal blanketing effect of the envelope.
    Calculates the temperature and entropy contrast across the fluid domain.
    """
    temp_array = results['T']
    s_array = results['S']
    
    # Surface conditions (last element of the arrays)
    t_surf = temp_array[-1]
    s_surf = s_array[-1]
    
    # Deep conditions (base of the envelope / top of the rock core)
    env_mask = s_array > 0.0
    
    # The first element [0] of the masked envelope is the deepest part (CMB)
    t_deep = temp_array[env_mask][0] if np.any(env_mask) else t_surf
    s_deep = s_array[env_mask][0] if np.any(env_mask) else s_surf

    return {
        'T_top': t_surf,
        'T_deep': t_deep,
        'S_top' : s_surf,
        'S_deep' : s_deep,
        'Delta_T': t_deep - t_surf,
        'Delta_S': s_deep - s_surf
    }