"""
Utility Functions for Planetary Interior Modeling

This module provides helper functions and classes for generating compositional 
gradients, integrating mass profiles, and handling single-threaded execution 
locks during structural parameter sweeps.
"""

import numpy as np
from . import constants as c


def generate_gaussian_z_profile(
    n_layers: int = 25, 
    sigma: float = 0.15, 
    z_base: float = 0.02, 
    z_core: float = 0.98
) -> np.ndarray:
    """
    Generates a heavy element (Z) mass fraction profile following a Gaussian decay.
    
    This function creates a discrete 1D array representing the transition from a 
    metal-rich core boundary out to a metal-poor atmospheric baseline. If `sigma` 
    is set to 0.0 or None, it acts as a toggle to return a single-layer profile, 
    triggering the solver to model a purely adiabatic (fully convective) envelope.
    
    Args:
        n_layers (int, optional): Number of discrete steps/layers in the envelope. 
            Defaults to 25.
        sigma (float, optional): 'Width' of the compositional transition. Smaller 
            values (e.g., 0.1) create a sharp boundary; larger values (e.g., 0.3) 
            create a highly diffuse "fuzzy" core. If `<= 0.0` or `None`, triggers 
            a purely adiabatic well-mixed envelope. Defaults to 0.15.
        z_base (float, optional): The baseline Z mass fraction at the atmospheric 
            surface. Defaults to 0.02.
        z_core (float, optional): The maximum Z mass fraction at the inner 
            core-envelope interface. Defaults to 0.98.
            
    Returns:
        np.ndarray: A 1D array containing the Z fraction for each layer, safely 
            clipped between [0.0, 0.99]. If `sigma` <= 0, returns a single-element 
            array `[z_base]`.
    """
    # Trigger for purely adiabatic / well-mixed envelope
    if sigma is None or sigma <= 0.0:
        return np.array([z_base])
        
    # Spatial domain: x goes from 0 (surface) to 1 (core interface)
    x = np.linspace(0, 1, n_layers)
    
    # Generate a raw Gaussian curve centered exactly at the core boundary (x = 1.0)
    # This means raw_z decays as it moves outwards toward the surface (x = 0.0)
    raw_z = np.exp(-((x - 1.0) ** 2) / (2 * sigma ** 2))
    
    # Normalize the raw Gaussian so it spans exactly [0, 1] before applying bounds
    raw_z = (raw_z - raw_z.min()) / (raw_z.max() - raw_z.min())
    
    # Scale and shift the normalized curve to fit between the atmospheric 
    # baseline (z_base) and the dense interior (z_core)
    z_profile = z_base + (z_core - z_base) * raw_z
    
    # Ensure physical validity (Z cannot be less than 0 or strictly 1.0 in fluid)
    return np.clip(z_profile, 0.0, 0.99)


def evaluate_heavy_element_mass(results: dict, z_top_atmosphere: float) -> float:
    """
    Calculates the total mass of heavy elements (metals/rock/water) within the planet.
    
    The function evaluates the mass by taking the central pure condensed core 
    and integrating the "excess" heavy element mass fraction suspended within 
    the gaseous envelope's compositional gradient.
    
    Args:
        results (dict): The integrated planetary profile dictionary returned 
            by the main solver (must contain 'R', 'M', 'Rho', and 'Z' arrays).
        z_top_atmosphere (float): The background heavy element mass fraction 
            (metallicity) of the pure envelope/surface.
            
    Returns:
        float: The total absolute mass of heavy elements in the planet (in kg).
    """
    # Extract physical profiles from the solver results
    R = results['R']
    M = results['M']
    Rho = results['Rho']
    Z = results['Z']
    
    # -------------------------------------------------------------------------
    # 1. Establish the Mass of the Condensed Core
    # -------------------------------------------------------------------------
    m_z_total = results.get('M_core_actual', 0.0)
    
    if m_z_total == 0.0 and len(Z) > 0:
        # Fallback Logic: If the explicit core mass wasn't saved, locate the 
        # physical interface where the composition drops below pure rock/ice (Z < 1.0)
        idx_int = np.where(Z < 1.0)[0]
        if len(idx_int) > 0:
            m_z_total = M[idx_int[0]]
        else:
            # If Z never drops below 1.0, the entire planet is a solid rock
            m_z_total = M[-1]

    # -------------------------------------------------------------------------
    # 2. Integrate the Dilute Envelope
    # -------------------------------------------------------------------------
    # Compute the radial thickness of each integration step
    dr = np.diff(R)
    
    # Calculate the total fluid mass within each discrete spherical shell.
    # Equation: dM = 4 * pi * r^2 * rho * dr
    # Array slicing [:-1] aligns the arrays with the spatial differences (dr).
    shell_masses = 4 * np.pi * R[:-1]**2 * Rho[:-1] * dr
    
    # Determine the "excess" heavy elements suspended in the envelope.
    # We subtract the baseline atmospheric metallicity because that is considered
    # part of the standard well-mixed envelope, not the dilute core.
    z_envelope = Z[:-1]
    z_excess = np.maximum(0, z_envelope - z_top_atmosphere)
    
    # Multiply the mass of each shell by its excess heavy element fraction
    m_z_envelope = np.sum(shell_masses * z_excess)
    
    # Add the integrated envelope metals to the solid core
    m_z_total += m_z_envelope
    
    return m_z_total


class DummyLock:
    """
    A no-operation (no-op) context manager designed to mimic a multiprocessing Lock.
    
    This is utilized during single-threaded executions (e.g., inside Jupyter 
    Notebooks or simple debugging scripts) to satisfy the solver's requirement 
    for a `write_lock` when saving intermediate CSV steps, preventing the need 
    to rewrite the solver logic for different environments.
    """
    
    def __enter__(self):
        """Enter the runtime context without acquiring any actual thread lock."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context without needing to release anything."""
        pass


def calculate_staircase_dt_ds(results: dict, t_int: float) -> dict:
    """
    Calculates the inverse cooling rate (dt/dS) for a non-adiabatic staircase model.
    
    This function expands the thermal integral by grouping the interior into 
    discrete compositional (Z) layers. It returns both the global homologous 
    cooling rate and a breakdown of how much thermal inertia is contributed 
    by each specific layer in the dilute core.
    
    Args:
        results (dict): The converged planetary structure dictionary returned 
            by the solver (must contain 'R', 'T', 'Rho', and 'Z' arrays).
        t_int (float): The internal effective temperature driving the cooling 
            luminosity (in Kelvin).
            
    Returns:
        dict: A dictionary containing:
            - 'total_dt_ds' (float): The global inverse cooling rate.
            - 'layer_contributions' (dict): A mapping of Z-fraction to its 
              specific contribution to dt/dS.
    """
    R = results['R']
    T = results['T']
    Rho = results['Rho']
    Z = results['Z']
    
    R_p = R[-1]
    
    # Calculate the Stefan-Boltzmann denominator (Luminosity / 4*pi)
    denominator = c.SIGMA_SB * (R_p ** 2) * (t_int ** 4)
    
    if denominator <= 0:
        return {'total_dt_ds': np.inf, 'layer_contributions': {}}
        
    # Calculate radial step sizes
    dr = np.diff(R)
    
    # Evaluate the base integrand: T(r) * rho(r) * r^2
    integrand = T[:-1] * Rho[:-1] * (R[:-1] ** 2)
    
    # Find all unique compositional steps in the staircase
    unique_z = np.unique(Z)
    
    layer_contributions = {}
    total_dt_ds = 0.0
    
    # Integrate layer-by-layer
    for z_val in unique_z:
        mask = np.isclose(Z[:-1], z_val, atol=1e-4)
        
        layer_integral = np.sum(integrand[mask] * dr[mask])
        
        # --- FIX: Added the minus sign required by the energy balance equation! ---
        layer_dt_ds = - (layer_integral / denominator)
        
        layer_contributions[z_val] = layer_dt_ds
        total_dt_ds += layer_dt_ds
        
    return {
        'total_dt_ds': total_dt_ds,
        'layer_contributions': layer_contributions
    }