"""
Utility Functions for Planetary Interior Modeling.

This module provides helper functions and classes for generating compositional 
gradients, integrating exact mass profiles, calculating thermal properties, 
and handling execution locks during parallelized parameter sweeps.
"""

import numpy as np

from . import constants as c


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
    Calculate the inverse cooling rate (dt/dS) for a non-adiabatic staircase.

    This expands the thermal integral by grouping the interior into discrete 
    compositional (Z) layers. It returns both the global homologous cooling 
    rate and a breakdown of the thermal inertia contributed by each layer.
    Uses exact mass differentials to avoid numerical integration artifacts.

    Args:
        results (dict): The converged planetary structure dictionary. Must 
            contain 'R', 'T', 'M', and 'Z'.
        t_int (float): The internal effective temperature driving the cooling 
            luminosity (in Kelvin).

    Returns:
        dict: A dictionary containing:
            - 'total_dt_ds' (float): The global inverse cooling rate.
            - 'layer_contributions' (dict): A mapping of Z-fraction to its 
              specific contribution to dt/dS.
    """
    radius_array = results['R']
    temp_array = results['T']
    mass_array = results['M']
    z_array = results['Z']

    radius_planet = radius_array[-1]

    # Calculate the Stefan-Boltzmann denominator (Luminosity / 4*pi)
    denominator = c.SIGMA_SB * (radius_planet ** 2) * (t_int ** 4)

    if denominator <= 0:
        return {'total_dt_ds': np.inf, 'layer_contributions': {}}

    # Exact mass of each discrete spherical shell
    dm = np.diff(mass_array)

    # Approximate average temperature of the shell
    t_shell = (temp_array[:-1] + temp_array[1:]) / 2.0

    # Evaluate the integrand: T(r) * (dm / 4*pi)
    integrand = t_shell * (dm / (4 * np.pi))

    # Classify shells by their outer edge Z
    z_shell = z_array[1:]
    unique_z = np.unique(z_shell)

    layer_contributions = {}
    total_dt_ds = 0.0

    # Integrate layer-by-layer
    for z_val in unique_z:
        mask = np.isclose(z_shell, z_val, atol=1e-4)

        layer_integral = np.sum(integrand[mask])

        # Apply the minus sign required by the energy balance equation
        layer_dt_ds = -(layer_integral / denominator)

        layer_contributions[z_val] = layer_dt_ds
        total_dt_ds += layer_dt_ds

    return {
        'total_dt_ds': total_dt_ds,
        'layer_contributions': layer_contributions
    }