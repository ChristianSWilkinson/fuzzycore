# utils.py
import numpy as np

def generate_gaussian_z_profile(n_layers=25, sigma=0.15, z_base=0.02, z_core=0.98):
    """
    Generates a Z mass fraction profile following a Gaussian curve.
    
    Parameters:
        n_layers (int): Number of discrete steps.
        sigma (float): 'Width' of the transition (0.1 = sharp, 0.3 = diffuse).
        z_base (float): The baseline Z fraction at the surface/atmosphere.
        z_core (float): The maximum Z fraction at the core-envelope interface.
    """
    # x goes from 0 (surface) to 1 (core interface)
    x = np.linspace(0, 1, n_layers)
    
    # Gaussian centered at 1.0 (Core boundary)
    # raw_z goes from nearly 0 at x=0 to 1.0 at x=1
    raw_z = np.exp(-((x - 1.0)**2) / (2 * sigma**2))
    
    # Normalize raw_z to be exactly [0, 1] before scaling
    raw_z = (raw_z - raw_z.min()) / (raw_z.max() - raw_z.min())
    
    # Scale and shift to [z_base, z_core]
    # This ensures the atmosphere starts at z_base
    z_profile = z_base + (z_core - z_base) * raw_z
    
    return np.clip(z_profile, 0.0, 0.99)

def evaluate_heavy_element_mass(results, z_top_atmosphere):
    """
    Calculates total mass of heavy elements above the atmospheric Z level.
    
    Parameters:
        results (dict): The dictionary returned by integrate_planet.
        z_top_atmosphere (float): The Z fraction at the surface (e.g., params['z_profile'][0]).
        
    Returns:
        float: Total mass of heavy elements in kg.
    """
    R = results['R']
    M = results['M']
    Rho = results['Rho']
    Z = results['Z']
    
    # 1. Start with the mass of the pure rock core (where Z is nominally 1)
    # R_int corresponds to the last step of the core loop
    m_z_total = results.get('M_core_actual', 0.0)
    if m_z_total == 0.0 and len(Z) > 0:
        # Fallback: if M_core wasn't saved, use mass at the interface
        idx_int = np.where(Z < 1.0)[0]
        if len(idx_int) > 0:
            m_z_total = M[idx_int[0]]
        else:
            m_z_total = M[-1] # Pure rock planet case

    # 2. Integrate the dilute region in the envelope
    # We only count Z values that exceed the atmospheric background
    dr = np.diff(R)
    
    # Calculate mass of each shell: dM = 4 * pi * r^2 * rho * dr
    # We use index [:-1] to match the length of np.diff(R)
    shell_masses = 4 * np.pi * R[:-1]**2 * Rho[:-1] * dr
    
    # Calculate the 'excess' heavy element fraction in each shell
    # Z_eff = (Z_local - Z_atm)
    z_envelope = Z[:-1]
    z_excess = np.maximum(0, z_envelope - z_top_atmosphere)
    
    # Sum the excess mass in the envelope
    m_z_envelope = np.sum(shell_masses * z_excess)
    
    m_z_total += m_z_envelope
    
    return m_z_total

# utils.py

class DummyLock:
    """A no-op lock for single-threaded runs in notebooks."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass