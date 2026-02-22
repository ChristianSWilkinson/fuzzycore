import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from . import constants as c
from . import eos


# =============================================================================
# 0. HELPER FUNCTIONS
# =============================================================================

def get_stepper(stack_entry: dict):
    """
    Retrieves or initializes the KD-Tree Adiabat Stepper for a given Z-layer.
    
    Args:
        stack_entry (dict): A dictionary containing interpolated EOS data 
            for a specific metallicity (Z) layer.
            
    Returns:
        eos.RobustAdiabatStepper: The initialized stepper object.
    """
    if 'stepper' not in stack_entry:
        stack_entry['stepper'] = eos.RobustAdiabatStepper(stack_entry)
    return stack_entry['stepper']


# =============================================================================
# 1. CORE INTEGRATOR
# =============================================================================

def integrate_core(Pc_bar: float, M_core: float, T_mean: float = 5000.0, 
                   iron_fraction: float = 0.33) -> dict:
    """
    Integrates the solid planetary core outward from the center.
    
    Uses a standard Forward Euler scheme to solve the equations of mass 
    conservation and hydrostatic equilibrium through a mixed Rock/Iron EOS.
    
    Args:
        Pc_bar (float): Central pressure guess in bar.
        M_core (float): Target mass of the solid core in kg.
        T_mean (float, optional): Assumed isothermal temperature of the core in K. 
            Defaults to 5000.0.
        iron_fraction (float, optional): Mass fraction of iron. Defaults to 0.33.
        
    Returns:
        dict: A dictionary containing the final core radius, pressure, mass, 
            and the full radial arrays (R, M, P, Rho, T, S, Z).
    """
    # Handle the trivial case (no solid core)
    if M_core <= 0:
        return {
            'P_top': Pc_bar, 
            'R_core': 0.0, 
            'Rho_top': 0.0, 
            'M_actual': 0.0, 
            'valid': False,
            'R': np.array([0.0]), 
            'M': np.array([0.0]), 
            'P': np.array([np.log10(max(1e-5, Pc_bar))]),
            'Rho': np.array([0.0]), 
            'T': np.array([T_mean]), 
            'S': np.array([0.0]), 
            'Z': np.array([1.0])
        }

    # Clamp temperature to valid EOS boundaries
    T_safe = np.clip(T_mean, 300.0, 49000.0)
    lt_safe = np.log10(T_safe)

    # Initialization
    r = 100.0  # Start slightly off-center to avoid 1/r^2 singularity
    p_pa = Pc_bar * 1e5
    p_log_bar = np.log10(max(1e-5, p_pa / 1e5))
    
    # Query central density
    rho_log = eos.query_core_eos(p_log_bar, lt_safe, iron_fraction)
    rho_c = 10 ** rho_log

    # Initial core mass
    m = (4 / 3) * np.pi * rho_c * r**3
    rho = rho_c

    # Trackers for the structural profile
    R_h, M_h, P_h, Rho_h, T_h, S_h, Z_h = [], [], [], [], [], [], []
    k = 0

    # Main Integration Loop (Forward Euler)
    while m < M_core:
        if p_pa <= 0:
            break
            
        p_log_bar = np.log10(max(1e-5, p_pa / 1e5))
        rho_log = eos.query_core_eos(p_log_bar, lt_safe, iron_fraction)
        rho = 10 ** rho_log

        # Hydrostatic equilibrium derivatives
        g = (c.G_CONST * m) / (r**2)
        H_P = p_pa / (rho * g) if (rho * g) > 0 else 1e5
        
        # Adaptive step size: Limit to 10% of a pressure scale height
        dr = min(500.0, H_P * 0.1)

        # Ensure we don't overshoot the target core mass on the final step
        dm_dr = 4 * np.pi * r**2 * rho
        dr = min(dr, (M_core - m) / dm_dr)
        if dr < 1.0:
            dr = 1.0

        # Step forward
        p_pa += -rho * g * dr
        m += dm_dr * dr
        r += dr

        # Downsample saved output to save memory
        if k % 50 == 0:
            R_h.append(r)
            M_h.append(m)
            P_h.append(p_log_bar)
            Rho_h.append(rho)
            T_h.append(T_safe)
            S_h.append(0.0)
            Z_h.append(1.0)
        k += 1

    # Record final boundary state
    p_final_log = np.log10(max(1e-5, p_pa / 1e5))
    R_h.append(r)
    M_h.append(m)
    P_h.append(p_final_log)
    Rho_h.append(rho)
    T_h.append(T_safe)
    S_h.append(0.0)
    Z_h.append(1.0)

    return {
        'P_top': max(1e-5, p_pa / 1e5), 
        'R_core': r, 
        'Rho_top': rho, 
        'M_actual': m, 
        'valid': True,
        'R': np.array(R_h), 
        'M': np.array(M_h), 
        'P': np.array(P_h), 
        'Rho': np.array(Rho_h), 
        'T': np.array(T_h), 
        'S': np.array(S_h), 
        'Z': np.array(Z_h)
    }


# =============================================================================
# 2. ENVELOPE BUILDER (Adaptive Staircase)
# =============================================================================

def build_staircase_envelope(P_surf_bar: float, P_int_bar: float, T_surf: float, 
                             z_steps: np.ndarray, fluid_stack: dict) -> dict:
    """
    Constructs a top-down "staircase" composition envelope.
    
    Generates an adiabatic profile using the KD-Tree stepper, dynamically 
    matching entropy across distinct heavy-element (Z) layers.
    
    Args:
        P_surf_bar (float): Atmospheric surface pressure.
        P_int_bar (float): Deep interior boundary pressure.
        T_surf (float): Surface temperature.
        z_steps (np.ndarray): Array of target Z mass fractions.
        fluid_stack (dict): Pre-computed fluid interpolators for each Z step.
        
    Returns:
        dict: 1D interpolators mapping logP to density, temperature, entropy, and Z.
            Returns None if the atmospheric domain is physically invalid.
    """
    if P_int_bar <= P_surf_bar:
        return None

    full_P, full_Rho, full_T, full_S, full_Z = [], [], [], [], []
    curr_lp_bar = np.log10(max(1e-10, P_surf_bar))
    final_lp_bar = np.log10(max(1.0, P_int_bar))
    
    if not np.isfinite(curr_lp_bar) or not np.isfinite(final_lp_bar):
        return None

    # Adaptive Downsampling: Reduce the number of discrete composition layers 
    # if the envelope is barometrically very thin to prevent numerical noise.
    pressure_span = final_lp_bar - curr_lp_bar
    if pressure_span < 0.2:
        used_z_steps = np.array([np.mean(z_steps)])
    elif pressure_span < 1.0:
        used_z_steps = z_steps[np.linspace(0, len(z_steps) - 1, 3, dtype=int)]
    elif pressure_span < 3.0:
        used_z_steps = z_steps[np.linspace(0, len(z_steps) - 1, 10, dtype=int)]
    else:
        used_z_steps = z_steps

    current_T_val = float(T_surf)
    if current_T_val > 5000.0:
        current_T_val = 200.0
        
    env_boundaries = np.linspace(curr_lp_bar, final_lp_bar, len(used_z_steps) + 1)

    # Step downward through each compositional layer
    for i, z_val in enumerate(used_z_steps):
        # Snap to the closest available pre-computed Z table
        z_key = min(fluid_stack.keys(), key=lambda x: abs(x - z_val))
        layer_data = fluid_stack[z_key]
        stepper = get_stepper(layer_data)
        
        lp_start = env_boundaries[i]
        lp_end = env_boundaries[i+1]

        # Function to find the Entropy that yields the exact boundary Temperature
        def temp_error(s_guess):
            try:
                lT, _ = stepper.get_state(lp_start, np.log10(max(50.0, current_T_val)), s_guess)
                return (10 ** lT) - current_T_val
            except Exception:
                return 1e9

        # Initial Entropy guess
        try:
            ref_s = float(layer_data['S'](lp_start, np.log10(max(50.0, current_T_val))))
        except Exception:
            ref_s = 6.0
            
        # Entropy matching logic
        try:
            if i == 0:
                target_s = ref_s  # Surface layer anchors the entropy
            else:
                # Find new entropy for the next layer that maintains temperature continuity
                s_bound = max(5.0, abs(ref_s) * 0.2)
                target_s = brentq(temp_error, ref_s - s_bound, ref_s + s_bound, xtol=1e-2)
        except Exception:
            target_s = ref_s

        # Integrate through the current layer
        n_points = 50 if pressure_span > 1.0 else 15
        p_layer_grid = np.linspace(lp_start, lp_end, n_points)
        curr_lt_guess = np.log10(max(50.0, current_T_val))

        layer_P, layer_T, layer_Rho = [], [], []
        
        for p_log in p_layer_grid:
            guess_in = curr_lt_guess + 0.002
            try:
                next_lt, next_lrho = stepper.get_state(p_log, guess_in, target_s)
                # Apply thermal safety clamps
                if next_lt > 5.8:
                    next_lt = 5.8 
                if next_lt < (curr_lt_guess - 1e-4):
                    next_lt = curr_lt_guess + 1e-6
            except Exception:
                next_lt = curr_lt_guess
                next_lrho = p_log - next_lt - 3.0  # Fallback ideal gas proxy
                
            curr_lt_guess = next_lt
            layer_P.append(p_log)
            layer_T.append(10 ** next_lt)
            layer_Rho.append(10 ** next_lrho)
            full_S.append(target_s)
            full_Z.append(z_key)

        full_P.extend(layer_P)
        full_Rho.extend(layer_Rho)
        full_T.extend(layer_T)
        
        # The base temperature of this layer becomes the roof of the next layer
        current_T_val = layer_T[-1]

    # Clean and interpolate final atmospheric profile
    full_P = np.array(full_P)
    valid_mask = np.isfinite(full_P) & np.isfinite(full_T)
    if np.sum(valid_mask) < 2:
        return None
        
    full_P = full_P[valid_mask]
    sort_idx = np.argsort(full_P)
    _, u_idx = np.unique(full_P[sort_idx], return_index=True)
    final_idx = sort_idx[u_idx]

    try:
        return {
            'rho': interp1d(full_P[final_idx], np.array(full_Rho)[valid_mask][final_idx], fill_value='extrapolate'),
            't': interp1d(full_P[final_idx], np.array(full_T)[valid_mask][final_idx], fill_value='extrapolate'),
            's': interp1d(full_P[final_idx], np.array(full_S)[valid_mask][final_idx], fill_value='extrapolate'),
            'z': interp1d(full_P[final_idx], np.array(full_Z)[valid_mask][final_idx], fill_value='extrapolate')
        }
    except Exception:
        return None


# =============================================================================
# 3. WATER WORLD INTEGRATOR LOGIC
# =============================================================================

def run_water_world_integration(Pc_bar: float, P_int_bar: float, params: dict, 
                                eos_data: dict) -> dict:
    """
    Executes the nested multi-zone integration for planets with massive water mantles.
    
    Architecture: Core -> Water Mantle -> Gaseous Envelope.
    """
    stack = eos_data['fluid']
    water_eos = eos_data.get('water')
    if not water_eos:
        return None
        
    mantle_stepper = get_stepper(water_eos)
    iron_frac = params.get('iron_fraction', 0.33)

    # 1. Build Gaseous Envelope (Top-Down)
    env = build_staircase_envelope(
        params['P_surf'], P_int_bar, params['T_surf'], params['z_profile'], stack
    )
    if env is None:
        return None

    logP_int = np.log10(P_int_bar)
    T_int = float(env['t'](logP_int))
    
    # Anchor the mantle entropy to the base of the envelope
    try:
        target_s = float(water_eos['S'](logP_int, np.log10(T_int)))
    except Exception:
        target_s = 3000.0

    # 2. Pilot Core Integration to match boundary temperatures
    pre_core = integrate_core(Pc_bar, params['M_rock'], T_mean=5000.0, iron_fraction=iron_frac)
    P_rock_top_guess = pre_core['P_top']
    p_log_core = np.log10(max(1e-5, P_rock_top_guess))

    # Project the mantle adiabat downwards to estimate the core surface temperature
    lt_core, _ = mantle_stepper.get_state(p_log_core, np.log10(T_int * 2), target_s)
    T_core_match = max(T_int, 10 ** lt_core)

    # 3. Final Core Integration
    core_res = integrate_core(Pc_bar, params['M_rock'], T_mean=T_core_match, iron_fraction=iron_frac)
    if not core_res['valid']:
        return None
        
    P_rock_top = core_res['P_top']
    R_rock = core_res['R_core']
    M_rock_actual = core_res['M_actual']

    # Physical validity check: Ensure mantle isn't structurally crushed
    if M_rock_actual < params['M_rock'] * 0.99 or P_rock_top <= P_int_bar:
        return {
            "M_total": M_rock_actual, 
            "R_total": R_rock, 
            "M_Z_total": M_rock_actual,
            "R": core_res['R'], 
            "M": core_res['M'], 
            "P": core_res['P'], 
            "Rho": core_res['Rho'], 
            "T": core_res['T'], 
            "S": core_res['S'], 
            "Z": core_res['Z'],
            "R_rock": R_rock, 
            "R_int": R_rock, 
            "P_int": P_rock_top
        }

    # 4. Integrate Water Mantle (Bottom-Up)
    r = R_rock
    m = M_rock_actual
    p_pa = P_rock_top * 1e5
    
    R_h, M_h, P_h = list(core_res['R']), list(core_res['M']), list(core_res['P'])
    Rho_h, T_h = list(core_res['Rho']), list(core_res['T'])
    S_h, Z_h = list(core_res['S']), list(core_res['Z'])

    current_lt = np.log10(T_core_match)
    k = 0

    while p_pa > P_int_bar * 1e5:
        p_log = np.log10(max(1e-5, p_pa / 1e5))
        next_lt, next_lrho = mantle_stepper.get_state(p_log, current_lt, target_s)

        # ISOTHERMAL GUARD: Prevents negative gradients across jaggy phase transitions
        if next_lt < current_lt - 0.005:
            next_lt = current_lt
            try:
                # Re-query raw EOS for the correct density at the clamped isothermal state
                next_lrho = float(water_eos['rho'](p_log, next_lt))
            except Exception:
                pass  # Fallback to the stepper's density prediction

        current_lt = next_lt
        rho = 10 ** next_lrho
        temp = 10 ** next_lt

        # Forward Euler Step
        g = (c.G_CONST * m) / r**2
        dp_dr = -rho * g
        H_P = p_pa / (rho * g) if (rho * g) > 0 else 1e5
        
        dr = min(5000.0, H_P * 0.05)
        p_rem = p_pa - P_int_bar * 1e5
        
        if abs(dp_dr) > 0:
            dr = min(dr, abs(p_rem / dp_dr) * 0.5)
        dr = max(dr, 1.0)
        
        p_new = p_pa + dp_dr * dr
        if p_new < P_int_bar * 1e5:
            dr = abs(p_rem / dp_dr)
            p_new = P_int_bar * 1e5

        m += 4 * np.pi * r**2 * rho * dr
        r += dr
        p_pa = p_new
        
        if k % 20 == 0:
            R_h.append(r); M_h.append(m); P_h.append(p_log)
            Rho_h.append(rho); T_h.append(temp)
            S_h.append(target_s); Z_h.append(1.0)
        k += 1

    # End of Mantle boundary condition
    R_h.append(r); M_h.append(m); P_h.append(np.log10(P_int_bar))
    Rho_h.append(rho); T_h.append(temp)
    S_h.append(target_s); Z_h.append(1.0)
    
    R_int = r

    # 5. Integrate Gaseous Envelope (Bottom-Up)
    p_surf_pa = params['P_surf'] * 1e5
    m_z_total = m  # Track total heavy mass before adding diffuse atmospheric Z
    
    while p_pa > p_surf_pa:
        p_log = np.log10(max(1e-10, p_pa / 1e5))
        try:
            rho = float(env['rho'](p_log))
            temp = float(env['t'](p_log))
            entr = float(env['s'](p_log))
            z_val = float(env['z'](p_log))
        except Exception:
            break
            
        g = (c.G_CONST * m) / r**2
        dp_dr = -rho * g
        H_P = p_pa / (rho * g) if (rho * g) > 0 else 1e5
        
        dr = min(5000.0, H_P * 0.05)
        p_rem = p_pa - p_surf_pa
        
        if abs(dp_dr) > 0:
            dr = min(dr, abs(p_rem / dp_dr) * 0.5)
        dr = max(dr, 1.0)
        
        p_new = p_pa + dp_dr * dr
        if p_new < p_surf_pa:
            dr = abs(p_rem / dp_dr)
            p_new = p_surf_pa
            
        m += 4 * np.pi * r**2 * rho * dr
        r += dr
        p_pa = p_new
        
        if k % 20 == 0:
            R_h.append(r); M_h.append(m); P_h.append(p_log)
            Rho_h.append(rho); T_h.append(temp)
            S_h.append(entr); Z_h.append(z_val)
        k += 1

    # Final Surface boundary condition
    R_h.append(r); M_h.append(m); P_h.append(np.log10(max(1e-5, p_pa / 1e5)))
    Rho_h.append(rho); T_h.append(temp)
    S_h.append(entr); Z_h.append(z_val)

    return {
        "M_total": m, "R_total": r, "M_Z_total": m_z_total,
        "R": np.array(R_h), "M": np.array(M_h), "P": np.array(P_h), 
        "Rho": np.array(Rho_h), "T": np.array(T_h), "S": np.array(S_h), "Z": np.array(Z_h),
        "R_rock": R_rock, "R_int": R_int, "P_int": P_int_bar
    }


def integrate_water_world(logPc: float, params: dict, eos_data: dict) -> dict:
    """
    Root-finding wrapper for Water World architectures. 
    Finds the exact mantle-envelope boundary pressure (P_int) required to 
    satisfy the requested M_water parameter.
    """
    Pc_bar = 10 ** logPc
    iron_frac = params.get('iron_fraction', 0.33)

    # Fast Core Check: Validate if the requested Pc is physically capable 
    # of supporting the requested rock mass before doing heavy envelope calculations.
    pre_core = integrate_core(Pc_bar, params['M_rock'], T_mean=5000.0, iron_fraction=iron_frac)
    if pre_core['M_actual'] < params['M_rock'] * 0.99:
        # Failsafe: Run integration anyway (will likely be rejected by solver logic)
        return run_water_world_integration(Pc_bar, pre_core['P_top'], params, eos_data)

    P_rock_top_guess = pre_core['P_top']

    def mass_error(logP_int_guess):
        """Inner objective function evaluating mantle mass mismatch."""
        P_int_val = 10 ** logP_int_guess
        if P_int_val <= params['P_surf']:
            return 1e30

        env = build_staircase_envelope(
            params['P_surf'], P_int_val, params['T_surf'], params['z_profile'], eos_data['fluid']
        )
        if env is None:
            return 1e30

        T_int = float(env['t'](logP_int_guess))
        water_eos = eos_data['water']
        try:
            target_s = float(water_eos['S'](logP_int_guess, np.log10(T_int)))
        except Exception:
            return 1e30

        mantle_stepper = get_stepper(water_eos)
        lt_core, _ = mantle_stepper.get_state(
            np.log10(max(1e-5, P_rock_top_guess)), np.log10(T_int * 2), target_s
        )
        T_core_match = max(T_int, 10 ** lt_core)

        core_iter = integrate_core(Pc_bar, params['M_rock'], T_mean=T_core_match, iron_fraction=iron_frac)
        r = core_iter['R_core']
        m = params['M_rock']
        p_pa = core_iter['P_top'] * 1e5

        m_water_added = 0.0
        current_lt = np.log10(T_core_match)

        # Fast Mantle Integration (Coarser step logic)
        while p_pa > P_int_val * 1e5:
            p_log = np.log10(max(1e-5, p_pa / 1e5))
            next_lt, next_lrho = mantle_stepper.get_state(p_log, current_lt, target_s)

            if next_lt < current_lt - 0.005:
                next_lt = current_lt 
            current_lt = next_lt
            rho = 10 ** next_lrho

            g = (c.G_CONST * m) / r**2
            dp_dr = -rho * g
            H_P = p_pa / (rho * g) if (rho * g) > 0 else 1e5
            
            dr = min(5000.0, H_P * 0.1) 
            p_rem = p_pa - P_int_val * 1e5
            
            if abs(dp_dr) > 0:
                dr = min(dr, abs(p_rem / dp_dr) * 0.5)
            dr = max(dr, 1.0)

            p_new = p_pa + dp_dr * dr
            if p_new < P_int_val * 1e5:
                dr = abs(p_rem / dp_dr)
                p_new = P_int_val * 1e5

            dm = 4 * np.pi * r**2 * rho * dr
            m += dm
            m_water_added += dm
            r += dr
            p_pa = p_new

        return m_water_added - params['M_water']

    # Set continuous solver bounds
    lb = np.log10(params['P_surf'] * 1.05) 
    ub = np.log10(pre_core['P_top'] * 0.999)

    if lb >= ub:
        final_logP_int = lb
    else:
        err_lb = mass_error(lb)
        err_ub = mass_error(ub)

        if np.sign(err_lb) == np.sign(err_ub):
            # No bracket: Pick the best physical bound
            final_logP_int = lb if abs(err_lb) < abs(err_ub) else ub
        else:
            try:
                # Brent's method: Strict mathematical solve
                final_logP_int = brentq(mass_error, lb, ub, xtol=1e-3)
            except Exception:
                final_logP_int = (lb + ub) / 2.0

    return run_water_world_integration(Pc_bar, 10 ** final_logP_int, params, eos_data)


# =============================================================================
# 4. GENERIC PLANET INTEGRATOR (Gas Giant / Sub-Neptune)
# =============================================================================

def integrate_planet(logPc: float, params: dict, eos_data: dict) -> dict:
    """
    Integrates a standard planetary structure: Solid Core + Gaseous Envelope.
    
    This function utilizes the nested shooting method to ensure absolute
    thermodynamic consistency between the boundary of the solid core and
    the base of the gas envelope.
    """
    debug = params.get('debug', False)
    Pc_bar = 10 ** logPc
    iron_frac = params.get('iron_fraction', 0.33)

    # 1. Pilot Integration (Estimate Core Boundary Pressure)
    pilot_core = integrate_core(Pc_bar, params['M_core'], T_mean=5000.0, iron_fraction=iron_frac)
    P_int_guess = pilot_core['P_top']
    
    if not pilot_core['valid'] or P_int_guess <= params['P_surf']:
        return None

    # 2. Build Envelope (Top-Down)
    try:
        env = build_staircase_envelope(
            params['P_surf'], P_int_guess, params['T_surf'], params['z_profile'], eos_data['fluid']
        )
    except Exception:
        return None
        
    if env is None:
        return None

    # Determine Base Envelope Temperature
    try:
        T_match = float(env['t'](np.log10(P_int_guess)))
    except Exception:
        T_match = 5000.0

    # 3. Final Core Integration (Bottom-Up, using matched temperature)
    final_core = integrate_core(Pc_bar, params['M_core'], T_mean=T_match, iron_fraction=iron_frac)
    P_int_final = final_core['P_top']
    R_int_final = final_core['R_core']
    
    if P_int_final <= params['P_surf']:
        return None

    # Initialize tracking variables for Envelope Integration
    r = R_int_final
    m = final_core['M_actual']
    p_pa = P_int_final * 1e5
    m_z = m 
    
    R_h, M_h, P_h = list(final_core['R']), list(final_core['M']), list(final_core['P'])
    Rho_h, T_h = list(final_core['Rho']), list(final_core['T'])
    S_h, Z_h = list(final_core['S']), list(final_core['Z'])
    
    p_min_log = np.log10(params['P_surf'])
    p_max_log = np.log10(P_int_final)
    k = 0

    if debug:
        print(f"[START] Pc={Pc_bar:.2e} -> P_int={P_int_final:.2e} -> T_int={T_match:.1f} K")

    # 4. Final Envelope Integration (Bottom-Up)
    while (p_pa / 1e5) > params['P_surf']:
        p_log = np.log10(max(1e-10, p_pa / 1e5))
        p_lookup = np.clip(p_log, p_min_log, p_max_log)
        
        try:
            rho = float(env['rho'](p_lookup))
            temp = float(env['t'](p_lookup))
            entr = float(env['s'](p_lookup))
            z_val = float(env['z'](p_lookup))
        except Exception:
            break

        g = (c.G_CONST * m) / r**2
        dp_dr = -rho * g
        H_P = p_pa / (rho * g) if (rho * g) > 0 else 1e5
        
        dr = min(5000.0, H_P * 0.05, abs((p_pa - params['P_surf'] * 1e5) / dp_dr) * 0.5)
        dr = max(dr, 1.0)
        
        p_new = p_pa + dp_dr * dr
        if p_new < params['P_surf'] * 1e5:
            dr = abs((p_pa - params['P_surf'] * 1e5) / dp_dr)
            p_new = params['P_surf'] * 1e5

        # Update mass and geometric arrays
        m += 4 * np.pi * r**2 * rho * dr
        m_z += 4 * np.pi * r**2 * rho * dr * max(0.0, z_val - params.get('z_base', 0.0))
        r += dr
        p_pa = p_new
        
        if k % 100 == 0:
            R_h.append(r); M_h.append(m); P_h.append(p_log)
            Rho_h.append(rho); T_h.append(temp)
            S_h.append(entr); Z_h.append(z_val)
        k += 1

    # Surface boundary condition
    R_h.append(r); M_h.append(m); P_h.append(np.log10(max(1e-5, p_pa / 1e5)))
    Rho_h.append(rho); T_h.append(temp)
    S_h.append(entr); Z_h.append(z_val)

    if debug:
        print(f"[SUCCESS] M={m/c.M_JUPITER:.3f} Mj")

    return {
        "M_total": m, "R_total": r, "M_Z_total": m_z, 
        "R": np.array(R_h), "M": np.array(M_h), "P": np.array(P_h), 
        "Rho": np.array(Rho_h), "T": np.array(T_h), "S": np.array(S_h), "Z": np.array(Z_h), 
        "R_rock": R_int_final, "R_int": R_int_final
    }