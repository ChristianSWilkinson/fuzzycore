import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from . import constants as c
from . import eos
from . import utils  

# =============================================================================
# 0. HELPER FUNCTIONS
# =============================================================================

def get_stepper(stack_entry: dict):
    if 'stepper' not in stack_entry:
        stack_entry['stepper'] = eos.RobustAdiabatStepper(stack_entry)
    return stack_entry['stepper']

# =============================================================================
# 1. CORE INTEGRATOR
# =============================================================================

def integrate_core(Pc_bar: float, M_core: float, T_center: float = 5000.0, 
                   iron_fraction: float = 0.33, env_rho_base: float = 0.0,
                   nabla_ad: float = 0.0) -> dict:
    
    if M_core <= 0:
        return {
            'P_top': Pc_bar, 'R_core': 0.0, 'Rho_top': 0.0, 'M_actual': 0.0, 'valid': False,
            'R': np.array([0.0]), 'M': np.array([0.0]), 'P': np.array([np.log10(max(1e-5, Pc_bar))]),
            'Rho': np.array([0.0]), 'T': np.array([T_center]), 'S': np.array([0.0]), 'Z': np.array([1.0])
        }

    r = 100.0  
    p_pa = Pc_bar * 1e5
    p_log_bar = np.log10(max(1e-5, p_pa / 1e5))
    
    lt_safe_c = np.log10(np.clip(T_center, 300.0, 49000.0))
    rho_log = eos.query_core_eos(p_log_bar, lt_safe_c, iron_fraction)
    rho_c = 10 ** rho_log

    m = (4 / 3) * np.pi * rho_c * r**3
    rho = rho_c

    R_h, M_h, P_h, Rho_h, T_h, S_h, Z_h = [], [], [], [], [], [], []
    k = 0

    while m < M_core:
        if p_pa <= 0:
            break
            
        p_log_bar = np.log10(max(1e-5, p_pa / 1e5))
        T_current = T_center * ((p_pa / 1e5) / Pc_bar) ** nabla_ad
        lt_safe = np.log10(np.clip(T_current, 300.0, 49000.0))
        
        rho_log = eos.query_core_eos(p_log_bar, lt_safe, iron_fraction)
        raw_rho = 10 ** rho_log
        
        if raw_rho < env_rho_base:
            rho = env_rho_base * (Pc_bar / max(1.0, p_pa / 1e5)) ** 0.05
        else:
            rho = raw_rho

        g = (c.G_CONST * m) / (r**2)
        H_P = p_pa / (rho * g) if (rho * g) > 0 else 1e5
        dr = min(500.0, H_P * 0.1)

        dm_dr = 4 * np.pi * r**2 * rho
        dr = min(dr, (M_core - m) / dm_dr)
        if dr < 1.0: dr = 1.0

        p_pa += -rho * g * dr
        m += dm_dr * dr
        r += dr

        if k % 50 == 0:
            R_h.append(r); M_h.append(m); P_h.append(p_log_bar)
            Rho_h.append(rho); T_h.append(T_current) 
            S_h.append(0.0); Z_h.append(1.0)
        k += 1

    p_final_log = np.log10(max(1e-5, p_pa / 1e5))
    R_h.append(r); M_h.append(m); P_h.append(p_final_log)
    Rho_h.append(rho); T_h.append(T_current)
    S_h.append(0.0); Z_h.append(1.0)

    # 🛑 THE FIX: Strictly enforce that the core actually reached the target mass!
    # If the central pressure is too weak, p_pa hits 0 early, and m < M_core.
    is_valid = (m >= M_core * 0.999) and (p_pa > 0)

    return {
        'P_top': max(1e-5, p_pa / 1e5), 'R_core': r, 'Rho_top': rho, 'M_actual': m, 'valid': is_valid,
        'R': np.array(R_h), 'M': np.array(M_h), 'P': np.array(P_h), 
        'Rho': np.array(Rho_h), 'T': np.array(T_h), 'S': np.array(S_h), 'Z': np.array(Z_h)
    }

# =============================================================================
# 2. ENVELOPE BUILDER (Adaptive Staircase)
# =============================================================================

def build_staircase_envelope(p_surf_bar: float, p_bottom_bar: float, 
                             T_surf: float, z_profile: np.ndarray, 
                             stack: dict, debug: bool = False) -> dict:
    
    if debug:
        print(f"  [Env Builder] Integrating from {p_surf_bar:.2f} to {p_bottom_bar:.2f} bar.")

    final_lp_bar = np.log10(max(1e-5, p_bottom_bar))
    curr_lp_bar = np.log10(max(1e-5, p_surf_bar))

    full_P, full_T, full_Rho, full_S, full_Z = [], [], [], [], []

    used_z_steps = np.unique(np.round(z_profile, 4))
    if used_z_steps[0] > z_profile[0]: 
        used_z_steps = np.insert(used_z_steps, 0, z_profile[0])
        
    current_T_val = float(T_surf)
    if current_T_val > 5000.0:
        current_T_val = 200.0
        
    env_boundaries = np.linspace(curr_lp_bar, final_lp_bar, len(used_z_steps) + 1)

    prev_z_key = None     
    prev_target_s = None  

    for i, z_val in enumerate(used_z_steps):
        z_key = min(stack.keys(), key=lambda x: abs(x - z_val))
        layer_data = stack[z_key]
        stepper = get_stepper(layer_data)
        
        lp_start = env_boundaries[i]
        lp_end = env_boundaries[i+1]

        def temp_error(s_guess):
            try:
                lT, _ = stepper.get_state(lp_start, np.log10(max(50.0, current_T_val)), s_guess)
                return (10 ** lT) - current_T_val
            except Exception:
                return 1e9

        try:
            ref_s = float(layer_data['S'](lp_start, np.log10(max(50.0, current_T_val))))
        except Exception:
            ref_s = 6.0
            
        try:
            if i == 0:
                target_s = ref_s  
            elif z_key == prev_z_key:
                target_s = prev_target_s
            else:
                s_bound = max(5.0, abs(ref_s) * 0.2)
                target_s = brentq(temp_error, ref_s - s_bound, ref_s + s_bound, xtol=1e-5)
                if debug:
                    print(f"  [Env Builder] Z-jump {prev_z_key:.3f}->{z_key:.3f}: Entropy jumped from {prev_target_s:.2f} to {target_s:.2f}")
        except Exception as e:
            if debug: print(f"  [Env Builder] Entropy root-find failed ({e}). Falling back to reference S.")
            target_s = ref_s

        prev_z_key = z_key
        prev_target_s = target_s

        n_points = 500 
        p_layer_grid = np.linspace(lp_start, lp_end, n_points)
        curr_lt_guess = np.log10(max(50.0, current_T_val))

        layer_P, layer_T, layer_Rho = [], [], []
        
        for p_log in p_layer_grid:
            guess_in = curr_lt_guess + 0.002
            try:
                next_lt, next_lrho = stepper.get_state(p_log, guess_in, target_s)
                if next_lt > 5.8: next_lt = 5.8 
                if next_lt < (curr_lt_guess - 1e-4): next_lt = curr_lt_guess + 1e-6
            except Exception:
                next_lt = curr_lt_guess
                next_lrho = p_log - next_lt - 3.0  
                
            curr_lt_guess = next_lt
            layer_P.append(p_log)
            layer_T.append(10 ** next_lt)
            layer_Rho.append(10 ** next_lrho)
            full_S.append(target_s)
            full_Z.append(z_key)

        full_P.extend(layer_P)
        full_Rho.extend(layer_Rho)
        full_T.extend(layer_T)
        
        current_T_val = layer_T[-1]

    if debug:
        print(f"  [Env Builder] Reached deep boundary P={p_bottom_bar:.2f} bar, T={current_T_val:.1f} K.")

    return {
        'p': interp1d(full_P, full_P, kind='linear', fill_value="extrapolate"),
        't': interp1d(full_P, np.log10(full_T), kind='linear', fill_value="extrapolate"),
        'rho': interp1d(full_P, np.log10(full_Rho), kind='linear', fill_value="extrapolate"),
        's': interp1d(full_P, full_S, kind='nearest', fill_value="extrapolate"),
        'z': interp1d(full_P, full_Z, kind='nearest', fill_value="extrapolate"),
    }

# =============================================================================
# 3. WATER WORLD INTEGRATOR LOGIC
# =============================================================================

def run_water_world_integration(Pc_bar: float, P_int_bar: float, params: dict, eos_data: dict) -> dict:
    debug = params.get('debug', False)
    fluid_stack = eos_data['fluid']
    water_eos = eos_data['water']
    iron_frac = params.get('iron_fraction', 0.33)
    
    if P_int_bar <= params['P_surf']:
        if debug: print(f"  [Water Builder] Bare Mantle Bypass! P_int ({P_int_bar:.2f}) < P_surf ({params['P_surf']:.2f})")
        env = None
        logP_int = np.log10(max(1e-5, P_int_bar))
        T_int = params.get('T_surf', 500.0)
    else:
        env = build_staircase_envelope(
            params['P_surf'], P_int_bar, params['T_surf'], params['z_profile'], fluid_stack, debug=debug
        )
        if env is None: return None
        logP_int = np.log10(P_int_bar)
        
        try:
            T_int = 10 ** float(env['t'](logP_int))
            if np.isnan(T_int): T_int = params.get('T_surf', 500.0)
        except Exception:
            T_int = params.get('T_surf', 500.0)
            
    try:
        target_s = float(water_eos['S'](logP_int, np.log10(T_int)))
        if np.isnan(target_s):
            target_s = float(water_eos['S_near'](logP_int, np.log10(T_int)))
    except Exception:
        target_s = 3000.0

    pre_core = integrate_core(Pc_bar, params['M_rock'], T_center=5000.0, iron_fraction=iron_frac)
    P_rock_top_guess = pre_core['P_top']
    p_log_core = np.log10(max(1e-5, P_rock_top_guess))

    mantle_stepper = get_stepper(water_eos)
    t_guess_deep = T_int * (P_rock_top_guess / max(1e-5, P_int_bar)) ** 0.25
    lt_core, lrho_core = mantle_stepper.get_state(p_log_core, np.log10(t_guess_deep), target_s)
    
    if np.isnan(lt_core) or np.isnan(lrho_core):
        lt_core = np.log10(t_guess_deep)
        lrho_core = float(water_eos['rho_near'](p_log_core, lt_core))

    T_core_match = max(T_int, 10 ** lt_core)
    Rho_core_match = 10 ** lrho_core 
    T_center_calc = T_core_match * (Pc_bar / max(1e-5, P_rock_top_guess)) ** 0.1

    core_res = integrate_core(
        Pc_bar, params['M_rock'], T_center=T_center_calc, 
        iron_fraction=iron_frac, env_rho_base=Rho_core_match, nabla_ad=0.1
    )
    if not core_res['valid']: return None
        
    P_rock_top = core_res['P_top']
    R_rock = core_res['R_core']

    r = R_rock
    m = core_res['M_actual']
    p_pa = P_rock_top * 1e5
    current_lt = np.log10(core_res['T'][-1])

    water_R, water_M, water_P, water_Rho, water_T, water_S, water_Z = [], [], [], [], [], [], []
    k = 0
    
    while p_pa > P_int_bar * 1e5:
        p_log = np.log10(max(1e-5, p_pa / 1e5))
        next_lt, next_lrho = mantle_stepper.get_state(p_log, current_lt, target_s)

        if np.isnan(next_lt) or np.isnan(next_lrho):
            next_lt = current_lt
            next_lrho = float(water_eos['rho_near'](p_log, next_lt))

        current_lt = next_lt
        rho = 10 ** next_lrho
        temp = 10 ** next_lt

        g = (c.G_CONST * m) / (r**2)
        H_P = p_pa / (rho * g) if (rho * g) > 0 else 1e5
        dr = min(5000.0, H_P * 0.05)

        dm_dr = 4 * np.pi * r**2 * rho
        target_water_mass = params['M_water']
        current_water_mass = m - core_res['M_actual']
        
        dr = min(dr, (target_water_mass - current_water_mass) / dm_dr)
        if dr < 1.0: dr = 1.0

        p_new = p_pa - rho * g * dr
        if p_new < P_int_bar * 1e5:
            frac = (p_pa - P_int_bar * 1e5) / (p_pa - p_new)
            dr *= frac
            p_new = P_int_bar * 1e5
            
        m += dm_dr * dr
        r += dr
        p_pa = p_new
        
        if k % 100 == 0:
            water_R.append(r); water_M.append(m); water_P.append(p_log)
            water_Rho.append(rho); water_T.append(temp)
            water_S.append(target_s); water_Z.append(1.0)
        k += 1

    water_R.append(r); water_M.append(m)
    water_P.append(np.log10(max(1e-5, p_pa / 1e5)))
    water_Rho.append(rho); water_T.append(temp)
    water_S.append(target_s); water_Z.append(1.0)

    p_surf_pa = params['P_surf'] * 1e5
    R_water_top = r
    
    temp = T_int if 'T_int' in locals() and not np.isnan(T_int) else params.get('T_surf', 500.0)
    rho = 10 ** next_lrho if 'next_lrho' in locals() and not np.isnan(next_lrho) else 10.0
    entr = target_s if 'target_s' in locals() and not np.isnan(target_s) else 3000.0
    z_val = params.get('z_base', 0.05)

    if env is not None:
        while p_pa > p_surf_pa:
            p_log = np.log10(max(1e-5, p_pa / 1e5))
            
            try:
                temp = 10 ** float(env['t'](p_log))
                rho = 10 ** float(env['rho'](p_log))
                entr = float(env['s'](p_log))
                z_val = float(env['z'](p_log))
            except Exception:
                pass 

            g = (c.G_CONST * m) / (r**2)
            H_P = p_pa / (rho * g) if (rho * g) > 0 else 1e5
            dr = min(5000.0, H_P * 0.05, abs((p_pa - p_surf_pa) / (rho * g)) * 0.5)
            if dr < 1.0: dr = 1.0

            dm_dr = 4 * np.pi * r**2 * rho
            p_new = p_pa - rho * g * dr

            if p_new < p_surf_pa:
                dr = (p_pa - p_surf_pa) / (rho * g)
                p_new = p_surf_pa

            m += dm_dr * dr
            r += dr
            p_pa = p_new
            
            if k % 100 == 0:
                water_R.append(r); water_M.append(m); water_P.append(p_log)
                water_Rho.append(rho); water_T.append(temp)
                water_S.append(entr); water_Z.append(z_val)
            k += 1

    water_R.append(r); water_M.append(m); water_P.append(np.log10(max(1e-5, p_pa / 1e5)))
    water_Rho.append(rho); water_T.append(temp)
    water_S.append(entr); water_Z.append(z_val)

    result = {
        "M_total": m, "R_total": r, 
        "M_core_actual": core_res['M_actual'], 
        "R": np.concatenate([core_res['R'], water_R]),
        "M": np.concatenate([core_res['M'], water_M]),
        "P": np.concatenate([core_res['P'], water_P]),
        "Rho": np.concatenate([core_res['Rho'], water_Rho]),
        "T": np.concatenate([core_res['T'], water_T]),
        "S": np.concatenate([core_res['S'], water_S]),
        "Z": np.concatenate([core_res['Z'], water_Z]),
        "R_rock": R_rock, "R_int": R_water_top
    }
    
    return result

def integrate_water_world(logPc: float, params: dict, eos_data: dict) -> dict:
    Pc_bar = 10 ** logPc
    debug = params.get('debug', False)
    fluid_stack = eos_data['fluid']
    water_eos = eos_data['water']
    iron_frac = params.get('iron_fraction', 0.33)
    target_water_mass = params['M_water']

    pre_core = integrate_core(Pc_bar, params['M_rock'], T_center=5000.0, iron_fraction=iron_frac)
    
    if not pre_core['valid']:
        return None
        
    P_rock_top_guess = pre_core['P_top']
    logP_rock_top = np.log10(max(1e-5, P_rock_top_guess))

    if P_rock_top_guess <= params['P_surf'] + 0.1:
        return None

    def mass_error(logP_int_guess: float) -> float:
        P_int_guess = 10 ** logP_int_guess
        p_surf_bar = params['P_surf']

        if P_int_guess <= p_surf_bar:
            env = None
            T_int = params.get('T_surf', 500.0)
        else:
            env = build_staircase_envelope(
                p_surf_bar, P_int_guess, params['T_surf'], params['z_profile'], fluid_stack, debug=False
            )
            if env is None: return 1e30
            
            try:
                T_int = 10 ** float(env['t'](logP_int_guess))
                if np.isnan(T_int): T_int = params.get('T_surf', 500.0)
            except Exception:
                T_int = params.get('T_surf', 500.0)
                
        try:
            target_s = float(water_eos['S'](logP_int_guess, np.log10(T_int)))
            if np.isnan(target_s):
                target_s = float(water_eos['S_near'](logP_int_guess, np.log10(T_int)))
        except Exception:
            return 1e30

        mantle_stepper = get_stepper(water_eos)
        t_guess_deep = T_int * (P_rock_top_guess / max(1e-5, P_int_guess)) ** 0.25
        
        lt_core, lrho_core = mantle_stepper.get_state(
            logP_rock_top, np.log10(t_guess_deep), target_s
        )
        
        if np.isnan(lt_core) or np.isnan(lrho_core):
            lt_core = np.log10(t_guess_deep)
            lrho_core = float(water_eos['rho_near'](logP_rock_top, lt_core))
            
        T_core_match = max(T_int, 10 ** lt_core)
        Rho_core_match = 10 ** lrho_core 

        T_center_calc = T_core_match * (Pc_bar / max(1e-5, P_rock_top_guess)) ** 0.1

        core_iter = integrate_core(
            Pc_bar, params['M_rock'], T_center=T_center_calc, 
            iron_fraction=iron_frac, env_rho_base=Rho_core_match, nabla_ad=0.1
        )
        
        if not core_iter['valid']: return 1e30

        r = core_iter['R_core']
        m = core_iter['M_actual']
        p_pa = core_iter['P_top'] * 1e5
        current_lt = np.log10(core_iter['T'][-1])

        while p_pa > P_int_guess * 1e5:
            p_log = np.log10(max(1e-5, p_pa / 1e5))
            next_lt, next_lrho = mantle_stepper.get_state(p_log, current_lt, target_s)

            if np.isnan(next_lt) or np.isnan(next_lrho):
                next_lt = current_lt
                next_lrho = float(water_eos['rho_near'](p_log, next_lt))
                
            current_lt = next_lt
            rho = 10 ** next_lrho

            g = (c.G_CONST * m) / (r**2)
            H_P = p_pa / (rho * g) if (rho * g) > 0 else 1e5
            
            # 🛑 THE FIX: MASSIVE step sizes for the Scout to fly through convergence!
            dr = min(100000.0, H_P * 0.2)  
            
            dm_dr = 4 * np.pi * r**2 * rho
            p_new = p_pa - rho * g * dr

            if p_new < P_int_guess * 1e5:
                frac = (p_pa - P_int_guess * 1e5) / (p_pa - p_new)
                dr *= frac
                p_new = P_int_guess * 1e5

            m += dm_dr * dr
            r += dr
            p_pa = p_new

        built_water_mass = m - core_iter['M_actual']
        error = built_water_mass - target_water_mass
        return error

    try:
        # 🛑 THE FIX: Reduced to 7 scan points to save processing time
        scan_pts = np.linspace(np.log10(params['P_surf']), logP_rock_top - 1e-4, 15)
        vals = [mass_error(p) for p in scan_pts]

        bracket = None
        for i in range(len(vals) - 1):
            if vals[i] < 1e29 and vals[i+1] < 1e29:
                if np.sign(vals[i]) != np.sign(vals[i + 1]):
                    bracket = (scan_pts[i], scan_pts[i + 1])
                    break

        if not bracket:
            return None

        # 🛑 THE FIX: Loosened xtol to 1e-1 so brentq stops immediately after finding a rough bracket!
        root_logP = brentq(mass_error, bracket[0], bracket[1], xtol=1e-3)
        converged_P_int = 10 ** root_logP
        
        return run_water_world_integration(Pc_bar, converged_P_int, params, eos_data)

    except Exception as e:
        return None

# =============================================================================
# 4. GENERIC PLANET INTEGRATOR (Gas Giant / Sub-Neptune)
# =============================================================================

def integrate_planet(logPc: float, params: dict, eos_data: dict) -> dict:
    debug = params.get('debug', False)
    Pc_bar = 10 ** logPc
    iron_frac = params.get('iron_fraction', 0.33)

    pilot_core = integrate_core(Pc_bar, params['M_core'], T_center=5000.0, iron_fraction=iron_frac)
    
    if not pilot_core['valid']:
        return None

    P_int_guess = pilot_core['P_top']
    
    if P_int_guess <= params['P_surf']:
        result = {
            "M_total": pilot_core['M_actual'], "R_total": pilot_core['R_core'], 
            "M_core_actual": pilot_core['M_actual'],
            "R": pilot_core['R'], "M": pilot_core['M'], "P": pilot_core['P'], 
            "Rho": pilot_core['Rho'], "T": pilot_core['T'], "S": pilot_core['S'], "Z": pilot_core['Z'], 
            "R_rock": pilot_core['R_core'], "R_int": pilot_core['R_core']
        }
        result["M_Z_total"] = pilot_core['M_actual']
        result["dt_ds_total"] = np.inf
        result["dt_ds_layers"] = {}
        return result

    try:
        env = build_staircase_envelope(
            params['P_surf'], P_int_guess, params['T_surf'], params['z_profile'], eos_data['fluid']
        )
    except Exception:
        return None
        
    if env is None:
        return None

    try:
        # 🛑 THE FIX: Convert BOTH Temperature and Density back to linear for Gas Giants!
        T_match = 10 ** float(env['t'](np.log10(P_int_guess)))
        if np.isnan(T_match): T_match = 5000.0
        Rho_match = 10 ** float(env['rho'](np.log10(P_int_guess)))
        if np.isnan(Rho_match): Rho_match = 0.0
    except Exception:
        T_match = 5000.0
        Rho_match = 0.0

    predictor_core = integrate_core(
        Pc_bar, params['M_core'], T_center=T_match, iron_fraction=iron_frac
    )
    P_int_predict = predictor_core['P_top']
    T_center_calc = T_match * (Pc_bar / max(1e-5, P_int_predict)) ** 0.1

    final_core = integrate_core(
        Pc_bar, params['M_core'], T_center=T_center_calc, 
        iron_fraction=iron_frac, env_rho_base=Rho_match, nabla_ad=0.1
    )
    
    P_int_final = final_core['P_top']
    R_int_final = final_core['R_core']
    
    if P_int_final <= params['P_surf']:
        return None

    r = R_int_final
    m = final_core['M_actual']
    p_pa = P_int_final * 1e5
        
    R_h, M_h, P_h = list(final_core['R']), list(final_core['M']), list(final_core['P'])
    Rho_h, T_h = list(final_core['Rho']), list(final_core['T'])
    S_h, Z_h = list(final_core['S']), list(final_core['Z'])
    
    p_min_log = np.log10(params['P_surf'])
    p_max_log = np.log10(P_int_final)
    k = 0

    while (p_pa / 1e5) > params['P_surf']:
        p_log = np.log10(max(1e-10, p_pa / 1e5))
        p_lookup = np.clip(p_log, p_min_log, p_max_log)
        
        try:
            # 🛑 THE FIX: Convert logarithmic outputs to physical linear units for Gas Giants!
            rho = 10 ** float(env['rho'](p_lookup))
            temp = 10 ** float(env['t'](p_lookup))
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

        m += 4 * np.pi * r**2 * rho * dr
        r += dr
        p_pa = p_new
        
        if k % 10 == 0:
            R_h.append(r); M_h.append(m); P_h.append(p_log)
            Rho_h.append(rho); T_h.append(temp)
            S_h.append(entr); Z_h.append(z_val)
        k += 1

    R_h.append(r); M_h.append(m); P_h.append(np.log10(max(1e-5, p_pa / 1e5)))
    Rho_h.append(rho); T_h.append(temp)
    S_h.append(entr); Z_h.append(z_val)

    result = {
        "M_total": m, "R_total": r, 
        "M_core_actual": final_core['M_actual'],
        "R": np.array(R_h), "M": np.array(M_h), "P": np.array(P_h), 
        "Rho": np.array(Rho_h), "T": np.array(T_h), "S": np.array(S_h), "Z": np.array(Z_h), 
        "R_rock": R_int_final, "R_int": R_int_final
    }

    result["M_Z_total"] = utils.evaluate_heavy_element_mass(result, params.get('z_base', 0.0))
    t_eff = params.get('T_int', params.get('T_surf', 500.0))
    c_info = utils.calculate_staircase_dt_ds(result, t_eff)
    result["dt_ds_total"] = c_info['total_dt_ds']
    result["dt_ds_layers"] = c_info['layer_contributions']

    return result