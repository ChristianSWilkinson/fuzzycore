import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

sys.path.append('./..')

# Import your framework modules
from src import solver
from src import constants as c

M_KEPLER11E = 7.95 * c.M_EARTH
RESULTS_FILE = "kepler11e_sweep_results.csv"
TEMP_FILE = "kepler11e_sweep_temp.csv"

def run_single_model(m_core, lock):
    """
    Function to be executed by a parallel worker.
    """
    params = {
        'M_core': m_core * c.M_EARTH,   
        'M_rock': m_core * c.M_EARTH,   
        'M_water': 0.0,                 
        'P_surf': 1.0,
        'T_surf': 900.0,                
        'z_base': 0.05,                 
        'z_profile': np.linspace(0.05, 1.0, 20), 
        'sigma_val': 0.25,              
        'iron_fraction': 0.33,          
        'debug': False
    }
    
    trial_id = f"K11e_core_{m_core:.3f}"
    
    try:
        res = solver.solve_structure(
            target_val=M_KEPLER11E,
            params=params,
            mode='mass',
            trial_id=trial_id,
            csv_file=TEMP_FILE,
            write_lock=lock
        )
        
        if res is not None:
            r_tot_re = res['R'][-1] / c.R_EARTH
            m_z_tot_kg = res.get('M_Z_total', m_core * c.M_EARTH)
            m_z_tot_me = m_z_tot_kg / c.M_EARTH
            m_dilute = max(m_z_tot_me - m_core, 0.0)
            
            result_dict = {
                'M_core_Me': m_core,
                'M_dilute_Me': m_dilute,
                'M_Z_total_Me': m_z_tot_me,
                'R_total_Re': r_tot_re
            }
            return (True, m_core, result_dict)
        else:
            return (False, m_core, "Solver returned None (failed to converge)")
    except Exception as e:
        return (False, m_core, str(e))

if __name__ == '__main__':
    print("==================================================")
    print(" Kepler-11e 1D Parameter Sweep (Parallel Mode) ")
    print("==================================================")
    
    # 1. Define the grid of core masses to test
    core_masses = np.linspace(7.5, 2.0, 15)
    
    # 2. Checkpoint Logic: Read what is already done
    completed_masses = []
    if os.path.exists(RESULTS_FILE):
        try:
            df_existing = pd.read_csv(RESULTS_FILE)
            completed_masses = df_existing['M_core_Me'].values.tolist()
            print(f"[*] Found {len(completed_masses)} previously completed runs in {RESULTS_FILE}.")
        except Exception as e:
            print(f"[*] Could not read {RESULTS_FILE}: {e}")
            
    # Filter out masses that are already completed (using np.isclose to avoid float errors)
    tasks_to_run = []
    for m in core_masses:
        if not any(np.isclose(m, comp, atol=1e-3) for comp in completed_masses):
            tasks_to_run.append(m)
            
    print(f"[*] Tasks remaining to compute: {len(tasks_to_run)} / {len(core_masses)}")
    
    # 3. Parallel Execution
    if tasks_to_run:
        # Memory Management: Limit workers to max 4, or CPU count - 1
        n_workers = min(mp.cpu_count() - 1, 4)
        n_workers = max(1, n_workers) 
        
        print(f"[*] Starting parallel execution with {n_workers} concurrent workers...")
        
        manager = mp.Manager()
        shared_lock = manager.Lock()
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all remaining tasks
            futures = {executor.submit(run_single_model, m, shared_lock): m for m in tasks_to_run}
            
            # As they finish (in any order), process the results
            for future in as_completed(futures):
                success, m_core, data = future.result()
                
                if success:
                    print(f"  [+] Success! M_core = {m_core:.2f} M_E -> Radius = {data['R_total_Re']:.2f} R_E")
                    # Immediately save to results file so we don't lose it if the script crashes later
                    df_new = pd.DataFrame([data])
                    df_new.to_csv(RESULTS_FILE, mode='a', header=not os.path.exists(RESULTS_FILE), index=False)
                else:
                    print(f"  [-] Failed for M_core = {m_core:.2f} : {data}")
                    
        print("[*] All parallel tasks finished.")

    # 4. Generate Plot from fully completed results file
    print("[*] Generating plot...")
    if os.path.exists(RESULTS_FILE):
        df_plot = pd.read_csv(RESULTS_FILE)
        
        # Sort by M_core to make the line plot continuous
        df_plot = df_plot.sort_values(by='M_core_Me', ascending=False)
        
        if len(df_plot) > 1:
            plt.figure(figsize=(8, 6))

            # Plot Dilute Mass vs Radius
            plt.plot(df_plot['M_dilute_Me'], df_plot['R_total_Re'], 
                     marker='o', markersize=8, linestyle='-', color='#d95f02', linewidth=2.5)

            # Highlight Kepler-11e observed radius
            plt.axhline(y=4.67, color='black', linestyle='--', alpha=0.6, label='Kepler-11e Observed Radius')

            plt.xlabel(r"Dilute Metal Mass in Envelope ($M_\oplus$)", fontsize=14)
            plt.ylabel(r"Total Planetary Radius ($R_\oplus$)", fontsize=14)
            plt.title(r"Kepler-11e Super-Puff Inflation via Fuzzy Core ($M_{tot} = 7.95 \ M_\oplus$)", fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend(fontsize=12)

            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            plt.tight_layout()
            plt.savefig("kepler11e_dilute_radius_curve.pdf", format='pdf', bbox_inches='tight')
            print("[*] Plot successfully saved as 'kepler11e_dilute_radius_curve.pdf'.")
        else:
            print("[!] Not enough successful data points to generate a line plot.")
    else:
        print("[!] No results file found to plot.")