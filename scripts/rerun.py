"""
Merge the polytropic_05 rerun result back into the main mantle CSV.

Reads:
  mantle_thermal_sensitivity.csv            (the main batch output)
  mantle_polytropic05_rerun.csv             (single corrected row from rerun)

Writes:
  mantle_thermal_sensitivity_corrected.csv  (merged result; main CSV untouched)

Logic
-----
- Finds the row in the main CSV matching the corrected row's
  (config_name, mode_label) pair.
- Replaces the structural / thermal columns with the corrected values,
  preserving everything else (input parameters, target mass, etc.).
- Records the initial_log_pc used in the rerun in a new column so the
  provenance is traceable.

The original CSV is not modified -- the merged output goes to a new file.
If you're happy with the result, just rename the new file to overwrite.
"""

import os
import shutil
import numpy as np
import pandas as pd


MAIN_CSV    = "../data/mantle_thermal_sensitivity.csv"
RERUN_CSV   = "../data/mantle_polytropic05_rerun.csv"
OUT_CSV     = "../data/mantle_thermal_sensitivity_corrected.csv"

# Columns to overwrite from the rerun (everything structure / thermal)
OVERWRITE_COLS = [
    'R_total_Re',
    'R_rock_Re',
    'R_water_Re',
    'R_mantle_thickness_Re',
    'M_total_Me_achieved',
    'P_center_GPa',
    'T_center_K',
    'T_cmb_K',          # rerun may not have this; pandas will keep NaN if missing
    'Status',
]


def main():
    main_df  = pd.read_csv(MAIN_CSV)
    rerun_df = pd.read_csv(RERUN_CSV)

    if len(rerun_df) != 1:
        raise ValueError(f"Expected exactly one corrected row in {RERUN_CSV}, "
                         f"found {len(rerun_df)}.")
    rerun_row = rerun_df.iloc[0]

    cfg  = rerun_row['config_name']
    mode = rerun_row['mode_label']

    # Find matching row in the main CSV
    mask = (main_df['config_name'] == cfg) & (main_df['mode_label'] == mode)
    n_matches = mask.sum()
    if n_matches != 1:
        raise ValueError(f"Expected exactly 1 matching row in main CSV for "
                         f"({cfg}, {mode}); found {n_matches}.")

    idx = main_df.index[mask][0]

    print("=" * 72)
    print(f" Merging rerun result for ({cfg}, {mode})")
    print("=" * 72)
    print(f"\n  BEFORE (main CSV row {idx}):")
    for col in OVERWRITE_COLS:
        if col in main_df.columns:
            v = main_df.at[idx, col]
            print(f"    {col:<28s} = {v}")

    # Apply the overwrite
    for col in OVERWRITE_COLS:
        if col in main_df.columns and col in rerun_row.index:
            new_val = rerun_row[col]
            # Only overwrite if the rerun has a finite value
            try:
                if isinstance(new_val, str) or np.isfinite(new_val):
                    main_df.at[idx, col] = new_val
            except (TypeError, ValueError):
                main_df.at[idx, col] = new_val

    # Add provenance column for the rerun
    if '_rerun_initial_log_pc' in rerun_row.index:
        if '_rerun_initial_log_pc' not in main_df.columns:
            main_df['_rerun_initial_log_pc'] = np.nan
        main_df.at[idx, '_rerun_initial_log_pc'] = rerun_row['_rerun_initial_log_pc']

    print(f"\n  AFTER:")
    for col in OVERWRITE_COLS:
        if col in main_df.columns:
            v = main_df.at[idx, col]
            print(f"    {col:<28s} = {v}")
    if '_rerun_initial_log_pc' in main_df.columns:
        v = main_df.at[idx, '_rerun_initial_log_pc']
        print(f"    {'_rerun_initial_log_pc':<28s} = {v}")

    out_dir = os.path.dirname(OUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    main_df.to_csv(OUT_CSV, index=False)
    print(f"\n[*] Merged CSV written to: {OUT_CSV}")
    print(f"[*] Original CSV preserved at: {MAIN_CSV}")
    print(f"[*] To make the merged version the default, run:")
    print(f"      mv {OUT_CSV} {MAIN_CSV}")
    print("=" * 72)


if __name__ == '__main__':
    main()