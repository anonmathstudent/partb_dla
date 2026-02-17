"""
Targeted update script: Recalculates BOTH the Box-Counting Dimension (D)
and its Error (D_err) using the project's standard metric function.

- Uses src.analysis.metrics.calculate_box_dim (enforces 8px cutoff)
- Updates both 'D' and 'D_err' columns in scalars.csv
- Preserves all other data (Beta, Rg, etc.)
"""
import os
import sys
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# --- 1. SETUP PATHS ---
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == "notebooks" else _SCRIPT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- KEY IMPORT ---
from src.analysis.metrics import calculate_box_dim

# Define directories
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "processed")
CLUSTER_DIR = os.path.join(PROJECT_ROOT, "results", "analysis_clusters")

MODELS = ["Lattice", "OffLattice", "Hybrid"]
SIZES = ["1M", "10M"]

# --- 2. PROCESSING LOOP ---
def update_model_scalars(model, size_label):
    csv_path = os.path.join(RESULTS_DIR, f"{model}_{size_label}_scalars.csv")
    
    if not os.path.exists(csv_path):
        print(f"Skipping {model}_{size_label}: scalars.csv not found.")
        return

    print(f"Updating {model}_{size_label}...")
    
    # Load existing CSV
    df = pd.read_csv(csv_path)
    
    # Find cluster files
    # We sort them to ensure alignment with CSV rows
    cluster_pattern = os.path.join(CLUSTER_DIR, model, size_label, "*.npz")
    cluster_files = sorted(glob.glob(cluster_pattern))
    
    # Safety Check
    if len(cluster_files) != len(df):
        print(f"Warning: CSV has {len(df)} rows but found {len(cluster_files)} files.")
        print("  - Aborting update for this model to prevent row mismatch.")
        return

    new_D_values = []
    new_D_err_values = []
    debug_printed = False
    for fpath in tqdm(cluster_files, desc="Recalculating D & Error"):
        try:
            data = np.load(fpath)
            
            # --- MODIFIED EXTRACTION LOGIC ---
            if 'positions' in data:
                coords = data['positions']
            elif 'coords' in data:
                coords = data['coords']
            else:
                if not debug_printed:
                    print(f"\n[ERROR] Key mismatch in file: {os.path.basename(fpath)}")
                    print(f"Found keys: {data.files}")
                    debug_printed = True # Stop printing after the first error
                
                new_D_values.append(np.nan)
                new_D_err_values.append(np.nan)
                continue

            if not debug_printed:
                print(f"\n--- DEBUG CHECK for {model} ---")
                print(f"File: {os.path.basename(fpath)}")
                print(f"Coords Shape: {coords.shape}")
                print(f"Coords Type: {coords.dtype}")
                print(f"Max Coord Value: {np.max(np.abs(coords))}")
                
                # Test the function explicitly and print result
                test_res = calculate_box_dim(coords)
                print(f"Function Result: {test_res}")
                debug_printed = True
            result = calculate_box_dim(coords)
            #print(result)
            new_D_values.append(result['D'])
            new_D_err_values.append(result['D_err'])
            
        except Exception as e:
            print(f"Failed on {fpath}: {e}")
            new_D_values.append(np.nan)
            new_D_err_values.append(np.nan)

    # Update DataFrame
    if len(new_D_values) == len(df):
        df['D'] = new_D_values
        df['D_err'] = new_D_err_values
        
        df.to_csv(csv_path, index=False)
        print(f"Saved updated {csv_path} with new D and D_err.")
    else:
        print(f"Error: Length mismatch. CSV not updated.")

# --- 3. MAIN ---
if __name__ == "__main__":
    for model in MODELS:
        for size in SIZES:
            update_model_scalars(model, size)