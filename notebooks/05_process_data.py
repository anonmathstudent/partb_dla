"""
Heavy data processing for DLA analysis: extract scalars, sector data, and density
grids once; save to results/processed/ for fast dashboard loading.

Run from project root:  python notebooks/05_process_data.py
Or from notebooks/:     python 05_process_data.py
"""
import os
import sys

_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == "notebooks" else _SCRIPT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from tqdm import tqdm

from src.analysis import processing as proc

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODELS = ["Lattice", "OffLattice", "Hybrid"]
SIZES = ["1M", "10M"]  # size labels for paths and filenames
SIZE_TO_N = {"1M": 1_000_000, "10M": 10_000_000}
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "processed")
LIMIT = 1000
NUM_WORKERS = 4
NUM_SECTORS = 360


def file_pattern(model: str, size_label: str) -> str:
    """Glob pattern for cluster NPZ files for a given model and size."""
    return os.path.join(PROJECT_ROOT, "results", "analysis_clusters", model, size_label, "*.npz")


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def process_model_size(model: str, size_label: str) -> None:
    """Run Steps A, B, C for one (model, size) and save to results/processed/."""
    pattern = file_pattern(model, size_label)
    snapshot_N = SIZE_TO_N[size_label]
    base_name = f"{model}_{size_label}"

    # Step A: Scalars (beta, Rg, D, etc.)
    # Take regular snapshots from 10^5 particles to max
    start_log = 5.0  
    end_log = np.log10(snapshot_N)
    num_snaps = int(5 * (end_log - start_log)) + 1 
    snapshots = np.logspace(start_log, end_log, num=num_snaps).astype(int)
    snapshots = np.unique(snapshots) # Ensure no duplicate sizes
    if len(snapshots) == 0 or snapshots[-1] != snapshot_N: 
        snapshots = np.append(snapshots, snapshot_N) # Guarantee the absolute final N is always the last snapshot

    df_scalars = proc.batch_analysis(
        file_pattern=pattern,
        snapshots=snapshots.tolist(), # Convert to standard Python list
        num_workers=NUM_WORKERS,
        limit=LIMIT,
    )
    scalars_path = os.path.join(OUT_DIR, f"{base_name}_scalars.csv")
    df_scalars.to_csv(scalars_path, index=False)

    # Step B: Sector data (raw 360-sector resolution)
    times, sector_grid = proc.batch_sector_analysis(
        file_pattern=pattern,
        max_time=snapshot_N,
        num_sectors=NUM_SECTORS,
        num_workers=NUM_WORKERS,
        limit=LIMIT,
    )
    if times is not None and sector_grid is not None:
        sectors_path = os.path.join(OUT_DIR, f"{base_name}_sectors.npz")
        np.savez(sectors_path, times=times, grid=sector_grid)
    else:
        # Save empty placeholder so dashboard can still load
        np.savez(
            os.path.join(OUT_DIR, f"{base_name}_sectors.npz"),
            times=np.array([]),
            grid=np.zeros((0, NUM_SECTORS, 0)),
        )

    # Step C: Density map
    grid, bins, max_r, rg = proc.generate_density_grid(
        file_pattern=pattern,
        snapshot_N=snapshot_N,
        limit_files=LIMIT,
    )
    density_path = os.path.join(OUT_DIR, f"{base_name}_density.npz")
    np.savez(density_path, grid=grid, bins=bins, max_r=max_r, rg=rg)


def main():
    ensure_out_dir()
    combinations = [(m, s) for m in MODELS for s in SIZES]
    for model, size_label in tqdm(combinations, desc="Model × Size"):
        try:
            process_model_size(model, size_label)
        except Exception as e:
            print(f"Error processing {model} / {size_label}: {e}")
            raise
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
