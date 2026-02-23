import numpy as np
import pandas as pd
import glob
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import gc
from . import metrics

def load_cluster(filepath):
    """Load a single .npz file"""
    try:
        with np.load(filepath, allow_pickle=True) as data:
            coords = data['positions']
            if coords.shape[0] == 2 and coords.shape[1] > 2:
                coords = coords.T
        return coords
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_single(filepath, snapshots):
    """
    load one cluster, truncate it to various sized 'snapshots', and calculate metrics for each one.
    """

    coords = load_cluster(filepath)
    if coords is None:
        return []
    
    results = []
    full_N = len(coords)
    full_t = np.arange(1, full_N + 1)

    #center cluster
    com = coords[0]
    coords_centered = coords - com

    for N in snapshots:
        if N > full_N:
            continue

        # --- Truncation ---
        sub_coords = coords_centered[:N]
        sub_t = full_t[:N]

        # --- Metrics Calculation ---
        # 1. growth rate
        fit_start = 1000 if N > 2000 else N//4
        res_beta = metrics.calculate_radius_gyration(sub_coords, sub_t, fit_start=fit_start)

        # 2. box-counting dimension
        max_box = int(np.max(np.abs(sub_coords))) + 1
        res_box = metrics.calculate_box_dim(sub_coords, max_box_size=max_box)

        # --- Aggregate Results ---
        row = {
            'filepath': filepath,
            'snapshot_N': N,
            **res_beta,
            **res_box
        }
        results.append(row)
    return results

def batch_analysis(file_pattern, snapshots, num_workers=4, limit=None):
    """
    Process multiple cluster files in parallel and aggregate results.
    """
    file_list = glob.glob(file_pattern)
    if limit is not None:
        print(f'Limiting to first {limit} files for analysis.')
        file_list = file_list[:limit]
    print(f'Found {len(file_list)} files to process. Starting analysis on {num_workers} cores...')
    all_results = []

    with Pool(processes=num_workers) as pool:
        func = partial(process_single, snapshots=snapshots)
        for result in tqdm(pool.imap_unordered(func, file_list), total=len(file_list)):
            all_results.extend(result)

    df_results = pd.DataFrame(all_results)
    return df_results

def process_sector_single(filepath, checkpoints, num_sectors=360):
    """
    load one cluster, and calculate its sector-wise evolution
    returns the grid (sectors x time) or None if failed
    """
    coords = load_cluster(filepath)
    if coords is None:
        return None
    
    t = np.arange(1, len(coords) + 1)
    
    # We deleted all the local math. Just pass the checkpoints straight through!
    _, _, grid = metrics.calculate_sector_evolution(
        coords, t, 
        num_sectors=num_sectors, 
        checkpoints=checkpoints
    )
    return grid

def batch_sector_analysis(file_pattern, max_time, num_sectors=360, num_workers=4, limit=None):
    """
    Runs sector evolution analysis on multiple files.
    returns time points (checkpoints used)and agg_data (3D array containing Rg vales)
        """
    files = glob.glob(file_pattern)
    if limit is not None:
        files = files[:limit]
        print(f'Limiting to first {limit} files for sector analysis.')
    print(f'Found {len(files)} files to process for sector analysis on {num_workers} cores...')
    
    # generate universal checkpoints
    start_log = 3.0  # Start at 1000
    end_log = np.log10(max_time)
    num_points = int(20 * (end_log - start_log))
    
    universal_checkpoints = np.unique(np.logspace(start_log, end_log, num=num_points).astype(int))
    universal_checkpoints = universal_checkpoints[universal_checkpoints <= max_time]
    results = []
    with Pool(processes=num_workers) as pool:
        func = partial(process_sector_single, checkpoints=universal_checkpoints, num_sectors=num_sectors)
        for grid in tqdm(pool.imap_unordered(func, files), total=len(files)):
            if grid is not None:
                results.append(grid)

    if not results:
        print("No valid sector data computed.")
        return None, None

    agg_data = np.stack(results)  # shape: (num_files, num_sectors, time_points)

    print(f'Aggregated shape: {agg_data.shape} (files, sectors, time_steps)')
    
    return universal_checkpoints, agg_data

def compute_anisotropy_metrics(time_points, agg_data):
    """
    Iterates over the batch of sector grids and computes anisotropy statistics.
    Returns:
        df_metrics: DataFrame with one row per file.
        summary: Dictionary with Mean +/- Std for key metrics.
    """
    num_files = agg_data.shape[0]
    records = []
    
    print(f"Computing anisotropy metrics for {num_files} files...")
    
    for i in tqdm(range(num_files), desc ="Anisotropy Metrics"):
        grid = agg_data[i]
        
        # 1. axes vs diag ratio
        b_ax, b_dg, b_ratio = metrics.anisotropy_ratio(time_points, grid)
        
        # 2. fourier score
        a0, a4, f_score = metrics.anisotropy_fourier(time_points, grid)
        
        records.append({
            'beta_axis': b_ax,
            'beta_diag': b_dg,
            'beta_ratio': b_ratio,
            'A0': a0,
            'A4': a4,
            'fourier_score': f_score
        })
        
    df_metrics = pd.DataFrame(records)
    
    # Summary Statistics (Mean +/- Std)
    summary = {
        'beta_ratio_mean': df_metrics['beta_ratio'].mean(),
        'beta_ratio_std':  df_metrics['beta_ratio'].std(),
        'fourier_score_mean': df_metrics['fourier_score'].mean(),
        'fourier_score_std':  df_metrics['fourier_score'].std()
    }
    
    return df_metrics, summary 

def generate_density_grid(file_pattern, snapshot_N, grid_size=1000, limit_files=None):
    """
    Generate a 2D histogram (density map) for later visualisation.
    Runs sequentially to avoid memory issues.
    """
    files = glob.glob(file_pattern)
    if limit_files:
        files = files[:limit_files]
    

    max_r = 1.2 * (snapshot_N ** 0.6) * 1.5 # heuristic bound for max radius
    if max_r < 100: max_r = 100
    
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    bins = np.linspace(-max_r, max_r, grid_size + 1)

    num_angles = 360
    max_radius_per_sector = np.zeros(num_angles) # outer contour
    sum_r_sq_per_sector = np.zeros(num_angles) # for inner contour
    count_sector = np.zeros(num_angles)

    print(f"Generating density grid for N={snapshot_N} across {len(files)} files...")
    for f in tqdm(files):
        coords = load_cluster(f)
        if coords is None or len(coords) < snapshot_N:
            continue

        #truncate and center
        sub_coords  = coords[:snapshot_N]
        sub_coords = sub_coords - sub_coords[0]

        #density map
        H, xedges, yedges = np.histogram2d(
            sub_coords[:,0], 
            sub_coords[:,1], 
            bins=[bins, bins]
            )
        grid += H.astype(np.int32)
        
        # Contours
        r = np.sqrt(np.sum(sub_coords**2, axis=1))
        theta = np.arctan2(sub_coords[:,1], sub_coords[:,0])
        
        sector_indices = ((theta + np.pi) / (2 * np.pi) * num_angles).astype(int) % num_angles # convert angles to [0, 359]
        for i in range(num_angles):
            in_sector = (sector_indices == i)
            if not np.any(in_sector):
                continue
            r_sector = r[in_sector]

            #update outer contour
            current_max = np.max(r_sector)
            if current_max > max_radius_per_sector[i]:
                max_radius_per_sector[i] = current_max
            #update inner contour
            sum_r_sq_per_sector[i] += np.sum(r_sector**2)
            count_sector[i] += len(r_sector)
        
        del sub_coords 
        del coords 
        gc.collect()
        
    with np.errstate(divide='ignore', invalid='ignore'):
        rg_per_sector = np.sqrt(sum_r_sq_per_sector / count_sector)
        rg_per_sector[np.isnan(rg_per_sector)] = 0

    return grid, bins, max_radius_per_sector, rg_per_sector

def get_log_checkpoints(max_n, pts_per_decade=50):
    """
    Returns log-spaced time points.
    Used for extracting Rg history consistently across all files.
    """
    if max_n < 1000: return np.array([max_n])
    
    start_log = 3.0 # Start at 10^3 = 1000
    end_log = np.log10(max_n)
    
    num_points = int(pts_per_decade * (end_log - start_log))
    
    # Unique integer checkpoints
    checkpoints = np.unique(np.logspace(start_log, end_log, num=num_points).astype(int))
    return checkpoints[checkpoints <= max_n]