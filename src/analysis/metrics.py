import numpy as np
from scipy.stats import linregress
from scipy.fft import rfft

from scipy.stats import linregress
import numpy as np

def calculate_radius_gyration(coords, t, fit_start=1000, pts_per_decade=50):
    """
    Calculate the growth rate (beta) from the radius of gyration.
    Matches Grebenkov and Beliaev (2017): Fit log(Rg) ~ beta * log(t).
    """
    N_total = len(t)

    # 1. Calculate full Rg trajectory
    r_sq = np.sum(coords**2, axis=1) # squared distance from origin
    cum_r_sq = np.cumsum(r_sq)       # cumulative sum
    Rg = np.sqrt(cum_r_sq / t)       # radius of gyration
    
    Rg_final = Rg[-1] if len(Rg) > 0 else 0

    # Handle tiny clusters
    start_idx = fit_start if N_total > fit_start + 10 else 10
    if start_idx >= N_total:
        return {'beta': np.nan, 'beta_err': np.nan, 'Rg_final': Rg_final}
    
    # --- 2. Generate Log-Spaced Checkpoints ---
    start_log = np.log10(start_idx)
    end_log = np.log10(N_total)
    
    # Determine number of points based on the decades
    num_points = int(pts_per_decade * (end_log - start_log))
    
    if num_points < 2: # Failsafe
         return {'beta': np.nan, 'beta_err': np.nan, 'Rg_final': Rg_final}
         
    checkpoints = np.unique(np.logspace(start_log, end_log, num=num_points).astype(int))
    checkpoints = checkpoints[checkpoints <= N_total]

    # --- 3. Extract Sampled Data ---
    # Since 't' usually goes [1, 2, ..., N], checkpoint 1000 is at index 999.
    # We subtract 1 to get the correct 0-based index from the Rg array.
    sampled_indices = checkpoints - 1
    
    t_sampled = t[sampled_indices]
    Rg_sampled = Rg[sampled_indices]
    assert np.array_equal(t_sampled, checkpoints), "Time alignment failed!"

    # --- 4. Fit Log-Log to get Beta ---
    log_t = np.log(t_sampled)
    log_Rg = np.log(Rg_sampled)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_Rg)
    
    return {'beta': slope, 'beta_err': std_err, 'Rg_final': Rg_final}

def calculate_box_dim(coords, max_box_size = None):
    """
    Calculate box-counting dimension (D) using 1D Hashing
    Scales: 2^4 to max_size/16 (skips atoms and finite effects)
    """
    if max_box_size is None:
        max_box_size = int(np.max(np.abs(coords))) + 1

    #define scales
    upper_limit = max_box_size / 16
    if upper_limit < 4:
        return {'D': np.nan, 'D_err': np.nan}
    
    scales = np.logspace(4, np.log2(upper_limit), num=30, base=2, dtype=int)
    scales = np.unique(scales)

    counts = []
    coords_pos = coords + max_box_size # Shift coordinates to be strictly positive
    max_possible_coord = np.max(coords_pos) # The absolute maximum coordinate value in our new positive space

    for eps in scales:
        grid_x = np.floor(coords_pos[:,0] / eps).astype(np.int64)
        grid_y = np.floor(coords_pos[:,1] / eps).astype(np.int64)
        safe_multiplier = int((max_possible_coord / eps) + 1) # Calculate a mathematically perfect multiplier for this specific eps

        # 1D Fast Hashing
        grid_hash = grid_x + (grid_y * safe_multiplier)
        unique_count = len(np.unique(grid_hash))
        counts.append(unique_count)
    
    # fit N(eps) ~ eps^{-D}
    if len(counts) > 2:
        slope, intercept, r_value, p_value, std_err = linregress(np.log(scales), np.log(counts))
        D = -slope
    else:
        D, std_err = np.nan, np.nan
    return {'D': D, 'D_err': std_err}

def calculate_sector_evolution(coords, t, num_sectors=360, checkpoints=None):
    """
    Calculate Radius of Gyration (RMS distance from Seed) per angular sector.
    """
    N_total = len(t)

    # --- 1. Log-Spaced Checkpoints ---
    if checkpoints is None:
        start_log = 3.0  # Start at particle 1000 to avoid small-number noise
        end_log = np.log10(N_total)
        num_points = int(20 * (end_log - start_log)) # 20 pts/decade
        checkpoints = np.unique(np.logspace(start_log, end_log, num=num_points).astype(int))
        checkpoints = checkpoints[checkpoints < N_total]

    # --- 2. Calculate Polar Coordinates ---
    # r_sq is the squared distance from the Seed (0,0)
    r_sq = np.sum(coords**2, axis=1)
    
    # Calculate angles [-pi, pi]
    angles = np.arctan2(coords[:, 1], coords[:, 0])

    # --- 3. Center Bins on Axes (Shift-Bin Strategy) ---
    # We want Bin 0 to be centered exactly on 0 degrees (the Lattice Axis).
    # Bin 0 limits: [-0.5 deg, +0.5 deg]
    # To use fast binning, we shift all angles by +0.5 deg (half a bin).
    
    d_theta = 2 * np.pi / num_sectors
    
    # Modulo 2pi ensures everything is in range [0, 2pi]
    angles_shifted = (angles + (d_theta / 2)) % (2 * np.pi)
    
    # Define edges 0..2pi. 
    # Because we shifted the data, 'digitize' will put particles 
    # centered at 0 deg into Bin 0.
    bin_edges = np.linspace(0, 2 * np.pi, num_sectors + 1)
    
    # Get sector indices (0 to num_sectors-1)
    sector_indices = np.digitize(angles_shifted, bin_edges) - 1
    
    # Handle rare edge case where a value equals exactly 2pi
    sector_indices[sector_indices == num_sectors] = 0

    # --- 4. Sort by Time ---
    # Critical for cumulative calculation.
    # We sort once globally to avoid sorting inside the loop.
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    r_sq_sorted = r_sq[sort_idx]
    sector_indices_sorted = sector_indices[sort_idx]

    # Initialize Output: [Sectors x Time_Checkpoints]
    rg_history = np.zeros((num_sectors, len(checkpoints)))
    rg_history[:] = np.nan 

    # --- 5. Process Each Sector ---
    for s in range(num_sectors):
        # Extract particles for this sector
        in_sector = (sector_indices_sorted == s)
        
        # If sector is empty, leave as NaNs and continue
        if not np.any(in_sector):
            continue

        t_sec = t_sorted[in_sector]
        r_sq_sec = r_sq_sorted[in_sector]
        
        # Calculate Cumulative RMS Distance (Rg)
        # Rg(t) = sqrt( Sum(r^2) / Count(t) )
        cum_r_sq = np.cumsum(r_sq_sec)
        counts = np.arange(1, len(t_sec) + 1)
        rg_evolution = np.sqrt(cum_r_sq / counts)
        
        # --- 6. Map to Checkpoints (The "Freezing" Logic) ---
        # Find the index of the particle that exists at or before each checkpoint.
        # side='right' ensures we grab the latest possible value.
        idx_map = np.searchsorted(t_sec, checkpoints, side='right') - 1
        
        # If idx_map is -1, the checkpoint is before the first particle arrived (keep NaN).
        # If idx_map is max_index, the checkpoint is after the last particle (keep max value).
        valid_mask = (idx_map >= 0)
        
        if np.any(valid_mask):
            valid_indices = idx_map[valid_mask]
            rg_history[s, valid_mask] = rg_evolution[valid_indices]

    # --- 7. Prepare Return Values ---
    bin_centers = np.linspace(0, 2*np.pi, num_sectors, endpoint=False)
    
    return bin_centers, checkpoints, rg_history

def downsample_centered(grid, input_sectors=360, target_sectors=90):
    """
    Correctly aggregates 1-degree bins into 4-degree bins while 
    maintaining the center-alignment on the lattice axes.
    """
    if grid.shape[0] != input_sectors:
        return grid
        
    factor = input_sectors // target_sectors # factor = 4
    shift = factor // 2                      # shift = 2
    
    # 1. Roll the array so that the axis (0°) is in the middle of a group
    grid_rolled = np.roll(grid, shift=shift, axis=0)
    
    # 2. Reshape to (90, 4, Time) and take the mean of those 4 bins
    coarse_grid = grid_rolled.reshape(target_sectors, factor, -1).mean(axis=1)
    
    return coarse_grid

def calculate_beta_profile(time_points, sector_grid, fit_start_N=1000, min_points=3):
    """
    Computes the growth exponent (beta) for every sector.
    """
    num_sectors = sector_grid.shape[0]
    beta_profile = []
    
    # Pre-calculate log time for the valid window to speed up loop
    # (We still need to filter for r > 0 per sector, but this helps)
    time_mask = (time_points >= fit_start_N)
    
    for s in range(num_sectors):
        r_curve = sector_grid[s, :]
        
        # Combine time filter with data validity filter
        valid = time_mask & (r_curve > 0)

        if np.sum(valid) >= min_points:
            # Fit power law: log(r) = beta * log(t) + C
            res = linregress(np.log(time_points[valid]), np.log(r_curve[valid]))
            beta_profile.append(res.slope)
        else:
            beta_profile.append(np.nan)
            
    return np.array(beta_profile)


def anisotropy_fourier(time_points, sector_grid, fit_start_N=1000, input_sectors=360, target_sectors=90):
    """
    Calculate Anisotropy Score (A4/A0) using Fourier decomp of angular growth rates
    """
    if sector_grid is None or sector_grid.size == 0 or sector_grid.shape[1] == 0:
        return np.nan, np.nan, np.nan
    sector_grid = downsample_centered(sector_grid, input_sectors=input_sectors, target_sectors=target_sectors)
    # calc angular growth rates
    beta_profile = calculate_beta_profile(time_points, sector_grid, fit_start_N)

    num_sectors = len(beta_profile)
    nans = np.isnan(beta_profile)
    
    # Safety check: if too many sectors failed to fit, return NaNs
    if np.sum(nans) > (num_sectors // 2):
        return np.nan, np.nan, np.nan
    
    # Interpolate over NaNs
    x = np.arange(num_sectors)
    beta_filled = np.interp(x, x[~nans], beta_profile[~nans])

    # Decompose
    coeffs = rfft(beta_filled)
    magnitudes = np.abs(coeffs) / num_sectors
    

    A0, A4 = magnitudes[0], magnitudes[4]
    
    # Avoid division by zero
    score = A4 / A0 if A0 != 0 else np.nan
    return A0, A4, score


def anisotropy_ratio(time_points, sector_grid, fit_start_N=1000, input_sectors=360, target_sectors=90):
    """
    Calculates ratio of axis growth rate vs diagonal growth rate.
    """
    if sector_grid is None or sector_grid.size == 0 or sector_grid.shape[1] == 0:
        return np.nan, np.nan, np.nan
    sector_grid = downsample_centered(sector_grid, input_sectors=input_sectors, target_sectors=target_sectors)

    num_sectors = sector_grid.shape[0]
    angles_deg = np.linspace(0, 360, num_sectors, endpoint=False)
    
    # define masks (+/- 5 degrees)
    axis_mask = np.zeros(num_sectors, dtype=bool)
    diag_mask = np.zeros(num_sectors, dtype=bool)
    
    bin_width = 360.0 / num_sectors
    # Bins are aligned (4 deg); use small tolerance to select axis/diagonal bins only
    tolerance = max(5.0, 1.01 * bin_width)

    for t in [0, 90, 180, 270]:
        diff = np.abs(angles_deg - t)
        axis_mask |= (np.minimum(diff, 360 - diff) < tolerance)
    for t in [45, 135, 225, 315]:
        diff = np.abs(angles_deg - t)
        diag_mask |= (np.minimum(diff, 360 - diff) < tolerance)

    # average Rg curves for Axes and Diagonals
    r_axis = np.nanmean(sector_grid[axis_mask, :], axis=0)
    r_diag = np.nanmean(sector_grid[diag_mask, :], axis=0)
    
    # calculate beta_axis and beta_diag 
    try:
        valid = (time_points >= fit_start_N) & (r_axis > 0) & (r_diag > 0)
        # Need at least 3 points for a line
        if np.sum(valid) < 3: return np.nan, np.nan, np.nan

        log_t = np.log(time_points[valid])
        
        # axes
        res_a = linregress(log_t, np.log(r_axis[valid]))
        # diagonals
        res_d = linregress(log_t, np.log(r_diag[valid]))
        
        beta_axis = res_a.slope
        beta_diag = res_d.slope
        
        # The Metric: Ratio of the growth rates (guard against division by zero)
        ratio = beta_axis / beta_diag if beta_diag > 1e-10 else np.nan
        
        return beta_axis, beta_diag, ratio
        
    except Exception:
        return np.nan, np.nan, np.nan
   