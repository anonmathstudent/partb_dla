import numpy as np
from scipy.stats import linregress
from scipy.fft import rfft

def calculate_radius_gyration(coords, t, fit_start=1000):
    """
    Calculate the growth rate (beta) from the radius of gyration
    Matches Grebenkov and Beliaev (2017): Fit log(Rg) ~ beta * log(t)
    """
    # 1. calculate Rg trajectory
    r_sq = np.sum(coords**2, axis=1) # squared distance from origin
    cum_r_sq = np.cumsum(r_sq) # cumulative sum
    Rg = np.sqrt(cum_r_sq / t) # radius of gyration

    # 2. fit log-log to get beta
    start_idx = fit_start if len(t) > fit_start + 10 else 10
    if start_idx >= len(t):
        return{'beta': np.nan, 'beta_err': np.nan, 'Rg_final': Rg[-1] if len(Rg)>0 else 0}
    
    log_t = np.log(t[start_idx:])
    log_Rg = np.log(Rg[start_idx:])
    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_Rg)
    return {'beta': slope, 'beta_err': std_err, 'Rg_final': Rg[-1]}

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

    #1D Hashing
    y_multiplier = int(max_box_size * 2 + 10000) #large constant to avoid collisions
    coords_pos = coords + max_box_size

    for eps in scales:
        grid_x = np.floor(coords_pos[:,0] / eps).astype(np.int64)
        grid_y = np.floor(coords_pos[:,1] / eps).astype(np.int64)

        grid_hash = grid_x + (grid_y * y_multiplier)
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
    Calculate radius of gyration per angular sector over time
    Returns: 1D array of angles, 1D array of times, 2D array of Rg values [sector, time]
    """
    if checkpoints is None:
        #default is powers of 2 from 2^10 up until max time
        max_p = int(np.log2(len(t)))
        checkpoints = np.logspace(10, max_p, num=(max_p-10)+1, base=2, dtype=int) #why is this num?
        checkpoints = np.unique(checkpoints[checkpoints < len(t)])

    #1. prepare bins
    bin_edges = np.linspace(-np.pi, np.pi, num_sectors + 1)
    angles = np.arctan2(coords[:,1], coords[:,0])

    #2. sort data spatially
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    coords_sorted = coords[sort_idx]
    t_sorted = t[sort_idx]

    bin_indices = np.searchsorted(angles_sorted, bin_edges)

    rg_history = np.zeros((num_sectors, len(checkpoints)))
    rg_history[:] = np.nan

    # 3. loop through sectors
    for i in range(num_sectors):
        start_idx = bin_indices[i]
        end_idx = bin_indices[i+1]

        t_wedge = t_sorted[start_idx:end_idx]
        coords_wedge = coords_sorted[start_idx:end_idx]

        if len(t_wedge) < 10: continue

        # sort by time
        t_sort_wedge = np.argsort(t_wedge)
        t_wedge = t_wedge[t_sort_wedge]
        coords_wedge = coords_wedge[t_sort_wedge]

        r_sq = np.sum(coords_wedge**2, axis=1)
        cum_r_sq = np.cumsum(r_sq)
        counts = np.arange(1, len(t_wedge) + 1)

        #lookup checkpoints
        indices = np.searchsorted(t_wedge, checkpoints, side='right') - 1
        valid = (indices >= 0) 

        if np.any(valid):
            Rg_values = np.sqrt(cum_r_sq[indices[valid]] / counts[indices[valid]])
            rg_history[i, valid] = Rg_values
    return bin_edges[:-1], checkpoints, rg_history

def downsample_centered(sector_grid, input_sectors=360, target_sectors=90):
    """
    Groups high-res sector data into centered coarse bins.
    Default is it combines four 1 degree bins together to turn 360 sectors into 90. 
    
    """
    if sector_grid.shape[0] != input_sectors:
        return sector_grid # Return as-is if already downsampled
        
    window = input_sectors // target_sectors
    shift = window // 2 
    
    # Roll array to bring the "left" neighbors of 0 to the start
    grid_rolled = np.roll(sector_grid, shift=shift, axis=0)
    
    # Reshape and Average
    # shape becomes (90, 4, Time) -> mean over axis 1 -> (90, Time)
    grid_reshaped = grid_rolled.reshape(target_sectors, window, -1)
    
    # Use nanmean to ignore empty sub-sectors
    grid_coarse = np.nanmean(grid_reshaped, axis=1)
    
    return grid_coarse

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
    
def calculate_anisotropy_old(coords,t,num_bins=360):
    """
    Calculate Anisotropy Score (A4/A0) using Fourier decomp of angular growth rates
    """
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    angles = np.arctan2(coords[:,1], coords[:,0])

    beta_thetas = []

    #loop over wedges
    for i in range(num_bins):
        in_bin = (angles >= bin_edges[i]) & (angles < bin_edges[i+1])
        
        coords_bin = coords[in_bin]
        t_wedge = t[in_bin]
        n_wedge = len(t_wedge)

        if n_wedge < 50:
            beta_thetas.append(np.nan)
            continue

        #local Rg calculation
        coords_wedge = coords[in_bin]
        r_sq_wedge = np.sum(coords_wedge**2, axis=1)
        cumsum_wedge = np.cumsum(r_sq_wedge)
        wedge_count = np.arange(1, n_wedge + 1)
        Rg_wedge = np.sqrt(cumsum_wedge / wedge_count)

        #regression: local size vs global time (early fit start)
        fit_start = 20
        if n_wedge > fit_start + 10:
            log_t_wedge = np.log(t_wedge[fit_start:])
            log_Rg_wedge = np.log(Rg_wedge[fit_start:])
            slope, intercept, r_value, p_value, std_err = linregress(log_t_wedge, log_Rg_wedge)
            beta_thetas.append(slope)
        else:
            beta_thetas.append(np.nan)
        
    #clean NaNs
    beta_arrray = np.array(beta_thetas)
    valid = ~np.isnan(beta_arrray)
    if np.sum(valid) < num_bins / 2:
        return {'A0': np.nan, 'A4': np.nan, 'A4_A0': np.nan}
    
    beta_cleaned = np.interp(
        np.arange(num_bins),
        np.arange(num_bins)[valid],
        beta_arrray[valid]
    )

    #Fourier Decomposition
    #A0 is the average growth rate, A4 is the 4th harmonic amplitude (corresponding to grid anisotropy)
    coeffs = rfft(beta_cleaned)
    magnitudes = np.abs(coeffs) / num_bins

    A0 = magnitudes[0]
    A4 = magnitudes[4]
    score = A4 / A0 if A0 != 0 else np.nan
    return {'A0': A0, 'A4': A4, 'A4_A0': score}