import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import linregress
import src.analysis.metrics as metrics 
from scipy.signal import savgol_filter


def plot_density_map(grid, bins, max_r_curve=None, rg_curve=None, title="Cluster Density", ax=None):
    """
    Plots density map with ALL details (Legend, Labels, Colorbar) always enabled.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))
    else:
        fig = ax.figure
    
    # 1. Plot Heatmap
    extent = [bins[0], bins[-1], bins[0], bins[-1]]
    im = ax.imshow(grid.T, origin='lower', extent=extent, 
                   norm=LogNorm(vmin=1, vmax=np.max(grid)),
                   cmap='magma')
    
    # 2. Add Colorbar
    # fraction=0.046 and pad=0.04 align the colorbar height to the plot height
    plt.colorbar(im, ax=ax, label='Density', fraction=0.046, pad=0.04)
    
    # 3. Overlay Contours
    if max_r_curve is not None or rg_curve is not None:
        num_angles = 360
        thetas = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)
        
        def polar_to_cart(r_values):
            x = r_values * np.cos(thetas)
            y = r_values * np.sin(thetas)
            return np.append(x, x[0]), np.append(y, y[0])

        if max_r_curve is not None:
            mx, my = polar_to_cart(max_r_curve)
            ax.plot(mx, my, color='cyan', linestyle='--', linewidth=1, alpha=0.8, label='Outer Boundary')
            
        if rg_curve is not None:
            gx, gy = polar_to_cart(rg_curve)
            ax.plot(gx, gy, color='white', linestyle='-', linewidth=1.5, label='Avg Mass ($R_g$)')
            
        #ax.legend(loc='upper right', frameon=True, facecolor='black', framealpha=0.6, labelcolor='white', fontsize='small')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    
    return fig

def angular_rg_plot(time_points, agg_data, ax_polar=None, ax_linear=None, ax_log=None):
    """
    Plots the angular radius of gyration vs angle for specific time points.
    Selects time points closest to powers of 2 (2^10, 2^11...) for plotting.
    """
    mean_profile = np.nanmean(agg_data, axis=0)  # shape: (num_sectors, time_points)
    num_sectors = mean_profile.shape[0]
    angles_deg = np.linspace(0, 360, num_sectors, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)

    # We want to plot N = 2^10, 2^11, ... up to the max size
    min_power = 10
    max_power = int(np.log2(time_points[-1]))
    
    indices = []
    # Loop through powers 10, 11, 12...
    for p in range(min_power, max_power + 1):
        target_N = 2**p
        # Find the index in time_points closest to this target
        idx = (np.abs(time_points - target_N)).argmin()
        indices.append(idx)
    
    # Remove duplicates (in case resolution is low) and sort
    indices = np.unique(indices)

    # Use 'viridis' color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    for idx, col in zip(indices, colors):
        t_val = time_points[idx]
        r_vals = mean_profile[:, idx]

       
        power_label = int(round(np.log2(t_val)))
        label = f'$N=2^{{{power_label}}}$'

        if ax_polar is not None:
            theta_plot = np.append(angles_rad, angles_rad[0])
            r_plot = np.append(r_vals, r_vals[0])
            ax_polar.plot(theta_plot, r_plot, color=col, label=label, linewidth=1.2)

        if ax_linear is not None:
            ax_linear.plot(angles_deg, r_vals, color=col, label=label)

        if ax_log is not None:
            ax_log.semilogy(angles_deg, r_vals, color=col, label=label)

    # Define Ticks
    ticks_45 = np.arange(0, 361, 45)

    if ax_polar is not None:
        ax_polar.set_title('Polar Scale', pad=20)
        ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='x-small', title='Cluster Size')

    if ax_linear is not None:
        ax_linear.set_title('Linear Scale')
        ax_linear.set_xlabel('Angle (degrees)')
        ax_linear.set_ylabel('Radius of Gyration $R_g$')
        ax_linear.set_xlim(0, 360)
        ax_linear.set_xticks(ticks_45)
        ax_linear.grid(True, alpha=0.3)

    if ax_log is not None:
        ax_log.set_title('Semi-Log Scale')
        ax_log.set_xlabel('Angle (degrees)')
        ax_log.set_ylabel('Radius of Gyration $R_g$')
        ax_log.set_xlim(0, 360)
        ax_log.set_xticks(ticks_45)
        ax_log.grid(True, which="both", alpha=0.3)

def comparing_axes_plot(time_points, agg_data, summary_stats=None, ax=None):
    """
    Plot aggregated Rg(t) for axes vs diagonals on log-log scale.
    Requires summary_stats to display the beta/fourier values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    num_sectors = agg_data.shape[1]
    angles_deg = np.linspace(0, 360, num_sectors, endpoint=False)

    axis_mask, diag_mask = np.zeros(num_sectors, dtype=bool), np.zeros(num_sectors, dtype=bool)
    tolerance = 5.0 

    targets_axis = [0, 90, 180, 270]
    targets_diag = [45, 135, 225, 315]

    for t in targets_axis:
        diff = np.abs(angles_deg - t)
        diff = np.minimum(diff, 360 - diff)
        axis_mask |= (diff <= tolerance) 

    for t in targets_diag:
        diag_mask |= (np.abs(angles_deg - t) <= tolerance)

    # Aggregation for visualisation (Mean of all files)
    mean_grid = np.nanmean(agg_data, axis=0)
    r_axis = np.nanmean(mean_grid[axis_mask, :], axis=0)
    r_diag = np.nanmean(mean_grid[diag_mask, :], axis=0)

    # Plotting
    ax.loglog(time_points, r_axis, 'o', label='Axes (0°, 90°...)', markersize=5, color='blue', linewidth=1.5)
    ax.loglog(time_points, r_diag, 's', label='Diagonals (45°, 135°...)', markersize=5, color='red', linewidth=1.5)

    fit_start_N = 1000
    valid_idx = (time_points >= fit_start_N) & (r_axis > 0) & (r_diag > 0)

    if np.sum(valid_idx) > 2:
        x_fit = np.log(time_points[valid_idx])
        
        # Fit & Plot Axis Line
        res_axis = linregress(x_fit, np.log(r_axis[valid_idx]))
        y_pred_axis = np.exp(res_axis.intercept + res_axis.slope * x_fit)
        ax.plot(time_points[valid_idx], y_pred_axis, '-', color='darkblue', linewidth=2, label=f'Fit Axes ($\\beta={res_axis.slope:.3f}$)')

        # Fit & Plot Diagonal Line
        res_diag = linregress(x_fit, np.log(r_diag[valid_idx]))
        y_pred_diag = np.exp(res_diag.intercept + res_diag.slope * x_fit)
        ax.plot(time_points[valid_idx], y_pred_diag, '-', color='darkred', linewidth=2, label=f'Fit Diags ($\\beta={res_diag.slope:.3f}$)')

    # display stats
    if summary_stats is not None:
        ratio_mu = summary_stats['beta_ratio_mean']
        ratio_std = summary_stats['beta_ratio_std']
        four_mu  = summary_stats['fourier_score_mean']
        four_std = summary_stats['fourier_score_std']
        
        b_ax_mean = summary_stats.get('beta_axis_mean', res_axis.slope)
        b_dg_mean = summary_stats.get('beta_diag_mean', res_diag.slope)

        text_str = (
            f"Growth Rates:\n"
            f"$\\beta_{{axis}} = {b_ax_mean:.3f}$\n"
            f"$\\beta_{{diag}} = {b_dg_mean:.3f}$\n\n"
            )
        
        ax.text(0.05, 0.9, text_str, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'))
        
    ax.set_xlabel("Cluster Size ($N$)")
    ax.set_ylabel("Aggregated $R_g$")
    ax.set_title("Fig 5: Anisotropic Growth Scaling")
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    
    return ax

def plot_anisotropy_evolution(df_precision, df_extension=None, ax=None):
    """
    Plots Anisotropy Score (A4/A0) vs Cluster Size (N) with error bars.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Helper to calculate stats
    def get_stats(df):
        return df.groupby('snapshot_N')['anisotropy_score'].agg(['mean', 'std']).reset_index()

    # Plot Precision Batch
    stats_p = get_stats(df_precision)
    ax.errorbar(stats_p['snapshot_N'], stats_p['mean'], yerr=stats_p['std'], 
                fmt='-o', markersize=4, capsize=3, label='Precision (N=1000)')
    
    # Plot Extension Batch
    if df_extension is not None:
        stats_e = get_stats(df_extension)
        ax.errorbar(stats_e['snapshot_N'], stats_e['mean'], yerr=stats_e['std'], 
                    fmt='--s', markersize=5, capsize=3, label='Extension (N=100)')
        
    ax.set_xscale('log')
    ax.set_xlabel('Cluster Size ($N$)')
    ax.set_ylabel('Anisotropy Score ($A_4/A_0$)')
    ax.set_title('Evolution of Anisotropy')
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    return ax


def plot_beta_profile(time_points, agg_data, fit_start_N=1000, ax=None, input_sectors=360, target_sectors=90, trend_line = False):
    """
    Plots the Angular Growth Rate (Beta) for each sector.
    Downsamples to target_sectors (default 90 = 4 deg) so 0 deg is aligned with the lattice axis.
    """
    if ax is None: fig, ax = plt.subplots()

    mean_r_matrix = np.nanmean(agg_data, axis=0)  # average across all clusters (sectors x time)
    # Downsample to coarse bins so x-axis 0 deg represents the true peak (lattice axis)
    mean_coarse = metrics.downsample_centered(mean_r_matrix, input_sectors=input_sectors, target_sectors=target_sectors)
    beta_profile = metrics.calculate_beta_profile(time_points, mean_coarse, fit_start_N)

    num_sectors = len(beta_profile)
    angles = np.linspace(0, 360, num_sectors, endpoint=False)

    ax.plot(angles, beta_profile, color='red', linewidth=1)

    if trend_line:
        window = max(5, num_sectors // 20)
        if window % 2 == 0: window += 1
        try:
            beta_smooth = savgol_filter(beta_profile, window, 2)
            ax.plot(angles, beta_smooth, color='purple', alpha=0.3, linewidth=2, label=r'$\beta(\theta)$ Trend')
        except Exception:
            pass

    # Mean line
    mean_beta = np.nanmean(beta_profile)
    ax.axhline(mean_beta, color='green', linestyle='--', label=f'Mean: {mean_beta:.3f}')

    # Styling (raw string so \beta is not interpreted as backspace)
    ax.set_title(rf"Angular Growth Rate ({num_sectors} bins): $\beta(\theta)$")
    ax.set_xlabel("Angle (Deg)")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)
    
    return ax