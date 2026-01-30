import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def plot_density_map(grid, bins, max_r_curve=None, rg_curve=None, title="Cluster Density"):
    """
    Plots the accumulated 2D histogram using Log normalisation.
    Optionally overlays the 'Outer' (Max R) and 'Inner' (Rg) contours.
    
    Parameters:
    - grid: 2D numpy array of density counts
    - bins: 1D numpy array of spatial bin edges
    - max_r_curve: (Optional) 1D array of length 360, max radius per degree
    - rg_curve: (Optional) 1D array of length 360, radius of gyration per degree
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # 1. Plot the Heatmap
    # Use extent to map array indices back to physical coordinates
    extent = [bins[0], bins[-1], bins[0], bins[-1]]
    
    # LogNorm is crucial for DLA: vmin=1 ensures empty background is distinct
    im = ax.imshow(grid.T, origin='lower', extent=extent, 
                   norm=LogNorm(vmin=1, vmax=np.max(grid)),
                   cmap='magma')
    
    cbar = plt.colorbar(im, ax=ax, label='Particle Density (Counts)')
    
    # 2. Overlay Contours (if provided)
    if max_r_curve is not None or rg_curve is not None:
        # Reconstruct angles (0 to 360 degrees mapped to -pi to pi)
        # In processing.py, index 0 corresponds to -pi
        num_angles = 360
        thetas = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)
        
        # Helper to convert polar -> cartesian for plotting
        def polar_to_cart(r_values):
            # We roll the array because linspace starts at -pi, 
            # but sometimes data alignment needs checking. 
            # Based on processing logic: index 0 is -pi. Logic holds.
            x = r_values * np.cos(thetas)
            y = r_values * np.sin(thetas)
            # Close the loop for the plot
            return np.append(x, x[0]), np.append(y, y[0])

        if max_r_curve is not None:
            mx, my = polar_to_cart(max_r_curve)
            ax.plot(mx, my, color='cyan', linestyle='--', linewidth=1, alpha=0.8, label='Outer Boundary ($R_{max}$)')
            
        if rg_curve is not None:
            gx, gy = polar_to_cart(rg_curve)
            ax.plot(gx, gy, color='white', linestyle='-', linewidth=1.5, label='Avg Mass ($R_g$)')
            
        ax.legend(loc='upper right', frameon=True, facecolor='black', framealpha=0.6, labelcolor='white')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Remove gridlines as they clutter the density map
    ax.grid(False)
    
    return fig

def angular_rg_plot(time_points, agg_data, ax_polar=None, ax_linear=None, ax_log=None):
    """
    plots the angular radius of gyration vs angle for different time points.
    Parameters:
    - time_points: array of N values
    - agg_data: 3D array of shape (num_files, num_sectors, time_points)
    - ax_linear: optional axis for linear scale plot
    - ax_polar: optional axis for polar plot
    - ax_log: optional axis for semi-log scale plot
    """
    mean_profile = np.nanmean(agg_data, axis=0)  # shape: (num_sectors, time_points)
    num_sectors = mean_profile.shape[0]
    angles_deg = np.linspace(0, 360, num_sectors, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)

    step = max(1, len(time_points) // 8)  # plot up to 8 time points
    indices = np.arange(0, len(time_points), step)
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    for idx, col in zip(indices, colors):
        t_val = time_points[idx]
        r_vals = mean_profile[:, idx]

        label = f'$N=2^{{{int(np.log2(t_val))}}}$'

        if ax_polar is not None:
            theta_plot = np.append(angles_rad, angles_rad[0])
            r_plot = np.append(r_vals, r_vals[0])
            ax_polar.plot(theta_plot, r_plot, color=col, label=label)

        if ax_linear is not None:
            ax_linear.plot(angles_deg, r_vals, color=col, label=label)

        if ax_log is not None:
            ax_log.semilogy(angles_deg, r_vals, color=col, label=label)

    if ax_polar is not None:
        ax_polar.set_title('Polar Scale', pad=20)
        ax_polar.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize = 'x-small', title='Cluster Size')
    if ax_linear is not None:
        ax_linear.set_title('Linear Scale')
        ax_linear.set_xlabel('Angle (degrees)')
        ax_linear.set_ylabel('Radius of Gyration $R_g$')
        ax_linear.set_xlim(0, 360)
        ax_linear.grid(True, alpha=0.3)
    if ax_log is not None:
        ax_log.set_title('Semi-Log Scale')
        ax_log.set_xlabel('Angle (degrees)')
        ax_log.set_ylabel('Radius of Gyration $R_g$')
        ax_log.set_xlim(0, 360)
        ax_log.grid(True, which="both", alpha=0.3)

def comparing_axes_plot(time_points, agg_data, ax=None):
    return
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