"""
Coarse-Grained Density Plotter for DLA Clusters.
"""
import argparse
import sys
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dla_sim import utils

@njit(cache=True)
def compute_density_grid(x_coords, y_coords, grid, scale, pad_x, pad_y):
    """
    Numba-accelerated 2D Histogram.
    Iterates through particles and increments the corresponding pixel counter.
    """
    H, W = grid.shape
    num_particles = len(x_coords)
    
    for i in range(num_particles):
        px = int((x_coords[i] - pad_x) * scale)
        py = int((y_coords[i] - pad_y) * scale)
        
        if 0 <= px < W and 0 <= py < H:
            grid[py, px] += 1


def render_density(
    positions, 
    title=None,
    output_path=None, 
    res_power=11, 
    mode="log", 
    cmap="Greys",
    ax=None
):
    """
    Renders coarse-grained density map. 
    Can plot to a standalone figure, or onto an existing Matplotlib Axis.
    """
    # 1. Setup Resolution
    res = 1 << res_power  # e.g., 2^11 = 2048
    print(f"Target Resolution: {res}x{res} (2^{res_power})")
    
    # 2. Get Coordinates
    if hasattr(positions, 'meta') and "x_coords" in positions.meta:
        x = np.array(positions.meta["x_coords"])
        y = np.array(positions.meta["y_coords"])
    elif hasattr(positions, 'positions') and positions.positions is not None:
        x = positions.positions[:, 0]
        y = positions.positions[:, 1]
    elif isinstance(positions, tuple) or isinstance(positions, list) or isinstance(positions, np.ndarray):
        # Allow passing raw coordinate arrays directly from Notebook dictionaries!
        coords = np.asarray(positions)
        if len(coords.shape) > 1 and coords.shape[1] == 2:
            x = coords[:, 0]
            y = coords[:, 1]
        else:
            x, y = coords[0], coords[1]
    else:
        raise ValueError("No coordinates found.")
        
    # 3. Calculate Scale & Grain
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    width_data = x_max - x_min
    height_data = y_max - y_min
    max_dim = max(width_data, height_data)
    
    # Add 2% padding so it doesn't touch the edge
    max_dim *= 1.02
    
    scale = res / max_dim
    grain_size = 1.0 / scale
    
    # 4. Center Data
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    pad_x = center_x - (max_dim / 2)
    pad_y = center_y - (max_dim / 2)
    
    # 5. Binning
    grid = np.zeros((res, res), dtype=np.int32)
    compute_density_grid(x, y, grid, scale, pad_x, pad_y)
    
    max_count = np.max(grid)

    # 6. Plotting Setup - AXIS INJECTION LOGIC
    if ax is None:
        fig, current_ax = plt.subplots(figsize=(10, 10))
        standalone = True
    else:
        current_ax = ax
        standalone = False
        plt.sca(current_ax) # Ensure Matplotlib targets this axis
    
    # Convert to float for plotting
    grid_float = grid.astype(np.float32)
    
    # MASKING: Make empty pixels transparent (or white)
    grid_masked = np.ma.masked_where(grid == 0, grid_float)
    
    # 7. Apply Modes
    if mode == "binary":
        norm = mcolors.Normalize(vmin=0, vmax=0.1) 
        title_mode = "Binary"
    elif mode == "log":
        norm = mcolors.LogNorm(vmin=1, vmax=max_count)
        title_mode = "Log Density"
    elif mode == "linear":
        norm = mcolors.Normalize(vmin=0, vmax=max_count)
        title_mode = "Linear Density"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Plot Image onto the correct axis
    im = current_ax.imshow(
        grid_masked, 
        cmap=cmap, 
        norm=norm, 
        interpolation='nearest', 
        origin='lower'
    )
    
    # Add Colorbar (attached explicitly to current_ax so it doesn't break grids)
    if mode != "binary":
        plt.colorbar(im, ax=current_ax, label="Particles per Pixel", fraction=0.046, pad=0.04)
    
    current_ax.axis('off')
    
    # Apply Title if provided
    if title:
        current_ax.set_title(title, pad=10)
    
    if standalone and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close(fig)
        
    return im


def main():
    parser = argparse.ArgumentParser(description="Coarse-Grained Density Plotter")
    parser.add_argument("file", help="Input .npz file")
    parser.add_argument("--res-power", type=int, default=11, help="Resolution power of 2 (default 11 -> 2048px)")
    parser.add_argument("--mode", choices=["binary", "log", "linear"], default="log", help="Plotting mode")
    parser.add_argument("--cmap", default="Greys", help="Matplotlib colormap (default: Greys)")
    parser.add_argument("--out", default=None, help="Output filename")

    args = parser.parse_args()
    
    result = utils.load_cluster(args.file)
    
    if args.out is None:
        input_path = Path(args.file)
        out_name = input_path.stem + f"_density_{args.mode}.png"
        out_path = input_path.parent / out_name
    else:
        out_path = args.out
        
    render_density(positions=result, output_path=out_path, res_power=args.res_power, mode=args.mode, cmap=args.cmap)

if __name__ == "__main__":
    main()