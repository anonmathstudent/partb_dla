import argparse
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dla_sim import utils  # type: ignore[import]


@njit(parallel=True, cache=True)
def render_grid_numba(x_coords, y_coords, ages, grid, scale, pad_x, pad_y, radius=0.5):
    """
    Numba-accelerated rasterizer that paints circles onto a grid.
    """
    H, W = grid.shape
    num_particles = len(x_coords)
    r_px = radius * scale
    
    # Optimization: if radius is less than 0.5 pixels, just set single pixel
    if r_px < 0.75:
        for i in prange(num_particles):
            px = int((x_coords[i] - pad_x) * scale)
            py = int((y_coords[i] - pad_y) * scale)
            
            if 0 <= px < W and 0 <= py < H:
                # Only overwrite if new age is greater (or if current is NaN)
                current = grid[py, px]
                if np.isnan(current) or ages[i] > current:
                    grid[py, px] = ages[i]
    else:
        # Full circle rendering: iterate bounding box
        r_px_sq = r_px * r_px
        r_int = int(np.ceil(r_px)) + 1  # Integer radius for bounding box
        
        for i in prange(num_particles):
            px_center = (x_coords[i] - pad_x) * scale
            py_center = (y_coords[i] - pad_y) * scale
            
            px_min = int(np.floor(px_center - r_int))
            px_max = int(np.ceil(px_center + r_int)) + 1
            py_min = int(np.floor(py_center - r_int))
            py_max = int(np.ceil(py_center + r_int)) + 1
            
            # Clamp to grid bounds
            px_min = max(0, px_min)
            px_max = min(W, px_max)
            py_min = max(0, py_min)
            py_max = min(H, py_max)
            
            # Fill pixels within circle
            for py in range(py_min, py_max):
                for px in range(px_min, px_max):
                    dx = px - px_center
                    dy = py - py_center
                    dist_sq = dx * dx + dy * dy
                    
                    if dist_sq <= r_px_sq:
                        # Only overwrite if new age is greater (or if current is NaN)
                        current = grid[py, px]
                        if np.isnan(current) or ages[i] > current:
                            grid[py, px] = ages[i]


def format_title(meta, num_particles=None):
    """
    Format a title string with important statistics from metadata.
    """
    if not meta:
        return None
    
    model = meta.get("model", "?")
    if num_particles is not None:
        num = num_particles
    else:
        num = meta.get("num", "?")
    seed = meta.get("seed")
    seed_str = str(seed) if seed is not None else "?"
    
    parts = [f"Model={model}", f"N={num}", f"seed={seed_str}"]
    
    if model == "continuous":
        pr = meta.get("particle_radius")
        if pr is not None:
            parts.append(f"PR={pr:.2f}")
    elif model == "koh":
        lmax = meta.get("lmax")
        if lmax is not None:
            parts.append(f"L={lmax}")
    elif model == "lattice":
        radius = meta.get("radius")
        if radius is not None:
            parts.append(f"R={radius}")
    elif model == "offlattice":
        rb = meta.get("Rb")
        rd = meta.get("Rd")
        if rb is not None:
            parts.append(f"Rb={rb:.1f}")
        if rd is not None:
            parts.append(f"Rd={rd:.1f}")
    
    return " | ".join(parts)


def get_coordinates_from_result(result):
    """
    Extract x and y coordinates from a ClusterResult, handling all formats.
    """
    meta = result.meta or {}
    
    if "x_coords" in meta and "y_coords" in meta:
        x = np.asarray(meta["x_coords"], dtype=np.float64)
        y = np.asarray(meta["y_coords"], dtype=np.float64)
        return x, y
    
    if result.positions is not None:
        pos = np.asarray(result.positions)
        
        if np.iscomplexobj(pos):
            x = pos.real.astype(np.float64)
            y = pos.imag.astype(np.float64)
            return x, y
            
        if pos.ndim == 2 and pos.shape[1] >= 2:
            x = pos[:, 0].astype(np.float64)
            y = pos[:, 1].astype(np.float64)
            return x, y
    
    raise ValueError(
        "Could not find valid coordinates. Checked: meta['x_coords'], positions(complex), positions(N,2)."
    )


def render(positions, title=None, output=None, cmap="magma", dpi=300, res=2048, auto_res=False, ax=None):
    """
    Unified high-performance rasterizer for DLA clusters.
    Can plot to a standalone figure, or onto an existing Matplotlib Axis.
    """
    if hasattr(positions, 'meta') or hasattr(positions, 'positions'):
        result = positions
        x, y = get_coordinates_from_result(result)
    else:
        x, y = positions
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    num_particles = len(x)
    if num_particles == 0:
        print("No particles to render")
        return None
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    num_particles = len(x)
    
    if num_particles == 0:
        print("No valid particles to render")
        return None
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    W_data = x_max - x_min
    H_data = y_max - y_min
    
    max_dim = max(W_data, H_data)
    
    if max_dim == 0.0:
        max_dim = 1.0 
    
    if auto_res:
        res = int(max_dim * 2.0)
        res = max(512, min(res, 16384))
    else:
        res = int(res)
        res = max(256, min(res, 16384))
    
    padding_factor = 1.05
    scale = res / (max_dim * padding_factor)
    
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    pad_x = center_x - (max_dim * padding_factor) / 2.0
    pad_y = center_y - (max_dim * padding_factor) / 2.0
    
    H = res
    W = res
    grid = np.full((H, W), np.nan, dtype=np.float32)
    
    ages = np.linspace(0.0, 1.0, num_particles, dtype=np.float32)
    
    # Run Numba kernel
    render_grid_numba(x, y, ages, grid, scale, pad_x, pad_y, radius=0.5)
    
    # --- Plotting Logic Update ---
    if ax is None:
        fig, current_ax = plt.subplots(figsize=(6, 6))
        standalone = True
    else:
        current_ax = ax
        standalone = False

    bg_color = "white"
    if standalone:
        fig.patch.set_facecolor(bg_color)
    current_ax.set_facecolor(bg_color)
    
    if cmap.lower() == "black":
        black_cmap = mcolors.ListedColormap(['black'])
        im = current_ax.imshow(grid, interpolation='nearest', origin='lower', cmap=black_cmap, vmin=0.0, vmax=1.0)
    else:
        im = current_ax.imshow(grid, interpolation='nearest', origin='lower', cmap=cmap, vmin=0.0, vmax=1.0)
    
    current_ax.set_aspect("equal")
    current_ax.axis("off")
    
    if title:
        current_ax.set_title(title, pad=10)
    
    if standalone and output:
        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
        plt.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
        print(f"Saved figure to {output} ({W}x{H} @ {dpi} DPI)")
        plt.close(fig)
        
    return im


def main():
    parser = argparse.ArgumentParser(
        description="Plot a saved DLA cluster .npz using high-performance rasterizer"
    )
    parser.add_argument("file", nargs="?", default="results/cluster.npz", help="Path to .npz cluster file")
    parser.add_argument("--out", default=None, help="Output image path")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap")
    parser.add_argument("--res", type=int, default=2048, help="Output image width in pixels")
    parser.add_argument("--full-res", action="store_true", help="Auto-calculate resolution")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output file")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        return

    if args.out is None:
        input_path = Path(args.file)
        base_name = input_path.stem
        known_colormaps = ["magma", "plasma", "inferno", "viridis", "cividis", "turbo", "hot", "cool", "jet", "black"]
        for cmap in known_colormaps:
            if base_name.endswith(f"_{cmap}"):
                base_name = base_name[:-len(f"_{cmap}")]
                break
        output_path = input_path.parent / f"{base_name}_{args.cmap}.png"
        args.out = str(output_path)

    result = utils.load_cluster(args.file)
    meta = result.meta or {}
    
    try:
        x, y = get_coordinates_from_result(result)
        num_particles = len(x)
    except (ValueError, AttributeError):
        if result.positions is not None:
            num_particles = len(result.positions)
        else:
            num_particles = None
    
    title = format_title(meta, num_particles=num_particles)
    
    render(positions=result, title=title, output=args.out, cmap=args.cmap, dpi=args.dpi, res=args.res, auto_res=args.full_res)
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()