# Refactoring Plan for Phase B (DLA Analysis)

Short plan to streamline the analysis workflow after Phase A fixes (Sample Fine 1°, Analyze Coarse 4°).

---

## 1. Bottlenecks: Separate Processing from Plotting

**Current:** Heavy processing (`batch_sector_analysis`, `batch_analysis`, `generate_density_grid`) runs in the same notebooks as dashboard plotting. Re-running the notebook re-runs all processing.

**Recommendation:**

- **Process script:** `notebooks/05_process_data.py` (run from repo root: `python notebooks/05_process_data.py`). It:
  - Runs `batch_analysis` and `batch_sector_analysis` (with `num_sectors=360`) per model/size.
  - Optionally runs `generate_density_grid` for dashboards.
  - Saves intermediate results to disk, e.g.:
    - Sector data: `results/processed/{model}_{size}_sectors.npz` (or `.pkl` with `times` + 3D array).
    - Scalar metrics: `results/processed/{model}_{size}_basic.csv` (from `batch_analysis`).
    - Anisotropy metrics: `results/processed/{model}_{size}_anisotropy.csv` (from `compute_anisotropy_metrics`).
    - Density grids: `results/processed/{model}_{size}_density.npz` (grid, bins, max_r, rg) if needed.
- **Keep `03_batch_analysis1M.ipynb` and `04_batch_analysis10M.ipynb`** as “visualisation / dashboard” notebooks that:
  - Load the pre-saved CSVs and NPZ/PKL (or assume in-memory variables from a single “run processing” cell at the top).
  - Only run plotting and summary tables.

This way, Phase B can re-plot quickly from cached data and only re-run processing when inputs change.

---

## 2. Code Duplication: Move Helpers into `src/analysis/metrics.py`

**Current:** `calculate_box_dim` and growth/dimension logic already live in `src/analysis/metrics.py` and are used via `processing.batch_analysis`. No duplicate definitions of `calculate_box_dim` were found in the notebooks.

**Action:**

- **Keep all metric helpers in `src/analysis/metrics.py`.** If any notebook defines its own “calculate_box_dim” or “D = 1/beta” helpers, remove them and import from `src.analysis.metrics` (or `processing` if they are used only in batch context).
- **Notebooks:** Use only `proc.batch_analysis`, `proc.batch_sector_analysis`, `proc.compute_anisotropy_metrics`, and `viz.*` for plots. Dimension from scaling: compute `D_i = 1/beta_i` per row in the notebook (or in a small helper in `metrics.py`, e.g. `dimension_from_beta(beta_series)`) and then take mean/std.

---

## 3. Suggested Block Mapping (What to Move Where)

| Block / responsibility              | Current location              | Move to / keep as                          |
|-------------------------------------|-------------------------------|--------------------------------------------|
| `batch_analysis` + `batch_sector_analysis` + optional density | 03 / 04 notebooks (inline)   | **Process_Data.ipynb** or **run_batch_analysis.py**; save CSVs + NPZ |
| Loading saved sector/basic/anisotropy data | —                             | **03 / 04 notebooks** (first cells: load from `results/processed/`) |
| Dashboard (5-row: Density, Polar, Semi-Log, Beta, Scaling) | 03 / 04 notebooks            | **Keep in 03 / 04**; ensure they only plot and build summary tables |
| Per-row D = 1/β then mean/std        | 03 comparison table (fixed)  | **Keep in notebook**; optional: add `metrics.dimension_from_beta(df['beta'])` in `metrics.py` |
| `calculate_box_dim`, `downsample_centered`, anisotropy metrics | `src/analysis/metrics.py`     | **Keep** (no duplication found)            |

---

## 4. Summary of Phase A Fixes Applied

- **metrics.py:** `downsample_centered` (360→90, shift=2, `nanmean`); `anisotropy_ratio` tolerance small (1.0), diagonal wraparound, division-by-zero guard; `anisotropy_fourier`/`anisotropy_ratio` empty-grid guards; `calculate_anisotropy_old` bug fix (`np.sum(valid) < num_bins/2`).
- **plotting.py:** `plot_beta_profile` downsamples to 90 bins before plotting; title uses raw string for `\beta`.
- **Notebooks:** Extraction already uses `num_sectors=360`; comparison dashboard subplot order set to Row 3 = Semi-Log, Row 4 = Beta, Row 5 = Scaling; single-model views updated to 5 rows with Beta; dimension statistics use mean(1/β) and std(1/β) from per-row `df_basic['beta']` in the comparison table.
