# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDO CLV Pipeline processes Solar Dynamics Observatory (SDO) FITS files to recover center-to-limb variability (CLV) in solar radial velocity. It ingests synchronized HMI continuum, magnetogram, and Dopplergram images alongside AIA 1700 Å filtergrams, segments the solar disk into region types, and computes velocity statistics as a function of disk position (mu = cos(heliocentric angle)).

## Environment and Installation

The project uses `uv` for dependency management with Python 3.12.

```bash
# Install in editable mode
uv pip install -e .

# Run scripts via uv (used in SLURM batch jobs)
uv run scripts/run_pipe.py
```

A `pytest` suite lives in `tests/`. Alongside the unit tests (`test_quality.py`,
`test_moat.py`, `test_region_flags.py`, `test_feature_catalog.py`, …) two files gate
numerical equivalence of the fused kernels against their retained numpy oracles:
`test_kernel_equivalence.py` and `test_aggregate_context.py`. Both run on synthetic
frames so they need no FITS data. Committed end-to-end goldens live in
`tests/fixtures/golden/`. Run the suite with:

```bash
uv run --extra test pytest
```

CI (`.github/workflows/ci.yml`) installs `pytest` and runs `python -m pytest`. `tests/test_quality.py` covers the pure-Python `quality.py` module and is designed to pass without numba installed (the `--no-deps` CI job).

## Key Commands

**Run the pipeline** (reads from `--fitsdir`, defaults to `/mnt/ceph/users/mpalumbo/sdo_data/`):
```bash
uv run scripts/run_pipe.py --fitsdir /path/to/fits --globexp "2014*01*07*" --clobber
```

**Download SDO data** via JSOC/sunpy Fido:
```bash
uv run sdo_clv_pipeline/sdo_download.py --outdir /path/to/output -s 2019-02-01 -e 2019-02-28 -c 4
```
`scripts/download_stuff.py` is a hardcoded convenience wrapper calling the same function.

**Merge per-run CSV outputs** into combined files at `data/`:
```bash
uv run scripts/merge_output.py
```

**Preprocess merged output** into per-region CSV files under `data/processed/`:
```bash
uv run scripts/preprocess_output.py
```

**Ad hoc / interactive run** (hardcoded date glob, serial):
```bash
uv run scripts/michael_run.py
```

**Process a single epoch** (idempotent, resumable — the unit of work for failure-isolated batch runs; select an epoch with `--index` or by explicit `--con/--mag/--dop/--aia`):
```bash
SDO_THREADS=4 uv run scripts/run_one.py --datadir data/<globdir> --fitsdir /path/to/fits --globexp "2014*01*07*" --index 0
```

**Generate a disBatch task file** (one `run_one.py` line per epoch, for failure-isolated fan-out under a Slurm allocation):
```bash
uv run scripts/make_disbatch_tasks.py --datadir data/<globdir> --globexp "2014*" --threads 1 > tasks.db
```

**Stitch per-PID temp CSVs** into the final outputs after a batch run (used to recover/resume; `--keep-tmp` preserves the temp files):
```bash
uv run scripts/stitch_tmp.py --datadir data/<globdir>
```

**Benchmark / diagnose the pipeline** (see the performance section below):
```bash
uv run scripts/benchmark_pipeline.py stages --globexp "2014*01*07*" --index 0
uv run scripts/benchmark_pipeline.py bandwidth --procs 1,2,4,8,16,31
uv run scripts/benchmark_pipeline.py pool --globexp "2014*01*" --epochs 32 --workers 4,16,31
```
`stages` also takes `--trace-alloc` for per-stage allocation high-water marks; the other
subcommands are `threads`, `io`, and `reproject`.

**SLURM batch submission** (Flatiron CCA cluster, `cca` partition, 64 tasks, 72h):
```bash
bash batch/run_pipe.sh
```
This chains: `runall.sh` → `merge_output.sh` → `preprocess_output.sh` via `--dependency=afterok`.

**`SDO_THREADS`** controls within-epoch numba threading (default 1). Keep it at 1 when fanning epochs out across the process pool or disBatch (one epoch per process — more threads would oversubscribe); raise it only for single-process/interactive runs.

## Architecture

### Pipeline stages (in `sdo_process.py`)

`reduce_sdo_images()` is the core single-epoch function, called by `process_data_set()`:

1. Load four FITS files into `SDOImage` objects (continuum `con`, magnetogram `mag`, Dopplergram `dop`, AIA filtergram `aia`).
2. Compute geometry (`dop.calc_geometry()`) — heliographic/heliocentric coordinate transforms computed analytically in `geometry.py` (`compute_geometry`, a numba-accelerated `pixel_to_hpc` → `hpc_to_hcc` → `hcc_to_hgs` chain validated against the sunpy/astropy `SkyCoord` path it replaced). The Dopplergram geometry is authoritative; `con` and `mag` inherit it via `inherit_geometry()`.
3. Reproject AIA onto HMI pixel scale via `aia.rescale_to_hmi(con)` (bilinear interpolation using a numba-JIT'd kernel in `reproject.py`).
4. Fit quadratic limb darkening law per image; divide out to get `iflat` (limb-flattened intensity).
5. Correct magnetogram for foreshortening (`image /= mu`).
6. Correct Dopplergram: subtract spacecraft velocity (`calc_spacecraft_vel`), then fit and remove differential rotation + meridional circulation + convective blueshift (CBS) using Legendre polynomial decomposition (`calc_bulk_vel`). Residual is `v_corr`.
7. Mask pixels with `mu < mu_thresh` (default 0.1).
8. Build `SunMask` — classifies every pixel into one of six region types (see below).
9. Compute velocity/magnetic/intensity statistics per mu-annulus per region; write to CSV.
   `aggregate.py` builds one shared per-epoch context (valid mask, disk totals, ring
   edges, region LUT) and two fused numba kernels that produce every (ring, region) and
   (flag, ring) sum in a single traversal each, replacing ~40 `np.bincount` passes.

Allocation-heavy full-frame numpy chains have been fused into single-pass numba
kernels, each keeping its numpy version in-tree as an oracle: `geometry.pixel_area_kernel`,
`doppler_kernels.spacecraft_vel_kernel`, and `limbdark.ld_bin_stats` /
`ld_bin_stats_clipped` / `ld_flatten`. See "Numerical-equivalence discipline" below
before touching any of them.

### Region classification (in `SunMask.identify_regions`, `sdo_image.py`)

Two planes, both module-level globals (not enums).

**`SunMask.regions` — the base partition.** Every on-disk pixel gets exactly one code:
- `umbrae_code = 1` — intensity < 45% quiet-sun mean
- `penumbrae_code = 2` — intensity 45–89%
- `quiet_sun_code = 3` — intensity > 89%, weak B field
- `network_code = 4` — intensity > 89%, strong B field or bright in AIA, area < 20 µhemispheres
- `plage_code = 5` — same as network but area ≥ 20 µhemispheres

**`SunMask.flags` — a non-exclusive uint8 bitmask plane** (like `quality.py`). A pixel
may carry several bits, and setting them never changes its base code: `blue_pen_flag`
/ `red_pen_flag` (penumbra `v_corr` ≤ 0 / > 0), `moat_left_flag` / `moat_right_flag`
(moat ring by hemisphere, only when `classify_moat=True`).

`flag_selections()` maps the flag plane to the derived `region_output.csv` codes, which
**overlap the base rows by design** (a moat pixel that is also plage contributes to
both): `moat_code = 6`, `blue_penumbra_code = 7`, `red_penumbra_code = 8`,
`left_moat_code = 9`, `right_moat_code = 10`, `plage_no_moat_code = 11`,
`network_no_moat_code = 12`.

The AIA 1700 Å image contributes to plage/network detection. Isolated single-pixel bright features are reclassified as quiet sun.

### Legendre polynomial decomposition (in `legendre.py`)

Adapted from Kashyap et al. (2021, arXiv:2105.12055). `gen_leg_vec` operates on heliographic latitude; `gen_leg_x_vec` operates on the disk-plane radial coordinate (rho). The model is a 6-term system (or 11 terms with `fit_cbs=True`): 3 differential rotation terms (odd l=1,3,5), 2 meridional circulation terms (even l=2,4), and 1–6 CBS Legendre terms. Solved via normal equations (`np.linalg.solve`).

### Parallel execution

`run_pipe.py` uses `multiprocessing` with `spawn` context and `maxtasksperchild=4` to control memory growth. Each worker writes temporary CSV files to `data/<globdir>/tmp/` named by PID; the main process stitches them afterward with `stitch_output_files` (re-runnable standalone via `scripts/stitch_tmp.py`). The numba JIT kernel is warmed up before `pool.starmap`.

There are two axes of parallelism: across epochs (the process pool above, or disBatch via `make_disbatch_tasks.py` + `run_one.py`) and within an epoch (numba thread count, set by `parallel.set_compute_threads` / the `SDO_THREADS` env var, default 1). Use one or the other, not both — `__init__.py` pins threads to 1 on import so pool workers don't oversubscribe. `logging_setup.py` configures structured per-epoch logging used by the entry-point scripts.

#### Performance: the pool is memory-bandwidth bound (measured)

Measured on an idle 32-core ccalin node, warm cache, via `scripts/benchmark_pipeline.py`
(subcommands `stages`, `threads`, `bandwidth`, `io`, `reproject`, `pool`):

- **One epoch, 1 process, 1 thread: ~14.3 s, ~4.5 GB peak RSS.** No single hotspot;
  the largest costs are `compute_pixel_mapping` (~3.4 s), `compute_geometry` (~2.2 s),
  `bulk_vel_design` (~2.5 s), `get_areas` (~2.2 s) and the FITS reads (~2.2 s).
- **Aggregate memory bandwidth saturates at ~130 GB/s by 16 processes**; per-process
  bandwidth falls 18.9 → 7.9 → 4.3 GB/s at 1 / 16 / 31 workers. Pool throughput
  follows: 4 workers 3.2×, 16 workers 7.8×, **31 workers 7.1× — slower than 16**.
  `run_pipe.py` uses `ncpus = affinity - 1`, which is past the optimum; **capping
  workers near cores/2 is the lever**, and it also halves the N × 4.5 GB footprint.
  This is still an open item: the measurement is in, the code has not been changed.
- **The BLAS/OpenMP env pin is not the fix.** `OPENBLAS/OMP/MKL/NUMEXPR=1` measures
  ~1.02×, flat across worker count. Harmless hygiene, not a remedy.
- **Cold I/O is not the bottleneck either.** A cold FITS read costs only ~1 s/epoch
  more than a warm one, and the read is RICE-*decompression*-bound (~57 MB of
  compressed input expands to ~268 MB of arrays), not transfer-bound.
- **`SDO_THREADS` buys little.** Measured Amdahl parallel fraction is p ≈ 0.18
  (1 → 16 threads takes 19.6 → 16.2 s on the pre-refactor code). `bulk_vel_design`
  shows *no* speedup at all: it is bandwidth-bound writing a ~640 MB design matrix.

Production's historical ~60 s/epoch is contention, not I/O: 31 workers × ~2 s/epoch
throughput is ~70 worker-seconds per epoch.

#### Numerical-equivalence discipline

Several hot paths keep a `*_numpy` oracle beside the fused fast path
(`calc_geometry_numpy`, `calc_bulk_vel_numpy`, `calculate_pixel_area_numpy`,
`calc_spacecraft_vel_numpy`, `calc_limb_darkening_numpy`,
`compute_pixel_mapping_highlevel`). **Every one of them has a test that compares it**
— an oracle nobody checks is pure maintenance cost, so if you add one, wire it in.
(`compute_pixel_mapping_highlevel` is the exception to the pytest rule: it is the
reference for the stride sweep in `benchmark_pipeline.py reproject`, since its cost
is ~13 s per call.)

Both gates live in the pytest suite, so they run automatically:

- `tests/test_kernel_equivalence.py` — synthetic frames, no data needed, runs in CI.
- `tests/test_real_epoch.py` — full 4096x4096 frames plus the end-to-end golden CSVs.
  **Skips automatically** when FITS input is unreachable, so CI stays green. Point it
  elsewhere with `SDO_TEST_FITSDIR` / `SDO_TEST_GLOBEXP`.

```bash
uv run --extra test pytest                      # everything, real-data gates included
uv run --extra test pytest -k "not real_epoch"  # fast: skip the ~110 s real-data pass
SDO_REGEN_GOLDEN=1 uv run --extra test pytest tests/test_real_epoch.py -k golden
```

The last form rewrites `tests/fixtures/golden/` after an *intended* numerical change;
review `git diff` on it before keeping the result. Note `*.csv` is gitignored, so the
goldens need `git add -f`.

The shared gate is in `tests/conftest.py`. Default is rtol 1e-12 with a
**scale-aware** atol (`1e-12 * max|reference|`), because `v_corr` and `v_conv` cross
zero by construction and `atol=0` false-fails on meaningless residual cancellations.
Two escapes, both to be used deliberately:

- `assert_bit_identical` — for paths that only reorder work and never change the
  arithmetic. Most of the fused kernels meet this; a near-miss there is a bug.
- `ulp_floor=True` — raises rtol to a few epsilons of the *stored* dtype. Required
  when comparing two genuinely different **algorithms** whose results land in a
  narrow type: float32 has eps = 1.2e-7, so the default 1e-12 asks for more
  precision than the array can hold. The geometry pair (inline TAN inverse vs
  astropy `wcs_pix2world`) and the bulk-velocity pair (in-kernel Legendre recurrence
  vs `scipy.special.eval_legendre`) are both in this category; their measured
  agreement is recorded in the test comments.

Traps this codebase has actually hit, all invisible to a tolerance-only gate:

- **NEP 50 weak vs strong scalars.** `ldark`/`iflat` are **float64** because `b`/`c`
  come from `np.polyfit` as `np.float64` (strong) and promote the float32 `mu`; but
  `1.0 - mu` uses a *Python* float (weak) and stays float32. Kernels must reproduce
  the mixed sequence — numba has no weak scalars, so casts must be explicit.
- **`float32 / np.int64` → float64, `float32 / python_int` → float32.** Counts taken
  from `np.nansum(bool)` must stay `np.int64` or `avg_int` silently loses precision.
- **`np.nansum` on a float32 array accumulates in float32**, while `np.bincount`
  upcasts weights to float64. Two aggregations that look equivalent are not, which is
  why `compute_region_only_flag_results` stays on numpy.
- **Fusing elementwise chains is bit-identical; fusing reductions usually is not.**
  Per-bin `np.bincount` accumulation is sequential in C order, so a single-threaded
  loop reproduces it exactly — hence the aggregation kernels are `parallel=False`.
  `np.nansum` uses pairwise summation and cannot be reproduced by a loop.

### Output schema

Three CSV files per run, then merged and post-processed. Schemas are defined once in
`sdo_io.py` (`header_thresholds`, `header_region`, `header_feature`):
- `thresholds.csv`: one row per epoch — MJD, limb-darkening coefficients, intensity thresholds, spacecraft/rotation/meridional velocity statistics, `quality_flag`
- `region_output.csv`: one row per (epoch, region-or-flag code, mu-bin) — `mjd`, `region` (integer code, 1–5 base or 6–12 flag-derived), `lo_mu`, `hi_mu`, `pixel_frac`, `light_frac`, `v_hat`, `v_phot`, `v_quiet`, `v_conv`, `mag_unsigned`, `avg_int`, `avg_int_flat`, `quality_flag`
- `feature_output.csv`: one row per connected umbra/penumbra blob — geometry, both LOS and radial field conventions (`_los` / `_rad`), unsigned flux, and velocities

`preprocess_output.py` splits `region_output.csv` into per-region CSV files under `data/processed/`. Full-disk integrated rows have `lo_mu = hi_mu = NaN`.

### Data locations

- Raw FITS input: `/mnt/ceph/users/mpalumbo/sdo_data/` (default, Flatiron Ceph)
- Pipeline output CSVs: `data/<globdir>/` relative to repo root
- Processed output: `data/processed/`
- `paths.py` provides `root`, `src`, `data`, `scripts`, `figures` as `pathlib.Path` objects

### File naming conventions

`sdo_io.get_date()` parses timestamps from three FITS filename formats:
- HMI 45s: `hmi.*YYYY_MM_DD_HH_MM_SS*.continuum.fits` / `.magnetogram.fits` / `.Dopplergram.fits`
- HMI 720s: `hmi.*.YYYYmmdd_HHMMSS*.continuum.fits` etc.
- AIA: `aia*lev1*1700*YYYY_MM_DDtHH_MM_SS*.fits`

All timestamps are rounded to the nearest hour before matching across instrument types.

## Dependencies

Key non-standard dependencies: `sunpy[all]` (coordinate transforms, Fido downloads), `astropy`, `numba` (JIT bilinear reprojection), `scikit-image` (region labeling via `regionprops`), `scipy` (curve fitting, ndimage), `tqdm`, `PyQt5`/`PyQt6` (matplotlib backend).

## Notes

- `matplotlib` style is set globally from `my.mplstyle` in the repo root; scripts call `plt.style.use(str(root) + "/my.mplstyle")` and `plt.ioff()`.
- `QUALITY` is the data quality gate, decoded in `quality.py` against the JSOC HMI bit definitions. A clean (`0`) flag passes; an epoch setting only "soft" bits (`is_tolerable_quality` — bits 10/15/16: unreadable cosmic-ray list, closest-neighbor keyword interp, low interp-point count) is still processed but the flag is recorded in the output so it can be masked downstream; any other nonzero bit is fatal and the epoch is skipped. `decode_quality`/`format_quality` turn the bitmask into human-readable skip reasons.
- FITS verification warnings from astropy are globally silenced (`silentfix` + `simplefilter("ignore")`).
- `scripts/michael_run.py` has hardcoded paths and date globs — it is an interactive development script, not a general-purpose entry point.
- **Public attributes and methods on `SDOImage` and `SunMask` are external API.** An
  external analysis repo imports this package directly, so an attribute with no reader
  *inside* this repo is not necessarily dead. Prefer converting it to a lazy
  `@property` (as the `ff` / `*_frac` area fractions are) or deprecating it, rather
  than deleting it outright.
- Modules re-export each other with `from .x import *`, so a name has several live
  bindings: `sdo_process` holds its own `write_results_to_file`, `sdo_image` its own
  `compute_geometry`. Anything that patches, probes, or reasons about which function
  actually runs must target the **consuming** module, not the defining one.
- `.gitignore` ignores `.claude/*` but re-includes this file via `!.claude/CLAUDE.md`,
  so **this file is tracked and shared** — keep it accurate for collaborators.
  `.claude/settings.local.json` stays ignored (personal permissions). Separately,
  `*.csv` is ignored, so the golden fixtures need `git add -f`.
