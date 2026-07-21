"""Contained benchmark of the three SDOImage geometry implementations.

Compares, on a single real FITS header, the wall-clock cost of:

  * calc_geometry_sunpy  -- reference SkyCoord.transform_to chain (slow oracle)
  * calc_geometry_numpy  -- astropy wcs_pix2world + vectorized numpy analytic
  * calc_geometry        -- fused analytic numba kernel (production default)

Geometry cost is a pure function of the WCS grid, not the pixel data, so one
FITS file is sufficient. The numba path is timed across a sweep of thread counts
(via parallel.set_compute_threads) because its prange kernel is thread-parallel;
the numpy/sunpy paths are single-threaded references.

The benchmark is "contained": each method reuses one preloaded SDOImage,
warmup calls (which pay numba's JIT/cache load) are excluded from the stats, and
gc is disabled around each timed block to suppress collection-induced variance.

    uv run scripts/benchmark_geometry.py --file /path/to/some.continuum.fits
"""

import argparse, gc, statistics, time

import numba

from sdo_clv_pipeline.sdo_image import SDOImage
from sdo_clv_pipeline.parallel import set_compute_threads


def time_method(fn, repeats, warmup):
    """Return per-call durations (seconds) for fn, excluding warmup calls."""
    # warmup: pays numba JIT/cache load and any lazy imports so they don't
    # contaminate the timed samples
    for _ in range(warmup):
        fn()

    times = []
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    finally:
        gc.enable()
    return times


def summarize(times):
    """Reduce a list of durations (s) to (min, median, std) in milliseconds."""
    ms = [t * 1e3 for t in times]
    lo = min(ms)
    med = statistics.median(ms)
    std = statistics.stdev(ms) if len(ms) > 1 else 0.0
    return lo, med, std


def run_benchmark(file, repeats, warmup, thread_sweep):
    # load one image; every method overwrites the same .xx/.mu/... attributes,
    # so re-calling on this single object is clean and isolated
    print(f"file           : {file}", flush=True)
    img = SDOImage(file)
    print(f"image shape    : {img.naxis2} x {img.naxis1}", flush=True)
    print(f"NUMBA_NUM_THREADS (max): {numba.config.NUMBA_NUM_THREADS}", flush=True)
    print(f"repeats={repeats}  warmup={warmup}", flush=True)
    print("", flush=True)

    # (label, threads_actual, times) rows for the results table
    rows = []

    # single-threaded reference paths (pin numba to 1 so a stray thread setting
    # can't affect the pieces of the numpy path that touch parallel kernels)
    set_compute_threads(1)
    rows.append(("sunpy", None, time_method(img.calc_geometry_sunpy, repeats, warmup)))
    rows.append(("numpy", None, time_method(img.calc_geometry_numpy, repeats, warmup)))

    # numba path across the thread sweep; dedupe on the value actually set
    # (set_compute_threads clamps to NUMBA_NUM_THREADS)
    seen_threads = set()
    for req in thread_sweep:
        actual = set_compute_threads(req)
        if actual in seen_threads:
            continue
        seen_threads.add(actual)
        rows.append(("numba", actual, time_method(img.calc_geometry, repeats, warmup)))

    # baseline for speedup column: the sunpy reference median
    sunpy_med = summarize(rows[0][2])[1]

    print(f"{'method':<8}{'threads':>9}{'min (ms)':>12}{'median':>12}"
          f"{'std':>10}{'speedup':>10}", flush=True)
    print("-" * 61, flush=True)
    for label, threads, times in rows:
        lo, med, std = summarize(times)
        tstr = "-" if threads is None else str(threads)
        speedup = sunpy_med / med if med > 0 else float("nan")
        print(f"{label:<8}{tstr:>9}{lo:>12.2f}{med:>12.2f}"
              f"{std:>10.2f}{speedup:>9.1f}x", flush=True)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="path to a single real SDO FITS file")
    p.add_argument("--repeats", type=int, default=7, help="timed calls per config")
    p.add_argument("--warmup", type=int, default=1, help="untimed warmup calls per config")
    p.add_argument("--threads", default="1,2,4,8",
                   help="comma-separated numba thread counts to sweep")
    args = p.parse_args()

    thread_sweep = [int(x) for x in args.threads.split(",") if x.strip()]
    run_benchmark(args.file, args.repeats, args.warmup, thread_sweep)
    return None


if __name__ == "__main__":
    main()
