"""Diagnostic benchmark for the run_pipe per-epoch analysis workflow.

Supplements static reading of the pipeline with measurements. Five subcommands,
each aimed at a specific question:

  stages     Where does one epoch's wall clock go? Wraps the real functions in
             sdo_process/sdo_image/sdo_vels/geometry/reproject in place and runs
             the actual process_data_set(), so the profile cannot drift from the
             pipeline. Reports inclusive and *exclusive* time per stage (the
             exclusive column sums to the total), so a parent stage's own glue is
             separated from its children.
  threads    How much of an epoch is actually thread-scalable? Sweeps
             SDO_THREADS and re-profiles, then solves Amdahl for the parallel
             fraction p. Only the three @njit(parallel=True) kernels can scale;
             everything else is the serial floor that caps single-epoch latency.
  bandwidth  Is the across-epoch pool memory-bandwidth bound? Runs a STREAM-like
             triad in N concurrent processes and reports aggregate GB/s vs N. If
             aggregate bandwidth saturates at a small N, the pool cannot scale
             past it no matter how the threads are pinned.
  io         How expensive is a cold FITS read? Uses posix_fadvise(DONTNEED) to
             evict a file's page cache without root, then times a cold read and a
             warm re-read of the same file. This is the measurement that decides
             whether production's ~60 s/epoch is I/O or compute.
  pool       End-to-end throughput vs worker count, the run_pipe configuration
             itself. Reports epochs/s, speedup, and parallel efficiency.

Notes on honest measurement:
  * Repeats 2..N are the steady-state samples; repeat 1 additionally pays numba
    JIT/cache load and lazy astropy/sunpy imports and is reported separately.
  * Timing runs and allocation-tracing runs are separate (--trace-alloc adds
    tracemalloc overhead to every allocation).
  * Re-running the same epoch warms the page cache, so `stages` numbers are
    warm-cache numbers and understate production I/O. Use `io` for that.
  * `pool` needs epochs >> workers, otherwise spawn/import storm dominates the
    high-worker cells and fakes a scaling ceiling.

Examples:
  uv run scripts/benchmark_pipeline.py stages --globexp '2014*01*07*' --index 0
  uv run scripts/benchmark_pipeline.py threads --globexp '2014*01*07*' --threads 1,2,4,8,16
  uv run scripts/benchmark_pipeline.py bandwidth --procs 1,2,4,8,16,31
  uv run scripts/benchmark_pipeline.py io --globexp '2014*01*07*' --epochs 3
  uv run scripts/benchmark_pipeline.py pool --globexp '2014*01*' --epochs 32 --workers 1,4,8,16
"""

import argparse, os, resource, shutil, statistics, time, tracemalloc

import numpy as np


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def read_rss_bytes():
    """Current resident set size in bytes (cheap /proc read, no fork)."""
    with open("/proc/self/statm", "r") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def peak_rss_bytes():
    """Peak RSS of this process in bytes (ru_maxrss is in KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def fmt_bytes(n):
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return None


def resolve_epochs(args, n_wanted):
    """Return up to n_wanted (con, mag, dop, aia) tuples for this run."""
    from sdo_clv_pipeline.sdo_io import find_data

    if args.con:
        assert args.mag and args.dop and args.aia, \
            "explicit mode needs all of --con/--mag/--dop/--aia"
        return [(args.con, args.mag, args.dop, args.aia)]

    con, mag, dop, aia = find_data(args.fitsdir, globexp=args.globexp)
    n_found = len(con)
    assert n_found > 0, "no matched epochs for globexp=%r in %s" % (args.globexp, args.fitsdir)
    start = args.index
    assert 0 <= start < n_found, "index %d out of range [0, %d)" % (start, n_found)
    sel = list(zip(con, mag, dop, aia))[start:start + n_wanted]
    print("matched %d epochs; using %d starting at index %d"
          % (n_found, len(sel), start), flush=True)
    return sel


def make_scratch_datadir(path):
    """Fresh output dir for benchmark CSVs, so the user's data/ is untouched."""
    tmpdir = os.path.join(path, "tmp")
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(tmpdir)
    return path


# ---------------------------------------------------------------------------
# nesting-aware profiler
# ---------------------------------------------------------------------------

class Profiler(object):
    """Wall-clock profiler that wraps existing callables in place.

    Maintains a stack of per-frame child-time accumulators, so each probe can
    report inclusive time (itself plus children) and exclusive time (itself
    only). Exclusive times over the whole probe set sum to the root's inclusive
    time, which means any unprobed inline work shows up as the exclusive time of
    whichever probed parent contains it -- nothing goes missing.

    Parameters
    ----------
    trace_alloc : bool, optional
        Also record the peak allocation high-water mark per top-level stage via
        tracemalloc. Adds real overhead, so timings from a traced run should not
        be compared against an untraced one.
    """

    def __init__(self, trace_alloc=False):
        self.trace_alloc = trace_alloc
        self.stack = []
        self.records = {}
        self.order = []
        self.saved = []
        return None

    def wrap(self, owner, attr, label, depth):
        """Replace owner.attr with a timing probe recorded under label."""
        original = getattr(owner, attr)
        self.records[label] = {"depth": depth, "calls": 0, "incl": 0.0,
                               "excl": 0.0, "alloc": 0, "rss": 0}
        self.order.append(label)
        rec = self.records[label]
        prof = self

        def probe(*args, **kwargs):
            # frame = [child inclusive time, max absolute traced peak seen].
            # tracemalloc's peak is global, so a nested probe would clobber its
            # parent's high-water mark. Each frame therefore carries the largest
            # absolute peak observed so far, children hand theirs up on exit, and
            # the peak counter is reset after every pop. That makes the alloc
            # column "high-water above this stage's entry level" at any depth.
            traced = prof.trace_alloc
            cur0 = 0
            rss0 = 0
            if traced:
                cur0, _ = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
                rss0 = read_rss_bytes()
            prof.stack.append([0.0, cur0])
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                child, peak_seen = prof.stack.pop()
                if prof.stack:
                    prof.stack[-1][0] += dt
                rec["calls"] += 1
                rec["incl"] += dt
                rec["excl"] += dt - child
                if traced:
                    _, peak_now = tracemalloc.get_traced_memory()
                    peak_abs = max(peak_now, peak_seen)
                    rec["alloc"] = max(rec["alloc"], peak_abs - cur0)
                    rec["rss"] += read_rss_bytes() - rss0
                    if prof.stack:
                        prof.stack[-1][1] = max(prof.stack[-1][1], peak_abs)
                    tracemalloc.reset_peak()

        probe.__name__ = getattr(original, "__name__", attr)
        probe.__doc__ = getattr(original, "__doc__", None)
        setattr(owner, attr, probe)
        self.saved.append((owner, attr, original))
        return None

    def reset(self):
        """Zero the counters (between repeats) without unwrapping."""
        for rec in self.records.values():
            rec.update({"calls": 0, "incl": 0.0, "excl": 0.0, "alloc": 0, "rss": 0})
        self.stack = []
        return None

    def snapshot(self):
        """Deep-ish copy of the current counters."""
        return {k: dict(v) for k, v in self.records.items()}

    def restore(self):
        """Put every original callable back."""
        for owner, attr, original in reversed(self.saved):
            setattr(owner, attr, original)
        self.saved = []
        return None


def install_probes(prof, root_label):
    """Wrap the pipeline call path. Order here is the report's display order.

    Probes are installed on the *consuming* namespace where that matters: the
    package uses `from .x import *` liberally, so e.g. sdo_process holds its own
    reference to write_results_to_file and sdo_image holds its own reference to
    compute_geometry. Class methods are patched on the class, which covers every
    instance.
    """
    from sdo_clv_pipeline import sdo_image, sdo_process
    from sdo_clv_pipeline.sdo_image import SDOImage, SunMask

    specs = [
        # ---- reduction (reduce_sdo_images) ----
        (sdo_process, "_reduce_sdo_images", "reduce (load + correct + classify)", 1),
        (SDOImage, "__init__", "  FITS load x4 (SDOImage init)", 2),
        (sdo_image, "read_data", "    read_data (pixels)", 3),
        (sdo_image, "read_header", "    read_header (WCS)", 3),
        (SDOImage, "calc_geometry", "  geometry (dop, authoritative)", 2),
        (sdo_image, "compute_geometry", "    compute_geometry (numba)", 3),
        (sdo_image, "calculate_pixel_area", "    calculate_pixel_area", 3),
        (SDOImage, "inherit_geometry", "  inherit_geometry (con, mag)", 2),
        (SDOImage, "rescale_to_hmi", "  AIA reproject to HMI grid", 2),
        (sdo_image, "compute_pixel_mapping", "    compute_pixel_mapping (SkyCoord)", 3),
        (sdo_image, "bilinear_reproject", "    bilinear_reproject (numba)", 3),
        (SDOImage, "calc_limb_darkening", "  limb darkening (con + aia)", 2),
        (SDOImage, "correct_magnetogram", "  magnetogram foreshortening", 2),
        (SDOImage, "correct_dopplergram", "  dopplergram correction", 2),
        (SDOImage, "calc_spacecraft_vel", "    spacecraft velocity", 3),
        (SDOImage, "calc_bulk_vel", "    bulk velocity fit", 3),
        (sdo_image, "bulk_vel_design", "      design matrix (numba)", 4),
        (SDOImage, "mask_low_mu", "  mask_low_mu x4", 2),
        (SunMask, "__init__", "  region classification (SunMask)", 2),
        (SunMask, "identify_regions", "    identify_regions", 3),
        (sdo_image, "calculate_weights", "      calculate_weights (convolve)", 4),
        (sdo_image, "get_areas", "      get_areas (regionprops)", 4),
        (sdo_image, "detect_moats", "      detect_moats (opt-in)", 4),
        # ---- aggregation + output (process_data_set body) ----
        (sdo_process, "shared_products", "shared_products", 1),
        (sdo_process, "build_agg_inputs", "build_agg_inputs (shared ctx)", 1),
        (sdo_process, "_region_index", "region index gather", 1),
        (sdo_process, "compute_disk_results", "disk-integrated row", 1),
        (sdo_process, "compute_region_only_results", "region rows (no mu bins)", 1),
        (sdo_process, "compute_region_results", "region x mu-ring rows", 1),
        (sdo_process, "flag_selections", "flag selections", 1),
        (sdo_process, "compute_region_only_flag_results", "flag rows (no mu bins)", 1),
        (sdo_process, "compute_region_flag_results", "flag x mu-ring rows", 1),
        (sdo_process, "compute_feature_catalog", "feature catalog (per blob)", 1),
        (sdo_process, "write_results_to_file", "CSV write", 1),
        (sdo_process, "gc", "gc.collect (per-epoch)", 1),
    ]

    # root probe: its exclusive time is the un-probed inline work in
    # process_data_set (the ravel/np.abs block, k_hat, quality, row tagging)
    prof.wrap(sdo_process, "process_data_set", root_label, 0)
    for owner, attr, label, depth in specs:
        if attr == "gc":
            # gc.collect() is called on the module object inside a finally block;
            # patch the function on the gc module itself, not the name.
            prof.wrap(sdo_process.gc, "collect", label, depth)
            continue
        if not hasattr(owner, attr):
            print("WARNING: probe target missing, skipping: %s.%s" % (owner, attr), flush=True)
            continue
        prof.wrap(owner, attr, label, depth)
    return None


def print_profile(snap, total, trace_alloc=False):
    """Print the hierarchical profile followed by the exclusive-time ranking."""
    head = "%-42s %6s %10s %10s %7s" % ("stage", "calls", "incl ms", "excl ms", "% excl")
    if trace_alloc:
        head += " %11s" % "peak alloc"
    print(head, flush=True)
    print("-" * (len(head)), flush=True)

    for label, rec in snap.items():
        if rec["calls"] == 0:
            continue
        pct = 100.0 * rec["excl"] / total if total > 0 else float("nan")
        line = "%-42s %6d %10.1f %10.1f %6.1f%%" % (
            label, rec["calls"], rec["incl"] * 1e3, rec["excl"] * 1e3, pct)
        if trace_alloc:
            line += " %11s" % fmt_bytes(rec["alloc"])
        print(line, flush=True)

    print("", flush=True)
    print("exclusive-time ranking (where the wall clock actually goes):", flush=True)
    ranked = sorted((r["excl"], k) for k, r in snap.items() if r["calls"] > 0)
    cum = 0.0
    for excl, label in reversed(ranked):
        if excl <= 0.0005 * total:
            continue
        cum += excl
        # the root's exclusive time is by construction the un-probed inline work
        name = "inline glue in process_data_set" if snap[label]["depth"] == 0 \
            else label.strip()
        print("  %-44s %8.1f ms  %5.1f%%   (cum %5.1f%%)"
              % (name, excl * 1e3, 100.0 * excl / total,
                 100.0 * cum / total), flush=True)
    return None


# ---------------------------------------------------------------------------
# subcommand: stages
# ---------------------------------------------------------------------------

def run_stages(args):
    """Profile one epoch of the real process_data_set call path."""
    from sdo_clv_pipeline import sdo_process
    from sdo_clv_pipeline.parallel import set_compute_threads

    threads = set_compute_threads(args.threads)
    epoch = resolve_epochs(args, 1)[0]
    datadir = make_scratch_datadir(args.datadir)

    print("", flush=True)
    print("=" * 92, flush=True)
    print("STAGES: single-epoch profile", flush=True)
    print("=" * 92, flush=True)
    for label, path in zip(("con", "mag", "dop", "aia"), epoch):
        print("  %s %12s  %s" % (label, fmt_bytes(os.path.getsize(path)),
                                 os.path.basename(path)), flush=True)
    print("  numba threads=%d  fit_cbs=%s  classify_moat=%s  n_rings=%d  mu_thresh=%.2f"
          % (threads, args.fit_cbs, args.classify_moat, args.n_rings, args.mu_thresh),
          flush=True)
    print("  repeats=%d (repeat 1 includes numba JIT / lazy imports)" % args.repeats,
          flush=True)
    print("", flush=True)

    if args.trace_alloc:
        tracemalloc.start()

    prof = Profiler(trace_alloc=args.trace_alloc)
    root = "process_data_set (TOTAL)"
    install_probes(prof, root)

    snaps = []
    try:
        for i in range(args.repeats):
            prof.reset()
            t0 = time.perf_counter()
            status = sdo_process.process_data_set(
                *epoch, mu_thresh=args.mu_thresh, n_rings=args.n_rings,
                suffix="bench%d" % i, datadir=datadir, fit_cbs=args.fit_cbs,
                plot_moat=False, classify_moat=args.classify_moat)
            wall = time.perf_counter() - t0
            assert status == "ok", "epoch did not reduce cleanly: status=%r" % status
            snaps.append((wall, prof.snapshot()))
            print("  repeat %d: %.2f s  (peak RSS %s)"
                  % (i + 1, wall, fmt_bytes(peak_rss_bytes())), flush=True)
    finally:
        prof.restore()
        if args.trace_alloc:
            tracemalloc.stop()

    print("", flush=True)
    walls = [w for w, _ in snaps]
    if len(walls) > 1:
        steady = walls[1:]
        print("JIT / first-call overhead: %.2f s  (repeat 1 %.2f s vs steady median %.2f s)"
              % (walls[0] - statistics.median(steady), walls[0], statistics.median(steady)),
              flush=True)
        # report the steady repeat closest to the median so the table is a real
        # single observation, not an average of structurally different runs
        target = statistics.median(steady)
        pick = min(range(1, len(snaps)), key=lambda i: abs(snaps[i][0] - target))
    else:
        pick = 0
    wall, snap = snaps[pick]
    print("", flush=True)
    print("profile of repeat %d (%.2f s):" % (pick + 1, wall), flush=True)
    print("", flush=True)
    print_profile(snap, snap[root]["incl"], trace_alloc=args.trace_alloc)

    probed = sum(r["excl"] for r in snap.values() if r["depth"] > 0)
    print("", flush=True)
    print("accounting: total %.2f s = probed stages %.2f s + unprobed inline glue %.2f s"
          % (snap[root]["incl"], probed, snap[root]["excl"]), flush=True)
    print("peak RSS: %s" % fmt_bytes(peak_rss_bytes()), flush=True)
    return None


# ---------------------------------------------------------------------------
# subcommand: threads
# ---------------------------------------------------------------------------

def run_threads(args):
    """Sweep SDO_THREADS and report the thread-scalable fraction of an epoch."""
    from sdo_clv_pipeline import sdo_process
    from sdo_clv_pipeline.parallel import set_compute_threads
    import numba

    epoch = resolve_epochs(args, 1)[0]
    datadir = make_scratch_datadir(args.datadir)
    sweep = [int(x) for x in args.threads_sweep.split(",") if x.strip()]

    print("", flush=True)
    print("=" * 92, flush=True)
    print("THREADS: SDO_THREADS sweep on one epoch (single process)", flush=True)
    print("=" * 92, flush=True)
    print("NUMBA_NUM_THREADS ceiling: %d" % numba.config.NUMBA_NUM_THREADS, flush=True)

    # the three parallel=True kernels are the only thread-scalable work
    kernels = ["    compute_geometry (numba)",
               "    bilinear_reproject (numba)",
               "      design matrix (numba)"]

    prof = Profiler()
    root = "process_data_set (TOTAL)"
    install_probes(prof, root)
    rows = []
    try:
        # untimed warmup so JIT is paid before the first timed cell
        sdo_process.process_data_set(*epoch, mu_thresh=args.mu_thresh,
                                     n_rings=args.n_rings, suffix="warm",
                                     datadir=datadir, fit_cbs=args.fit_cbs,
                                     plot_moat=False, classify_moat=args.classify_moat)
        seen = set()
        for req in sweep:
            actual = set_compute_threads(req)
            if actual in seen:
                continue
            seen.add(actual)
            best = None
            for i in range(args.repeats):
                prof.reset()
                sdo_process.process_data_set(
                    *epoch, mu_thresh=args.mu_thresh, n_rings=args.n_rings,
                    suffix="t%d_%d" % (actual, i), datadir=datadir,
                    fit_cbs=args.fit_cbs, plot_moat=False,
                    classify_moat=args.classify_moat)
                snap = prof.snapshot()
                if best is None or snap[root]["incl"] < best[root]["incl"]:
                    best = snap
            rows.append((actual, best))
            print("  threads=%2d  total %.2f s" % (actual, best[root]["incl"]), flush=True)
    finally:
        prof.restore()
        set_compute_threads(1)

    print("", flush=True)
    print("%-8s %9s %11s %11s %11s %11s %9s"
          % ("threads", "total s", "geometry", "reproject", "design mat",
             "kernels", "rest s"), flush=True)
    print("-" * 76, flush=True)
    for threads, snap in rows:
        ktimes = [snap[k]["excl"] for k in kernels if k in snap]
        ksum = sum(ktimes)
        total = snap[root]["incl"]
        cells = ["%11.3f" % t for t in ktimes]
        print("%-8d %9.2f %s %11.3f %9.2f"
              % (threads, total, " ".join(cells), ksum, total - ksum), flush=True)

    if len(rows) > 1:
        t1 = rows[0][1][root]["incl"]
        n1 = rows[0][0]
        assert n1 == 1, "sweep must start at 1 thread to get a baseline"
        print("", flush=True)
        print("Amdahl: parallel fraction p implied by each cell "
              "(t_n/t_1 = (1-p) + p/n)", flush=True)
        for threads, snap in rows[1:]:
            tn = snap[root]["incl"]
            p = (1.0 - tn / t1) / (1.0 - 1.0 / threads)
            print("  n=%2d: t_n/t_1 = %.3f  ->  p = %.3f  (serial floor %.2f s)"
                  % (threads, tn / t1, p, (1.0 - p) * t1), flush=True)
    return None


# ---------------------------------------------------------------------------
# subcommand: bandwidth
# ---------------------------------------------------------------------------

def _triad_worker(rank, barrier, queue, n_elem, seconds):
    """STREAM-like triad in one process; reports (iters, elapsed)."""
    a = np.zeros(n_elem, dtype=np.float64)
    b = np.ones(n_elem, dtype=np.float64)
    c = np.full(n_elem, 0.5, dtype=np.float64)
    scalar = 3.0

    # one untimed pass so pages are faulted in before the barrier
    np.multiply(c, scalar, out=a)
    a += b

    barrier.wait()
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < seconds:
        np.multiply(c, scalar, out=a)
        a += b
        iters += 1
    elapsed = time.perf_counter() - t0
    queue.put((rank, iters, elapsed))
    return None


def run_bandwidth(args):
    """Measure aggregate memory bandwidth vs concurrent process count."""
    from multiprocessing import get_context

    ctx = get_context("fork")
    sweep = [int(x) for x in args.procs.split(",") if x.strip()]
    n_elem = args.array_mb * 1024 * 1024 // 8
    arr_bytes = n_elem * 8
    # per triad iteration: read c, write a, then read a + read b + write a.
    # Counted as 5 array touches (the fused-multiply pass moves 2, the add 3).
    bytes_per_iter = 5 * arr_bytes

    print("", flush=True)
    print("=" * 92, flush=True)
    print("BANDWIDTH: aggregate STREAM-triad throughput vs concurrent processes", flush=True)
    print("=" * 92, flush=True)
    print("array %s x3 per process, %.1f s per cell, %d cores available"
          % (fmt_bytes(arr_bytes), args.seconds, len(os.sched_getaffinity(0))), flush=True)
    print("", flush=True)
    print("%-8s %12s %14s %12s" % ("procs", "GB/s total", "GB/s per proc", "scaling"),
          flush=True)
    print("-" * 50, flush=True)

    base = None
    for n_proc in sweep:
        barrier = ctx.Barrier(n_proc)
        queue = ctx.Queue()
        procs = [ctx.Process(target=_triad_worker,
                             args=(i, barrier, queue, n_elem, args.seconds))
                 for i in range(n_proc)]
        for p in procs:
            p.start()
        results = [queue.get() for _ in range(n_proc)]
        for p in procs:
            p.join()

        total_bytes = sum(it * bytes_per_iter for _, it, _ in results)
        span = max(el for _, _, el in results)
        gbs = total_bytes / span / 1e9
        if base is None:
            base = gbs
        print("%-8d %12.1f %14.2f %11.2fx"
              % (n_proc, gbs, gbs / n_proc, gbs / base), flush=True)
    return None


# ---------------------------------------------------------------------------
# subcommand: io
# ---------------------------------------------------------------------------

def _evict(path):
    """Drop a file's page cache without root via posix_fadvise(DONTNEED)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return None


def run_io(args):
    """Time cold vs warm FITS reads for the four files of each epoch."""
    from sdo_clv_pipeline.sdo_io import read_data, read_header

    epochs = resolve_epochs(args, args.epochs)

    print("", flush=True)
    print("=" * 92, flush=True)
    print("IO: cold vs warm FITS read (page cache evicted with posix_fadvise)", flush=True)
    print("=" * 92, flush=True)
    print("caveat: fadvise(DONTNEED) drops local page cache. It does not reproduce "
          "Ceph-side\n         cache or metadata latency, so cold numbers here are a "
          "lower bound on a truly\n         cold production read.", flush=True)
    print("", flush=True)
    print("%-34s %10s %10s %10s %10s %8s"
          % ("file", "size", "cold s", "cold MB/s", "warm s", "speedup"), flush=True)
    print("-" * 86, flush=True)

    cold_total = 0.0
    warm_total = 0.0
    bytes_total = 0
    for epoch in epochs:
        for path in epoch:
            size = os.path.getsize(path)
            _evict(path)
            t0 = time.perf_counter()
            read_header(path)
            read_data(path)
            cold = time.perf_counter() - t0
            t0 = time.perf_counter()
            read_header(path)
            read_data(path)
            warm = time.perf_counter() - t0
            cold_total += cold
            warm_total += warm
            bytes_total += size
            print("%-34s %10s %10.2f %10.1f %10.2f %7.1fx"
                  % (os.path.basename(path)[:34], fmt_bytes(size), cold,
                     size / cold / 1e6, warm, cold / warm), flush=True)

    n_ep = len(epochs)
    print("", flush=True)
    print("per-epoch (4 files, %s): cold %.2f s, warm %.2f s, penalty %.2f s"
          % (fmt_bytes(bytes_total / n_ep), cold_total / n_ep, warm_total / n_ep,
             (cold_total - warm_total) / n_ep), flush=True)
    print("aggregate cold read bandwidth: %.1f MB/s"
          % (bytes_total / cold_total / 1e6), flush=True)
    return None


# ---------------------------------------------------------------------------
# subcommand: pool
# ---------------------------------------------------------------------------

def run_pool(args):
    """Throughput vs worker count for the run_pipe across-epoch pool."""
    from multiprocessing import get_context
    from sdo_clv_pipeline.sdo_process import process_data_set_parallel
    from sdo_clv_pipeline.reproject import bilinear_reproject

    epochs = resolve_epochs(args, args.epochs)
    sweep = [int(x) for x in args.workers.split(",") if x.strip()]
    assert len(epochs) >= max(sweep), \
        ("need epochs >= max workers or the high-worker cells measure spawn "
         "overhead, not throughput (got %d epochs, max %d workers)"
         % (len(epochs), max(sweep)))

    print("", flush=True)
    print("=" * 92, flush=True)
    print("POOL: across-epoch throughput vs worker count (spawn, maxtasksperchild=4)",
          flush=True)
    print("=" * 92, flush=True)
    print("%d epochs per cell, %d cores available. NOTE the first cell warms the "
          "page cache\n      for all later cells, so cell 1 carries the cold-read cost."
          % (len(epochs), len(os.sched_getaffinity(0))), flush=True)
    # single-process per-epoch time, used as the speedup/efficiency baseline so a
    # 1-worker cell (which costs epochs x ~20 s) need not be run every time
    serial = args.serial_seconds
    if serial is None:
        print("no --serial-seconds given; the first cell is the baseline", flush=True)
    else:
        print("serial baseline: %.2f s/epoch (from the `stages` subcommand)"
              % serial, flush=True)
    print("", flush=True)
    print("%-9s %10s %12s %10s %12s %12s"
          % ("workers", "wall s", "s/epoch", "speedup", "efficiency",
             "max child RSS"), flush=True)
    print("-" * 72, flush=True)

    base = serial * len(epochs) if serial is not None else None
    for n_work in sweep:
        datadir = make_scratch_datadir("%s_w%d" % (args.datadir, n_work))
        items = [(c, m, d, a, args.mu_thresh, args.n_rings, datadir)
                 for c, m, d, a in epochs]
        t0 = time.perf_counter()
        with get_context("spawn").Pool(n_work, maxtasksperchild=4) as pool:
            dummy = np.empty((1, 1), dtype=np.float32)
            bilinear_reproject(np.zeros((1, 1), np.float32),
                               np.zeros((1, 1), np.float32),
                               np.zeros((1, 1), np.float32), dummy)
            statuses = pool.starmap(process_data_set_parallel, items, chunksize=1)
        wall = time.perf_counter() - t0
        n_ok = sum(1 for s in statuses if s == "ok")
        if base is None:
            base = wall
        child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        print("%-9d %10.1f %12.2f %10.2fx %11.0f%% %12s   (%d/%d ok)"
              % (n_work, wall, wall / len(epochs), base / wall,
                 100.0 * (base / wall) / n_work, fmt_bytes(child_rss),
                 n_ok, len(epochs)), flush=True)
        shutil.rmtree(datadir, ignore_errors=True)
    return None


# ---------------------------------------------------------------------------
# subcommand: reproject
# ---------------------------------------------------------------------------

def run_reproject(args):
    """Sweep the AIA->HMI pixel-mapping stride: cost vs accuracy.

    compute_pixel_mapping evaluates the expensive SkyCoord transform on a coarse
    subgrid and bilinearly upsamples. ``stride`` is the only knob, and its cost is
    quadratic in 1/stride. The accuracy reference is the full-resolution
    high-level transform the fast path was derived from, restricted to the on-disk
    pixels the pipeline actually keeps (mu >= mu_thresh).
    """
    from sdo_clv_pipeline.sdo_image import SDOImage
    from sdo_clv_pipeline.reproject import (compute_pixel_mapping,
                                            compute_pixel_mapping_highlevel)

    epoch = resolve_epochs(args, 1)[0]
    con_file, _, _, aia_file = epoch
    con = SDOImage(con_file)
    aia = SDOImage(aia_file)
    con.calc_geometry()
    shape = con.image.shape
    on_disk = con.mu >= args.mu_thresh

    print("", flush=True)
    print("=" * 92, flush=True)
    print("REPROJECT: compute_pixel_mapping stride sweep (cost vs accuracy)", flush=True)
    print("=" * 92, flush=True)
    print("destination grid %d x %d, %d on-disk pixels (mu >= %.2f)"
          % (shape[0], shape[1], int(on_disk.sum()), args.mu_thresh), flush=True)

    t0 = time.perf_counter()
    ref_x, ref_y = compute_pixel_mapping_highlevel(aia.wcs, con.wcs, shape)
    ref_time = time.perf_counter() - t0
    print("full-resolution high-level oracle: %.2f s" % ref_time, flush=True)
    print("", flush=True)
    print("%-8s %10s %12s %14s %14s"
          % ("stride", "time s", "vs stride 4", "max dev px", "p99.9 dev px"), flush=True)
    print("-" * 62, flush=True)

    base = None
    for stride in [int(x) for x in args.strides.split(",") if x.strip()]:
        t0 = time.perf_counter()
        src_x, src_y = compute_pixel_mapping(aia.wcs, con.wcs, shape, stride=stride)
        dt = time.perf_counter() - t0
        dev = np.hypot(src_x[on_disk] - ref_x[on_disk], src_y[on_disk] - ref_y[on_disk])
        dev = dev[np.isfinite(dev)]
        if stride == 4:
            base = dt
        rel = "%11.2fx" % (base / dt) if base else "%12s" % "-"
        print("%-8d %10.3f %s %14.4f %14.4f"
              % (stride, dt, rel, np.max(dev), np.percentile(dev, 99.9)), flush=True)
    return None


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, needs_data=True):
        if needs_data:
            sp.add_argument("--fitsdir", default="/mnt/ceph/users/mpalumbo/sdo_data/")
            sp.add_argument("--globexp", default="")
            sp.add_argument("--index", type=int, default=0,
                            help="index of the first matched epoch to use")
            sp.add_argument("--con"); sp.add_argument("--mag")
            sp.add_argument("--dop"); sp.add_argument("--aia")
        sp.add_argument("--mu-thresh", type=float, default=0.1, dest="mu_thresh")
        sp.add_argument("--n-rings", type=int, default=10, dest="n_rings")
        sp.add_argument("--fit-cbs", action="store_true", dest="fit_cbs",
                        help="fit the 11-term CBS system (sibling-repo config)")
        sp.add_argument("--classify-moat", action="store_true", dest="classify_moat",
                        help="enable moat classification (off in run_pipe)")
        sp.add_argument("--datadir", default="/tmp/sdo_bench",
                        help="scratch output dir (wiped on start)")
        return None

    sp = sub.add_parser("stages", help="single-epoch hierarchical profile")
    add_common(sp)
    sp.add_argument("--repeats", type=int, default=3)
    sp.add_argument("--threads", type=int, default=1, help="numba threads")
    sp.add_argument("--trace-alloc", action="store_true", dest="trace_alloc",
                    help="also record peak allocation per stage (adds overhead)")
    sp.set_defaults(func=run_stages)

    sp = sub.add_parser("threads", help="SDO_THREADS sweep / Amdahl fraction")
    add_common(sp)
    sp.add_argument("--repeats", type=int, default=2)
    sp.add_argument("--threads", dest="threads_sweep", default="1,2,4,8,16")
    sp.set_defaults(func=run_threads)

    sp = sub.add_parser("bandwidth", help="memory bandwidth vs process count")
    add_common(sp, needs_data=False)
    sp.add_argument("--procs", default="1,2,4,8,16,31")
    sp.add_argument("--array-mb", type=int, default=128, dest="array_mb")
    sp.add_argument("--seconds", type=float, default=3.0)
    sp.set_defaults(func=run_bandwidth)

    sp = sub.add_parser("io", help="cold vs warm FITS read cost")
    add_common(sp)
    sp.add_argument("--epochs", type=int, default=2)
    sp.set_defaults(func=run_io)

    sp = sub.add_parser("reproject", help="AIA pixel-mapping stride cost vs accuracy")
    add_common(sp)
    sp.add_argument("--strides", default="2,4,8,16,32")
    sp.set_defaults(func=run_reproject)

    sp = sub.add_parser("pool", help="across-epoch pool throughput vs workers")
    add_common(sp)
    sp.add_argument("--epochs", type=int, default=32)
    sp.add_argument("--workers", default="1,4,8,16")
    sp.add_argument("--serial-seconds", type=float, default=None,
                    dest="serial_seconds",
                    help="single-process s/epoch baseline (from `stages`); avoids "
                         "paying for a 1-worker cell")
    sp.set_defaults(func=run_pool)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)
    return None


if __name__ == "__main__":
    main()
