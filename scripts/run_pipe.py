import numpy as np
import os, pdb, glob, time, argparse, logging
# force a non-interactive backend before pyplot is imported, so this batch entry
# point works headless (no DISPLAY) regardless of the configured GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import exists, split, isdir, getsize
from collections import Counter

# bring functions into scope
from sdo_clv_pipeline.paths import root
from sdo_clv_pipeline.sdo_io import *
from sdo_clv_pipeline.sdo_process import *
from sdo_clv_pipeline.reproject import *
from sdo_clv_pipeline.parallel import set_compute_threads
from sdo_clv_pipeline.logging_setup import configure_logging


def _worker_init(log_level):
    """Pool-worker initializer: set up logging and pin numba to one thread.

    One epoch per worker process, so each worker must stay single-threaded to
    avoid oversubscribing the box (N processes x M numba threads). The package
    __init__ already defaults to 1 thread on import; this is the explicit guard.
    """
    configure_logging(log_level)
    set_compute_threads(1)
    return None

# multiprocessing imports
from multiprocessing import get_context
import multiprocessing as mp

logger = logging.getLogger(__name__)

# use style
plt.style.use(str(root) + "/" + "my.mplstyle"); plt.ioff()

def print_run_summary(statuses):
    """Log an end-of-run tally of processed vs. skipped epochs by reason."""
    tally = Counter(statuses)
    total = sum(tally.values())
    n_ok = tally.get(status_ok, 0)
    lines = ["RUN SUMMARY: %d epochs, %d processed, %d skipped"
             % (total, n_ok, total - n_ok)]
    for reason in (skip_quality, skip_limb_dark, skip_doppler,
                   skip_regions, skip_invalid_file, skip_unknown):
        lines.append("    skipped[%s] = %d" % (reason, tally.get(reason, 0)))
    logger.info("\n".join(lines))
    return None

def get_parser_args():
    # initialize argparser
    parser = argparse.ArgumentParser(description="Analyze SDO data")
    parser.add_argument("--fitsdir", type=str, default="/mnt/ceph/users/mpalumbo/sdo_data/")
    parser.add_argument("--clobber", action="store_true", default=False)
    parser.add_argument("--globexp", type=str, default="")
    parser.add_argument("--mu-thresh", type=float, default=0.1, dest="mu_thresh",
                        help="mask pixels with mu below this (default: %(default)s)")
    parser.add_argument("--n-rings", type=int, default=10, dest="n_rings",
                        help="number of mu rings for disk-resolved aggregation (default: %(default)s)")
    parser.add_argument("--max-epochs", type=int, default=None, dest="max_epochs",
                        help="process at most this many epochs (for testing)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="logging level (DEBUG, INFO, WARNING, ERROR)")

    # parse the command line arguments
    return parser.parse_args()

def main():
    # make raw data dir if it does not exist
    if not isdir(os.path.join(root, "data")):
        os.mkdir(os.path.join(root, "data"))

    # sort out input/output data files
    args = get_parser_args()
    log_level = args.log_level
    configure_logging(log_level)
    globdir = args.globexp.replace("*","")
    # fitsdir = os.path.join(root, "data", "fits")
    files = organize_IO(args.fitsdir, clobber=args.clobber, globexp=args.globexp)
    con_files, mag_files, dop_files, aia_files = files

    # optionally cap the number of epochs (smoke testing)
    if args.max_epochs is not None:
        con_files = con_files[:args.max_epochs]
        mag_files = mag_files[:args.max_epochs]
        dop_files = dop_files[:args.max_epochs]
        aia_files = aia_files[:args.max_epochs]

    # get output datadir
    datadir = os.path.join(root, "data", globdir)
    if not isdir(datadir):
        os.mkdir(datadir)

    # mu threshold / number of mu rings (from CLI)
    n_rings = args.n_rings
    mu_thresh = args.mu_thresh
    plot = False

    # get number of cpus
    try:
        from os import sched_getaffinity
        logger.info("OS claims %d CPUs are available", len(sched_getaffinity(0)))
        ncpus = len(sched_getaffinity(0)) - 1
        # ncpus = 33 - 1
    except Exception as e:
        # ncpus = np.min([len(con_files), mp.cpu_count()])
        logger.warning("Could not query CPU affinity (%s: %s); falling back to 1 process",
                       type(e).__name__, e)
        ncpus = 1

    # ncpus = 1

    # process the data either in parallel or serially
    if ncpus > 1:
        # make tmp directory
        tmpdir = os.path.join(datadir, "tmp")
        if not isdir(tmpdir):
            os.mkdir(tmpdir)

        # remove any stale per-worker temp files from a previous (possibly
        # crashed) run; otherwise they would be glob'd into this run's stitch
        # and silently corrupt the output
        for stale in glob.glob(os.path.join(tmpdir, "thresholds_*")) + \
                     glob.glob(os.path.join(tmpdir, "region_output_*")) + \
                     glob.glob(os.path.join(tmpdir, "feature_output_*")):
            os.remove(stale)

        # prepare arguments for starmap
        items = []
        for i in range(len(con_files)):
            items.append((con_files[i], mag_files[i], dop_files[i], aia_files[i], mu_thresh, n_rings, datadir))

        # run in parellel
        logger.info("Processing %d epochs with %d processes", len(con_files), ncpus)
        t0 = time.time()
        pids = []
        with get_context("spawn").Pool(ncpus, maxtasksperchild=4,
                                       initializer=_worker_init,
                                       initargs=(log_level,)) as pool:
            # get PIDs of workers
            for child in mp.active_children():
                pids.append(child.pid)
        
            # warm up jit
            dummy_dst = np.empty((1,1), dtype=np.float32)
            bilinear_reproject(np.zeros((1,1),np.float32),
                               np.zeros((1,1),np.float32),
                               np.zeros((1,1),np.float32),
                               dummy_dst)

            # run the analysis; keep the per-epoch statuses for the summary.
            # chunksize=1: epochs are heavy and uneven, so dynamic dispatch
            # load-balances better than static chunking (which would also collide
            # with maxtasksperchild=4, respawning a worker every chunk boundary).
            statuses = pool.starmap(process_data_set_parallel, items, chunksize=1)

        # find the output data sets
        outfiles1 = glob.glob(os.path.join(tmpdir,"thresholds_*"))
        outfiles2 = glob.glob(os.path.join(tmpdir,"region_output_*"))
        outfiles3 = glob.glob(os.path.join(tmpdir,"feature_output_*"))

        # stitch them together on the main process, then remove the temp files
        # so a later non-clobber rerun cannot pick up stale per-worker output
        delete = True
        stitch_output_files(os.path.join(datadir, "thresholds.csv"), outfiles1, delete=delete)
        stitch_output_files(os.path.join(datadir, "region_output.csv"), outfiles2, delete=delete)
        stitch_output_files(os.path.join(datadir, "feature_output.csv"), outfiles3, delete=delete)

        # log run time
        logger.info("Parallel run complete in %.1f seconds", time.time() - t0)
        print_run_summary(statuses)
    else:
        # run serially
        logger.info("Processing %d epochs on a single process", len(con_files))
        t0 = time.time()
        statuses = []
        for i in range(len(con_files)):
            statuses.append(
                process_data_set(con_files[i], mag_files[i], dop_files[i], aia_files[i],
                                 mu_thresh=mu_thresh, n_rings=n_rings, datadir=datadir,
                                 plot_moat=False, classify_moat=False))

        # log run time
        logger.info("Serial run complete in %.1f seconds", time.time() - t0)
        print_run_summary(statuses)
    return None

if __name__ == "__main__":
    main()
