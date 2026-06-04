import numpy as np
import matplotlib.pyplot as plt
import os, pdb, glob, time, argparse, logging
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
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="logging level (DEBUG, INFO, WARNING, ERROR)")

    # parse the command line arguments
    args = parser.parse_args()
    fitsdir = args.fitsdir
    clobber = args.clobber
    globexp = args.globexp
    log_level = args.log_level
    return fitsdir, clobber, globexp, log_level

def main():
    # make raw data dir if it does not exist
    if not isdir(os.path.join(root, "data")):
        os.mkdir(os.path.join(root, "data"))

    # sort out input/output data files
    fitsdir, clobber, globexp, log_level = get_parser_args()
    configure_logging(log_level)
    globdir = globexp.replace("*","")
    # fitsdir = os.path.join(root, "data", "fits")
    files = organize_IO(fitsdir, clobber=clobber, globexp=globexp)
    con_files, mag_files, dop_files, aia_files = files

    # get output datadir
    datadir = os.path.join(root, "data", globdir)
    if not isdir(datadir):
        os.mkdir(datadir)

    # set mu threshold, number of mu rings
    n_rings = 10
    mu_thresh = 0.1
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
                     glob.glob(os.path.join(tmpdir, "region_output_*")):
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

            # run the analysis; keep the per-epoch statuses for the summary
            statuses = pool.starmap(process_data_set_parallel, items, chunksize=4)

        # find the output data sets
        outfiles1 = glob.glob(os.path.join(tmpdir,"thresholds_*"))
        outfiles2 = glob.glob(os.path.join(tmpdir,"region_output_*"))

        # stitch them together on the main process, then remove the temp files
        # so a later non-clobber rerun cannot pick up stale per-worker output
        delete = True
        stitch_output_files(os.path.join(datadir, "thresholds.csv"), outfiles1, delete=delete)
        stitch_output_files(os.path.join(datadir, "region_output.csv"), outfiles2, delete=delete)

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
