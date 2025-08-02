import numpy as np
import matplotlib.pyplot as plt
import os, pdb, glob, time, argparse
from os.path import exists, split, isdir, getsize

# bring functions into scope
from sdo_clv_pipeline.paths import root
from sdo_clv_pipeline.sdo_io import *
from sdo_clv_pipeline.sdo_process import *

# multiprocessing imports
from multiprocessing import get_context
import multiprocessing as mp

# use style
plt.style.use(str(root) + "/" + "my.mplstyle"); plt.ioff()

def get_parser_args():
    # initialize argparser
    parser = argparse.ArgumentParser(description="Analyze SDO data")
    parser.add_argument("--fitsdir", type=str, default="/mnt/ceph/users/mpalumbo/sdo_data/")
    parser.add_argument("--clobber", action="store_true", default=False)
    parser.add_argument("--globexp", type=str, default="")

    # parse the command line arguments
    args = parser.parse_args()
    fitsdir = args.fitsdir
    clobber = args.clobber
    globexp = args.globexp
    return fitsdir, clobber, globexp

def main():
    # make raw data dir if it does not exist

    if not isdir(os.path.join(root, "data")):
        os.mkdir(os.path.join(root, "data"))

    # sort out input/output data files
    fitsdir, clobber, globexp = get_parser_args()
    globdir = globexp.replace("*","")
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
        print()
        print(">>> OS claims %s CPUs are available..." % len(sched_getaffinity(0)))
        ncpus = len(sched_getaffinity(0)) - 1
        # ncpus = 33 - 1
    except:
        # ncpus = np.min([len(con_files), mp.cpu_count()])
        ncpus = 1

    # ncpus = 1

    # process the data either in parallel or serially
    if ncpus > 1:
        # make tmp directory
        tmpdir = os.path.join(datadir, "tmp")
        if not isdir(tmpdir):
            os.mkdir(tmpdir)

        # prepare arguments for starmap
        items = []
        for i in range(len(con_files)):
            items.append((con_files[i], mag_files[i], dop_files[i], aia_files[i], mu_thresh, n_rings, datadir))

        # run in parellel
        print(">>> Processing %s epochs with %s processes..." % (len(con_files), ncpus))
        print()
        t0 = time.time()
        pids = []
        with get_context("spawn").Pool(ncpus, maxtasksperchild=4) as pool:
            # get PIDs of workers
            for child in mp.active_children():
                pids.append(child.pid)

            # run the analysis
            pool.starmap(process_data_set_parallel, items, chunksize=4)

        # find the output data sets
        outfiles1 = glob.glob(os.path.join(tmpdir,"thresholds_*"))
        outfiles2 = glob.glob(os.path.join(tmpdir,"region_output_*"))

        # stitch them together on the main process
        delete = False
        stitch_output_files(os.path.join(datadir, "thresholds.csv"), outfiles1, delete=delete)
        stitch_output_files(os.path.join(datadir, "region_output.csv"), outfiles2, delete=delete)

        # print run time
        print("Parallel: --- %s seconds ---" % (time.time() - t0))
    else:
        # run serially
        print(">>> Processing %s epochs on a single process" % len(con_files))
        print()
        t0 = time.time()
        for i in range(len(con_files)):
            process_data_set(con_files[i], mag_files[i], dop_files[i], aia_files[i],
                             mu_thresh=mu_thresh, n_rings=n_rings, datadir=datadir,
                             plot_moat=False)

        # print run time
        print("Serial: --- %s seconds ---" % (time.time() - t0))
    return None

if __name__ == "__main__":
    main()
