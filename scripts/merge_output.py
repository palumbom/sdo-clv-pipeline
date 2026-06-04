import numpy as np
import pandas as pd
import os, pdb, glob, time, argparse, logging
from os.path import exists, split, isdir, getsize

# bring functions into scope
from sdo_clv_pipeline.paths import root
from sdo_clv_pipeline.sdo_io import *
from sdo_clv_pipeline.sdo_process import *
from sdo_clv_pipeline.logging_setup import configure_logging

logger = logging.getLogger(__name__)

def main():
    configure_logging()
    # figure out data directories
    files = []
    datadir = os.path.join(root, "data")

    # find the files to combine
    for file in os.listdir(datadir):
        file = os.path.join(datadir, file)
        if not isdir(file):
            continue
        else:
            # for f in glob.glob(file + "/" + "*.csv"):
            for f in glob.glob(os.path.join(file, "*.csv")):
                files.append(f)

    # names for output files
    fname1 = os.path.join(datadir, "thresholds.csv")
    fname2 = os.path.join(datadir, "region_output.csv")

    # headers for output files (shared schema; defined once in sdo_io)
    header1 = header_thresholds
    header2 = header_region

    # delete old files if they exists
    fileset = (fname1, fname2)
    clean_output_directory(*fileset)

    # create the files with headers
    create_file(fname1, header1)
    create_file(fname2, header2)

    # now loop over files to combine
    for file in fileset:
        # find files to merge into output
        in_fname = []
        for f in files:
            if os.path.split(f)[-1] == os.path.split(file)[-1]:
                in_fname.append(f)

        # read in each file sequentially and append its data to output file
        for f in in_fname:
            data = pd.read_csv(f)
            data.to_csv(file, mode="a", index=False, header=False)
        logger.info("merged %d files into %s", len(in_fname), file)

    return None


if __name__ == "__main__":
    main()

