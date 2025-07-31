import numpy as np
import pandas as pd
import os, pdb, glob, time, argparse
from os.path import exists, split, isdir, getsize

# bring functions into scope
from sdo_clv_pipeline.paths import root
from sdo_clv_pipeline.sdo_io import *
from sdo_clv_pipeline.sdo_process import *

def main():
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

    # headers for output files
    header1 = ["mjd", "aia_thresh", "a_aia", "b_aia", "c_aia",
               "hmi_thresh1", "hmi_thresh2", "a_hmi", "b_hmi", "c_hmi",
               "vel_cbs_off",
               "min_vel_sat", "max_vel_sat", "avg_vel_sat",
               "min_vel_rot", "max_vel_rot", "avg_vel_rot",
               "min_vel_mer", "max_vel_mer", "avg_vel_mer"]
    header2 = ["mjd", "region", "lo_mu", "hi_mu", "pixel_frac", "light_frac", "v_hat", "v_phot", "v_quiet", "v_conv", "mag_unsigned", "avg_int", "avg_int_flat"]

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

    return None


if __name__ == "__main__":
    main()

