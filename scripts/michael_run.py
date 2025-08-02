# external package imports
import numpy as np
import pandas as pd
import os, pdb, glob, time, argparse
from tqdm import tqdm, trange
from os.path import exists, split, isdir, getsize

# internal package imports
from sdo_clv_pipeline.paths import root
from sdo_clv_pipeline.sdo_io import *
from sdo_clv_pipeline.sdo_process import *
from sdo_clv_pipeline.sdo_io import * 
from sdo_clv_pipeline.sdo_plot import *
import sdo_clv_pipeline

# returns four sorted lists with the path to each of the file types
# fits_dir = os.path.abspath("/mnt/ceph/users/mpalumbo/sdo_data")
fits_dir = os.path.join(root, "data", "fits")

globexp ="2014*0*"
files = organize_IO(fits_dir, clobber=True, globexp=globexp)
con_files, mag_files, dop_files, aia_files = files

globdir = globexp.replace("*","")
datadir = os.path.join(root, "data", globdir)
if not isdir(datadir):
    os.mkdir(datadir)

# loop over data
print()
for i in range(len(con_files)):
    # define the paths where files live
    con_file = con_files[i]
    mag_file = mag_files[i]
    dop_file = dop_files[i]
    aia_file = aia_files[i]

    process_data_set(con_file, mag_file, dop_file, aia_file,
                     mu_thresh=0.1, n_rings=10, suffix=None, 
                     datadir=datadir, plot_moat=False)