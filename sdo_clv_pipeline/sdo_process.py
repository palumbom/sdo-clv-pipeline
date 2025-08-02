import numpy as np
import matplotlib.pyplot as plt
import gc, os, re, pdb, csv, glob, time, argparse
from astropy.time import Time
from os.path import exists, split, isdir, getsize
from multiprocessing import get_context
import multiprocessing as mp

# bring functions into scope
from .paths import root
from .sdo_io import *
from .sdo_vels import *
from .sdo_image import *

# multiprocessing imports
from multiprocessing import get_context
import multiprocessing as mp

def is_quality_data(sdo_image):
    return sdo_image.quality == 0

def reduce_sdo_images(con_file, mag_file, dop_file, aia_file, mu_thresh=0.1, fit_cbs=False, plot_moat=True):
    # assertions
    assert exists(con_file)
    assert exists(mag_file)
    assert exists(dop_file)
    assert exists(aia_file)

    # get the datetime
    iso = get_date(con_file).isoformat()

    # make SDOImage instances
    try:
        con = SDOImage(con_file)
        mag = SDOImage(mag_file)
        dop = SDOImage(dop_file)
        aia = SDOImage(aia_file)
    except OSError:
        print("\t >>> Invalid file, skipping " + iso, flush=True)
        return None

    # check for data quality issue
    if not all(list(map(is_quality_data, [con, mag, dop, aia]))):
        print("\t >>> Data quality issue, skipping " + iso, flush=True)
        return None

    # calculate geometries
    dop.calc_geometry()
    con.inherit_geometry(dop)
    mag.inherit_geometry(dop)

    # interpolate aia image onto hmi image scale and inherit geometry
    aia.rescale_to_hmi(con)

    # calculate limb darkening/brightening in continuum map and filtergram
    try:
        con.calc_limb_darkening()
        aia.calc_limb_darkening()
    except:
        print("\t >>> Limb darkening fit failed, skipping " + iso, flush=True)
        return None

    # correct magnetogram for foreshortening
    # print("correct mag")
    mag.correct_magnetogram()

    # calculate differential rot., meridional circ., obs. vel, grav. redshift, cbs
    # print("correct dop")
    dop.correct_dopplergram(fit_cbs=fit_cbs)

    # check that the dopplergram correction went well
    # print("checking")
    if np.nanmax(np.abs(dop.v_rot)) < 1000.0:
        print("\t >>> Dopplergram correction failed, skipping " + iso, flush=True)
        return None
    
    # print("set to nan")
    # set values to nan for mu less than mu_thresh
    con.mask_low_mu(mu_thresh)
    dop.mask_low_mu(mu_thresh)
    mag.mask_low_mu(mu_thresh)
    aia.mask_low_mu(mu_thresh)

    # print("mask")

    # identify regions for thresholding
    try:
    # print("About to construct SunMask")
        mask = SunMask(con, mag, dop, aia, plot_moat=plot_moat)
        mask.mask_low_mu(mu_thresh)
    except:
        print("\t >>> Region identification failed, skipping " + iso, flush=True)
        return None

    return con, mag, dop, aia, mask


def process_data_set_parallel(con_file, mag_file, dop_file, aia_file, mu_thresh, n_rings, datadir):
    process_data_set(con_file, mag_file, dop_file, aia_file,
                     mu_thresh=mu_thresh, n_rings=n_rings,
                     suffix=str(mp.current_process().pid), datadir=datadir,
                     plot_moat=False)
    return None


def process_data_set(con_file, mag_file, dop_file, aia_file, 
                     mu_thresh, n_rings=10, suffix=None, 
                     datadir=None, plot_moat=True):

    print(">>> Running epoch %s " % get_date(con_file).isoformat(), flush=True)

    start_time = time.perf_counter()
    #figure out data directories
    if not isdir(datadir):
        os.mkdir(datadir)

    # name output files
    if suffix is None:
        fname1 = os.path.join(datadir, "thresholds.csv")
        fname2 = os.path.join(datadir, "region_output.csv")
    else:
        # make tmp directory
        tmpdir = os.path.join(datadir, "tmp")

        # filenames
        fname1 = os.path.join(tmpdir, "thresholds_" + suffix + ".csv")
        fname2 = os.path.join(tmpdir, "region_output_" + suffix + ".csv")

    # check if the files exist, create otherwise
    for file in (fname1, fname2):
            if not exists(file):
                create_file(file)

    # reduce the data set
    try:
        con, mag, dop, aia, mask = reduce_sdo_images(con_file, mag_file, dop_file, aia_file, plot_moat=plot_moat)
    except:
        print("\t >>> Epoch %s reduction failed for unknown reasons :(" % iso, flush=True)
        return None

    # get the MJD of the obs
    mjd = Time(con.date_obs).mjd

    # write the limb darkening parameters, velocities, etc. to disk
    write_results_to_file(fname1, mjd, mask.aia_thresh, *aia.ld_coeffs,
                          mask.con_thresh1, mask.con_thresh2, *con.ld_coeffs,
                          np.nanmax(dop.v_cbs),
                          np.nanmin(dop.v_obs), np.nanmax(dop.v_obs), np.nanmean(dop.v_obs),
                          np.nanmin(dop.v_rot), np.nanmax(dop.v_rot), np.nanmean(dop.v_rot),
                          np.nanmin(dop.v_mer), np.nanmax(dop.v_mer), np.nanmean(dop.v_mer))

    # create arrays to hold velocity magnetic fiel, and pixel fraction results
    results = []

    # calculate number of pixels and total light
    all_pixels = np.nansum(con.mu >= mu_thresh)
    all_light = np.nansum(con.image * (con.mu >= mu_thresh))

    # calculate disk-integrated velocities, mag field, and intensity
    vels = calc_velocities(con, mag, dop, aia, mask)
    mags = calc_mag_stats(con, mag)
    ints = calc_int_stats(con)

    # append full-disk results
    results.append([mjd, np.nan, np.nan, np.nan, all_pixels, all_light, *vels, mags, *ints])

    # loop over the mu annuli
    mu_grid = np.linspace(mu_thresh, 1.0, n_rings)

    regions = [1, 2, 2.5, 3, 4, 5, 6, 8, 9]
    for j in range(n_rings-1):
        # mu values for annuli
        lo_mu=mu_grid[j]
        hi_mu=mu_grid[j+1]

        # compute the region mask
        region_mask = calc_region_mask(mask, region=None, hi_mu=hi_mu, lo_mu=lo_mu)

        # compute quiet-sun velocity in mu annulus
        v_quiet = np.nansum(dop.v_corr * con.image * mask.is_quiet_sun() * region_mask)
        v_quiet /= np.nansum(con.image * mask.is_quiet_sun() * region_mask)

        # loop over unique region identifiers
        for k in regions:
            # compute the region mask
            region_mask = calc_region_mask(mask, region=k, hi_mu=hi_mu, lo_mu=lo_mu)

            # get total pix and light
            pixels = np.nansum(region_mask)/all_pixels
            light = np.nansum(region_mask * con.image)/all_light

            if ((pixels == 0.0) | (light == 0.0)):
                vels = [0.0, 0.0, 0.0, 0.0]
                mags = 0.0
                ints = [0.0, 0.0]
                results.append([mjd, k, lo_mu, hi_mu, pixels, light, *vels, mags, *ints])
                continue

            # compute velocity components in each mu annulus by region
            if k != 4:
                # case where region is quiet sun, return zero for v_quiet
                vels = calc_velocities(con, mag, dop, aia, mask, region_mask=region_mask, v_quiet=v_quiet)
            else:
                # case where region is quiet sun, return nonzero v_quiet
                vels = calc_velocities(con, mag, dop, aia, mask, region_mask=region_mask, v_quiet=None)

            # calculate mag and intensity stats
            mags = calc_mag_stats(con, mag, region_mask=region_mask)
            ints = calc_int_stats(con, region_mask=region_mask)

            # append the velocity results
            results.append([mjd, k, lo_mu, hi_mu, pixels, light, *vels, mags, *ints])

    # write to disk
    write_results_to_file(fname2, results)

    # do some memory cleanup
    del con
    del mag
    del dop
    del aia
    del mask
    del vels
    del mags
    del ints
    del results
    del regions
    del mu_grid
    del v_quiet
    del region_mask
    gc.collect() 

    end_time = time.perf_counter()

    # report success and return
    print("\t >>> Run successfully in %s seconds" % str(end_time - start_time), flush=True)
    return None
