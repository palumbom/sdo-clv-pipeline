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

def reduce_sdo_images(con_file, mag_file, dop_file, aia_file, mu_thresh=0.1, fit_cbs=False, **kwargs):
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
    mag.correct_magnetogram()

    # calculate differential rot., meridional circ., obs. vel, grav. redshift, cbs
    dop.correct_dopplergram(fit_cbs=fit_cbs)

    # check that the dopplergram correction went well
    if np.nanmax(np.abs(dop.v_rot)) < 1000.0:
        print("\t >>> Dopplergram correction failed, skipping " + iso, flush=True)
        return None
    
    # set values to nan for mu less than mu_thresh
    con.mask_low_mu(mu_thresh)
    dop.mask_low_mu(mu_thresh)
    mag.mask_low_mu(mu_thresh)
    aia.mask_low_mu(mu_thresh)

    # identify regions for thresholding
    try:
        # print("About to construct SunMask")
        mask = SunMask(con, mag, dop, aia, **kwargs)
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
                     datadir=None, **kwargs):
    iso = get_date(con_file).isoformat()
    print(">>> Running epoch %s " % iso, flush=True)

    start_time = time.perf_counter()
    #figure out data directories
    if not isdir(datadir): os.mkdir(datadir)

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
        con, mag, dop, aia, mask = reduce_sdo_images(con_file, mag_file, dop_file, aia_file, **kwargs)
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

    # # calculate number of pixels and total light
    # all_pixels = np.nansum(con.mu >= mu_thresh)
    # all_light = np.nansum(con.image * (con.mu >= mu_thresh))

    # flatten per-pixel arrays
    flat_mu = mask.mu.ravel()
    flat_reg = mask.regions.ravel()
    flat_int = con.image.ravel()
    flat_v_corr = dop.v_corr.ravel()
    flat_v_rot = dop.v_rot.ravel()
    flat_abs_mag = np.abs(mag.B_obs).ravel()
    flat_iflat = con.iflat.ravel()
    flat_ld = con.ldark.ravel()
    flat_w_quiet = mask.is_quiet_sun().ravel()

    # calculate k_hat
    w_quiet = mask.is_quiet_sun()
    k_hat_con = np.nansum(con.image * con.ldark * w_quiet) / np.nansum(con.ldark**2 * w_quiet)

    # create arrays to hold velocity, magnetic field, and pixel fraction results
    results = []

    # calculate disk-integrataed quantities
    results.append(compute_disk_results(mjd, flat_mu, flat_int, flat_v_corr,
                                        flat_v_rot, flat_ld, flat_iflat,
                                        flat_w_quiet, flat_abs_mag,
                                        mu_thresh, k_hat_con))

    # calculate disk-resovled quantities
    results.extend(compute_region_results(mjd, flat_mu, flat_int, flat_v_corr,
                                          flat_v_rot, flat_ld, flat_iflat,
                                          flat_abs_mag, flat_w_quiet,
                                          flat_reg, region_codes,
                                          quiet_sun_code,
                                          mu_thresh, n_rings, k_hat_con))

    # write to disk
    write_results_to_file(fname2, results)

    # do some memory cleanup
    # del con
    # del mag
    # del dop
    # del aia
    # del bins
    # del valid
    # del denom
    # del results
    # del regions
    # del flat_mu
    # del flat_reg
    # del flat_int
    # del flat_v_corr
    # del flat_v_rot
    # del flat_abs_mag
    # del flat_iflat
    # del flat_ld
    # del flat_w_quiet
    # del flat_w_active
    # gc.collect() 
    
    # end the timer
    end_time = time.perf_counter()

    # report success and return
    print("\t >>> Run successfully in %s seconds" % str(end_time - start_time), flush=True)
    return None