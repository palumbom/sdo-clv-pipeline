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
    iso = get_date(con_file).isoformat()
    print(">>> Running epoch %s " % iso, flush=True)

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

    # create arrays to hold velocity, magnetic field, and pixel fraction results
    results = []

    # calculate number of pixels and total light
    all_pixels = np.nansum(con.mu >= mu_thresh)
    all_light = np.nansum(con.image * (con.mu >= mu_thresh))

    # quiet‐sun mask & continuum scaling
    w_quiet = mask.is_quiet_sun()
    k_hat_con = np.nansum(con.image * con.ldark * w_quiet) / np.nansum(con.ldark**2 * w_quiet)

    # flatten per-pixel arrays
    flat_mu = mask.mu.ravel()
    flat_reg = mask.regions.ravel()
    flat_int = con.image.ravel()
    flat_v_corr = dop.v_corr.ravel()
    flat_v_rot = dop.v_rot.ravel()
    flat_abs_mag = np.abs(mag.B_obs).ravel()
    flat_iflat = con.iflat.ravel()
    flat_ld = con.ldark.ravel()
    flat_w_quiet = w_quiet.ravel()

    # get disk-integrated quantitites
    valid = flat_mu >= mu_thresh
    denom = np.nansum(flat_int[valid])

    v_hat_di = np.nansum(flat_int[valid] * flat_v_corr[valid]) 
    v_hat_di /= denom

    v_phot_di = np.nansum(flat_v_rot[valid] * (flat_int - k_hat_con * flat_ld)[valid] * ~flat_w_quiet[valid]) 
    v_phot_di /= denom

    v_quiet_di = np.nansum(flat_v_corr[valid] * flat_int[valid] * flat_w_quiet[valid])
    v_quiet_di /= np.nansum(flat_int[valid] * flat_w_quiet[valid])

    v_cbs_di = v_hat_di - v_quiet_di

    # do unsigned mag flux 
    mag_unsigned = np.nansum(flat_abs_mag[valid] * flat_int[valid]) / denom

    # do avg intensities
    denom = np.nansum(valid)
    avg_int = np.nansum(flat_int[valid])
    avg_int /= denom
    avg_int_flat = np.nansum(flat_iflat[valid])
    avg_int_flat /= denom

    # append full-disk results
    results.append([mjd, np.nan, np.nan, np.nan, all_pixels, 
                    all_light, v_hat_di, v_phot_di, v_quiet_di, 
                    v_cbs_di, mag_unsigned, avg_int, avg_int_flat])

    # bins & region mapping
    bins = np.linspace(mu_thresh, 1.0, n_rings)
    bin_idx = np.digitize(flat_mu, bins) - 1
    regions = region_codes
    region_map = {r:i for i,r in enumerate(regions)}
    reg_idx = np.array([region_map.get(r, -1) for r in flat_reg])

    n_bins = n_rings - 1
    valid = (flat_mu >= mu_thresh) & (bin_idx >= 0) & (bin_idx < n_bins) & (reg_idx >= 0)
    grp = bin_idx[valid] * len(regions) + reg_idx[valid]
    M = n_bins * len(regions)

    # compute sums
    sum_vhat = np.bincount(grp, weights=flat_int[valid] * flat_v_corr[valid], minlength=M)
    sum_vphot = np.bincount(grp, weights=flat_v_rot[valid] * (flat_int - k_hat_con * flat_ld)[valid] * ~flat_w_quiet[valid], minlength=M)
    sum_int = np.bincount(grp, weights=flat_int[valid], minlength=M)
    sum_iflat = np.bincount(grp, weights=flat_iflat[valid], minlength=M)
    sum_mag = np.bincount(grp, weights=flat_abs_mag[valid] * flat_int[valid], minlength=M)
    sum_pix = np.bincount(grp, weights=valid.astype(int)[valid], minlength=M)

    # quiet-sun sums
    qidx = valid & flat_w_quiet
    grp_q = bin_idx[qidx] * len(regions) + reg_idx[qidx]
    sum_vquiet = np.bincount(grp_q, weights=flat_v_corr[qidx] * flat_int[qidx], minlength=M)
    sum_int_q = np.bincount(grp_q, weights=flat_int[qidx], minlength=M)

    # reshape
    sum_vhat = sum_vhat.reshape(n_bins, len(regions))
    sum_vphot = sum_vphot.reshape(n_bins, len(regions))
    sum_int = sum_int.reshape(n_bins, len(regions))
    sum_iflat = sum_iflat.reshape(n_bins, len(regions))
    sum_mag = sum_mag.reshape(n_bins, len(regions))
    sum_vquiet = sum_vquiet.reshape(n_bins, len(regions))
    sum_int_q = sum_int_q.reshape(n_bins, len(regions))
    sum_pix = sum_pix.reshape(n_bins, len(regions))

    # get quiet index
    quiet_idx = np.argmax(np.array(regions) == quiet_sun_code)

    # loop over mu bins
    for j in range(n_bins):
        lo_mu = bins[j]
        hi_mu = bins[j+1]

        # get v_quiet for the mu bin for vconv
        dq_mu_bin = sum_int_q[j, quiet_idx]
        v_q_mu_bin = (sum_vquiet[j, quiet_idx] / dq_mu_bin) if dq_mu_bin > 0 else 0.0

        # loop over regions
        for ir, r in enumerate(regions):
            denom_int = sum_int[j, ir]
            denom_pix = sum_pix[j, ir]
            if denom_int == 0:
                results.append([mjd, r, lo_mu, hi_mu, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                continue

            # calculate values
            pix_frac = denom_pix / all_pixels
            light_frac = denom_int / all_light

            v_hat = sum_vhat[j, ir] / denom_int
            v_phot = sum_vphot[j, ir] / denom_int

            dq = sum_int_q[j, ir]
            v_q = (sum_vquiet[j, ir] / dq) if dq > 0 else 0.0
            v_conv = v_hat - v_q_mu_bin

            mag_u = sum_mag[j, ir] / denom_int
            avg_i = sum_int[j, ir] / denom_pix 
            avg_if = sum_iflat[j, ir] / denom_pix 

            # append the velocity results
            results.append([mjd, r, lo_mu, hi_mu, pix_frac, 
                            light_frac, v_hat, v_phot, v_q, 
                            v_conv, mag_u, avg_i, avg_if])

    # write to disk
    write_results_to_file(fname2, results)

    # do some memory cleanup
    del con
    del mag
    del dop
    del aia
    del bins
    del valid
    del denom
    del results
    del regions
    del flat_mu
    del flat_reg
    del flat_int
    del flat_v_corr
    del flat_v_rot
    del flat_abs_mag
    del flat_iflat
    del flat_ld
    del flat_w_quiet
    gc.collect() 
    
    # end the timer
    end_time = time.perf_counter()

    # report success and return
    print("\t >>> Run successfully in %s seconds" % str(end_time - start_time), flush=True)
    return None