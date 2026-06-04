"""SDO processing pipeline entry points for per-epoch reductions."""

import numpy as np
import matplotlib.pyplot as plt
import gc, os, re, pdb, csv, glob, time, argparse, logging
from astropy.time import Time
from os.path import exists, split, isdir, getsize
from multiprocessing import get_context
import multiprocessing as mp

# bring functions into scope
from .paths import root
from .sdo_io import *
from .sdo_vels import *
from .sdo_image import *
from .quality import *

# multiprocessing imports
from multiprocessing import get_context
import multiprocessing as mp

logger = logging.getLogger(__name__)

# per-epoch outcome codes; STATUS_OK marks success, the SKIP_* codes name the
# reason an epoch was skipped so callers can tally a run summary
status_ok = "ok"
skip_invalid_file = "invalid_file"
skip_quality = "quality"
skip_limb_dark = "limb_darkening"
skip_doppler = "dopplergram"
skip_regions = "region_id"
skip_unknown = "unknown"

def is_quality_data(sdo_image):
    """Return True when the SDO image passes the QUALITY flag check."""
    return sdo_image.quality == 0

def reduce_sdo_images(con_file, mag_file, dop_file, aia_file, mu_thresh=0.1, fit_cbs=False, **kwargs):
    """Load, validate, and reduce a set of SDO images for a single epoch.

    Public API. Returns a 5-tuple ``(con, mag, dop, aia, mask)`` on success or
    ``None`` when the epoch is skipped; in the skip case a specific diagnostic
    has already been printed by ``_reduce_sdo_images``. External callers rely on
    this exact contract, so the skip reason is intentionally not exposed here.
    """
    images, _reason = _reduce_sdo_images(con_file, mag_file, dop_file, aia_file,
                                         mu_thresh=mu_thresh, fit_cbs=fit_cbs, **kwargs)
    return images


def _reduce_sdo_images(con_file, mag_file, dop_file, aia_file, mu_thresh=0.1, fit_cbs=False, **kwargs):
    """Load, validate, and reduce a set of SDO images for a single epoch.

    This handles geometry setup, limb darkening, doppler/magnetogram corrections,
    and region classification.

    Returns ``(images, reason)`` where on success ``images`` is the 5-tuple
    ``(con, mag, dop, aia, mask)`` and ``reason`` is None, and on a skip
    ``images`` is None and ``reason`` is one of the module-level skip_* codes.
    A specific diagnostic is logged at each skip point.
    """
    # assertions
    assert exists(con_file), "missing continuum file: " + str(con_file)
    assert exists(mag_file), "missing magnetogram file: " + str(mag_file)
    assert exists(dop_file), "missing dopplergram file: " + str(dop_file)
    assert exists(aia_file), "missing aia file: " + str(aia_file)

    # get the datetime
    iso = get_date(con_file).isoformat()

    # make SDOImage instances; load one at a time so a bad file is named
    images = {}
    for label, fname in (("CON", con_file), ("MAG", mag_file),
                         ("DOP", dop_file), ("AIA", aia_file)):
        try:
            images[label] = SDOImage(fname)
        except OSError as e:
            logger.warning("Invalid file, skipping %s: %s %s (%r)", iso, label, fname, e)
            return None, skip_invalid_file
    con, mag, dop, aia = images["CON"], images["MAG"], images["DOP"], images["AIA"]

    # check QUALITY flags: any fatal bit skips the epoch; tolerable (soft) bits are
    # processed but warned about, and recorded downstream via the combined quality_flag
    fatal = [(label, images[label]) for label in ("CON", "MAG", "DOP", "AIA")
             if not is_tolerable_quality(images[label].quality)]
    if fatal:
        for label, im in fatal:
            logger.warning("Quality skip %s: %s (%s)",
                           iso, format_quality(im.instrument, im.quality), label)
        return None, skip_quality
    flagged = [(label, images[label]) for label in ("CON", "MAG", "DOP", "AIA")
               if images[label].quality != 0]
    for label, im in flagged:
        logger.warning("Quality warning %s: %s (%s) [tolerated, processing]",
                       iso, format_quality(im.instrument, im.quality), label)

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
    except Exception:
        logger.exception("Limb darkening fit failed, skipping %s", iso)
        return None, skip_limb_dark

    # correct magnetogram for foreshortening
    mag.correct_magnetogram()

    # calculate differential rot., meridional circ., obs. vel, grav. redshift, cbs
    dop.correct_dopplergram(fit_cbs=fit_cbs)

    # check that the dopplergram correction went well
    v_rot_max = float(np.nanmax(np.abs(dop.v_rot)))
    if v_rot_max < 1000.0:
        logger.warning("Dopplergram correction failed %s: max|v_rot|=%.3f m/s < 1000.0, skipping",
                       iso, v_rot_max)
        return None, skip_doppler

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
    except Exception:
        logger.exception("Region identification failed, skipping %s", iso)
        return None, skip_regions

    return (con, mag, dop, aia, mask), None


def process_data_set_parallel(con_file, mag_file, dop_file, aia_file, mu_thresh, n_rings, datadir):
    """Wrapper for multiprocessing that forwards to process_data_set.

    Returns the per-epoch status so pool.starmap carries it back to the parent
    for the end-of-run summary.
    """
    return process_data_set(con_file, mag_file, dop_file, aia_file,
                            mu_thresh=mu_thresh, n_rings=n_rings,
                            suffix=str(mp.current_process().pid), datadir=datadir,
                            plot_moat=False, classify_moat=False)

def process_data_set(con_file, mag_file, dop_file, aia_file, 
                     mu_thresh, n_rings=10, suffix=None, 
                     datadir=None, **kwargs):
    """Run the full processing pipeline and persist per-epoch outputs."""
    iso = get_date(con_file).isoformat()
    logger.info("Running epoch %s", iso)

    # start the timer
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

    try:
        # reduce the data set; skip cleanly (no disk touched) on a known issue
        images, reason = _reduce_sdo_images(con_file, mag_file, dop_file, aia_file, **kwargs)
        if images is None:
            return reason
        con, mag, dop, aia, mask = images

        # single combined quality flag (bitwise-OR across instruments); nonzero means
        # the epoch carried tolerated soft QUALITY bits and can be masked downstream
        quality_flag = combine_quality(con.quality, mag.quality, dop.quality, aia.quality)

        # now that we have results, create the output files if needed
        for file in (fname1, fname2):
            if not exists(file):
                create_file(file)

        # get the MJD of the obs
        mjd = Time(con.date_obs).mjd

        # write the limb darkening parameters, velocities, etc. to disk
        write_results_to_file(fname1, mjd, mask.aia_thresh, *aia.ld_coeffs,
                              mask.con_thresh1, mask.con_thresh2, *con.ld_coeffs,
                              np.nanmax(dop.v_cbs),
                              np.nanmin(dop.v_obs), np.nanmax(dop.v_obs), np.nanmean(dop.v_obs),
                              np.nanmin(dop.v_rot), np.nanmax(dop.v_rot), np.nanmean(dop.v_rot),
                              np.nanmin(dop.v_mer), np.nanmax(dop.v_mer), np.nanmean(dop.v_mer),
                              quality_flag)

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
        flat_w_active = np.logical_not(flat_w_quiet)

        # calculate k_hat
        w_quiet = mask.is_quiet_sun()
        k_hat_con = np.nansum(con.image * con.ldark * w_quiet) / np.nansum(con.ldark**2 * w_quiet)

        # shared per-pixel weighted products (computed once, reused by all three
        # aggregations below instead of being recomputed in each)
        p_vhat, p_vphot, p_mag = shared_products(flat_int, flat_v_corr, flat_v_rot,
                                                 flat_ld, flat_w_active, flat_abs_mag, k_hat_con)

        # create arrays to hold velocity, magnetic field, and pixel fraction results
        results = []

        # calculate disk-integrataed quantities
        results.append(compute_disk_results(mjd, flat_mu, flat_int, flat_v_corr,
                                            flat_v_rot, flat_ld, flat_iflat,
                                            flat_w_quiet, flat_w_active, flat_abs_mag,
                                            mu_thresh, k_hat_con,
                                            p_vhat=p_vhat, p_vphot=p_vphot, p_mag=p_mag))

        # calculate velocities for regions, not binning by mu
        results.extend(compute_region_only_results(mjd, flat_mu, flat_int,
                                                   flat_v_corr, flat_v_rot,
                                                   flat_ld, flat_iflat,
                                                   flat_abs_mag, flat_w_quiet,
                                                   flat_w_active,
                                                   flat_reg, region_codes,
                                                   mu_thresh, k_hat_con,
                                                   p_vhat=p_vhat, p_vphot=p_vphot, p_mag=p_mag))

        # calculate disk-resovled quantities
        results.extend(compute_region_results(mjd, flat_mu, flat_int, flat_v_corr,
                                              flat_v_rot, flat_ld, flat_iflat,
                                              flat_abs_mag, flat_w_quiet,
                                              flat_w_active, flat_reg,
                                              region_codes, mu_thresh,
                                              n_rings, k_hat_con,
                                              p_vhat=p_vhat, p_vphot=p_vphot, p_mag=p_mag))

        # tag every region row with the per-epoch quality flag, then write to disk
        for row in results:
            row.append(quality_flag)
        write_results_to_file(fname2, results)

        # do some memory cleanup (success path: all names are bound)
        del con
        del mag
        del dop
        del aia
        del flat_mu
        del flat_ld
        del flat_reg
        del flat_int
        del flat_v_rot
        del flat_iflat
        del flat_v_corr
        del flat_abs_mag
        del flat_w_quiet

        # end the timer
        end_time = time.perf_counter()

        # report success and return
        logger.info("Epoch %s reduced successfully in %.2f seconds", iso, end_time - start_time)
        return status_ok
    except Exception:
        logger.exception("Epoch %s failed unexpectedly", iso)
        return skip_unknown
    finally:
        gc.collect()
