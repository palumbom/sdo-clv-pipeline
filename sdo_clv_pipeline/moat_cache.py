"""Reduce SDO epochs once and cache the arrays the moat kernel needs.

The moat step is trivially cheap, but it sits behind a ~15-60 s per-epoch
reduction (FITS loads, geometry, reproject, limb-darkening, Doppler correction).
This module runs that reduction once per epoch and stores only the arrays
detect_moats consumes, so parameter sweeps in a notebook reload in well under a
second. See docs/superpowers/specs/2026-07-01-moat-tuning-design.md.

Import-heavy (pulls sdo_process -> sdo_image -> numba/sunpy); intended for
interactive/harness use, not the numba-free test job.
"""

import os
import numpy as np
from pathlib import Path
import logging

from astropy.time import Time

from .sdo_io import find_data, get_date
from .sdo_process import _reduce_sdo_images

logger = logging.getLogger(__name__)

# large arrays -> Ceph bulk storage per site filesystem policy
default_cache_dir = Path("/mnt/ceph/users/mpalumbo/sdo_data/moat_cache")

# the arrays detect_moats() consumes, in call order
kernel_keys = ("v_corr", "mu", "lon", "con_image", "mag_image",
               "invalid_mask", "umbra_mask", "penumbra_mask")


def reduce_epoch_for_moats(con_file, mag_file, dop_file, aia_file, mu_thresh=0.1):
    """Reduce one epoch and extract the arrays + metadata the moat kernel needs.

    Returns a dict, or None if the epoch was skipped by the reduction (bad file,
    quality, failed fit -- diagnostics already logged by _reduce_sdo_images).
    """
    images, reason = _reduce_sdo_images(con_file, mag_file, dop_file, aia_file,
                                        mu_thresh=mu_thresh)
    if images is None:
        logger.warning("Epoch skipped during reduction (%s): %s", reason, con_file)
        return None
    con, mag, dop, aia, mask = images

    # off-disk / low-mu pixels, matching identify_regions' invalid_mask
    invalid_mask = np.logical_or(con.mu <= mu_thresh, np.isnan(con.mu))

    # float64 for the physical fields (guard the known float32 downcast trap);
    # intensity/field stay native
    return {
        "v_corr": np.asarray(dop.v_corr, dtype=np.float64),
        "mu": np.asarray(con.mu, dtype=np.float64),
        "lon": np.asarray(dop.lon.value, dtype=np.float64),
        "con_image": con.image,
        "mag_image": mag.image,
        "invalid_mask": invalid_mask.astype(bool),
        "umbra_mask": mask.is_umbra().astype(bool),
        "penumbra_mask": mask.is_penumbra().astype(bool),
        "mjd": float(Time(con.date_obs).mjd),
        "iso": get_date(con_file).isoformat(),
        "mu_thresh": float(mu_thresh),
        "con_file": str(con_file),
        "mag_file": str(mag_file),
        "dop_file": str(dop_file),
        "aia_file": str(aia_file),
    }


def _cache_path(iso, cache_dir):
    return Path(cache_dir) / f"moat_epoch_{iso}.npz"


def cache_epoch(con_file, mag_file, dop_file, aia_file, mu_thresh=0.1,
                cache_dir=default_cache_dir, clobber=False):
    """Reduce one epoch and write its arrays to a per-epoch .npz keyed by ISO time.

    Returns the cache Path, or None if the epoch was skipped. Skips the reduction
    entirely when a cache file already exists and clobber is False.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    iso = get_date(con_file).isoformat()
    path = _cache_path(iso, cache_dir)
    if path.exists() and not clobber:
        logger.info("Cache hit, skipping reduction: %s", path)
        return path

    data = reduce_epoch_for_moats(con_file, mag_file, dop_file, aia_file,
                                  mu_thresh=mu_thresh)
    if data is None:
        return None

    np.savez_compressed(path, **data)
    logger.info("Wrote moat cache: %s", path)
    return path


def load_cached_epoch(iso_or_path, cache_dir=default_cache_dir):
    """Load a cached epoch into a plain dict of arrays/scalars.

    Accepts an ISO timestamp (resolved against cache_dir) or a direct path.
    """
    path = Path(iso_or_path)
    if not path.exists():
        path = _cache_path(str(iso_or_path), cache_dir)
    if not path.exists():
        raise FileNotFoundError("no cached epoch at %s" % path)

    with np.load(path, allow_pickle=False) as npz:
        out = {}
        for key in npz.files:
            arr = npz[key]
            # unwrap 0-d scalars (mjd, mu_thresh, iso, filenames)
            out[key] = arr.item() if arr.ndim == 0 else arr
    return out


def cache_epochs(fitsdir, globexp="", cache_dir=default_cache_dir,
                 mu_thresh=0.1, clobber=False):
    """Cache every matched epoch under fitsdir/globexp.

    Reuses the pipeline's file-matching (find_data). Per-epoch failures are
    logged and skipped, not fatal, so one bad epoch doesn't abort the batch.
    Returns the list of successfully written cache Paths.
    """
    con_files, mag_files, dop_files, aia_files = find_data(fitsdir, globexp=globexp)
    logger.info("Caching %d matched epochs from %s (glob=%r)",
                len(con_files), fitsdir, globexp)

    written = []
    for con_f, mag_f, dop_f, aia_f in zip(con_files, mag_files, dop_files, aia_files):
        try:
            path = cache_epoch(con_f, mag_f, dop_f, aia_f,
                               mu_thresh=mu_thresh, cache_dir=cache_dir,
                               clobber=clobber)
            if path is not None:
                written.append(path)
        except Exception:
            logger.exception("Failed to cache epoch: %s", con_f)
    return written
