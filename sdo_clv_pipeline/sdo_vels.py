"""Velocity aggregation helpers for disk- and region-level summaries."""

import numpy as np
import pdb
from scipy import ndimage
from .sdo_image import *


def _region_index(flat_reg, region_codes):
    """Map a flat region-code array to dense indices [0, len(region_codes)).

    Vectorized replacement for the per-pixel ``dict.get`` loop: builds a small
    lookup table indexed by region code and gathers in one pass. Codes not in
    ``region_codes`` (including NaN, mapped to 0) become -1, matching the old
    ``reg_map.get(r, -1)`` semantics.
    """
    maxc = max(region_codes)
    lut = np.full(maxc + 1, -1, dtype=np.int64)
    for i, r in enumerate(region_codes):
        lut[r] = i
    codes = np.nan_to_num(flat_reg, nan=0.0).astype(np.int64)
    np.clip(codes, 0, maxc, out=codes)
    return lut[codes]


def shared_products(flat_int, flat_v_corr, flat_v_rot, flat_ld,
                    flat_w_active, flat_abs_mag, k_hat_con):
    """Per-pixel weighted products reused by all three aggregation functions.

    Computing these once (in process_data_set) instead of inside each of
    compute_disk/region_only/region_results avoids recomputing the same
    16.8M-element products three times. Returns (p_vhat, p_vphot, p_mag) where
    p_vhat = int*v_corr (also used for the quiet-sun numerator), p_vphot is the
    photometric term, p_mag = |B|*int.
    """
    p_vhat = flat_int * flat_v_corr
    p_vphot = flat_v_rot * (flat_int - k_hat_con * flat_ld) * flat_w_active
    p_mag = flat_abs_mag * flat_int
    return p_vhat, p_vphot, p_mag

def compute_disk_results(mjd, flat_mu, flat_int, flat_v_corr, flat_v_rot,
                         flat_ld, flat_iflat, flat_w_quiet, flat_w_active,
                         flat_abs_mag, mu_thresh, k_hat_con,
                         p_vhat=None, p_vphot=None, p_mag=None):
    """Compute disk-integrated velocity and intensity metrics.

    Parameters
    ----------
    mjd : float
        Observation time in modified Julian date.
    flat_mu, flat_int, flat_v_corr, flat_v_rot : array-like
        Flattened arrays of mu, intensity, corrected velocity, and rotation velocity.
    flat_ld, flat_iflat : array-like
        Flattened limb-darkening model and flattened intensity.
    flat_w_quiet, flat_w_active : array-like
        Boolean weights for quiet and active regions.
    flat_abs_mag : array-like
        Absolute magnetogram values.
    mu_thresh : float
        Minimum mu to include in the aggregation.
    k_hat_con : float
        Continuum scaling factor for the photometric term.
    """
    if p_vhat is None:
        p_vhat, p_vphot, p_mag = shared_products(flat_int, flat_v_corr, flat_v_rot,
                                                 flat_ld, flat_w_active, flat_abs_mag, k_hat_con)

    valid = flat_mu >= mu_thresh

    # boolean indexing materializes a new array each time, and the masked
    # intensity sum / valid-pixel count are each reused several times below.
    # Compute them once; the operands and operation order are unchanged, so
    # every result is bit-identical to recomputing them inline.
    fi = flat_int[valid]
    fwq = flat_w_quiet[valid]
    light_sum = np.nansum(fi)
    n_valid = np.nansum(valid)

    all_pixels = n_valid
    all_light = light_sum

    denom = light_sum
    v_hat_di = np.nansum(p_vhat[valid]) / denom
    v_phot_di = np.nansum(p_vphot[valid]) / denom
    v_quiet_di = np.nansum(p_vhat[valid] * fwq) / np.nansum(fi * fwq)
    v_cbs_di = v_hat_di - v_quiet_di

    mag_unsigned = np.nansum(p_mag[valid]) / denom

    avg_int = light_sum / n_valid
    avg_int_flat = np.nansum(flat_iflat[valid]) / n_valid

    return [mjd, np.nan, np.nan, np.nan,
            all_pixels, all_light,
            v_hat_di, v_phot_di, v_quiet_di, v_cbs_di,
            mag_unsigned, avg_int, avg_int_flat]

def compute_region_only_results(mjd, flat_mu, flat_int, flat_v_corr, flat_v_rot,
                                flat_ld, flat_iflat, flat_abs_mag, flat_w_quiet,
                                flat_w_active, flat_reg, region_codes,
                                mu_thresh, k_hat_con,
                                p_vhat=None, p_vphot=None, p_mag=None,
                                reg_idx=None):
    """Compute region-aggregated metrics across the full disk."""
    if p_vhat is None:
        p_vhat, p_vphot, p_mag = shared_products(flat_int, flat_v_corr, flat_v_rot,
                                                 flat_ld, flat_w_active, flat_abs_mag, k_hat_con)

    valid_mask = flat_mu >= mu_thresh

    # aggregate sums by region
    regions = np.array(region_codes)
    if reg_idx is None:
        reg_idx = _region_index(flat_reg, region_codes)
    valid = np.logical_and(valid_mask, reg_idx >= 0)
    grp = reg_idx[valid]
    M = len(regions)

    sum_vhat = np.bincount(grp, weights=p_vhat[valid], minlength=M)
    sum_vphot = np.bincount(grp, weights=p_vphot[valid], minlength=M)
    sum_int = np.bincount(grp, weights=flat_int[valid], minlength=M)
    sum_iflat = np.bincount(grp, weights=flat_iflat[valid], minlength=M)
    sum_mag = np.bincount(grp, weights=p_mag[valid], minlength=M)
    sum_pix = np.bincount(grp, weights=valid.astype(int)[valid], minlength=M)

    quiet_idx = np.where(regions == quiet_sun_code)[0][0]
    q_valid = valid & flat_w_quiet
    grp_q = reg_idx[q_valid]
    sum_vquiet = np.bincount(grp_q, weights=p_vhat[q_valid], minlength=M)
    sum_int_q = np.bincount(grp_q, weights=flat_int[q_valid], minlength=M)

    total_pixels = np.nansum(valid_mask)
    total_light = np.nansum(flat_int[valid_mask])
    pix_frac = sum_pix / total_pixels
    light_frac = sum_int / total_light

    sum_int_safe = np.where(sum_int > 0, sum_int, 1)
    sum_int_q_safe = np.where(sum_int_q > 0, sum_int_q, 1)
    sum_pix_safe = np.where(sum_pix > 0, sum_pix, 1)

    v_hat = sum_vhat / sum_int_safe
    v_phot = sum_vphot / sum_int_safe
    v_q = sum_vquiet / sum_int_q_safe
    v_conv = np.where(v_hat != 0, v_hat - v_q[quiet_idx], 0)

    mag_u = sum_mag / sum_int_safe
    avg_i = sum_int / sum_pix_safe
    avg_if = sum_iflat / sum_pix_safe

    regs = np.array(regions)
    mjd_arr = np.full_like(regs, mjd, dtype=float)
    nan_arr = np.full_like(regs, np.nan, dtype=float)
    data = np.vstack([mjd_arr, regs, nan_arr, nan_arr,
                      pix_frac, light_frac, v_hat, 
                      v_phot, v_q, v_conv, mag_u, 
                      avg_i, avg_if]).T.tolist()
    return data

def compute_region_results(mjd, flat_mu, flat_int, flat_v_corr, flat_v_rot,
                           flat_ld, flat_iflat, flat_abs_mag, flat_w_quiet, flat_w_active,
                           flat_reg, region_codes, mu_thresh, n_rings, k_hat_con,
                           p_vhat=None, p_vphot=None, p_mag=None,
                           reg_idx=None):
    """Compute region-aggregated metrics in mu rings."""
    if p_vhat is None:
        p_vhat, p_vphot, p_mag = shared_products(flat_int, flat_v_corr, flat_v_rot,
                                                 flat_ld, flat_w_active, flat_abs_mag, k_hat_con)

    bins = np.linspace(mu_thresh, 1.0, n_rings)
    bin_idx = np.clip(np.digitize(flat_mu, bins) - 1, 0, n_rings-2)
    valid_mask = (flat_mu >= mu_thresh)

    # aggregate sums by (bin, region)
    regions = np.array(region_codes)
    if reg_idx is None:
        reg_idx = _region_index(flat_reg, region_codes)
    valid = valid_mask & (reg_idx>=0)
    grp = bin_idx[valid] * len(regions) + reg_idx[valid]
    M = (n_rings-1) * len(regions)

    sum_vhat = np.bincount(grp, weights=p_vhat[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_vphot = np.bincount(grp, weights=p_vphot[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_int = np.bincount(grp, weights=flat_int[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_iflat = np.bincount(grp, weights=flat_iflat[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_mag = np.bincount(grp, weights=p_mag[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_pix = np.bincount(grp, weights=valid.astype(int)[valid], minlength=M).reshape(n_rings-1,len(regions))

    # quiet-sun sums
    quiet_idx = np.where(regions == quiet_sun_code)[0][0]
    q_valid = valid & flat_w_quiet
    grp_q = bin_idx[q_valid] * len(regions) + reg_idx[q_valid]
    sum_vquiet_flat = np.bincount(grp_q, weights=p_vhat[q_valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_int_q_flat = np.bincount(grp_q, weights=flat_int[q_valid], minlength=M).reshape(n_rings-1,len(regions))

    # compute metrics
    total_pixels = np.nansum(valid_mask)
    total_light = np.nansum(flat_int[valid_mask])
    pix_frac = sum_pix/total_pixels
    light_frac = sum_int/total_light

    # avoid division by zero
    sum_int_safe = np.where(sum_int > 0, sum_int, 1)
    sum_int_q_safe = np.where(sum_int_q_flat > 0, sum_int_q_flat, 1)
    sum_pix_safe = np.where(sum_pix > 0, sum_pix, 1)

    # get velocities
    v_hat = sum_vhat/sum_int_safe
    v_phot = sum_vphot/sum_int_safe
    v_q = sum_vquiet_flat/sum_int_q_safe
    v_q_mu = v_q[:,quiet_idx][:,None]
    v_conv = np.where(v_hat != 0, v_hat - v_q_mu, 0)

    mag_u = sum_mag/sum_int_safe
    avg_i = sum_int/sum_pix_safe
    avg_if = sum_iflat/sum_pix_safe

    # build rows
    lo_mu = bins[:-1]
    hi_mu = bins[1:]
    bin_idxs = np.repeat(lo_mu, len(regions)), np.repeat(hi_mu, len(regions))
    region_list = np.tile(regions, len(lo_mu))

    data = np.vstack([np.full_like(v_phot.ravel(), mjd), region_list, bin_idxs[0],
                      bin_idxs[1], pix_frac.ravel(), light_frac.ravel(),
                      v_hat.ravel(), v_phot.ravel(), v_q.ravel(), v_conv.ravel(),
                      mag_u.ravel(), avg_i.ravel(), avg_if.ravel()]).T.tolist()
    # also surface the per-ring quiet-Sun reference (the scalar subtracted to form
    # v_conv) and the ring edges, so the feature catalog can subtract the identical
    # reference per pixel without recomputing it.
    return data, v_q[:, quiet_idx], bins

def compute_feature_catalog(mjd, regions, flat_int, flat_iflat, flat_mu,
                            flat_abs_mag, flat_abs_mag_rad, flat_pix_area,
                            flat_lon, flat_lat, p_vhat, p_vphot,
                            flat_v_quiet_ref=None,
                            region_codes_to_label=(umbrae_code, penumbrae_code)):
    """Per-feature catalog of connected umbra/penumbra blobs.

    Each connected component (8-connectivity) of a region-type mask is one
    feature; field strength can then be split downstream without re-running the
    pipeline. Velocities and field means use the same intensity-weighted
    estimator as the region curves; area and unsigned flux use the per-pixel
    solar area (microhemispheres). ``regions`` is the 2D region-code map (NaN
    off-disk / sub-threshold); all other per-pixel inputs are flattened in the
    same C-order.

    Two magnetic-field conventions are recorded, matching how the literature uses
    each (see Haywood et al. 2016 vs Yeo et al. 2013 / Milbourne et al. 2019):
    ``flat_abs_mag`` is the line-of-sight field ``|B_obs|`` (the Haywood unsigned-
    flux proxy, matching the pipeline's ``mag_unsigned``), reported in the ``_los``
    columns; ``flat_abs_mag_rad`` is the foreshortening-corrected radial field
    ``|B_obs|/mu`` (the canonical field for magnetic-strength classification),
    reported in the ``_rad`` columns and used for the unsigned flux. NOTE the
    radial field amplifies noise at low mu (x10 at mu=0.1), so ``max_abs_b_rad``
    is spike-prone near the limb.

    ``flat_lon``/``flat_lat`` are plain degree arrays (caller extracts ``.value``
    from the astropy Quantities). ``flat_lat`` is the pipeline's internal
    heliographic colatitude (0=N pole, 90=equator); ``centroid_lat`` is converted
    to standard Stonyhurst latitude in [-90, 90] (lat - 90).

    ``unsigned_flux_rad_g_uhem`` = sum(|B_rad| * pix_area) in Gauss*microhemisphere
    -- proportional to the true vertical flux (multiply by ~3.0e16 cm^2/uHem, i.e.
    1e-6 * 2*pi*R_sun^2, for Maxwells).

    ``flat_v_quiet_ref`` is the per-pixel quiet-Sun velocity reference of each
    pixel's mu-ring (``v_q[ring, quiet_idx]`` from compute_region_results, mapped
    to pixels). When supplied, ``v_quiet`` is its intensity-weighted mean over the
    feature and ``v_conv = v_hat - v_quiet`` -- the exact per-pixel-ring analogue
    of the region-level convective term, correct even when a feature straddles
    rings. When omitted (e.g. unit tests), both columns are NaN.

    Returns one row per feature (quality_flag appended by the caller):
      [mjd, region, feature_id, n_pix, area_uhem, mean_mu_iw, min_mu, max_mu,
       centroid_lon, centroid_lat, mean_abs_b_iw_los, mean_abs_b_aw_los,
       max_abs_b_los, mean_abs_b_iw_rad, mean_abs_b_aw_rad, max_abs_b_rad,
       unsigned_flux_rad_g_uhem, v_hat, v_phot, avg_int, avg_int_flat,
       v_conv, v_quiet]
    """
    corners = ndimage.generate_binary_structure(2, 2)
    rows = []
    for code in region_codes_to_label:
        labels2d, n = ndimage.label(regions == code, structure=corners)
        if n == 0:
            continue
        # restrict to labeled pixels first: features cover <<1% of the disk, so
        # doing the arithmetic on this small subset instead of the full
        # ~16.8M-pixel frame is the difference between ~8s and well under 1s.
        # np.flatnonzero is a single full pass; the gathers below touch only the
        # feature pixels.
        lab = labels2d.ravel()
        idx = np.flatnonzero(lab)
        labs = lab[idx]
        index = np.arange(1, n + 1)
        nb = n + 1

        # subset weights once, then reuse
        i = flat_int[idx]
        b = flat_abs_mag[idx]        # line-of-sight |B_obs|
        b_rad = flat_abs_mag_rad[idx]  # radial |B_obs|/mu
        a = flat_pix_area[idx]
        mu_ = flat_mu[idx]

        # per-label weighted sums; bin 0 is empty (labs >= 1) but kept for index
        # alignment, then sliced off. Feature pixels carry no NaN (sub-threshold
        # pixels were masked to regions=NaN before labeling), so bincount is safe.
        s_pix = np.bincount(labs, minlength=nb)[1:]
        s_int = np.bincount(labs, weights=i, minlength=nb)[1:]
        s_iflat = np.bincount(labs, weights=flat_iflat[idx], minlength=nb)[1:]
        s_area = np.bincount(labs, weights=a, minlength=nb)[1:]
        s_mu_i = np.bincount(labs, weights=mu_ * i, minlength=nb)[1:]
        s_b_i = np.bincount(labs, weights=b * i, minlength=nb)[1:]
        s_b_area = np.bincount(labs, weights=b * a, minlength=nb)[1:]
        s_brad_i = np.bincount(labs, weights=b_rad * i, minlength=nb)[1:]
        s_brad_area = np.bincount(labs, weights=b_rad * a, minlength=nb)[1:]
        s_lon_i = np.bincount(labs, weights=flat_lon[idx] * i, minlength=nb)[1:]
        s_lat_i = np.bincount(labs, weights=flat_lat[idx] * i, minlength=nb)[1:]
        s_vhat = np.bincount(labs, weights=p_vhat[idx], minlength=nb)[1:]
        s_vphot = np.bincount(labs, weights=p_vphot[idx], minlength=nb)[1:]

        # per-label extrema (bincount only sums), on the subset
        min_mu = np.atleast_1d(ndimage.minimum(mu_, labels=labs, index=index))
        max_mu = np.atleast_1d(ndimage.maximum(mu_, labels=labs, index=index))
        max_b_los = np.atleast_1d(ndimage.maximum(b, labels=labs, index=index))
        max_b_rad = np.atleast_1d(ndimage.maximum(b_rad, labels=labs, index=index))

        # intensity-weighted (iw) and area-weighted (aw) means
        mean_mu_iw = s_mu_i / s_int
        mean_abs_b_iw_los = s_b_i / s_int
        mean_abs_b_aw_los = s_b_area / s_area
        mean_abs_b_iw_rad = s_brad_i / s_int
        mean_abs_b_aw_rad = s_brad_area / s_area
        centroid_lon = s_lon_i / s_int
        centroid_lat = s_lat_i / s_int - 90.0  # colatitude -> Stonyhurst latitude
        v_hat = s_vhat / s_int
        v_phot = s_vphot / s_int
        avg_int = s_int / s_pix
        avg_int_flat = s_iflat / s_pix

        # convective term: subtract the intensity-weighted per-pixel quiet-Sun
        # ring reference (matches the region v_conv definition exactly)
        if flat_v_quiet_ref is not None:
            s_vq_i = np.bincount(labs, weights=flat_v_quiet_ref[idx] * i, minlength=nb)[1:]
            v_quiet = s_vq_i / s_int
            v_conv = v_hat - v_quiet
        else:
            v_quiet = np.full(n, np.nan)
            v_conv = np.full(n, np.nan)

        for j in range(n):
            rows.append([mjd, code, int(index[j]), int(s_pix[j]), s_area[j],
                         mean_mu_iw[j], min_mu[j], max_mu[j],
                         centroid_lon[j], centroid_lat[j],
                         mean_abs_b_iw_los[j], mean_abs_b_aw_los[j], max_b_los[j],
                         mean_abs_b_iw_rad[j], mean_abs_b_aw_rad[j], max_b_rad[j],
                         s_brad_area[j], v_hat[j], v_phot[j],
                         avg_int[j], avg_int_flat[j], v_conv[j], v_quiet[j]])
    return rows
