import numpy as np
import pdb
from .sdo_image import *

def compute_disk_results(mjd, flat_mu, flat_int, flat_v_corr, flat_v_rot,
                         flat_ld, flat_iflat, flat_w_quiet, flat_w_active,
                         flat_abs_mag, mu_thresh, k_hat_con):
    valid = flat_mu >= mu_thresh
    all_pixels = np.nansum(valid)
    all_light = np.nansum(flat_int[valid])

    denom = np.nansum(flat_int[valid])
    v_hat_di = np.nansum(flat_int[valid] * flat_v_corr[valid]) / denom
    v_phot_di = np.nansum(flat_v_rot[valid] * (flat_int - k_hat_con * flat_ld)[valid] * flat_w_active[valid]) / denom
    v_quiet_di = np.nansum(flat_v_corr[valid] * flat_int[valid] * flat_w_quiet[valid]) / np.nansum(flat_int[valid] * flat_w_quiet[valid])
    v_cbs_di = v_hat_di - v_quiet_di

    mag_unsigned = np.nansum(flat_abs_mag[valid] * flat_int[valid]) / denom

    count = np.nansum(valid)
    avg_int = np.nansum(flat_int[valid]) / count
    avg_int_flat = np.nansum(flat_iflat[valid]) / count

    return [mjd, np.nan, np.nan, np.nan,
            all_pixels, all_light,
            v_hat_di, v_phot_di, v_quiet_di, v_cbs_di,
            mag_unsigned, avg_int, avg_int_flat]

def compute_region_only_results(mjd, flat_mu, flat_int, flat_v_corr, flat_v_rot,
                                flat_ld, flat_iflat, flat_abs_mag, flat_w_quiet, 
                                flat_w_active, flat_reg, region_codes, 
                                mu_thresh, k_hat_con):
    valid_mask = flat_mu >= mu_thresh

    # aggregate sums by region
    regions = np.array(region_codes)
    reg_map = {r:i for i,r in enumerate(region_codes)}
    reg_idx = np.array([reg_map.get(r,-1) for r in flat_reg])
    valid = np.logical_and(valid_mask, reg_idx >= 0)
    grp = reg_idx[valid]
    M = len(regions)

    sum_vhat = np.bincount(grp, weights=flat_int[valid] * flat_v_corr[valid], minlength=M)
    sum_vphot = np.bincount(grp, weights=flat_v_rot[valid] * (flat_int - k_hat_con * flat_ld)[valid] * flat_w_active[valid], minlength=M)
    sum_int = np.bincount(grp, weights=flat_int[valid], minlength=M)
    sum_iflat = np.bincount(grp, weights=flat_iflat[valid], minlength=M)
    sum_mag = np.bincount(grp, weights=flat_abs_mag[valid] * flat_int[valid], minlength=M)
    sum_pix = np.bincount(grp, weights=valid.astype(int)[valid], minlength=M)

    quiet_idx = np.where(regions == quiet_sun_code)[0][0]
    q_valid = valid & flat_w_quiet
    grp_q = reg_idx[q_valid]
    sum_vquiet = np.bincount(grp_q, weights=flat_v_corr[q_valid] * flat_int[q_valid], minlength=M)
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
                           flat_reg, region_codes, mu_thresh, n_rings, k_hat_con):
    bins = np.linspace(mu_thresh, 1.0, n_rings)
    bin_idx = np.clip(np.digitize(flat_mu, bins) - 1, 0, n_rings-2)
    valid_mask = (flat_mu >= mu_thresh)

    # aggregate sums by (bin, region)
    regions = np.array(region_codes)
    reg_map = {r:i for i,r in enumerate(region_codes)}
    reg_idx = np.array([reg_map.get(r,-1) for r in flat_reg])
    valid = valid_mask & (reg_idx>=0)
    grp = bin_idx[valid] * len(regions) + reg_idx[valid]
    M = (n_rings-1) * len(regions)

    sum_vhat = np.bincount(grp, weights=flat_int[valid]*flat_v_corr[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_vphot = np.bincount(grp, weights=flat_v_rot[valid] * (flat_int-k_hat_con*flat_ld)[valid] * flat_w_active[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_int = np.bincount(grp, weights=flat_int[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_iflat = np.bincount(grp, weights=flat_iflat[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_mag = np.bincount(grp, weights=flat_abs_mag[valid]*flat_int[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_pix = np.bincount(grp, weights=valid.astype(int)[valid], minlength=M).reshape(n_rings-1,len(regions))

    # quiet-sun sums
    quiet_idx = np.where(regions == quiet_sun_code)[0][0]
    q_valid = valid & flat_w_quiet
    grp_q = bin_idx[q_valid] * len(regions) + reg_idx[q_valid]
    sum_vquiet_flat = np.bincount(grp_q, weights=flat_v_corr[q_valid]*flat_int[q_valid], minlength=M).reshape(n_rings-1,len(regions))
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
    return data