import numpy as np
import pdb

def calc_region_mask(mask, region=None, hi_mu=None, lo_mu=None):
    # get mask for region type specified
    if region is None:
        region_mask = True
    else:
        region_mask = np.logical_and((region == mask.regions), (mask.mu >= mask.mu_thresh))

    # get masks for mu annuli
    if ((hi_mu is None) | (lo_mu is None)):
        region_mask *= True
    else:
        assert lo_mu < hi_mu
        cond = [(mask.mu > lo_mu), (mask.mu <= hi_mu) , (mask.mu >= mask.mu_thresh)]
        region_mask *= np.logical_and.reduce(cond)

    return region_mask

def calc_velocities(con, mag, dop, aia, mask, region_mask=None, v_quiet=None):
    # don't bother doing math if there is nothing in the mask
    if (type(region_mask) is np.ndarray) and (~region_mask.any()):
        return 0.0, 0.0, 0.0, 0.0

    # get default region_mask
    if region_mask is None:
        region_mask = (con.mu >= con.mu_thresh)

    # get weights
    w_quiet = mask.is_quiet_sun()
    w_active = ~w_quiet

    # calculate scaling factor for continuum and filtergrams
    k_hat_con = np.nansum(con.image * con.ldark * w_quiet) / np.nansum(con.ldark**2 * w_quiet)

    # calculate velocity terms
    v_hat = np.nansum(dop.v_corr * con.image * region_mask)

    # TODO add v_mer???
    v_phot = np.nansum(dop.v_rot * (con.image - k_hat_con * con.ldark) * w_active * region_mask)

    # divide velocities by the denominator (only calculate it once)
    denom = np.nansum(con.image * region_mask)
    v_hat /= denom
    v_phot /= denom

    if v_quiet is None:
        # calculate v_quiet
        v_quiet = np.nansum(dop.v_corr * con.image * w_quiet * region_mask)
        v_quiet /= np.nansum(con.image * w_quiet * region_mask)

        # get convective velocity by subtracting off other terms
        v_conv = v_hat - v_quiet # - v_phot
        return v_hat, v_phot, v_quiet, v_conv
    else:
        # get convective velocity by subtracting off other terms
        v_conv = v_hat - v_quiet # - v_phot
        return v_hat, v_phot, 0.0, v_conv
    return None

def calc_mag_stats(con, mag, region_mask=True):
    # don't bother doing math if there is nothing in the mask
    if (type(region_mask) is np.ndarray) and (~region_mask.any()):
        return 0.0, 0.0, 0.0

    # get intensity weighted unsigned magnetic field strength
    mag_unsigned = np.nansum(np.abs(mag.B_obs) * con.image * region_mask)

    # divide by the denominator
    mag_unsigned /= np.nansum(con.image * region_mask)

    return mag_unsigned


def calc_int_stats(con, region_mask=True):
    # don't bother doing math if there is nothing in the mask
    if (type(region_mask) is np.ndarray) and (~region_mask.any()):
        return 0.0, 0.0, 0.0

    # get numerator
    avg_int = np.nansum(con.image * region_mask)
    avg_int_flat = np.nansum(con.iflat * region_mask)

    # divide by the denominator
    if (type(region_mask) is not np.ndarray):
        denom = np.nansum(con.mu > con.mu_thresh)
    else:
        denom = np.nansum(region_mask)

    avg_int /= denom
    avg_int_flat /= denom

    return avg_int, avg_int_flat

def compute_disk_results(mjd, flat_mu, flat_int, flat_v_corr, flat_v_rot,
                         flat_ld, flat_iflat, flat_w_quiet, flat_abs_mag,
                         mu_thresh, k_hat_con):
    valid = flat_mu >= mu_thresh
    all_pixels = np.nansum(valid)
    all_light = np.nansum(flat_int[valid])

    flat_w_active = np.logical_not(flat_w_quiet)

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

def compute_region_results(mjd, flat_mu, flat_int, flat_v_corr, flat_v_rot,
                           flat_ld, flat_iflat, flat_abs_mag, flat_w_quiet,
                           flat_reg, region_codes, quiet_sun_code,
                           mu_thresh, n_rings, k_hat_con):
    bins = np.linspace(mu_thresh, 1.0, n_rings)
    bin_idx = np.clip(np.digitize(flat_mu, bins) - 1, 0, n_rings-2)
    valid_mask = (flat_mu >= mu_thresh)
    flat_w_active = np.logical_not(flat_w_quiet)

    # aggregate sums by (bin, region)
    regions = np.array(region_codes)
    reg_map = {r:i for i,r in enumerate(region_codes)}
    reg_idx = np.array([reg_map.get(r,-1) for r in flat_reg])
    valid = valid_mask & (reg_idx>=0)
    grp = bin_idx[valid] * len(regions) + reg_idx[valid]
    M = (n_rings-1)*len(regions)

    sum_vhat = np.bincount(grp, weights=flat_int[valid]*flat_v_corr[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_vphot = np.bincount(grp, weights=flat_v_rot[valid] * (flat_int-k_hat_con*flat_ld)[valid] * flat_w_active[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_int = np.bincount(grp, weights=flat_int[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_iflat = np.bincount(grp, weights=flat_iflat[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_mag = np.bincount(grp, weights=flat_abs_mag[valid]*flat_int[valid], minlength=M).reshape(n_rings-1,len(regions))
    sum_pix = np.bincount(grp, weights=valid.astype(int)[valid], minlength=M).reshape(n_rings-1,len(regions))

    # quiet-sun sums
    quiet_idx = np.where(regions==quiet_sun_code)[0][0]
    q_valid = valid & flat_w_quiet
    grp_q = bin_idx[q_valid]*len(regions)+reg_idx[q_valid]
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