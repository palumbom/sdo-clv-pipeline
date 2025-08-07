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
    rows = []
    bins = np.linspace(mu_thresh, 1.0, n_rings)
    bin_idx = np.digitize(flat_mu, bins) - 1

    region_map = {r: i for i, r in enumerate(region_codes)}
    reg_idx = np.array([region_map.get(r, -1) for r in flat_reg])

    n_bins = n_rings - 1
    valid = (flat_mu >= mu_thresh) & (bin_idx >= 0) & (bin_idx < n_bins) & (reg_idx >= 0)
    grp = bin_idx[valid] * len(region_codes) + reg_idx[valid]
    M = n_bins * len(region_codes)

    sum_vhat = np.bincount(grp, weights=flat_int[valid] * flat_v_corr[valid], minlength=M)
    sum_vphot = np.bincount(grp, weights=flat_v_rot[valid] * (flat_int - k_hat_con * flat_ld)[valid] * ~flat_w_quiet[valid], minlength=M)
    sum_int = np.bincount(grp, weights=flat_int[valid], minlength=M)
    sum_iflat = np.bincount(grp, weights=flat_iflat[valid], minlength=M)
    sum_mag = np.bincount(grp, weights=flat_abs_mag[valid] * flat_int[valid], minlength=M)
    sum_pix = np.bincount(grp, weights=valid.astype(int)[valid], minlength=M)

    # quiet-sun sums
    qidx = np.logical_and(valid, flat_w_quiet)
    grp_q = bin_idx[qidx] * len(region_codes) + reg_idx[qidx]
    sum_vquiet = np.bincount(grp_q, weights=flat_v_corr[qidx] * flat_int[qidx], minlength=M)
    sum_int_q = np.bincount(grp_q, weights=flat_int[qidx], minlength=M)

    # reshape
    sum_vhat = sum_vhat.reshape(n_bins, len(region_codes))
    sum_vphot = sum_vphot.reshape(n_bins, len(region_codes))
    sum_int = sum_int.reshape(n_bins, len(region_codes))
    sum_iflat = sum_iflat.reshape(n_bins, len(region_codes))
    sum_mag = sum_mag.reshape(n_bins, len(region_codes))
    sum_vquiet = sum_vquiet.reshape(n_bins, len(region_codes))
    sum_int_q = sum_int_q.reshape(n_bins, len(region_codes))
    sum_pix = sum_pix.reshape(n_bins, len(region_codes))

    quiet_idx = np.argmax(np.array(region_codes) == quiet_sun_code)

    for j in range(n_bins):
        lo_mu = bins[j]
        hi_mu = bins[j+1]

        dq_mu = sum_int_q[j, quiet_idx]
        v_q_mu = sum_vquiet[j, quiet_idx] / dq_mu if dq_mu > 0 else 0.0

        for ir, r in enumerate(region_codes):
            denom_int = sum_int[j, ir]
            denom_pix = sum_pix[j, ir]
            if denom_int == 0:
                rows.append([mjd, r, lo_mu, hi_mu, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                continue

            pix_frac = denom_pix / np.nansum(flat_mu >= mu_thresh)
            light_frac = denom_int / np.nansum(flat_int[flat_mu >= mu_thresh])

            v_hat = sum_vhat[j, ir] / denom_int
            v_phot = sum_vphot[j, ir] / denom_int

            dq = sum_int_q[j, ir]
            v_q = sum_vquiet[j, ir] / dq if dq > 0 else 0.0
            v_conv = v_hat - v_q_mu

            mag_u = sum_mag[j, ir] / denom_int
            avg_i = sum_int[j, ir] / denom_pix
            avg_if = sum_iflat[j, ir] / denom_pix

            rows.append([mjd, r, lo_mu, hi_mu,
                         pix_frac, light_frac,
                         v_hat, v_phot, v_q, v_conv,
                         mag_u, avg_i, avg_if])
    return rows