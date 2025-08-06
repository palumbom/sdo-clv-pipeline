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
