"""SDO image handling and region masking utilities."""

import numpy as np
import pdb, ipdb, time, warnings
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sunpy.map import Map as sun_map
from sunpy.coordinates import frames

from numba import njit
from scipy import ndimage
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from skimage.measure import regionprops, regionprops_table
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from string import ascii_letters
from scipy.ndimage import distance_transform_edt

from .sdo_io import *
from .limbdark import *
from .legendre import *
from .legendre import bulk_vel_design, basis_scale
from .reproject import *
from .geometry import pixel_to_hpc, hpc_to_hcc, hcc_to_hgs, compute_geometry

from .moat import detect_moats

warnings.simplefilter("ignore", category=VerifyWarning)
warnings.simplefilter("ignore", category=FITSFixedWarning)

# set globals for region IDS. These are the mutually-exclusive morphological
# classes: every on-disk pixel gets exactly one, so they form a partition and
# are stored as a single integer code per pixel in SunMask.regions.
umbrae_code = 1
penumbrae_code = 2
quiet_sun_code = 3
network_code = 4
plage_code = 5

# the base partition membership list (moat is NOT here -- it is a non-exclusive
# overlay carried in the feature-flag plane, see below)
region_codes = [umbrae_code, penumbrae_code, quiet_sun_code, network_code, plage_code]

# non-exclusive feature flags (SunMask.flags, a bitmask plane like quality.py).
# A pixel may carry several: a moat pixel can also be plage. Sub-types that
# would otherwise need their own exclusive code live here instead.
blue_pen_flag = 1 << 0    # penumbra pixel with v_corr <= 0 (blueshifted)
red_pen_flag = 1 << 1     # penumbra pixel with v_corr > 0 (redshifted)
moat_left_flag = 1 << 2   # moat pixel, left hemisphere (lon >= 0)
moat_right_flag = 1 << 3  # moat pixel, right hemisphere (lon < 0)
moat_any_flag = moat_left_flag | moat_right_flag

# output codes for the flag-derived region_output.csv rows. moat_code is kept
# (a derived left|right query) so existing moat.csv consumers are unaffected.
moat_code = 6
blue_penumbra_code = 7
red_penumbra_code = 8
left_moat_code = 9
right_moat_code = 10
plage_no_moat_code = 11     # plage pixels NOT also in a moat
network_no_moat_code = 12   # network pixels NOT also in a moat


def flag_selections(flat_reg, flat_flags):
    """Map the feature-flag plane to (output_code, pixel_mask) selections.

    Unlike region_codes these masks may overlap (a moat pixel can also be plage),
    so each is aggregated independently downstream. The *_no_moat variants pair a
    base class with the negation of the moat flag, letting callers separate moat
    overlap from the rest of the bright region.
    """
    moat = (flat_flags & moat_any_flag) != 0
    return [
        (blue_penumbra_code,   (flat_flags & blue_pen_flag) != 0),
        (red_penumbra_code,    (flat_flags & red_pen_flag) != 0),
        (moat_code,            moat),
        (left_moat_code,       (flat_flags & moat_left_flag) != 0),
        (right_moat_code,      (flat_flags & moat_right_flag) != 0),
        (plage_no_moat_code,   (flat_reg == plage_code) & ~moat),
        (network_no_moat_code, (flat_reg == network_code) & ~moat),
    ]

class SDOImage(object):
    """Load an SDO image and provide geometry, corrections, and metadata.

    Parameters
    ----------
    file : str
        Path to the FITS file to load.
    dtype : numpy dtype, optional
        Data type for the image array.
    """
    def __init__(self, file, dtype=np.float32):
        # set the filename
        self.filename = file

        # get the image and the header
        self.image = read_data(self.filename, dtype=dtype)
        self.parse_header()

        # initialize mu_thresh
        self.mu_thresh = 0.0
        return None

    def parse_header(self):
        # read the header
        head = read_header(self.filename)
        self.wcs = WCS(head)

        # parse it
        self.naxis1 = head["NAXIS1"]
        self.naxis2 = head["NAXIS2"]
        self.crpix1 = head["CRPIX1"]
        self.crpix2 = head["CRPIX2"]
        self.cdelt1 = head["CDELT1"]
        self.cdelt2 = head["CDELT2"]
        self.date_obs = head["DATE-OBS"]

        self.L0 = head["CRLN_OBS"]
        self.B0 = head["CRLT_OBS"]

        self.dsun_obs = head["DSUN_OBS"]
        self.dsun_ref = head["DSUN_REF"]
        self.rsun_obs = head["RSUN_OBS"]
        self.rsun_ref = head["RSUN_REF"]

        self.obs_vr = head["OBS_VR"]
        self.obs_vw = head["OBS_VW"]
        self.obs_vn = head["OBS_VN"]

        if "CONTENT" in head.keys():
            self.content = head["CONTENT"]
        else:
            self.content = "FILTERGRAM"

        # get data quality flag
        self.instrument = head["TELESCOP"]
        self.quality = head["QUALITY"]

        # export full header
        self.head = head
        return None

    def calc_geometry(self):
        # Analytic replacement for the sunpy/astropy coordinate-transform chain,
        # fused into a single numba pass (see geometry.compute_geometry). Matches
        # calc_geometry_numpy to machine precision and calc_geometry_sunpy to the
        # tolerances in scripts/verify_geometry.py, but avoids the per-pixel
        # SkyCoord/frame machinery and the intermediate full-frame temporaries.

        # the authoritative observer (B0, L0) comes from the same sunpy machinery
        # the reference path uses; this is a cheap scalar lookup (~20 ms)
        smap = sun_map(self.image, self.head)
        obs = smap.observer_coordinate
        b0 = obs.lat.to_value(u.rad)
        l0 = obs.lon.to_value(u.rad)

        self.rsun_solrad = self.dsun_obs / self.rsun_ref

        xx, yy, rr, mu, lat_deg, lon_deg = compute_geometry(
            self.wcs, self.naxis1, self.naxis2,
            self.dsun_obs, self.rsun_ref, self.rsun_obs,
            b0, l0, self.image.dtype)

        self.xx = xx * u.m
        self.yy = yy * u.m
        self.rr = rr * u.dimensionless_unscaled
        self.lat = lat_deg * u.deg
        self.lon = lon_deg * u.deg
        self.mu = mu

        # calculate the pixel areas (unchanged helper)
        self.pix_area = calculate_pixel_area(self.lat, self.lon)
        return None

    def calc_geometry_numpy(self):
        # Pure-numpy analytic path; retained as a machine-precision oracle for the
        # fused numba calc_geometry. Same formulas, just not fused.
        smap = sun_map(self.image, self.head)
        obs = smap.observer_coordinate
        b0 = obs.lat.to_value(u.rad)
        l0 = obs.lon.to_value(u.rad)

        Tx, Ty = pixel_to_hpc(self.wcs, self.naxis1, self.naxis2)
        self.rsun_solrad = self.dsun_obs / self.rsun_ref

        x, y, z = hpc_to_hcc(Tx, Ty, self.dsun_obs, self.rsun_ref)
        self.xx = x * u.m
        self.yy = y * u.m

        Tx_arcsec = (Tx * u.rad).to_value(u.arcsec)
        Ty_arcsec = (Ty * u.rad).to_value(u.arcsec)
        self.rr = (np.sqrt(Tx_arcsec**2 + Ty_arcsec**2) / self.rsun_obs) * u.dimensionless_unscaled

        lon, lat = hcc_to_hgs(x, y, z, b0, l0)
        self.lat = (np.rad2deg(lat) + 90.0) * u.deg
        self.lon = np.rad2deg(lon) * u.deg

        self.pix_area = calculate_pixel_area(self.lat, self.lon)

        rr2 = self.rr.value**2.0
        diff = 1.0 - rr2
        np.clip(diff, 0.0, None, out=diff)
        with np.errstate(invalid='ignore'):
            np.sqrt(diff, out=diff)
        self.mu = diff.astype(self.image.dtype)
        self.mu[rr2 >= 1.0] = np.nan
        return None

    def calc_geometry_sunpy(self):
        # Reference implementation (slow). Retained as the verification oracle for
        # the analytic calc_geometry above; not used in the production pipeline.
        # methods adapted from https://arxiv.org/abs/2105.12055
        # original implementation at https://github.com/samarth-kashyap/hmi-clean-ls
        # get sun map
        smap = sun_map(self.image, self.head)

        # do coordinate transforms / calculations
        paxis1 = np.arange(self.naxis1)
        paxis2 = np.arange(self.naxis2)
        xx, yy = np.meshgrid(paxis1, paxis2)
        hpc = smap.pixel_to_world(xx * u.pix, yy * u.pix)   # helioprojective cartesian
        self.rsun_solrad = self.dsun_obs / self.rsun_ref

        # transform to other coordinate systems
        hgs = hpc.transform_to(frames.HeliographicStonyhurst)
        hcc = hpc.transform_to(frames.Heliocentric)

        # get cartesian and radial coordinates
        self.xx = hcc.x
        self.yy = hcc.y
        self.rr = np.sqrt(hpc.Tx**2 + hpc.Ty**2) / (self.rsun_obs * u.arcsec)

        # heliocgraphic latitude and longitude
        self.lat = hgs.lat + 90 * u.deg
        self.lon = hgs.lon

        # calculate the pixel areas
        self.pix_area = calculate_pixel_area(self.lat, self.lon)

        # get mu
        rr2 = self.rr.value**2.0
        diff = 1.0 - rr2
        np.clip(diff, 0.0, None, out=diff)
        with np.errstate(invalid='ignore'):
            np.sqrt(diff, out=diff)
        self.mu = diff.astype(self.image.dtype)
        self.mu[rr2 >= 1.0] = np.nan
        return None

    def inherit_geometry(self, other_image):
        # self.xx = np.copy(other_image.xx)
        # self.yy = np.copy(other_image.yy)
        # self.rr = np.copy(other_image.rr)
        self.mu = np.copy(other_image.mu)
        self.pix_area = np.copy(other_image.pix_area)
        # self.lat = np.copy(other_image.lat)
        # self.lon = np.copy(other_image.lon)
        return None

    def is_magnetogram(self):
        return self.content == "MAGNETOGRAM"

    def is_dopplergram(self):
        return self.content == "DOPPLERGRAM"

    def is_continuum(self):
        return self.content == "CONTINUUM INTENSITY"

    def is_filtergram(self):
        return self.content == "FILTERGRAM"

    def mask_low_mu(self, mu_thresh):
        self.mu_thresh = mu_thresh
        mask_idx = np.logical_or(self.mu < mu_thresh, np.isnan(self.mu))
        self.image[mask_idx] = np.nan

        if self.is_continuum() | self.is_filtergram():
            self.ldark[mask_idx] = np.nan
            self.iflat[mask_idx] = np.nan
        elif self.is_magnetogram():
            self.B_obs[mask_idx] = np.nan
        elif self.is_dopplergram():
            self.v_corr[mask_idx] = np.nan
            self.v_obs[mask_idx] = np.nan
            self.v_rot[mask_idx] = np.nan
            self.v_mer[mask_idx] = np.nan
            self.v_cbs[mask_idx] = np.nan

        return None

    def correct_magnetogram(self):
        """Apply mu correction to magnetogram values."""
        assert self.is_magnetogram(), "expected magnetogram, got content=%s (%s)" % (self.content, self.filename)
        self.B_obs = self.image.copy()
        self.image /= self.mu
        return None

    def correct_dopplergram(self, fit_cbs=False):
        """Compute velocity corrections and derived components for dopplergrams.

        Parameters
        ----------
        fit_cbs : bool, optional
            If True, fit convective blueshift components in the bulk velocity model.
        """
        assert self.is_dopplergram(), "expected dopplergram, got content=%s (%s)" % (self.content, self.filename)

        # get mask excluding nans / sqrts of negatives
        # self.mask_nan = np.logical_and((self.rr <= 0.95), ~np.isnan(self.lat))
        # self.mask_nan = np.logical_and((self.mu >= 0.1), ~np.isnan(self.image))
        self.mask_nan = (self.mu >= 0.1)

        # velocity components
        self.v_grav = 633 # m/s, constant 
        self.calc_spacecraft_vel() # spacecraft velocity
        self.calc_bulk_vel(fit_cbs=fit_cbs) # differential rotation + meridional flows + cbs
        return None

    def calc_spacecraft_vel(self):
        # methods adapted from https://arxiv.org/abs/2105.12055
        # original implementation at https://github.com/samarth-kashyap/hmi-clean-ls
        assert self.is_dopplergram(), "expected dopplergram, got content=%s (%s)" % (self.content, self.filename)

        # pre-compute trigonometric quantities on the valid (mask_nan) pixels
        # only -- the result is discarded off-mask anyway, so computing the full
        # 16.8M-pixel trig wastes ~30% of the work
        m = self.mask_nan
        sig = np.arctan(self.rr[m] / self.rsun_solrad)
        chi = np.arctan2(self.xx[m], self.yy[m])
        sin_sig = np.sin(sig)
        cos_sig = np.cos(sig)
        sin_chi = np.sin(chi)
        cos_chi = np.cos(chi)

        # project satellite velocity into coordinate frame
        vr1 = self.obs_vr * cos_sig
        vr2 = -self.obs_vw * sin_sig * sin_chi
        vr3 = -self.obs_vn * sin_sig * cos_chi

        # scatter back into the full frame
        self.v_obs = np.zeros_like(self.image)
        self.v_obs[m] = -(vr1 + vr2 + vr3)
        self.v_obs[~m] = np.nan
        return None

    def calc_bulk_vel(self, fit_cbs=False):
        # Fused numba implementation of the bulk-velocity fit. Reproduces
        # calc_bulk_vel_numpy (retained below) but generates the Legendre design
        # matrix via an in-kernel recurrence instead of gen_leg_vec/gen_leg_x_vec,
        # then runs the identical normal-equations solve. With the rho basis bug
        # fixed (gen_leg_x_vec no longer mis-scales rho), the fit_cbs=True system
        # is well-conditioned (cond ~4e7), so the fast path handles both cases.
        # methods adapted from https://arxiv.org/abs/2105.12055
        assert self.is_dopplergram(), "expected dopplergram, got content=%s (%s)" % (self.content, self.filename)

        # B0 (CRLT_OBS) is in degrees; convert to radians. The upstream Kashyap
        # code used a sunpy Quantity here and let np.cos auto-convert; this port
        # read the raw header float, so the conversion must be explicit.
        cos_B0 = np.cos(np.deg2rad(self.B0))
        sin_B0 = np.sin(np.deg2rad(self.B0))
        n_poly = 11 if fit_cbs else 6

        # masked inputs as plain arrays (lat/lon in degrees, rho dimensionless)
        lat_deg = self.lat[self.mask_nan].to_value(u.deg)
        lon_deg = self.lon[self.mask_nan].to_value(u.deg)
        rho = self.rr[self.mask_nan].value

        # build the design matrix via the fused numba kernel (replaces the slow
        # gen_leg_vec/gen_leg_x_vec basis generation), then run the identical
        # normal-equations solve as calc_bulk_vel_numpy.
        self.im_arr = bulk_vel_design(lat_deg, lon_deg, rho, cos_B0, sin_B0,
                                      n_poly, basis_scale)

        # subtract on the masked subset directly, avoiding a full-frame temporary.
        # Bitwise-identical to (image - v_obs - v_grav)[mask_nan] (v_grav is scalar;
        # the result is already a fresh array, so no .copy() is needed).
        m = self.mask_nan
        self.dat = self.image[m] - self.v_obs[m] - self.v_grav
        self.RHS = self.im_arr.dot(self.dat)
        A = self.im_arr @ self.im_arr.T
        self.fit_params = np.linalg.solve(A, self.RHS)

        self.v_rot = np.zeros_like(self.image)
        self.v_rot[self.mask_nan] = self.fit_params[:3].dot(self.im_arr[:3, :])
        self.v_rot[~self.mask_nan] = np.nan

        self.v_mer = np.zeros_like(self.image)
        self.v_mer[self.mask_nan] = self.fit_params[3:5].dot(self.im_arr[3:5, :])
        self.v_mer[~self.mask_nan] = np.nan

        self.v_cbs = np.zeros_like(self.image)
        self.v_cbs[self.mask_nan] = self.fit_params[5:].dot(self.im_arr[5:, :])
        self.v_cbs[~self.mask_nan] = np.nan

        self.dat -= self.fit_params.dot(self.im_arr)
        self.v_corr = np.zeros_like(self.image)
        self.v_corr[self.mask_nan] = self.dat
        self.v_corr[~self.mask_nan] = np.nan
        return None

    def calc_bulk_vel_numpy(self, fit_cbs=False):
        # Reference numpy/scipy implementation; retained as the oracle for the
        # fused numba calc_bulk_vel above.
        # methods adapted from https://arxiv.org/abs/2105.12055
        # original implementation at https://github.com/samarth-kashyap/hmi-clean-ls
        assert self.is_dopplergram(), "expected dopplergram, got content=%s (%s)" % (self.content, self.filename)

        # pre-compute trigonometric quantities (B0 in degrees -> radians;
        # see calc_bulk_vel for why the conversion must be explicit here)
        cos_B0 = np.cos(np.deg2rad(self.B0))
        sin_B0 = np.sin(np.deg2rad(self.B0))

        self.lat_mask = self.lat[self.mask_nan]#.copy()
        self.lon_mask = self.lon[self.mask_nan]#.copy()
        self.rho_mask = self.rr[self.mask_nan]#.copy()

        cos_theta = np.cos(self.lat_mask)
        sin_theta = np.sin(self.lat_mask)
        cos_phi = np.cos(self.lon_mask)
        sin_phi = np.sin(self.lon_mask)

        self.lt = sin_B0 * sin_theta - cos_B0 * cos_theta * cos_phi
        self.lp = cos_B0 * sin_phi

        # calculate legendre poylnomials
        pl_theta, dt_pl_theta = gen_leg_vec(5, self.lat_mask)
        if fit_cbs:
            pl_rho, dt_pl_rho = gen_leg_x_vec(5, self.rho_mask)
        else:
            pl_rho, dt_pl_rho = gen_leg_x_vec(0, self.rho_mask)

        # figure out how many polynomials we need
        if fit_cbs:
            n_poly = 11
        else:
            n_poly = 6

        # allocate memory
        self.im_arr = np.zeros((n_poly, self.lt.shape[0]))

        # differential rotation (axisymmetric feature; s = 1, 3, 5)
        self.im_arr[0, :] = dt_pl_theta[1, :] * self.lp
        self.im_arr[1, :] = dt_pl_theta[3, :] * self.lp
        self.im_arr[2, :] = dt_pl_theta[5, :] * self.lp

        # meridional circulation (axisymmetric feature; s = 2, 4)
        # s = 0 is 0
        self.im_arr[3, :] = dt_pl_theta[2, :] * self.lt
        self.im_arr[4, :] = dt_pl_theta[4, :] * self.lt

        # axisymmetric feature (frame=pole at disk-center)
        # s = 0-5
        self.im_arr[5, :] = pl_rho[0, :]
        if fit_cbs:
            self.im_arr[6, :] = pl_rho[1, :]
            self.im_arr[7, :] = pl_rho[2, :]
            self.im_arr[8, :] = pl_rho[3, :]
            self.im_arr[9, :] = pl_rho[4, :]
            self.im_arr[10, :] = pl_rho[5, :]

        # get the data to fit and compute RHS
        self.dat = (self.image - self.v_obs - self.v_grav)[self.mask_nan].copy()
        self.RHS = self.im_arr.dot(self.dat)

        # fill the matrix and compute fit params
        A = self.im_arr @ self.im_arr.T
        self.fit_params = np.linalg.solve(A, self.RHS)

        # get rotation component
        self.v_rot = np.zeros_like(self.image)
        self.v_rot[self.mask_nan] = self.fit_params[:3].dot(self.im_arr[:3, :])
        self.v_rot[~self.mask_nan] = np.nan

        # get meridional circulation component
        self.v_mer = np.zeros_like(self.image)
        self.v_mer[self.mask_nan] = self.fit_params[3:5].dot(self.im_arr[3:5, :])
        self.v_mer[~self.mask_nan] = np.nan

        # get convective blueshift w/ limb component
        self.v_cbs = np.zeros_like(self.image)
        self.v_cbs[self.mask_nan] = self.fit_params[5:].dot(self.im_arr[5:, :])
        self.v_cbs[~self.mask_nan] = np.nan

        # get corrected velocity
        self.dat -= self.fit_params.dot(self.im_arr)
        self.v_corr = np.zeros_like(self.image)
        self.v_corr[self.mask_nan] = self.dat
        self.v_corr[~self.mask_nan] = np.nan
        return None

    def calc_limb_darkening(self, mu_lim=0.1, num_mu=25, n_sigma=2.0):
        """Estimate limb darkening and flatten continuum/filtergram intensity.

        Parameters
        ----------
        mu_lim : float, optional
            Minimum mu used in the fit.
        num_mu : int, optional
            Number of mu bins between mu_lim and 1.
        n_sigma : float, optional
            Sigma clipping threshold within each bin.
        """
        assert (self.is_continuum() | self.is_filtergram()), "expected continuum or filtergram, got content=%s (%s)" % (self.content, self.filename)

        # flatten & mask
        mu_flat = self.mu.ravel()
        I_flat = self.image.ravel()
        valid = (~np.isnan(I_flat)) & (mu_flat >= mu_lim)
        mu_valid = mu_flat[valid]
        I_valid = I_flat[valid]

        # build bin edges
        mu_edges = np.linspace(mu_lim, 1.0, num=num_mu+1)
        bin_idx = np.digitize(mu_valid, mu_edges) - 1
        bin_idx = np.clip(bin_idx, 0, num_mu-1)
        nbins = num_mu

        # first pass: sums, counts, sum of squares
        sums = np.bincount(bin_idx, weights=I_valid, minlength=nbins)
        counts = np.bincount(bin_idx, minlength=nbins)
        sum2 = np.bincount(bin_idx, weights=I_valid**2, minlength=nbins)
        
        # compute mean & std per bin
        means = sums / counts
        stds = np.sqrt(np.clip(sum2/counts - means**2.0, 0, None))
        
        # sigma-clip outliers
        mask_out = np.abs(I_valid - means[bin_idx]) > (n_sigma * stds[bin_idx])
        if np.any(mask_out):
            sums = np.bincount(bin_idx[~mask_out], weights=I_valid[~mask_out], minlength=nbins)
            counts = np.bincount(bin_idx[~mask_out], minlength=nbins)
        avg_int = sums / counts

        # bin-center mu values
        mu_avgs = 0.5 * (mu_edges[:-1] + mu_edges[1:])

        # fit and fix the coeffs for LD law
        p = np.polyfit(1.0 - mu_avgs, avg_int, 2)
        a = p[2]
        b = -p[1] / p[2]
        c = -p[0] / p[2]

        # flatten
        self.ld_coeffs = np.array([a, b, c])
        self.ldark = quad_darkening_two(self.mu, b, c)
        self.iflat = self.image / self.ldark
        return None

    def rescale_to_hmi(self, hmi_image):
        """Resample a filtergram onto an HMI image grid and inherit geometry.

        Parameters
        ----------
        hmi_image : SDOImage
            Target HMI image providing the output WCS and geometry.
        """
        assert self.is_filtergram(), "expected filtergram, got content=%s (%s)" % (self.content, self.filename)

        # compute pixel mapping 
        H, W = hmi_image.image.shape
        src_x, src_y = compute_pixel_mapping(self.wcs, hmi_image.wcs, (H, W))

        # do the interpolation (bilinear)
        dst = np.empty((H, W), dtype=np.float32)
        bilinear_reproject(self.image, src_x, src_y, dst)

        # set attributes
        self.image = dst
        self.inherit_geometry(hmi_image)
        self.wcs = hmi_image.wcs

# for creating pixel mask with thresholded regions
def calculate_weights(mag):
    """Return active and quiet masks based on magnetogram thresholds."""
    # set magnetic threshold
    mag_thresh = 24.0 / mag.mu

    # make flag array for magnetically active areas
    w_active = (np.abs(mag.image) > mag_thresh).astype(float)

    # convolve with boxcar filter to remove isolated pixels
    w_conv = ndimage.convolve(w_active, np.ones([3,3]), mode="constant")
    w_active = np.logical_and(w_conv >= 2., w_active == 1.)
    w_active[np.logical_or(mag.mu < mag.mu_thresh, np.isnan(mag.mu))] = False

    # make weights array for magnetically quiet areas
    w_quiet = ~w_active
    w_quiet[np.logical_or(mag.mu < mag.mu_thresh, np.isnan(mag.mu))] = False
    return w_active, w_quiet

def calculate_pixel_area(lat, lon):
    """Compute per-pixel areas from heliographic latitude/longitude grids."""
    # convert to radians
    lat_rad = lat.value * np.pi / 180.0
    lon_rad = lon.value * np.pi / 180.0
    d_lat = np.diff(lat_rad, 1, 0)
    d_lon = np.diff(lon_rad, 1, 1)

    # pad the edge
    d_lat2 = np.pad(d_lat, ((0, 1), (0, 0)), mode="constant")
    d_lon2 = np.pad(d_lon, ((0, 0), (0, 1)), mode="constant")

    # compute the areas of pixels. Same operation order as
    #   sin(lat) * |d_lon2| * |d_lat2| / (2*pi) * 1e6
    # but accumulated in place to avoid ~6 full-frame (134 MB) temporaries.
    np.abs(d_lon2, out=d_lon2)
    np.abs(d_lat2, out=d_lat2)
    pix_area = np.sin(lat_rad)
    pix_area *= d_lon2
    pix_area *= d_lat2
    pix_area /= (2 * np.pi)
    pix_area *= 1e6
    return pix_area

def pad_max_len(data, max_length):
    """Pad a 1D array with NaNs up to max_length."""
    return np.hstack([data, np.repeat(np.nan, max_length - len(data))]).astype(float)

def get_areas(labels, intensity_image):
    """Compute region areas and intensity-weighted areas from label map."""
    properties = ("label","area","mean_intensity")
    rprops_tab = regionprops_table(labels, intensity_image=intensity_image, properties=properties)    
    area = np.r_[0, rprops_tab['area']]
    mean = np.r_[0, rprops_tab['mean_intensity']]
    areas_pix = area[labels]
    areas_mic = (area * mean)[labels]
    return areas_pix, areas_mic

class SunMask(object):
    """Classify solar regions using continuum, magnetogram, doppler, and AIA data.

    Parameters
    ----------
    con, mag, dop, aia : SDOImage
        Continuum, magnetogram, dopplergram, and filtergram images. These are
        used to derive masks for umbrae, penumbrae, quiet sun, network, plage,
        and moat flow regions.
    """
    def __init__(self, con, mag, dop, aia, **kwargs):
        # check argument order/names are correct
        # print("Entered SunMask.__init__")
        
        assert con.is_continuum(), "con is not a continuum image: content=%s (%s)" % (con.content, con.filename)
        assert mag.is_magnetogram(), "mag is not a magnetogram: content=%s (%s)" % (mag.content, mag.filename)
        assert dop.is_dopplergram(), "dop is not a dopplergram: content=%s (%s)" % (dop.content, dop.filename)
        assert aia.is_filtergram(), "aia is not a filtergram: content=%s (%s)" % (aia.content, aia.filename)

        # copy observation date
        self.date_obs = con.date_obs

        # inherit the geometry and the WCS
        self.wcs = WCS(read_header(con.filename))
        self.inherit_geometry(con)

        # calculate weights
        self.w_active, self.w_quiet = calculate_weights(mag)

        # calculate magnetic filling factor
        npix = np.nansum(con.mu >= con.mu_thresh)
        self.ff = np.nansum(self.w_active[con.mu >= con.mu_thresh]) / npix

        # identify regions
        self.identify_regions(con, mag, dop, aia, **kwargs)

        # get region fracs
        self.umb_frac = np.nansum(self.is_umbra()) / npix
        self.pen_frac = np.nansum(self.is_penumbra()) / npix
        self.blu_pen_frac = np.nansum(self.is_blue_penumbra()) / npix
        self.red_pen_frac = np.nansum(self.is_red_penumbra()) / npix
        self.quiet_frac = np.nansum(self.is_quiet_sun()) / npix
        self.network_frac = np.nansum(self.is_network()) / npix
        self.plage_frac = np.nansum(self.is_plage()) / npix
        self.moat_frac = np.nansum(self.is_moat_flow()) / npix
        self.left_moat_frac = np.nansum(self.is_left_moat()) / npix
        self.right_moat_frac = np.nansum(self.is_right_moat()) / npix
        return None

    def inherit_geometry(self, other_image):
        # self.xx = np.copy(other_image.xx)
        # self.yy = np.copy(other_image.yy)
        # self.rr = np.copy(other_image.rr)
        self.mu = np.copy(other_image.mu)
        # self.lat = np.copy(other_image.lat)
        # self.lon = np.copy(other_image.lon)
        return None

    def identify_regions(self, con, mag, dop, aia, plot_moat=False, classify_moat=False):
        invalid_mask = np.logical_or(con.mu <= con.mu_thresh, np.isnan(con.mu))

        # allocate memory for mask array
        self.regions = np.zeros_like(con.image)
        # non-exclusive feature flags (see sdo_image module globals); a pixel may
        # carry several bits (e.g. moat AND plage) without losing its base label
        self.flags = np.zeros(self.regions.shape, dtype=np.uint8)

        # calculate intensity thresholds for HMI. thresh1 and thresh2 are the
        # same quiet-sun mean intensity scaled by two different constants, so the
        # full-frame product and its two reductions are computed once here. The
        # operand order (const * sum / sum) is preserved exactly, so the results
        # are bit-identical to evaluating each line independently.
        qs_int_sum = np.nansum(con.iflat * self.w_quiet)
        qs_weight_sum = np.nansum(self.w_quiet)
        self.con_thresh1 = 0.89 * qs_int_sum / qs_weight_sum
        self.con_thresh2 = 0.45 * qs_int_sum / qs_weight_sum

        # get indices for umbrae
        ind1 = con.iflat <= self.con_thresh2      # if intensity less than thresh2, umbra (ind1)

        # get indices for penumbrae
        indp = np.logical_and(con.iflat <= self.con_thresh1, con.iflat > self.con_thresh2)
            # if flattened continuum intensity less than thresh1 and greater than thresh2, penumbra (ind1 or ind2)
        if hasattr(dop, "v_corr"):
            ind2 = np.logical_and(indp, dop.v_corr <= 0)    # if penumbra and bluehift or 0, ind2
            ind3 = np.logical_and(indp, dop.v_corr > 0)     # if penumbra and redshift, ind3
        else: # no attribute v_corr
            ind2 = indp
            ind3 = np.zeros_like(indp)

        """
        # find contiguous penumbra regions
        structure = ndimage.generate_binary_structure(2,2)
        labels, nlabels = ndimage.label(indp, structure=structure)

        # find the mean mean of each bin
        index = np.arange(1, nlabels+1)
        mean_mus = ndimage.labeled_comprehension(self.mu, labels, index, np.mean, float, np.nan)

        # allocate memory, get indices for near- and far- penumbrae
        ind2_new = np.zeros(np.shape(self.regions), dtype=bool)
        ind3_new = np.zeros(np.shape(self.regions), dtype=bool)
        for i in index:
            ind2_new += ((labels == i) & (self.mu < mean_mus[i-1]))
            ind3_new += ((labels == i) & (self.mu >= mean_mus[i-1]))
        """

        # get indices for quiet sun
        # if continuum intensity is greater than thresh1 and weak B field
        ind4 = np.logical_and(con.iflat > self.con_thresh1, self.w_quiet)
        
        # calculate intensity thresholds for AIA
        weights = np.logical_and.reduce([self.w_active, ~ind1, ~ind2, ~ind3])
        self.aia_thresh = np.nansum(aia.iflat * weights) / np.nansum(weights)

        # get indices for bright regions (plage/faculae + network)
        ind5a = np.logical_and(con.iflat > self.con_thresh1, self.w_active)  # intensity greater than thresh 1 and strong B field
        ind5b = np.logical_and.reduce([aia.iflat > self.aia_thresh, ~ind1, ~ind2, ~ind3]) # > aia thresh and not umbra or penumbra
        ind5 = np.logical_or(ind5a, ind5b) # if ind5a or ind5b, bright

        # set mask indices
        self.regions[ind1] = umbrae_code 
        self.regions[ind2] = penumbrae_code # blue_pen_code 
        self.regions[ind3] = penumbrae_code # red_pen_code 
        self.regions[ind4] = quiet_sun_code 
        self.regions[ind5] = network_code # bright areas (will separate into plage + network)

        # tag the penumbra velocity split as non-exclusive flags (both pixels
        # remain penumbrae_code in the base partition)
        self.flags[ind2] |= blue_pen_flag
        self.flags[ind3] |= red_pen_flag

        # create structures for dilations
        corners = ndimage.generate_binary_structure(2,2) # array of bools, defines feature connections
        no_corners = ndimage.generate_binary_structure(2,1)

        # label unique contiguous bright regions (label islands of bright stuff)
        binary_img = self.regions == network_code  # get bright areas 
        labels, nlabels = ndimage.label(binary_img, structure=corners) # takes bright areas and feature connections, gives each island a label
        areas_pix, areas_mic = get_areas(labels, dop.pix_area) # get areas

        # area thresh is 20 microhemispheres
        area_thresh = 20.0

        # assign region type to plage for ratios less than ratio thresh
        ind6 = areas_mic >= area_thresh  # areas_mic
        self.regions[ind6] = plage_code # plage

        # set isolated bright pixels to quiet sun
        ind_iso = areas_pix == 1.0
        self.regions[ind_iso] = quiet_sun_code # quiet sun

        if classify_moat:
            # grow moat rings outward from each large spot. The tunable logic
            # lives in the pure kernel moat.detect_moats (see
            # docs/superpowers/specs/2026-07-01-moat-tuning-design.md); this
            # applies its masks to the region map. Behavior-preserving vs the
            # former inline implementation.
            moat_result = detect_moats(dop.v_corr, con.mu, dop.lon.value,
                                       con.image, mag.image, invalid_mask,
                                       self.is_umbra(), self.is_penumbra())
            # non-destructive: set moat flags but leave the base label (a moat
            # ring overlapping plage/network keeps its plage/network code)
            self.flags[moat_result.left_moat] |= moat_left_flag
            self.flags[moat_result.right_moat] |= moat_right_flag
            self.moat_profiles = moat_result.profiles

            if plot_moat:
                plt.imshow(self.is_moat_flow())
                plt.show()

        # make any remaining unclassified pixels quiet sun
        ind_rem = np.logical_and(~invalid_mask, self.is_unclassified())
        self.regions[ind_rem] = quiet_sun_code

        # set values beyond mu_thresh to nan (if they weren't already)
        self.regions[invalid_mask] = np.nan
        self.flags[invalid_mask] = 0

        return None

    def get_moat_properties(self):
        """Per-spot moat radial profiles (populated when classify_moat=True)."""
        return self.moat_profiles

    def mask_low_mu(self, mu_thresh):
        self.mu_thresh = mu_thresh
        self.regions[np.logical_or(self.mu < mu_thresh, np.isnan(self.mu))] = np.nan
        return None
    
    def is_unclassified(self):
        """Return mask for pixels with no valid region classification."""
        return np.logical_or(np.isnan(self.regions), ~np.isin(self.regions, region_codes))

    def is_umbra(self):
        """Return mask for umbra regions."""
        return self.regions == umbrae_code

    def is_penumbra(self):
        """Return mask for penumbra regions."""
        # return np.logical_or(self.regions == 2, self.regions == 3)
        return self.regions == penumbrae_code

    def is_blue_penumbra(self):
        """Return mask for blue-shifted penumbra regions."""
        return (self.flags & blue_pen_flag) != 0

    def is_red_penumbra(self):
        """Return mask for red-shifted penumbra regions."""
        return (self.flags & red_pen_flag) != 0

    def is_quiet_sun(self):
        """Return mask for quiet sun regions."""
        return self.regions == quiet_sun_code

    def is_network(self):
        """Return mask for network regions."""
        return self.regions == network_code

    def is_plage(self):
        """Return mask for plage regions."""
        return self.regions == plage_code
    
    def is_moat_flow(self):
        """Return mask for moat flow regions (either hemisphere)."""
        return (self.flags & moat_any_flag) != 0

    def is_left_moat(self):
        """Return mask for left-hand moat flow regions."""
        return (self.flags & moat_left_flag) != 0

    def is_right_moat(self):
        """Return mask for right-hand moat flow regions."""
        return (self.flags & moat_right_flag) != 0
