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
from .reproject import *

import sdo_clv_pipeline.plot_moats_data as plot_moats_data
from sdo_clv_pipeline.plot_moats_data import load_and_plot

warnings.simplefilter("ignore", category=VerifyWarning)
warnings.simplefilter("ignore", category=FITSFixedWarning)

# set globals for region IDS
umbrae_code = 1 
penumbrae_code = 2
quiet_sun_code = 3 
network_code = 4 
plage_code = 5 
moat_code = 6 

# set all region codes
region_codes = [umbrae_code, penumbrae_code, quiet_sun_code, network_code, plage_code, moat_code]

class SDOImage(object):
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
        if self.instrument == "SDO/AIA":
            self.quality = head["QUALLEV0"]
        elif self.instrument == "SDO/HMI":
            self.quality = head["QUALLEV1"]

        self.head = head
        return None

    def calc_geometry(self):
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
        assert self.is_magnetogram()
        self.B_obs = self.image.copy()
        self.image /= self.mu
        return None

    def correct_dopplergram(self, fit_cbs=False):
        assert self.is_dopplergram()

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
        assert self.is_dopplergram()

        # pre-compute trigonometric quantities
        sig = np.arctan(self.rr / self.rsun_solrad)
        chi = np.arctan2(self.xx, self.yy)
        sin_sig = np.sin(sig)
        cos_sig = np.cos(sig)
        sin_chi = np.sin(chi)
        cos_chi = np.cos(chi)

        # project satellite velocity into coordinate frame
        vr1 = self.obs_vr * cos_sig
        vr2 = -self.obs_vw * sin_sig * sin_chi
        vr3 = -self.obs_vn * sin_sig * cos_chi

        # reshape into 4096 x 4096 array
        self.v_obs = np.zeros_like(self.image)
        self.v_obs[self.mask_nan] = -(vr1 + vr2 + vr3)[self.mask_nan]
        self.v_obs[~self.mask_nan] = np.nan
        return None

    def calc_bulk_vel(self, fit_cbs=False):
        # methods adapted from https://arxiv.org/abs/2105.12055
        # original implementation at https://github.com/samarth-kashyap/hmi-clean-ls
        assert self.is_dopplergram()

        # pre-compute trigonometric quantities
        cos_B0 = np.cos(self.B0)
        sin_B0 = np.sin(self.B0)

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
        assert (self.is_continuum() | self.is_filtergram())

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
        assert self.is_filtergram()

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
    # convert to radians
    lat_rad = lat.value * np.pi / 180.0
    lon_rad = lon.value * np.pi / 180.0
    d_lat = np.diff(lat_rad, 1, 0)
    d_lon = np.diff(lon_rad, 1, 1)

    # pad the edge
    d_lat2 = np.pad(d_lat, ((0, 1), (0, 0)), mode="constant")
    d_lon2 = np.pad(d_lon, ((0, 0), (0, 1)), mode="constant")
    
    # compute the areas of pixels
    pix_area = np.sin(lat_rad) * np.abs(d_lon2) * np.abs(d_lat2) / (2 * np.pi) * 1e6
    return pix_area

def pad_max_len(data, max_length):
    return np.hstack([data, np.repeat(np.nan, max_length - len(data))]).astype(float)

def get_areas(labels, intensity_image):
    properties = ("label","area","mean_intensity")
    rprops_tab = regionprops_table(labels, intensity_image=intensity_image, properties=properties)    
    area = np.r_[0, rprops_tab['area']]
    mean = np.r_[0, rprops_tab['mean_intensity']]
    areas_pix = area[labels]
    areas_mic = (area * mean)[labels]
    return areas_pix, areas_mic

class SunMask(object):
    def __init__(self, con, mag, dop, aia, plot_moat=True):
        # check argument order/names are correct
        # print("Entered SunMask.__init__")
        
        assert con.is_continuum()
        assert mag.is_magnetogram()
        assert dop.is_dopplergram()
        assert aia.is_filtergram()

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
        self.identify_regions(con, mag, dop, aia, plot_moat=plot_moat)

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

    def identify_regions(self, con, mag, dop, aia, plot_moat=True):
        invalid_mask = np.logical_or(con.mu <= con.mu_thresh, np.isnan(con.mu))

        # allocate memory for mask array
        self.regions = np.zeros_like(con.image)

        # calculate intensity thresholds for HMI
        self.con_thresh1 = 0.89 * np.nansum(con.iflat * self.w_quiet) / np.nansum(self.w_quiet)
        self.con_thresh2 = 0.45 * np.nansum(con.iflat * self.w_quiet) / np.nansum(self.w_quiet)

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

        # set blue and red penumbrae
        self.blue_penumbrae = np.zeros_like(con.image).astype(bool)
        self.blue_penumbrae[ind2] = True
        self.red_penumbrae = np.zeros_like(con.image).astype(bool)
        self.red_penumbrae[ind3] = True

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

        # label each penumbra island and include umbra so we only expand outwards
        binary_img = np.logical_or.reduce([self.is_umbra(), self.is_penumbra()])
        labels, nlabels = ndimage.label(binary_img, structure=corners) # label each island of umbra and penumbra
        rprops = regionprops(labels, dop.pix_area)
        areas_pix, areas_mic = get_areas(labels, dop.pix_area) # get areas

        if plot_moat:
            plt.imshow(labels)
            plt.colorbar()
            plt.show()

        # save original array first
        # save_arr = areas_pix.copy()
        # a_label = [36, 17, 19, 15, 37, 49]
        # b_label = [71, 86, 81, 62, 52, 36, 39, 26]

        # store left/right pixels
        left_moat_pixels = np.zeros_like(self.regions, dtype=bool)
        right_moat_pixels = np.zeros_like(self.regions, dtype=bool)

        # collect areas and centroids
        area_thresh = 600
        rprops_tab = regionprops_table(labels, properties=("label","area","centroid"))  
        areas = np.array(rprops_tab["area"])
        x_centroids = np.array((rprops_tab["centroid-1"][areas > area_thresh])).astype(int)
        y_centroids = np.array((rprops_tab["centroid-0"][areas > area_thresh])).astype(int)
        labels = np.array(rprops_tab["label"][areas > area_thresh]).astype(int)
        areas = areas[areas > area_thresh]

        # allocate for moats
        moat_idx = np.zeros_like(con.image).astype(bool)
        moats = np.zeros((len(areas), *np.shape(con.image))).astype(bool)
        area_idx_arr = np.zeros((len(areas), *np.shape(con.image))).astype(bool)

        # allocate for hemisphere indicator 
        left_hemisphere = np.zeros(len(areas)).astype(bool)
        lon = dop.lon.value

        # allocate for average quantities
        avg_mu = np.zeros(len(areas))
        max_dilations = np.zeros(len(areas))
        max_rings = np.zeros(len(areas))
        avg_vels = []
        avg_mags = []
        avg_ints = []

        # allocate for mask
        valid_mask = np.zeros_like(invalid_mask)

        # define function to dilate moats
        def dilate_moat(idx, pre_factor, compute_avgs=True):
            # get area of that region              
            max_area = areas[idx]
            x_centroid = x_centroids[idx]
            y_centroid = y_centroids[idx]

            # get pixels in that region
            max_area_idx = areas_pix == max_area
            area_idx_arr[idx, :, :] = max_area_idx
            valid_list = [~max_area_idx, ~invalid_mask, ~self.is_umbra(), ~self.is_penumbra()]
            np.logical_and.reduce(valid_list, out=valid_mask)

            # get the distance transform
            dist = distance_transform_edt(~max_area_idx)
            rings = dist.astype(int)
            max_ring = np.nanmax(rings[valid_mask]) + 1
            flat_rings = rings[valid_mask].ravel()

            # func to compute inclusive averages (capturing variables above)
            def cum_avg_val(val):
                sums = np.bincount(flat_rings, weights=val, minlength=max_ring)
                counts = np.bincount(flat_rings, minlength=max_ring)
                cumsum_sums = np.cumsum(sums)
                cumsum_counts = np.cumsum(counts)
                return cumsum_sums[1:] / cumsum_counts[1:]

            # select and ravel averaging quantities
            if compute_avgs:
                flat_vel = dop.v_corr[valid_mask].ravel()
                flat_mag = np.abs(mag.image[valid_mask].ravel())
                flat_int = con.image[valid_mask].ravel()
                
                # compute averages and append
                avg_vels.append(cum_avg_val(flat_vel))
                avg_mags.append(cum_avg_val(flat_mag))
                avg_ints.append(cum_avg_val(flat_int))

            # select initially on the larger moat radius
            rings_cond = rings < pre_factor * (1.2 * np.sqrt(max_area / pi))
            cond = [rings_cond, valid_mask] 
            np.logical_and.reduce(cond, out=moat_idx)
            moats[idx, :, :] = moat_idx

            # get average mu
            avg_mu[idx] = np.average(con.mu[moat_idx])

            # get max number of dilations
            max_dilations[idx] = np.nanmax(rings[rings_cond])
            max_rings[idx] = max_ring

            # make hemisphere indicator
            left_hemisphere[idx] = lon[y_centroid, x_centroid] >= 0.0
            return None

        # iterate over regions
        for idx in range(len(areas)):
            dilate_moat(idx, 0.97, compute_avgs=True)

        # now re-do overlapping moats
        collision_map = np.count_nonzero(moats, axis=0) > 1
        overlapping = np.any(np.logical_and(moats, collision_map), axis=(1, 2))
        redo_idxs = np.nonzero(overlapping)[0]

        # iterate again over regions which overlap
        for idx in redo_idxs:
            dilate_moat(idx, 0.65, compute_avgs=False)

        # get average theta
        avg_theta = np.arccos(avg_mu)

        # discriminate based on hemisphere
        ind8 = moats[left_hemisphere, :, :].any(axis=0)
        ind9 = moats[~left_hemisphere, :, :].any(axis=0)

        # set values 
        self.regions[ind8] = moat_code
        self.regions[ind9] = moat_code

        # set left and right moat
        self.left_moat = np.zeros_like(self.regions).astype(bool)
        self.left_moat[ind8] = True
        self.right_moat = np.zeros_like(self.regions).astype(bool)
        self.right_moat[ind9] = True

        # pad data for write out
        if plot_moat:
            max_length = np.max([(len(arr) + 1) for arr in avg_vels])
            vels = np.vstack([pad_max_len(arr, max_length) for arr in avg_vels])
            mags = np.vstack([pad_max_len(arr, max_length) for arr in avg_mags])
            ints = np.vstack([pad_max_len(arr, max_length) for arr in avg_ints])

            # letters for labeling
            letters = [ascii_letters[i%52] for i in range(len(areas))]
            
            # directory to write out moat data to
            moat_path = os.path.join(root, "data", "moat_data")
            if not os.path.exists(moat_path):
                os.mkdir(moat_path)

            # write it out
            iso = get_date(con.filename).isoformat()
            fname = os.path.join(moat_path, f"moats_data_{iso}.npz")
            np.savez_compressed(fname, x=max_rings, vels=vels, mags=mags, ints=ints, 
                                areas=areas, mus=avg_mu, area_idx_arr=area_idx_arr, 
                                letters=letters, dilated_spots=moats)
        
        # print("before plotting")
        # TODO this needs to be checked
        if plot_moat:
            load_and_plot(fname)
            # plot_loop()

        # make any remaining unclassified pixels quiet sun
        ind_rem = np.logical_and(~invalid_mask, self.is_unclassified())
        self.regions[ind_rem] = quiet_sun_code

        # set values beyond mu_thresh to nan (if they weren't already)
        self.regions[invalid_mask] = np.nan

        # fig, ax = plt.subplots()
        # colors = ['r', 'g', 'b']
        # print(len(dilated_spots))
        # for i in range(0, len(dilated_spots)+1):
        #     ax.imshow(dilated_spots[i], alpha=0.4)
        # ax.set_title("Regions and overlaps")
        # plt.show()

        # TODO not sure what this does
        # self.moat_vals = moat_vals
        # self.moat_dilations = moat_dilations
        # self.moat_thetas = moat_thetas
        # self.moat_areas = moat_areas

        # print("moat pixels")
        if plot_moat:
            plt.imshow(self.is_moat_flow())
            plt.show()

        return None

    def get_moat_properties(self):
        return self.moat_vals, self.moat_dilations, self.moat_thetas, self.moat_areas

    def mask_low_mu(self, mu_thresh):
        self.mu_thresh = mu_thresh
        self.regions[np.logical_or(self.mu < mu_thresh, np.isnan(self.mu))] = np.nan
        return None
    
    def is_unclassified(self):
        return np.logical_or(np.isnan(self.regions), ~np.isin(self.regions, region_codes))

    def is_umbra(self):
        return self.regions == umbrae_code

    def is_penumbra(self):
        # return np.logical_or(self.regions == 2, self.regions == 3)
        return self.regions == penumbrae_code

    def is_blue_penumbra(self):
        return self.blue_penumbrae

    def is_red_penumbra(self):
        return self.red_penumbrae

    def is_quiet_sun(self):
        return self.regions == quiet_sun_code

    def is_network(self):
        return self.regions == network_code

    def is_plage(self):
        return self.regions == plage_code
    
    def is_moat_flow(self):
        return self.regions == moat_code
    
    def is_left_moat(self):
        return self.left_moat
    
    def is_right_moat(self):
        return self.right_moat