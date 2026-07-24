"""Analytic helioprojective->heliocentric->heliographic transforms.

Pure-numpy replacements for the sunpy ``SkyCoord.transform_to`` chain used by
``SDOImage.calc_geometry``. These mirror sunpy's own definitions exactly (see
``sunpy/coordinates/_transformations.py``: ``Helioprojective.make_3d``,
``hpc_to_hcc``, ``_rotation_matrix_hcc_to_hgs``) so the outputs agree to
numerical precision, but skip per-call SkyCoord construction and the frame
machinery that dominate runtime on 4096x4096 arrays.

Conventions (matched to sunpy):
  * HPC angles Tx, Ty are the spherical lon/lat of the Helioprojective frame.
  * Heliocentric x, y, z are in meters with z toward the observer; the origin is
    shifted from observer to Sun center.
  * Heliographic Stonyhurst longitude of the observer is 0 by definition, so the
    HCC->HGS rotation reduces to a single rotation by the observer latitude B0.
  * Off-disk lines of sight (no intersection with the sphere of radius RSUN_REF)
    yield NaN, as in ``make_3d``.

Reference: Thompson, W. T. 2006, A&A 449, 791.
"""

import logging
import math
import numpy as np
import astropy.units as u
from numba import njit, prange

logger = logging.getLogger(__name__)

_RAD2DEG = 180.0 / math.pi
_RAD2ARCSEC = _RAD2DEG * 3600.0


def pixel_to_hpc(wcs, naxis1, naxis2):
    """Return helioprojective Tx, Ty (radians, 2D arrays) for every pixel.

    Uses astropy's low-level array API ``wcs_pix2world`` (pure projection math in
    C, no SkyCoord), then converts from the WCS native angular unit to radians.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Image WCS (expected HPLN-TAN/HPLT-TAN).
    naxis1, naxis2 : int
        Image dimensions (columns, rows).
    """
    paxis1 = np.arange(naxis1)
    paxis2 = np.arange(naxis2)
    xx, yy = np.meshgrid(paxis1, paxis2)
    shape = xx.shape

    # low-level transform: returns freshly-allocated float64 world coords in the
    # WCS's native CUNIT, which we are free to mutate in place below
    lon, lat = wcs.wcs_pix2world(xx.ravel(), yy.ravel(), 0)

    # convert to degrees using the live CUNIT (do not assume deg vs arcsec).
    # The unit factor is a scalar, so scale in place rather than building
    # full-frame Quantity temporaries (this is astropy's own internal multiply).
    cunit = wcs.wcs.cunit
    lon *= (1.0 * u.Unit(cunit[0])).to_value(u.deg)
    lat *= (1.0 * u.Unit(cunit[1])).to_value(u.deg)

    # wcs_pix2world reports longitude in [0, 360); helioprojective Tx near the
    # disk is small, so unwrap to a symmetric branch [-180, 180). Without this,
    # disk-center pixels come back as ~360 deg and inflate rr (sin/cos-based
    # quantities are immune, but rr = sqrt(Tx^2+Ty^2) is not).
    lon += 180.0
    np.mod(lon, 360.0, out=lon)
    lon -= 180.0

    # to radians, in place (same values as np.deg2rad of the above)
    np.deg2rad(lon, out=lon)
    np.deg2rad(lat, out=lat)
    return lon.reshape(shape), lat.reshape(shape)


def hpc_to_hcc(Tx, Ty, dsun, rsun):
    """Helioprojective (Tx, Ty in rad) -> Heliocentric x, y, z in meters.

    Mirrors ``Helioprojective.make_3d`` (law of cosines, "near" solution) and
    ``hpc_to_hcc`` (axis permutation + origin shift to Sun center). Off-disk
    pixels (negative discriminant) become NaN.

    Parameters
    ----------
    Tx, Ty : ndarray
        Helioprojective longitude/latitude in radians.
    dsun : float
        Observer-Sun distance in meters (DSUN_OBS).
    rsun : float
        Solar radius in meters (RSUN_REF).
    """
    # Buffer-reuse form: computes x, y, z with 4 full-frame allocations instead of
    # ~11, by threading out= through every intermediate product. Tx, Ty are read
    # only (the callers still need them for rr), so they are never written to.
    # Reassociation vs the naive form perturbs results by <=1 ULP.
    d2 = dsun * dsun - rsun * rsun

    cos_Ty = np.cos(Ty)                            # buf A (also reused later for y)
    cos_alpha = np.cos(Tx)                         # buf B -> cos_alpha -> x
    np.multiply(cos_alpha, cos_Ty, out=cos_alpha)  # cos_alpha = cos_Ty*cos_Tx

    # distance observer->surface via law of cosines, "near" root: d = b - sqrt(b^2 - d2)
    d = np.multiply(cos_alpha, dsun)               # buf C = b (dsun*cos_alpha)
    disc = np.multiply(d, d)                       # buf D = b^2
    disc -= d2
    with np.errstate(invalid="ignore"):
        np.sqrt(disc, out=disc)                    # NaN where disc < 0 (off-disk)
    np.subtract(d, disc, out=d)                    # d = b - sqrt(disc)

    # z = dsun - d*cos_alpha (reuse buf D; this is cos_alpha's last use)
    z = disc
    np.multiply(d, cos_alpha, out=z)
    np.subtract(dsun, z, out=z)

    # x = d*cos_Ty*sin_Tx (reuse buf B for sin_Tx -> x; cos_Ty's last read)
    x = cos_alpha
    np.sin(Tx, out=x)
    np.multiply(x, cos_Ty, out=x)
    np.multiply(x, d, out=x)

    # y = d*sin_Ty (reuse buf A for sin_Ty -> y)
    y = cos_Ty
    np.sin(Ty, out=y)
    np.multiply(y, d, out=y)
    return x, y, z


def hcc_to_hgs(x, y, z, b0, l0=0.0):
    """Heliocentric x, y, z -> Heliographic Stonyhurst lon, lat in radians.

    Implements ``_rotation_matrix_hcc_to_hgs`` = rot_z(-l0) @ rot_y(b0) @ axes.
    The rot_z(-l0) term is a pure rotation about the pole, so it leaves latitude
    unchanged and simply adds ``l0`` to every longitude. The remainder gives
    ``lat = arcsin((cosB0*y + sinB0*z)/r)`` and
    ``lon = arctan2(x, cosB0*z - sinB0*y) + l0``.

    Note that the observer's Stonyhurst longitude ``l0`` is small but nonzero for
    SDO (the spacecraft is offset from the Sun-Earth line); omitting it produces
    a constant longitude bias of ~0.066 deg. The caller passes the value from the
    map's ``observer_coordinate`` to match sunpy exactly.

    Parameters
    ----------
    x, y, z : ndarray
        Heliocentric Cartesian components (any consistent length unit).
    b0 : float
        Observer heliographic latitude in radians.
    l0 : float, optional
        Observer heliographic Stonyhurst longitude in radians (default 0).
    """
    cos_b0 = np.cos(b0)
    sin_b0 = np.sin(b0)

    # Buffer-reuse form: 3 full-frame allocations (r, and the two outputs) instead
    # of ~8. x, y, z are read only (noquant still needs them after this call), so
    # they are never written to. Reassociation perturbs results by <=1 ULP.
    r = np.multiply(x, x)                # buf R
    scratch = np.multiply(y, y)          # buf S
    r += scratch
    np.multiply(z, z, out=scratch)
    r += scratch
    np.sqrt(r, out=r)                    # r = |(x,y,z)|

    # lat = arcsin((cos_b0*y + sin_b0*z)/r); reuse buf S for hgs_z -> lat
    tmp = np.multiply(z, sin_b0)         # buf T = sin_b0*z
    lat = scratch
    np.multiply(y, cos_b0, out=lat)      # lat = cos_b0*y
    lat += tmp                           # lat = hgs_z
    lat /= r
    lat = np.arcsin(lat, out=lat)

    # lon = arctan2(x, cos_b0*z - sin_b0*y) + l0; reuse buf R for hgs_x -> lon
    lon = r
    np.multiply(z, cos_b0, out=lon)      # lon = cos_b0*z
    np.multiply(y, sin_b0, out=tmp)      # tmp = sin_b0*y
    lon -= tmp                           # lon = hgs_x
    lon = np.arctan2(x, lon, out=lon)
    lon += l0
    return lon, lat


@njit(cache=True, parallel=True)
def _geometry_kernel(Tx, Ty, dsun, rsun, rsun_obs, cos_b0, sin_b0, l0,
                     xx, yy, rr, mu, lat_deg, lon_deg):
    """Single fused pass computing all geometry arrays from Tx, Ty (radians).

    Reproduces the numpy chain hpc_to_hcc -> hcc_to_hgs plus rr/mu in one loop,
    avoiding ~10 intermediate full-frame temporaries. Each pixel is independent
    (a pure map, no cross-pixel reduction), so the prange loop is thread-count
    invariant: results are identical whether run on 1 thread (batch) or many
    (single-epoch). Thread count is controlled at runtime via
    parallel.set_compute_threads (default 1). Off-disk pixels (negative
    discriminant) get NaN for xx/yy/lat/lon; mu is NaN wherever rr >= 1.
    """
    n = Tx.shape[0]
    d2 = dsun * dsun - rsun * rsun
    for i in prange(n):
        tx = Tx[i]
        ty = Ty[i]
        ctx = math.cos(tx)
        stx = math.sin(tx)
        cty = math.cos(ty)
        sty = math.sin(ty)

        # rr and mu (defined for all pixels; same formula as the sunpy path)
        rho = math.sqrt(tx * tx + ty * ty) * _RAD2ARCSEC / rsun_obs
        rr[i] = rho
        rho2 = rho * rho
        if rho2 >= 1.0:
            mu[i] = np.nan
        else:
            mu[i] = math.sqrt(1.0 - rho2)

        # HPC -> HCC (law of cosines, near root); off-disk -> NaN
        cos_alpha = cty * ctx
        b = dsun * cos_alpha
        disc = b * b - d2
        if disc < 0.0:
            xx[i] = np.nan
            yy[i] = np.nan
            lat_deg[i] = np.nan
            lon_deg[i] = np.nan
        else:
            d = b - math.sqrt(disc)
            x = d * cty * stx
            y = d * sty
            z = dsun - d * cos_alpha
            xx[i] = x
            yy[i] = y
            # HCC -> Heliographic Stonyhurst
            r = math.sqrt(x * x + y * y + z * z)
            hgs_z = cos_b0 * y + sin_b0 * z
            hgs_x = cos_b0 * z - sin_b0 * y
            lat_deg[i] = math.asin(hgs_z / r) * _RAD2DEG + 90.0
            lon_deg[i] = (math.atan2(x, hgs_x) + l0) * _RAD2DEG


@njit(cache=True, parallel=True)
def _geometry_kernel_analytic(naxis1, naxis2, cr1, cr2, cd1, cd2,
                              pc00, pc01, pc10, pc11, av1_deg, fp_rad, sin_dp, cos_dp,
                              dsun, rsun, rsun_obs, cos_b0, sin_b0, l0,
                              xx, yy, rr, mu, lat_deg, lon_deg):
    """Fully analytic geometry pass: pixel index -> all geometry arrays.

    Replaces the astropy ``wcs_pix2world`` call in ``pixel_to_hpc`` with an inline
    rotated-TAN inverse (FITS WCS Paper II), then runs the same HPC->HCC->HGS +
    rr/mu math as ``_geometry_kernel``. Valid only for a clean HPLN/HPLT-TAN WCS
    with no distortion (the caller guards this); WCS scalars are passed in
    post-``wcs.wcs.set()`` (cunit normalized to deg). The longitude unwrap and all
    formulas reproduce ``pixel_to_hpc`` to machine precision (~1e-10 arcsec).

    Like ``_geometry_kernel`` this is a pure per-pixel map in ``prange``, so it is
    thread-count invariant. Flat index ``idx`` decomposes as the original
    ``meshgrid(arange(naxis1), arange(naxis2))`` raveling: col=idx%naxis1 (pixel x),
    row=idx//naxis1 (pixel y).
    """
    n = naxis1 * naxis2
    d2 = dsun * dsun - rsun * rsun
    deg_per_rad = _RAD2DEG          # 180/pi (also the TAN constant)
    deg2rad = 1.0 / _RAD2DEG
    for idx in prange(n):
        col = idx % naxis1          # pixel x (naxis1 axis)
        row = idx // naxis1         # pixel y (naxis2 axis)
        q1 = col - (cr1 - 1.0)
        q2 = row - (cr2 - 1.0)

        # pixel -> intermediate projection-plane coords (deg)
        xdeg = cd1 * (pc00 * q1 + pc01 * q2)
        ydeg = cd2 * (pc10 * q1 + pc11 * q2)

        # TAN deprojection -> native spherical (rad)
        Rdeg = math.sqrt(xdeg * xdeg + ydeg * ydeg)
        phi = math.atan2(xdeg, -ydeg)
        theta = math.atan2(deg_per_rad, Rdeg)

        # native -> celestial (Paper II eq 2)
        st = math.sin(theta)
        ct = math.cos(theta)
        dphi = phi - fp_rad
        cdp_ = math.cos(dphi)
        sdp_ = math.sin(dphi)
        ty = math.asin(st * sin_dp + ct * cos_dp * cdp_)            # rad
        alpha_deg = av1_deg + deg_per_rad * math.atan2(
            -ct * sdp_, st * cos_dp - ct * sin_dp * cdp_)

        # unwrap lon to [-180,180) then to radians (matches pixel_to_hpc)
        t = alpha_deg + 180.0
        t = t - 360.0 * math.floor(t / 360.0)
        tx = (t - 180.0) * deg2rad

        # ---- identical to _geometry_kernel from here ----
        ctx = math.cos(tx)
        stx = math.sin(tx)
        cty = math.cos(ty)
        sty = math.sin(ty)

        rho = math.sqrt(tx * tx + ty * ty) * _RAD2ARCSEC / rsun_obs
        rr[idx] = rho
        rho2 = rho * rho
        if rho2 >= 1.0:
            mu[idx] = np.nan
        else:
            mu[idx] = math.sqrt(1.0 - rho2)

        cos_alpha = cty * ctx
        b = dsun * cos_alpha
        disc = b * b - d2
        if disc < 0.0:
            xx[idx] = np.nan
            yy[idx] = np.nan
            lat_deg[idx] = np.nan
            lon_deg[idx] = np.nan
        else:
            d = b - math.sqrt(disc)
            x = d * cty * stx
            y = d * sty
            z = dsun - d * cos_alpha
            xx[idx] = x
            yy[idx] = y
            r = math.sqrt(x * x + y * y + z * z)
            hgs_z = cos_b0 * y + sin_b0 * z
            hgs_x = cos_b0 * z - sin_b0 * y
            lat_deg[idx] = math.asin(hgs_z / r) * _RAD2DEG + 90.0
            lon_deg[idx] = (math.atan2(x, hgs_x) + l0) * _RAD2DEG


def _wcs_is_clean_tan(wcs):
    """True if ``wcs`` is a plain HPLN/HPLT-TAN with no distortion (analytic-safe)."""
    try:
        ctype = [str(c) for c in wcs.wcs.ctype]
    except Exception:
        return False
    return (wcs.sip is None and not wcs.has_distortion
            and len(ctype) == 2
            and ctype[0].endswith("-TAN") and ctype[1].endswith("-TAN"))


@njit(cache=True, parallel=True)
def pixel_area_kernel(lat_deg, lon_deg, out):
    """Per-pixel solar area in microhemispheres, in one fused pass.

    Oracle: ``sdo_image.calculate_pixel_area_numpy``. Operation order is preserved
    (radians via ``deg * pi / 180``, then
    ``sin(lat) * |d_lon| * |d_lat| / (2*pi) * 1e6``), so results are bit-identical.
    Forward differences take a zero edge, matching the oracle's
    ``np.pad(..., mode="constant")``: the last row of d_lat and last column of
    d_lon are 0, giving those pixels zero area.

    Each pixel reads only (i, j), (i+1, j) and (i, j+1), so the prange loop is a
    pure map and thread-count invariant.
    """
    n_row, n_col = lat_deg.shape
    two_pi = 2.0 * math.pi
    for i in prange(n_row):
        for j in range(n_col):
            lr = lat_deg[i, j] * math.pi / 180.0
            if i < n_row - 1:
                d_lat = lat_deg[i + 1, j] * math.pi / 180.0 - lr
            else:
                d_lat = 0.0
            if j < n_col - 1:
                d_lon = lon_deg[i, j + 1] * math.pi / 180.0 - lon_deg[i, j] * math.pi / 180.0
            else:
                d_lon = 0.0
            if d_lat < 0.0:
                d_lat = -d_lat
            if d_lon < 0.0:
                d_lon = -d_lon
            out[i, j] = math.sin(lr) * d_lon * d_lat / two_pi * 1e6


def compute_geometry(wcs, naxis1, naxis2, dsun, rsun, rsun_obs, b0, l0, image_dtype):
    """Compute (xx, yy, rr, mu, lat_deg, lon_deg) arrays for an image grid.

    For a clean HPLN/HPLT-TAN WCS (the SDO case), runs a fully analytic fused
    numba pass (``_geometry_kernel_analytic``) that reproduces the astropy
    ``wcs_pix2world`` deprojection inline -- no astropy in the hot path. For any
    other WCS (distortion/SIP/non-TAN) it falls back to the astropy
    ``pixel_to_hpc`` path + ``_geometry_kernel`` so correctness is never silently
    compromised. Returns plain ndarrays in the same units/dtypes the numpy path
    produced: xx, yy in meters; rr dimensionless; mu in ``image_dtype`` (NaN at the
    limb); lat_deg, lon_deg in degrees (lat already +90).
    """
    n = naxis1 * naxis2
    shape = (naxis2, naxis1)
    xx = np.empty(n, dtype=np.float64)
    yy = np.empty(n, dtype=np.float64)
    rr = np.empty(n, dtype=np.float64)
    lat_deg = np.empty(n, dtype=np.float64)
    lon_deg = np.empty(n, dtype=np.float64)
    mu = np.empty(n, dtype=image_dtype)

    # A clean HPLN/HPLT-TAN WCS (the SDO case) takes the analytic path; anything
    # else falls back to the astropy pixel_to_hpc path (with a warning) so a
    # distortion/SIP/non-TAN WCS is never silently pushed through the analytic
    # kernel, which does not model it.
    if _wcs_is_clean_tan(wcs):
        wcs.wcs.set()  # canonicalize celestial units to deg, populate lonpole
        w = wcs.wcs
        cr1, cr2 = float(w.crpix[0]), float(w.crpix[1])
        cd1, cd2 = float(w.cdelt[0]), float(w.cdelt[1])
        pc = w.get_pc()
        av1, av2 = float(w.crval[0]), float(w.crval[1])
        lonpole = float(w.lonpole)
        if math.isnan(lonpole):
            lonpole = 180.0 if av2 < 90.0 else 0.0
        fp_rad = math.radians(lonpole)
        dp_rad = math.radians(av2)
        _geometry_kernel_analytic(
            int(naxis1), int(naxis2), cr1, cr2, cd1, cd2,
            float(pc[0, 0]), float(pc[0, 1]), float(pc[1, 0]), float(pc[1, 1]),
            av1, fp_rad, math.sin(dp_rad), math.cos(dp_rad),
            float(dsun), float(rsun), float(rsun_obs),
            math.cos(b0), math.sin(b0), float(l0),
            xx, yy, rr, mu, lat_deg, lon_deg)
    else:
        logger.warning("WCS is not a clean HPLN/HPLT-TAN (ctype=%s, distortion=%s); "
                       "using astropy pixel_to_hpc fallback",
                       list(wcs.wcs.ctype), wcs.has_distortion)
        Tx, Ty = pixel_to_hpc(wcs, naxis1, naxis2)
        txf = np.ascontiguousarray(Tx.ravel())
        tyf = np.ascontiguousarray(Ty.ravel())
        _geometry_kernel(txf, tyf, float(dsun), float(rsun), float(rsun_obs),
                         math.cos(b0), math.sin(b0), float(l0),
                         xx, yy, rr, mu, lat_deg, lon_deg)

    return (xx.reshape(shape), yy.reshape(shape), rr.reshape(shape),
            mu.reshape(shape), lat_deg.reshape(shape), lon_deg.reshape(shape))
