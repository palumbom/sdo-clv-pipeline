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

import numpy as np
import astropy.units as u


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

    # low-level transform: returns world coords in the WCS's native CUNIT
    lon, lat = wcs.wcs_pix2world(xx.ravel(), yy.ravel(), 0)

    # convert to degrees using the live CUNIT (do not assume deg vs arcsec)
    cunit = wcs.wcs.cunit
    lon = (lon * u.Unit(cunit[0])).to_value(u.deg)
    lat = (lat * u.Unit(cunit[1])).to_value(u.deg)

    # wcs_pix2world reports longitude in [0, 360); helioprojective Tx near the
    # disk is small, so unwrap to a symmetric branch [-180, 180). Without this,
    # disk-center pixels come back as ~360 deg and inflate rr (sin/cos-based
    # quantities are immune, but rr = sqrt(Tx^2+Ty^2) is not).
    lon = (lon + 180.0) % 360.0 - 180.0

    Tx = np.deg2rad(lon).reshape(xx.shape)
    Ty = np.deg2rad(lat).reshape(xx.shape)
    return Tx, Ty


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
    cos_Tx = np.cos(Tx)
    sin_Tx = np.sin(Tx)
    cos_Ty = np.cos(Ty)
    sin_Ty = np.sin(Ty)

    # cos of the angle between disk center and the point (sunpy: cos_alpha)
    cos_alpha = cos_Ty * cos_Tx

    # distance observer->surface via law of cosines, "near" root
    b = dsun * cos_alpha
    disc = b * b - (dsun**2 - rsun**2)
    with np.errstate(invalid="ignore"):
        d = b - np.sqrt(disc)  # NaN where disc < 0 (off-disk)

    # HPC-equivalent Cartesian permuted to HCC, origin shifted observer->Sun
    x = d * cos_Ty * sin_Tx
    y = d * sin_Ty
    z = dsun - d * cos_alpha
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

    r = np.sqrt(x * x + y * y + z * z)
    hgs_x = cos_b0 * z - sin_b0 * y
    hgs_y = x
    hgs_z = cos_b0 * y + sin_b0 * z

    lat = np.arcsin(hgs_z / r)
    lon = np.arctan2(hgs_y, hgs_x) + l0
    return lon, lat
