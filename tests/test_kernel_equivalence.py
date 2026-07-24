"""Fast-path kernels must reproduce their retained numpy oracles.

These run on small synthetic frames so they need no FITS data and gate every CI
run. Tolerance mirrors the project gate: rtol 1e-12 with a scale-aware atol.
"""

import pytest

pytest.importorskip("numba")

import numpy as np
import astropy.units as u

from conftest import assert_within_gate as assert_close


def _synthetic_latlon(n=64):
    """Smooth, monotonic lat/lon grids shaped like a real disk quadrant."""
    lat = np.linspace(30.0, 150.0, n)[:, None] + np.zeros((1, n))
    lon = np.zeros((n, 1)) + np.linspace(-60.0, 60.0, n)[None, :]
    # add a small smooth perturbation so d_lat/d_lon are not constant
    lat = lat + 0.01 * np.sin(np.linspace(0, 3, n))[None, :]
    lon = lon + 0.01 * np.cos(np.linspace(0, 3, n))[:, None]
    return lat * u.deg, lon * u.deg


def test_pixel_area_matches_numpy_oracle():
    from sdo_clv_pipeline.sdo_image import (calculate_pixel_area,
                                            calculate_pixel_area_numpy)
    lat, lon = _synthetic_latlon()
    assert_close(calculate_pixel_area(lat, lon),
                 calculate_pixel_area_numpy(lat, lon), "pix_area")
    return None


def test_pixel_area_is_bit_identical():
    """Elementwise fusion preserves op order, so expect exactness, not tolerance."""
    from sdo_clv_pipeline.sdo_image import (calculate_pixel_area,
                                            calculate_pixel_area_numpy)
    lat, lon = _synthetic_latlon()
    assert np.array_equal(calculate_pixel_area(lat, lon),
                          calculate_pixel_area_numpy(lat, lon), equal_nan=True), \
        "pix_area must be bit-identical to the oracle"
    return None


def _synthetic_dop(n=64):
    """A minimal SDOImage exposing the attributes calc_spacecraft_vel reads."""
    from sdo_clv_pipeline.sdo_image import SDOImage

    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    xx = (xx - n / 2.0) / (n / 2.0)
    yy = (yy - n / 2.0) / (n / 2.0)
    rr = np.hypot(xx, yy)
    mu = np.sqrt(np.clip(1.0 - rr ** 2, 0.0, None)).astype(np.float32)
    mu[rr >= 1.0] = np.nan

    img = SDOImage.__new__(SDOImage)          # bypass FITS I/O
    img.content = "DOPPLERGRAM"
    img.filename = "synthetic"
    img.image = (rng.standard_normal((n, n)) * 500.0).astype(np.float32)
    img.mu = mu
    img.rr = rr * u.dimensionless_unscaled
    img.xx = (xx * 7.0e8) * u.m
    img.yy = (yy * 7.0e8) * u.m
    img.rsun_solrad = 215.0
    img.obs_vr, img.obs_vw, img.obs_vn = 3000.0, -120.0, 45.0
    img.mask_nan = (img.mu >= 0.1)
    return img


def test_spacecraft_vel_matches_numpy_oracle():
    img = _synthetic_dop()
    img.calc_spacecraft_vel()
    fast = img.v_obs.copy()
    img.calc_spacecraft_vel_numpy()
    assert_close(fast, img.v_obs, "v_obs")
    return None


def _synthetic_continuum(n=64):
    """A continuum-like SDOImage with a plausible limb-darkened profile."""
    from sdo_clv_pipeline.sdo_image import SDOImage

    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    xx = (xx - n / 2.0) / (n / 2.0)
    yy = (yy - n / 2.0) / (n / 2.0)
    rr = np.hypot(xx, yy)
    mu = np.sqrt(np.clip(1.0 - rr ** 2, 0.0, None)).astype(np.float32)
    mu[rr >= 1.0] = np.nan

    # limb-darkened continuum plus noise plus a few dark outliers to exercise
    # the sigma-clipping branch
    base = 5.0e4 * (1.0 - 0.4 * (1.0 - mu) - 0.2 * (1.0 - mu) ** 2)
    img_arr = (base + rng.standard_normal((n, n)) * 300.0).astype(np.float32)
    img_arr[n // 3, n // 3] = 1.0e3
    img_arr[n // 2, n // 4] = 1.0e3

    img = SDOImage.__new__(SDOImage)
    img.content = "CONTINUUM INTENSITY"
    img.filename = "synthetic"
    img.image = img_arr
    img.mu = mu
    img.mu_thresh = 0.0
    return img


def test_limb_darkening_matches_numpy_oracle():
    fast = _synthetic_continuum()
    fast.calc_limb_darkening()
    slow = _synthetic_continuum()
    slow.calc_limb_darkening_numpy()

    assert_close(fast.ld_coeffs, slow.ld_coeffs, "ld_coeffs")
    assert_close(fast.ldark, slow.ldark, "ldark")
    assert_close(fast.iflat, slow.iflat, "iflat")
    return None


def test_limb_darkening_flatten_is_bit_identical():
    """The flatten half is fixed-order elementwise math -- expect exactness.

    Also pins the dtype: ldark/iflat are float64 because polyfit returns
    np.float64 coefficients, which are strong under NEP 50 and promote the
    float32 mu. A float32 result here would be a ~1e-7 error.
    """
    fast = _synthetic_continuum()
    fast.calc_limb_darkening()
    slow = _synthetic_continuum()
    slow.calc_limb_darkening_numpy()
    assert fast.ldark.dtype == np.float64, "ldark must be float64"
    assert fast.iflat.dtype == np.float64, "iflat must be float64"
    assert np.array_equal(fast.ldark, slow.ldark, equal_nan=True), \
        "ldark must be bit-identical (float32/float64 sequence preserved)"
    assert np.array_equal(fast.iflat, slow.iflat, equal_nan=True), \
        "iflat must be bit-identical"
    return None
