"""Tests for feature-flag wiring in SunMask.identify_regions.

Drives identify_regions with duck-typed image stand-ins (it reads only a fixed
set of attributes), so no FITS/WCS is needed.
"""

import pytest

pytest.importorskip("numba")   # sdo_image imports numba at module load
pytest.importorskip("skimage")

import numpy as np
from types import SimpleNamespace

from sdo_clv_pipeline.sdo_image import (SunMask, quiet_sun_code, penumbrae_code,
                                        moat_code,
                                        blue_pen_flag, red_pen_flag,
                                        moat_left_flag, moat_right_flag)


def test_predicates_read_the_flag_plane():
    m = object.__new__(SunMask)
    m.regions = np.array([penumbrae_code, penumbrae_code, quiet_sun_code, quiet_sun_code],
                         dtype=float)
    m.flags = np.array([blue_pen_flag, red_pen_flag,
                        moat_left_flag, moat_right_flag], dtype=np.uint8)

    assert list(m.is_blue_penumbra()) == [True, False, False, False]
    assert list(m.is_red_penumbra()) == [False, True, False, False]
    assert list(m.is_left_moat()) == [False, False, True, False]
    assert list(m.is_right_moat()) == [False, False, False, True]
    assert list(m.is_moat_flow()) == [False, False, True, True]


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _spot_scene(h=120, w=120, cy=60, cx=60, r=16, lon_sign=1.0):
    """A central penumbra spot (blueshifted) on a quiet-Sun background.

    The spot is large enough (r=16 -> ~800 px > the 600-px moat threshold) to
    grow a moat ring, which lands on surrounding quiet-Sun pixels.
    """
    disk = _disk(h, w, cy, cx, r)
    ones = np.ones((h, w))
    mu = np.full((h, w), 0.8)
    iflat = np.where(disk, 0.6, 1.0)          # 0.45 < 0.6 <= 0.89 -> penumbra
    v_corr = np.full((h, w), -100.0)          # <= 0 -> blue penumbra
    lon = np.full((h, w), lon_sign)

    con = SimpleNamespace(mu=mu, mu_thresh=0.1, iflat=iflat, image=ones)
    mag = SimpleNamespace(image=ones)
    dop = SimpleNamespace(v_corr=v_corr, pix_area=ones,
                          lon=SimpleNamespace(value=lon))
    aia = SimpleNamespace(iflat=np.zeros((h, w)))

    m = object.__new__(SunMask)
    m.w_active = np.zeros((h, w), dtype=bool)
    m.w_quiet = ~disk
    return m, con, mag, dop, aia, disk


def test_moat_flag_does_not_overwrite_base_label():
    m, con, mag, dop, aia, disk = _spot_scene(lon_sign=1.0)
    # no active pixels in this scene -> the AIA threshold is a benign 0/0
    with np.errstate(invalid="ignore"):
        m.identify_regions(con, mag, dop, aia, classify_moat=True)

    moat = m.is_moat_flow()
    assert moat.any()                                  # a ring was grown
    assert m.is_left_moat().any()                      # lon >= 0 -> left
    assert not m.is_right_moat().any()

    # the headline behavior: moat pixels keep their underlying quiet-Sun code
    # (the old code overwrote them to moat_code, destroying the base label)
    assert np.all(m.regions[moat] == quiet_sun_code)
    assert not np.any(m.regions == moat_code)          # moat is no longer a base code

    # the moat ring lies outside the spot; the penumbra base label is intact
    assert not (moat & m.is_penumbra()).any()
    assert np.array_equal(m.is_penumbra(), disk)


def test_penumbra_velocity_split_sets_blue_flag():
    m, con, mag, dop, aia, disk = _spot_scene()
    with np.errstate(invalid="ignore"):
        m.identify_regions(con, mag, dop, aia, classify_moat=False)

    # every penumbra pixel is blueshifted here -> all carry the blue flag, none red
    assert np.array_equal(m.is_blue_penumbra(), disk)
    assert not m.is_red_penumbra().any()
