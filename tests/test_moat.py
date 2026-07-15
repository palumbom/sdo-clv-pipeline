"""Tests for the pure moat-detection kernel (sdo_clv_pipeline.moat)."""

import pytest

# the kernel needs numpy/scipy/skimage but NOT numba; skip cleanly on the
# numba-free / --no-deps CI job rather than erroring at collection.
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from sdo_clv_pipeline.moat import detect_moats, MoatResult


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _synthetic(spot_r=16, lon_sign=1.0, h=120, w=120, cy=60, cx=60):
    """One circular spot (all penumbra) on an otherwise blank, on-disk frame."""
    penumbra = _disk(h, w, cy, cx, spot_r)
    umbra = np.zeros((h, w), dtype=bool)
    invalid = np.zeros((h, w), dtype=bool)
    mu = np.full((h, w), 0.8)
    lon = np.full((h, w), lon_sign)          # sign -> hemisphere
    v_corr = np.full((h, w), 100.0)
    con = np.ones((h, w))
    mag = np.ones((h, w))
    return dict(v_corr=v_corr, mu=mu, lon=lon, con_image=con, mag_image=mag,
                invalid_mask=invalid, umbra_mask=umbra, penumbra_mask=penumbra)


def test_grows_ring_around_spot():
    syn = _synthetic()
    res = detect_moats(**syn)
    assert isinstance(res, MoatResult)
    assert res.moat_mask.any()
    # the moat is a ring OUTSIDE the spot, never overlapping it
    assert not (res.moat_mask & syn["penumbra_mask"]).any()


def test_larger_radius_factor_grows_more_pixels():
    syn = _synthetic()
    small = detect_moats(**syn, radius_factor=1.2).moat_mask.sum()
    big = detect_moats(**syn, radius_factor=2.0).moat_mask.sum()
    assert big > small


def test_smaller_shrink_grows_fewer_pixels():
    syn = _synthetic()
    base = detect_moats(**syn, shrink=0.97).moat_mask.sum()
    less = detect_moats(**syn, shrink=0.5).moat_mask.sum()
    assert less < base


def test_hemisphere_split_left():
    res = detect_moats(**_synthetic(lon_sign=1.0))
    assert res.left_moat.any()
    assert not res.right_moat.any()
    assert np.array_equal(res.moat_mask, res.left_moat)


def test_hemisphere_split_right():
    res = detect_moats(**_synthetic(lon_sign=-1.0))
    assert res.right_moat.any()
    assert not res.left_moat.any()
    assert np.array_equal(res.moat_mask, res.right_moat)


def test_area_threshold_excludes_small_spots():
    # r=5 -> ~81 px, below the default 600 px threshold
    res = detect_moats(**_synthetic(spot_r=5))
    assert not res.moat_mask.any()
    assert res.profiles == []


def test_profiles_have_matching_ring_axis():
    res = detect_moats(**_synthetic())
    assert len(res.profiles) == 1
    p = res.profiles[0]
    assert len(p["v_corr"]) == len(p["rings"])
    assert len(p["abs_mag"]) == len(p["rings"])
    assert len(p["intensity"]) == len(p["rings"])
    assert p["rings"][0] == 1
    assert p["hemisphere"] == "left"
    # v_corr is a constant 100 everywhere -> cumulative average stays 100
    assert np.allclose(p["v_corr"], 100.0)
