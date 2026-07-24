"""Tests for non-exclusive feature-flag aggregation in region output."""

import pytest

# sdo_vels imports sdo_image -> numba at import time; skip on the numba-free CI job.
pytest.importorskip("numba")

import numpy as np
from sdo_clv_pipeline.sdo_vels import (compute_region_only_flag_results,
                                       compute_region_flag_results)
from sdo_clv_pipeline.sdo_image import (flag_selections, plage_code,
                                        network_code, quiet_sun_code,
                                        moat_code, plage_no_moat_code,
                                        network_no_moat_code,
                                        moat_left_flag, moat_right_flag)

# column positions in a region_output row (pre-quality_flag), matching header_region.
COL = {name: i for i, name in enumerate([
    "mjd", "region", "lo_mu", "hi_mu", "pixel_frac", "light_frac",
    "v_hat", "v_phot", "v_quiet", "v_conv", "mag_unsigned",
    "avg_int", "avg_int_flat"])}


def _flat_epoch():
    """A 4-pixel, all-on-disk (mu=1) frame with hand-computable weighted sums.

    Two overlapping selections share pixel 1 so the double-count behavior is
    observable: selection A = pixels {0,1}, selection B = pixels {1,2}.
    """
    flat_mu = np.array([1.0, 1.0, 1.0, 1.0])
    flat_int = np.array([1.0, 1.0, 1.0, 1.0])
    flat_iflat = np.array([0.5, 0.5, 0.5, 0.5])
    p_vhat = np.array([2.0, 4.0, 6.0, 8.0])   # int*v_corr; here int=1 so == v_corr
    p_vphot = np.array([1.0, 1.0, 1.0, 1.0])
    p_mag = np.array([10.0, 20.0, 30.0, 40.0])
    sel_a = np.array([True, True, False, False])   # pixels 0,1
    sel_b = np.array([False, True, True, False])   # pixels 1,2  (overlap at 1)
    return dict(mjd=100.0, flat_mu=flat_mu, flat_int=flat_int,
                flat_iflat=flat_iflat,
                selections=[(6, sel_a), (5, sel_b)],
                mu_thresh=0.1, quiet_ref=1.0,
                p_vhat=p_vhat, p_vphot=p_vphot, p_mag=p_mag)


def test_overlapping_selections_double_count_shared_pixel():
    rows = compute_region_only_flag_results(**_flat_epoch())
    by_code = {r[COL["region"]]: r for r in rows}
    assert set(by_code) == {6, 5}

    a, b = by_code[6], by_code[5]
    # both selections have 2 pixels; pixel 1 is counted in BOTH
    assert a[COL["pixel_frac"]] == pytest.approx(2.0 / 4.0)
    assert b[COL["pixel_frac"]] == pytest.approx(2.0 / 4.0)


def test_flag_row_velocity_and_intensity_values():
    rows = compute_region_only_flag_results(**_flat_epoch())
    a = next(r for r in rows if r[COL["region"]] == 6)   # pixels 0,1

    assert a[COL["mjd"]] == 100.0
    assert np.isnan(a[COL["lo_mu"]]) and np.isnan(a[COL["hi_mu"]])
    assert a[COL["light_frac"]] == pytest.approx(2.0 / 4.0)   # sum_int=2, total=4
    assert a[COL["v_hat"]] == pytest.approx((2.0 + 4.0) / 2.0)   # 3.0
    assert a[COL["v_phot"]] == pytest.approx((1.0 + 1.0) / 2.0)  # 1.0
    assert a[COL["mag_unsigned"]] == pytest.approx((10.0 + 20.0) / 2.0)  # 15.0
    assert a[COL["avg_int"]] == pytest.approx(1.0)              # 2/2
    assert a[COL["avg_int_flat"]] == pytest.approx(0.5)         # 1.0/2

    # flags are never the quiet reference: v_quiet column is 0, v_conv = v_hat - quiet_ref
    assert a[COL["v_quiet"]] == pytest.approx(0.0)
    assert a[COL["v_conv"]] == pytest.approx(3.0 - 1.0)


def test_empty_selection_emits_zero_row_not_nan():
    epoch = _flat_epoch()
    epoch["selections"] = [(6, np.zeros(4, dtype=bool))]
    rows = compute_region_only_flag_results(**epoch)
    r = rows[0]
    assert r[COL["region"]] == 6
    assert r[COL["pixel_frac"]] == pytest.approx(0.0)
    assert r[COL["v_hat"]] == pytest.approx(0.0)      # no div-by-zero -> 0
    assert r[COL["v_conv"]] == pytest.approx(0.0)     # v_hat==0 -> conv 0


def _binned_epoch():
    """4 pixels split across 2 mu rings; one selection spanning both rings.

    With mu_thresh=0.1, n_rings=3 -> bins [0.1, 0.55, 1.0]:
      pixels 0,1 (mu=0.3) fall in ring 0; pixel 2 (mu=0.8) in ring 1.
    Selection A = pixels {0,1,2}. Per-ring quiet reference = [1.0, 2.0].
    """
    flat_mu = np.array([0.3, 0.3, 0.8, 0.8])
    flat_int = np.array([1.0, 1.0, 1.0, 1.0])
    flat_iflat = np.array([0.5, 0.5, 0.5, 0.5])
    p_vhat = np.array([2.0, 4.0, 6.0, 8.0])
    p_vphot = np.array([1.0, 1.0, 1.0, 1.0])
    p_mag = np.array([10.0, 20.0, 30.0, 40.0])
    sel_a = np.array([True, True, True, False])
    return dict(mjd=100.0, flat_mu=flat_mu, flat_int=flat_int,
                flat_iflat=flat_iflat, selections=[(6, sel_a)],
                mu_thresh=0.1, n_rings=3, quiet_ref_by_bin=np.array([1.0, 2.0]),
                p_vhat=p_vhat, p_vphot=p_vphot, p_mag=p_mag)


def test_mu_binned_flag_rows_split_by_ring():
    rows = compute_region_flag_results(**_binned_epoch())
    assert len(rows) == 2   # 1 selection x (n_rings-1) bins

    ring0 = next(r for r in rows if r[COL["lo_mu"]] == pytest.approx(0.1))
    ring1 = next(r for r in rows if r[COL["lo_mu"]] == pytest.approx(0.55))

    assert ring0[COL["region"]] == 6
    assert ring0[COL["hi_mu"]] == pytest.approx(0.55)
    assert ring0[COL["pixel_frac"]] == pytest.approx(2.0 / 4.0)   # pixels 0,1
    assert ring0[COL["v_hat"]] == pytest.approx((2.0 + 4.0) / 2.0)   # 3.0
    assert ring0[COL["v_conv"]] == pytest.approx(3.0 - 1.0)          # quiet_ref[0]

    assert ring1[COL["hi_mu"]] == pytest.approx(1.0)
    assert ring1[COL["pixel_frac"]] == pytest.approx(1.0 / 4.0)      # pixel 2
    assert ring1[COL["v_hat"]] == pytest.approx(6.0 / 1.0)           # 6.0
    assert ring1[COL["v_conv"]] == pytest.approx(6.0 - 2.0)          # quiet_ref[1]
    assert ring1[COL["v_quiet"]] == pytest.approx(0.0)


def test_flag_selections_no_moat_variants_exclude_moat_overlap():
    #             plage   plage+moat   network+moat   quiet
    flat_reg = np.array([plage_code, plage_code, network_code, quiet_sun_code],
                        dtype=float)
    flat_flags = np.array([0, moat_left_flag, moat_right_flag, 0], dtype=np.uint8)
    sels = dict(flag_selections(flat_reg, flat_flags))

    # moat (any hemisphere) = the two flagged pixels, regardless of base class
    assert list(sels[moat_code]) == [False, True, True, False]
    # plage-not-moat drops the plage pixel that is also a moat
    assert list(sels[plage_no_moat_code]) == [True, False, False, False]
    # network-not-moat drops the network pixel that is also a moat
    assert list(sels[network_no_moat_code]) == [False, False, False, False]


def _bare_mask(n=8):
    """A SunMask with only the fields the area-fraction properties read.

    Bypasses __init__ (which needs four real SDOImages). Layout: a 8x8 frame with
    a 2-pixel umbra, a 2-pixel penumbra, one plage pixel, one network pixel, the
    rest quiet sun, and a ring of off-disk pixels excluded by mu_thresh.
    """
    from sdo_clv_pipeline.sdo_image import (SunMask, umbrae_code, penumbrae_code,
                                            plage_code, network_code,
                                            quiet_sun_code, blue_pen_flag)
    mask = SunMask.__new__(SunMask)
    mask.mu = np.full((n, n), 0.5, dtype=np.float32)
    mask.mu[0, :] = 0.0                       # one row below threshold
    mask.mu_thresh = 0.1
    mask.regions = np.full((n, n), quiet_sun_code, dtype=np.float32)
    mask.regions[0, :] = np.nan               # off-disk
    mask.regions[1, 0:2] = umbrae_code
    mask.regions[2, 0:2] = penumbrae_code
    mask.regions[3, 0] = plage_code
    mask.regions[3, 1] = network_code
    mask.flags = np.zeros((n, n), dtype=np.uint8)
    mask.flags[2, 0:2] = blue_pen_flag
    mask.w_active = np.zeros((n, n), dtype=bool)
    mask.w_active[1, 0:2] = True
    return mask


def test_area_fraction_properties_partition_the_disk():
    """The five base-region fractions must sum to exactly 1 over on-disk pixels."""
    mask = _bare_mask()
    total = (mask.umb_frac + mask.pen_frac + mask.quiet_frac
             + mask.network_frac + mask.plage_frac)
    assert total == 1.0, "base region fractions must partition the disk, got %r" % total
    assert mask.npix == 56          # 8x8 minus the one off-disk row
    assert mask.umb_frac == 2 / 56
    assert mask.plage_frac == 1 / 56
    return None


def test_area_fraction_properties_are_lazy_not_attributes():
    """They are computed on access, so a stale cached value cannot be served."""
    from sdo_clv_pipeline.sdo_image import SunMask, umbrae_code
    mask = _bare_mask()
    before = mask.umb_frac
    mask.regions[4, 0:2] = umbrae_code        # two more umbra pixels
    assert mask.umb_frac > before, "property did not reflect the updated region map"
    assert isinstance(SunMask.umb_frac, property)
    assert mask.ff == 2 / 56                  # w_active covers the 2 umbra pixels
    return None
