"""The shared aggregation context must not change any aggregation output.

The fused kernels change how pixels are visited, not the arithmetic, so their
rows must be *bit-identical* to the np.bincount path rather than merely within
tolerance -- a near-miss is a bug, not floating-point noise. The one exception is
the region-only rows, which sum ring partials and so reassociate; those are held
to the 1e-12 tolerance.
"""

import pytest

pytest.importorskip("numba")

import numpy as np

from sdo_clv_pipeline.aggregate import build_agg_inputs
from sdo_clv_pipeline.sdo_image import (region_codes, quiet_sun_code,
                                        flag_selections)
from sdo_clv_pipeline.sdo_vels import (shared_products, compute_disk_results,
                                       compute_region_only_results,
                                       compute_region_results,
                                       compute_region_only_flag_results,
                                       compute_region_flag_results)

MU_THRESH = 0.1
N_RINGS = 10


def _toy(n=997):
    """A pseudo-random frame with every region code and a spread of mu."""
    rng = np.random.default_rng(7)
    flat_mu = rng.uniform(0.0, 1.0, n).astype(np.float32)
    flat_mu[::50] = np.nan
    flat_int = rng.uniform(1.0e4, 6.0e4, n).astype(np.float32)
    flat_iflat = rng.uniform(0.5, 1.2, n).astype(np.float64)
    flat_v_corr = rng.normal(0.0, 300.0, n).astype(np.float32)
    flat_v_rot = rng.normal(0.0, 2000.0, n).astype(np.float32)
    flat_ld = rng.uniform(0.5, 1.0, n).astype(np.float64)
    flat_abs_mag = np.abs(rng.normal(0.0, 60.0, n)).astype(np.float32)
    flat_reg = rng.choice(np.array(region_codes, dtype=np.float32), n)
    flat_flags = rng.integers(0, 16, n).astype(np.uint8)
    flat_w_quiet = flat_reg == quiet_sun_code
    flat_w_active = ~flat_w_quiet
    k_hat = 1.03
    p_vhat, p_vphot, p_mag = shared_products(flat_int, flat_v_corr, flat_v_rot,
                                             flat_ld, flat_w_active,
                                             flat_abs_mag, k_hat)
    return dict(flat_mu=flat_mu, flat_int=flat_int, flat_iflat=flat_iflat,
                flat_v_corr=flat_v_corr, flat_v_rot=flat_v_rot, flat_ld=flat_ld,
                flat_abs_mag=flat_abs_mag, flat_reg=flat_reg,
                flat_flags=flat_flags, flat_w_quiet=flat_w_quiet,
                flat_w_active=flat_w_active, k_hat=k_hat,
                p_vhat=p_vhat, p_vphot=p_vphot, p_mag=p_mag)


def _agg(d):
    return build_agg_inputs(d["flat_mu"], d["flat_int"], region_codes,
                            MU_THRESH, N_RINGS)


def _rows_within_gate(a, b, name, rtol=1e-12):
    """Project gate: rtol 1e-12 with an atol scaled to each column's range."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "%s: shape %s != %s" % (name, a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "%s: NaN pattern moved" % name
    f = ~np.isnan(b)
    scale = float(np.max(np.abs(b[f]))) if f.any() else 0.0
    np.testing.assert_allclose(a[f], b[f], rtol=rtol, atol=rtol * scale,
                               err_msg="%s exceeded the 1e-12 gate" % name)
    return None


def _rows_equal(a, b, name):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "%s: shape %s != %s" % (name, a.shape, b.shape)
    assert np.array_equal(a, b, equal_nan=True), \
        "%s: must be bit-identical, max|d| = %g" % (
            name, np.nanmax(np.abs(a - b)))
    return None


def test_disk_row_bit_identical_with_context():
    d = _toy()
    common = (1.0, d["flat_mu"], d["flat_int"], d["flat_v_corr"], d["flat_v_rot"],
              d["flat_ld"], d["flat_iflat"], d["flat_w_quiet"], d["flat_w_active"],
              d["flat_abs_mag"], MU_THRESH, d["k_hat"])
    kw = dict(p_vhat=d["p_vhat"], p_vphot=d["p_vphot"], p_mag=d["p_mag"])
    old = compute_disk_results(*common, **kw)
    new = compute_disk_results(*common, agg=_agg(d), **kw)
    _rows_equal(new, old, "disk row")
    return None


def test_region_only_rows_bit_identical_with_context():
    d = _toy()
    common = (1.0, d["flat_mu"], d["flat_int"], d["flat_v_corr"], d["flat_v_rot"],
              d["flat_ld"], d["flat_iflat"], d["flat_abs_mag"], d["flat_w_quiet"],
              d["flat_w_active"], d["flat_reg"], region_codes, MU_THRESH,
              d["k_hat"])
    kw = dict(p_vhat=d["p_vhat"], p_vphot=d["p_vphot"], p_mag=d["p_mag"])
    old = compute_region_only_results(*common, **kw)
    new = compute_region_only_results(*common, agg=_agg(d), **kw)
    # the only intentionally non-exact path: these rows sum the shared (ring,
    # region) accumulator over rings instead of accumulating once over pixels, so
    # they are held to the 1e-12 project gate rather than bit-identity
    _rows_within_gate(new, old, "region-only rows")
    return None


def test_region_ring_rows_bit_identical_with_context():
    d = _toy()
    common = (1.0, d["flat_mu"], d["flat_int"], d["flat_v_corr"], d["flat_v_rot"],
              d["flat_ld"], d["flat_iflat"], d["flat_abs_mag"], d["flat_w_quiet"],
              d["flat_w_active"], d["flat_reg"], region_codes, MU_THRESH, N_RINGS,
              d["k_hat"])
    kw = dict(p_vhat=d["p_vhat"], p_vphot=d["p_vphot"], p_mag=d["p_mag"])
    old_rows, old_vq, old_bins = compute_region_results(*common, **kw)
    new_rows, new_vq, new_bins = compute_region_results(*common, agg=_agg(d), **kw)
    _rows_equal(new_rows, old_rows, "region x ring rows")
    _rows_equal(new_vq, old_vq, "per-ring quiet reference")
    _rows_equal(new_bins, old_bins, "ring edges")
    return None


def test_flag_rows_bit_identical_with_context():
    """Flag rows must be bit-identical: the mu-binned ones fuse (bincount upcasts
    to float64) and the disk-level ones stay on the numpy path on purpose."""
    d = _toy()
    agg = _agg(d)
    sel = flag_selections(d["flat_reg"], d["flat_flags"])
    quiet_ref = -123.0

    old = compute_region_only_flag_results(
        1.0, d["flat_mu"], d["flat_int"], d["flat_iflat"], sel, MU_THRESH,
        quiet_ref, d["p_vhat"], d["p_vphot"], d["p_mag"])
    new = compute_region_only_flag_results(
        1.0, d["flat_mu"], d["flat_int"], d["flat_iflat"], sel, MU_THRESH,
        quiet_ref, d["p_vhat"], d["p_vphot"], d["p_mag"], agg=agg)
    _rows_equal(new, old, "flag rows (no mu bins)")

    ref_by_bin = np.linspace(-50.0, 50.0, N_RINGS - 1)
    old_r = compute_region_flag_results(
        1.0, d["flat_mu"], d["flat_int"], d["flat_iflat"], sel, MU_THRESH,
        N_RINGS, ref_by_bin, d["p_vhat"], d["p_vphot"], d["p_mag"])
    new_r = compute_region_flag_results(
        1.0, d["flat_mu"], d["flat_int"], d["flat_iflat"], sel, MU_THRESH,
        N_RINGS, ref_by_bin, d["p_vhat"], d["p_vphot"], d["p_mag"], agg=agg)
    _rows_equal(new_r, old_r, "flag x ring rows")
    return None
