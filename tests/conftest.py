"""Shared numerical gates for the equivalence tests."""

import numpy as np


def assert_within_gate(new, old, name, rtol=1e-12, ulp_floor=False):
    """Project gate: rtol with an atol scaled to the reference's own range.

    Scale-aware because v_corr and v_conv cross zero by construction, so a plain
    atol of 0 fails on residual cancellations that carry no physical meaning.
    Dtype and NaN placement are checked first: a changed dtype or a moved mask is
    a logic error, not floating-point noise.

    ``ulp_floor`` raises rtol to a few epsilons of the *stored* dtype. Use it when
    comparing two genuinely different algorithms whose results are stored in a
    narrow type: float32 has eps = 1.2e-7, so the default 1e-12 would demand more
    precision than the array can represent. Fused rewrites of a single algorithm
    do not need it -- they should be bit-identical and pass any tolerance.
    """
    new = np.asarray(new)
    old = np.asarray(old)
    assert new.shape == old.shape, \
        "%s: shape %s != reference %s" % (name, new.shape, old.shape)
    assert new.dtype == old.dtype, \
        "%s: dtype %s != reference %s" % (name, new.dtype, old.dtype)
    assert np.array_equal(np.isnan(new), np.isnan(old)), \
        "%s: NaN pattern moved" % name

    if ulp_floor and np.issubdtype(old.dtype, np.floating):
        rtol = max(rtol, 4.0 * float(np.finfo(old.dtype).eps))

    finite = ~np.isnan(old)
    scale = float(np.max(np.abs(old[finite]))) if finite.any() else 0.0
    np.testing.assert_allclose(new[finite], old[finite], rtol=rtol,
                               atol=rtol * scale,
                               err_msg="%s exceeded the rtol=%g gate" % (name, rtol))
    return None


def assert_bit_identical(new, old, name):
    """Stricter gate for paths that only reorder work, never the arithmetic."""
    new = np.asarray(new)
    old = np.asarray(old)
    assert new.dtype == old.dtype, \
        "%s: dtype %s != reference %s" % (name, new.dtype, old.dtype)
    assert np.array_equal(new, old, equal_nan=True), \
        "%s: must be bit-identical, max|d| = %g" % (
            name, np.nanmax(np.abs(new.astype(float) - old.astype(float))))
    return None
