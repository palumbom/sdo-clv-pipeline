"""Limb-darkening law and the fused kernels for fitting and removing it."""

import numpy as np
from numba import njit, prange


def quad_darkening(x, a, b, c):
    return a * (1.0 - b * (1.0 - x) - c * (1.0 - x)**2)

def quad_darkening_two(x, b, c):
    return 1.0 - b * (1.0 - x) - c * (1.0 - x)**2


@njit(cache=True)
def bin_index(v, edges, n_bins):
    """Bin index for v, equal to np.clip(np.digitize(v, edges) - 1, 0, n_bins-1).

    np.digitize on increasing edges is np.searchsorted(edges, v, side="right").
    Takes the edges array rather than a bin width: linspace edges are not exactly
    representable, so an arithmetic index puts boundary values in a different bin
    than digitize does.
    """
    lo = 0
    hi = edges.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if v < edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    idx = lo - 1
    if idx < 0:
        idx = 0
    elif idx > n_bins - 1:
        idx = n_bins - 1
    return idx


@njit(cache=True)
def ld_bin_stats(mu, image, edges, mu_lim, n_bins):
    """Per-mu-bin count, sum, and sum of squares over valid pixels, in one pass.

    Oracle: the first half of ``SDOImage.calc_limb_darkening_numpy``. Pixels are
    visited in C order and each bin accumulates sequentially, matching np.bincount
    over a boolean-gathered array, so the sums are bit-identical. Single-threaded
    for that reason.

    Dtypes follow the oracle: ``sums`` accumulates float64(v) because np.bincount
    casts float32 weights to float64, while ``sum2`` accumulates the float32
    square because the oracle forms ``I_valid ** 2`` before the bincount. A NaN mu
    fails ``mu >= mu_lim`` and is skipped, as it is in the oracle.
    """
    n = mu.shape[0]
    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    sum2 = np.zeros(n_bins, dtype=np.float64)
    for i in range(n):
        m = mu[i]
        v = image[i]
        if np.isnan(v) or not (m >= mu_lim):
            continue
        b = bin_index(np.float64(m), edges, n_bins)
        sums[b] += np.float64(v)
        counts[b] += 1.0
        sum2[b] += np.float64(v * v)
    return sums, counts, sum2


@njit(cache=True)
def ld_bin_stats_clipped(mu, image, edges, mu_lim, n_bins, means, stds, n_sigma):
    """Second pass: per-bin count and sum after sigma clipping.

    Mirrors ``abs(I_valid - means[bin_idx]) > n_sigma * stds[bin_idx]``; numpy
    promotes the float32 pixel against the float64 per-bin mean, so the difference
    is formed in float64 here too. ``n_clipped`` lets the caller reproduce the
    oracle's ``if np.any(mask_out)`` branch.
    """
    n = mu.shape[0]
    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    n_clipped = 0
    for i in range(n):
        m = mu[i]
        v = image[i]
        if np.isnan(v) or not (m >= mu_lim):
            continue
        b = bin_index(np.float64(m), edges, n_bins)
        if abs(np.float64(v) - means[b]) > n_sigma * stds[b]:
            n_clipped += 1
            continue
        sums[b] += np.float64(v)
        counts[b] += 1.0
    return sums, counts, n_clipped


@njit(cache=True, parallel=True)
def ld_flatten(mu, image, b, c, ldark_out, iflat_out):
    """Fill ldark = 1 - b(1-mu) - c(1-mu)^2 and iflat = image / ldark.

    Oracle: ``quad_darkening_two(self.mu, b, c)`` followed by ``self.image /
    self.ldark``.

    The mixed precision is required, not incidental. In the oracle ``1.0 - x``
    pairs a Python float (weak under NEP 50) with float32 ``mu``, so that
    subtraction and the square happen in float32; ``b`` and ``c`` are np.float64
    scalars from np.polyfit, which are not weak, so multiplying by them promotes
    to float64. Hence float64 outputs. numba has no weak scalars (a bare 1.0 is
    float64), so the float32 stage needs the explicit casts below.

    Pure per-pixel map, so the prange loop is thread-count invariant.
    """
    n = mu.shape[0]
    one32 = np.float32(1.0)
    for i in prange(n):
        t32 = one32 - mu[i]
        sq32 = t32 * t32
        d = 1.0 - b * np.float64(t32) - c * np.float64(sq32)
        ldark_out[i] = d
        iflat_out[i] = np.float64(image[i]) / d
