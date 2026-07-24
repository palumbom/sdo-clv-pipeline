"""Limb-darkening law and the fused kernels for fitting and removing it."""

import numpy as np
from numba import njit, prange


def quad_darkening(x, a, b, c):
    return a * (1.0 - b * (1.0 - x) - c * (1.0 - x)**2)

def quad_darkening_two(x, b, c):
    return 1.0 - b * (1.0 - x) - c * (1.0 - x)**2


@njit(cache=True)
def _bin_of(v, edges, n_bins):
    """Bin index matching np.clip(np.digitize(v, edges) - 1, 0, n_bins - 1).

    np.digitize on increasing edges is np.searchsorted(edges, v, side="right"),
    so this is that search minus one, clipped. The edges array is passed in
    rather than the bin width recomputed, because linspace edges are not exactly
    representable and an arithmetic bin index would put boundary pixels in a
    different bin than digitize does, perturbing the fitted coefficients.
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

    Replaces the boolean mask, two gathers, np.digitize and three np.bincount
    calls in the first half of ``calc_limb_darkening_numpy``. Pixels are visited
    in C order -- the same order boolean-mask gathering preserves -- and each bin
    accumulates sequentially, exactly as np.bincount does, so the sums are
    bit-identical. Single-threaded for that reason.

    Dtype discipline, matching the oracle exactly:
      * ``sums`` accumulates float64(v), as np.bincount casts its float32 weights
        to float64 before accumulating.
      * ``sum2`` accumulates the *float32* square, because the oracle forms
        ``I_valid ** 2`` on the float32 array before handing it to bincount.
      * the validity test mirrors ``(~isnan(I)) & (mu >= mu_lim)``; a NaN mu fails
        ``mu >= mu_lim`` and is skipped, as it is in the oracle.
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
        b = _bin_of(np.float64(m), edges, n_bins)
        sums[b] += np.float64(v)
        counts[b] += 1.0
        sum2[b] += np.float64(v * v)
    return sums, counts, sum2


@njit(cache=True)
def ld_bin_stats_clipped(mu, image, edges, mu_lim, n_bins, means, stds, n_sigma):
    """Second pass: per-bin count and sum after sigma clipping.

    Mirrors ``np.abs(I_valid - means[bin_idx]) > n_sigma * stds[bin_idx]`` from
    the oracle. numpy promotes the float32 pixel against the float64 per-bin
    mean, so the difference is formed in float64 here too. ``n_clipped`` lets the
    caller reproduce the oracle's ``if np.any(mask_out)`` branch exactly.
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
        b = _bin_of(np.float64(m), edges, n_bins)
        if abs(np.float64(v) - means[b]) > n_sigma * stds[b]:
            n_clipped += 1
            continue
        sums[b] += np.float64(v)
        counts[b] += 1.0
    return sums, counts, n_clipped


@njit(cache=True, parallel=True)
def ld_flatten(mu, image, b, c, ldark_out, iflat_out):
    """Fill ldark = 1 - b(1-mu) - c(1-mu)^2 and iflat = image / ldark.

    Replaces ``quad_darkening_two(self.mu, b, c)`` plus ``self.image /
    self.ldark``, which together allocate ~5 full-frame temporaries.

    Dtype discipline here is subtle and load-bearing. In the oracle:
      * ``1.0 - x`` pairs a *Python* float (weak under NEP 50) with the float32
        ``mu``, so that subtraction and the ``** 2`` happen in **float32**;
      * ``b`` and ``c`` are ``np.float64`` scalars from np.polyfit, which are NOT
        weak, so multiplying by them promotes to **float64**;
      * therefore ``ldark`` and ``iflat`` come out float64, not float32.
    numba does not implement NEP 50 weak scalars (a bare 1.0 literal is float64),
    so the float32 stage is forced with explicit np.float32 casts. Getting this
    wrong is a ~1e-7 relative error -- well outside the 1e-12 gate, but easy to
    mistake for harmless noise.

    Pure per-pixel map, so the prange loop is thread-count invariant.
    """
    n = mu.shape[0]
    one32 = np.float32(1.0)
    for i in prange(n):
        t32 = one32 - mu[i]                    # float32, as in the oracle
        sq32 = t32 * t32                       # float32 (1.0 - x) ** 2
        d = 1.0 - b * np.float64(t32) - c * np.float64(sq32)
        ldark_out[i] = d
        iflat_out[i] = np.float64(image[i]) / d
