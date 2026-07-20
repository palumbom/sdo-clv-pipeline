"""Weighted mean/std/SEM for combining many epoch- or feature-level rows."""

import numpy as np


def weighted_stats(vals, weights):
    """Weighted mean, std, and standard error of ``vals`` given relative ``weights``.

    Uses the Kish/Cochran reliability-weights formalism, which reduces exactly to
    the ordinary unbiased sample mean/``std(ddof=1)``/SEM when all weights are
    equal:

    mean = sum(w*x)/sum(w)
    std  = sqrt(sum(w*(x-mean)**2) / (V1 - V2/V1))   -- bias-corrected
    err  = std / sqrt(n_eff),  n_eff = V1**2/V2       -- Kish effective sample size

    where V1 = sum(weights), V2 = sum(weights**2). Returns (mean, 0.0, 0.0) when
    n_eff <= 1 (e.g. a single row).
    """
    w = np.asarray(weights, dtype=float)
    x = np.asarray(vals, dtype=float)
    assert len(w) == len(x), "vals and weights must be the same length"
    assert np.all(w >= 0), "weights must be non-negative"
    v1 = w.sum()
    assert v1 > 0, "weights must not sum to zero"

    mean = np.average(x, weights=w)
    v2 = np.sum(w ** 2)
    denom = v1 - v2 / v1
    if denom <= 0:
        return mean, 0.0, 0.0

    std = np.sqrt(np.sum(w * (x - mean) ** 2) / denom)
    n_eff = v1 ** 2 / v2
    return mean, std, std / np.sqrt(n_eff)
