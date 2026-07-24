"""Fused per-pixel kernels for the dopplergram velocity corrections."""

import math
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def spacecraft_vel_kernel(rr, xx, yy, mask, rsun_solrad,
                          obs_vr, obs_vw, obs_vn, out):
    """Project the spacecraft velocity into the image frame, in one pass.

    Oracle: ``SDOImage.calc_spacecraft_vel_numpy``. The mask is a branch here
    rather than a gather, so no intermediate arrays are built and the astropy
    Quantity wrappers stay out of the loop.

    Arithmetic is float64, stored once into the float32 ``out``, as the oracle
    does when it assigns into a float32 array. Off-mask pixels are NaN.

    Pure per-pixel map, so the prange loop is thread-count invariant.
    """
    n_row, n_col = rr.shape
    for i in prange(n_row):
        for j in range(n_col):
            if not mask[i, j]:
                out[i, j] = np.nan
                continue
            sig = math.atan(rr[i, j] / rsun_solrad)
            chi = math.atan2(xx[i, j], yy[i, j])
            sin_sig = math.sin(sig)
            cos_sig = math.cos(sig)
            sin_chi = math.sin(chi)
            cos_chi = math.cos(chi)

            vr1 = obs_vr * cos_sig
            vr2 = -obs_vw * sin_sig * sin_chi
            vr3 = -obs_vn * sin_sig * cos_chi
            out[i, j] = -(vr1 + vr2 + vr3)
