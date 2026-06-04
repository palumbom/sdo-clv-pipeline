# code modified from https://github.com/samarth-kashyap/hmi-clean-ls
# original author: Samarth Ganesh Kashyap (g.samarth@tifr.res.in)
# original associated publication: https://arxiv.org/abs/2105.12055

"""Legendre basis helpers used for dopplergram bulk-velocity fits."""

import math
import numpy as np
from math import pi
from scipy.special import eval_legendre
from numba import njit, prange


def gen_leg_vec(lmax, theta):
    """Generate normalized Legendre basis and derivative vs latitude.

    This is used by dopplergram bulk-velocity modeling to build design matrices.

    Parameters
    ----------
    lmax : int
        Maximum degree of the Legendre series.
    theta : astropy.units.Quantity
        Colatitude in degrees.
    """
    # build cos, sin and l arrays
    theta = theta.value * np.pi / 180.0
    cost = np.cos(theta)
    sint = np.sin(theta)
    ell = np.arange(lmax+1)

    # l(l+1)–scaling (avoid div 0 at l = 0)
    norm_l = np.sqrt(ell*(ell+1))
    norm_l[0] = 1

    # shtools "geodesy" normalization sqrt(2l+1)
    norm_sht = np.sqrt(2*ell + 1)

    # evaluate P_l(cost) for all l and all theta
    P = eval_legendre(ell[:,None], cost[None,:])

    # compute dP_l / dz via (1–z^2)P'_l(z) = l[P_(l-1)(z) – z P_l(z)] (recurrence relation)
    dP = np.zeros_like(P)
    z = cost[None,:]
    l = ell[1:,None]
    dP[1:,:] = l * (P[:-1,:] - z*P[1:,:]) / (1.0 - z*z)

    # apply sqrt(2l+1) normalization
    leg = P * norm_sht[:,None]
    leg_dz = dP * norm_sht[:,None]

    # final sqrt2 and sqrtl(l+1) scalings, and change of variable d theta = -sin theta dz
    leg_out = leg / np.sqrt(2) / norm_l[:,None]
    leg_d1_out = -sint[None,:] * leg_dz / np.sqrt(2) / norm_l[:,None]
    return leg_out, leg_d1_out


def gen_leg_x_vec(lmax, x):
    """Generate normalized Legendre basis and derivative vs x.

    This is used by dopplergram bulk-velocity modeling to build design matrices.

    Parameters
    ----------
    lmax : int
        Maximum degree of the Legendre series.
    x : astropy.units.Quantity
        Dimensionless disk-plane radial coordinate (rho in [0, 1]). Used directly
        as the Legendre argument, matching the upstream hmi-clean-ls gen_leg_x.
        (A previous version erroneously scaled this by pi/180 as if it were an
        angle in degrees, which collapsed the argument to ~[0, 0.017] and made
        the fit_cbs=True normal-equations system singular; cond ~3e18.)
    """
    # rho is already the (dimensionless) Legendre argument; no angle conversion
    x = x.value

    # degree index l=0…lmax
    ell = np.arange(lmax+1)

    # l(l+1)–scaling (avoid div 0 at l = 0)
    norm_l = np.sqrt(ell*(ell+1))
    norm_l[0] = 1

    # shtools "geodesy" normalization sqrt(2l+1)
    norm_sht = np.sqrt(2*ell + 1)

    # evaluate all P_l(x)
    P = eval_legendre(ell[:,None], x[None,:])

    # compute dP_l/dx via (1−x^2) dP_l/dx = l [P_(l-1)(x) − x P_l(x)]
    dP = np.zeros_like(P)
    l = ell[1:,None]
    z = x[None,:]
    dP[1:,:] = l * (P[:-1,:] - z*P[1:,:])/(1 - z*z)

    # apply sqrt(2l+1) normalization
    leg = P * norm_sht[:,None]
    leg_dx = dP * norm_sht[:,None]

    # final sqrt2 and sqrtl(l+1) scalings
    leg_out = leg / np.sqrt(2) / norm_l[:,None]
    leg_d1_out = leg_dx / np.sqrt(2) / norm_l[:,None]
    return leg_out, leg_d1_out


# ---------------------------------------------------------------------------
# Fused numba kernels for the dopplergram bulk-velocity fit.
#
# These reproduce, in a single per-pixel pass, the design-matrix construction in
# SDOImage.calc_bulk_vel: the theta basis from gen_leg_vec (derivative form,
# degrees l=1..5) and the rho basis from gen_leg_x_vec (value form, l=0..5),
# combined with the projection factors lt, lp. Instead of materializing the
# (n_poly x N) design matrix and ~6 (6 x N) Legendre intermediates, we directly
# accumulate the normal-equations A (n_poly x n_poly) and RHS (n_poly), then a
# second pass reconstructs the velocity components. Single-threaded, so it is
# safe under the per-epoch multiprocessing pool.
#
# Normalizations match gen_leg_vec / gen_leg_x_vec exactly:
#   norm_l   = sqrt(l(l+1)), with norm_l[0] := 1
#   norm_sht = sqrt(2l+1)
#   basis scaling = norm_sht[l] / (sqrt(2) * norm_l[l])
# theta basis additionally carries the -sin(theta) factor and uses the Legendre
# derivative dP_l/dz; rho basis uses P_l(x) directly with x = rho * pi/180.
# ---------------------------------------------------------------------------

# precomputed norm_sht[l] / (sqrt(2)*norm_l[l]) for l = 0..5
_SQRT2 = math.sqrt(2.0)
basis_scale = np.array([
    1.0 / (_SQRT2 * 1.0),                        # l=0 (norm_l[0] := 1)
    math.sqrt(3.0) / (_SQRT2 * math.sqrt(2.0)),  # l=1
    math.sqrt(5.0) / (_SQRT2 * math.sqrt(6.0)),  # l=2
    math.sqrt(7.0) / (_SQRT2 * math.sqrt(12.0)), # l=3
    3.0 / (_SQRT2 * math.sqrt(20.0)),            # l=4 (sqrt(9)=3)
    math.sqrt(11.0) / (_SQRT2 * math.sqrt(30.0)),# l=5
])


@njit(cache=True, inline="always")
def _bulk_vel_basis(lat_deg, lon_deg, rho, cos_B0, sin_B0, n_poly, scale, b):
    """Fill b[0:n_poly] with the bulk-velocity design row for one pixel."""
    deg2rad = math.pi / 180.0

    theta = lat_deg * deg2rad
    cost = math.cos(theta)
    sint = math.sin(theta)
    phi = lon_deg * deg2rad
    cphi = math.cos(phi)
    sphi = math.sin(phi)

    lt = sin_B0 * sint - cos_B0 * cost * cphi
    lp = cos_B0 * sphi

    # Legendre P_l(cost), l=0..5 (standard recurrence, matches eval_legendre)
    p0 = 1.0
    p1 = cost
    p2 = (3.0 * cost * p1 - p0) / 2.0
    p3 = (5.0 * cost * p2 - 2.0 * p1) / 3.0
    p4 = (7.0 * cost * p3 - 3.0 * p2) / 4.0
    p5 = (9.0 * cost * p4 - 4.0 * p3) / 5.0

    # derivative dP_l/dz = l*(P_{l-1} - z*P_l)/(1 - z^2), z = cost
    den = 1.0 - cost * cost
    dp1 = 1.0 * (p0 - cost * p1) / den
    dp2 = 2.0 * (p1 - cost * p2) / den
    dp3 = 3.0 * (p2 - cost * p3) / den
    dp4 = 4.0 * (p3 - cost * p4) / den
    dp5 = 5.0 * (p4 - cost * p5) / den

    # theta basis dt_pl_theta[l] = -sint * dP_l * scale[l]
    s = -sint
    dt1 = s * dp1 * scale[1]
    dt2 = s * dp2 * scale[2]
    dt3 = s * dp3 * scale[3]
    dt4 = s * dp4 * scale[4]
    dt5 = s * dp5 * scale[5]

    b[0] = dt1 * lp
    b[1] = dt3 * lp
    b[2] = dt5 * lp
    b[3] = dt2 * lt
    b[4] = dt4 * lt

    if n_poly > 5:
        # rho is the dimensionless Legendre argument directly (see gen_leg_x_vec)
        x = rho
        px0 = 1.0
        b[5] = px0 * scale[0]
        if n_poly > 6:
            px1 = x
            px2 = (3.0 * x * px1 - px0) / 2.0
            px3 = (5.0 * x * px2 - 2.0 * px1) / 3.0
            px4 = (7.0 * x * px3 - 3.0 * px2) / 4.0
            px5 = (9.0 * x * px4 - 4.0 * px3) / 5.0
            b[6] = px1 * scale[1]
            b[7] = px2 * scale[2]
            b[8] = px3 * scale[3]
            b[9] = px4 * scale[4]
            b[10] = px5 * scale[5]


@njit(cache=True)
def bulk_vel_design(lat_deg, lon_deg, rho, cos_B0, sin_B0, n_poly, scale):
    """Build the design matrix (n_poly x N) for the bulk-velocity fit.

    Drop-in fast replacement for the gen_leg_vec / gen_leg_x_vec basis generation
    plus the im_arr assembly in calc_bulk_vel: same columns, generated by an
    in-kernel Legendre recurrence instead of scipy.special.eval_legendre, and
    without the six (6 x N) Legendre intermediates. The caller then runs the
    identical normal-equations solve (im_arr @ im_arr.T), which keeps results
    faithful even for the ill-conditioned fit_cbs=True system.

    Returns im_arr with shape (n_poly, N), matching the original layout. Each
    pixel writes a disjoint column, so the prange loop is thread-count invariant
    (identical results on 1 vs many threads); thread count is set at runtime via
    parallel.set_compute_threads (default 1). The per-pixel basis buffer is
    allocated inside the loop so it is private to each thread.
    """
    n = lat_deg.shape[0]
    im_arr = np.empty((n_poly, n))
    for i in prange(n):
        b = np.empty(n_poly)
        _bulk_vel_basis(lat_deg[i], lon_deg[i], rho[i], cos_B0, sin_B0, n_poly, scale, b)
        for j in range(n_poly):
            im_arr[j, i] = b[j]
    return im_arr
