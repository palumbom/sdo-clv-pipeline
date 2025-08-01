# code modified from https://github.com/samarth-kashyap/hmi-clean-ls
# original author: Samarth Ganesh Kashyap (g.samarth@tifr.res.in)
# original associated publication: https://arxiv.org/abs/2105.12055

import numpy as np
from math import pi
from scipy.special import eval_legendre


def gen_leg_vec(lmax, theta):
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
    # switch to radians
    x = x.value * np.pi / 180.0

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