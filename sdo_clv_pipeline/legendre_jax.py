# code copied/modified from https://github.com/samarth-kashyap/hmi-clean-ls
# original author: Samarth Ganesh Kashyap (g.samarth@tifr.res.in)
# original associated publication: https://arxiv.org/abs/2105.12055

import numpy as np
import jax
import jax.numpy as jnp
from math import pi
from pyshtools import legendre as pleg

def get_pleg_index_jax(l, m):
    return int(l*(l+1)/2 + m)

jaxPlBar_d1 = jax.jit(pleg.PlBar_d1)

def gen_leg_jax(lmax, theta):
    cost = jnp.cos(theta)
    sint = jnp.sin(theta).reshape(1, theta.shape[0])

    maxIndex = int(lmax+1)
    ell = jnp.arange(maxIndex)
    norm = jnp.sqrt(ell*(ell+1)).reshape(maxIndex, 1)
    norm = norm.at[norm==0].set(1)

    # leg = jnp.zeros((maxIndex, theta.size))
    # leg_d1 = jnp.zeros((maxIndex, theta.size))

    # for i,z in enumerate(cost):
    #     leg[:, i], leg_d1[:, i] = jaxPlBar_d1(lmax, z)

    def PlBar_loop(cost):
        leg = []
        leg_d1 = []
        for i,z in enumerate(cost):
            out1, out2 = pleg.PlBar_d1(lmax, z)
            leg.append(out1)
            leg_d1.append(out2)
        return leg, leg_d1

    auto_batch_PlBar = jax.vmap(PlBar_loop)

    derp1, derp2 = auto_batch_PlBar(cost)
    return leg/jnp.sqrt(2)/norm, leg_d1 * (-sint)/jnp.sqrt(2)/norm


def gen_leg_x(lmax, x):
    maxIndex = int(lmax+1)
    ell = np.arange(maxIndex)
    norm = np.sqrt(ell*(ell+1)).reshape(maxIndex, 1)
    norm[norm == 0] = 1

    leg = np.zeros((maxIndex, x.size))
    leg_d1 = np.zeros((maxIndex, x.size))

    for i,z in enumerate(x):
        leg[:, i], leg_d1[:, i] = pleg.PlBar_d1(lmax, z)
    return leg/np.sqrt(2)/norm, leg_d1/np.sqrt(2)/norm


def inv_SVD(A, svdlim):
    u, s, v = np.linalg.svd(A, full_matrices=False)
    sinv = s**-1
    sinv[sinv/sinv[0] > svdlim] = 0.0  # svdlim
    return np.dot(v.transpose().conjugate(),
                  np.dot(np.diag(sinv), u.transpose().conjugate()))
