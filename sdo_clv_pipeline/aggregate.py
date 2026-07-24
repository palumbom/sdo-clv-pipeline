"""Fused single-traversal aggregation for the per-epoch region statistics.

Two kernels replace the ~40 separate ``np.bincount`` passes the region and
feature-flag aggregations used to make over the 16.8M-pixel frame, along with the
valid mask, ring index and region index that fed them. They read the full-frame
arrays and skip sub-threshold pixels with a branch, so no compacted copies of the
per-pixel arrays are materialized.

Bit-identity: np.bincount accumulates sequentially into each bin, and
boolean-mask gathering preserves C order, so a full-frame loop that skips invalid
pixels feeds each accumulator the same addends in the same sequence. This is why
the kernels are single-threaded -- per-thread partials would merge in a different
order. It costs nothing, as the loops are bandwidth-bound.

The disk-integrated row keeps its numpy gathers: np.nansum uses pairwise
summation, which a sequential loop cannot reproduce.
"""

import numpy as np
from numba import njit

from .limbdark import bin_index

# last-axis slots of the region_ring accumulator
Q_VHAT = 0
Q_VPHOT = 1
Q_INT = 2
Q_IFLAT = 3
Q_MAG = 4
Q_PIX = 5
Q_VQUIET = 6
Q_INT_QUIET = 7

# last-axis slots of the flag_ring accumulator
F_VHAT = 0
F_VPHOT = 1
F_INT = 2
F_IFLAT = 3
F_MAG = 4
F_PIX = 5


def region_lut(region_codes):
    """Lookup table mapping a region code to its dense index, or -1."""
    lut = np.full(max(region_codes) + 1, -1, dtype=np.int64)
    for i, r in enumerate(region_codes):
        lut[r] = i
    return lut


class AggInputs(object):
    """Per-epoch valid-pixel mask, disk totals, ring edges, and region LUT.

    The heavy accumulators are built on first use and cached, so the region and
    flag aggregations each traverse the frame once however many callers ask for
    the sums.
    """

    def __init__(self, valid, n_valid, total_light, bins, lut):
        self.valid = valid
        self.n_valid = n_valid
        self.total_light = total_light
        self.bins = bins
        self.lut = lut
        self.region_ring_acc = None
        self.flag_ring_acc = None
        return None


def build_agg_inputs(flat_mu, flat_int, region_codes, mu_thresh, n_rings):
    """Compute the valid-pixel mask, disk totals, and region LUT once per epoch.

    ``n_valid`` must stay np.int64: the value it replaces came from
    ``np.nansum(bool_array)``, and ``float32_sum / np.int64`` promotes to float64
    while ``float32_sum / python_int`` stays float32 under NEP 50, which silently
    drops avg_int to float32 precision.

    NaN mu compares False against ``mu_thresh`` and so is already excluded; an
    explicit isnan test here would change every pixel_frac denominator.
    """
    valid = flat_mu >= mu_thresh
    return AggInputs(valid=valid,
                     n_valid=np.nansum(valid),
                     total_light=np.nansum(flat_int[valid]),
                     bins=np.linspace(mu_thresh, 1.0, n_rings),
                     lut=region_lut(region_codes))


@njit(cache=True)
def region_ring_sums(flat_mu, flat_int, flat_iflat, flat_vhat, flat_vphot,
                     flat_mag, flat_w_quiet, flat_reg, mu_thresh, bins, lut,
                     n_bins, n_regions):
    """All (ring, region) sums in a single full-frame traversal.

    The accumulator is (n_bins, n_regions, 8), a few kB, so it stays in cache
    while the pixel arrays stream past once.

    The region index is looked up inline to match ``_region_index``: NaN maps to
    0, ``int()`` truncates toward zero (region codes are exact small integers in
    float32), and the code is clipped before the LUT gather. Weights are cast to
    float64 before accumulating, as np.bincount does.
    """
    n = flat_mu.shape[0]
    maxc = lut.shape[0] - 1
    acc = np.zeros((n_bins, n_regions, 8), dtype=np.float64)
    for i in range(n):
        m = flat_mu[i]
        if not (m >= mu_thresh):          # NaN fails this, as in the numpy path
            continue
        rv = flat_reg[i]
        if np.isnan(rv):
            code = 0
        else:
            code = int(rv)
        if code < 0:
            code = 0
        elif code > maxc:
            code = maxc
        r = lut[code]
        if r < 0:
            continue
        b = bin_index(np.float64(m), bins, n_bins)

        iv = np.float64(flat_int[i])
        vh = np.float64(flat_vhat[i])
        acc[b, r, Q_VHAT] += vh
        acc[b, r, Q_VPHOT] += np.float64(flat_vphot[i])
        acc[b, r, Q_INT] += iv
        acc[b, r, Q_IFLAT] += np.float64(flat_iflat[i])
        acc[b, r, Q_MAG] += np.float64(flat_mag[i])
        acc[b, r, Q_PIX] += 1.0
        if flat_w_quiet[i]:
            acc[b, r, Q_VQUIET] += vh
            acc[b, r, Q_INT_QUIET] += iv
    return acc


@njit(cache=True)
def flag_ring_sums(flat_mu, flat_int, flat_iflat, flat_vhat, flat_vphot,
                   flat_mag, sel_bits, n_sel, mu_thresh, bins, n_bins):
    """Per-(selection, ring) sums for the non-exclusive feature flags, one pass.

    ``sel_bits`` packs the k-th selection's membership into bit k, so overlapping
    selections (a moat pixel that is also plage) are handled in one traversal.
    Same bit-identity and single-thread reasoning as region_ring_sums.
    """
    n = flat_mu.shape[0]
    acc = np.zeros((n_sel, n_bins, 6), dtype=np.float64)
    for i in range(n):
        m = flat_mu[i]
        if not (m >= mu_thresh):
            continue
        bits = sel_bits[i]
        if bits == 0:
            continue
        b = bin_index(np.float64(m), bins, n_bins)
        iv = np.float64(flat_int[i])
        ifl = np.float64(flat_iflat[i])
        vh = np.float64(flat_vhat[i])
        vp = np.float64(flat_vphot[i])
        mg = np.float64(flat_mag[i])
        for k in range(n_sel):
            if (bits >> k) & 1:
                acc[k, b, F_VHAT] += vh
                acc[k, b, F_VPHOT] += vp
                acc[k, b, F_INT] += iv
                acc[k, b, F_IFLAT] += ifl
                acc[k, b, F_MAG] += mg
                acc[k, b, F_PIX] += 1.0
    return acc


def pack_selection_bits(selections, n_pix):
    """Pack a list of (code, bool mask) into one uint8 bit-plane.

    Keeps ``sdo_image.flag_selections`` the single source of truth for what each
    selection means while giving the kernel one array to read.
    """
    assert len(selections) <= 8, \
        "pack_selection_bits supports at most 8 selections, got %d" % len(selections)
    bits = np.zeros(n_pix, dtype=np.uint8)
    for k, (_code, sel) in enumerate(selections):
        bits |= (sel.astype(np.uint8) << k)
    return bits


def region_acc(agg, flat_mu, flat_int, flat_iflat, flat_vhat, flat_vphot,
               flat_mag, flat_w_quiet, flat_reg, mu_thresh, n_bins, n_regions):
    """Return the shared (ring, region, 8) accumulator, traversing once."""
    if agg.region_ring_acc is None:
        agg.region_ring_acc = region_ring_sums(
            flat_mu, flat_int, flat_iflat, flat_vhat, flat_vphot, flat_mag,
            flat_w_quiet, flat_reg, mu_thresh, agg.bins, agg.lut,
            n_bins, n_regions)
    return agg.region_ring_acc


def flag_acc(agg, flat_mu, flat_int, flat_iflat, flat_vhat, flat_vphot,
             flat_mag, selections, mu_thresh, n_bins):
    """Return the shared (selection, ring, 6) accumulator, traversing once."""
    if agg.flag_ring_acc is None:
        sel_bits = pack_selection_bits(selections, flat_mu.shape[0])
        agg.flag_ring_acc = flag_ring_sums(
            flat_mu, flat_int, flat_iflat, flat_vhat, flat_vphot, flat_mag,
            sel_bits, len(selections), mu_thresh, agg.bins, n_bins)
    return agg.flag_ring_acc
