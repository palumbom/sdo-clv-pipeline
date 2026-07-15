"""Pure moat-flow detection kernel: grow rings outward from large spots.

Extracted from SunMask.identify_regions so it can be tuned in isolation (see
docs/superpowers/specs/2026-07-01-moat-tuning-design.md). Takes plain arrays and
returns a MoatResult; computes no plots and touches no disk. Depends only on
numpy/scipy/skimage (deliberately NOT numba or sdo_image, to stay importable in
the numba-free test job and to avoid a circular import).
"""

import numpy as np
from dataclasses import dataclass, field

from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table


@dataclass
class MoatResult:
    """Masks and per-spot radial profiles from detect_moats."""
    moat_mask: "np.ndarray"          # bool, union of all moat rings
    left_moat: "np.ndarray"          # bool, left-hemisphere (lon >= 0) moats
    right_moat: "np.ndarray"         # bool, right-hemisphere (lon < 0) moats
    profiles: list = field(default_factory=list)  # one dict per spot (see below)


def detect_moats(v_corr, mu, lon, con_image, mag_image,
                 invalid_mask, umbra_mask, penumbra_mask, *,
                 area_thresh_pix=600, radius_factor=1.2,
                 shrink=0.97, overlap_shrink=0.65):
    """Grow a moat ring outward from each large sunspot via a distance transform.

    Spots are contiguous umbra+penumbra islands larger than ``area_thresh_pix``.
    For each, the ring extends to ``shrink * radius_factor * sqrt(area / pi)``
    pixels; overlapping moats are regrown with ``overlap_shrink`` instead.

    Parameters
    ----------
    v_corr, mu, lon, con_image, mag_image : ndarray
        Corrected Doppler velocity, cos(heliocentric angle), heliographic
        longitude (degrees), continuum intensity, and line-of-sight magnetogram.
    invalid_mask, umbra_mask, penumbra_mask : ndarray of bool
        Off-disk/low-mu pixels, umbra pixels, and penumbra pixels.
    area_thresh_pix, radius_factor, shrink, overlap_shrink : float
        The four tunable knobs (defaults reproduce the original inline behavior).

    Returns
    -------
    MoatResult
    """
    assert umbra_mask.dtype == bool and penumbra_mask.dtype == bool
    assert invalid_mask.dtype == bool

    shape = con_image.shape

    # structure that connects diagonally-touching pixels (8-connectivity)
    corners = ndimage.generate_binary_structure(2, 2)

    # label each contiguous umbra+penumbra island; include umbra so rings only
    # ever expand outward from the spot
    binary_img = np.logical_or(umbra_mask, penumbra_mask)
    labels, nlabels = ndimage.label(binary_img, structure=corners)

    # per-pixel map of each label's pixel-count area (area[label][pixel])
    rprops_tab = regionprops_table(labels, properties=("label", "area", "centroid"))
    all_areas = np.asarray(rprops_tab["area"])
    area_by_label = np.r_[0, all_areas]      # index 0 == background
    areas_pix = area_by_label[labels]

    # keep only spots above the area threshold
    keep = all_areas > area_thresh_pix
    x_centroids = np.asarray(rprops_tab["centroid-1"][keep]).astype(int)
    y_centroids = np.asarray(rprops_tab["centroid-0"][keep]).astype(int)
    areas = all_areas[keep]
    n_spots = len(areas)

    # allocate per-spot scratch and outputs
    moat_idx = np.zeros(shape, dtype=bool)
    moats = np.zeros((n_spots, *shape), dtype=bool)
    left_hemisphere = np.zeros(n_spots, dtype=bool)
    avg_mu = np.zeros(n_spots)
    max_dilations = np.zeros(n_spots)
    max_rings = np.zeros(n_spots, dtype=int)
    cutoff_radii = np.zeros(n_spots)
    avg_vels, avg_mags, avg_ints = [], [], []
    valid_mask = np.zeros_like(invalid_mask)

    def dilate_moat(idx, pre_factor, compute_avgs=True):
        max_area = areas[idx]
        x_centroid = x_centroids[idx]
        y_centroid = y_centroids[idx]

        # NOTE: the spot is selected by matching its area VALUE, preserved verbatim
        # from the original. Two spots of identical pixel count would merge here --
        # a latent issue flagged for the tuning phase, not fixed in this extraction.
        max_area_idx = areas_pix == max_area
        valid_list = [~max_area_idx, ~invalid_mask, ~umbra_mask, ~penumbra_mask]
        np.logical_and.reduce(valid_list, out=valid_mask)

        # rings = integer distance (in pixels) from the spot boundary
        dist = distance_transform_edt(~max_area_idx)
        rings = dist.astype(int)
        max_ring = np.nanmax(rings[valid_mask]) + 1
        flat_rings = rings[valid_mask].ravel()

        def cum_avg_val(val):
            sums = np.bincount(flat_rings, weights=val, minlength=max_ring)
            counts = np.bincount(flat_rings, minlength=max_ring)
            cumsum_sums = np.cumsum(sums)
            cumsum_counts = np.cumsum(counts)
            return cumsum_sums[1:] / cumsum_counts[1:]

        if compute_avgs:
            flat_vel = v_corr[valid_mask].ravel()
            flat_mag = np.abs(mag_image[valid_mask].ravel())
            flat_int = con_image[valid_mask].ravel()
            avg_vels.append(cum_avg_val(flat_vel))
            avg_mags.append(cum_avg_val(flat_mag))
            avg_ints.append(cum_avg_val(flat_int))

        cutoff = pre_factor * (radius_factor * np.sqrt(max_area / np.pi))
        rings_cond = rings < cutoff
        np.logical_and.reduce([rings_cond, valid_mask], out=moat_idx)
        moats[idx, :, :] = moat_idx

        avg_mu[idx] = np.average(mu[moat_idx])
        max_dilations[idx] = np.nanmax(rings[rings_cond])
        max_rings[idx] = max_ring
        cutoff_radii[idx] = cutoff
        left_hemisphere[idx] = lon[y_centroid, x_centroid] >= 0.0
        return None

    # first pass over every spot
    for idx in range(n_spots):
        dilate_moat(idx, shrink, compute_avgs=True)

    # regrow any moats that collide with a neighbor, using the tighter factor
    collision_map = np.count_nonzero(moats, axis=0) > 1
    overlapping = np.any(np.logical_and(moats, collision_map), axis=(1, 2))
    for idx in np.nonzero(overlapping)[0]:
        dilate_moat(idx, overlap_shrink, compute_avgs=False)

    # split by hemisphere (empty-safe when n_spots == 0)
    left_moat = moats[left_hemisphere, :, :].any(axis=0)
    right_moat = moats[~left_hemisphere, :, :].any(axis=0)
    moat_mask = np.logical_or(left_moat, right_moat)

    profiles = []
    for idx in range(n_spots):
        profiles.append({
            "rings": np.arange(1, max_rings[idx]),
            "v_corr": avg_vels[idx],
            "abs_mag": avg_mags[idx],
            "intensity": avg_ints[idx],
            "avg_mu": avg_mu[idx],
            "area": areas[idx],
            "centroid": (int(y_centroids[idx]), int(x_centroids[idx])),
            "hemisphere": "left" if left_hemisphere[idx] else "right",
            "max_dilation": max_dilations[idx],
            "cutoff_radius": cutoff_radii[idx],
        })

    return MoatResult(moat_mask=moat_mask, left_moat=left_moat,
                      right_moat=right_moat, profiles=profiles)
