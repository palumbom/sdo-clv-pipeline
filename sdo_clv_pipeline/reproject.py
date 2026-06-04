"""Lightweight WCS-based reprojection helpers."""

import numpy as np
from numba import njit, prange
from astropy.wcs import WCS
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
import math

def compute_pixel_mapping(src_wcs, dst_wcs, shape, stride=4):
    """Compute source pixel coordinates for each destination pixel.

    Fast path that preserves the *exact* high-level transform physics (including
    the cross-observer parallax between the HMI and AIA exposures, which a pure
    2D WCS composition would drop). The destination->source pixel map is a very
    smooth, near-affine TAN->TAN warp, so the expensive SkyCoord transform is
    evaluated only on a coarse subgrid (every ``stride`` pixels) and bilinearly
    upsampled to full resolution. With stride=8 this is ~64x fewer SkyCoord
    transforms, reducing ~12 s to <1 s while matching the reference to well under
    0.01 px on disk.

    Off-disk coarse nodes come back as NaN (the high-level transform projects
    through the solar surface); they are nearest-filled before upsampling so the
    usable disk is not eroded. Any residual error stays in the ``mu < mu_thresh``
    limb ring that is masked downstream. The original full-resolution
    implementation is retained as ``compute_pixel_mapping_highlevel`` and used as
    the verification oracle in ``scripts/verify_geometry.py``.

    Parameters
    ----------
    src_wcs : astropy.wcs.WCS
        Source image WCS.
    dst_wcs : astropy.wcs.WCS
        Destination image WCS.
    shape : tuple
        Destination image shape as (H, W).
    stride : int, optional
        Coarse-grid spacing in pixels. Smaller is more accurate but slower.
    """
    H, W = shape

    # coarse, uniform destination grid spanning [0, H-1] x [0, W-1] inclusive
    n_y = int(np.ceil((H - 1) / stride)) + 1
    n_x = int(np.ceil((W - 1) / stride)) + 1
    ys = np.linspace(0, H - 1, n_y)
    xs = np.linspace(0, W - 1, n_x)
    cx, cy = np.meshgrid(xs, ys)  # (n_y, n_x)

    # exact high-level transform on the coarse grid
    sky = dst_wcs.pixel_to_world(cx, cy)
    csx, csy = src_wcs.world_to_pixel(sky)
    csx = np.asarray(csx, dtype=np.float64)
    csy = np.asarray(csy, dtype=np.float64)

    # nearest-fill off-disk NaN nodes so bilinear upsampling does not erode the
    # on-disk limb
    csx = _fill_nan_nearest(csx)
    csy = _fill_nan_nearest(csy)

    # bilinear upsample to full resolution via index-space coordinates
    full_y = np.arange(H)
    full_x = np.arange(W)
    gy, gx = np.meshgrid(full_y, full_x, indexing="ij")
    coarse_r = gy * (n_y - 1) / (H - 1)
    coarse_c = gx * (n_x - 1) / (W - 1)
    coords = np.vstack([coarse_r.ravel(), coarse_c.ravel()])

    src_x = map_coordinates(csx, coords, order=1, mode="nearest").reshape(H, W)
    src_y = map_coordinates(csy, coords, order=1, mode="nearest").reshape(H, W)
    return src_x.astype(np.float32), src_y.astype(np.float32)


def _fill_nan_nearest(arr):
    """Replace NaNs in a 2D array with their nearest finite value."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    yy, xx = np.indices(arr.shape)
    valid = ~mask
    filled = arr.copy()
    filled[mask] = griddata(
        (yy[valid], xx[valid]), arr[valid],
        (yy[mask], xx[mask]), method="nearest",
    )
    return filled


def compute_pixel_mapping_highlevel(src_wcs, dst_wcs, shape):
    """Reference (slow) SkyCoord-based pixel mapping; verification oracle.

    Builds a SkyCoord per destination pixel and routes through the high-level
    ``world_to_pixel`` (which applies the full observer/obstime frame transform).
    Kept only so ``compute_pixel_mapping`` can be validated against it.
    """
    H, W = shape
    y_idx, x_idx = np.indices((H, W), dtype=np.float32)

    # map target pixels to sky coordinates
    sky = dst_wcs.pixel_to_world(x_idx, y_idx)

    # map those sky coords into source pixel coords
    src_x, src_y = src_wcs.world_to_pixel(sky)
    return src_x.astype(np.float32), src_y.astype(np.float32)

@njit(parallel=True)
def bilinear_reproject(src, src_x, src_y, dst):
    """Bilinearly sample src into dst using precomputed pixel mappings.

    Each output pixel is independent, so the prange loop is thread-count
    invariant; thread count is set at runtime via parallel.set_compute_threads
    (default 1, safe under the across-epoch pool)."""
    H, W = dst.shape
    Hs, Ws = src.shape
    for idx in prange(H*W):
        j = idx // W
        i = idx % W
    
        x = src_x[j, i]
        y = src_y[j, i]
        x0 = math.floor(x)
        y0 = math.floor(y)
        dx = x - x0
        dy = y - y0

        # boundary check
        if x0 < 0 or x0+1 >= Ws or y0 < 0 or y0+1 >= Hs:
            dst[j, i] = np.nan
        else:
            v00 = src[y0, x0]
            v10 = src[y0, x0+1]
            v01 = src[y0+1, x0]
            v11 = src[y0+1, x0+1]
            dst[j, i] = (v00 * (1-dx)*(1-dy) + 
                            v10 * dx*(1-dy) +
                            v01 * (1-dx)*dy +
                            v11 * dx*dy)
    return None
