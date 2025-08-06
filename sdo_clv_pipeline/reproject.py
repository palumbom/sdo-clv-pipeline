import numpy as np
from numba import njit, prange
from astropy.wcs import WCS
import math

def compute_pixel_mapping(src_wcs, dst_wcs, shape):
    H, W = shape
    y_idx, x_idx = np.indices((H, W), dtype=np.float32)

    # map target pixels to sky coordinates
    sky = dst_wcs.pixel_to_world(x_idx, y_idx)

    # map those sky coords into source pixel coords
    src_x, src_y = src_wcs.world_to_pixel(sky)
    return src_x.astype(np.float32), src_y.astype(np.float32)

@njit(parallel=False)
def bilinear_reproject(src, src_x, src_y, dst):
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