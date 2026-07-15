"""Notebook-facing helpers to sweep moat parameters and visualize the results.

Load a cached epoch (see sdo_clv_pipeline.moat_cache), run the pure kernel
sdo_clv_pipeline.moat.detect_moats with candidate knobs, and inspect the moat
masks (tune by eye) and the per-spot radial profiles (tune against the flow
profile). See docs/superpowers/specs/2026-07-01-moat-tuning-design.md.
"""

import numpy as np
import matplotlib.pyplot as plt

from sdo_clv_pipeline.moat import detect_moats
from sdo_clv_pipeline.moat_cache import load_cached_epoch, kernel_keys, default_cache_dir

# HMI continuum plate scale near disk center: ~0.504 arcsec/pix, ~725 km/arcsec
default_pixel_km = 0.504 * 725.0


def run_on_cached(iso_or_path, cache_dir=default_cache_dir, **knobs):
    """Load a cached epoch and run detect_moats with the given knobs.

    Returns (epoch_dict, MoatResult). knobs are forwarded to detect_moats
    (area_thresh_pix, radius_factor, shrink, overlap_shrink).
    """
    epoch = load_cached_epoch(iso_or_path, cache_dir=cache_dir)
    result = detect_moats(*[epoch[k] for k in kernel_keys], **knobs)
    return epoch, result


def overlay_moats(epoch, result, background="continuum", ax=None):
    """Overlay left/right moat masks on a background image.

    background: 'continuum' (grayscale) or 'v_corr' (diverging velocity).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    if background == "v_corr":
        bg = np.array(epoch["v_corr"], dtype=float)
        vlim = np.nanpercentile(np.abs(bg), 99)
        ax.imshow(bg, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    else:
        bg = np.array(epoch["con_image"], dtype=float)
        ax.imshow(bg, origin="lower", cmap="Greys_r")

    # translucent color per hemisphere
    left = np.ma.masked_where(~result.left_moat, result.left_moat)
    right = np.ma.masked_where(~result.right_moat, result.right_moat)
    ax.imshow(left, origin="lower", cmap="autumn", alpha=0.5, vmin=0, vmax=1)
    ax.imshow(right, origin="lower", cmap="winter", alpha=0.5, vmin=0, vmax=1)

    n = len(result.profiles)
    ax.set_title("%s  |  %d spot(s)  |  %d moat px"
                 % (epoch.get("iso", ""), n, int(result.moat_mask.sum())))
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_profiles(result, quantity="v_corr", distance="rings",
                  pixel_km=default_pixel_km, ax=None):
    """Plot per-spot cumulative-average radial profiles.

    quantity: 'v_corr' | 'abs_mag' | 'intensity'
    distance: 'rings' (pixels) or 'Mm' (physical, via pixel_km)
    A vertical line marks each spot's current cutoff radius.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    for i, p in enumerate(result.profiles):
        rings = np.asarray(p["rings"], dtype=float)
        cutoff = p["cutoff_radius"]
        if distance == "Mm":
            x = rings * pixel_km / 1000.0
            cutoff = cutoff * pixel_km / 1000.0
            xlabel = "distance from spot [Mm]"
        else:
            x = rings
            xlabel = "distance from spot [pix]"
        line, = ax.plot(x, p[quantity],
                        label="spot %d (%s, mu=%.2f)" % (i, p["hemisphere"], p["avg_mu"]))
        ax.axvline(cutoff, color=line.get_color(), ls="--", alpha=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(quantity)
    ax.legend(fontsize=7)
    return ax


def compare_params(iso_or_path, param_list, cache_dir=default_cache_dir,
                   background="continuum"):
    """Side-by-side moat overlays for a list of knob dicts on one cached epoch.

    param_list: e.g. [{"radius_factor": 1.0}, {"radius_factor": 1.5}, ...]
    Returns the list of (params, MoatResult).
    """
    epoch = load_cached_epoch(iso_or_path, cache_dir=cache_dir)
    n = len(param_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    out = []
    for ax, knobs in zip(axes[0], param_list):
        result = detect_moats(*[epoch[k] for k in kernel_keys], **knobs)
        overlay_moats(epoch, result, background=background, ax=ax)
        ax.set_title(", ".join("%s=%s" % (k, v) for k, v in knobs.items()))
        out.append((knobs, result))
    fig.tight_layout()
    return out
