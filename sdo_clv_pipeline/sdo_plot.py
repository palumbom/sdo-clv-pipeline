"""Plotting helpers for SDO images and region masks."""

import numpy as np
from .sdo_io import *
from .sdo_image import *

import pdb, warnings
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning


# shared cosmetics (identical across every image type)
_xlabel = r"${\rm Helioprojective\ Longitude}$"
_ylabel = r"${\rm Helioprojective\ Latitude}$"
_figsize = (6.4, 4.8)
_timestamp_fontsize = 8   # DATE-OBS corner annotation; tuned to _figsize


def _annotate_timestamp(ax, date_obs):
    """Stamp the observation timestamp in the upper-left corner.

    Anchored in axes-fraction coords (transAxes) so it stays in the corner
    regardless of the data-axis inversions applied later in _finalize, and
    clears the inscribed solar disk (the frame corners are empty pixels).
    """
    # DATE-OBS is ISO 8601 (e.g. "2014-01-07T00:00:00.00"); drop the 'T'
    # separator and any fractional seconds for a clean YYYY-MM-DD HH:MM:SS label
    stamp = date_obs.replace("T", " ").split(".")[0]
    ax.text(0.02, 0.98, stamp, transform=ax.transAxes,
            ha="left", va="top", fontsize=_timestamp_fontsize, color="black")
    return None


def _continuum_intensity(im):
    """Limb-flattened intensity normalized by the disk-center darkening coeff."""
    return im.iflat / im.ld_coeffs[0]


# Per-type plotting recipe. Everything that differs between the four image types
# is data here, not control flow: which array to display, the colormap, how to
# set the color scale, and the colorbar label / output filename prefix. `scale`
# returns the imshow color-scale kwargs (either a norm or vmin/vmax), resolving
# None to each type's fixed fallback limits so colorbars can be held constant
# across an animated sequence.
_PLOT_SPECS = {
    "mag": dict(
        predicate="is_magnetogram",
        get_data=lambda im: im.image,
        cmap="RdYlBu",
        scale=lambda vmin, vmax: dict(norm=colors.SymLogNorm(
            1, vmin=-4200 if vmin is None else vmin,
               vmax=4200 if vmax is None else vmax)),
        label=r"${\rm Magnetic\ Field\ Strength\ (G)}$",
    ),
    "dop": dict(
        predicate="is_dopplergram",
        get_data=lambda im: im.v_corr,
        cmap="seismic",
        scale=lambda vmin, vmax: dict(vmin=-2000 if vmin is None else vmin,
                                      vmax=2000 if vmax is None else vmax),
        label=r"${\rm LOS\ Velocity\ } {\rm(m\ s}^{-1}{\rm )}$",
    ),
    "con": dict(
        predicate="is_continuum",
        get_data=_continuum_intensity,
        cmap="afmhot",
        scale=lambda vmin, vmax: dict(vmin=vmin, vmax=vmax),
        label=r"${\rm Relative\ HMI\ Continuum\ Intensity}$",
    ),
    "aia": dict(
        predicate="is_filtergram",
        get_data=_continuum_intensity,
        cmap="Purples_r",
        scale=lambda vmin, vmax: dict(vmin=vmin, vmax=vmax),
        label=r"${\rm Relative\ 1700\ \AA \ Continuum\ Intensity}$",
    ),
}


def _draw_limb_and_grid(ax, wcs, mu):
    """Overlay the heliographic lat/lon grid and the solar limb on a WCS axis."""
    sp.visualization.wcsaxes_compat.wcsaxes_heliographic_overlay(
        ax, grid_spacing=15 * u.deg, annotate=True,
        color="k", alpha=0.5, ls="--", lw=0.5)
    # The limb is rr == 1. The disk is a filled circle of area pi*r^2, so derive
    # the radius from the finite-mu pixel count rather than marching-squares
    # contouring the full 4k frame on every figure. Center is the WCS reference
    # pixel (Sun center); CRPIX is 1-based, imshow data coords are 0-based.
    cx, cy = wcs.wcs.crpix[0] - 1.0, wcs.wcs.crpix[1] - 1.0
    r_pix = np.sqrt(np.count_nonzero(np.isfinite(mu)) / np.pi)
    ax.add_patch(Circle((cx, cy), r_pix, fill=False,
                        color="k", ls="--", lw=0.5, alpha=0.5))
    return None


def _finalize(fig, ax, outdir, fname, default_name, dpi):
    """Apply shared axis cosmetics, then save (and close) or show the figure."""
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)
    ax.grid(False)
    if outdir is not None:
        if fname is None:
            fname = default_name
        fig.savefig(os.path.join(outdir, fname), bbox_inches="tight", dpi=dpi)
        plt.close(fig)   # close THIS figure so batch loops don't leak figures
    else:
        plt.show()
    return None


def plot_image(sdo_image, outdir=None, fname=None, vmin=None, vmax=None, dpi=500):
    """Plot an SDOImage with WCS axes and type-specific styling.

    Parameters
    ----------
    sdo_image : SDOImage
        Image to plot (magnetogram, dopplergram, continuum, or filtergram).
    outdir : str, optional
        If provided, save to this directory instead of showing.
    fname : str, optional
        Filename for the saved figure (auto-generated if None).
    vmin, vmax : float, optional
        Fixed color-scale limits. When None, magnetogram/dopplergram fall back
        to their built-in fixed limits and continuum/filtergram autoscale to the
        frame. Pass explicit values to keep the colorbar identical across frames
        (e.g. when animating a sequence).
    dpi : int, optional
        Raster resolution for the saved figure. The solar image is the only
        rasterized element in the vector PDF; lower this to shrink files and
        speed up batch runs.
    """
    # pick the per-type recipe; bail on anything that isn't a known image type
    key = next((k for k, s in _PLOT_SPECS.items()
                if getattr(sdo_image, s["predicate"])()), None)
    if key is None:
        return None
    spec = _PLOT_SPECS[key]

    # initialize the figure
    fig = plt.figure(figsize=_figsize)
    ax1 = fig.add_subplot(111, projection=sdo_image.wcs)
    ax1.set_aspect("equal")

    # get cmap (bad/masked pixels show white)
    cmap = plt.get_cmap(spec["cmap"]).copy()
    cmap.set_bad(color="white")

    # plot the sun
    img = ax1.imshow(spec["get_data"](sdo_image), cmap=cmap, origin="lower",
                     interpolation=None, **spec["scale"](vmin, vmax))
    _draw_limb_and_grid(ax1, sdo_image.wcs, sdo_image.mu)
    fig.colorbar(img).set_label(spec["label"])

    _annotate_timestamp(ax1, sdo_image.date_obs)
    _finalize(fig, ax1, outdir, fname,
              key + "_" + sdo_image.date_obs + ".pdf", dpi)
    return None


def plot_mask(mask, outdir=None, fname=None, dpi=500):
    """Plot a SunMask classification map with labeled region colors."""
    # get cmap
    cmap = colors.ListedColormap(["black", "saddlebrown", "orange", "yellow", "white", "blue"])
    cmap.set_bad(color="white")
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6], ncolors=cmap.N, clip=True)

    # plot the sun (regions - 0.5 centers each integer code in its color band)
    fig = plt.figure(figsize=_figsize)
    ax1 = fig.add_subplot(111, projection=mask.wcs)
    img = ax1.imshow(mask.regions - 0.5, cmap=cmap, norm=norm,
                     origin="lower", interpolation=None)
    _draw_limb_and_grid(ax1, mask.wcs, mask.mu)
    clb = fig.colorbar(img, ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    clb.ax.set_yticklabels([r"${\rm Umbra}$", r"${\rm Penumbra}$", r"${\rm Quiet\ Sun}$",
                            r"${\rm Network}$", r"${\rm Plage}$", r"${\rm Moat}$"])

    _annotate_timestamp(ax1, mask.date_obs)
    _finalize(fig, ax1, outdir, fname, "mask_" + mask.date_obs + ".pdf", dpi)
    return None


def _plot_spot_overlay(mask_copy, spots, letters, base_cmap, overlay_cmap,
                       title, label_spots):
    """Render the merged-penumbra mask in greyscale with a per-spot color overlay."""
    h, w = mask_copy.shape
    overlay = np.full((h, w), np.nan)               # overlay array, one number per spot
    for i, spot_mask in enumerate(spots):           # boolean-mask assignment; no np.where
        overlay[spot_mask] = i

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mask_copy, cmap=base_cmap, origin="lower")
    ax.imshow(overlay, cmap=plt.get_cmap(overlay_cmap, len(spots)),
              origin="lower", alpha=0.7)

    if label_spots:                                 # letter at the centroid of each spot
        for i, spot_mask in enumerate(spots):
            y, x = np.where(spot_mask)
            ax.text(x.mean(), y.mean(), letters[i], ha="center", va="center",
                    color="black", fontsize=10, weight="bold")

    ax.set_title(title)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.show()
    return None


def label_moats_on_sun(mask, outdir=None, fname=None):
    """Overlay moat region labels using precomputed moat data."""
    # get spot masks and label letters from the separate moats data file
    moat_file = os.path.join(root, "data", "moats_data.npz")
    data = np.load(moat_file, allow_pickle=True)  # allow_pickle for arrays of arrays
    letters = data["letters"]

    # merged-penumbra copy of the classification map (shared by both panels)
    mask_copy = np.copy(mask.regions)
    mask_copy[mask_copy >= 3] -= 1

    # panel 1: detected spots, labeled with letters
    _plot_spot_overlay(mask_copy, data["area_idx_arr"], letters,
                       "gray", "tab20", "Colored Spots with Labels",
                       label_spots=True)
    # panel 2: moats dilated to the Solanki radius (labels suppressed, as before)
    _plot_spot_overlay(mask_copy, data["dilated_spots"], letters,
                       "Greys_r", "rainbow", "Moats to Solanki Radius",
                       label_spots=False)
    return None
