"""Dump the full-disk velocity component arrays for a single epoch to an npz.

The batch pipeline (run_one.py / run_pipe.py) reduces each epoch down to
per-region CSV summaries and discards the 4096x4096 arrays. Some downstream
work wants the maps themselves -- plotting a velocity field over the solar
disk, or exporting it on a lat/lon grid. This script re-runs one epoch through
the same public reduction and writes the arrays out instead of aggregating.

Array keys are the five the velocity-field plots expect, plus extras:

    raw        dop.image    dopplergram as read (mu-masked)
    satellite  dop.v_obs    projected SDO spacecraft velocity
    rotation   dop.v_rot    differential rotation, Legendre s = 1, 3, 5
    mer_flows  dop.v_mer    meridional circulation, Legendre s = 2, 4
    corrected  dop.v_corr   residual after removing every fitted component
    cbs        dop.v_cbs    fitted radial (limb/CB) profile incl. constant; removed
                            from corrected only when fit_cbs

All five velocity arrays and ``raw`` share one footprint: the bulk-velocity fit
is defined on mu >= 0.1 and mask_low_mu NaNs everything below --mu-thresh.

The scalar keys (obstime, B0, fit_params, mu_thresh, fit_cbs, and the four
source paths) make the npz self-describing. A consumer must take the epoch from
``obstime`` rather than hardcoding one: the arrays carry no time of their own,
and a mismatched epoch silently rotates any heliographic transform applied to
them.

Usage (explicit files):
  uv run scripts/dump_velocity_npz.py --con C.fits --mag M.fits --dop D.fits \
      --aia A.fits --outdir DIR
Usage (index into a matched glob, as run_one.py does):
  uv run scripts/dump_velocity_npz.py --fitsdir DIR --globexp '20231014' \
      --index 0 --outdir DIR --fit-cbs both

The Oct 2023 eclipse inputs were fetched with:
  download_data(series="720", email="mlp95@psu.edu",
                outdir="/mnt/ceph/users/mpalumbo/sdo_eclipse_data",
                start="2023-10-14T15:00:00", end="2023-10-14T19:00:00",
                sample=1, overwrite=False, progress=True)
"""

import os, argparse, logging

import numpy as np

from sdo_clv_pipeline.parallel import set_compute_threads
from sdo_clv_pipeline.sdo_io import find_data, get_date
from sdo_clv_pipeline.sdo_process import reduce_sdo_images
from sdo_clv_pipeline.logging_setup import configure_logging

logger = logging.getLogger(__name__)

# npz key -> SDOImage attribute on the reduced dopplergram
array_keys = {"raw": "image",
              "satellite": "v_obs",
              "rotation": "v_rot",
              "mer_flows": "v_mer",
              "corrected": "v_corr",
              "cbs": "v_cbs"}


def resolve_epoch(args):
    """Return the (con, mag, dop, aia) paths for this epoch."""
    if args.index is not None:
        con, mag, dop, aia = find_data(args.fitsdir, globexp=args.globexp)
        n = len(con)
        assert n > 0, f"no complete epochs matched {args.globexp!r} in {args.fitsdir}"
        assert 0 <= args.index < n, f"index {args.index} out of range [0, {n})"
        i = args.index
        return con[i], mag[i], dop[i], aia[i]
    assert args.con and args.mag and args.dop and args.aia, \
        "provide either --index (+--fitsdir/--globexp) or all of --con/--mag/--dop/--aia"
    return args.con, args.mag, args.dop, args.aia


def describe(name, arr):
    """Print shape, coverage and magnitude of one array."""
    finite = np.isfinite(arr)
    n_finite = int(finite.sum())
    if n_finite == 0:
        print(f"  {name:<10s} {str(arr.shape):>13s}  all NaN", flush=True)
        return None

    vals = arr[finite]
    rms = float(np.sqrt(np.mean(vals.astype(np.float64) ** 2)))
    print(f"  {name:<10s} {str(arr.shape):>13s}  "
          f"finite {100.0 * n_finite / arr.size:5.1f}%  "
          f"min {np.min(vals):10.2f}  max {np.max(vals):10.2f}  rms {rms:9.2f}",
          flush=True)
    return None


def dump_epoch(con, mag, dop_file, aia, outpath, mu_thresh=0.1, fit_cbs=False):
    """Reduce one epoch and write its velocity component arrays to outpath."""
    images = reduce_sdo_images(con, mag, dop_file, aia,
                               mu_thresh=mu_thresh, fit_cbs=fit_cbs)
    assert images is not None, \
        f"epoch was skipped by the pipeline (quality/limb-darkening/region failure): {con}"
    _con, _mag, dop, _aia, _mask = images

    out = {key: getattr(dop, attr) for key, attr in array_keys.items()}
    out["obstime"] = np.array(dop.date_obs)
    out["B0"] = np.array(dop.B0)
    out["fit_params"] = np.asarray(dop.fit_params)
    out["mu_thresh"] = np.array(mu_thresh)
    out["fit_cbs"] = np.array(fit_cbs)
    out["con_file"] = np.array(con)
    out["mag_file"] = np.array(mag)
    out["dop_file"] = np.array(dop_file)
    out["aia_file"] = np.array(aia)

    print(f"\nepoch {dop.date_obs}  B0 = {dop.B0:.4f} deg  fit_cbs = {fit_cbs}", flush=True)
    print(f"  fit_params = {np.array2string(np.asarray(dop.fit_params), precision=3)}", flush=True)
    for key in array_keys:
        describe(key, out[key])

    np.savez_compressed(outpath, **out)
    print(f"  wrote {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser(description="dump full-disk velocity components for one SDO epoch")
    ap.add_argument("--con"); ap.add_argument("--mag")
    ap.add_argument("--dop"); ap.add_argument("--aia")
    ap.add_argument("--fitsdir", default="/mnt/ceph/users/mpalumbo/sdo_eclipse_data/")
    ap.add_argument("--globexp", default="")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out", default=None,
                    help="output basename; default sdo_dop_<stamp>.npz, with _cbs appended "
                         "for the fit_cbs=True variant")
    ap.add_argument("--mu-thresh", type=float, default=0.1)
    ap.add_argument("--fit-cbs", choices=("false", "true", "both"), default="false")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(args.log_level)
    set_compute_threads()  # SDO_THREADS env (default 1)

    con, mag, dop_file, aia = resolve_epoch(args)
    os.makedirs(args.outdir, exist_ok=True)
    stamp = get_date(con).strftime("%Y%m%dT%H%M%S")

    variants = {"false": [False], "true": [True], "both": [False, True]}[args.fit_cbs]
    for fit_cbs in variants:
        if args.out is not None:
            stem, ext = os.path.splitext(args.out)
            ext = ext or ".npz"
            base = f"{stem}_cbs{ext}" if fit_cbs else f"{stem}{ext}"
        else:
            base = f"sdo_dop_{stamp}_cbs.npz" if fit_cbs else f"sdo_dop_{stamp}.npz"
        dump_epoch(con, mag, dop_file, aia, os.path.join(args.outdir, base),
                   mu_thresh=args.mu_thresh, fit_cbs=fit_cbs)

    return None


if __name__ == "__main__":
    main()
