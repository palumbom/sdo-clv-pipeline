"""Reproduce the fig4 CLV curves and plot umbra/penumbra CLV split by |B|.

This is a self-contained analysis script that reads a single pipeline run
directly (``region_output.csv`` + ``feature_output.csv`` under ``--datadir``)
and produces three figures in ``figures/``:

  fig4_repro.pdf        reproduction of the published fig4 (region CLV curves,
                        v_hat and v_conv vs mu) as a sanity check on the run
  umbra_clv_bfield.pdf  umbra CLV (v_hat, v_conv vs mu) split into |B| bins
  penumbra_clv_bfield.pdf  same, for penumbrae

v_conv is not stored per feature, so it is derived here.  v_conv is defined in
sdo_vels.py as ``v_hat - v_quiet_ref(epoch, mu-ring)``, where v_quiet_ref is the
quiet-Sun row's v_quiet column in region_output.csv.  We assign each feature the
quiet reference of the single mu-ring containing its intensity-weighted mean mu
(mean_mu_iw) and subtract per-epoch, before any binning.

Minor approximations (removed by the per-pixel pipeline version):
  - one ring per feature via mean_mu_iw, vs per-pixel ring assignment.  Negligible
    for umbrae (min_mu ~= max_mu); slightly larger for large penumbrae.
  - simple mean over features per mu-bin (matches the reference estimator); an
    intensity-weighted variant is not implemented here.
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sdo_clv_pipeline.paths import root, data, figures

plt.style.use(os.path.join(root, "my.mplstyle"))
plt.ioff()

# region codes (mirrors sdo_image.py module-level globals)
umbrae_code = 1
penumbrae_code = 2
quiet_sun_code = 3
network_code = 4
plage_code = 5

# reference palette / markers (from sdo-clv/src/scripts/fig4_5_6.py)
pl_color = "tab:purple"
nw_color = "tab:pink"
qs_color = "tab:orange"
pu_color = "sienna"
um_color = "tab:gray"

pl_marker = "s"
nw_marker = "p"
qs_marker = "o"
pu_marker = "X"
um_marker = "D"

bfield_col = "mean_abs_b_iw_rad" # radial, intensity-weighted |B|, the binning axis


def load_region_output(datadir, keep_soft=False):
    """Load region_output.csv, apply the quality policy, drop full-disk rows."""
    df = pd.read_csv(os.path.join(datadir, "region_output.csv"))
    df = apply_quality(df, keep_soft)
    # keep only the mu-binned rows (full-disk rows have NaN lo_mu)
    df = df[~df.lo_mu.isna()].copy()
    return df


def load_feature_output(datadir, keep_soft=False):
    """Load feature_output.csv and apply the quality policy."""
    df = pd.read_csv(os.path.join(datadir, "feature_output.csv"))
    df = apply_quality(df, keep_soft)
    return df


def apply_quality(df, keep_soft):
    # quality_flag == 0 is clean; nonzero flags are tolerable soft bits or fatal.
    # The default keeps only clean epochs; --keep-soft keeps everything recorded.
    if keep_soft:
        return df.copy()
    return df[df.quality_flag == 0].copy()


def mask_all_zero_rows(df):
    """Drop empty mu-rings (no pixels of that region), mirroring preprocess_output.

    Empty rings are written with all velocity columns identically zero (the
    np.where(v_hat != 0, ..., 0) guard in sdo_vels.py).  Averaging over them
    would bias region means toward zero.
    """
    vel_cols = ["v_hat", "v_phot", "v_quiet", "v_conv"]
    nonzero = (df[vel_cols] != 0).any(axis=1)
    return df[nonzero].copy()


def calc_region_stats(region_df, colname="v_hat"):
    """Per mu-bin mean, std and standard error of ``colname`` over all rows.

    Port of the reference calc_region_stats (simple statistics over epochs).
    """
    lo_mus = np.unique(region_df.lo_mu[~np.isnan(region_df.lo_mu)])
    nn = len(lo_mus)
    reg_avg = np.zeros(nn)
    reg_std = np.zeros(nn)
    reg_err = np.zeros(nn)
    for i in range(nn):
        idx = region_df.lo_mu == lo_mus[i]
        vals = region_df[colname][idx]
        reg_avg[i] = np.mean(vals)
        reg_std[i] = np.std(vals)
        # standard error = std/sqrt(n); the reference script used avg/sqrt(n),
        # which is not an error and goes negative when avg < 0 (it was only ever
        # drawn with elinewidth=0, so the bug was invisible there).
        reg_err[i] = reg_std[i] / np.sqrt(len(vals))
    return reg_avg, reg_std, reg_err


def ring_edges(region_df):
    """Return the mu-ring edges actually used in this run (e.g. 0.1, 0.2, ... 1.0)."""
    lo = np.sort(region_df.lo_mu.unique())
    hi = np.sort(region_df.hi_mu.unique())
    edges = np.append(lo, hi[-1])
    return edges


def quiet_reference(region_df):
    """Quiet-Sun reference table keyed by (mjd, lo_mu) -> v_quiet.

    This is the exact scalar subtracted to form v_conv in sdo_vels.py
    (the quiet_sun region's v_quiet column = v_q[ring, quiet_idx]).
    """
    qs = region_df[region_df.region == quiet_sun_code][["mjd", "lo_mu", "v_quiet"]]
    return qs.rename(columns={"v_quiet": "v_quiet_ref"})


def assign_ring(mean_mu, edges):
    """Map each feature's mean_mu_iw to its ring's lo_mu edge."""
    # digitize returns the index of the bin [edges[i-1], edges[i]); clip the
    # endpoints so mu == edges[0] or mu == edges[-1] land in the first/last ring.
    idx = np.clip(np.digitize(mean_mu, edges) - 1, 0, len(edges) - 2)
    return edges[idx]


def derive_feature_vconv(feature_df, region_df):
    """Add ring_lo_mu and v_conv to ``feature_df`` via the quiet-Sun join.

    v_conv = v_hat - v_quiet_ref(epoch, ring).  Subtraction is per-epoch and per
    feature, BEFORE any binning -- a global epoch-averaged quiet curve would be
    wrong because spots populate only a subset of epochs.
    """
    edges = ring_edges(region_df)
    out = feature_df.copy()
    out["ring_lo_mu"] = assign_ring(out.mean_mu_iw.values, edges)
    ref = quiet_reference(region_df)
    out = out.merge(ref, left_on=["mjd", "ring_lo_mu"], right_on=["mjd", "lo_mu"],
                    how="left").drop(columns=["lo_mu"])
    out["v_conv"] = out.v_hat - out.v_quiet_ref
    n_missing = out.v_quiet_ref.isna().sum()
    if n_missing:
        print(f"  dropping {n_missing} features with no quiet reference", flush=True)
    return out.dropna(subset=["v_conv"])


def bin_by_bfield(feature_df, nbins):
    """Add an integer b_bin column via equal-count quantile bins on |B|.

    Returns (df, edges) where edges has length nbins+1 (Gauss).
    """
    out = feature_df.copy()
    bins, edges = pd.qcut(out[bfield_col], nbins, labels=False, retbins=True)
    out["b_bin"] = bins.astype(int)
    return out, edges


def feature_clv_stats(feature_df, edges, colname):
    """Per mu-ring mean/std/err of ``colname``, binning features by mean_mu_iw.

    Returns (mu_centers, avg, std, err) over the rings that contain features.
    """
    lo = edges[:-1]
    hi = edges[1:]
    centers, avg, std, err = [], [], [], []
    ring = assign_ring(feature_df.mean_mu_iw.values, edges)
    for l, h in zip(lo, hi):
        vals = feature_df[colname].values[ring == l]
        if len(vals) == 0:
            continue
        centers.append((l + h) / 2.0)
        avg.append(np.mean(vals))
        std.append(np.std(vals))
        err.append(np.std(vals) / np.sqrt(len(vals)))
    return (np.array(centers), np.array(avg), np.array(std), np.array(err))


def plot_fig4(region_df, fname):
    """Reproduce the published fig4: 2x2 region CLV curves (v_hat | v_conv)."""
    capsize = capthick = elinewidth = 0.0
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(12.8, 7.2))
    fig.subplots_adjust(hspace=0.05, wspace=0.1)

    # split by region and drop empty rings before averaging
    regions = {
        "qs": mask_all_zero_rows(region_df[region_df.region == quiet_sun_code]),
        "pl": mask_all_zero_rows(region_df[region_df.region == plage_code]),
        "nw": mask_all_zero_rows(region_df[region_df.region == network_code]),
        "pu": mask_all_zero_rows(region_df[region_df.region == penumbrae_code]),
        "um": mask_all_zero_rows(region_df[region_df.region == umbrae_code]),
    }
    style = {
        "qs": (qs_color, qs_marker, r"${\rm Quiet\ Sun}$"),
        "pl": (pl_color, pl_marker, r"${\rm Plage}$"),
        "nw": (nw_color, nw_marker, r"${\rm Network}$"),
        "pu": (pu_color, pu_marker, r"${\rm Penumbrae}$"),
        "um": (um_color, um_marker, r"${\rm Umbrae}$"),
    }

    def draw(ax, key, colname):
        mu = (np.unique(regions[key].lo_mu) + np.unique(regions[key].hi_mu)) / 2.0
        avg, sd, er = calc_region_stats(regions[key], colname=colname)
        c, m, lab = style[key]
        ax.errorbar(mu, avg, yerr=er, fmt=m, capsize=capsize, capthick=capthick,
                    elinewidth=elinewidth, color=c, label=lab)
        ax.fill_between(mu, avg - sd, avg + sd, color=c, alpha=0.4)

    # v_hat (left column)
    for key in ("qs", "pl", "nw"):
        draw(axs[0, 0], key, "v_hat")
    for key in ("pu", "um"):
        draw(axs[1, 0], key, "v_hat")
    # v_conv (right column); quiet sun is the reference (~0), not plotted
    for key in ("pl", "nw"):
        draw(axs[0, 1], key, "v_conv")
    for key in ("pu", "um"):
        draw(axs[1, 1], key, "v_conv")

    axs[0, 0].set_ylim(-210, 260)
    axs[0, 1].set_ylim(-210, 260)
    axs[1, 0].set_ylim(-600, 600)
    axs[1, 1].set_ylim(-600, 600)

    axs[1, 0].set_xlabel(r"$\mu$", fontsize=18)
    axs[1, 1].set_xlabel(r"$\mu$", fontsize=18)
    axs[0, 0].set_xticks(np.arange(0.1, 1.1, 0.1))
    axs[0, 0].invert_xaxis()

    ylabel1 = r"$\hat{v}_{k, \mu}\ {\rm(m\ s}^{-1}{\rm )}$"
    ylabel2 = r"$\Delta \hat{v}_{{\rm conv},k,\mu}\ {\rm(m\ s}^{-1}{\rm )}$"
    axs[0, 0].set_ylabel(ylabel1, fontsize=18)
    axs[1, 0].set_ylabel(ylabel1, fontsize=18)
    axs[0, 1].set_ylabel(ylabel2, fontsize=18)
    axs[1, 1].set_ylabel(ylabel2, fontsize=18)
    axs[0, 1].yaxis.tick_right()
    axs[1, 1].yaxis.tick_right()
    axs[0, 1].yaxis.set_label_position("right")
    axs[1, 1].yaxis.set_label_position("right")

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    _, idx = np.unique(labels, return_index=True)
    fig.legend([lines[i] for i in np.sort(idx)], [labels[i] for i in np.sort(idx)],
               ncol=7, fontsize=14, loc="upper center", handletextpad=0.15,
               bbox_to_anchor=(0.51, 0.95))

    fig.savefig(os.path.join(figures, fname), bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_clv_by_bfield(feature_df, edges, b_edges, region_name, fname):
    """1x2 CLV (v_hat | v_conv vs mu), one curve per |B| quantile bin."""
    nbins = len(b_edges) - 1
    colors = cm.viridis(np.linspace(0.1, 0.9, nbins))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5.0))
    fig.subplots_adjust(wspace=0.28)

    for b in range(nbins):
        sub = feature_df[feature_df.b_bin == b]
        lab = (r"$" + f"{b_edges[b]:.0f}" + r"\!-\!" + f"{b_edges[b + 1]:.0f}"
               + r"\ {\rm G}$")
        for ax, col in ((axs[0], "v_hat"), (axs[1], "v_conv")):
            mu, avg, sd, er = feature_clv_stats(sub, edges, col)
            ax.errorbar(mu, avg, yerr=er, fmt="o", ms=5, color=colors[b], label=lab)
            # ax.fill_between(mu, avg - sd, avg + sd, color=colors[b], alpha=0.2)

    axs[0].set_ylabel(r"$\hat{v}_{k, \mu}\ {\rm(m\ s}^{-1}{\rm )}$", fontsize=18)
    axs[1].set_ylabel(r"$\Delta \hat{v}_{{\rm conv},k,\mu}\ {\rm(m\ s}^{-1}{\rm )}$",
                      fontsize=18)
    for ax in axs:
        ax.set_xlabel(r"$\mu$", fontsize=18)
        ax.set_xticks(np.arange(0.1, 1.1, 0.1))
        ax.invert_xaxis()
    axs[1].legend(fontsize=12, title=r"$|B|_{\rm rad}$")
    fig.suptitle(r"${\rm " + region_name + r"}$", fontsize=16)

    fig.savefig(os.path.join(figures, fname), bbox_inches="tight")
    plt.clf()
    plt.close()


def crosscheck_vconv(feature_df, region_df, code, name):
    """Sanity check vs the region_output curves (no B split).

    The feature curve is a simple mean over features; the region_output curve is
    an intensity-weighted mean over all region pixels per epoch.  These are
    different estimators, so a nonzero gap is expected.  We report it for both
    v_hat and the derived v_conv: if the two gaps are similar, the gap comes from
    the per-feature vs per-pixel weighting and the quiet-reference subtraction
    (v_conv = v_hat - v_quiet_ref) is consistent -- the derivation is sound.
    """
    edges = ring_edges(region_df)
    feat = feature_df[feature_df.region == code]
    reg = mask_all_zero_rows(region_df[region_df.region == code])
    mu_r = (np.unique(reg.lo_mu) + np.unique(reg.hi_mu)) / 2.0
    for col in ("v_hat", "v_conv"):
        mu_f, vf, _, _ = feature_clv_stats(feat, edges, col)
        vr, _, _ = calc_region_stats(reg, colname=col)
        common = np.isin(np.round(mu_r, 3), np.round(mu_f, 3))
        diff = np.abs(vf - vr[common])
        print(f"  {name} {col:6s}: feature-mean vs region_output  "
              f"median |diff| = {np.median(diff):6.2f} m/s, max = {np.max(diff):6.2f} m/s",
              flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datadir", default=os.path.join(data, "2014"),
                   help="directory holding region_output.csv and feature_output.csv")
    p.add_argument("--nbins", type=int, default=3, help="number of |B| quantile bins")
    p.add_argument("--keep-soft", action="store_true",
                   help="also include tolerable soft-quality epochs (quality_flag != 0)")
    args = p.parse_args()

    os.makedirs(figures, exist_ok=True)

    print(f"reading {args.datadir}", flush=True)
    region_df = load_region_output(args.datadir, keep_soft=args.keep_soft)
    feature_df = load_feature_output(args.datadir, keep_soft=args.keep_soft)
    edges = ring_edges(region_df)

    print("reproducing fig4 -> fig4_repro.pdf", flush=True)
    plot_fig4(region_df, "fig4_repro.pdf")

    print("deriving per-feature v_conv", flush=True)
    feature_df = derive_feature_vconv(feature_df, region_df)

    print("cross-checking derived v_conv against region_output:", flush=True)
    crosscheck_vconv(feature_df, region_df, umbrae_code, "umbrae")
    crosscheck_vconv(feature_df, region_df, penumbrae_code, "penumbrae")

    for code, name, fname in ((umbrae_code, "Umbrae", "umbra_clv_bfield.pdf"),
                              (penumbrae_code, "Penumbrae", "penumbra_clv_bfield.pdf")):
        sub = feature_df[feature_df.region == code]
        sub, b_edges = bin_by_bfield(sub, args.nbins)
        print(f"{name}: |B| bin edges (G) = "
              f"{np.round(b_edges, 0)} ; n = {len(sub)} -> {fname}", flush=True)
        plot_clv_by_bfield(sub, edges, b_edges, name, fname)

    print("done", flush=True)


if __name__ == "__main__":
    main()
