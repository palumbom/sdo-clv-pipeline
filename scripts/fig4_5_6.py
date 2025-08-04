import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import os, sys, pdb, csv, glob
import pandas as pd
from sdo_clv_pipeline.paths import root

# get paths
plotdir = os.path.join(root, "figures")
os.makedirs(plotdir, exist_ok=True)

datadir = os.path.join(root, "data")
os.makedirs(datadir, exist_ok=True)

# datafile = 'C:\\Users\\srugi\\Documents\\sdo-clv-pipeline\\data\\region_output.csv'
datafile = os.path.join(datadir, "region_output.csv")

# staticdir = os.path.join(paths, "static")
# os.makedirs(staticdir, exist_ok=True)

scripts_dir = os.path.join(root, "scripts")

# with open(datafile) as f:
#     exec(f.read())

plt.style.use(os.path.join(root, "my.mplstyle"))
# plt.ioff()

# # consolidate output
# exec(open(str(paths.scripts) + "/" + "preprocess_output.py").read())

# get color palette (seaborn colorblind hex)
colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc',
          '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']

pl_color = "tab:purple" # colors[4]
nw_color = "tab:pink" # colors[6]
qs_color = "tab:orange" # colors[1]
rp_color = "tab:red" # colors[3]
bp_color = "tab:blue" # colors[9]
pu_color = "sienna" # colors[5]
um_color = "tab:gray" # colors[7]
mt_color = "tab:green" 
lm_color = "tab:olive"
rm_color = "tab:cyan"

pl_marker = "s"
nw_marker = "p"
qs_marker = "o"
rp_marker = "v"
bp_marker = "^"
pu_marker = "X"
um_marker = "D"
mt_marker = "*"
lm_marker = "<"
rm_marker = ">"

mu_bins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def calc_region_stats(region_df, colname="v_hat"):
    # get number elements
    lo_mus = mu_bins
    nn = len(lo_mus)

    # allocate memory
    reg_avg = np.zeros(nn)
    reg_std = np.zeros(nn)
    reg_err = np.zeros(nn)

    if len(region_df[colname]) == 0:
        reg_avg[:] = np.nan
        reg_std[:] = np.nan
        reg_err[:] = np.nan
        return reg_avg, reg_std, reg_err

    # loop over mu rings
    for i in range(nn):
        # get idx
        # idx = region_df.lo_mu.values == lo_mus[i]
        idx = np.isclose(lo_mus[i], region_df.lo_mu.values, atol=1e-2)

        # calculate the stats
        reg_avg[i] = np.mean(region_df[colname][idx])
        reg_std[i] = np.std(region_df[colname][idx])
        reg_err[i] = reg_avg[i]/np.sqrt(len(region_df[colname][idx]))
    return reg_avg, reg_std, np.abs(reg_err)

# read in by region
df_vels_full = pd.read_csv(os.path.join(datadir, "processed", "full_disk.csv"))
plage = pd.read_csv(os.path.join(datadir, "processed", "plage.csv"))
network = pd.read_csv(os.path.join(datadir, "processed", "network.csv"))
quiet_sun = pd.read_csv(os.path.join(datadir, "processed", "quiet_sun.csv"))
penumbrae = pd.read_csv(os.path.join(datadir, "processed", "penumbrae.csv"))
red_penumbrae = pd.read_csv(os.path.join(datadir, "processed", "red_penumbrae.csv"))
blu_penumbrae = pd.read_csv(os.path.join(datadir, "processed", "blu_penumbrae.csv"))
umbrae = pd.read_csv(os.path.join(datadir, "processed", "umbrae.csv"))
# moat = pd.read_csv(os.path.join(datadir, "processed", "moat.csv"))
left_moat = pd.read_csv(os.path.join(datadir, "processed", "left_moat.csv"))
right_moat = pd.read_csv(os.path.join(datadir, "processed", "right_moat.csv"))

# get centers of mu bins
lo_mus = np.unique(plage.lo_mu)
hi_mus = np.unique(plage.hi_mu)
mu_bin = (lo_mus + hi_mus) / 2.0

# print(np.unique(left_moat.lo_mu))
# print(np.unique(right_moat.lo_mu))

# plot red/blue
plot_rb = False

# get stats
def clv_plot(fname=None):
    # set error bar props
    capsize = 0.0
    capthick = 0.0
    elinewidth = 0.0

    # make figure objects
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(12.8, 7.2))
    fig.subplots_adjust(hspace=0.05, wspace=0.1)

    # get stats in v_hat
    umbrae_avg, umbrae_std, umbrae_err = calc_region_stats(umbrae, colname="v_hat")
    blue_penumbrae_avg, blue_penumbrae_std, blue_penumbrae_err = calc_region_stats(blu_penumbrae, colname="v_hat")
    red_penumbrae_avg, red_penumbrae_std, red_penumbrae_err = calc_region_stats(red_penumbrae, colname="v_hat")
    penumbrae_avg, penumbrae_std, penumbrae_err = calc_region_stats(penumbrae, colname="v_hat")
    quiet_sun_avg, quiet_sun_std, quiet_sun_err = calc_region_stats(quiet_sun, colname="v_hat")
    network_avg, network_std, network_err = calc_region_stats(network, colname="v_hat")
    plage_avg, plage_std, plage_err = calc_region_stats(plage, colname="v_hat")
    # moat_avg, moat_std, moat_err = calc_region_stats(moat, colname="v_hat")
    left_moat_avg, left_moat_std, left_moat_err = calc_region_stats(left_moat, colname="v_hat")
    right_moat_avg, right_moat_std, right_moat_err = calc_region_stats(right_moat, colname="v_hat")


    # write out table
    tabfile1 = os.path.join(datadir, "processed", "v_hat_table.tex")
    df_v_hat = pd.DataFrame()
    df_v_hat["mu"] = np.round(mu_bin, decimals=2)
    df_v_hat["v_avg_qs"] = np.round(quiet_sun_avg, decimals=2)
    df_v_hat["v_std_qs"] = np.round(quiet_sun_std, decimals=2)
    df_v_hat["v_avg_nw"] = np.round(network_avg, decimals=2)
    df_v_hat["v_std_nw"] = np.round(network_std, decimals=2)
    df_v_hat["v_avg_pl"] = np.round(plage_avg, decimals=2)
    df_v_hat["v_std_pl"] = np.round(plage_std, decimals=2)
    df_v_hat["v_avg_um"] = np.round(umbrae_avg, decimals=2)
    df_v_hat["v_std_um"] = np.round(umbrae_std, decimals=2)
    df_v_hat["v_avg_pu"] = np.round(penumbrae_avg, decimals=2)
    df_v_hat["v_std_pu"] = np.round(penumbrae_std, decimals=2)
    df_v_hat["v_avg_lm"] = np.round(left_moat_avg, decimals=2)
    df_v_hat["v_std_rm"] = np.round(left_moat_std, decimals=2)
    df_v_hat["v_avg_lm"] = np.round(right_moat_avg, decimals=2)
    df_v_hat["v_std_rm"] = np.round(right_moat_std, decimals=2)
    
    # df_v_hat["v_avg_mt"] = np.round(moat_avg, decimals=2)
    # df_v_hat["v_std_mt"] = np.round(moat_std, decimals=2)
    print("df_v_hat[v_avg_lm]: ", df_v_hat["v_std_pu"].values)
    # print(len(left_moat_avg)) # 3
    # print(len(df_v_hat)) # 9
    
    df_v_hat["v_avg_lm"] = np.round(left_moat_avg, decimals=2)
    df_v_hat["v_std_lm"] = np.round(left_moat_std, decimals=2)
    df_v_hat["v_avg_rm"] = np.round(right_moat_avg, decimals=2)
    df_v_hat["v_std_rm"] = np.round(right_moat_std, decimals=2)
    #df_v_hat.to_latex(buf=tabfile1, na_rep="-", index=False, float_format="%.2f")

    print(df_v_hat["v_avg_lm"].to_numpy())

    # polyfit each of them
    # mu_fit = np.linspace(0.15, 0.95, num=100)

    # qs_wts = 1.0 / (quiet_sun_std)
    # qs_fit = np.polyfit(mu_bin, quiet_sun_avg, w=qs_wts, deg=5)

    # umbrae_wts = 1.0 / (umbrae_std)
    # umbrae_fit = np.polyfit(mu_bin, umbrae_avg, w=umbrae_wts, deg=5)

    # penumbrae_wts = 1.0 / (penumbrae_std)
    # penumbrae_fit = np.polyfit(mu_bin, penumbrae_avg, w=penumbrae_wts, deg=5)

    # network_wts = 1.0 / (network_std)
    # network_fit = np.polyfit(mu_bin, network_avg, w=network_wts, deg=5)

    # plage_wts = 1.0 / (plage_std)
    # plage_fit = np.polyfit(mu_bin, plage_avg, w=plage_wts, deg=5)

    # moat_wts = 1.0 / (moat_std)
    # moat_fit = np.polyfit(mu_bin, moat_avg, w=moat_wts, deg=5)

    # left_moat_wts = 1.0 / (left_moat_std)
    # left_moat_fit = np.polyfit(mu_bin, left_moat_avg, w=left_moat_wts, deg=5)

    # right_moat_wts = 1.0 / (right_moat_std)
    # right_moat_fit = np.polyfit(mu_bin, right_moat_avg, w=right_moat_wts, deg=5)

    # plot v_hat
    axs[0,0].errorbar(mu_bin, quiet_sun_avg, yerr=np.abs(quiet_sun_err), fmt=qs_marker, capsize=capsize,
                      capthick=capthick, elinewidth=elinewidth, color=qs_color, label=r"${\rm Quiet\ Sun}$")
    axs[0,0].fill_between(mu_bin, quiet_sun_avg - quiet_sun_std, quiet_sun_avg + quiet_sun_std, color=qs_color, alpha=0.5)
    # axs[0,0].plot(mu_fit, np.polyval(qs_fit, mu_fit), color=qs_color, ls="--")

    axs[0,0].errorbar(mu_bin, plage_avg, yerr=plage_err, fmt=pl_marker, capsize=capsize,
                      capthick=capthick, elinewidth=elinewidth, color=pl_color, label=r"${\rm Plage}$")
    axs[0,0].fill_between(mu_bin, plage_avg - plage_std, plage_avg + plage_std, color=pl_color, alpha=0.4)
    # axs[0,0].plot(mu_fit, np.polyval(plage_fit, mu_fit), color=pl_color, ls="--")

    axs[0,0].errorbar(mu_bin, network_avg, yerr=network_err, fmt=nw_marker, capsize=capsize,
                      capthick=capthick, elinewidth=elinewidth, color=nw_color, label=r"${\rm Network}$")
    axs[0,0].fill_between(mu_bin, network_avg - network_std, network_avg + network_std, color=nw_color, alpha=0.4)
    # axs[0,0].plot(mu_fit, np.polyval(network_fit, mu_fit), color=nw_color, ls="--")

    # axs[0,0].errorbar(mu_bin, moat_avg, yerr=moat_err, fmt=mt_marker, capsize=capsize,
    #                   capthick=capthick, elinewidth=elinewidth, color=mt_color, label=r"${\rm Moat}$")
    # axs[0,0].fill_between(mu_bin, moat_avg - moat_std, moat_avg + moat_std, color=mt_color, alpha=0.4)
    # # axs[0,0].plot(mu_fit, np.polyval(moat_fit, mu_fit), color=um_color, ls="--")

    axs[0,0].errorbar(mu_bin, left_moat_avg, yerr=left_moat_err, fmt=lm_marker, capsize=capsize,
                      capthick=capthick, elinewidth=elinewidth, color=lm_color, label=r"${\rm Left\ Moat}$")
    axs[0,0].fill_between(mu_bin, left_moat_avg - left_moat_std, left_moat_avg + left_moat_std, color=lm_color, alpha=0.4)
    # axs[1,0].plot(mu_fit, np.polyval(moat_fit, mu_fit), color=um_color, ls="--")

    axs[0,0].errorbar(mu_bin, right_moat_avg, yerr=right_moat_err, fmt=rm_marker, capsize=capsize,
                      capthick=capthick, elinewidth=elinewidth, color=rm_color, label=r"${\rm Right\ Moat}$")
    axs[0,0].fill_between(mu_bin, right_moat_avg - right_moat_std, right_moat_avg + right_moat_std, color=rm_color, alpha=0.4)
    # axs[1,0].plot(mu_fit, np.polyval(moat_fit, mu_fit), color=um_color, ls="--")

    if plot_rb:
        axs[1,0].errorbar(mu_bin, red_penumbrae_avg, yerr=red_penumbrae_err, fmt=rp_marker, capsize=capsize,
                          capthick=capthick, elinewidth=elinewidth, color=rp_color, label=r"${\rm Red\ Penumbrae}$")
        axs[1,0].fill_between(mu_bin, red_penumbrae_avg - red_penumbrae_std, red_penumbrae_avg + red_penumbrae_std, color=rp_color, alpha=0.4)

        axs[1,0].errorbar(mu_bin, blue_penumbrae_avg, yerr=blue_penumbrae_err, fmt=bp_marker, capsize=capsize,
                          capthick=capthick, elinewidth=elinewidth, color=bp_color, label=r"${\rm Blue\ Penumbrae}$")
        axs[1,0].fill_between(mu_bin, blue_penumbrae_avg - blue_penumbrae_std, blue_penumbrae_avg + blue_penumbrae_std, color=bp_color, alpha=0.4)

    axs[1,0].errorbar(mu_bin, penumbrae_avg, yerr=penumbrae_err, fmt=pu_marker, capsize=capsize,
                      capthick=capthick, elinewidth=elinewidth, color=pu_color, label=r"${\rm Penumbrae}$")
    axs[1,0].fill_between(mu_bin, penumbrae_avg - penumbrae_std, penumbrae_avg + penumbrae_std, color=pu_color, alpha=0.4)
    # axs[1,0].plot(mu_fit, np.polyval(penumbrae_fit, mu_fit), color=pu_color, ls="--")

    axs[1,0].errorbar(mu_bin, umbrae_avg, yerr=umbrae_err, fmt=um_marker, capsize=capsize,
                      capthick=capthick, elinewidth=elinewidth, color=um_color, label=r"${\rm Umbrae}$")
    axs[1,0].fill_between(mu_bin, umbrae_avg - umbrae_std, umbrae_avg + umbrae_std, color=um_color, alpha=0.4)
    # axs[1,0].plot(mu_fit, np.polyval(umbrae_fit, mu_fit), color=um_color, ls="--")

    # get stats for v_conv
    umbrae_avg, umbrae_std, umbrae_err = calc_region_stats(umbrae, colname="v_conv")
    blue_penumbrae_avg, blue_penumbrae_std, blue_penumbrae_err = calc_region_stats(blu_penumbrae, colname="v_conv")
    red_penumbrae_avg, red_penumbrae_std, red_penumbrae_err = calc_region_stats(red_penumbrae, colname="v_conv")
    penumbrae_avg, penumbrae_std, penumbrae_err = calc_region_stats(penumbrae, colname="v_conv")
    quiet_sun_avg, quiet_sun_std, quiet_sun_err = calc_region_stats(quiet_sun, colname="v_conv")
    network_avg, network_std, network_err = calc_region_stats(network, colname="v_conv")
    plage_avg, plage_std, plage_err = calc_region_stats(plage, colname="v_conv")
    # moat_avg, moat_std, moat_err = calc_region_stats(moat, colname="v_conv")
    left_moat_avg, left_moat_std, left_moat_err = calc_region_stats(left_moat, colname="v_conv")
    right_moat_avg, right_moat_std, right_moat_err = calc_region_stats(right_moat, colname="v_conv")

    tabfile2 = os.path.join(datadir, "v_conv_table.tex")
    df_v_conv = pd.DataFrame()
    df_v_conv["mu"] = mu_bin
    df_v_conv["v_avg_nw"] = network_avg
    df_v_conv["v_std_nw"] = network_std
    df_v_conv["v_avg_pl"] = plage_avg
    df_v_conv["v_std_pl"] = plage_std
    df_v_conv["v_avg_um"] = umbrae_avg
    df_v_conv["v_std_um"] = umbrae_std
    df_v_conv["v_avg_pu"] = penumbrae_avg
    df_v_conv["v_std_pu"] = penumbrae_std
    # df_v_conv["v_avg_mt"] = moat_avg
    # df_v_conv["v_std_mt"] = moat_std
    df_v_conv["v_avg_lm"] = left_moat_avg
    df_v_conv["v_std_lm"] = left_moat_std
    df_v_conv["v_avg_rm"] = left_moat_avg
    df_v_conv["v_std_rm"] = left_moat_std
    # df_v_conv.to_latex(buf=tabfile2, na_rep="-", index=False, float_format="%.2f")

    # polyfit each of them
    # umbrae_wts = 1.0 / (umbrae_std)
    # umbrae_fit = np.polyfit(mu_bin, umbrae_avg, w=umbrae_wts, deg=5)

    # penumbrae_wts = 1.0 / (penumbrae_std)
    # penumbrae_fit = np.polyfit(mu_bin, penumbrae_avg, w=penumbrae_wts, deg=5)

    # network_wts = 1.0 / (network_std)
    # network_fit = np.polyfit(mu_bin, network_avg, w=network_wts, deg=5)

    # plage_wts = 1.0 / (plage_std)
    # plage_fit = np.polyfit(mu_bin, plage_avg, w=plage_wts, deg=5)

    # moat_wts = 1.0 / (moat_std)
    # moat_fit = np.polyfit(mu_bin, moat_avg, w=moat_wts, deg=5)

    # moat_wts = 1.0 / (left_moat_std)
    # moat_fit = np.polyfit(mu_bin, left_moat_avg, w=plage_wts, deg=5)

    # moat_wts = 1.0 / (right_moat_std)
    # moat_fit = np.polyfit(mu_bin, right_moat_avg, w=plage_wts, deg=5)

    # plot v_conv
    axs[0,1].errorbar(mu_bin, plage_avg, yerr=plage_err, fmt=pl_marker, capsize=capsize,
                 capthick=capthick, elinewidth=elinewidth, color=pl_color, label=r"${\rm Plage}$")
    axs[0,1].fill_between(mu_bin, plage_avg - plage_std, plage_avg + plage_std, color=pl_color, alpha=0.4)
    # axs[0,1].plot(mu_fit, np.polyval(plage_fit, mu_fit), color=pl_color, ls="--")


    axs[0,1].errorbar(mu_bin, network_avg, yerr=network_err, fmt=nw_marker, capsize=capsize,
                 capthick=capthick, elinewidth=elinewidth, color=nw_color, label=r"${\rm Network}$")
    axs[0,1].fill_between(mu_bin, network_avg - network_std, network_avg + network_std, color=nw_color, alpha=0.4)
    # axs[0,1].plot(mu_fit, np.polyval(network_fit, mu_fit), color=nw_color, ls="--")

    # axs[0,1].errorbar(mu_bin, moat_avg, yerr=moat_err, fmt=mt_marker, capsize=capsize,
    #              capthick=capthick, elinewidth=elinewidth, color=mt_color, label=r"${\rm Moat}$")
    # axs[0,1].fill_between(mu_bin, moat_avg - moat_std, moat_avg + moat_std, color=mt_color, alpha=0.4)
    # # axs[0,1].plot(mu_fit, np.polyval(network_fit, mu_fit), color=nw_color, ls="--")

    axs[0,1].errorbar(mu_bin, left_moat_avg, yerr=left_moat_err, fmt=lm_marker, capsize=capsize,
                 capthick=capthick, elinewidth=elinewidth, color=lm_color, label=r"${\rm Left\ Moat}$")
    axs[0,1].fill_between(mu_bin, left_moat_avg - left_moat_std, left_moat_avg + left_moat_std, color=lm_color, alpha=0.4)
    # axs[0,1].plot(mu_fit, np.polyval(network_fit, mu_fit), color=nw_color, ls="--")

    axs[0,1].errorbar(mu_bin, right_moat_avg, yerr=right_moat_err, fmt=rm_marker, capsize=capsize,
                 capthick=capthick, elinewidth=elinewidth, color=rm_color, label=r"${\rm Right\ Moat}$")
    axs[0,1].fill_between(mu_bin, right_moat_avg - right_moat_std, right_moat_avg + right_moat_std, color=rm_color, alpha=0.4)
    # axs[0,1].plot(mu_fit, np.polyval(network_fit, mu_fit), color=nw_color, ls="--")



    if plot_rb:
        axs[1,1].errorbar(mu_bin, red_penumbrae_avg, yerr=red_penumbrae_err, fmt=rp_marker, capsize=capsize,
                     capthick=capthick, elinewidth=elinewidth, color=rp_color, label=r"${\rm Red\ Penumbrae}$")
        axs[1,1].fill_between(mu_bin, red_penumbrae_avg - red_penumbrae_std, red_penumbrae_avg + red_penumbrae_std, color=rp_color, alpha=0.4)

        axs[1,1].errorbar(mu_bin, blue_penumbrae_avg, yerr=blue_penumbrae_err, fmt=bp_marker, capsize=capsize,
                     capthick=capthick, elinewidth=elinewidth, color=bp_color, label=r"${\rm Blue\ Penumbrae}$")
        axs[1,1].fill_between(mu_bin, blue_penumbrae_avg - blue_penumbrae_std, blue_penumbrae_avg + blue_penumbrae_std, color=bp_color, alpha=0.4)

    axs[1,1].errorbar(mu_bin, penumbrae_avg, yerr=penumbrae_err, fmt=pu_marker, capsize=capsize,
                 capthick=capthick, elinewidth=elinewidth, color=pu_color, label=r"${\rm Penumbrae}$")
    axs[1,1].fill_between(mu_bin, penumbrae_avg - penumbrae_std, penumbrae_avg + penumbrae_std, color=pu_color, alpha=0.4)
    # axs[1,1].plot(mu_fit, np.polyval(penumbrae_fit, mu_fit), color=pu_color, ls="--")


    axs[1,1].errorbar(mu_bin, umbrae_avg, yerr=umbrae_err, fmt=um_marker, capsize=capsize,
                 capthick=capthick, elinewidth=elinewidth, color=um_color, label=r"${\rm Umbrae}$")
    axs[1,1].fill_between(mu_bin, umbrae_avg - umbrae_std, umbrae_avg + umbrae_std, color=um_color, alpha=0.4)
    # axs[1,1].plot(mu_fit, np.polyval(umbrae_fit, mu_fit), color=um_color, ls="--")

    # axs[1,1].errorbar(mu_bin, moat_avg, yerr=moat_err, fmt=mt_marker, capsize=capsize,
    #              capthick=capthick, elinewidth=elinewidth, color=mt_color, label=r"${\rm Moat}$")
    # axs[1,1].fill_between(mu_bin, moat_avg - moat_std, moat_avg + moat_std, color=mt_color, alpha=0.4)
    # # axs[1,1].plot(mu_fit, np.polyval(moat_fit, mu_fit), color=um_color, ls="--")


    # set limits
    axs[0,0].set_ylim(-210, 260)
    axs[0,1].set_ylim(-210, 260)
    axs[1,0].set_ylim(-600, 600)
    axs[1,1].set_ylim(-600, 600)

    # deal with xlabels
    axs[1,0].set_xlabel(r"$\mu$", fontsize=18)
    axs[1,1].set_xlabel(r"$\mu$", fontsize=18)
    axs[0,0].set_xticks(np.arange(0.1, 1.1, 0.1))
    axs[0,0].invert_xaxis()

    # deal with ylabels
    ylabel1 = r"$\hat{v}_{k, \mu}\ {\rm(m\ s}^{-1}{\rm )}$"
    ylabel2 = r"$\Delta \hat{v}_{{\rm conv},k,\mu}\ {\rm(m\ s}^{-1}{\rm )}$"
    axs[0,0].set_ylabel(ylabel1, fontsize=18)
    axs[1,0].set_ylabel(ylabel1, fontsize=18)
    axs[0,1].set_ylabel(ylabel2, fontsize=18)
    axs[1,1].set_ylabel(ylabel2, fontsize=18)
    axs[0,1].yaxis.tick_right()
    axs[1,1].yaxis.tick_right()
    # axs[0,1].set_yticklabels([])
    # axs[1,1].set_yticklabels([])

    # make the legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    _, idx = np.unique(labels, return_index=True)
    fig.legend([lines[i] for i in np.sort(idx)],
               [labels[i] for i in np.sort(idx)],
               ncol=7, fontsize=14, loc='upper center',
               handletextpad=0.15, bbox_to_anchor=(0.51, 0.95))

    # save the figure
    plt.savefig(os.path.join(plotdir, fname), bbox_inches="tight")
    plt.clf(); plt.close()
    return None

clv_plot("fig4.pdf")

# get distributions of velocities at different mu positions
mu_samps = [0.9, 0.8, 0.4, 0.2]
n_mu_samps = len(mu_samps)

# create figure objects
colname = "v_hat"
xlabel =  r"$\hat{v}_{k, \mu}\ {\rm(m\ s}^{-1}{\rm )}$"
fig, axs = plt.subplots(figsize=(11, 8.5), nrows=4, ncols=n_mu_samps)
# fig.subplots_adjust(wspace=0.07, hspace=0.15)

# loop over valus
ylims = np.zeros(n_mu_samps)
for i in range(n_mu_samps):
    # do all regs
    idx1 = umbrae.lo_mu == mu_samps[i]
    idx2 = red_penumbrae.lo_mu == mu_samps[i]
    idx3 = blu_penumbrae.lo_mu == mu_samps[i]
    idx4 = penumbrae.lo_mu == mu_samps[i]
    # idx5 = moat.lo_mu == mu_samps[i]
    idx6 = left_moat.lo_mu == mu_samps[i]
    idx7 = right_moat.lo_mu == mu_samps[i]


    # # get the FWHM the penumbra distribution
    # if i == 1:
    #     val, bin_edges = np.histogram(penumbrae[colname][idx4], bins="auto", density=True)
    #     halfmax = np.max(val)/2.0
    #     idx = np.where(val > halfmax)[0]
    #     idxl = idx[0]
    #     idxr = idx[-1]
    #     bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.0
    #     fwhm = bin_centers[idxr] - bin_centers[idxl]
    #     pdb.set_trace()

    # plot this mu
    axs[0,i].hist(umbrae[colname][idx1], bins="auto", density=True, color=um_color, histtype="step", label=r"{\rm Umbrae}")
    if plot_rb:
        axs[0,i].hist(red_penumbrae[colname][idx2], bins="auto", density=True, color=rp_color, histtype="step", label=r"{\rm Red\ Penumbrae}")
        axs[0,i].hist(blu_penumbrae[colname][idx3], bins="auto", density=True, color=bp_color, histtype="step", label=r"{\rm Blue\ Penumbrae}")
    axs[0,i].hist(penumbrae[colname][idx4], bins="auto", density=True, color=pu_color, histtype="step", label=r"{\rm Penumbrae}")
    # axs[0,i].hist(moat[colname][idx5], bins="auto", density=True, color=mt_color, histtype="step", label=r"{\rm Moat}")

    # label stuff
    axs[0,i].set_title(r"$\mu = " + str(mu_samps[i] + 0.05)[0:4]+ r"$")
    axs[0,i].set_xlim(-1200, 1200)

    ylims[i] = axs[0,i].get_ylim()[1]

    if i > 0:
        axs[0,i].set_yticklabels([])

    if i == n_mu_samps - 1:
        axs[0,i].legend(fontsize=10)

# loop again to set yvals
for i in range(n_mu_samps):
    axs[0,i].set_ylim(0.0, np.max(ylims))

# loop over valus
ylims = np.zeros(n_mu_samps)
for i in range(n_mu_samps):
    # do all regs
    idx1 = plage.lo_mu == mu_samps[i]
    idx2 = network.lo_mu == mu_samps[i]

    # pl1,ot this mu
    axs[1,i].hist(plage[colname][idx1], bins="auto", density=True, color=pl_color, histtype="step", label=r"{\rm Plage}")
    axs[1,i].hist(network[colname][idx2], bins="auto", density=True, color=nw_color, histtype="step", label=r"{\rm Network}")
    #axs[1,i].hist(moat[colname][idx5], bins="auto", density=True, color=mt_color, histtype="step", label=r"{\rm Moat}")
    # axs[2,i].hist(left_moat[colname][idx6], bins="auto", density=True, color=lm_color, histtype="step", label=r"{\rm Left\ Moat}")
    # axs[2,i].hist(right_moat[colname][idx7], bins="auto", density=True, color=rm_color, histtype="step", label=r"{\rm Right\ Moat}")

    # label stuff
    axs[1,i].set_xlim(-250,250)
    #axs[1,i].set_xlim(-500,500)

    # axs[1,i].set_xlim(0,500)

    ylims[i] = axs[1,i].get_ylim()[1]

    if i > 0:
        axs[1,i].set_yticklabels([])

    if i == n_mu_samps - 1:
        axs[1,i].legend(fontsize=10)

# loop again to set yvals
for i in range(n_mu_samps):
    axs[1,i].set_ylim(0.0, np.max(ylims))

# loop over valus
ylims = np.zeros(n_mu_samps)
for i in range(n_mu_samps):
    # do all regs
    idx3 = quiet_sun.lo_mu == mu_samps[i]
    axs[2,i].hist(quiet_sun[colname][idx3], bins="auto", density=True, color=qs_color, histtype="step", label=r"{\rm Quiet Sun}")

    # label stuff
    # axs[2,i].set_xlabel(xlabel)
    axs[2,i].set_xlim(-165,165)
    # axs[2,i].set_xlim(0,50)

    ylims[i] = axs[2,i].get_ylim()[1]

    if i > 0:
        axs[2,i].set_yticklabels([])

    if i == n_mu_samps - 1:
        axs[2,i].legend(fontsize=10)

# loop again to set yvals
for i in range(n_mu_samps):
    axs[2,i].set_ylim(0.0, np.max(ylims))

# loop over valus
ylims = np.zeros(n_mu_samps)
for i in range(n_mu_samps):
    # do all regs
    # idx5 = moat.lo_mu == mu_samps[i]
    idx6 = left_moat.lo_mu == mu_samps[i]
    idx7 = right_moat.lo_mu == mu_samps[i]

    # axs[3,i].hist(moat[colname][idx5], bins="auto", density=True, color=mt_color, histtype="step", label=r"{\rm Moat}")
    axs[3,i].hist(left_moat[colname][idx6], bins="auto", density=True, color=lm_color, histtype="step", label=r"{\rm Left\ Moat}")
    axs[3,i].hist(right_moat[colname][idx7], bins="auto", density=True, color=rm_color, histtype="step", label=r"{\rm Right\ Moat}")

    # label stuff
    # axs[2,i].set_xlabel(xlabel)
    axs[3,i].set_xlim(-700,700)
    # axs[2,i].set_xlim(0,50)

    ylims[i] = axs[3,i].get_ylim()[1]

    if i > 0:
        axs[3,i].set_yticklabels([])

    if i == n_mu_samps - 1:
        axs[3,i].legend(fontsize=10)

# loop again to set yvals
for i in range(n_mu_samps):
    axs[3,i].set_ylim(0.0, np.max(ylims))


# save the figure
fig.supxlabel(xlabel, fontsize=16, y=0.04)
fig.supylabel(r"${\rm Probability\ Density}$", fontsize=16, x=0.05)
fig.savefig(os.path.join(plotdir, "fig5.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# create figure objects
colname = "v_conv"
xlabel = r"$\Delta \hat{v}_{{\rm conv},k,\mu}\ {\rm(m\ s}^{-1}{\rm )}$"
fig, axs = plt.subplots(figsize=(11, 5.8), nrows=3, ncols=n_mu_samps)
# fig.subplots_adjust(wspace=0.07, hspace=0.15)

# loop over values
ylims = np.zeros(n_mu_samps)
for i in range(n_mu_samps):
    # do all regs
    idx1 = umbrae.lo_mu == mu_samps[i]
    idx2 = red_penumbrae.lo_mu == mu_samps[i]
    idx3 = blu_penumbrae.lo_mu == mu_samps[i]
    idx4 = penumbrae.lo_mu == mu_samps[i]
    # idx5 = moat.lo_mu == mu_samps[i]
    # idx6 = left_moat.lo_mu == mu_samps[i]
    # idx7 = right_moat.lo_mu == mu_samps[i]

    # plot this mu
    axs[0,i].hist(umbrae[colname][idx1], bins="auto", density=True, color=um_color, histtype="step", label=r"{\rm Umbrae}")
    if plot_rb:
        axs[0,i].hist(red_penumbrae[colname][idx2], bins="auto", density=True, color=rp_color, histtype="step", label=r"{\rm Red\ Penumbrae}")
        axs[0,i].hist(blu_penumbrae[colname][idx3], bins="auto", density=True, color=bp_color, histtype="step", label=r"{\rm Blue\ Penumbrae}")
    axs[0,i].hist(penumbrae[colname][idx4], bins="auto", density=True, color=pu_color, histtype="step", label=r"{\rm Penumbrae}")
    # axs[0,i].hist(moat[colname][idx5], bins="auto", density=True, color=mt_color, histtype="step", label=r"{\rm Moat}")

    # label stuff
    axs[0,i].set_title(r"$\mu = " + str(mu_samps[i] + 0.05)[0:4]+ r"$")
    axs[0,i].set_xlim(-1500, 1500)

    ylims[i] = axs[0,i].get_ylim()[1]

    if i == n_mu_samps - 1:
        axs[0,i].legend(fontsize=10)

    if i > 0:
        axs[0,i].set_yticklabels([])

# loop again to set yvals
for i in range(n_mu_samps):
    axs[0,i].set_ylim(0.0, np.max(ylims))

# loop over values
ylims = np.zeros(n_mu_samps)
for i in range(n_mu_samps):
    # do all regs
    idx1 = plage.lo_mu == mu_samps[i]
    idx2 = network.lo_mu == mu_samps[i]

    # plot this mu
    axs[1,i].hist(plage[colname][idx1], bins="auto", density=True, color=pl_color, histtype="step", label=r"{\rm Plage}")
    axs[1,i].hist(network[colname][idx2], bins="auto", density=True, color=nw_color, histtype="step", label=r"{\rm Network}")
    # axs[1,i].hist(moat[colname][idx5], bins="auto", density=True, color=mt_color, histtype="step", label=r"{\rm Moat}")
    # axs[1,i].hist(left_moat[colname][idx6], bins="auto", density=True, color=lm_color, histtype="step", label=r"{\rm Left\ Moat}")
    # axs[1,i].hist(right_moat[colname][idx7], bins="auto", density=True, color=rm_color, histtype="step", label=r"{\rm Right\ Moat}")

    # label stuff
    # axs[1,i].set_xlabel(xlabel)
    axs[1,i].set_xlim(-250,250)
    #axs[1,i].set_xlim(-500,500)

    if i == n_mu_samps - 1:
        axs[1,i].legend(fontsize=10)

    ylims[i] = axs[1,i].get_ylim()[1]

    if i > 0:
        axs[1,i].set_yticklabels([])

# loop again to set yvals
for i in range(n_mu_samps):
    axs[1,i].set_ylim(0.0, np.max(ylims))

# loop over values
ylims = np.zeros(n_mu_samps)
for i in range(n_mu_samps):
    # do all regs
    idx6 = left_moat.lo_mu == mu_samps[i]
    idx7 = right_moat.lo_mu == mu_samps[i]
    

    # plot this mu
    axs[2,i].hist(left_moat[colname][idx6], bins="auto", density=True, color=lm_color, histtype="step", label=r"{\rm Left\ Moat}")
    axs[2,i].hist(right_moat[colname][idx7], bins="auto", density=True, color=rm_color, histtype="step", label=r"{\rm Right\ Moat}")
    # axs[2,i].hist(moat[colname][idx5], bins="auto", density=True, color=mt_color, histtype="step", label=r"{\rm Moat}")

    # label stuff
    # axs[1,i].set_xlabel(xlabel)
    #axs[2,i].set_xlim(-250,250)
    axs[2,i].set_xlim(-700,700)

    if i == n_mu_samps - 1:
        axs[2,i].legend(fontsize=10)

    ylims[i] = axs[2,i].get_ylim()[1]

    if i > 0:
        axs[2,i].set_yticklabels([])

# loop again to set yvals
for i in range(n_mu_samps):
    axs[2,i].set_ylim(0.0, np.max(ylims))


# set axes labels
fig.supxlabel(xlabel, fontsize=16)
fig.supylabel(r"${\rm Probability\ Density}$", fontsize=16, x=0.05)
fig.savefig(os.path.join(plotdir, "fig6.pdf"), bbox_inches="tight")
plt.clf(); plt.close()
