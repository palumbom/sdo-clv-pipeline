import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import os, sys, pdb, csv, glob, copy
import pandas as pd
from astropy.time import Time
from sdo_clv_pipeline.paths import root
from sdo_clv_pipeline.sdo_image import *

# use style
plt.style.use(os.path.join(root, "my.mplstyle"))

# sort out paths
plotdir = os.path.join(root, "figures")
os.makedirs(plotdir, exist_ok=True)

datadir = os.path.join(root, "data", "processed")
os.makedirs(datadir, exist_ok=True)

# read in the data and sort by mjd
df_vels = pd.read_csv(os.path.join(datadir,"full_disk.csv"))

# get mjds rounded to nearrest day
mjds_round = np.round(df_vels.mjd)

vels_bin = []
for i in range(len(mjds_round)):
    idx = mjds_round == mjds_round[i]
    vels_bin.append(np.mean(df_vels.v_conv[idx]))

# get indices for full disk
no_reg = np.isnan(df_vels.region)
is_reg = np.logical_not(no_reg)
is_moat = df_vels.region == moat_code

# plot it
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(df_vels.mjd[no_reg], df_vels.v_conv[no_reg], s=5, label=r"${\rm Entire\ Sun}$")
ax1.set_xlabel(r"${\rm MJD}$")
ax1.set_ylabel(r"$ \Delta v_{\rm conv}\ {\rm [ m s}^{-1} {\rm ]}$")
ax1.legend(ncol=2)#, fontsize=9)
fig.savefig(os.path.join(plotdir, "full_disk_series.pdf"))
# plt.show()
plt.clf(); plt.close()

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(df_vels.mjd[no_reg], df_vels.v_conv[no_reg], s=5, label=r"${\rm Entire\ Sun}$")
ax1.scatter(df_vels.mjd[is_moat], df_vels.v_conv[is_moat], s=5, label=r"${\rm Moat}$")
ax1.set_xlabel(r"${\rm MJD}$")
ax1.set_ylabel(r"$ \Delta v_{\rm conv}\ {\rm [ m s}^{-1} {\rm ]}$")
ax1.legend(ncol=2)#, fontsize=9)
fig.savefig(os.path.join(plotdir, "full_disk_region_time_series.pdf"))
# plt.show()
plt.clf(); plt.close()


# # get regions only
# df_vels_regs = df_vels[df_vels.region > 0.0]

# # plot quiet sun velocity by mu
# df_vels_quiet_sun = df_vels_regs[df_vels_regs.region == 4]

# # init fig
# fig = plt.figure()
# ax1 = fig.add_subplot()

# # get colormarp
# mus = np.unique(df_vels_quiet_sun.lo_mu)
# pdb.set_trace()
# mu_step = np.diff(mus)[0]
# cbar_vals = np.linspace(mus[0] - 0.5 * mu_step, mus[-1] + 0.5 * mu_step, len(mus)+1)
# cmap = copy.copy(plt.cm.rainbow)
# norm = mpl.colors.BoundaryNorm(cbar_vals, cmap.N)

# # loop over mu annuli
# for i in mus:
#     df_temp = df_vels_quiet_sun[df_vels_quiet_sun.lo_mu == i]
#     ax1.scatter(df_temp.mjd, df_temp.v_quiet, color=cmap(i), s=0.5)

# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=mus)
# cbar.set_label(r"$\mu\ {\rm bin center}$")
# cbar.ax.set_yticklabels(np.round(mus + 0.05, 2))

# ax1.set_xlabel(r"${\rm MJD}$")
# ax1.set_ylabel(r"${\rm Quiet-Sun\ Velocity\ (m/s)}$")
# fig.savefig(plotdir + "v_quiet_time_by_mu.pdf")
# plt.show()
# plt.clf(); plt.close()

# pdb.set_trace()


# # get full disk velocities
# df_full = df_all[np.isnan(df_all.lo_mu)]

# # get velocites for full mu annuli
# df_mu = df_all[(df_all.region == 0.0) & (~np.isnan(df_all.lo_mu))]

# # get velocities for regions in mu
# df_regs = df_all[(df_all.region > 0.0) & (~np.isnan(df_all.lo_mu))]

# # make time series
# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(df_full.mjd, df_full.v_hat, s=2, label="v_hat")
# ax1.scatter(df_full.mjd, df_full.v_phot, s=2, label="v_phot")
# ax1.scatter(df_full.mjd, df_full.v_conv, s=2, label="v_conv")
# ax1.scatter(df_full.mjd, df_full.v_quiet, s=2, label="v_quiet")
# ax1.set_xlabel(r"${\rm MJD}$")
# ax1.set_ylabel(r"${\rm Velocity (m/s)}$")
# ax1.legend(ncol=2, fontsize=9)
# fig.savefig(plotdir + "full_disk_series.pdf")
# plt.clf(); plt.close()

# # make time series
# fig = plt.figure()
# ax1 = fig.add_subplot()
# idx = (df_regs.region == 1) & (df_regs.v_hat != 0.0)
# # ax1.scatter(df_regs.mjd[idx], df_regs.v_hat[idx], s=2, label="v_hat")
# ax1.scatter(df_regs.mjd[idx], df_regs.v_phot[idx], s=2, label="v_phot")
# ax1.scatter(df_regs.mjd[idx], df_regs.v_conv[idx], s=2, label="v_conv")
# # ax1.scatter(df_regs.mjd[idx], df_regs.v_quiet[idx], s=2, label="v_quiet")
# ax1.set_xlabel(r"${\rm MJD}$")
# ax1.set_ylabel(r"${\rm Velocity (m/s)}$")
# ax1.set_title("Velocity in Spots")
# ax1.legend()
# fig.savefig(plotdir + "spot_series.pdf")
# plt.clf(); plt.close()

# # make time series
# fig = plt.figure()
# ax1 = fig.add_subplot()
# idx = (df_regs.region == 5) & (df_regs.v_hat != 0.0)
# # ax1.scatter(df_regs.mjd[idx], df_regs.v_hat[idx], s=2, label="v_hat")
# ax1.scatter(df_regs.mjd[idx], df_regs.v_phot[idx], s=2, label="v_phot")
# ax1.scatter(df_regs.mjd[idx], df_regs.v_conv[idx], s=2, label="v_conv")
# # ax1.scatter(df_regs.mjd[idx], df_regs.v_quiet[idx], s=2, label="v_quiet")
# ax1.set_xlabel(r"${\rm MJD}$")
# ax1.set_ylabel(r"${\rm Velocity (m/s)}$")
# ax1.set_title("Velocity in Plage")
# ax1.legend()
# fig.savefig(plotdir + "plage_series.pdf")
# plt.clf(); plt.close()

# # time series of other stuff
# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(df_stats.mjd, df_stats.ffactor, s=2, label="Mag. Filling Factor")
# ax1.scatter(df_stats.mjd, df_stats.plage_frac, s=2, label="Plage")
# ax1.scatter(df_stats.mjd, df_stats.network_frac, s=2, label="Network")
# ax1.scatter(df_stats.mjd, df_stats.pen_frac, s=2, label="Penmbrae")
# ax1.scatter(df_stats.mjd, df_stats.umb_frac, s=2, label="Umbrae")
# ax1.set_xlabel(r"${\rm MJD}$")
# ax1.set_ylabel(r"${\rm Fraction}$")
# ax1.set_ylim(0.0, 0.12)
# ax1.legend(fontsize=9)
# fig.savefig(plotdir + "frac_series.pdf")
# plt.clf(); plt.close()

# # correlation
# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(df_stats.ffactor, df_stats.plage_frac, s=2, label="Plage")
# ax1.scatter(df_stats.ffactor, df_stats.plage_frac + df_stats.network_frac, s=2, label="Plage + Network")
# ax1.set_xlabel(r"${\rm Magnetic\ Filling\ Factor}$")
# ax1.set_ylabel(r"${\rm Fraction}$")
# ax1.set_xlim(0.05, 0.125)
# ax1.set_ylim(0.00, 0.125)
# ax1.legend(fontsize=9)
# fig.savefig(plotdir + "frac_corr.pdf")
# plt.clf(); plt.close()
