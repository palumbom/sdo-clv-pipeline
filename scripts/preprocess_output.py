import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import os, sys, pdb, csv, glob
import pandas as pd

from sdo_clv_pipeline.paths import root


def mask_all_zero_rows(df, return_idx=False):
    idx = (df.v_hat == 0.0) & (df.v_phot == 0.0) & (df.v_conv == 0.0) & (df.v_quiet == 0.0)
    if return_idx:
        return df[~idx], ~(idx.values)
    else:
        return df[~idx]
    return None

def mask_zero_v_conv(df):
    idx = (df.v_hat == df.v_quiet)
    return df[~idx]

def daily_bin(df):
    # create output df
    df_out = pd.DataFrame(columns=df.columns)

    # do a daily binning
    mjds, idx = np.unique(np.round(df.mjd), return_index=True)
    mus = np.unique(df.lo_mu)


    if np.all(np.isnan(mus)):
        # loop over unique dates and mus
        for i in range(len(mjds)-1):
            rows = df[idx[i]:idx[i+1]]
            to_append = rows.mean().to_frame(1).T
            df_out = pd.concat([df_out, to_append], ignore_index=True)
    else:
        # loop over unique dates and mus
        for i in range(len(mjds)-1):
            rows = df[idx[i]:idx[i+1]]
            for j in range(len(mus)):
                # select the rows
                to_append = (rows[rows.lo_mu == mus[j]]).mean().to_frame(1).T
                if np.all(np.isnan(to_append)):
                    continue

                # append to the output data frame
                df_out = pd.concat([df_out, to_append], ignore_index=True)

    # deal with inexact errors in mu values
    df_out.lo_mu = np.round(df_out.lo_mu.to_numpy(), decimals=1)
    df_out.hi_mu = np.round(df_out.hi_mu.to_numpy(), decimals=1)

    return df_out

# sort out paths
datadir = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))

# make directory for processed output
if not os.path.isdir(os.path.join(datadir, "processed")):
    os.mkdir(os.path.join(datadir, "processed"))

outdir = os.path.join(datadir, "processed")

print(datadir)
print(outdir)

# read in and sort by mjd
# df_all = pd.read_csv(os.path.join(datadir, "region_output.csv"))
# df_all = pd.read_csv("C:\\Users\\srugi\\Documents\\sdo-clv-pipeline\\data\\region_output.csv")
df_all = pd.read_csv("/Users/srugins/sdo-clv-pipeline/data/region_output.csv")
df_all.sort_values(by=["mjd", "region", "lo_mu"], inplace=True)
df_all.drop_duplicates()
df_all.reset_index(drop=True, inplace=True)

# get full disk only
df_full_disk = df_all[(np.isnan(df_all.lo_mu)) & np.isnan(df_all.region)]
df_full_disk.reset_index(drop=True, inplace=True)
df_full_disk.to_csv(os.path.join(outdir, "full_disk.csv"), index=False)
# full_disk_daily = daily_bin(df_full_disk)
# full_disk_daily.to_csv(outdir + "full_disk_daily.csv", index=False)

"""
# find dates with extreme outliers
v_conv_rolling_avg = df_full_disk.v_conv.rolling(30).mean()
v_conv_rolling_std = df_full_disk.v_conv.rolling(30).std()

# find distance from rolling avergge
dist = np.abs(df_full_disk.v_conv - v_conv_rolling_avg)
idx = dist[dist > 2.0 * v_conv_rolling_std].index
"""

# make dfs by mu
moat = df_all[df_all.region == 7.0]
plage = df_all[df_all.region == 6.0]
network = df_all[df_all.region == 5.0]
quiet_sun = df_all[df_all.region == 4.0]
red_penumbrae = df_all[df_all.region == 3.0]
all_penumbrae = df_all[df_all.region == 2.5]
blu_penumbrae = df_all[df_all.region == 2.0]
umbrae = df_all[df_all.region == 1.0]

# mask rows where all vels are 0.0 (i.e., region isn't present in that annulus)
moat = mask_all_zero_rows(moat)
moat.reset_index(drop=True, inplace=True)
moat.to_csv(os.path.join(outdir, "moat.csv"), index=False)
# moat_daily = daily_bin(moat)
# moat_daily.to_csv(outdir + "moat_daily.csv", index=False)

plage = mask_all_zero_rows(plage)
plage.reset_index(drop=True, inplace=True)
plage.to_csv(os.path.join(outdir, "plage.csv"), index=False)
# plage_daily = daily_bin(plage)
# plage_daily.to_csv(outdir + "plage_daily.csv", index=False)

network = mask_all_zero_rows(network)
network.reset_index(drop=True, inplace=True)
network.to_csv(os.path.join(outdir, "network.csv"), index=False)
# network_daily = daily_bin(network)
# network_daily.to_csv(outdir + "network_daily.csv", index=False)

quiet_sun = mask_all_zero_rows(quiet_sun)
quiet_sun.reset_index(drop=True, inplace=True)
quiet_sun.to_csv(os.path.join(outdir, "quiet_sun.csv"), index=False)
# quiet_sun_daily = daily_bin(quiet_sun)
# quiet_sun_daily.to_csv(outdir + "quiet_sun_daily.csv", index=False)

red_penumbrae = mask_all_zero_rows(red_penumbrae)
red_penumbrae.reset_index(drop=True, inplace=True)
red_penumbrae.to_csv(os.path.join(outdir, "red_penumbrae.csv"), index=False)
# red_penumbrae_daily = daily_bin(red_penumbrae)
# red_penumbrae_daily.to_csv(outdir + "red_penumbrae_daily.csv", index=False)

all_penumbrae = mask_all_zero_rows(all_penumbrae)
all_penumbrae.reset_index(drop=True, inplace=True)
all_penumbrae.to_csv(os.path.join(outdir, "penumbrae.csv"), index=False)
# all_penumbrae_daily = daily_bin(all_penumbrae)
# all_penumbrae_daily.to_csv(outdir + "penumbrae_daily.csv", index=False)

blu_penumbrae = mask_all_zero_rows(blu_penumbrae)
blu_penumbrae.reset_index(drop=True, inplace=True)
blu_penumbrae.to_csv(os.path.join(outdir, "blu_penumbrae.csv"), index=False)
# blu_penumbrae_daily = daily_bin(blu_penumbrae)
# blu_penumbrae_daily.to_csv(outdir + "blu_penumbrae_daily.csv", index=False)

umbrae = mask_all_zero_rows(umbrae)
umbrae.reset_index(drop=True, inplace=True)
umbrae.to_csv(os.path.join(outdir, "umbrae.csv"), index=False)
# umbrae_daily = daily_bin(umbrae)
# umbrae_daily.to_csv(outdir + "umbrae_daily.csv", index=False)


