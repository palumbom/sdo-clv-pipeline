import matplotlib.cm as cm
import numpy as np
import pdb, warnings
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sunpy.map import Map as sun_map
from sunpy.coordinates import frames

from scipy import ndimage
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from skimage.measure import regionprops
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from string import ascii_letters

from .sdo_io import *
from .limbdark import *
from .legendre import *
from .reproject import *


def load_and_plot():
    moat_file = os.path.join(root, "data", "moats_data.npz")
    data = np.load(moat_file)

    x = data['x']
    vels = data['vels']
    mags = data['mags'] 
    ints = data['ints']
    areas = data['areas']
    mus = data['mus']
    moats = []
    moats.append(vels)  # 0 
    moats.append(mags)  # 1
    moats.append(ints)  # 2 
    moats.append(areas) # 3
    moats.append(mus)   # 4

    # print(np.shape(x))
    print("outside plot loop")

    for j in range (0, 3): # vel, mag, int
        print ("trying to plot...")
        # layered plots for different moats
        thetas = []
        # plot avg velocities / dilations, mu
        for i in mus:
            i = np.arccos(i)
            thetas.append(i)
        cmap = cm.plasma
        norm = colors.Normalize(vmin=min(thetas), vmax=np.max(thetas))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        letters = []
        for i in range (0, len(thetas)):
            print(thetas[i])
            color = cmap(norm(thetas[i]))
            label = ascii_letters[i%52]
            letters.append(label)
            plt.plot(x[i], moats[j][i], color = color)
            # marker
            mark_dilation = np.sqrt(areas[i] / np.pi)
            vel_at_mark = np.interp(mark_dilation, x[i], moats[j][i])
            plt.plot(mark_dilation, vel_at_mark, marker='o', color=color, markersize=5)
            # letter
            plt.text(mark_dilation+0.2, vel_at_mark+0.5, f' {label}', fontsize=9)
        plt.colorbar(sm, label='Average Theta (rad)', ax=plt.gca())
        plt.xlabel("# of Dilations")
        if j == 0: 
            plt.ylabel("Average Velocity (m/s)")
            plt.title("Average Velocity vs # of Dilations")
        elif j == 1:
            plt.ylabel("Average Magnetic Field (G)")
            plt.title("Average Magnetic Field Strength vs # of Dilations")
        else:
            plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
            plt.title("Average Intensity vs # of Dilations")
        plt.show()

        # plot avg velocities / dilations, area
        cmap = cm.plasma
        norm = colors.Normalize(vmin=min(areas), vmax=np.max(areas))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        for i in range (0, len(areas)):
            color = cmap(norm(areas[i]))
            label = ascii_letters[i%52] 
            plt.plot(x[i], moats[j][i], color = color)
            # marker
            mark_dilation = np.sqrt(areas[i] / np.pi)
            vel_at_mark = np.interp(mark_dilation, x[i], moats[j][i])
            plt.plot(mark_dilation, vel_at_mark, marker='o', color=color, markersize=5)
            # letter
            plt.text(mark_dilation+0.2, vel_at_mark+0.5, f' {label}', fontsize=9)
        plt.colorbar(sm, label='Area of Spot in Pixels', ax=plt.gca())
        plt.xlabel("# of Dilations")
        if j == 0:
            plt.ylabel("Average Velocity (m/s)")
            plt.title("Average Velocity vs # of Dilations")
        elif j == 1: 
            plt.ylabel("Average Magnetic Field (G)")
            plt.title("Average Magnetic Field Strength vs # of Dilations")
        else:
            plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
            plt.title("Average Intensity vs # of Dilations")
        plt.show()

def plot_loop(moat_vals, moat_dilations, moat_thetas, moat_areas):

    for j in range (0, 3): # vel, mag, int
        print ("trying to plot...")
        # layered plots for different moats
        cmap = cm.plasma
        norm = colors.Normalize(vmin=min(moat_thetas), vmax=np.max(moat_thetas))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        letters = []
        for i in range (0, len(moat_thetas)):
            # print(moat_thetas[i])
            color = cmap(norm(moat_thetas[i]))
            #label = ascii_letters[i%52]
            if moat_vals[5][i] == 1:
                label = f"*{moat_thetas[i]:.3f}"
            else:
                label = f"{moat_thetas[i]:.3f}"
            letters.append(label)
            plt.plot(moat_dilations[i], moat_vals[j][i], color = color)
            # marker
            mark_dilation = np.sqrt(moat_areas[i] / np.pi)
            vel_at_mark = np.interp(mark_dilation, moat_dilations[i], moat_vals[j][i])
            plt.plot(mark_dilation, vel_at_mark, marker='o', color=color, markersize=5)
            # letter
            plt.text(mark_dilation+1, vel_at_mark+0.5, f' {label}', fontsize=7)
        plt.colorbar(sm, label='Average Theta (rad)', ax=plt.gca())
        plt.xlabel("# of Dilations")
        if j == 0: 
            plt.ylabel("Average Velocity (m/s)")
            plt.title("Average Velocity vs # of Dilations")
        elif j == 1:
            plt.ylabel("Average Magnetic Field (G)")
            plt.title("Average Magnetic Field Strength vs # of Dilations")
        else:
            plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
            plt.title("Average Intensity vs # of Dilations")
        plt.tight_layout()
        plt.show()

        # # plot avg velocities / dilations, area
        # cmap = cm.plasma
        # norm = colors.Normalize(vmin=min(areas), vmax=np.max(areas))
        # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        # sm.set_array([])
        # for i in range (0, len(areas)):
        #     color = cmap(norm(areas[i]))
        #     label = ascii_letters[i%52] 
        #     plt.plot(x[i], moats[j][i], color = color)
        #     # marker
        #     mark_dilation = np.sqrt(areas[i] / np.pi)
        #     vel_at_mark = np.interp(mark_dilation, x[i], moats[j][i])
        #     plt.plot(mark_dilation, vel_at_mark, marker='o', color=color, markersize=5)
        #     # letter
        #     plt.text(mark_dilation+0.2, vel_at_mark+0.5, f' {label}', fontsize=9)
        # plt.colorbar(sm, label='Area of Spot in Pixels', ax=plt.gca())
        # plt.xlabel("# of Dilations")
        # if j == 0:
        #     plt.ylabel("Average Velocity (m/s)")
        #     plt.title("Average Velocity vs # of Dilations")
        # elif j == 1: 
        #     plt.ylabel("Average Magnetic Field (G)")
        #     plt.title("Average Magnetic Field Strength vs # of Dilations")
        # else:
        #     plt.ylabel("Average Intensity (ergs / s / Hz / m^2)")
        #     plt.title("Average Intensity vs # of Dilations")
        # plt.show()

def plot_avg_vel (moat_avg_vels, moat_thetas, moat_vals):
    
    for i in range (0, len(moat_thetas)):
            if moat_vals[5][i] == 1:
                label = f"*{moat_thetas[i]:.3f}"
            else:
                label = f"{moat_thetas[i]:.3f}"
            plt.scatter(moat_thetas, moat_avg_vels)
            plt.text(moat_thetas[i], moat_avg_vels[i], f' {label}', fontsize=7)
    plt.xlabel("Theta (rad)")
    plt.ylabel("Moat Total Average Velocity (m/s)")
    plt.title("Average Velocity of Moat vs Theta")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # load_and_plot()
    plot_loop()