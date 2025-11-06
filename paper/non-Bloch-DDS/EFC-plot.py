'''
author:        Wang Chenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-06-24
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' Plot equi-frequency countour '''

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, pi
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from color_set import create_white_to_color_cmap

CM_IN_INCH = 1 / 2.54


def plot_EFC():
    omega0 = sqrt(3)

    theta_y = np.linspace(-pi, pi, 5000)
    cos_theta_x = omega0 - sqrt(3) / 2 * np.cos(theta_y)
    selected_idx = (cos_theta_x >= -1) & (cos_theta_x <= 1)
    theta_y_real = theta_y[selected_idx]
    theta_x_real = np.arccos(cos_theta_x[selected_idx])

    theta_x_total = np.hstack([
        theta_x_real, - theta_x_real[::-1], [theta_x_real[0]]
    ])
    theta_y_total = np.hstack([
        theta_y_real, theta_y_real[::-1], [theta_y_real[0]]
    ])

    E_imag = np.cos(theta_y_total) / 2

    # plot
    # 1. set colormap
    cmap = create_white_to_color_cmap(np.array([0xff, 0x7f, 0x0e]) / 255, gray=0.9)

    # 2. plot with cmap
    points = np.array([theta_x_total, theta_y_total]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments, 
        cmap=cmap, 
        norm=plt.Normalize(E_imag.min(), E_imag.max())
    )
    lc.set_array(E_imag)

    fig = plt.figure(
        figsize=(4 * CM_IN_INCH, 4 * CM_IN_INCH)
    )
    plt.style.use("../settings-and-materials/paper_plot.mplstyle")
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    line = ax.add_collection(lc)
    ax.set_xlim([-pi / 2, pi / 2])
    ax.set_ylim([-pi / 2, pi / 2])
    ax.set_aspect(1)
    clb = plt.colorbar(line, ax=ax)
    clb.set_ticks([])

    fig.savefig("Figures/EFC.pdf")


if __name__ == "__main__":
    plot_EFC()