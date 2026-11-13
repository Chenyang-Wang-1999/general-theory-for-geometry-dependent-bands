'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-05-26
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

# The color palette is shared by all the plotting scripts of this repository,
# and lives in colors.py at its root.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from colors import COLORS
import figure_data as fd

DEFAULT_PARAMETERS = {
    "a0": 0.8,
    "c0": 0.4 * np.exp(1j * np.pi / 12), # b0 in revised manuscript
}
DATA_FOLDER = "../../phcpy-free-algorithm/data"
plt.style.use("../settings-and-materials/paper_plot.mplstyle")
CM_IN_INCH = 1/2.54
SQUARE_ZOOM_IN = ((1, 3), (-0.15, 0.15))

def get_theoretical_spectrum(Lx: int, Ly: int):
    m = np.arange(1, Lx + 1)
    a0 = DEFAULT_PARAMETERS["a0"]
    b0 = DEFAULT_PARAMETERS["c0"]
    Am = 2 * (1 + a0 * np.cos(m * np.pi / (Lx + 1)))
    Bm = 2 * (1 + b0 * np.cos(m * np.pi / (Lx + 1)))
    E_mesh = np.zeros((Lx, Ly), dtype=complex)
    sqrt_AB = np.sqrt(Am * Bm)
    for n_ind in range(Ly):
        E_mesh[:, n_ind] = 2 * sqrt_AB * np.cos((n_ind + 1) * np.pi / (Ly + 1))
    E_list = E_mesh.flatten()
    return E_list


def save_spectrum_data(stem, panel, E_num, amoeba_boundary_pts):
    '''Write the numerical spectrum and the amoebic boundary of one panel.'''
    fd.save_csv(
        "Figures/%s" % stem,
        panel + " | numerical OBC spectrum and the amoebic region boundary of the "
        "characteristic polynomial; Re(E), Im(E)",
        series=np.concatenate([np.full(E_num.size, "numerical"),
                               np.full(amoeba_boundary_pts.size, "amoebic")]),
        Re_E=np.concatenate([E_num.real, amoeba_boundary_pts.real]),
        Im_E=np.concatenate([E_num.imag, amoeba_boundary_pts.imag]),
    )

def plot_square_region():

    #### load data ####
    with open("%s/special-test-amoeba-boundary-rearranged.pkl" % DATA_FOLDER, "rb") as fp:
        amoeba_boundary_pts  = pickle.load(fp)

    with open("%s/special-test-OBC-spectrum-xy-57.pkl" % DATA_FOLDER, "rb") as f:
        eigv, eigvec, coords = pickle.load(f)
    E_num = eigv

    E_ana = get_theoretical_spectrum(57, 57)

    fig = plt.figure(figsize=(
        4 * CM_IN_INCH / 0.8,
        3 * CM_IN_INCH / 0.8
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlim([-6.6, 6.6])
    ax.set_ylim([-0.9, 0.9])
    ax.set_xticks(range(-8, 9, 4))
    ax.set_yticks(np.arange(-0.8, 0.9, 0.4))

    pl_ana = ax.plot(E_ana.real, E_ana.imag, 'o', label="Analytical", color=COLORS["gl"], markeredgewidth=0.1, markersize=0.8, markerfacecolor="none")
    pl_num = ax.plot(E_num.real, E_num.imag, '.', label="Numerical", color=COLORS["ol"], markersize=1, markeredgecolor="none")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig("Figures/square-region.png", dpi=1200, transparent=True)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.fill(amoeba_boundary_pts.real, amoeba_boundary_pts.imag, color=COLORS["bf"], label="Amoebic")
    ax.legend()

    pl_num[0].remove()
    pl_ana[0].remove()

    x_range, y_range = SQUARE_ZOOM_IN
    ax.plot(
        [x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]],
        [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]],
        "--",
        color=COLORS["rl"]
    )

    fig.savefig("Figures/square-region.pdf")
    plt.close(fig)

    save_spectrum_data("square-region", "Fig. S18(c) | square geometry, 57 x 57",
                       E_num, amoeba_boundary_pts)

def plot_diamond_region():
    #### load data ####
    with open("%s/special-test-amoeba-boundary-rearranged.pkl" % DATA_FOLDER, "rb") as fp:
        amoeba_boundary_pts  = pickle.load(fp)

    with open("%s/special-test-OBC-spectrum-diamond-40.pkl" % DATA_FOLDER, "rb") as f:
        eigv, eigvec, coords = pickle.load(f)
    E_num = eigv

    fig = plt.figure(figsize=(
        4 * CM_IN_INCH / 0.8,
        3 * CM_IN_INCH / 0.8
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlim([-6.6, 6.6])
    ax.set_ylim([-0.9, 0.9])
    ax.set_xticks(range(-8, 9, 4))
    ax.set_yticks(np.arange(-0.8, 0.9, 0.4))

    pl_num = ax.plot(E_num.real, E_num.imag, '.', label="Numerical", color=COLORS["ol"], markersize=1.5, markeredgecolor="none")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig("Figures/diamond-region.png", dpi=1200, transparent=True)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.fill(amoeba_boundary_pts.real, amoeba_boundary_pts.imag, color=COLORS["bf"], label="Amoebic")
    ax.legend()

    pl_num[0].remove()

    fig.savefig("Figures/diamond-region.pdf")
    plt.close(fig)

    save_spectrum_data("diamond-region", "Fig. S18(d) | diamond geometry, 40 x 40",
                       E_num, amoeba_boundary_pts)

def plot_square_region_zoomed_in():
    #### load data ####
    with open("%s/special-test-amoeba-boundary-rearranged.pkl" % DATA_FOLDER, "rb") as fp:
        amoeba_boundary_pts = pickle.load(fp)

    with open("%s/special-test-OBC-spectrum-xy-57.pkl" % DATA_FOLDER, "rb") as f:
        eigv, eigvec, coords = pickle.load(f)
    E_num = eigv

    E_ana = get_theoretical_spectrum(57, 57)

    x_range, y_range = SQUARE_ZOOM_IN
    w_parent = 6.6 * 2
    h_parent = 0.9 * 2
    x_ratio = (x_range[1] - x_range[0]) / w_parent
    y_ratio = (y_range[1] - y_range[0]) / h_parent
    fig = plt.figure(figsize=(
        4 * CM_IN_INCH / 0.8 * x_ratio * 5,
        3 * CM_IN_INCH / 0.8 * y_ratio * 5
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xticks([])
    ax.set_yticks([])

    pl_ana = ax.plot(E_ana.real, E_ana.imag, 'o', label="Analytical", color=COLORS["gl"], markeredgewidth=0.3, markersize=2.4, markerfacecolor="none")
    pl_num = ax.plot(E_num.real, E_num.imag, '.', label="Numerical", color=COLORS["ol"], markersize=3, markeredgecolor="none")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig("Figures/square-region-zoomed-in.png", dpi=1200, transparent=True)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.fill(amoeba_boundary_pts.real, amoeba_boundary_pts.imag, color=COLORS["bf"], label="Amoebic")
    ax.legend()

    pl_num[0].remove()
    pl_ana[0].remove()

    fig.savefig("Figures/square-region-zoomed-in.pdf")
    plt.close(fig)

    save_spectrum_data("square-region-zoomed-in",
                       "Fig. S18(c) zoom | square geometry, 57 x 57",
                       E_num, amoeba_boundary_pts)

if __name__ == "__main__":
    plot_square_region()
    plot_diamond_region()
    plot_square_region_zoomed_in()
