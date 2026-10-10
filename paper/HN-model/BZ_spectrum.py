'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-02-26
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import HN_model_common as HN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import linalg as la
import pickle

# The color palette is shared by all the plotting scripts of this repository,
# and lives in colors.py at its root.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from colors import COLORS

CM_IN_INCH = 0.393701

# 创建一个纯色的colormap
def create_solid_color_cmap(rgb_color=(0.2, 0.4, 0.8)):
    """
    创建一个纯色的colormap，所有值都映射到相同的RGB颜色
    
    Parameters:
    - rgb_color: RGB颜色值，范围为0-1，例如(0.2, 0.4, 0.8)表示蓝色
    
    Returns:
    - ListedColormap对象
    """
    # 创建一个只有一个颜色的colormap
    colors = [rgb_color]  # 所有值都映射到这个颜色
    cmap = ListedColormap(colors)
    return cmap

def get_BZ_enenergy(kx, ky, Jx1, Jx2, Jy1, Jy2):
    return (
        Jx1 * np.exp(-1j * kx) + Jx2 * np.exp(1j * kx) +
        Jy1 * np.exp(-1j * ky) + Jy2 * np.exp(1j * ky)
    )

def plot_spectra(Jx1, Jx2, Jy1, Jy2):
    plt.style.use("../settings-and-materials/paper_plot.mplstyle")
    # 1. BZ spectra
    kx = np.linspace(-np.pi, np.pi, 1000)
    ky = np.linspace(-np.pi, np.pi, 1000)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    E = get_BZ_enenergy(kx_mesh, ky_mesh, Jx1, Jx2, Jy1, Jy2)

    cm = create_solid_color_cmap(COLORS["bf"])
    fig = plt.figure(figsize=(3.5 / 0.8 * CM_IN_INCH, 2.5344 / 0.8 * CM_IN_INCH))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-1.6, 1.6])
    pcl = ax.pcolor(E.real, E.imag, 0 * kx_mesh, cmap=cm, label="BZ")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig("Figures/BZ-spectrum.png", dpi=1200, transparent=True)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    bound, loop = HN.get_x_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2)
    line = ax.plot(loop.real, loop.imag, label="SGBZ/OBC", color=COLORS["ol"])
    ax.legend(handles=[pcl, line[0]])
    pcl.set_visible(False)

    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")
    fig.savefig("Figures/BZ-spectrum.pdf")

def check_spectra():
    Jx1, Jx2, Jy1, Jy2 = np.random.randn(4) + 1j * np.random.randn(4)
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    kx = np.random.rand() * 2 * np.pi
    ky = np.random.rand() * 2 * np.pi
    E1 = model.get_bulk_Hamiltonian_complex((np.exp(1j * kx), np.exp(1j * ky))).todense()[0,0]
    E2 = get_BZ_enenergy(kx, ky, Jx1, Jx2, Jy1, Jy2)
    print(E1, E2)
    assert np.isclose(E1, E2)

def calculate_OBC_eigenenergy(nx, ny, Jx1, Jx2, Jy1, Jy2):
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model_1d = model.get_supercell(
        [(j, 0) for j in range(nx)],
        np.array(
            [[nx, 0],
             [0, 1]], 
            dtype=int
        )
    )
    model_2d = model_1d.get_supercell(
        [(0, j) for j in range(ny)],
        np.array(
            [[1, 0],
             [0, ny]], 
            dtype=int
        )
    )
    H_mat = model_2d.get_bulk_Hamiltonian_complex((None, None)).todense()
    eigv, eigvec = la.eig(H_mat)
    with open("data/OBC_eigenenergy_no_phase.pkl", "wb") as f:
        pickle.dump((eigv, eigvec), f)


def plot_OBC_spectrum():
    Jx1, Jx2, Jy1, Jy2 = 1, 2, 0.5, 1
    with open("data/OBC_eigenenergy_no_phase.pkl", "rb") as f:
        eigv, _ = pickle.load(f)
    plt.plot(eigv.real, eigv.imag, '.', label="OBC")
    bound, loop = HN.get_x_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2)
    plt.plot(loop.real, loop.imag, label="SGBZ")
    plt.legend()
    plt.savefig("Figures/OBC_eigenenergy_no_phase.png")


if __name__ == "__main__":
    # plot_spectra(1, 2, 0.5, 1)
    # plot_SGBZ_bands(1, 2, 0.5, 1)
    # plt.savefig("Figures/SGBZ-bands.png")
    # check_spectra()
    # calculate_OBC_eigenenergy(50, 50, 1, 2, 0.5, 1)
    plot_OBC_spectrum()
