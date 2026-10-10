'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-03-18
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt

CM_IN_INCH = 1 / 2.54
plt.style.use("../settings-and-materials/paper_plot.mplstyle")
DATA_DIR = "../../phcpy-free-algorithm/data"
wx = 3.929 # cm
wy = 3.1712 # cm
XLIM = (-3.4, 4.7)
YLIM = (-0.53, 0.53)
XTICKS = (-2, 0, 2, 4)
YTICKS = (-0.4, -0.2, 0, 0.2, 0.4)

def main_plot_DOS(n1, n2, N_real_grids, N_imag_grids, which):
    with open(f"{DATA_DIR}/paper-Haldane-DOS-OBC{which}-{n1}-{n2}-{N_real_grids}-{N_imag_grids}.pkl", "rb") as fp:
        point_num, real_grids, imag_grids = pickle.load(fp)
    if which == "":
        total_num = n1 * n2
    else:
        total_num = 2 * n1 * n2
    
    E_area = (real_grids[1] - real_grids[0]) * (imag_grids[1] - imag_grids[0])
    dos = point_num / E_area

    E_real_mesh, E_imag_mesh = np.meshgrid(real_grids, imag_grids, indexing="ij")

    fig = plt.figure(
        figsize=(
            wx * CM_IN_INCH / 0.8,
            wy * CM_IN_INCH / 0.8,
        )
    )
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlabel(r"Re($E$)")
    ax.set_ylabel(r"Im($E$)")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)

    pcl = ax.pcolor(E_real_mesh, E_imag_mesh, dos, vmin=0, vmax=total_num * 4)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig(f"Figures/paper-Haldane-DOS-OBC{which}-{n1}-{n2}.png", dpi=1200)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    pcl.remove()
    fig.savefig(f"Figures/paper-Haldane-DOS-OBC{which}-{n1}-{n2}.pdf")


if __name__ == "__main__":
    main_plot_DOS(80, 80, 201, 201, "")
    main_plot_DOS(120, 120, 201, 201, "")
    main_plot_DOS(80, 40, 201, 201, "-square")
    main_plot_DOS(120, 60, 201, 201, "-square")