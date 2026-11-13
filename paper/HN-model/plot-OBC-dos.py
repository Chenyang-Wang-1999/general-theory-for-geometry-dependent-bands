'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-03-14
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle

import sys
sys.path.append("..")
import figure_data as fd

DATA_FOLDER = "../../puncture-calculation/data"
CM_IN_INCH = 1 / 2.54

def plot_OBC_dos(n1, n2, with_axes=False):
    plt.style.use("../settings-and-materials/paper_plot.mplstyle")
    fig = plt.figure(
        figsize=(2 * CM_IN_INCH / 0.8, 1.5 * CM_IN_INCH / 0.8)
    )
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])

    result_fname = DATA_FOLDER + "/paper-HN-OBC-2-DOS-n1-%d-n2-%d.pkl"%(n1, n2)
    with open(result_fname, "rb") as f:
        state_count, E_real_grid, E_imag_grid = pickle.load(f)
    E_area = (E_real_grid[1] - E_real_grid[0]) * (E_imag_grid[1] - E_imag_grid[0])

    ax.set_xlim(E_real_grid[0], E_real_grid[-1])
    ax.set_xticks(range(-4, 5, 4))
    ax.set_yticks([-2,0,2])
    ax.set_ylim(E_imag_grid[0], E_imag_grid[-1])

    print(500 * E_area)
    pcl = ax.pcolormesh(
        E_real_grid, E_imag_grid, state_count.T / E_area,
        vmin=0, vmax=n1*n2/6)

    fd.save_grid(
        "Figures/paper-HN-OBC-2-DOS-n1-%d-n2-%d" % (n1, n2),
        E_real_grid, E_imag_grid, state_count.T / E_area,
        "Fig. S10 | density of states of the 2D HN model with (L1, L2) = (%d, %d), "
        "normalized per cell area; the panel's vmax is L1*L2/6" % (n1, n2),
        x_name="Re_E", y_name="Im_E", value_name="dos",
    )
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig("Figures/paper-HN-OBC-2-DOS-n1-%d-n2-%d.png"%(n1, n2), dpi=1200)

    if with_axes:
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        pcl.remove()
        ax.set_xlabel("Re($E$)")
        ax.set_ylabel("Im($E$)")
        fig.savefig("Figures/paper-HN-OBC-2-DOS-n1-%d-n2-%d-axes.pdf"%(n1, n2))


def batch_plot(plot=True):
    for n1_over_n2 in [2, 4, 6, 8, 10, 12, 14, 16]:
        totol_num_list = np.linspace(3600, 25600, 8)
        # totol_num_list = np.array([51200])
        n2_list = np.round(np.sqrt(totol_num_list / n1_over_n2)).astype(int)
        n2_list = np.unique(n2_list)
        print("Aspect: ", n1_over_n2, ":1 =========")
        for n2 in n2_list:
            print(n1_over_n2 * n2, n2)
            if plot:
                plot_OBC_dos(n1_over_n2 * n2, n2, with_axes=False)


if __name__ == "__main__":
    # plot_OBC_dos(640, 40, with_axes=True)
    batch_plot()
