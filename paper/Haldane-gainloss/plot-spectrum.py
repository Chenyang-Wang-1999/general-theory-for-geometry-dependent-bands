'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-03-03
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append("..")
import figure_data as fd

CM_IN_INCH = 1 / 2.54
DATA_FOLDER = "../../phcpy-free-algorithm/data"

wx = 3.929 # cm
wy = 3.1712 # cm
XLIM = (-3.4, 4.7)
YLIM = (-0.53, 0.53)
XTICKS = (-2, 0, 2, 4)
YTICKS = (-0.4, -0.2, 0, 0.2, 0.4)

def plot_OBC_spectrum(nx, ny, which=""):
    plt.style.use("../settings-and-materials/paper_plot.mplstyle")
    with open(f"{DATA_FOLDER}/paper-Haldane-gain-loss-OBC{which}-{nx}-{ny}.pkl", "rb") as f:
        eigv, eigvec, coord = pickle.load(f)
    
    fig = plt.figure(figsize=(
        wx * CM_IN_INCH / 0.8,
        wy * CM_IN_INCH / 0.8,
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    data_plt = ax.plot(eigv.real, eigv.imag, '.', markeredgecolor="none")
    fig.savefig(f"Figures/Haldane-OBC-spectrum-{nx}-{ny}{which}.png", dpi=1200, transparent=True)

    data_plt[0].remove()

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")
    
    fig.savefig(f"Figures/Haldane-OBC-spectrum-{nx}-{ny}{which}.pdf")
    plt.close(fig)

    fd.save_csv(
        f"Figures/Haldane-OBC-spectrum-{nx}-{ny}{which}",
        ("Fig. S12(e) | rhombic geometry, " if which == "" else
         "Fig. S12(f) | rectangular geometry, ") +
        f"(L1, L2) = ({nx}, {ny}); Re(E), Im(E) of the OBC spectrum",
        Re_E=eigv.real, Im_E=eigv.imag,
    )


def plot_SGBZ_spectrum(which="a1"):
    with open(f"{DATA_FOLDER}/Haldane-gain-loss-{which}-SGBZ-0.pkl", "rb") as fp:
        data = pickle.load(fp)
    E_real, E_imag, results, params = data["E_real"], data["E_imag"], data["results"], data["params"]
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    success_mesh = np.array([res["success"] and res["is_PMGBZ"] for res in results]).reshape(E_real_mesh.shape)

    plt.style.use("../settings-and-materials/paper_plot.mplstyle")
    fig = plt.figure(figsize=(
        wx * CM_IN_INCH / 0.8,
        wy * CM_IN_INCH / 0.8,
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)

    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    data_plt = ax.plot(E_real_mesh[success_mesh], E_imag_mesh[success_mesh], '.', markeredgecolor="none")
    fig.savefig(f"Figures/Haldane-SGBZ-spectrum-{which}.png", dpi=1200, transparent=True)

    data_plt[0].remove()

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    fig.savefig(f"Figures/Haldane-SGBZ-spectrum-{which}.pdf")

    fd.save_csv(
        f"Figures/Haldane-SGBZ-spectrum-{which}",
        ("Fig. S12(b) | SGBZ along the a1, a2 or x axis, " if which != "y" else
         "Fig. S12(c) | SGBZ along the y axis, ") +
        f"data set {which}: the scanned real energies whose SGBZ condition holds, "
        "with their imaginary part; Re(E), Im(E)",
        Re_E=E_real_mesh[success_mesh], Im_E=E_imag_mesh[success_mesh],
    )


if __name__ == "__main__":
    plot_OBC_spectrum(80, 80)
    plot_OBC_spectrum(80, 40, "-square")
    plot_SGBZ_spectrum("a1")
    plot_SGBZ_spectrum("a2")
    plot_SGBZ_spectrum("x")
    plot_SGBZ_spectrum("y")
