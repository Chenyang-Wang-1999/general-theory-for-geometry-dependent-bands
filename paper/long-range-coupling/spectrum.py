'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-03-17
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

DATA_DIR = "../../long-range-coupling/data"
CM_IN_INCH = 1 / 2.54

def plot_spectrum(eigv, fname):
    plt.style.use("../settings-and-materials/paper_plot.mplstyle")
    fig = plt.figure(figsize=(
        4 / 0.8 * CM_IN_INCH, 4 / 0.8 * CM_IN_INCH
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlabel(r"Re($E$)")
    ax.set_ylabel(r"Im($E$)")
    ax.set_ylim([-0.42, 0.42])
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xlim([-2.6, 6.5])
    ax.set_xticks([-2, 0, 2, 4, 6])
    pl = ax.plot(eigv.real, eigv.imag, '.', markeredgecolor="none")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(f"Figures/{fname}.png", dpi=1200)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    pl[0].remove()

    fig.savefig(f"Figures/{fname}.pdf")


def plot_OBC_spectrum(nx, ny):
    with open(f"{DATA_DIR}/OBC-default-nx{nx}-ny{ny}.pkl", "rb") as f:
        params, coord_list, eigv, eigvec = pickle.load(f)
    plot_spectrum(eigv, f"OBC-spectrum-nx{nx}-ny{ny}")


def plot_x_SGBZ():
    results = []
    fid_start = 0
    fid_end = 21
    for fid in range(fid_start, fid_end + 1):
        with open(f"{DATA_DIR}/SGBZ-x{fid:03d}.pkl", "rb") as f:
            results.extend(pickle.load(f))

    err_list = []
    E_list = []
    for r in results:
        if r["success"]:
            E_list.extend(r["SGBZ_points"])
        else:
            err_list.append(r["mux"])

    E_list = np.array(E_list)
    plot_spectrum(E_list, "x-SGBZ-spectrum")


def plot_y_SGBZ():
    results = []
    fid_start = 0
    fid_end = 4
    for fid in range(fid_start, fid_end + 1):
        with open(f"{DATA_DIR}/y-SGBZ-results-{fid}.pkl", "rb") as f:
            results.extend(pickle.load(f))

    E_list = []
    for r in results:
        if r["success"]:
            for E_val, mu, theta1, theta2 in r["GBZ_points"]:
                E_list.append(E_val)
    E_list = np.array(E_list)
    plot_spectrum(E_list, "y-SGBZ-spectrum")


def plot_mux(E_real, mux, fname):
    plt.style.use("../settings-and-materials/paper_plot.mplstyle")
    fig = plt.figure(figsize=(
        4 / 0.8 * CM_IN_INCH, 2 / 0.8 * CM_IN_INCH
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlabel(r"Re($E$)")
    ax.set_ylabel(r"$\mu_x$")
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_xlim([-2.6, 6.5])
    ax.set_xticks([-2, 0, 2, 4, 6])
    pl = ax.plot(E_real, mux, '.', markeredgecolor="none")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(f"Figures/{fname}.png", dpi=1200)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    pl[0].remove()

    ax.plot([-2.6, 6.5], [1, 1], "k--")
    # ax.plot([-2.6, 6.5], [-1, -1], "k--")

    fig.savefig(f"Figures/{fname}.pdf")


def plot_x_SGBZ_mux():
    results = []
    fid_start = 0
    fid_end = 21
    for fid in range(fid_start, fid_end + 1):
        with open(f"{DATA_DIR}/SGBZ-x{fid:03d}.pkl", "rb") as f:
            results.extend(pickle.load(f))

    err_list = []
    E_list = []
    mux_list = []
    for r in results:
        if r["success"]:
            E_list.extend(r["SGBZ_points"])
            mux_list.extend(r["mux"] * np.ones_like(r["SGBZ_points"], dtype=float))
        else:
            err_list.append(r["mux"])

    E_list = np.array(E_list)
    mux_list = np.array(mux_list)
    plot_mux(E_list.real, mux_list, "x-SGBZ-mux")


def plot_y_SGBZ_mux():
    results = []
    fid_start = 0
    fid_end = 4
    for fid in range(fid_start, fid_end + 1):
        with open(f"{DATA_DIR}/y-SGBZ-results-{fid}.pkl", "rb") as f:
            results.extend(pickle.load(f))

    err_list = []
    E_list = []
    mux_list = []
    for r in results:
        if r["success"]:
            for E_val, mu, theta1, theta2 in r["GBZ_points"]:
                E_list.append(E_val)
                mux_list.append(mu)
        else:
            err_list.append(r["ky"])

    E_list = np.array(E_list)
    mux_list = np.array(mux_list)
    plot_mux(E_list.real, mux_list, "y-SGBZ-mux")


if __name__ == "__main__":
    # plot_OBC_spectrum(20, 300)
    # plot_OBC_spectrum(300, 20)
    # plot_OBC_spectrum(77, 77)
    # plot_x_SGBZ()
    # plot_y_SGBZ()
    plot_x_SGBZ_mux()
    plot_y_SGBZ_mux()
