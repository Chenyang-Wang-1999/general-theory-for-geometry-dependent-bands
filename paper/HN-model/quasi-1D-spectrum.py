'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-05-30 00:00:00
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import pickle
import matplotlib.pyplot as plt
from cmath import sqrt

import numpy as np
import numpy_extended as npe
from scipy import sparse
from scipy.sparse import linalg as spla

import sys
sys.path.append("..")
import figure_settings_common as fs
import HN_model_common as HN

import palette_manager as pm

COLORs = pm.get_palette("matplotlib")


plt.style.use("../settings-and-materials/paper_plot.mplstyle")

FILL_COLOR = "#ff7f0eff"


def plot_quasi_1D_x():
    with open("../../puncture-calculation/data/paper-quasi-1D-x-N_15.pkl", "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)

    fig = plt.figure(figsize=(4*fs.cm, 4 * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    plt_E = ax.plot(E.real, E.imag, '.', markersize=2, markeredgecolor='none', label="QMGBZ")
    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-4, 4])
    ax.minorticks_off()

    # theoretical
    Jx1, Jx2, Jy1, Jy2 = params
    all_points = [
        2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
        2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
        -2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
        -2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
        2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
    ]
    all_points = np.asarray(all_points)
    # ax.plot(all_points.real, all_points.imag, 'r-')

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(
        "Figures/quasi-1D-x-N_15-E.png",
        dpi=1200,
        transparent=True
    )

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.fill(all_points.real, all_points.imag, color=FILL_COLOR, linewidth=0, label="SGBZ")
    ax.legend(handlelength=1)
    plt_E[0].remove()
    fig.savefig("Figures/quasi-1D-x-N_15-E.pdf")

    fig = plt.figure(figsize=(4*fs.cm, 2 * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.3, 1.3])
    # ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("$\\theta_x$ ($\pi$)")
    ax.set_ylabel("$\\mu_x$")
    ax.set_xticks([-1, 0, 1])

    r = sqrt(abs(Jx1/Jx2))
    # r_circ = r * np.exp(1j * np.linspace(0, 2 * np.pi, 500))
    # plt_beta = ax.plot(beta.real, beta.imag, '.', markersize=2)
    plt_beta = ax.plot(np.log(beta).imag / np.pi, np.log(beta).real, '.', markersize=2, label="QMGBZ")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(
        "Figures/quasi-1D-x-N_15-beta.png",
        dpi=1200,
        transparent=True
    )

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.plot([-1, 1], [np.log(r), np.log(r)], "--", label="SGBZ")
    # ax.plot(
    #     np.cos(np.linspace(0, 2 * np.pi, 100)),
    #     np.sin(np.linspace(0, 2 * np.pi, 100)), ":"
    # )
    # ax.plot([-1, 1], [1, 1], ":")
    ax.legend()
    plt_beta[0].remove()
    fig.savefig("Figures/quasi-1D-x-N_15-beta.pdf")


def plot_quasi_1D_y():
    with open("../../puncture-calculation/data/paper-quasi-1D-y-N_15.pkl", "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)

    fig = plt.figure(figsize=(4*fs.cm, 4 * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    plt_E = ax.plot(E.real, E.imag, '.', markersize=2, markeredgecolor='none', label="QMGBZ")
    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-4, 4])

    # theoretical
    Jx1, Jx2, Jy1, Jy2 = params
    all_points = [
        2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
        2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
        -2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
        -2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
        2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
    ]
    all_points = np.asarray(all_points)
    # ax.plot(all_points.real, all_points.imag, 'r-')

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(
        "Figures/quasi-1D-y-N_15-E.png",
        dpi=1200,
        transparent=True
    )

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.fill(all_points.real, all_points.imag, color=FILL_COLOR, linewidth=0, label="SGBZ")
    ax.legend(handlelength=1)
    plt_E[0].remove()
    fig.savefig("Figures/quasi-1D-y-N_15-E.pdf")

    fig = plt.figure(figsize=(4*fs.cm, 2 * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    ax.set_xlabel("$\\theta_y$ ($\\pi$)")
    ax.set_ylabel("$\\mu_y$")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.3, 1.3])
    ax.set_xticks([-1, 0, 1])

    r = sqrt(abs(Jy1/Jy2))
    # r_circ = r * np.exp(1j * np.linspace(0, 2 * np.pi, 500))
    plt_beta = ax.plot(np.log(beta).imag / np.pi, np.log(beta).real, '.', markersize=2, label="QMGBZ")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(
        "Figures/quasi-1D-y-N_15-beta.png",
        dpi=1200,
        transparent=True
    )

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    # ax.plot(r_circ.real, r_circ.imag, 'r--')
    ax.plot([-1, 1], [np.log(r), np.log(r)], '--', label="SGBZ")
    # ax.plot(
    #     np.cos(np.linspace(0, 2 * np.pi, 100)),
    #     np.sin(np.linspace(0, 2 * np.pi, 100)), ":"
    # )
    # ax.plot([-1,1], [1, 1], ":")
    ax.legend()
    plt_beta[0].remove()
    fig.savefig("Figures/quasi-1D-y-N_15-beta.pdf")


def plot_quasi_1D_11():
    with open("../../puncture-calculation/data/paper-quasi-1D-11-N_15.pkl", "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)
    Jx1, Jx2, Jy1, Jy2 = params

    fig = plt.figure(figsize=(4*fs.cm, 4 * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-4, 4])

    # theory
    bound = 2 * npe.continuous_sqrt(
        2 * sqrt(Jx1 * Jx2 * Jy1 * Jy2) * np.linspace(-1, 1, 500)
        + (Jx1 * Jx2)
        + (Jy1 * Jy2)
    )
    bound = np.hstack([bound, [0]])
    plt_E = ax.plot(E.real, E.imag, '.', markersize=2, markeredgecolor='none', label="QMGBZ")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(
        "Figures/quasi-1D-11-N_15-E.png",
        dpi=1200,
        transparent=True
    )

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.fill(bound.real, bound.imag, color=FILL_COLOR, linewidth=0, label="SGBZ")
    ax.fill(-bound.real, -bound.imag, color=FILL_COLOR, linewidth=0)
    ax.legend(handlelength=1)
    plt_E[0].remove()

    fig.savefig("Figures/quasi-1D-11-N_15-E.pdf")

    fig = plt.figure(figsize=(4*fs.cm, 2 * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    ax.set_xlabel(r"$\theta_{[11]}$ ($\pi$)")
    ax.set_ylabel(r"$\mu_{[11]}$")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.3, 1.3])
    # ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks([-1, 0, 1])

    # plt_beta = ax.plot(beta.real, beta.imag, '.', markersize=2)
    plt_beta = ax.plot(np.log(beta).imag / np.pi, np.log(beta).real, '.', markersize=2, label="QMGBZ")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(
        "Figures/quasi-1D-11-N_15-beta.png",
        dpi=1200,
        transparent=True
    )

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    r = sqrt(abs(Jx1 * Jy1 / Jx2 / Jy2))
    plt.plot([-1,1], [np.log(r), np.log(r)], '--', label="SGBZ")
    # ax.plot(
    #     np.cos(np.linspace(0, 2 * np.pi, 100)),
    #     np.sin(np.linspace(0, 2 * np.pi, 100)), ":"
    # )
    plt.legend()
    plt_beta[0].remove()
    plt.savefig("Figures/quasi-1D-11-N_15-beta.pdf")


def plot_minor_beta_11():
    # 11 direction
    with open("../../puncture-calculation/data/paper-quasi-1D-11-N_15.pkl", "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)
    Jx1, Jx2, Jy1, Jy2 = params

    fig = plt.figure(figsize=(4*fs.cm, 2 * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    beta_plt = ax.plot(np.log(beta).imag / np.pi, np.log(r_fit_list), '.', 
                       markersize=3, markeredgecolor='none', label="QMGBZ")
    theta = np.linspace(-np.pi, np.pi, 100)
    beta_th = np.sqrt(np.abs(Jx1 * Jy1 / Jx2 / Jy2)) * np.exp(1j * theta)
    r_th = np.sqrt(np.abs((Jx2 * beta_th + Jy1) / (Jx1 / beta_th + Jy2)))
    ax.set_xlabel(r"$\theta_{[11]}$ ($\pi$)")
    ax.set_ylabel(r"$\mu_{y}$")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.3, 1.3])
    # ax.set_yticks([0, 1, 2, 3])

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig(
        "Figures/quasi-1D-11-N_15-minor.png",
        dpi=1200,
        transparent=True
    )
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.plot(theta / np.pi, np.log(r_th), '--', label="SGBZ")
    ax.legend()
    beta_plt[0].remove()
    fig.savefig("Figures/quasi-1D-11-N_15-minor.pdf")


def plot_minor_beta_x():
    # 11 direction
    with open("../../puncture-calculation/data/paper-quasi-1D-x-N_15.pkl", "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)
    Jx1, Jx2, Jy1, Jy2 = params

    fig = plt.figure(figsize=(4*fs.cm, 2 * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    beta_plt = ax.plot(np.log(beta).imag / np.pi, np.log(r_fit_list), '.', 
                       markersize=3, markeredgecolor='none', label="QMGBZ")
    theta = np.linspace(-np.pi, np.pi, 100)
    r_th = np.sqrt(np.abs(Jy1 / Jy2)) * np.ones_like(theta)
    ax.set_xlabel(r"$\theta_{x}$ ($\pi$)")
    ax.set_ylabel(r"$\mu_y$")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.3, 1.3])
    # ax.set_yticks([0, 1, 2, 3])

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig(
        "Figures/quasi-1D-x-N_15-minor.png",
        dpi=1200,
        transparent=True
    )
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.plot(theta / np.pi, np.log(r_th), '--', label="SGBZ")
    ax.legend()
    beta_plt[0].remove()
    fig.savefig("Figures/quasi-1D-x-N_15-minor.pdf")


def plot_minor_beta_y():
    # 11 direction
    with open("../../puncture-calculation/data/paper-quasi-1D-y-N_15.pkl", "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)
    Jx1, Jx2, Jy1, Jy2 = params

    fig = plt.figure(figsize=(4*fs.cm, 2 * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    beta_plt = ax.plot(np.log(beta).imag / np.pi, np.log(r_fit_list), '.', 
                       markersize=3, markeredgecolor='none', label="QMGBZ")
    theta = np.linspace(-np.pi, np.pi, 100)
    r_th = np.sqrt(np.abs(Jx1 / Jx2)) * np.ones_like(theta)
    ax.set_xlabel(r"$\theta_{y}$ ($\pi$)")
    ax.set_ylabel(r"$\mu_x$")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.3, 1.3])
    # ax.set_yticks([0, 1, 2, 3])

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig(
        "Figures/quasi-1D-y-N_15-minor.png",
        dpi=1200,
        transparent=True
    )
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.plot(theta / np.pi, np.log(r_th), '--', label="SGBZ")
    ax.legend()
    beta_plt[0].remove()
    fig.savefig("Figures/quasi-1D-y-N_15-minor.pdf")


def plot_eigenstates_beta_11():
    ''' Plot distributions for eigenstates '''
    with open("../../puncture-calculation/data/paper-quasi-1D-11-N_15.pkl", "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)
    model = HN.get_HN_model(*params)
    model = model.get_supercell(
        [(0, 0)],
        np.array([
            [1, 0],
            [1, 1]
        ], dtype=int)
    )

    model_super = model.get_supercell(
        [(0,j) for j in range(N)],
        np.array([
            [1, 0],
            [0, N]
        ], dtype=int)
    )
    model_super = model_super.truncate(1)

    fig = plt.figure(figsize=(4,2))
    ax = fig.gca()

    color_map = plt.cm.viridis
    norm_fun = plt.Normalize(-np.pi, np.pi)
    for point_ind in range(len(E)):
        curr_E = E[point_ind]
        curr_beta = beta[point_ind]
        curr_H = model_super.get_bulk_Hamiltonian_complex((curr_beta,))
        curr_eigv, curr_eigvec = spla.eigs(
            curr_E * sparse.eye(curr_H.shape[0]) - curr_H,
            k=1, which='SM'
        )

        # Check eigenstate solution
        if np.abs(curr_eigv[0]) > 1e-4:
            print(curr_eigv[0])
        ax.plot(np.arange(len(curr_eigvec)), np.abs(curr_eigvec.flatten()), linewidth=0.5, 
                color=color_map(norm_fun(np.log(curr_beta).imag)))
    plt.show()


if __name__ == '__main__':
    plot_quasi_1D_x()
    plot_quasi_1D_y()
    plot_quasi_1D_11()
    plot_minor_beta_11()
    plot_minor_beta_x()
    plot_minor_beta_y()
    # plot_eigenstates_beta_11()
