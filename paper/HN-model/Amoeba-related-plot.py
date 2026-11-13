'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-10-28
Copyright © Department of Physics, Tsinghua University. All rights reserved

Compare the Amoeba and 11-directional GBZ spectra along with the non-Bloch solutions
'''

import numpy as np
import HN_model_common as HN
import pickle
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use


import sys
sys.path.append("..")
import figure_settings_common as fs
import figure_data as fd
plt.style.use("../settings-and-materials/paper_plot.mplstyle")


NON_BLOCH_DATA = "../../puncture-calculation/data/"
AMOEBA_DATA = "../../Amoeba-calculation/data/"


def plot_non_Bloch_sols(E_ref):
    ''' Plot non-Bloch solutions for a given reference energy E_ref '''
    name_str = "paper-non-Bloch-sol-E_%.2f_%.2f" % (E_ref.real, E_ref.imag)
    # load data
    with open(NON_BLOCH_DATA + name_str + '.pkl', "rb") as fp:
        all_loops = pickle.load(fp)

    # plot
    fig = plt.figure(figsize=(
        3 * fs.cm, 3 * fs.cm
    ))
    ax = fig.gca()
    ax.set_position([
        0.2, 0.2, 0.75, 0.75
    ])
    ax.set_xlabel("Reb1")
    ax.set_ylabel("Imb1")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])

    for curr_loop in all_loops:
        ax.plot(curr_loop[:,0].real, curr_loop[:,0].imag, color=fs.DEFAULT_COLORS['blue'])
    theta = np.linspace(0, 2 * np.pi, 100)
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    r = np.sqrt(np.abs(Jx1 * Jy1 / Jx2 / Jy2))
    ax.plot(r * np.cos(theta), r * np.sin(theta), color=fs.DEFAULT_COLORS['red'],
            linewidth=0.5)

    fig.savefig("Figures/%s.pdf" % (name_str))


def plot_Amoeba(E_ref):
    ''' Plot Amoeba for the reference energy E_ref '''
    name_str = "paper-Amoeba-E_%.2f_%.2f" % (E_ref.real, E_ref.imag)
    
    # load data
    with open(AMOEBA_DATA + name_str + ".pkl", "rb") as fp:
        all_amoeba = pickle.load(fp)

    # Text data behind the panel.  Each entry of ``all_amoeba`` is a degenerate
    # mesh: the rows of ``x_norm`` are all equal and ``y_norm[:, :, k]`` spans a
    # band for every column, so the shaded region is the band [min, max] of
    # ``ln|y|`` at a fixed ``ln|x|``.  The pcolor call decides which of the two
    # is drawn horizontally: ``x_major_flag`` puts ``ln|x|`` on the x axis (the
    # band runs along y) and the other orientation puts ``ln|y|`` on the x axis
    # (the band runs along x), so the flag is written as the ``axis`` column.
    # Storing the two envelopes reproduces the drawn region exactly and keeps
    # the file at about 0.1 MB instead of the 45 MB of the full mesh.
    envelopes = []
    for entry_ind, (curr_x_norm, curr_y_norm, x_major_flag) in enumerate(all_amoeba):
        for branch_ind in range(curr_y_norm.shape[2]):
            envelopes.append((
                entry_ind, branch_ind,
                "x" if x_major_flag else "y",
                np.log(curr_x_norm[0, :]),
                np.log(curr_y_norm[:, :, branch_ind]),
            ))
    fd.save_csv(
        "Figures/%s" % name_str,
        "Fig. S19 | amoeba of the characteristic polynomial at E = (%.2f%+.2fj); every "
        "entry/branch of the shaded region is the band [mu_min, mu_max] at a fixed "
        "mu; axis = x means that the band runs along mu_y at a fixed mu_x, axis = y "
        "the other way round" % (E_ref.real, E_ref.imag),
        entry=np.concatenate([np.full(band.size, entry)
                              for entry, _, _, band, _ in envelopes]),
        branch=np.concatenate([np.full(band.size, branch)
                               for _, branch, _, band, _ in envelopes]),
        axis=np.concatenate([np.full(band.size, axis)
                             for _, _, axis, band, _ in envelopes]),
        mu=np.concatenate([band for _, _, _, band, _ in envelopes]),
        mu_min=np.concatenate([mesh.min(axis=0)
                               for _, _, _, _, mesh in envelopes]),
        mu_max=np.concatenate([mesh.max(axis=0)
                               for _, _, _, _, mesh in envelopes]),
    )

    # plot
    fig = plt.figure(figsize=(
        3 * fs.cm, 3 * fs.cm
    ))
    ax = fig.gca()
    ax.set_position([
        0.2, 0.2, 0.75, 0.75
    ])
    ax.set_xlabel("mux")
    ax.set_ylabel("muy")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])

    fig.savefig("Figures/%s.pdf" % (name_str))
    for curr_pair in all_amoeba:
        curr_x_norm, curr_y_norm, x_major_flag = curr_pair
        for layer_ind in range(curr_y_norm.shape[2]):
            if x_major_flag:
                ax.pcolor(
                    np.log(curr_x_norm),
                    np.log(curr_y_norm[:,:,layer_ind]),
                    np.ones_like(curr_x_norm) * 0.5,
                    cmap="Greys",
                    vmax=1,
                    vmin=0
                )
            else:
                ax.pcolor(
                    np.log(curr_y_norm[:,:,layer_ind]),
                    np.log(curr_x_norm),
                    np.ones_like(curr_x_norm) * 0.5,
                    cmap="Greys",
                    vmax=1,
                    vmin=0
                )
    fig.savefig("Figures/%s.png" % (name_str), dpi=4800)


def get_minor_beta(E_ref, beta1):
    ''' Get minor beta for a given value of E and major beta, (1,1)-direction '''
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    beta2_sol = np.roots([
        Jx1 / beta1 + Jy2,
        -E_ref,
        Jx2 * beta1 + Jy1
    ])
    return beta2_sol


def batch_get_minor_beta_abs(E_ref, beta1_arr):
    ''' Get the absolute value of minor beta with an array of beta1 '''
    beta2_norm = np.zeros((2, len(beta1_arr)), dtype=float)
    for point_ind, beta1 in enumerate(beta1_arr):
        beta2_sol = get_minor_beta(E_ref, beta1)
        beta2_norm[:,point_ind] = np.sort(np.abs(beta2_sol))
    return beta2_norm


def plot_minor_beta_abs(E_ref):
    ''' Plot the absolute value of minor beta '''
    mpl_use('pdf')
    # Generate beta1 
    theta = np.linspace(-np.pi, np.pi, 2001)
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    r = np.sqrt(np.abs(Jx1 * Jy1 / Jx2 / Jy2))
    beta1 = r * np.exp(1j * theta)

    # Generate abs of beta2
    beta2_abs = batch_get_minor_beta_abs(E_ref, beta1)

    # Plot
    fig = plt.figure(figsize=(
        3 * fs.cm, 3 * fs.cm
    ))
    ax = fig.gca()
    ax.set_position([
        0.2, 0.2, 0.75, 0.75
    ])
    ax.set_xlabel("Arg(beta1)")
    ax.set_ylabel("ln|beta2|")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-3, 3])
    ax.set_xticks([-1, 1])
    ax.set_yticks([-3, 3])
    ax.set_xticklabels(["-pi", "pi"])

    ax.plot(theta / np.pi, np.log(beta2_abs[0,:]), color=fs.DEFAULT_COLORS['blue'], linewidth=0.7)
    ax.plot(theta / np.pi, np.log(beta2_abs[1,:]), color=fs.DEFAULT_COLORS['orange'], linewidth=0.7)

    ax.legend(['(M2)', '(M2+1)'])

    fig.savefig("Figures/paper-abs-minor-beta-E_%.2f_%.2f.pdf" % (E_ref.real, E_ref.imag))


def plot_Amoeba_and_GBZ_spectra():
    fig = plt.figure(figsize=(
        4 * fs.cm, 4 * 3.5 / 5 * fs.cm
    ))
    ax = fig.gca()
    ax.set_position([
        0.2, 0.25, 0.75, 0.7
    ])
    ax.set_ylim([-3.5, 3.5])
    ax.set_xlabel("ReE")
    ax.set_ylabel("ImE")

    bound, loop = HN.get_x_DGBZ_spectrum(*HN.Default_Model_Params)
    ax.fill(loop.real, loop.imag)
    bound, loop = HN.get_11_DGBZ_spectrum(*HN.Default_Model_Params)
    ax.fill(loop.real, loop.imag)

    ax.plot([1, 2, 3], [1, 2, 3], '.')

    fig.savefig("Figures/Amoeba-spectra.pdf")



if __name__ == '__main__':
    for E_ref in [1 + 1j, 2 + 2j, 3 + 3j]:
        # plot_non_Bloch_sols(E_ref)
        plot_Amoeba(E_ref)
        # plot_minor_beta_abs(E_ref)
    plot_Amoeba_and_GBZ_spectra()
