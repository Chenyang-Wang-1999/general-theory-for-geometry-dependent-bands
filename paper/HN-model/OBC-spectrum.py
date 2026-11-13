'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-09-04
Copyright © Department of Physics, Tsinghua University. All rights reserved

Plot OBC spectrum and compare with strip GBZ
'''

import pickle
import HN_model_common as HN
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import figure_settings_common as fs
import figure_data as fd
import numpy as np

plt.style.use('../settings-and-materials/paper_plot.mplstyle')
J_X1, J_X2, J_Y1, J_Y2 = HN.Default_Model_Params
DATA_DIR = "../../puncture-calculation/data"


def test_OBC_xy():
    ''' Plot OBC spectrum in x-y direction '''
    for N in [20, 30, 40]:
        for aspect in [1/8, 1/4, 1/2, 1, 2, 4, 8]:
            with open("%s/paper-HN-OBC-2-xy-aspect_%.6f-N_%d.pkl" % (
                DATA_DIR, aspect, N
            ), "rb") as fp:
                eigv, eigvec, point_vec, params, shape = pickle.load(fp)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(eigv.real, eigv.imag, '.')
            ax.set_aspect(1)
            ax.set_title("%d, %d, total %d" % (
                shape[0],
                shape[1],
                shape[0] * shape[1]
            ))
            plt.show()


def plot_OBC_with_given_aspect(
        aspect: float,
        N: int
):
    ''' Given the aspect and site number, plot the spectrum and save '''
    ### load data ###
    with open("%s/paper-HN-OBC-2-xy-aspect_%.6f-N_%d.pkl" % (
        DATA_DIR, aspect, N
    ), "rb") as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    ### create figure ###
    ax_size = 2.45
    fig_size = ax_size / 0.8
    fig = plt.figure(figsize=(fig_size * fs.cm, fig_size * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlabel("ReE")
    ax.set_ylabel("ImE")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-4, 4])

    ### plot spectrum and save as bitmap
    dots = ax.plot(
        eigv.real, eigv.imag, '.',
        markersize=2, markeredgecolor='none'
    )

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(
        "Figures/xy-%.6f-%d-%d-%d.png" % (aspect, shape[0], shape[1], shape[0] * shape[1]),
        dpi=600
    )

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ### plot strip GBZ ###
    bound, loop = HN.get_x_DGBZ_spectrum(J_X1, J_X2, J_Y1, J_Y2)
    strip_GBZ = ax.plot(loop.real, loop.imag, 'r--')
    ax.legend(dots + strip_GBZ, ["OBC", "strip GBZ"])
    dots[0].remove()
    fig.savefig(
        "Figures/xy-%.6f-%d-%d-%d.pdf" % (aspect, shape[0], shape[1], shape[0] * shape[1]),
    )

    fd.save_csv(
        "Figures/xy-%.6f-%d-%d-%d" % (aspect, shape[0], shape[1], shape[0] * shape[1]),
        "Fig. 3(e) | OBC spectrum of the 2D HN model in the xy-parallelogram with "
        "(L1, L2) = (%d, %d); Re(E), Im(E)" % (shape[0], shape[1]),
        Re_E=eigv.real, Im_E=eigv.imag,
    )


def plot_11_OBC_spectrum(
        aspect: float,
        N: int,
        which_strip_GBZ=0    # 0: None, 1: (1,1), 2: xy
):
    ''' Given the aspect and site number, plot the spectrum and save '''
    ### load data ###
    with open("%s/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (
        DATA_DIR, aspect, N
    ), "rb") as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    ### create figure ###
    ax_size = 2.45
    fig_size = ax_size / 0.8
    fig = plt.figure(figsize=(fig_size * fs.cm, fig_size * fs.cm))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlabel("ReE")
    ax.set_ylabel("ImE")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-4, 4])

    ### plot spectrum and save as bitmap
    dots = ax.plot(
        eigv.real, eigv.imag, '.',
        markersize=2, markeredgecolor='none'
    )

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(
        "Figures/11-%.6f-%d-%d-%d.png" % (aspect, shape[0], shape[1], shape[0] * shape[1]),
        dpi=600
    )

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ### plot strip GBZ ###
    if which_strip_GBZ == 1:
        bound, loop = HN.get_11_DGBZ_spectrum(J_X1, J_X2, J_Y1, J_Y2)
        strip_GBZ = ax.plot(loop.real, loop.imag, 'r--')
        # ax.legend(dots + strip_GBZ, ["OBC", "(1,1) Direction"])
    elif which_strip_GBZ == 2:
        bound, loop = HN.get_x_DGBZ_spectrum(J_X1, J_X2, J_Y1, J_Y2)
        strip_GBZ = ax.plot(loop.real, loop.imag, 'r--')
        # ax.legend(dots + strip_GBZ, ["OBC", "y "])
    dots[0].remove()
    fig.savefig(
        "Figures/11-%.6f-%d-%d-%d.pdf" % (aspect, shape[0], shape[1], shape[0] * shape[1]),
    )

    fd.save_csv(
        "Figures/11-%.6f-%d-%d-%d" % (aspect, shape[0], shape[1], shape[0] * shape[1]),
        "Fig. 3(f) | OBC spectrum of the 2D HN model in the [11]-parallelogram with "
        "(L1, L2) = (%d, %d) in the cell basis of the [11] direction; Re(E), Im(E)"
        % (shape[0], shape[1]),
        Re_E=eigv.real, Im_E=eigv.imag,
    )


def plot_11_OBC_spectrum_marked_SGBZ(
        aspect: float,
        N: int,
        which_strip_GBZ=0    # 0: None, 1: (1,1), 2: xy
):
    ''' Given the aspect and site number, plot the spectrum and save '''
    # Only this routine needs shapely; importing it lazily keeps the figure
    # routines (plot_OBC_with_given_aspect / plot_11_OBC_spectrum) down to
    # NumPy + SciPy + Matplotlib + BerryPy.
    import shapely

    ### load data ###
    with open("%s/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (
        DATA_DIR, aspect, N
    ), "rb") as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    Jx1, Jx2, Jy1, Jy2 = params
    boundary_list = np.array([
        4 * (2 * np.sqrt(Jx1 * Jx2 * Jy1 * Jy2) 
             +Jx1 * Jx2 + Jy1 * Jy2),
        4 * (-2 * np.sqrt(Jx1 * Jx2 * Jy1 * Jy2) 
             +Jx1 * Jx2 + Jy1 * Jy2),
        0
    ], dtype=complex)
    boundary_matrix = np.column_stack(
        [boundary_list.real, boundary_list.imag]
    )
    boundary_loop = np.vstack([boundary_matrix, boundary_matrix[0,:]])
    polygon_obj = shapely.Polygon(boundary_matrix)
    ### create figure ###
    fig = plt.figure(figsize=(4 * fs.cm, 4 * fs.cm))
    ax = fig.gca()
    ax.set_xlabel("ReE")
    ax.set_ylabel("ImE")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-4, 4])

    ### plot spectrum and save as bitmap
    dots = ax.plot(
        eigv.real, eigv.imag, '.',
        markersize=2, markeredgecolor='none'
    )

    SGBZ_eigv = []
    in_sgbz = np.zeros(len(eigv), dtype=int)
    for j in range(len(eigv)):
        if(polygon_obj.contains(shapely.Point([(eigv[j]**2).real, (eigv[j]**2).imag]))):
            SGBZ_eigv.append(eigv[j])
            in_sgbz[j] = 1
    SGBZ_eigv = np.asarray(SGBZ_eigv)
    ax.plot(
        SGBZ_eigv.real, SGBZ_eigv.imag, '.',
        markersize=2, markeredgecolor='none'
    )

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.savefig(
        "Figures/11-SGBZ-marked-%.6f-%d-%d-%d.png" % (aspect, shape[0], shape[1], shape[0] * shape[1]),
        dpi=600
    )

    fd.save_csv(
        "Figures/11-SGBZ-marked-%.6f-%d-%d-%d" % (aspect, shape[0], shape[1], shape[0] * shape[1]),
        "Fig. 3 | OBC spectrum of the [11]-parallelogram with (L1, L2) = (%d, %d); "
        "in_sgbz flags the eigenvalues whose E^2 falls inside the [11]-SGBZ region"
        % (shape[0], shape[1]),
        Re_E=eigv.real, Im_E=eigv.imag, in_sgbz=in_sgbz,
    )


if __name__ == '__main__':
    # pass
    plot_OBC_with_given_aspect(1/8, 40)
    plot_OBC_with_given_aspect(1, 40)
    plot_OBC_with_given_aspect(8, 40)
    plot_11_OBC_spectrum(1/8, 40, 1)
    plot_11_OBC_spectrum(1, 40, 0)
    plot_11_OBC_spectrum(8, 40, 2)
    # plot_11_OBC_spectrum_marked_SGBZ(1/8, 40, 1)
