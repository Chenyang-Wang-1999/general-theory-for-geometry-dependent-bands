'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-12-29
Copyright © Department of Physics, Tsinghua University. All rights reserved

Explore the non-trivial braiding group of the sub-GBZs. 
Model from Fu, Y. & Zhang, Y. Braiding topology of non-Hermitian open-boundary bands. Phys. Rev. B 110, L121401 (2024). 
'''

import numpy as np
import partial_GBZ_solver as pGs
import GBZ_manifold as Gm
import poly_tools as pt
import BerryPy.TightBinding as tb
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import linalg as la
import pickle

# The color palette is shared by all the plotting scripts of this repository,
# and lives in colors.py at its root.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from colors import COLORS

def Hbr_model(tp, tm, ta, lam1, lam2, lam3, m):
    dim = 1
    site_num = 2
    lat_vec = np.eye(1)
    intra_cell = [
        [0, 0, lam3],
        [1, 0, -1j * m],
        [0, 1, 1j * m],
        [1, 1, -lam3]
    ]

    inter_cell = [
        [0, 0, tp + lam1, (-1,)],
        [1, 0, -ta, (-1,)],
        [0, 1, ta, (-1,)],
        [1, 1, tp - lam1, (-1,)],

        [0, 0, tm + lam2, (1,)],
        [1, 0, ta, (1,)],
        [0, 1, -ta, (1,)],
        [1, 1, tm - lam2, (1,)]
    ]

    return tb.TightBindingModel(dim, site_num, lat_vec, intra_cell, inter_cell)

def Hbr_model_2D(tx, tp, tm, ta, lam1, lam2, lam3, m):
    dim = 2
    site_num = 2
    lat_vec = np.eye(2)
    intra_cell = [
        [0, 0, lam3],
        [1, 0, -1j * m],
        [0, 1, 1j * m],
        [1, 1, -lam3]
    ]

    inter_cell = [
        [0, 0, tp + lam1, (0, -1)],
        [1, 0, -ta, (0, -1)],
        [0, 1, ta, (0, -1)],
        [1, 1, tp - lam1, (0, -1)],

        [0, 0, tm + lam2, (0, 1)],
        [1, 0, ta, (0, 1)],
        [0, 1, -ta, (0, 1)],
        [1, 1, tm - lam2, (0, 1)],

        [0, 0, tx, (1, 0)],
        [0, 0, tx, (-1, 0)],
        [1, 1, tx, (1, 0)],
        [1, 1, tx, (-1, 0)]
    ]

    return tb.TightBindingModel(dim, site_num, lat_vec, intra_cell, inter_cell)

def replicate_reported_GBZs(lam3: float):
    tp, tm, ta, lam1, lam2, m = 0.15, 0.85, 0.2, 0, 0, 0.1

    model = Hbr_model(tp, tm, ta, lam1, lam2, lam3, m)
    # _, E, beta = model.get_GBZ(N_process=12)
    # plt.figure()
    # plt.plot(E.real, E.imag, '.')
    # plt.figure()
    # plt.plot(beta.real, beta.imag, '.')

    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(2)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    GBZ_loops = pGs.solve_GBZ_1D(char_poly, glue_tol=0.1)

    fig1 = plt.figure()
    ax1 = fig1.gca()
    fig2 = plt.figure()
    ax2 = fig2.gca()
    for curr_loop in GBZ_loops:
        coords = np.zeros((len(curr_loop.point_vec), 2), dtype=complex)
        for point_ind, point in enumerate(curr_loop.point_vec):
            Gm.GBZ_to_chart(point, [0])
            coords[point_ind,:] = point.coords
        ax1.plot(coords[:,0].real, coords[:,0].imag)
        ax2.plot(coords[:,1].real, coords[:,1].imag)
    
    plt.show()

def test_small_ring(lam3: float):
    tp, tm, ta, lam1, lam2, m = 0.15, 0.85, 0.2, 0, 0, 0.1

    model = Hbr_model(tp, tm, ta, lam1, lam2, lam3, m)

    n_cells = int(input("n_cells:"))
    model_1D = model.get_supercell(
        [(j,) for j in range(n_cells)],
        [[n_cells]]
    )
    H = model_1D.get_bulk_Hamiltonian_complex((None,)).todense()
    eigv, eigvec = la.eig(H)
    plt.plot(eigv.real, eigv.imag, '.')

    plt.show()

def calculate_minor_GBZ(betax_norm: float):
    tp, tm, ta, lam1, lam2, m = 0.15, 0.85, 0.2, 0, 0, 0.1
    lam3 = 0.4
    tx = 0.05
    model = Hbr_model_2D(tx, tp, tm, ta, lam1, lam2, lam3, m)
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    # Get winding loop of betax
    betax_loop = betax_norm * np.exp(1j * np.linspace(0, 2 * np.pi, 50))
    GBZ_torus = pGs.calculate_minor_GBZ_no_sort(
        char_poly, betax_loop, 101
    )

    with open("data/non-trivial-braiding/GBZ_torus_lam3%.3f_betax%.3f.pkl" % (lam3, betax_norm), "wb") as f:
        pickle.dump((betax_loop, GBZ_torus), f)

def plot_GBZ_torus():
    lam3 = 0.4
    betax_norm = 1
    fname = "data/non-trivial-braiding/GBZ_torus_lam3%.3f_betax%.3f.pkl" % (lam3, betax_norm)
    with open(fname, "rb") as fp:
        beta1_arr, mGBZ_torus = pickle.load(fp)

    print(np.max(np.abs(beta1_arr)), np.min(np.abs(beta1_arr)))
    color_list = [COLORS['bl'], COLORS['ol'], COLORS['gl'], COLORS['rl'], 
                  COLORS['pl'], COLORS['brl'], COLORS['cl'], COLORS['ml'], COLORS['yl']]
    # plt.plot(E_ref.real, E_ref.imag, 'x')

    fig = plt.figure()
    ax = fig.gca()
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_position([0.1, 0.25, 0.8, 0.7])
    # ax2 = fig.add_subplot(1, 2, 2)

    ax_scroll = plt.axes([0.2, 0.1, 0.65, 0.03])
    scroll_slider = Slider(
        ax=ax_scroll,
        label='',
        valmin=0,
        valmax=len(beta1_arr) - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        mGBZ_ind = int(val)
        ax.cla()
        # ax2.cla()
        ax.set_title("%.2f pi" % (np.log(beta1_arr[mGBZ_ind]).imag / np.pi))
        mGBZ_loops = mGBZ_torus[mGBZ_ind]
        for band_ind in range(len(mGBZ_loops)):
            E_list = np.zeros(len(mGBZ_loops[band_ind].point_vec), dtype=complex)
            beta2_list = np.zeros(len(mGBZ_loops[band_ind].point_vec), dtype=complex)
            for point_ind, point in enumerate(mGBZ_loops[band_ind].point_vec):
                Gm.GBZ_to_chart(point, [0])
                E_list[point_ind] = point.coords[0]
                beta2_list[point_ind] = point.coords[1]
            ax.plot(E_list.real, E_list.imag, color=color_list[band_ind])
            # plt.plot(beta2_list.real, beta2_list.imag, color=color_list[band_ind])
            # ax.text(band_ind, 0, "%.0f" % (pGs.get_winding_number(E_list - E_ref)), color=color_list[band_ind])
            # ax.plot(beta2_list.real, beta2_list.imag, E_list.real, color=color_list[band_ind])
        fig.canvas.draw_idle()

    scroll_slider.on_changed(update)

    plt.show()

if __name__ == '__main__':
    # lam3 = float(input("lambda 3:"))
    # # which_test = int(input("Which test? 1: small ring, 2: replicate reported GBZs:"))
    # which_test = 2
    # if which_test == 1:
    #     test_small_ring(lam3)
    # elif which_test == 2:
    #     replicate_reported_GBZs(lam3)

    # calculate_minor_GBZ(1)
    plot_GBZ_torus()
