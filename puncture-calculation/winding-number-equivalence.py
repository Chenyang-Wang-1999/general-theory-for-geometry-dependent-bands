'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-12-23
Copyright © Department of Physics, Tsinghua University. All rights reserved

Check the equivalence of the winding number

'''

import numpy as np
import partial_GBZ_solver as pGs
import GBZ_manifold as Gm
import poly_tools as pt
import BerryPy.TightBinding as tb
import BerryPy as bp
from math import sin, cos, sqrt, pi
from cmath import exp
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
from scipy import linalg as la

# The color palette is shared by all the plotting scripts of this repository,
# and lives in colors.py at its root.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from colors import COLORS

def SSH_2D(ux1, ux2, vx1, vx2, uy1, uy2, vy1, vy2):
    dim = 2
    site_num = 2
    lat_vec = np.array([
        [1, 0],
        [0, 1]
    ])
    intra_cell = [
        [1, 0, ux1],
        [0, 1, ux2],
    ]
    inter_cell = [
        [0, 1, vx1, (1, 0)],
        [1, 0, vx2, (-1, 0)],
        [0, 0, uy1, (0, 1)],
        [0, 0, uy2, (0, -1)],
        [1, 1, vy1, (0, 1)],
        [1, 1, vy2, (0, -1)]
    ]

    return tb.TightBindingModel(
        dim, site_num, lat_vec, intra_cell, inter_cell
    )

def SSH_quasi1D(ux1, ux2, vx1, vx2, uy1, uy2, vy1, vy2, betax) -> tb.TightBindingModel:
    dim = 1
    site_num = 2
    lat_vec = [[1]]
    intra_cell = [
        [1, 0, ux1 + vx2 * betax],
        [0, 1, ux2 + vx1 * betax]
    ]
    inter_cell = [
        [0, 0, uy1, (1,)],
        [0, 0, uy2, (-1,)],
        [1, 1, vy1, (1,)],
        [1, 1, vy2, (-1,)]
    ]
    return tb.TightBindingModel(
        dim, site_num, lat_vec, intra_cell, inter_cell
    )

def NH_Haldane(t1, t2, phi, m, gamma):
    dim = 2
    site_num = 2
    lat_vec = np.array([
        [1, cos(pi/3)],
        [0, sin(pi/3)]
    ])
    intra_cell = [
        [0, 0, m],
        [1, 1, -m],
        [0, 1, t1],
        [1, 0, t1]
    ]
    inter_cell = [
        [1, 0, t1, (0, 1)],
        [0, 1, t1, (0, -1)],
        [1, 0, t1, (-1, 1)],
        [0, 1, t1, (1, -1)],

        [0, 0, t2 * exp(1j * (phi + gamma)), (-1, 0)],
        [0, 0, t2 * exp(1j * (phi + gamma)), (0, 1)],
        [0, 0, t2 * exp(1j * (phi + gamma)), (1, -1)],
        [0, 0, t2 * exp(-1j * (phi - gamma)), (1, 0)],
        [0, 0, t2 * exp(-1j * (phi - gamma)), (0, -1)],
        [0, 0, t2 * exp(-1j * (phi - gamma)), (-1, 1)],

        [1, 1, t2 * exp(1j * (phi + gamma)), (1, 0)],
        [1, 1, t2 * exp(1j * (phi + gamma)), (0, -1)],
        [1, 1, t2 * exp(1j * (phi + gamma)), (-1, 1)],
        [1, 1, t2 * exp(-1j * (phi - gamma)), (-1, 0)],
        [1, 1, t2 * exp(-1j * (phi - gamma)), (0, 1)],
        [1, 1, t2 * exp(-1j * (phi - gamma)), (1, -1)],
    ]

    return tb.TightBindingModel(
        dim, site_num, lat_vec, intra_cell, inter_cell
    )

def run_all_Cherns(argin):
    t1 = 1
    m = argin[0]
    t2 = argin[1]
    N_Phi = argin[2]
    phi = np.linspace(-pi, pi, N_Phi)
    C_Grid = np.zeros_like(phi)
    for j in range(N_Phi):
        H = NH_Haldane(t1, t2, phi[j], m, 0)
        ms = bp.MatrixSolver(H.get_bulk_Hamiltonian_dense,
                             True,
                             2,
                             hermitian=True)
        bs = bp.BerrySolver(ms, "OpticalCouplingHaldaneModel_t2%.3f" % (t2),
                            no_time=True)

        N_grid = 20
        k2, k1 = np.meshgrid(np.linspace(-0.5, 0.5, N_grid), np.linspace(-0.5, 0.5, N_grid))
        ParamsGrid = np.zeros((2, N_grid, N_grid))
        ParamsGrid[0, :, :] = k1
        ParamsGrid[1, :, :] = k2
        Berry = bs.Curvature(ParamsGrid, 0)[0]
        C_Grid[j] = Berry.sum() / (2 * pi)
        print('phi = %.3fpi, M/t2 = %.3f, C = %.3f' % (phi[j] / pi, m / t2,
                                                       C_Grid[j]))
    return C_Grid

def check_Haldane_model():
    N_M = 50
    N_Phi = 50
    n_process = 6
    t2 = 1/3
    M = np.linspace(-3*sqrt(3),3*sqrt(3),N_M)*t2
    Phi_Grid, M_Grid = np.meshgrid(np.linspace(-pi,pi,N_Phi),M)
    parainputs = [(m,t2,N_Phi) for m in M]
    pool = Pool(processes=n_process)
    C_list = pool.map(run_all_Cherns,parainputs)
    C_Grid = np.vstack(C_list)
    pcl = plt.pcolor(Phi_Grid / pi, M_Grid, C_Grid,cmap='seismic_r',
                     vmin=-np.max(np.abs(C_Grid)),vmax=np.max(np.abs(C_Grid)))
    plt.xlabel(r'$\phi (\pi)$')
    plt.ylabel(r'$M/t_2$')
    plt.colorbar(pcl)
    plt.show()

def check_SSH_2D_ChP():
    # model_params = np.random.randn(8)
    model_params = [0.5002, -0.1031, -2.1329, -0.1353,  1.0413,  1.6011, -1.8877,  0.3436]
    print(model_params)
    model = SSH_2D(*model_params)
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    solver = pGs.GBZMainSolver(char_poly)
    # E_ref = np.random.randn() + 1j * np.random.randn()
    E_ref = -0.6784521839887232+0.306284263371469j
    print(E_ref)
    err_code = solver.get_beta_GBZ(E_ref)
    if err_code != pGs.GBZSolverErrorCode.SOLVER_SUCCESS:
        raise Exception('GBZ solver failed! Error code:', err_code)

    beta_GBZ = solver.beta_GBZ 
    all_holes = solver.GBZ_holes
    print(all_holes)
    for j, curr_hole in enumerate(all_holes[1:-1]):
        beta_norm = np.sqrt(
            (1 + curr_hole) / (1 - curr_hole)
        )

        k1_arr = np.concatenate([
            np.linspace(-0.06, 0.06, 51) * np.pi,
            np.linspace(0.06, 2 - 0.06, 51) * np.pi
        ])
        beta1_arr = beta_norm * np.exp(1j * k1_arr)

        mGBZ_torus = pGs.calculate_minor_GBZ_no_sort(
            char_poly, beta1_arr, 101, glue_tol=0.1, distance_tol=0.1
        )
        with open("data/SSH-2D-mGBZ-hole_%d.pkl" % (j), "wb") as fp:
            pickle.dump((beta1_arr, mGBZ_torus), fp)

def plot_SSH_2D_GBZ_torus():
    E_ref = -0.6784521839887232+0.306284263371469j
    with open("data/SSH-2D-mGBZ-hole_%d.pkl" % (0), "rb") as fp:
        beta1_arr, mGBZ_torus = pickle.load(fp)
    
    print(np.max(np.abs(beta1_arr)), np.min(np.abs(beta1_arr)))
    color_list = [COLORS['bl'], COLORS['ol'], COLORS['gl'], COLORS['rl'], 
                  COLORS['pl'], COLORS['brl'], COLORS['cl'], COLORS['ml'], COLORS['yl']]
    # plt.plot(E_ref.real, E_ref.imag, 'x')

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.set_position([0.1, 0.25, 0.4, 0.7])
    ax2 = fig.add_subplot(1, 2, 2)

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
        ax2.cla()
        ax.set_title("%.2f pi" % (np.log(beta1_arr[mGBZ_ind]).imag / np.pi))
        mGBZ_loops = mGBZ_torus[mGBZ_ind]
        for band_ind in range(len(mGBZ_loops)):
            E_list = np.zeros(len(mGBZ_loops[band_ind].point_vec), dtype=complex)
            beta2_list = np.zeros(len(mGBZ_loops[band_ind].point_vec), dtype=complex)
            for point_ind, point in enumerate(mGBZ_loops[band_ind].point_vec):
                Gm.GBZ_to_chart(point, [0])
                E_list[point_ind] = point.coords[0]
                beta2_list[point_ind] = point.coords[1]
            ax2.plot(E_list.real, E_list.imag, color=color_list[band_ind])
            # plt.plot(beta2_list.real, beta2_list.imag, color=color_list[band_ind])
            # ax.text(band_ind, 0, "%.0f" % (pGs.get_winding_number(E_list - E_ref)), color=color_list[band_ind])
            ax.plot(beta2_list.real, beta2_list.imag, E_list.real, color=color_list[band_ind])
        fig.canvas.draw_idle()

    scroll_slider.on_changed(update)

    plt.show()

def get_SSH_quasi_1D_GBZ():
    model_params = [0.5002, -0.1031, -2.1329, -0.1353,  1.0413,  1.6011, -1.8877,  0.3436]
    betax = 1.340745534377115
    model = SSH_quasi1D(*model_params, betax)

    # calculate GBZ
    _, E, beta = model.get_GBZ(N_process=6)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(beta.real, beta.imag, E.real, '.')
    
    N = 50
    model_OBC = model.get_supercell(
        [(j,) for j in range(N)],
        [[N]]
    )
    
    fig2 = plt.figure()
    ax = fig2.gca()
    H = model_OBC.get_bulk_Hamiltonian_complex((None,)).todense()
    eigv, eigvec = la.eig(H)
    ax.plot(eigv.real, eigv.imag, '.')

    fig3 = plt.figure()
    ax = fig3.gca()
    ax.plot(E.real, E.imag, '.')

    plt.show()

if __name__ == '__main__':
    # check_Haldane_model()
    # check_SSH_2D_ChP()
    # plot_SSH_2D_GBZ_torus()
    get_SSH_quasi_1D_GBZ()