'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-01-13
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from math import sin, cos, sqrt, pi
from cmath import exp, log
import numpy as np
from BerryPy import TightBinding as tb
import matplotlib.pyplot as plt
import GBZ_2d_tools as gbz
import c_solver_2D as csolver
pt = csolver
import pickle
from scipy import linalg as la
import brute_force_solver as bfs
import multiprocessing as mp


def non_Hermitian_Haldane_H(u1, u2, v1, v2, phi, M):
    dim = 2
    site_num = 2

    lattice_vec = np.array(
        [[-cos(pi/3), -cos(pi/3)],
         [-sin(pi/3), sin(pi/3)]]
    )

    intra_cell = [
        [0, 0, M],
        [1, 1, -M],
        [1, 0, u1],
        [0, 1, u2]
    ]
    inter_cell = [
        [0, 0, v2 * exp(1j*phi), (-1,0)],
        [0, 0, v2 * exp(1j*phi), (0, -1)],
        [0, 0, v2 * exp(1j*phi), (1,1)],
        [0, 0, v1 * exp(-1j*phi), (1,0)],
        [0, 0, v1 * exp(-1j*phi), (0,1)],
        [0, 0, v1 * exp(-1j*phi), (-1,-1)],
        [1, 0, u1, (0, -1)],
        [1, 0, u1, (1, 0)],
        [0, 1, u2, (0, 1)],
        [0, 1, u2, (-1, 0)],
        [1, 1, v2 * exp(-1j * phi), (-1, 0)],
        [1, 1, v2 * exp(-1j*phi), (0, -1)],
        [1, 1, v2 * exp(-1j*phi), (1, 1)],
        [1, 1, v1 * exp(1j * phi), (1, 0)],
        [1, 1, v1 * exp(1j * phi), (0, 1)],
        [1, 1, v1 * exp(1j * phi), (-1,-1)]
    ]

    site_coord_cart = np.array(
        [[0, 1 / (2 * sqrt(3))],
         [0, - 1 / (2 * sqrt(3))]]
    )

    model = tb.TightBindingModel(dim, site_num, lattice_vec, intra_cell, inter_cell)
    model.SiteCoord = model.cart2lattice(site_coord_cart.T).T

    return model


def Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma):
    # v1 = v2 = t2 * exp(i gamma)
    return non_Hermitian_Haldane_H(t1, t1, t2 * exp(1j * gamma), t2 * exp(1j * gamma), phi, M)


def check_geometry():
    t1, t2, phi, M, gamma = np.random.randn(5)
    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    Nx = 50
    Ny = 50
    model_1d = model.get_supercell(
        [(0,j) for j in range(Ny)],
        np.array([
            [1, 0],
            [0, Ny]
        ], dtype=int)
    )
    model_2d = model_1d.get_supercell(
        [(j, 0) for j in range(Nx)],
        np.array([
            [Nx, 0],
            [0, 1]
        ], dtype=int)
    )

    coord_cart = model_2d.lattice2cart(model_2d.SiteCoord.T).T
    plt.plot(
        coord_cart[::2, 0], 
        coord_cart[::2, 1],
        '.'
    )
    plt.plot(
        coord_cart[1::2, 0], 
        coord_cart[1::2, 1],
        '.'
    )
    plt.axis("equal")
    plt.show()


def calculate_Haldane_SGBZ():
    t1 = 1
    t2 = 0.5
    phi = pi / 3
    M = 0
    gamma = pi / 12

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = csolver.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        csolver.CScalarVec(coeffs),
        csolver.CLaurentIndexVec(degs.flatten())
    )
    gbz.sweep_2D_GBZ_x_major(
        "data/paper-Haldane-nH-phase",
        char_poly,
        np.linspace(np.pi / 51, np.pi * (1 - 1 / 51), 50),
        np.linspace(np.pi / 51, np.pi * (1 - 1 / 51), 50),
        0,
        100
    )


def calculate_Haldane_SGBZ_a2():
    t1 = 1
    t2 = 0.5
    phi = pi / 3
    M = 0
    gamma = pi / 12

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [0, 1],
            [1, 0]
        ], dtype=int)
    )

    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = csolver.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        csolver.CScalarVec(coeffs),
        csolver.CLaurentIndexVec(degs.flatten())
    )
    gbz.sweep_2D_GBZ_x_major(
        "data/paper-Haldane-nH-phase-a2",
        char_poly,
        np.linspace(np.pi / 51, np.pi * (1 - 1 / 51), 50),
        np.linspace(np.pi / 51, np.pi * (1 - 1 / 51), 50),
        0,
        100
    )


def plot_Haldane_SGBZ():
    # fig1 = plt.figure()
    # fig2 = plt.figure()
    # ax1 = fig1.gca()
    # ax2 = fig2.gca()
    for file_ind in range(1, 50):
        try:
            with open("data/paper-Haldane-nH-phase/x_major_%d.pkl" % file_ind, "rb") as fp:
                data = pickle.load(fp)
            for seg in data:
                plt.plot((seg[:,0]).real, (seg[:,0]).imag, 'b.')
                # ax1.plot((seg[:,0]).real, np.log(seg[:,1]).real, 'b.')
                # ax2.plot((seg[:,0]).real, np.log(seg[:,2]).real, 'b.')
        except:
            pass


def plot_BZ_bands():
    t1 = 1
    t2 = 0.5
    phi = pi / 3
    M = 0
    gamma = pi / 12

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
 
    k1 = np.linspace(-np.pi, np.pi, 100)
    k2 = np.linspace(-np.pi, np.pi, 100)
    k1_grid, k2_grid = np.meshgrid(k1, k2)
    E1_grid = np.zeros_like(k1_grid, dtype=complex)
    E2_grid = np.zeros_like(k1_grid, dtype=complex)
    for i in range(k1_grid.shape[0]):
        for j in range(k1_grid.shape[1]):
            H = model.get_bulk_Hamiltonian((k1_grid[i, j], k2_grid[i, j])).todense()

            E1_grid[i, j], E2_grid[i, j] = la.eig(H)[0]

    plt.plot(E1_grid.real, E1_grid.imag, 'b.')
    plt.plot(E2_grid.real, E2_grid.imag, 'b.')

def calculate_Haldane_OBC():
    t1 = 1
    t2 = 0.5
    phi = pi / 3
    M = 0
    gamma = pi / 6

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    
    Nx = 80
    Ny = 80
    model_1d = model.get_supercell(
        [(0,j) for j in range(Ny)],
        np.array([
            [1, 0],
            [0, Ny]
        ], dtype=int)
    )
    model_2d = model_1d.get_supercell(
        [(j, 0) for j in range(Nx)],
        np.array([
            [Nx, 0],
            [0, 1]
        ], dtype=int)
    )

    coord = model_2d.lattice2cart(model_2d.SiteCoord.T).T

    H_obc = model_2d.get_bulk_Hamiltonian_complex((None, None)).todense()
    eigv, eigvec = la.eig(H_obc)
    with open("data/paper-Haldane-nH-phase-pi_6-OBC-%d-%d.pkl" % (Nx, Ny), "wb") as fp:
        pickle.dump((eigv, eigvec, coord), fp)


def calculate_Haldane_OBC_square():
    t1 = 1
    t2 = 0.5
    phi = pi / 3
    M = 0
    gamma = pi / 6

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    model_square_unit = model.get_supercell(
        [(0, 0), (1, 0)],
        np.array([
            [1, 1],
            [1, -1]
        ], dtype=int)
    )
    
    Nx = 10
    Ny = 80
    model_1d = model_square_unit.get_supercell(
        [(0,j) for j in range(Ny)],
        np.array([
            [1, 0],
            [0, Ny]
        ], dtype=int)
    )
    model_2d = model_1d.get_supercell(
        [(j, 0) for j in range(Nx)],
        np.array([
            [Nx, 0],
            [0, 1]
        ], dtype=int)
    )

    coord = model_2d.lattice2cart(model_2d.SiteCoord.T).T

    H_obc = model_2d.get_bulk_Hamiltonian_complex((None, None)).todense()
    eigv, eigvec = la.eig(H_obc)
    with open("data/paper-Haldane-nH-phase-pi_6-OBC-square-%d-%d.pkl" % (Nx, Ny), "wb") as fp:
        pickle.dump((eigv, eigvec, coord), fp)


def plot_Haldane_OBC():
    with open("data/paper-Haldane-nH-phase-OBC-80-80.pkl", "rb") as fp:
        eigv, eigvec, coord = pickle.load(fp)
    plt.plot(eigv.real, eigv.imag, 'r.')
    # # normalization
    # eigvec = eigvec / np.linalg.norm(eigvec, axis=0)
    # eigvec_mean = np.mean(np.abs(eigvec), axis=1)
    # plt.scatter(coord[:, 0], coord[:, 1], c=eigvec_mean)
    # plt.axis("equal")
    # plt.show()


if __name__ == "__main__":
    # check_geometry()
    # calculate_Haldane_SGBZ()
    # calculate_Haldane_SGBZ_a2()
    # plot_Haldane_SGBZ()
    plot_BZ_bands()
    plot_Haldane_OBC()
    plt.show()
    # calculate_Haldane_OBC()
    # calculate_Haldane_OBC_square()
    # plot_Haldane_OBC()
