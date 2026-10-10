'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-01-15
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from math import sin, cos, sqrt, pi
from cmath import exp, log
import numpy as np
from BerryPy import TightBinding as tb
import matplotlib.pyplot as plt
import poly_tools as pt
import brute_force_solver as bfs
import brute_force_amoeba as bfa
import pickle
from scipy import linalg as la
from typing import Literal
import multiprocessing as mp
import os
from scipy import interpolate

# The color palette is shared by all the plotting scripts of this repository,
# and lives in colors.py at its root.
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from colors import COLORS

ALL_PARAMS = (
    1, #t1,
    0.5, #t2,
    pi / 3, # phi
    0.5j, # M
    0 # gamma
)

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

def calculate_Haldane_SGBZ():
    t1, t2, phi, M, gamma = ALL_PARAMS

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = csolver.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        csolver.CScalarVec(coeffs),
        csolver.CLaurentIndexVec(degs.flatten())
    )
    gbz.sweep_2D_GBZ_x_major(
        "data/paper-Haldane-gain-loss-a1",
        char_poly,
        np.linspace(np.pi / 51, np.pi * (1 - 1 / 51), 50),
        np.linspace(np.pi / 51, np.pi * (1 - 1 / 51), 50),
        0,
        100
    )

def plot_BZ_bands():
    t1, t2, phi, M, gamma = ALL_PARAMS

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

    plt.plot(E1_grid.real, E1_grid.imag, 'bx')
    plt.plot(E2_grid.real, E2_grid.imag, 'bx')
    plt.axis("equal")

def calculate_Haldane_OBC(Nx, Ny):
    t1, t2, phi, M, gamma = ALL_PARAMS

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    
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
    with open("data/paper-Haldane-gain-loss-OBC-%d-%d.pkl" % (Nx, Ny), "wb") as fp:
        pickle.dump((eigv, eigvec, coord), fp)

def calculate_Haldane_OBC_square():
    t1, t2, phi, M, gamma = ALL_PARAMS

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    model_square_unit = model.get_supercell(
        [(0, 0), (1, 0)],
        np.array([
            [1, 1],
            [1, -1]
        ], dtype=int)
    )
    
    Nx = 80
    Ny = 40
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
    with open("data/paper-Haldane-gain-loss-OBC-square-%d-%d.pkl" % (Nx, Ny), "wb") as fp:
        pickle.dump((eigv, eigvec, coord), fp)

def calculate_Haldane_OBC_rhombus_y(Nx, Ny):
    t1, t2, phi, M, gamma = ALL_PARAMS

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    model_square_unit = model.get_supercell(
        [(0, 0), (1, 0), (0, -1)],
        np.array([
            [2, -1],
            [1, -2]
        ], dtype=int)
    )
    
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
    with open("data/paper-Haldane-gain-loss-OBC-rhombus-y-%d-%d.pkl" % (Nx, Ny), "wb") as fp:
        pickle.dump((eigv, eigvec, coord), fp)

def plot_Haldane_OBC(Nx, Ny, which: Literal["", "-square"]=""):
    with open("data/paper-Haldane-gain-loss-OBC%s-%d-%d.pkl" % (which, Nx, Ny), "rb") as fp:
        eigv, eigvec, coord = pickle.load(fp)
    
    print(len(eigv))
    plt.plot(eigv.real, eigv.imag, '.')
    plt.title("OBC%s" % (which))

def plot_Haldane_eigenstates(Nx, Ny, which: Literal["", "-square"]=""):
    with open("data/paper-Haldane-gain-loss-OBC%s-%d-%d.pkl" % (which, Nx, Ny), "rb") as fp:
        eigv, eigvec, coord = pickle.load(fp)

    eigv_ind = np.random.randint(0, eigv.shape[0])

    plt.figure()
    plt.plot(eigv.real, eigv.imag, 'b.')
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, 'rx')
    
    plt.figure()
    # eigvec_mean = np.mean(np.abs(eigvec), axis=1)
    eigvec_abs = np.abs(eigvec[:, eigv_ind])
    sca = plt.scatter(coord[:,0], coord[:,1], c=eigvec_abs)
    plt.title("OBC-eigenstates%s" % (which))
    plt.colorbar(sca)
    plt.axis("equal")
    print(coord.shape)

def plot_SGBZ_bands():
    for file_ind in range(0, 50):
        with open("data/paper-Haldane-gain-loss-a1/x_major_%d.pkl" % file_ind, "rb") as fp:
            data = pickle.load(fp)
        print(len(data))
        for seg in data:
            print(len(seg))
            plt.plot((seg[:,0]).real, (seg[:,0]).imag, 'b.')

def check_SGBZ(
    coeffs: np.ndarray,
    degs: np.ndarray,
    E_ref: complex,
    perc: float
):
    print("%.2f" % (perc * 100) + r"%")
    coeffs = pt.CScalarVec(coeffs)
    degs = pt.CLaurentIndexVec(degs.flatten())
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(coeffs, degs)
    solver = bfs.SGBZSolver(char_poly)
    try:
        mu1, PMGBZ_points = solver.solve_for_E(E_ref, N_points=301)
        success = True
    except Exception as e:
        success = False
        return {"success": success, "error": str(e)}

    if PMGBZ_points:
        return {"success": success, "is_PMGBZ": True, "mu1": mu1, "PMGBZ_points": PMGBZ_points}
    else:
        return {"success": success, "is_PMGBZ": False, "mu1": mu1}

def sweep_amoeba():
    model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    coeffs, degs = model.get_characteristic_polynomial_data()

    N_points = 101
    E_real = np.linspace(-3.1, 4.6, N_points)
    E_imag = np.linspace(-0.51, 0.51, N_points)
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_mesh = E_real_mesh + 1j * E_imag_mesh
    E_list = E_mesh.flatten()

    pool = mp.Pool(10)
    results = pool.starmap(bfa.check_amoeba, [(coeffs, degs, E, j / len(E_list), True) for j, E in enumerate(E_list)])
    pool.close()
    pool.join()

    fname_prefix = "data/Haldane-gain-loss-amoeba"
    file_id = 0
    # while os.path.exists(fname_prefix + "-%d.pkl" % file_id):
    #     file_id += 1

    with open(fname_prefix + "-%d.pkl" % file_id, "wb") as fp:
        pickle.dump({
            "E_real": E_real,
            "E_imag": E_imag,
            "results": results,
            "params": ALL_PARAMS,
            "coeffs": coeffs,
            "degs": degs
        }, fp)

def sweep_SGBZ_a1():
    model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    coeffs, degs = model.get_characteristic_polynomial_data()

    N_points = 41
    E_real = np.linspace(-3.1, 4.6, N_points)
    E_imag = np.linspace(-0.51, 0.51, N_points)
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_mesh = E_real_mesh + 1j * E_imag_mesh
    E_list = E_mesh.flatten()

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(bfs.check_SGBZ, [(coeffs, degs, E, j / len(E_list), True) for j, E in enumerate(E_list)])
    pool.close()
    pool.join()

    fname_prefix = "data/debug-Haldane-gain-loss-a1-SGBZ"
    file_id = 0
    # while os.path.exists(fname_prefix + "-%d.pkl" % file_id):
    #     file_id += 1

    with open(fname_prefix + "-%d.pkl" % file_id, "wb") as fp:
        pickle.dump({
            "E_real": E_real,
            "E_imag": E_imag,
            "results": results,
            "params": ALL_PARAMS,
            "coeffs": coeffs,
            "degs": degs
        }, fp)

def sweep_SGBZ_a2():
    model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [0, 1],
            [1, 0]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()

    N_points = 101
    E_real = np.linspace(-3.1, 4.6, N_points)
    E_imag = np.linspace(-0.51, 0.51, N_points)
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_mesh = E_real_mesh + 1j * E_imag_mesh
    E_list = E_mesh.flatten()

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(check_SGBZ, [(coeffs, degs, E, j / len(E_list)) for j, E in enumerate(E_list)])
    pool.close()
    pool.join()

    fname_prefix = "data/Haldane-gain-loss-a2-SGBZ"
    file_id = 0
    while os.path.exists(fname_prefix + "-%d.pkl" % file_id):
        file_id += 1

    with open(fname_prefix + "-%d.pkl" % file_id, "wb") as fp:
        pickle.dump({
            "E_real": E_real,
            "E_imag": E_imag,
            "results": results,
            "params": ALL_PARAMS,
            "coeffs": coeffs,
            "degs": degs
        }, fp)

def sweep_SGBZ_x():
    model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    model = model.get_supercell(
        [(0, 0), (1, 0)],
        np.array([
            [1, 1],
            [1, -1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()

    N_points = 101
    E_real = np.linspace(-3.1, 4.6, N_points)
    E_imag = np.linspace(-0.51, 0.51, N_points)
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_mesh = E_real_mesh + 1j * E_imag_mesh
    E_list = E_mesh.flatten()

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(check_SGBZ, [(coeffs, degs, E, j / len(E_list)) for j, E in enumerate(E_list)])
    pool.close()
    pool.join()

    fname_prefix = "data/Haldane-gain-loss-x-SGBZ"
    file_id = 0
    while os.path.exists(fname_prefix + "-%d.pkl" % file_id):
        file_id += 1

    with open(fname_prefix + "-%d.pkl" % file_id, "wb") as fp:
        pickle.dump({
            "E_real": E_real,
            "E_imag": E_imag,
            "results": results,
            "params": ALL_PARAMS,
            "coeffs": coeffs,
            "degs": degs
        }, fp)

def sweep_SGBZ_y():
    model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    model = model.get_supercell(
        [(0, 0), (1, 0)],
        np.array([
            [1, 1],
            [-1, 1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()

    N_points = 201
    E_real = np.linspace(-3.1, 4.6, N_points)
    E_imag = np.linspace(-0.51, 0.51, N_points)
    # E_imag = np.array([0])
    # E_real = np.linspace(-3.1, 4.6, 5 * N_points)
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_mesh = E_real_mesh + 1j * E_imag_mesh
    E_list = E_mesh.flatten()

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(bfs.check_SGBZ, [(coeffs, degs, E, j / len(E_list)) for j, E in enumerate(E_list)])
    pool.close()
    pool.join()

    fname_prefix = "data/new-Haldane-gain-loss-y-SGBZ"
    file_id = 0
    # while os.path.exists(fname_prefix + "-%d.pkl" % file_id):
    #     file_id += 1

    with open(fname_prefix + "-%d.pkl" % file_id, "wb") as fp:
        pickle.dump({
            "E_real": E_real,
            "E_imag": E_imag,
            "results": results,
            "params": ALL_PARAMS,
            "coeffs": coeffs,
            "degs": degs,
        }, fp)

def plot_amoebic_spectrum():
    with open("data/Haldane-gain-loss-amoeba-0.pkl", "rb") as fp:
        data = pickle.load(fp)
    E_real, E_imag, results, params = data["E_real"], data["E_imag"], data["results"], data["params"]
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_list = (E_real_mesh + 1j * E_imag_mesh).flatten()
    coeffs, degs = data["coeffs"], data["degs"]

    success_mesh = np.array([res["success"] for res in results]).reshape(E_real_mesh.shape)
    failed_mesh = np.array([not res["success"] for res in results]).reshape(E_real_mesh.shape)
    success_mesh = np.array([res["success"] and res["is_amoeba"] for res in results]).reshape(E_real_mesh.shape)
    print(E_real_mesh[failed_mesh] + 1j * E_imag_mesh[failed_mesh])
    plt.plot(E_real_mesh[success_mesh], E_imag_mesh[success_mesh], '.', label="Amoebic")
    plt.plot(E_real_mesh[~success_mesh & ~failed_mesh], E_imag_mesh[~success_mesh & ~failed_mesh], '.', label="Non-amoebic")
    plt.plot(E_real_mesh[failed_mesh], E_imag_mesh[failed_mesh], '.', label="Failed")
    plt.legend()
    plt.show()

    # if which == "a1":
    #     model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    #     coeffs, degs = model.get_characteristic_polynomial_data()

    # else:
    #     coeffs, degs = data["coeffs"], data["degs"]

    # E_k_list = bfs.convert_results_to_triplet(results, E_list, coeffs, degs)
    # E_k_list = np.array(E_k_list)
    # plt.plot(E_k_list[:,0].real, E_k_list[:,1].imag, ".")
    # plt.plot(E_k_list[:,0].real, E_k_list[:,2].imag, ".")
    # plt.show()

def plot_SGBZ(which="a1"):
    with open("data/new-Haldane-gain-loss-%s-SGBZ-0.pkl" % which, "rb") as fp:
        data = pickle.load(fp)
    E_real, E_imag, results, params = data["E_real"], data["E_imag"], data["results"], data["params"]
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_list = (E_real_mesh + 1j * E_imag_mesh).flatten()
    coeffs, degs = data["coeffs"], data["degs"]

    # success_mesh = np.array([res["success"] for res in results]).reshape(E_real_mesh.shape)
    # failed_mesh = np.array([not res["success"] for res in results]).reshape(E_real_mesh.shape)
    # success_mesh = np.array([res["success"] and res["is_PMGBZ"] for res in results]).reshape(E_real_mesh.shape)
    # print(E_real_mesh[failed_mesh] + 1j * E_imag_mesh[failed_mesh])
    # plt.plot(E_real_mesh[success_mesh], E_imag_mesh[success_mesh], '.', label="SGBZ")
    # plt.plot(E_real_mesh[~success_mesh & ~failed_mesh], E_imag_mesh[~success_mesh & ~failed_mesh], '.', label="Non-SGBZ")
    # plt.plot(E_real_mesh[failed_mesh], E_imag_mesh[failed_mesh], '.', label="Failed")
    # plt.legend()
    # plt.show()

    # if which == "a1":
    #     model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    #     coeffs, degs = model.get_characteristic_polynomial_data()

    # else:
    #     coeffs, degs = data["coeffs"], data["degs"]

    E_k_list = bfs.convert_results_to_triplet(results, E_list, coeffs, degs)
    E_k_list = np.array(E_k_list)
    plt.plot(E_k_list[:,0].real, E_k_list[:,1].imag, ".")
    plt.plot(E_k_list[:,0].real, E_k_list[:,2].imag, ".")
    plt.show()

def plot_SGBZ_mu(which="a1"):
    with open("data/Haldane-gain-loss-%s-SGBZ-0.pkl" % which, "rb") as fp:
        data = pickle.load(fp)
    E_real, E_imag, results, params = data["E_real"], data["E_imag"], data["results"], data["params"]
    all_mu1 = []

    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_list = (E_real_mesh + 1j * E_imag_mesh).flatten()
    E_SGBZ = []
    for ind, res in enumerate(results):
        if res["success"]:
            if res["is_PMGBZ"]:
                all_mu1.append(res["mu1"])
                E_SGBZ.append(E_list[ind])
    E_SGBZ = np.asarray(E_SGBZ)
    sca = plt.scatter(E_SGBZ.real, E_SGBZ.imag, c=all_mu1)
    plt.colorbar(sca)

def debug_SGBZ():
    E_ref = -1.56 + 0.01j
    model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    coeffs, degs = model.get_characteristic_polynomial_data()
    print(bfs.check_SGBZ(coeffs, degs, E_ref, 1))
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs), 
        pt.CLaurentIndexVec(degs.flatten())
    )
    poly_diff = bfs.PolyDiffContext(char_poly)
    # print(bfs.get_strip_winding(poly_diff, E_ref, 0.06447462687012076))
    # mu_list = np.linspace(-0.1, 0.1, 31)
    # w_list = np.zeros_like(mu_list)
    # for ind in range(len(mu_list)):
    #     w_list[ind] = bfs.get_strip_winding(poly_diff, E_ref, mu_list[ind])[0]
    # plt.plot(mu_list, w_list)
    # plt.show()

    for mu1 in [0.05]:
        PMGBZ_points, theta1_arr, sols_arr, info = bfs.get_roots_and_PMGBZ(poly_diff, E_ref, mu1)
        M = info["M"]
        y_range = [np.min(np.log(np.abs(sols_arr))), np.max(np.log(np.abs(sols_arr)))]
        for p in PMGBZ_points:
            print(p)
            if p["is_continuum"]:
                plt.fill_betweenx(
                    y_range, 
                    [p["theta1_start"], p["theta1_start"]], 
                    [p["theta1_end"], p["theta1_end"]],
                    color=COLORS["of"])
            else:
                plt.plot([p["theta1"], p["theta1"]], y_range, color=COLORS["ol"])
        plt.plot(theta1_arr, np.log(np.abs(sols_arr)), ".-", color=COLORS["bl"])
        plt.show()
        mu2_max_arr = np.log(np.abs(sols_arr[:, M]))
        mu2_min_arr = np.log(np.abs(sols_arr[:, M - 1]))
        mu2_arr = (mu2_max_arr + mu2_min_arr) / 2
        # plt.plot(theta1_arr, mu2_arr, color=COLORS["gl"])
        mu2_fun = interpolate.CubicSpline(theta1_arr, mu2_arr, bc_type="periodic")

        plt.clf()
        theta2_samples = np.linspace(-pi, pi, 21)
        for p in PMGBZ_points:
            curr_theta1 = np.log(np.exp(1j * p["theta1"])).imag
            pos_sols, neg_sols, zero_sols = p["beta2_sols"]
            for beta_p in pos_sols:
                plt.plot(curr_theta1, np.log(beta_p).imag, 'r+')
            for beta_n in neg_sols:
                plt.plot(curr_theta1, np.log(beta_n).imag, 'bx')
            for beta_z in zero_sols:
                plt.plot(curr_theta1, np.log(beta_z).imag, 'ko')

        for theta2 in theta2_samples:
            plt.plot([-pi, pi], [theta2, theta2], '-')
            w0 = bfs.get_loop_winding(poly_diff, E_ref, mu1, mu2_fun, theta2)
            plt.text(np.random.rand() * 1.8 * pi - 0.9 * pi, theta2 + 0.1, f"{w0:.2f}")

        plt.show()

def plot_BZ_bands():
    model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
 
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

def debug_SGBZ_calculation():
    model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    model = model.get_supercell(
        [(0, 0), (1, 0)],
        np.array([
            [1, 1],
            [1, -1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()

    # with open("data/Haldane-gain-loss-y-SGBZ-1.pkl", "rb") as fp:
    #     data = pickle.load(fp)
    # E_real, E_imag, results, params = data["E_real"], data["E_imag"], data["results"], data["params"]
    # for j in range(len(results)):
    #     if not results[j]["success"]:
    #         E = E_real[j]
    #         print(E)
    #         break
    print(check_SGBZ(coeffs, degs, 3.136 - 0.02j, 0))

   
def plot_SGBZ_mu_major_and_minor(which="a1"):
    with open("data/Haldane-gain-loss-%s-SGBZ-0.pkl" % which, "rb") as fp:
        data = pickle.load(fp)
    E_real, E_imag, results, params = data["E_real"], data["E_imag"], data["results"], data["params"]
    all_mu1 = []

    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_list = (E_real_mesh + 1j * E_imag_mesh).flatten()
    mu1_list = []
    mu2_list = []
    SGBZ_E_list = []
    for ind, res in enumerate(results):
        if res["success"]:
            if res["is_PMGBZ"]:
                mu1 = res["mu1"]
                SGBZ_points = res["PMGBZ_points"]
                curr_mu2 = []
                for p in SGBZ_points:
                    pos_sols, neg_sols, zero_sols = p["beta2_sols"]
                    beta2_all = np.asarray(list(pos_sols) + list(neg_sols) + list(zero_sols), dtype=complex)
                    curr_mu2.append(np.log(np.abs(beta2_all)))
                curr_mu1 = [mu1 * np.ones_like(m) for m in curr_mu2]
                mu1_list.extend(curr_mu1)
                mu2_list.extend(curr_mu2)
                SGBZ_E_list.extend([E_list[ind] * np.ones_like(m) for m in curr_mu2])
    # E_SGBZ = np.asarray(E_SGBZ)
    # sca = plt.scatter(E_SGBZ.real, E_SGBZ.imag, c=all_mu1)
    # plt.colorbar(sca)
    mu1_list = np.concatenate(mu1_list)
    mu2_list = np.concatenate(mu2_list)
    SGBZ_E_list = np.concatenate(SGBZ_E_list)
    # order = np.argsort(mu1_list)
    non_zero_mask = abs(mu1_list) > 1e-6
    # plt.plot(mu1_list[order], '.')
    # plt.plot(mu2_list[order], '.')
    plt.plot(SGBZ_E_list[non_zero_mask].real, SGBZ_E_list[non_zero_mask].imag, '.')
    plt.plot(SGBZ_E_list[~non_zero_mask].real, SGBZ_E_list[~non_zero_mask].imag, '.')

def calculate_DOS(eigv: np.ndarray, real_grids: np.ndarray, imag_grids: np.ndarray):
    ''' Indexing: ij '''
    point_num = np.zeros((real_grids.shape[0] - 1, imag_grids.shape[0] - 1))
    for row_ind in range(real_grids.shape[0] - 1):
        for col_ind in range(imag_grids.shape[0] - 1):
            point_num[row_ind, col_ind] = np.sum(
                (eigv.real >= real_grids[row_ind]) & 
                (eigv.real < real_grids[row_ind + 1]) &
                (eigv.imag >= imag_grids[col_ind]) & 
                (eigv.imag < imag_grids[col_ind + 1]))
    return point_num

def run_DOS_calculation(n1, n2, N_real_grids=201, N_imag_grids=201, which=""):
    with open(f"data/paper-Haldane-gain-loss-OBC{which}-{n1}-{n2}.pkl", "rb") as fp:
        eigv, _, _ = pickle.load(fp)
    real_grids = np.linspace(-3.4, 4.7, N_real_grids)
    imag_grids = np.linspace(-0.53, 0.53, N_imag_grids)
    point_num = calculate_DOS(eigv, real_grids, imag_grids)
    with open(f"data/paper-Haldane-DOS-OBC{which}-{n1}-{n2}-{N_real_grids}-{N_imag_grids}.pkl", "wb") as fp:
        pickle.dump((point_num, real_grids, imag_grids), fp)

if __name__ == "__main__":
    # run_DOS_calculation(120, 120)
    # run_DOS_calculation(80, 80)
    # run_DOS_calculation(80, 40, which="-square")
    # run_DOS_calculation(120, 60, which="-square")
    # calculate_Haldane_SGBZ()
    # calculate_Haldane_OBC()
    # calculate_Haldane_OBC_square()
    # plot_SGBZ_bands()
    # plt.show()
    # sweep_SGBZ_a1()
    # sweep_SGBZ_x()
    # sweep_SGBZ_a2()
    # sweep_SGBZ_y()
    # plt.figure()
    # plot_SGBZ("a1")
    # plot_Haldane_OBC(120, 120)
    # plot_Haldane_eigenstates(120, 120)
    # plt.figure()
    # plot_Haldane_OBC(120, 60, "-square")
    # plot_Haldane_eigenstates(80, 40, "-square")

    # debug_SGBZ()
    # plt.figure()
    # plt.title("a1")
    # plt.figure()
    # plot_SGBZ("y")
    # plt.title("a2")
    # plt.figure()
    # plot_SGBZ("y")
    # plt.title("y")
    # plt.figure()
    # plot_SGBZ("a1")
    # plt.title("x")
    # plot_BZ_bands()
    # plot_SGBZ_mu("a1")
    # plt.title("a1")
    # plt.figure()
    # plot_SGBZ_mu("a2")
    # plt.title("a2")
    # plt.figure()
    # plot_SGBZ_mu("a1")
    # plt.title("y")
    # debug_SGBZ_calculation()
    # plt.show()

    # plot_SGBZ_mu_major_and_minor("a1")
    # calculate_Haldane_OBC(80, 80)
    # calculate_Haldane_OBC_rhombus_y(40, 40)
    # plot_Haldane_OBC(40, 40, "-rhombus-y")
    # plt.show()
    # sweep_amoeba()
    plot_amoebic_spectrum()
