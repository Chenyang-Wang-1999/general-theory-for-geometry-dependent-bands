'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-03-28
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from math import pi
import poly_tools as pt
import winding_GBZ_solver as wGs
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle
import pyvista as pv
import os
import sys
sys.path.append("../paper/HN-model")
import HN_model_common as HN


def get_ChP(gammax, gammay, Jx, Jy, alpha):
    coeffs = pt.CScalarVec(
        [1, -gammax * Jx, 
         -Jx.conjugate() / gammax, 
         -gammay * Jy, 
         -Jy.conjugate() / gammay]
    )
    degs = pt.CLaurentIndexVec([
        # E, beta1, beta2
        1, 0, 0,
        0, -1, alpha,
        0, 1, -alpha,
        0, 0, -1,
        0, 0, 1
    ])
    chp = pt.CLaurent(3)
    chp.set_Laurent_by_terms(coeffs, degs)
    return chp


def get_left_and_right_winding_number(
    chp: pt.CLaurent,
    E: complex,
    r: float,
    n_theta1: int = 100,
    glue_tol: float = 1e-2,
    min_dr: float = 1e-2,
    max_dr: float = 5e-2,
    n_points: int = 5,
    safe_mode: bool = True
):
    ''' Get the left and right limit of $W_{strip}(E, r)$ at r '''
    solver = wGs.StripGBZMainSolver(chp)
    err_code = solver.solve_non_Bloch_solutions(E, glue_tol=glue_tol)
    if err_code != 0 and safe_mode:
        raise ValueError("Error code %d when solving non-Bloch solutions" % err_code)

    deltar_list = r * np.linspace(min_dr, max_dr, n_points)

    # Get left limit
    w_list = np.zeros_like(deltar_list, dtype=float) * np.nan
    for i, deltar in enumerate(deltar_list):
        err_code = solver.get_total_winding(r - deltar, n_theta1)
        if err_code == 0:
            w_list[i] = solver.total_winding
        elif safe_mode:
            # print("Warning: error code %d when calculating total winding number" % err_code)
            raise ValueError("Error code %d when calculating total winding number" % err_code)
    # exclude the nan values
    w_idx = ~np.isnan(w_list)
    w_list = w_list[w_idx]
    deltar_list = deltar_list[w_idx]
    # fit a line
    coeffs = np.polyfit(deltar_list, w_list, 1)
    w_left = np.polyval(coeffs, 0)

    # Get right limit
    w_list = np.zeros_like(deltar_list, dtype=float) * np.nan
    for i, deltar in enumerate(deltar_list):
        err_code = solver.get_total_winding(r + deltar, n_theta1)
        if err_code == 0:
            w_list[i] = solver.total_winding
        elif safe_mode:
            # print("Warning: error code %d when calculating total winding number" % err_code)
            raise ValueError("Error code %d when calculating total winding number" % err_code)
    # exclude the nan values
    w_idx = ~np.isnan(w_list)
    w_list = w_list[w_idx]
    deltar_list = deltar_list[w_idx]
    # fit a line
    coeffs = np.polyfit(deltar_list, w_list, 1)
    w_right = np.polyval(coeffs, 0)
    return w_left, w_right


def test_left_and_right_winding_numbers():
    ''' Test the left and right limit of $W_{strip}(E, r)$ at r '''
    alpha = 3
    Jx, Jy = [(-0.59477-1.09826j), (1.18704-0.07472j)]
    gamma_x, gamma_y = [1.97069, 0.52332]
    # delta = 1.79515
    delta = 0
    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, delta, delta)
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1, 0],
            [alpha, 1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    chp = pt.CLaurent(3)
    chp.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    print(chp.to_str(pt.CStrVec(["E", "x", "y"])))
    chp = get_ChP(gamma_x, gamma_y, Jx, Jy, alpha)
    print(chp.to_str(pt.CStrVec(["E", "x", "y"])))

    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, delta, delta)

    kx = pi / 2
    ky = pi / 3

    beta_x = gamma_x * np.exp(1j * kx)
    beta_y = gamma_y * np.exp(1j * ky)
    E = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y

    r_mid = gamma_x * (gamma_y ** alpha)

    solver = wGs.StripGBZMainSolver(chp)
    err_code = solver.solve_non_Bloch_solutions(E, glue_tol=0.05)

    r_list = np.linspace(0.1, 0.3, 500)
    w_list = np.zeros_like(r_list, dtype=float) * np.nan
    for i, r in enumerate(r_list):
        err_code = solver.get_total_winding(r, 100)
        if err_code == 0:
            w_list[i] = solver.total_winding
        else:
            print("Warning: error code %d when calculating total winding number" % err_code)

    w_left, w_right = get_left_and_right_winding_number(chp, E, r_mid)
    print("W_left = %.3f, W_right = %.3f" % (w_left, w_right))

    plt.plot(r_list, w_list, '.', label="W(r)")
    plt.plot(r_list, 0 * r_list + w_left, '--')
    plt.plot(r_list, 0 * r_list + w_right, '--')
    plt.show()


def batch_fun(args: tuple):
    perc, coeffs, degs, params, k = args
    chp = pt.CLaurent(3)
    chp.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    gamma_x, gamma_y, Jx, Jy, delta, alpha = params
    r = gamma_x * (gamma_y ** alpha)
    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, delta, delta)
    kx, ky = k

    print(perc)
    perturb = 0
    for j in range(2):
        beta_x = gamma_x * np.exp(1j * kx)
        beta_y = gamma_y * np.exp(1j * ky)
        E_ref = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y 
        try:
            w_left, w_right = get_left_and_right_winding_number(chp, E_ref, r, glue_tol=2e-2, min_dr=5e-2, max_dr=8e-2, safe_mode=False)
            return kx, ky, w_left, w_right
        except Exception as e:
            print(os.getpid(), "Trial:",j, e, file=sys.stderr)
            kx += np.random.randn() * 1e-7
            ky += np.random.randn() * 1e-7

    return (kx, ky, None, None)


def sweep_HN_primitive(alpha: int):
    ''' Sweep primitive cells of HN model '''
    # Jx, Jy = np.random.randn(2) + 1j * np.random.randn(2)
    # gamma_x, gamma_y = np.random.rand(2) + 0.5
    # delta = np.random.rand() * 2 * pi
    # print(Jx, Jy)
    # print(gamma_x, gamma_y)
    # print(delta)
    # return 
    Jx, Jy = [(-0.59477-1.09826j), (1.18704-0.07472j)]
    gamma_x, gamma_y = [1.97069, 0.52332]
    delta = 1.79515
    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, delta, delta)
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1, 0],
            [alpha, 1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()

    # Generate data
    kx = np.linspace(-pi, pi, 21)
    ky = np.linspace(-pi, pi, 21)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    kx_data = kx_mesh.flatten()
    ky_data = ky_mesh.flatten()

    data_pack = []
    for idx in range(len(kx_data)):
        perc = "alpha %d: %.2f%%" % (alpha, 100 * idx / len(kx_data))
        # beta_x = gamma_x * np.exp(1j * kx_data[idx])
        # beta_y = gamma_y * np.exp(1j * ky_data[idx])
        # E_ref = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y 
        # r = gamma_x * (gamma_y ** alpha)
        data_pack.append((perc, coeffs, degs, (gamma_x, gamma_y, Jx, Jy, delta, alpha), (kx_data[idx], ky_data[idx])))
    pool = Pool(11)
    result = pool.map(batch_fun, data_pack)
    with open("data/HN-alpha-%d.pkl" % alpha, "wb") as f:
        pickle.dump(
            (
                (Jx, Jy, gamma_x, gamma_y, delta),
                (kx, ky),
                result
            ),
            f
        )


def visualization(alpha: int):
    ''' Visualization '''
    with open("data/HN-alpha-%d.pkl" % alpha, "rb") as f:
        data = pickle.load(f)
    (Jx, Jy, gamma_x, gamma_y, delta), (kx, ky), result = data
    # print(result)
    _, _, w_left_list, w_right_list = zip(*result)
    w_left_list = np.array(w_left_list)
    w_right_list = np.array(w_right_list)
    w_left_list = w_left_list.reshape(len(kx), len(ky))
    w_right_list = w_right_list.reshape(len(kx), len(ky))
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    left_mesh = pv.StructuredGrid(kx_mesh, ky_mesh, w_left_list)
    right_mesh = pv.StructuredGrid(kx_mesh, ky_mesh, w_right_list)
    zeros = pv.StructuredGrid(kx_mesh, ky_mesh, np.zeros_like(kx_mesh))
    pl = pv.Plotter()
    pl.add_mesh(left_mesh, opacity=0.5, scalars=w_left_list.flatten(), cmap="coolwarm", show_scalar_bar=False) 
    pl.add_mesh(right_mesh, opacity=0.5, scalars=w_right_list.flatten(), cmap="coolwarm", show_scalar_bar=False)
    pl.add_mesh(zeros)
    pl.show_grid()
    pl.show()


def test_fractional():
    ''' Test fractional direction '''

    Jx, Jy = [(-0.59477-1.09826j), (1.18704-0.07472j)]
    gamma_x, gamma_y = [1.97069, 0.52332]
    delta = 1.79515
    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, delta, delta)

    a1 = (3, 2)
    point_list = [
        (j, 
         int(np.ceil(j * a1[1] / a1[0])))
        for j in range(1, a1[0])
    ]
    print(point_list)

    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)] + point_list,
        np.array([
            [a1[0], 0],
            [a1[1], 1]
        ], dtype=int)
    )

    r_mid = (gamma_x ** a1[0]) * (gamma_y ** (a1[1]))
    print("r_mid = ", r_mid)

    coeffs, degs = model.get_characteristic_polynomial_data()
    chp = pt.CLaurent(3)
    chp.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    print(chp.to_str(pt.CStrVec(["E", "x", "y"])))
    kx = pi / 2
    ky = pi / 3
    beta_x = gamma_x * np.exp(1j * kx)
    beta_y = gamma_y * np.exp(1j * ky)
    E = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y

    solver = wGs.StripGBZMainSolver(chp)
    err_code = solver.solve_non_Bloch_solutions(E, glue_tol=0.05)
    print(solver.non_Bloch_holes)
    for j in range(len(solver.non_Bloch_holes)):
        print(solver.get_total_winding_at_hole(j, 300))
    for loop in solver.non_Bloch_sols:
        plt.plot(loop[:,0].real, loop[:,0].imag, '.-')
    plt.show()
    r_list = np.linspace(0.4, 0.8, 500)
    w_list = np.zeros_like(r_list, dtype=float) * np.nan
    for i, r in enumerate(r_list):
        err_code = solver.get_total_winding(r, 300)
        if err_code == 0:
            w_list[i] = solver.total_winding
        else:
            print("Warning: error code %d when calculating total winding number" % err_code)

    plt.plot(r_list, w_list, '.-')
    plt.show()


def find_failed_points(alpha: int):
    ''' Find failed points '''
    with open("data/HN-alpha-%d.pkl" % alpha, "rb") as f:
        data = pickle.load(f)
    (Jx, Jy, gamma_x, gamma_y, delta), (kx, ky), result = data
    # print(result)
    kx_list, ky_list, w_left_list, w_right_list = zip(*result)
    if not os.path.exists("data/failed_list.csv"):
        with open("data/failed_list.csv", "w") as fp:
            fp.write("# alpha, idx\n")

    with open("data/failed_list.csv", "a") as fp:
        for idx in range(len(kx_list)):
            if w_left_list[idx] is None or w_right_list[idx] is None:
                fp.write("%d, %d\n" % (alpha, idx))
            elif w_left_list[idx] > -1e-4 or w_right_list[idx] < 1e-4:
                fp.write("%d, %d\n" % (alpha, idx))
            elif abs(w_left_list[idx]) > 4:
                fp.write("%d, %d\n" % (alpha, idx))



def debug_failed_points():
    alpha = -3
    with open("data/HN-alpha-%d.pkl" % alpha, "rb") as f:
        data = pickle.load(f)
    (Jx, Jy, gamma_x, gamma_y, delta), (kx, ky), result = data
    kx, ky, w_left, w_right = zip(*result)
    idx = w_left.index(None)
    kx = kx[idx]
    ky = ky[idx]
    print(kx, ky)

    Jx, Jy = [(-0.59477-1.09826j), (1.18704-0.07472j)]
    gamma_x, gamma_y = [1.97069, 0.52332]
    delta = 1.79515
    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, delta, delta)
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1, 0],
            [alpha, 1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    chp = pt.CLaurent(3)
    chp.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    print(chp.to_str(pt.CStrVec(["E", "x", "y"])))
    beta_x = gamma_x * np.exp(1j * kx)
    beta_y = gamma_y * np.exp(1j * ky)
    E_ref = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y
    r_mid = gamma_x * (gamma_y ** alpha)
    solver = wGs.StripGBZMainSolver(chp)
    err_code = solver.solve_non_Bloch_solutions(E_ref, glue_tol=0.05)
    print("Error code:", err_code)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for loop in solver.non_Bloch_sols:
        # ax.plot(loop[:,0].real, loop[:,0].imag, np.log(loop[:,1]).imag, '.-')
        ax.plot(loop[:,0].real, loop[:,0].imag, np.abs(loop[:,1]), '.-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_zlabel('z')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    # ax.set_zlim([-pi, pi])
    plt.show()
    # print(solver.non_Bloch_holes)
    # w_left, w_right = get_left_and_right_winding_number(chp, E_ref, r_mid, n_theta1=100, min_dr=5e-2, max_dr=8e-2, n_points=5)
    # print(w_left, w_right)


def find_all_failed():
    os.remove("data/failed_list.csv")
    for alpha in range(-3, 4):
        find_failed_points(alpha)


def recalculate_failed_points(alpha, failed_idx, manual: bool = True):
    # alpha = -3
    # failed_idx = 201
    with open("data/HN-alpha-%d.pkl" % alpha, "rb") as f:
        data = pickle.load(f)
    (Jx, Jy, gamma_x, gamma_y, delta), k_grid, result = data
    kx_list, ky_list, _, _ = zip(*result)
    kx = kx_list[failed_idx]
    ky = ky_list[failed_idx]
    print(kx, ky)
    Jx, Jy = [(-0.59477-1.09826j), (1.18704-0.07472j)]
    gamma_x, gamma_y = [1.97069, 0.52332]
    delta = 1.79515
    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, delta, delta)
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1, 0],
            [alpha, 1]
        ])
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    chp = pt.CLaurent(3)
    chp.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    # 1. Calculate all mGBZ points
    new_glue_tol = [0.01, 0.02, 1e-4, 0.03]
    min_dr = 1.5e-2
    max_dr = 2e-2
    beta_x = gamma_x * np.exp(1j * kx)
    beta_y = gamma_y * np.exp(1j * ky)
    E_ref = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y
    r_mid = gamma_x * (gamma_y ** alpha)
    if manual:
        solver = wGs.StripGBZMainSolver(chp)
        err_code = solver.solve_non_Bloch_solutions(E_ref, glue_tol=new_glue_tol)
        print("Error code:", err_code)

        # plot
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.gca()
        theta = np.linspace(0, 2 * pi, 500)
        ax.plot(
            r_mid * (1 - min_dr) * np.cos(theta),
            r_mid * (1 - min_dr) * np.sin(theta),
            "k--"
        )
        ax.plot(
            r_mid * (1 + min_dr) * np.cos(theta),
            r_mid * (1 + min_dr) * np.sin(theta),
            "k--"
        )
        ax.plot(
            r_mid * (1 + max_dr) * np.cos(theta),
            r_mid * (1 + max_dr) * np.sin(theta),
            "k--"
        )
        ax.plot(
            r_mid * (1 - max_dr) * np.cos(theta),
            r_mid * (1 - max_dr) * np.sin(theta),
            "k--"
        )
        for loop in solver.non_Bloch_sols:
            ax.plot(loop[:,0].real, loop[:,0].imag, '.-')
            # ax.plot(loop[:,0].real, loop[:,0].imag, np.log(loop[:,1]).imag, '.-')
            # ax.plot(loop[:,0].real, loop[:,0].imag, np.abs(loop[:,1]), '.-')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # ax.set_zlabel('z')
        ax.set_xlim([-2 * r_mid, 2 * r_mid])
        ax.set_ylim([-2 * r_mid, 2 * r_mid])
        # ax.set_zlim([-pi, pi])
        plt.show()

        # pl = pv.Plotter()
        # for loop in solver.non_Bloch_sols:
        #     pl.add_lines(np.column_stack(
        #         loop[:,0].real,
        #         loop[:,0].imag,
        #         np.log(loop[:,1]).imag
        #     ))
        #     pl.add_points(np.column_stack(
        #         loop[:,0].real,
        #         loop[:,0].imag,
        #         np.log(loop[:,1]).imag
        #     ))
        # pl.show_grid()
        # pl.show() 

        cmd = input("Continue? (y/n)")
        if cmd != "y":
            return

    # 2. Calculate the left and right limit of $W_{strip}(E, r)$ at r
    w_left, w_right = get_left_and_right_winding_number(
        chp, E_ref, r_mid, n_theta1=100, glue_tol=new_glue_tol,
        min_dr=min_dr, max_dr=max_dr, n_points=5,
        safe_mode=False)
    print(w_left, w_right)

    # 3. Change the failed point to the new value
    if manual:
        cmd = input("Change the failed point to the new value? (y/n)")
        if cmd == "y":
            result[failed_idx] = (kx, ky, w_left, w_right)
            with open("data/HN-alpha-%d.pkl" % alpha, "wb") as f:
                pickle.dump(
                    (
                        (Jx, Jy, gamma_x, gamma_y, delta),
                        k_grid,
                        result
                    ),
                    f
                )
            print("Successfully changed the failed point to the new value")
    else:
        result[failed_idx] = (kx, ky, w_left, w_right)
        with open("data/HN-alpha-%d.pkl" % alpha, "wb") as f:
            pickle.dump(
                (
                    (Jx, Jy, gamma_x, gamma_y, delta),
                    k_grid,
                    result
                ),
                f
            )


def recalculate_all_failed_points():
    # find_all_failed()
    failed_list = np.loadtxt("data/failed_list.csv", delimiter=",", dtype=int)
    for alpha, idx in failed_list:
        recalculate_failed_points(alpha, idx)


def paper_calculate_W_against_r():
    alpha = 2
    Jx, Jy = [(-0.59477-1.09826j), (1.18704-0.07472j)]
    gamma_x, gamma_y = [1.97069, 0.52332]
    delta = 1.79515
    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, delta, delta)
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1, 0],
            [alpha, 1]
        ])
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    chp = pt.CLaurent(3)
    chp.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    # 1. Calculate all mGBZ points
    kx = 0.3 * pi
    ky = 0.3 * pi
    new_glue_tol = [0.01, 0.02, 1e-4, 0.03]
    beta_x = gamma_x * np.exp(1j * kx)
    beta_y = gamma_y * np.exp(1j * ky)
    E_ref = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y
    r_mid = gamma_x * (gamma_y ** alpha)

    solver = wGs.StripGBZMainSolver(chp)
    err_code = solver.solve_non_Bloch_solutions(E_ref, glue_tol=new_glue_tol)
    print("Error code:", err_code)

    # plot these points
    # plot
    # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca()
    # for loop in solver.non_Bloch_sols:
    #     ax.plot(loop[:,0].real, loop[:,0].imag, '.-')
    #     # ax.plot(loop[:,0].real, loop[:,0].imag, np.log(loop[:,1]).imag, '.-')
    #     # ax.plot(loop[:,0].real, loop[:,0].imag, np.abs(loop[:,1]), '.-')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # # ax.set_zlabel('z')
    # ax.set_xlim([-2 * r_mid, 2 * r_mid])
    # ax.set_ylim([-2 * r_mid, 2 * r_mid])
    # # ax.set_zlim([-pi, pi])
    # plt.show()

    r_list = np.linspace(0.01, 3, 500)
    w_list = np.zeros_like(r_list, dtype=float) * np.nan
    for idx, r in enumerate(r_list):
        perc = int(idx / len(r_list) * 100)
        print("#" * perc + "-" * (100 - perc), end='\r')
        err_code = solver.get_total_winding(r, 300)
        if err_code == 0:
            w_list[idx] = solver.total_winding
        else:
            print("Warning: error code %d when calculating total winding number" % err_code)
    plt.plot(r_list, w_list, '.-')
    plt.show()

    with open("data/paper-W_strip-no-GDSE.pkl", "wb") as fp:
        pickle.dump(
            (
                (Jx, Jy, gamma_x, gamma_y, delta),
                (kx, ky),
                E_ref,
                (r_list, w_list),
                solver.non_Bloch_sols
            ),
            fp
        )


if __name__ == "__main__":
    # test_left_and_right_winding_numbers()
    # for alpha in range(-3, 4):
    #     sweep_HN_primitive(alpha)
    # sweep_HN_primitive(-3)
    # visualization(-3)
    # for alpha in range(-3, 4):
    #     visualization(alpha)
    # test_fractional()
    # debug_failed_points()
    # recalculate_failed_points(1, 43)
    # find_all_failed()
    # recalculate_all_failed_points()
    paper_calculate_W_against_r()
