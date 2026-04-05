'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-25 15:56:42
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''
import numpy as np
import poly_tools as pt
import time
import BerryPy.TightBinding as tb
from math import sin, cos, sqrt, pi
from cmath import log, exp
import brute_force_solver as bfs
import matplotlib.pyplot as plt
from scipy import optimize
import sys
sys.path.append("../paper/HN-model")
import HN_model_common as HN


def test_SGBZ():
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0, 0)],
        np.array([
            [1, 0],
            [1, 1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    poly = pt.CLaurent(3)
    poly.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))

    mu1_0 = np.log(abs(Jx1 / Jx2)) / 2
    solver = bfs.SGBZSolver(poly)

    for E_ref in [1 + 1j, 2 + 2j, 3 + 3j]:
        t1 = time.time()
        mu1, PMGBZ_points = solver.solve_for_E(E_ref)
        print(mu1)
        t2 = time.time()
        if len(PMGBZ_points) == 0:
            # w0, _, gap_info = bfs.get_strip_winding(
            #     solver.winding_fun,
            #     E_ref,
            #     mu1,
            #     with_gap_info=True
            # )
            # print(w0)
            # print(gap_info)

            def gap_fun(test_mu1: float):
                w0, _, gap_info = bfs.get_strip_winding(
                    solver.winding_fun,
                    E_ref,
                    test_mu1,
                    with_gap_info=True
                )
                if gap_info is None:
                    return -w0 * np.inf
                else:
                    return gap_info["dmu2_diff"]
            
            if gap_fun(mu1) > 0:
                new_mu1 = optimize.brentq(gap_fun, mu1, mu1 + 1)
            else:
                new_mu1 = optimize.brentq(gap_fun, mu1 - 1, mu1)
            print(new_mu1)
            w0, _, gap_info = bfs.get_strip_winding(
                solver.winding_fun,
                E_ref,
                new_mu1,
                with_gap_info=True
            )
            print(w0)
            print(gap_info)


def show_sols():
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0, 0)],
        np.array([
            [1, 0],
            [1, 1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    poly = pt.CLaurent(3)
    poly.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))

    mu1_0 = np.log(abs(Jx1 * Jy1 / Jx2 / Jy2)) / 2
    mu2_0 = np.log(abs(Jy1 / Jy2)) / 2
    solver = bfs.SGBZSolver(poly)

    E_ref = 1 + 1j
    mu1, PMGBZ_points = solver.solve_for_E(E_ref)
    print(mu1, mu1_0)

    PMGBZ_points, theta1_arr, sols_arr, info = bfs.get_roots_and_PMGBZ(
        poly, E_ref, mu1
    )
    M = info['M']

    plt.plot(theta1_arr, np.log(np.abs(sols_arr[:, M-1])))
    plt.plot(theta1_arr, np.log(np.abs(sols_arr[:, M])))
    plt.plot(theta1_arr, mu2_0 * np.ones_like(theta1_arr), '--')
    plt.savefig("data/test.png")



if __name__ == '__main__':
    # test_SGBZ()
    show_sols()