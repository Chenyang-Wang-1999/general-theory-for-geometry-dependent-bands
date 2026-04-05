'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-19 16:20:40
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from brute_force_solver.root_solver import brute_force_roots
from brute_force_solver.strip_winding_number import get_roots_and_PMGBZ, get_strip_winding
from brute_force_solver.winding import WindingFun 
from brute_force_solver.SGBZ import SGBZSolver
import poly_tools as pt
import numpy as np
import matplotlib.pyplot as plt
import time
import BerryPy.TightBinding as tb
import sys
sys.path.append("../paper/HN-model")
import HN_model_common as HN


def test_1():
    poly = pt.CLaurent(3)
    coeffs = pt.CScalarVec([1, -1, -1, 1])
    degs = pt.CLaurentIndexVec([
        # z, z1, z2
        8, 0, 0,
        0, 1, 0,
        0, 0, 1,
        -1, 1, 1
    ])
    poly.set_Laurent_by_terms(coeffs, degs)

    r1 = 1
    r2 = 1
    delta = 1e-6
    c1 = 0
    c2 = (r1 + r2 + delta) * np.exp(3j)
    z1_list = c1 + r1 * np.exp(1j * np.linspace(0, 2*np.pi, 100))
    z2_list = c2 + r2 * np.exp(1j * np.linspace(0, 2*np.pi, 100))

    param_list = np.column_stack((z1_list, z2_list))

    # t1 = time.time()
    # continuation_solver(
    #     poly,
    #     (1, 2),
    #     param_list,
    #     0,
    #     np.array([1, 1]),
    # )

    # t2 = time.time()
    # print(f'continuation solver time: {t2 - t1}')
        
    t1 = time.time()
    flag, all_sols = brute_force_roots(
        poly,
        (1, 2),
        param_list,
        0,
        1, 8
    )
    t2 = time.time()
    print(f'brute force solver time: {t2 - t1}')

    for j in range(9):
        plt.plot(all_sols[:,j].real, all_sols[:,j].imag)
    plt.savefig('data/test_1.png')


def test_2():
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    coeffs, degs = model.get_characteristic_polynomial_data()
    poly = pt.CLaurent(3)
    poly.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))

    mu1 = np.log(abs(Jx1 / Jx2))
    E_ref = 0
    theta1 = np.linspace(0, 2*np.pi, 100)
    param_list = np.column_stack((np.ones_like(theta1) * E_ref, np.exp(mu1 + 1j * theta1)))

    PMGBZ_points, theta1_arr, sols_arr = get_roots_and_PMGBZ(
        poly,
        E_ref,
        mu1
    )

    plt.plot(theta1_arr, np.abs(sols_arr))
    for res, beta_list in PMGBZ_points:
        plt.plot([res, res], [0, 3], '--')
    plt.savefig('data/test_2.png')
    plt.clf()

    plt.plot(sols_arr.real, sols_arr.imag)
    for res, beta_list in PMGBZ_points:
        print(res, beta_list)
        for beta_p, beta_n in beta_list:
            plt.plot(beta_p.real, beta_p.imag, 'rx')
            plt.plot(beta_n.real, beta_n.imag, 'bx')
            theta = np.linspace(0, 2*np.pi, 100)
            cir = beta_p * np.exp(1j * theta)
            plt.plot(cir.real, cir.imag, '--')

    plt.savefig('data/test_2.1.png')


def test_3():
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    coeffs, degs = model.get_characteristic_polynomial_data()
    poly = pt.CLaurent(3)
    poly.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))

    mu1_0 = np.log(abs(Jx1 / Jx2)) / 2
    E_ref = 0
    mu1_list = np.linspace(-2, 2, 100)
    W_list = np.zeros_like(mu1_list)
    winding_fun = WindingFun(poly, None, (0, 2 * np.pi))
    for mu1_ind, mu1 in enumerate(mu1_list):
        W_list[mu1_ind], _ = get_strip_winding(winding_fun, E_ref, mu1)
        print(mu1_ind / len(mu1_list))

    plt.plot(mu1_list, W_list)
    plt.plot([mu1_0, mu1_0], [np.min(W_list), np.max(W_list)], '--')
    plt.plot([-2, 2], [0, 0], '--')
    plt.savefig('data/test_3.png')


def test_4():
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    model = HN.get_HN_model(Jx1.real, Jx2.real, Jy1.real, Jy2.real)
    coeffs, degs = model.get_characteristic_polynomial_data()
    poly = pt.CLaurent(3)
    poly.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))

    solver = SGBZSolver(poly)
    E_list = np.linspace(-10, 10, 100)
    len_SGBZ_list = []
    t1 = time.time()
    for E_ref in E_list:
        mu1_0, SGBZ_points = solver.solve_for_E(E_ref, (-1, 1))
        len_SGBZ_list.append(len(SGBZ_points))
        print(E_ref, mu1_0, len(SGBZ_points))
    t2 = time.time()
    print(f'SGBZ solver time: {t2 - t1}')
    # print(mu1_0, SGBZ_points)
    # print(np.log(abs(Jx1 / Jx2)) / 2)


if __name__ == '__main__':
    # test_1()
    # test_2()
    # test_3()
    test_4()
