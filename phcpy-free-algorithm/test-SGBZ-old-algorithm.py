'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-28 16:15:57
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import GBZ_2d_tools as gbz
import c_solver_2D as csolver
import numpy as np
import BerryPy.TightBinding as tb
import sys
import matplotlib.pyplot as plt
import brute_force_solver as bfs
from cmath import exp, log
import pickle
sys.path.append("../paper/HN-model")
import HN_model_common as HN

def test_main():
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    # model = HN.get_HN_model(Jx1.real, Jx2.real, Jy1.real, Jy2.real)
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0, 0)],
        np.array([
            [1, 0],
            [1, 1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = csolver.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        csolver.CScalarVec(coeffs),
        csolver.CLaurentIndexVec(degs.flatten())
    )
    gbz.sweep_2D_GBZ_x_major(
        "data/test-HN-2",
        char_poly,
        np.linspace(0, np.pi, 50),
        np.linspace(0, np.pi, 50),
        0,
        100
    )


def test_continuum_solutions():
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    model = HN.get_HN_model(Jx1.real, Jx2.real, Jy1.real, Jy2.real)
    # model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0, 0)],
        np.array([
            [1, 0],
            [1, 1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = csolver.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        csolver.CScalarVec(coeffs),
        csolver.CLaurentIndexVec(degs.flatten())
    )

    E_ref = 0.05
    mu1 = np.log(abs(Jx1 * Jy1 / Jx2 / Jy2)) / 2
    theta1_list = np.linspace(0, 2 * np.pi, 101)
    beta1 = np.exp(mu1 + 1j * theta1_list)
    E = E_ref * np.ones_like(beta1, dtype=complex)
    param_list = np.column_stack((E, beta1))

    # 2. solve the points
    event_call, sols = bfs.brute_force_roots(
        char_poly,
        (0, 1),
        param_list,
        2,
        1, 1
    )

    plt.plot(theta1_list, np.abs(sols))
    plt.savefig("data/test_4.png")


def compare_with_numerical_solutions():
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    model = HN.get_HN_model(Jx1.real, Jx2.real, Jy1.real, Jy2.real)
    # model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0, 0)],
        np.array([
            [1, 0],
            [1, 1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    N = 40
    model_q1D = model.get_supercell(
        [(0, j) for j in range(N)],
        np.array([
            [1, 0],
            [0, N]
        ], dtype=int)
    )
    model_q1D = model_q1D.truncate(1)

    char_poly = csolver.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        csolver.CScalarVec(coeffs),
        csolver.CLaurentIndexVec(degs.flatten())
    )

    E_ref = 0.05
    mu1_list = np.linspace(-1, 1, 101)
    w_num = np.zeros_like(mu1_list)
    w_theo = np.zeros_like(mu1_list)

    for mu1_ind, mu1 in enumerate(mu1_list):
        param_fun = lambda t: ((exp(mu1 + 1j * t),), (1j * exp(mu1 + 1j * t),))
        winding_fun = bfs.MatWindingFun(
            model_q1D.get_bulk_Hamiltonian_complex,
            model_q1D.get_bulk_Hamiltonian_diff,
            param_fun,
            (0, 2 * np.pi),
            E_ref
        )
        w_num[mu1_ind] = bfs.get_winding_number(winding_fun)
        w_theo[mu1_ind] = bfs.get_strip_winding(
            bfs.WindingFun(char_poly, None, None),
            E_ref,
            mu1,
            N_points = 301,
            continuum_perturb=0.1,
            zero_tol=1e-10
        )[0]
        print(w_num[mu1_ind], w_theo[mu1_ind])

        perc = (mu1_ind / len(mu1_list) * 100)
        print(" " * 100, end="\r")
        print(f"Progress: {perc: .2f}%", end="\r")
    
    plt.plot(mu1_list, w_num / N)
    plt.plot(mu1_list, w_theo, '--')
    plt.savefig("data/test_5_%d.png" % N)


def plot_SGBZ():
    plt.figure()
    for file_ind in range(1, 50):
        try:
            with open("data/test-HN-2/x_major_%d.pkl" % file_ind, "rb") as fp:
                data = pickle.load(fp)
            for seg in data:
                plt.plot(seg[:,0].real, seg[:,0].imag, 'b.')
        except:
            pass
    plt.savefig("data/test_6.png")



if __name__ == "__main__":
    # test_main()
    # test_continuum_solutions()
    # compare_with_numerical_solutions()
    plot_SGBZ()
