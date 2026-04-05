'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-18 15:49:49
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import poly_tools as pt
from math import sin, cos, sqrt, pi
from cmath import log, exp
from brute_force_solver.winding import WindingFun, get_winding_number
import brute_force_solver as bfs


def test_winding_fun():
    coeffs = pt.CScalarVec([
        1, 2
    ])
    degs = pt.CLaurentIndexVec([
        2, 0
    ])

    char_poly = pt.CLaurent(1)
    char_poly.set_Laurent_by_terms(coeffs, degs)

    def loop_fun(t: float):
        return [(exp(2 * pi * 1j * t),), (2 * pi * 1j * exp(2 * pi * 1j * t),)]

    winding_fun = WindingFun(char_poly, loop_fun, (0, 1))

    res = get_winding_number(winding_fun)
    print(res)


def test_dmu2():
    n_terms = 10
    coeffs = pt.CScalarVec(np.random.randn(n_terms) + 1j * np.random.randn(n_terms))
    degs = pt.CLaurentIndexVec(np.random.randint(-6, 6, n_terms * 3))

    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(coeffs, degs)

    E_ref = np.random.randn() + 1j * np.random.randn()
    mu1 = np.random.rand() + 0.5
    theta1 = np.random.rand() * 2 * pi

    dmu1 = 1e-3 * np.random.randn()
    dtheta1 = 1e-3 * np.random.randn()
    mu1_2 = mu1 + dmu1
    theta1_2 = theta1 + dtheta1

    def solve_roots(mu1: float, theta1: float):
        poly_1d = char_poly.partial_eval(
                    pt.CScalarVec([E_ref, exp(mu1 + 1j * theta1)]),
                    pt.CIndexVec([0, 1]),
                    pt.CIndexVec([2])
                )
        coeffs = pt.CScalarVec([])
        degs = pt.CIndexVec([])
        poly_1d.num.batch_get_data(coeffs, degs)
        np_coeffs = bfs.poly_to_np_coefficients(coeffs, degs)
        curr_roots = sorted(np.roots(np_coeffs), key=lambda x: abs(x))
        return curr_roots
    # calculate numerical difference
    beta2_1 = solve_roots(mu1, theta1)
    beta2_2 = solve_roots(mu1_2, theta1_2)
    dmu2_num = (np.log(np.abs(beta2_2)) - np.log(np.abs(beta2_1)))
    diff_list = np.zeros((len(beta2_1), 2), dtype=float)
    winding_fun = bfs.WindingFun(char_poly, None, None)
    for ind in range(len(beta2_1)):
        var = (E_ref, exp(mu1 + 1j * theta1), beta2_1[ind])
        diff_list[ind] = winding_fun.eval_dmu2(var)
    
    dmu2_theory = diff_list[:, 0] * dmu1 + diff_list[:, 1] * dtheta1

    for ind in range(len(beta2_1)):
        print(dmu2_num[ind], dmu2_theory[ind])


if __name__ == '__main__':
    # test_winding_fun()
    test_dmu2()
