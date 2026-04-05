'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-12-01 14:59:12
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import brute_force_solver as bfs
import c_solver_2D as csolver
import numpy as np

import sys
sys.path.append("../paper/HN-model")
import HN_model_common as HN

def test_1():
    model = HN.get_HN_model(*HN.Default_Model_Params)
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
    M = char_poly.denom_orders[2]

    E_ref = np.random.randn() + 1j * np.random.randn()
    mu1 = np.random.randn()

    theta1 = np.random.rand() * 2 * np.pi
    dtheta1 = np.random.randn() * 1e-3

    poly_1d = char_poly.partial_eval(
        csolver.CScalarVec([E_ref, np.exp(mu1 + 1j * theta1)]),
        csolver.CIndexVec([0, 1]),
        csolver.CIndexVec([2])
    )
    coeffs = csolver.CScalarVec([])
    degs = csolver.CIndexVec([])
    poly_1d.num.batch_get_data(coeffs, degs)
    np_coeffs = bfs.poly_to_np_coefficients(coeffs, degs)
    sols = np.roots(np_coeffs)
    searching_fun = bfs.PMGBZSearchingFun(
        char_poly, E_ref, mu1, sols, M, True
    )

    val, dval = searching_fun(theta1)
    val2, _ = searching_fun(theta1 + dtheta1)

    print(val2 - val)
    print(dval * dtheta1)
    print(abs(val2 - val - dval * dtheta1) / abs(dtheta1))


if __name__ == "__main__":
    test_1()
