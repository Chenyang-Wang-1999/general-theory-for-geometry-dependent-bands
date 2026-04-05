'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-18 17:53:46
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import BerryPy.TightBinding as tb
import numpy as np
from math import sin, cos, sqrt, pi
from cmath import exp, log
import poly_tools as pt
from brute_force_solver.strip_winding_number import get_diabolic_root


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
        [1, 1, v2 * exp(1j * phi), (1, 0)],
        [1, 1, v2 * exp(1j * phi), (0, 1)],
        [1, 1, v2 * exp(1j * phi), (-1,-1)]
    ]
    return tb.TightBindingModel(dim, site_num, lattice_vec, intra_cell, inter_cell)


def Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma):
    # v1 = v2 = t2 * exp(i gamma)
    return non_Hermitian_Haldane_H(t1, t1, t2 * exp(1j * gamma), t2 * exp(1j * gamma), phi, M)


def main():
    t1, t2, phi, M, gamma = np.random.randn(5)
    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))
    df = [char_poly.derivative(i) for i in [1, 2]]
    get_diabolic_root(
        char_poly,
        df[0],
        df[1],
        df[0].derivative(2),
        df[1].derivative(2),
        np.random.randn() + 1j * np.random.randn(),
        [0,0]
    )

if __name__ == '__main__':
    main()