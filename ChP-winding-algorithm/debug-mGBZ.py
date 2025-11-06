'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-03-29
Copyright © Department of Physics, Tsinghua University. All rights reserved

Debug the mGBZ calculations
'''

import numpy as np
from math import pi
from numpy import pi
import pyvista as pv
import poly_tools as pt
import winding_GBZ_solver as wGs
import sys
import matplotlib.pyplot as plt
sys.path.append("../paper/HN-model")
import HN_model_common as HN


def test_left_and_right_winding_numbers():
    ''' Test the left and right limit of $W_{strip}(E, r)$ at r '''
    alpha = 3
    Jx, Jy = [(-0.59477-1.09826j), (1.18704-0.07472j)]
    gamma_x, gamma_y = [1.97069, 0.52332]
    # delta = 1.79515
    delta = 0
    glue_tol = 0.05
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

    kx = pi / 2
    ky = pi / 3

    beta_x = gamma_x * np.exp(1j * kx)
    beta_y = gamma_y * np.exp(1j * ky)
    E_ref = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y

    char_poly = chp.partial_eval(
        pt.CScalarVec([E_ref]), pt.CIndexVec([0]), pt.CIndexVec([1,2])
    )

    # Solve non-Bloch loops
    GBZ_loops, phi_loop, aGBZ_loops = wGs.solve_GBZ(
        "CP1", char_poly, 200, glue_tol)
    GBZ_loops, phi_loop, aGBZ_loops = wGs.expand_solution(
        GBZ_loops, aGBZ_loops, phi_loop)
    all_segments = wGs.get_GBZ_segments(GBZ_loops, 1.1 * glue_tol)
    all_segments = wGs.remove_duplicated_points(all_segments, 0.5 * glue_tol)
    # all_segments = wGs.convert_to_complex_numbers(all_segments)
    all_loops = wGs.GBZ_segments_to_loops(all_segments, 5 * glue_tol)
    all_loops = wGs.convert_to_complex_numbers(all_loops)
    # print(all_segments)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-pi, pi])
    for loop in all_loops:
        loop_arr = np.asarray(loop)
        ax.plot(loop_arr[:,0].real, loop_arr[:,0].imag, np.log(loop_arr[:,1]).imag, '.-')
        plt.pause(1)
    plt.show()

if __name__ == "__main__":
    test_left_and_right_winding_numbers()
