'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-01-05
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import matplotlib.pyplot as plt
import poly_tools as pt
import winding_GBZ_solver as wGs


def plot_ellipse_abs():
    A1 = 2
    B1 = 1
    theta1 = np.linspace(0, 2 * np.pi, 100)
    h1 = A1 * np.cos(theta1) + B1 * 1j * np.sin(theta1)

    A2 = 1.8
    B2 = 1.2
    h2 = A2 * np.cos(theta1) + B2 * 1j * np.sin(theta1)

    plt.plot(theta1, np.abs(h1))
    plt.plot(theta1, np.abs(h2))
    plt.show()


def get_model_ChP(A1, A2, B1, B2):
    C1 = (A1 + B1) / 2
    D1 = (A1 - B1) / 2
    C2 = (A2 + B2) / 2
    D2 = (A2 - B2) / 2

    coeffs = pt.CScalarVec([
        1, -1, C2, D2, C1, D1,
        -C1 * C2, -D1 * C2, -C1 * D2, -D1 * D2,
    ])
    degs = pt.CLaurentIndexVec([
        # E, beta1, beta2
        1, 0, 0,
        0, 0, 1,
        0, 1, 0,
        0, -1, 0,
        0, 1, 0,
        0, -1, 0,
        0, 2, -1,
        0, 0, -1,
        0, 0, -1,
        0, -2, -1
    ])
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(coeffs, degs)
    return char_poly


def check_topological_charges():
    A1 = 2
    B1 = 1
    A2 = 1.8
    B2 = 1.2
    char_poly = get_model_ChP(A1, A2, B1, B2)
    # print(char_poly.to_str(pt.CStrVec(['E', 'x', 'y'])))
    solver = wGs.StripGBZMainSolver(char_poly)
    solver.solve_non_Bloch_solutions(0)
    dChP = [char_poly.derivative(i) for i in range(3)]
    radius_Cr = 2

    critical_segs = wGs.get_critical_segments(
        solver.non_Bloch_sols,
        radius_Cr
    )
    intersection_point_list = []
    for seg in critical_segs:
        for i in range(seg[0].shape[0]):
            intersection_point_list.append(
                wGs.interp_to_radius_r(
                    seg[0][i,:],
                    seg[1][i,:],
                    radius_Cr
                )
            )

    # 2. Calculate topological charges
    all_pairs = wGs.pair_points(intersection_point_list)
    positive_charges = []
    negative_charges = []
    for curr_pair in all_pairs:
        if wGs.calculate_topological_charge(dChP, 0, curr_pair) > 0:
            positive_charges.append(curr_pair[0])
            negative_charges.append(curr_pair[1])
        else:
            positive_charges.append(curr_pair[1])
            negative_charges.append(curr_pair[0])

    theta2_list = [np.log(p[1]).imag for p in (positive_charges + negative_charges)]
    theta2_list = np.sort(theta2_list)
    winding_loops = (theta2_list[1:] + theta2_list[:-1]) / 2
    winding_loops = np.hstack([
        winding_loops,
        [(theta2_list[0] + 2 * np.pi + theta2_list[-1]) / 2],
    ])

    theta1_list = np.linspace(-np.pi, np.pi, 1000)
    winding_numbers = []
    for curr_theta2 in winding_loops:
        winding_numbers.append(
            wGs.calculate_winding_around_loop(
                char_poly,
                0,
                radius_Cr,
                theta1_list,
                curr_theta2 * np.ones_like(theta1_list)
            )
        )

    # visualization
    for p in positive_charges:
        plt.plot(np.log(p[0]).imag, np.log(p[1]).imag, 'r.')
    for p in negative_charges:
        plt.plot(np.log(p[0]).imag, np.log(p[1]).imag, 'b.')

    for i in range(len(winding_numbers)):
        plt.plot(
            theta1_list,
            winding_loops[i] * np.ones_like(theta1_list)
        )
        plt.text(0, winding_loops[i], winding_numbers[i])

    plt.show()


if __name__ == "__main__":
    # plot_ellipse_abs()
    check_topological_charges()
