'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-12-31
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import winding_GBZ_solver as wGs
import poly_tools as pt
import BerryPy.TightBinding as tb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append("../paper/HN-model")
import HN_model_common as HN


Default_HN_Params = (
    1 + 1j,
    1.5 + 1.2j,
    -1 + 1j,
    -1.2 - 0.5j
)


def HN_model_2D(Jx1, Jx2, Jy1, Jy2):
    ''' Hamiltonian of 2D HN model '''
    dim = 2
    site_num = 1
    intra_cell = []
    inter_cell = [
        [0, 0, Jx1, (1, 0)],
        [0, 0, Jx2, (-1, 0)],
        [0, 0, Jy1, (0, 1)],
        [0, 0, Jy2, (0, -1)]
    ]
    return tb.TightBindingModel(
        dim, site_num, np.eye(2), intra_cell, inter_cell
    )


def calculate_non_Bloch_sols():
    Jx1, Jx2, Jy1, Jy2 = Default_HN_Params
    model = HN_model_2D(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array(
            [[1, 0],
             [1, 1]],
            dtype=int
        )
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    # solve
    GBZ_loops = wGs.solve_non_Bloch_loops(
        char_poly, -1 + 1j
    )
    GBZ_loops = wGs.convert_to_complex_numbers(GBZ_loops)

    # visualization
    for loop in GBZ_loops:
        loop_arr = np.array(loop)
        plt.plot(loop_arr[:,0].real, loop_arr[:,0].imag)
    plt.axis('equal')
    plt.show()


def check_critical_segments():
    Jx1, Jx2, Jy1, Jy2 = Default_HN_Params
    model = HN_model_2D(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array(
            [[1, 0],
             [1, 1]],
            dtype=int
        )
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    dchar_poly = [char_poly.derivative(i) for i in range(3)]
    # solve
    GBZ_loops = wGs.solve_non_Bloch_loops(
        char_poly, -1 + 1j
    )
    GBZ_loops = wGs.convert_to_complex_numbers(GBZ_loops)
    all_holes, all_intervals = wGs.get_holes_from_loops(GBZ_loops)
    print(all_holes, all_intervals)
    GBZ_loops = [np.array(loop) for loop in GBZ_loops]
    # check critical segments
    r = 0.9
    critical_seg = wGs.get_critical_segments(
        GBZ_loops, r
    )
    # visualization
    for loop in GBZ_loops:
        plt.plot(loop[:,0].real, loop[:,0].imag, 'b')
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(r * np.cos(theta), r * np.sin(theta), 'r')

    total_n_segs = 0
    for seg in critical_seg:
        total_n_segs += seg[0].shape[0]
        plt.plot(
            seg[0][:,0].real,
            seg[0][:,0].imag,
            'x'
        )
        plt.plot(
            seg[1][:,0].real,
            seg[1][:,0].imag,
            '+'
        )

        for i in range(seg[0].shape[0]):
            intersection_point = wGs.interp_to_radius_r(
                seg[0][i,:], seg[1][i,:], r
            )
            plt.plot(intersection_point[0].real, intersection_point[0].imag, '.')
            print(abs(intersection_point[0]))
            print(wGs.calculate_beta2_radius_diff(
                r, np.log(intersection_point[0]).imag, 
                char_poly, dchar_poly, 
                1+1j))

    print("Total segments:", total_n_segs)

    plt.axis('equal')
    plt.show()


def check_derivative():
    Jx1, Jx2, Jy1, Jy2 = Default_HN_Params
    model = HN_model_2D(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array(
            [[1, 0],
             [1, 1]],
            dtype=int
        )
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    dchar_poly = [char_poly.derivative(i) for i in range(3)]

    r = np.random.rand() * 0.5 + 0.5
    theta1 = np.random.rand() * 2 * np.pi
    dtheta1 = np.random.rand() * 1e-3

    diff1, der = wGs.calculate_beta2_radius_diff(
        r, theta1, char_poly, dchar_poly, 1 + 1j
    )
    diff2, _ = wGs.calculate_beta2_radius_diff(
        r, theta1 + dtheta1, char_poly, dchar_poly, 1 + 1j
    )

    print(diff1, diff2)
    print(diff2 - diff1)
    print(der * dtheta1)
    print(abs(diff2 - diff1 - der * dtheta1) / (der * dtheta1))


def check_winding_number():
    ''' Check the winding number calculation '''
    # 1. Get intersections
    Jx1, Jx2, Jy1, Jy2 = Default_HN_Params
    E_ref = 1 + 1j
    model = HN_model_2D(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array(
            [[1, 0],
             [1, 1]],
            dtype=int
        )
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    dchar_poly = [char_poly.derivative(i) for i in range(3)]
    # solve
    GBZ_loops = wGs.solve_non_Bloch_loops(
        char_poly, E_ref
    )
    GBZ_loops = wGs.convert_to_complex_numbers(GBZ_loops)
    all_holes, all_intervals = wGs.get_holes_from_loops(GBZ_loops)
    print(all_holes, all_intervals)
    GBZ_loops = [np.array(loop) for loop in GBZ_loops]
    # check critical segments
    r = 0.9
    critical_seg = wGs.get_critical_segments(
        GBZ_loops, r
    )

    intersection_point_list = []
    for seg in critical_seg:
        for i in range(seg[0].shape[0]):
            intersection_point = wGs.interp_to_radius_r(
                seg[0][i,:], seg[1][i,:], r
            )
            intersection_point_list.append(intersection_point)

    # 2. Sort intersections by the argument of beta2 
    intersection_point_list = sorted(
        intersection_point_list,
        key=lambda x: np.log(x[1]).imag
    )

    # 3. Calculate topological charges
    topo_charge = [
        wGs.calculate_topological_charge(
            dchar_poly, 
            E_ref,
            p[0],
            p[1]
        )
        for p in intersection_point_list
    ]

    # 4. Calculate loop winding numbers
    beta2_imag_list = [
        np.log(p[1]).imag for p in intersection_point_list
    ]

    winding_arg_list = np.zeros_like(beta2_imag_list)
    for ind in range(len(beta2_imag_list)):
        next_ind = (ind + 1)
        if next_ind == len(beta2_imag_list):
            winding_arg_list[ind] = (
                (beta2_imag_list[ind] + beta2_imag_list[0] + 2 * np.pi) / 2
            )
        else:
            winding_arg_list[ind] = (
                (beta2_imag_list[ind] + beta2_imag_list[next_ind]) / 2
            )

    theta1_list = np.linspace(-np.pi, np.pi, 100)
    winding_number_list = [
        wGs.calculate_winding_around_loop(
            char_poly, E_ref, r, 
            theta1_list,
            theta2 * np.ones(len(theta1_list))
        )
        for theta2 in winding_arg_list
    ]

    # 5. Visualization
    for point_ind, point in enumerate(intersection_point_list):
        if topo_charge[point_ind] > 0:
            plt.plot(np.log(point[0]).imag, np.log(point[1]).imag, 'rx')
        else:
            plt.plot(np.log(point[0]).imag, np.log(point[1]).imag, 'bx')
    
    for arg_ind, arg_beta2 in enumerate(winding_arg_list):
        plt.plot(
            theta1_list,
            arg_beta2 * np.ones_like(theta1_list)
        )
        plt.text(np.pi, arg_beta2, winding_number_list[arg_ind])
    
    plt.show()


def test_total_winding():
    # 1. Get intersections
    Jx1, Jx2, Jy1, Jy2 = Default_HN_Params
    E_ref = -1 + 1j
    model = HN_model_2D(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array(
            [[1, 0],
             [1, 1]],
            dtype=int
        )
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    # solve
    solver = wGs.StripGBZMainSolver(char_poly)
    err_code = solver.solve_non_Bloch_solutions(E_ref)
    r_list = np.linspace(0.5, 1.5, 50)
    w_list = np.zeros_like(r_list) * np.nan

    for r_ind, r_val in enumerate(r_list):
        err_code = solver.get_total_winding(
            r_val, 100
        )
        if err_code == 0:
            w_list[r_ind] = solver.total_winding
        else:
            print("Error at r =", r_val)

    plt.plot(r_list, w_list, '.-')
    plt.show()


def test_GBZ_sweep():
    Jx1, Jx2, Jy1, Jy2 = Default_HN_Params
    model = HN_model_2D(Jx1, Jx2, Jy1, Jy2)
    # model = model.get_supercell(
    #     [(0,0)],
    #     np.array(
    #         [[1, 0],
    #          [1, 1]],
    #         dtype=int
    #     )
    # )
    coeffs, degs = model.get_characteristic_polynomial_data()

    E_real = np.linspace(-5, 5, 101)
    E_imag = np.linspace(-3, 3, 51)
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_mesh = E_real_mesh + 1j * E_imag_mesh
    E_arr = E_mesh.flatten()

    flags = wGs.batch_sweep_strip_GBZ(
        coeffs, degs, E_arr, 12
    )

    with open("data/GBZ_numerical2.pkl", "wb") as fp:
        pickle.dump(
            (E_arr, flags),
            fp
        )


def plot_GBZ_sweep():
    with open("data/GBZ_numerical2.pkl", "rb") as fp:
        E_arr, flags = pickle.load(fp)
    E_spectrum = []
    for curr_ind, curr_flag in enumerate(flags):
        if curr_flag is True:
            E_spectrum.append(E_arr[curr_ind])

    E_spectrum = np.array(E_spectrum)
    plt.plot(E_spectrum.real, E_spectrum.imag, '.')
    bound, loop = HN.get_11_DGBZ_spectrum(*Default_HN_Params)
    plt.plot(loop.real, loop.imag)
    plt.show()
 

if __name__ == '__main__':
    # calculate_non_Bloch_sols()
    # check_critical_segments()
    # check_derivative()
    # check_winding_number()
    test_total_winding()
    # test_GBZ_sweep()
    # plot_GBZ_sweep()
