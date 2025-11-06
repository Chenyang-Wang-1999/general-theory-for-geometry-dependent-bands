'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-01-01
Copyright © Department of Physics, Tsinghua University. All rights reserved

Test the relation of the total winding number and the quasi-1D winding number
'''

import numpy as np
import winding_GBZ_solver as wGs
import BerryPy.TightBinding as tb
from scipy import linalg as la
import matplotlib.pyplot as plt
import poly_tools as pt


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


def calculate_spectral_winding_around_loop(
    model: tb.TightBindingModel,
    winding_loop: np.ndarray[1,complex],
    E_ref: complex
):
    '''
    Calculate the spectral winding number around a loop
    Args:
        model: tight binding model
        winding_loop: the loop around which the winding number is calculated
    '''
    winding_number = 0
    eigv_list = np.zeros(
        (model.SiteNum, len(winding_loop)),
        dtype=complex
    )
    # Calculate eigenvalues
    for point_ind in range(len(winding_loop)):
        H_mat = model.get_bulk_Hamiltonian_complex(
            (winding_loop[point_ind],)
        ).todense()
        eigv, _ = la.eig(H_mat)
        eigv_list[:,point_ind] = eigv

    # Calculate winding number
    for curr_ind in range(len(winding_loop)):
        next_ind = (curr_ind + 1) % len(winding_loop)
        eigv_curr = eigv_list[:,curr_ind]
        eigv_next = eigv_list[:,next_ind]
        winding_increment = np.log(np.prod((E_ref - eigv_next) / (E_ref - eigv_curr))).imag
        if abs(winding_increment) > np.pi/2:
            print("Warning: large increment detected! increment = %.2f pi" % (winding_increment / np.pi))
        winding_number += winding_increment

    winding_number /= 2 * np.pi
    return winding_number


def calculate_winding_loop_brute_force(
    model: tb.TightBindingModel,
    winding_loop: np.ndarray[1,complex],
    E_ref: complex
):
    '''
    Calculate the winding loop around a point with brute force
    '''
    det_val = np.zeros(len(winding_loop), dtype=complex)
    for point_ind in range(len(winding_loop)):
        H_mat = model.get_bulk_Hamiltonian_complex(
            (winding_loop[point_ind],)
        ).todense()
        det_val[point_ind] = la.det(E_ref * np.eye(model.SiteNum) - H_mat)

    return wGs.get_winding_number(det_val)


def test_winding_number_calculation():
    model = HN_model_2D(*Default_HN_Params)
    model = model.get_supercell(
        [(0,0)],
        np.array(
            [[1, 0],
             [1, 1]],
            dtype=int
        )
    )
    n_sites = 40
    model_q1D = model.get_supercell(
        [(0, j) for j in range(n_sites)],
        np.array(
            [[1, 0],
             [0, n_sites]],
            dtype=int
        )
    )
    model_q1D = model_q1D.truncate(1)
    E_ref = 1 + 1j
    r_list = np.linspace(0.5, 1.5, 100)
    w_list = np.zeros_like(r_list)
    for r_ind, r_val in enumerate(r_list):
        winding_loop = np.exp(1j * np.linspace(0, 2 * np.pi, 300)) * r_val
        w_list[r_ind] = calculate_spectral_winding_around_loop(model_q1D, winding_loop, E_ref)

    plt.plot(r_list, w_list)
    plt.show()


def test_total_winding():
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


if __name__ == '__main__':
    # test_winding_number_calculation()
    test_total_winding()