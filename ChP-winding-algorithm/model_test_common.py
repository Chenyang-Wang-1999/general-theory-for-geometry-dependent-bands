'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-01-02
Copyright © Department of Physics, Tsinghua University. All rights reserved

Common script for testing the new winding number algorithm
'''


import BerryPy.TightBinding as tb
import numpy as np
from scipy import linalg as la
import poly_tools as pt
import winding_GBZ_solver as wGs
import os
import pickle
from quasi_1D_winding import calculate_quasi_1D_winding


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
        # if abs(winding_increment) > np.pi/2:
        #     print("Warning: large increment detected! increment = %.2f pi" % (winding_increment / np.pi))
        winding_number += winding_increment

    winding_number /= 2 * np.pi
    return winding_number


def compare_winding_number(
    fname: str,
    model_2D: tb.TightBindingModel,
    E_ref: complex,
    n_minor: int,
    n_theta1: int,
    n_theta1_num: int,
    r_list: np.ndarray[1,float],
    model_info
):
    '''
    Compare the total winding number and the quasi-1D winding number
    '''
    print(E_ref)
    # 1. Calculate total winding number
    coeffs, degs = model_2D.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    solver = wGs.StripGBZMainSolver(char_poly)
    err_code = solver.solve_non_Bloch_solutions(E_ref)

    w_list = np.zeros_like(r_list, dtype=float) * np.nan
    for r_ind, r in enumerate(r_list):
        err_code = solver.get_total_winding(r, n_theta1)
        if err_code == 0:
            w_list[r_ind] = solver.total_winding
        else:
            print("Warning: error code %d when calculating total winding number" % err_code)

    # 2. Calculate quasi-1D winding number
    model_q1D = model_2D.get_supercell(
        [(0, j) for j in range(n_minor)],
        np.array(
            [[1, 0],
             [0, n_minor]],
            dtype=int
        )
    )
    model_q1D = model_q1D.truncate(1)
    w_list_q1D = np.zeros_like(r_list, dtype=float) * np.nan
    for r_ind, r in enumerate(r_list):
        w_list_q1D[r_ind] = calculate_quasi_1D_winding(
            model_q1D, E_ref, n_theta1_num, r
        )

    # 3. save
    file_id = 0
    while os.path.exists("%s-%d.pkl" % (fname, file_id)):
        file_id += 1

    with open("%s-%d.pkl" % (fname, file_id), "wb") as f:
        pickle.dump(
            {
                "w_list": w_list,
                "w_list_q1D": w_list_q1D,
                "r_list": r_list,
                "E_ref": E_ref,
                "n_minor": n_minor,
                "n_theta1": n_theta1,
                "n_theta1_num": n_theta1_num,
                "info": model_info
            },
            f
        )
