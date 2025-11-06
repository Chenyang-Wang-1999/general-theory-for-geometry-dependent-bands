'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-10-22
Copyright © Department of Physics, Tsinghua University. All rights reserved

Calculate Amoeba for a given 2D polynomial
'''

import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import pickle
from scipy import interpolate
sys.path.append("../puncture-calculation")
import poly_tools as pt
sys.path.append("../paper/HN-model")
import HN_model_common as HN


def solve_minor_variable(f: pt.CLaurent, x: complex):
    ''' Solve y from f(x, y) = 0 '''
    f_1d = f.partial_eval(
        pt.CScalarVec([x]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1])
    )
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    f_1d.num.batch_get_data(coeffs, degs)

    # Convert f_1d to numpy array
    max_deg = max(degs)
    eq_arr = np.zeros(max_deg + 1, dtype=complex)
    for term_ind in range(len(degs)):
        eq_arr[max_deg - degs[term_ind]] = coeffs[term_ind]
    
    return np.roots(eq_arr)


def sweep_Amoeba(
    f: pt.CLaurent,
    x_log_range: tuple[float, float],
    n_norm: int,
    n_arg: int
):
    ''' Sweep Amoeba with the given x range '''
    # Generate x mesh
    x_log = np.linspace(x_log_range[0], x_log_range[1], n_norm)
    x_norm = np.exp(x_log)
    x_arg = np.linspace(0, 2 * np.pi, n_arg)
    x_norm_mesh, x_arg_mesh = np.meshgrid(x_norm, x_arg)
    x_mesh = x_norm_mesh * np.exp(1j * x_arg_mesh)

    # Get maximal degree
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    f.num.batch_get_data(coeffs, degs)
    degs = np.reshape(degs, (-1, 2))
    max_deg = max(degs[:,1])

    # Solve y
    y_norm = np.zeros(
        x_mesh.shape + (max_deg,)
    )
    for row_ind in range(x_mesh.shape[0]):
        for col_ind in range(x_mesh.shape[1]):
            y_sol = solve_minor_variable(f, x_mesh[row_ind,col_ind])
            y_norm[row_ind,col_ind,:] = np.sort(np.abs(y_sol))
    
    return x_norm_mesh, y_norm


def test_Amoeba():
    ''' Test Amoeba calculation, Wang Zhong PRX 2024 '''
    t = 1
    t1 = 0.5
    gamma = 0.2
    energy_E3 = -2.5
    energy_E4 = -1
    reference_energy = energy_E3

    x_log_range = [-4, 4]
    n_norm = 200
    n_arg = 200

    coeffs = pt.CScalarVec(
        [1] 
        + [-t] * 4 
        + [-t1] * 4
        + [-gamma, gamma, -gamma, gamma]
    )
    degs = pt.CLaurentIndexVec([
        # E, betax, betay
        1, 0, 0,
        0, 1, 0,
        0, -1, 0,
        0, 0, 1,
        0, 0, -1,
        0, 1, 1,
        0, 1, -1,
        0, -1, 1,
        0, -1, -1,
        0, 1, 0,
        0, -1, 0,
        0, 0, 1,
        0, 0, -1
    ])
    f = pt.CLaurent(3)
    f.set_Laurent_by_terms(coeffs, degs)

    # Calculate
    f_beta = f.partial_eval(
        pt.CScalarVec([reference_energy]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1, 2])
    )
    x_norm, y_norm = sweep_Amoeba(
        f_beta, x_log_range, n_norm, n_arg
    )

    # Add triangular mesh
    for layer_ind in range(y_norm.shape[2]):
        plt.figure()
        plt.pcolor(
            np.log(x_norm),
            np.log(y_norm[:,:,layer_ind]),
            np.zeros_like(x_norm)
        )
    plt.show()

    # # plot
    # fig = plt.figure()
    # ax = fig.gca()
    # for row_ind in range(x_norm.shape[0]):
    #     for col_ind in range(x_norm.shape[1]):
    #         ax.plot(
    #             np.ones_like(y_norm[row_ind][col_ind]) * np.log(x_norm[row_ind,col_ind]),
    #             np.log(y_norm[row_ind][col_ind]),
    #             'b.'
    #         )
    # ax.set_xlim([-3, 3])
    # ax.set_ylim([-3, 3])
    # ax.set_aspect(1)
    # ax.set_xlabel("$\\mu_x$")
    # ax.set_ylabel("$\\mu_y$")
    # plt.show()


def paper_calculate_HN_Amoeba():
    model = HN.get_HN_model(*HN.Default_Model_Params)
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs), 
        pt.CLaurentIndexVec(degs.flatten())
    )

    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    rx = np.abs(Jx1 / Jx2)
    ry = np.abs(Jy1 / Jy2)

    E_ref1 = 1 + 1j
    E_ref2 = 2 + 2j
    E_ref3 = 3 + 3j
    E_ref = E_ref2
    # n_norm = 351
    # n_arg = 351
    n_norm = 101
    n_arg = 1001

    x_log_range = [-0.1 + np.log(rx), 0.1 + np.log(rx)]
    f_beta = char_poly.partial_eval(
        pt.CScalarVec([E_ref]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1, 2])
    )

    x_norm, y_norm = sweep_Amoeba(
        f_beta, x_log_range, n_norm, n_arg
    )

    x_log_range = [-0.1 + np.log(ry), 0.1 + np.log(ry)]
    f_beta2 = char_poly.partial_eval(
        pt.CScalarVec([E_ref]),
        pt.CIndexVec([0]),
        pt.CIndexVec([2, 1])
    )
    y_norm2, x_norm2 = sweep_Amoeba(
        f_beta2, x_log_range, n_norm, n_arg
    )

    fname = "data/paper-Amoeba-E_%.2f_%.2f.pkl" % (E_ref.real, E_ref.imag)
    if os.path.exists(fname):
        with open(fname, "rb") as fp:
            all_Amoeba_segs = pickle.load(fp)
    else:
        all_Amoeba_segs = []
    
    print(x_norm.shape, y_norm.shape)
    print(x_norm2.shape, y_norm2.shape)
    all_Amoeba_segs += [(x_norm, y_norm, True), (y_norm2, x_norm2, False)]

    with open(fname, "wb") as fp:
        pickle.dump(all_Amoeba_segs, fp)

    # Add triangular mesh
    for curr_pair in all_Amoeba_segs:
        curr_x_norm, curr_y_norm, x_major_flag = curr_pair
        for layer_ind in range(curr_y_norm.shape[2]):
            if x_major_flag:
                plt.pcolor(
                    np.log(curr_x_norm),
                    np.log(curr_y_norm[:,:,layer_ind]),
                    np.zeros_like(curr_x_norm)
                )
            else:
                plt.pcolor(
                    np.log(curr_y_norm[:,:,layer_ind]),
                    np.log(curr_x_norm),
                    np.zeros_like(curr_x_norm)
                )
    plt.show()


def plot_HN_Amoeba(E_ref: complex):
    model = HN.get_HN_model(*HN.Default_Model_Params)
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs), 
        pt.CLaurentIndexVec(degs.flatten())
    )

    x_log_range = [-3, 3]
    n_norm = 300
    n_arg = 300

    f_beta = char_poly.partial_eval(
        pt.CScalarVec([E_ref]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1, 2])
    )
    x_norm, y_norm = sweep_Amoeba(
        f_beta, x_log_range, n_norm, n_arg
    )

    f_beta2 = char_poly.partial_eval(
        pt.CScalarVec([E_ref]),
        pt.CIndexVec([0]),
        pt.CIndexVec([2, 1])
    )
    y_norm2, x_norm2 = sweep_Amoeba(
        f_beta2, x_log_range, n_norm, n_arg
    )

    fname = "data/paper-Amoeba-E_%.2f_%.2f.pkl" % (E_ref.real, E_ref.imag)
    if os.path.exists(fname):
        with open(fname, "rb") as fp:
            all_Amoeba_segs = pickle.load(fp)
    else:
        all_Amoeba_segs = []
    
    all_Amoeba_segs += [(x_norm, y_norm, True), (y_norm2, x_norm2, False)]

    # Add triangular mesh
    for curr_pair in all_Amoeba_segs:
        curr_x_norm, curr_y_norm, x_major_flag = curr_pair
        for layer_ind in range(curr_y_norm.shape[2]):
            if x_major_flag:
                plt.pcolor(
                    np.log(curr_x_norm),
                    np.log(curr_y_norm[:,:,layer_ind]),
                    np.zeros_like(curr_x_norm)
                )
            else:
                plt.pcolor(
                    np.log(curr_y_norm[:,:,layer_ind]),
                    np.log(curr_x_norm),
                    np.zeros_like(curr_x_norm)
                )
    plt.show()


def check_HN_Amoeba_spectrum():
    bound, loop = HN.get_x_DGBZ_spectrum(*HN.Default_Model_Params)
    DGBZ_interp = interpolate.interp1d(
        [0, 1, 2, 3, 4], loop
    )
    E_ref = DGBZ_interp(np.random.rand() * 4)

    E_in = E_ref * 0.99
    plot_HN_Amoeba(E_in)
    E_out = E_ref * 1.01
    plot_HN_Amoeba(E_out)


if __name__ == '__main__':
    # test_Amoeba()
    paper_calculate_HN_Amoeba()
    # check_HN_Amoeba_spectrum()
