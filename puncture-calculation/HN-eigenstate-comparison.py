'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-09-10
Copyright © Department of Physics, Tsinghua University. All rights reserved

Compare eigenstates with strip GBZ solutions
'''

import numpy as np
from scipy import linalg as la
from scipy import optimize
from scipy import fft
from math import pi
import sys
import poly_tools as pt
from BerryPy import TightBinding as tb
sys.path.append('../paper/HN-model')
import HN_model_common as HN
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import partial_GBZ_solver as pGs
import pyvista as pv

J_X1, J_X2, J_Y1, J_Y2 = HN.Default_Model_Params
PHI_X1 = np.log(J_X1).imag
PHI_X2 = np.log(J_X2).imag
PHI_Y1 = np.log(J_Y1).imag
PHI_Y2 = np.log(J_Y2).imag
PHI_XBAR = (PHI_X1 + PHI_X2) / 2
PHI_YBAR = (PHI_Y1 + PHI_Y2) / 2
THETA_2 = (PHI_X2 + PHI_Y2 - PHI_X1 - PHI_Y1) / 2
R = np.sqrt(abs(
    J_X1 * J_Y1 / J_Y2 / J_X2
))

R_VEC = [np.sqrt(np.abs(J_X1 / J_X2)), np.sqrt(np.abs(J_Y1 / J_Y2))]

MODEL = HN.get_HN_model(J_X1, J_X2, J_Y1, J_Y2)
MODEL11 = MODEL.get_supercell([(0, 0)], np.array(
    [[1, 0],
        [1, 1]],
    dtype=int
))

coeffs, degs = MODEL11.get_characteristic_polynomial_data()
CHP11 = pt.CLaurent(3)
CHP11.set_Laurent_by_terms(
    pt.CScalarVec(coeffs),
    pt.CLaurentIndexVec(degs.flatten())
)
del coeffs, degs


def HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2):
    inter_cell = [
        [0, 0, Jx1, (1,0)],
        [0, 0, Jx2, (-1,0)],
        [0, 0, Jy1, (0,1)],
        [0, 0, Jy2, (0,-1)]
    ]
    return tb.TightBindingModel(2, 1, np.array([[1,0],[0,1]]), [], inter_cell, [[0,0]])


def solve_poly_eq_1D(poly_eq: pt.CLaurent):
    ''' Solve polynomial equation with numpy.root '''
    # 1. Get coefficients and degrees
    if isinstance(poly_eq, pt.CLaurent):
        # Laurent polynomial
        coeffs = pt.CScalarVec([])
        degs = pt.CIndexVec([])
        poly_eq.num.batch_get_data(coeffs, degs)
    else:
        # polynomial
        coeffs, degs = poly_eq.batch_get_data()
    
    # 2. Generate 1D array
    max_deg = max(degs)
    equation_1d_nparray = np.zeros(max_deg + 1, dtype=complex)

    for term_id in range(len(coeffs)):
        equation_1d_nparray[max_deg - degs[term_id]] = coeffs[term_id]
    
    all_roots = np.roots(equation_1d_nparray)

    return all_roots


def generate_non_Bloch_states(
    beta1: complex,
    beta2: complex,
    basis_vec: np.ndarray[2, int],
    coords: np.ndarray[2, float],
    p0: tuple[float],
    p0_global: tuple[float],
    r_vec: tuple[float] = (1.0, 1.0)
) -> np.ndarray[1, complex]:
    ''' Generate non-Bloch states with given beta, and scaled by 'r_vec'''
    psi = np.zeros(coords.shape[0], dtype=complex)
    inv_basis_vec = la.inv(basis_vec)
    for coord_ind in range(coords.shape[0]):
        rel_coord = inv_basis_vec @ np.array([coords[coord_ind][0], coords[coord_ind][1]])
        psi[coord_ind] = (
            beta1 ** (rel_coord[0] - p0[0])
            * beta2 ** (rel_coord[1] - p0[1])
            / (r_vec[0] ** (coords[coord_ind][0] - p0_global[0]))
            / (r_vec[1] ** (coords[coord_ind][1] - p0_global[1]))
        )

    return psi / la.norm(psi)


def convert_Bloch_wavevec_to_non_Bloch(
    k1: float, k2: float
):
    '''Generate non-Bloch waves with the same energy'''
    beta1 = [
        R * np.exp(1j * k1),
        R * np.exp(-1j * k1) * np.exp(-2j * THETA_2)
    ]

    beta_pairs = []

    # first pair
    curr_beta1 = beta1[0]
    beta2_sq = (J_X2 * curr_beta1 + J_Y1) / (J_X1 / curr_beta1 + J_Y2)
    beta2 = np.sqrt(np.abs(beta2_sq)) * np.exp(1j * k2)
    beta2_another = beta2_sq / beta2
    beta_pairs += [(curr_beta1, beta2), (curr_beta1, beta2_another)]

    E = J_X1 * beta2 / curr_beta1 + J_X2 * curr_beta1 / beta2 + J_Y1 / beta2 + J_Y2 * beta2

    # second pair
    curr_beta1 = beta1[1]
    beta2_sol = np.roots([
        J_X1 / curr_beta1 + J_Y2,
        -E,
        J_X2 * curr_beta1 + J_Y1
    ])
    beta_pairs += [(curr_beta1, curr_beta2) for curr_beta2 in beta2_sol]
    
    return beta_pairs


class EigenstateGenerator:
    ''' Generate approximate eigenstates from given Bloch wavevectors and coefficients '''
    point_vec: np.ndarray[2, float]
    basis_vec: np.ndarray[2, int]
    p0: tuple[float]
    p0_global: tuple[float]
    r_vec: tuple[float]

    def __init__(self,
        basis_vec: np.ndarray[2, int],
        p0: tuple[float],
        r_vec: tuple[float] = None
    ):
        self.basis_vec = basis_vec
        self.p0 = p0
        self.p0_global = basis_vec @ np.asarray(p0)
        if r_vec is None:
            self.r_vec = (1.0, 1.0)
        else:
            self.r_vec = r_vec

    def __call__(self,
        point_vec: np.ndarray[2, float],
        k1, k2,                 # Bloch wavevector
        *A_list     # Coefficients
    ) -> np.ndarray[1, complex]:
        # 1. Get all beta pairs
        all_beta_pairs = convert_Bloch_wavevec_to_non_Bloch(
            k1, k2
        )

        # 2. Get the four non-Bloch waves and calculate
        eigvec = np.zeros(point_vec.shape[0], dtype=complex)
        for beta_ind, beta_pair in enumerate(all_beta_pairs):
            beta1, beta2 = beta_pair
            eigvec += generate_non_Bloch_states(
                beta1, beta2, self.basis_vec,
                point_vec, self.p0, self.p0_global, self.r_vec
            ) * (A_list[2 * beta_ind] + 1j * A_list[2 * beta_ind + 1])
        
        return np.hstack([eigvec.real, eigvec.imag])


def check_eigenenergy():
    k1, k2 = np.random.rand(2) * 2 * np.pi
    all_beta = convert_Bloch_wavevec_to_non_Bloch(k1, k2)
    for beta1, beta2 in all_beta:
        E = J_X1 * beta2 / beta1 + J_X2 * beta1 / beta2 + J_Y1 / beta2 + J_Y2 * beta2
        print(E)


def calculate_beta_given_E(E: complex):
    ''' Given energy E, calculate GBZ beta '''
    # 1. Calculate cos kappa and cos theta
    a = E ** 2 / 4
    b = (
        2 * np.exp(1j * (PHI_XBAR + PHI_YBAR))
        * np.sqrt(abs(J_X1 * J_X2 * J_Y1 * J_Y2))
    )
    c = J_X1 * J_X2 + J_Y1 * J_Y2
    lin_eq = np.array([
        [a.real, -b.real],
        [a.imag, -b.imag]
    ])
    t = la.solve(lin_eq, [[c.real], [c.imag]])

    # 2. calculate kappa and theta
    kappa_origin = np.arccos(1 / np.sqrt(t[0, 0]))
    kappa_list = [
        kappa_origin,
        - kappa_origin,
        kappa_origin + pi,
        - (kappa_origin + pi)
    ]
    theta_list = [
        np.arccos(t[1, 0]) - THETA_2,
        - np.arccos(t[1, 0]) - THETA_2
    ]

    # 3. calculate beta1 and beta2
    all_beta_pairs = []
    for theta in theta_list:
        beta1 = R * np.exp(1j * theta)
        exp_is0_half = np.sqrt(
            (J_X2 * beta1 + J_Y1) / (J_X1 / beta1 + J_Y2)
        )
        exp_is0_half /= abs(exp_is0_half)

        beta2_list = exp_is0_half * np.exp(1j * np.array(kappa_list)) * np.sqrt(
            abs(J_X2 * beta1 + J_Y1) / abs(J_X1 / beta1 + J_Y2)
        )
        for beta2 in beta2_list:
            curr_E = (
                J_X1 * beta2 / beta1
                + J_X2 * beta1 / beta2
                + J_Y1 / beta2
                + J_Y2 * beta2
            )
            if abs(E - curr_E) < 1e-6:
                all_beta_pairs.append((beta1, beta2))
    return all_beta_pairs


def solve_least_square(E, point_vec, eigvec):
    pass


def scale_eigenstates(eigvec, point_vec, p0):
    ''' Scale the eigenstates by R_VEC '''
    scaled_eigvec = eigvec.copy()
    for point_ind in range(point_vec.shape[0]):
        scaled_eigvec[point_ind] /= (
            R_VEC[0] ** (point_vec[point_ind, 0] - p0[0])
            * R_VEC[1] ** (point_vec[point_ind, 1] - p0[1])
        )
    return scaled_eigvec


def recover_eigenstates(eigvec, point_vec, p0):
    ''' Recover the eigenstates scaled by R_VEC '''
    recovered_eigvec = eigvec.copy()
    for point_ind in range(point_vec.shape[0]):
        recovered_eigvec[point_ind] *= (
            R_VEC[0] ** (point_vec[point_ind, 0] - p0[0]) 
            * R_VEC[1] ** (point_vec[point_ind, 1] - p0[1])
        )
    return recovered_eigvec


def compare_eigenstates(E: complex, aspect: float, N: int):
    ''' Compare eigenstate for given E '''
    with open(
        "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    p0 = (shape[0] / 2, shape[1] / 2)
    eigv_ind = np.argmin(np.abs(eigv - E))
    print("E index:", eigv_ind)
    print("E value:", eigv[eigv_ind])
    all_beta_pairs = calculate_beta_given_E(eigv[eigv_ind])

    # plot eigenstates
    curr_eigvec = scale_eigenstates(eigvec[:, eigv_ind], point_vec, p0)
    # curr_eigvec = eigvec[:, eigv_ind]
    states_num = curr_eigvec.reshape(shape, order='F')
    x_coords = point_vec[:, 0].reshape(shape, order='F')
    y_coords = point_vec[:, 1].reshape(shape, order='F')

    # for beta1, beta2 in all_beta_pairs:
    #     print(abs(beta1), abs(beta2))
    #     plt.figure()
    #     curr_strip_state = generate_non_Bloch_states(
    #         beta1, beta2, np.array([
    #             [1, 0],
    #             [1, 1]
    #         ]), point_vec, p0
    #     ).reshape(shape, order='F')
    #     pcl = plt.pcolor(x_coords, y_coords, np.abs(curr_strip_state))
    #     plt.axis('equal')

    # for row_ind in range(x_coords.shape[0]):
    #     for col_ind in range(x_coords.shape[1]):
    #         states_num[row_ind, col_ind] *= (
    #             R_VEC[0] ** (x_coords[row_ind, col_ind] - p0[0])
    #             * R_VEC[1] ** (y_coords[row_ind, col_ind] - p0[1])
    #         )
    # with open("data/temp.txt", "w") as fp:
    #     print(y_coords, file=fp)
    plt.figure()
    pcl = plt.pcolor(x_coords, y_coords, np.abs(states_num), cmap='Blues')
    # pcl = plt.pcolor(x_coords, y_coords, vals)
    plt.colorbar(pcl)
    plt.axis('equal')
    plt.show()
    # plt.savefig("Figures/temp.pdf")


def test_strip_GBZ_spectrum():
    bound, loop = HN.get_11_DGBZ_spectrum(J_X1, J_X2, J_Y1, J_Y2)
    loop = np.asarray(loop)
    plt.plot(loop.real, loop.imag)
    plt.show()


def test_eigenstates_scale(E: complex, aspect: float, N: int):
    ''' Test whether the eigenstates are scaled '''
    with open(
        "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    p0 = (shape[0] / 2, shape[1] / 2)
    eigv_ind = np.argmin(np.abs(eigv - E))
    print("E index:", eigv_ind)
    print("E value:", eigv[eigv_ind])
    all_beta_pairs = calculate_beta_given_E(eigv[eigv_ind])

    x_coords = point_vec[:, 0].reshape(shape, order='F')
    y_coords = point_vec[:, 1].reshape(shape, order='F')

    for beta1, beta2 in all_beta_pairs:
        print(abs(beta1), abs(beta2))
        plt.figure()
        curr_strip_state = generate_non_Bloch_states(
            beta1, beta2, np.array([
                [1, 0],
                [1, 1]
            ]), point_vec, p0
        )
        curr_strip_state = scale_eigenstates(curr_strip_state, point_vec, p0).reshape(shape, order='F')
        pcl = plt.pcolor(x_coords, y_coords, np.abs(curr_strip_state), cmap='Blues')
        plt.axis('equal')

    plt.show()


def get_component(
    E: complex,
    aspect: float,
    N: int
):
    ''' Get component of eigenstates under the basis of non-Bloch waves '''
    # 1. Get energy E
    with open(
        "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    p0 = (shape[0] / 2, shape[1] / 2)
    eigv_ind = np.argmin(np.abs(eigv - E))
    print("E index:", eigv_ind)
    print("E value:", eigv[eigv_ind])
    selected_eigvec_num = eigvec[:, eigv_ind]

    # 2. Generate eigen basis and scale
    all_beta_pairs = calculate_beta_given_E(eigv[eigv_ind])
    all_non_Bloch_waves = np.zeros(
        (point_vec.shape[0], len(all_beta_pairs)),
        dtype=complex
    )
    for j, curr_beta_pair in enumerate(all_beta_pairs):
        beta1, beta2 = curr_beta_pair
        curr_strip_state = generate_non_Bloch_states(
            beta1, beta2, np.array([
                [1, 0],
                [1, 1]
            ]), point_vec, p0
        )
        all_non_Bloch_waves[:, j] = scale_eigenstates(
            curr_strip_state, point_vec, p0
        )
    all_non_Bloch_waves /= 1e11

    # 4. Select coordinate range
    x_boundary = 20
    y_boundary = 100
    selected_range = (
        (point_vec[:, 0] >= x_boundary)
        & (point_vec[:, 0] < shape[0] - x_boundary)
        & ((point_vec[:, 1] <= point_vec[:, 0] + y_boundary)
        | (point_vec[:, 1] > point_vec[:, 0] + shape[1] - y_boundary))
    )
    plt.plot(point_vec[:, 0], point_vec[:, 1], '.')
    plt.plot(point_vec[selected_range, 0], point_vec[selected_range, 1], '.')
    plt.show()

    # 5. Solve the linear equation to get the coefficients
    coeffs = la.lstsq(
        all_non_Bloch_waves[selected_range, :],
        selected_eigvec_num[selected_range]
    )[0]

    # 6. Check
    # for j in range(4):
    #     pcl = plt.pcolor(
    #         point_vec[:, 0].reshape(shape, order='F'),
    #         point_vec[:, 1].reshape(shape, order='F'),
    #         np.abs(all_non_Bloch_waves[:, j].reshape(shape, order='F')),
    #         cmap='Blues'
    #     )
    #     plt.colorbar(pcl)
    #     plt.show()
    lstsq_non_Bloch = all_non_Bloch_waves @ coeffs

    plt.figure(1)
    pcl = plt.pcolor(
        point_vec[:, 0].reshape(shape, order='F'),
        point_vec[:, 1].reshape(shape, order='F'),
        np.abs(selected_eigvec_num.reshape(shape, order='F')),
        cmap='hot'
    )
    plt.colorbar(pcl)

    plt.figure(2)
    pcl = plt.pcolor(
        point_vec[:, 0].reshape(shape, order='F'),
        point_vec[:, 1].reshape(shape, order='F'),
        np.abs(lstsq_non_Bloch.reshape(shape, order='F')),
        cmap='hot', vmax=0.10
    )
    plt.colorbar(pcl)

    plt.figure(3)
    pcl = plt.pcolor(
        point_vec[:, 0].reshape(shape, order='F'),
        point_vec[:, 1].reshape(shape, order='F'),
        np.abs((selected_eigvec_num - lstsq_non_Bloch).reshape(shape, order='F')),
        cmap='hot', vmax=0.11
    )
    plt.colorbar(pcl)
    plt.show()


def fit_eigenstate(
    E: complex,
    aspect: float,
    N: int
):
    ''' Get component of eigenstates under the basis of non-Bloch waves '''
    # 1. Get energy E
    with open(
        "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    p0 = (shape[0] / 2, shape[1] / 2)

    eigstate_fitter = EigenstateGenerator(
        [[1,0], [1,1]], p0, R_VEC
    )


    eigv_ind = np.argmin(np.abs(eigv - E))
    print("E index:", eigv_ind)
    print("E value:", eigv[eigv_ind])
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, 'x')
    plt.plot(E.real, E.imag, '+')
    plt.show()
    all_beta_pairs = calculate_beta_given_E(eigv[eigv_ind])
    k1 = np.log(all_beta_pairs[0][0]).imag
    k2 = np.log(all_beta_pairs[0][1]).imag
    selected_eigvec_num = eigvec[:, eigv_ind]

    # 4. Select coordinate range
    x_boundary = 20
    y_boundary = 100
    selected_range = (
        (point_vec[:, 0] >= x_boundary)
        & (point_vec[:, 0] < shape[0] - x_boundary)
        & ((point_vec[:, 1] <= point_vec[:, 0] + y_boundary)
        | (point_vec[:, 1] > point_vec[:, 0] + shape[1] - y_boundary))
    )
    plt.plot(point_vec[:, 0], point_vec[:, 1], '.')
    plt.plot(point_vec[selected_range, 0], point_vec[selected_range, 1], '.')
    plt.show()

    # 5. Solve the linear equation to get the coefficients
    res = optimize.curve_fit(
        eigstate_fitter, point_vec[selected_range, :], 
        np.hstack([
            selected_eigvec_num[selected_range].real,
            selected_eigvec_num[selected_range].imag
        ]),
        (k1, k2) + (1.0, 0.0) * 4,
        method='dogbox',
        bounds=[
            (-np.pi, -np.pi) + (-1.0, -1.0) * 4,
            (np.pi, np.pi) + (1.0, 1.0) * 4
        ],
        full_output= True
    )

    print(res[0])

    eigvec_fit_real = eigstate_fitter(point_vec, *res[0])
    eigvec_fit = (
        eigvec_fit_real[:len(eigvec_fit_real) // 2] 
        + 1j * eigvec_fit_real[len(eigvec_fit_real) // 2:] 
    ) 

    # plot
    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    eigvec_num_mesh = selected_eigvec_num.reshape(shape, order='F')
    eigvec_fit_mesh = eigvec_fit.reshape(shape, order='F')
    plt.figure()
    pcl = plt.pcolor(x_mesh, y_mesh, np.abs(eigvec_num_mesh))
    plt.colorbar(pcl)
    plt.figure()
    pcl = plt.pcolor(x_mesh, y_mesh, np.abs(eigvec_fit_mesh))
    plt.colorbar(pcl)
    plt.figure()
    pcl = plt.pcolor(x_mesh, y_mesh, np.abs(eigvec_num_mesh - eigvec_fit_mesh))
    plt.colorbar(pcl)
    plt.show()


def plot_all_eigenstates(aspect, N):
    with open(
        "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    for j in range(len(eigv)):
        plt.pcolor(
            point_vec[:, 0].reshape(shape, order='F'),
            point_vec[:, 1].reshape(shape, order='F'),
            abs(eigvec[:, j]).reshape(shape, order='F'),
            cmap='Blues'
        )
        plt.savefig("data/E_%.2f_%.2f.png"%(eigv[j].real, eigv[j].imag))


def get_non_Bloch_solutions_numerical(E: complex, N: int):
    ''' Get numerical non-Bloch solutions of quasi-1D system '''
    model = HN_model_tightbinding(J_X1, J_X2, J_Y1, J_Y2)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )

    model_super = model.get_supercell(
        [(0,j) for j in range(N)],
        np.array([
            [1, 0],
            [0, N]
        ], dtype=int)
    )
    model_super = model_super.truncate(1)

    # get charpoly
    coeffs, degs = model_super.get_characteristic_polynomial_data()
    coeffs = pt.CScalarVec(coeffs)
    degs = pt.CLaurentIndexVec(degs.flatten())
    char_poly = pt.CLaurent(2)
    char_poly.set_Laurent_by_terms(coeffs, degs)

    # substitute E
    poly_eq = char_poly.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1])
    )
    all_roots = solve_poly_eq_1D(poly_eq)

    return all_roots


def check_quasi_1D_roots():
    ''' Check quasi 1D solution of E and beta '''
    N = 15
    with open("data/paper-quasi-1D-11-N_%d.pkl"%(N), "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)
        # E, beta, params, N = pickle.load(fp)
    
    plt.plot(E.real, E.imag, '.')
    plt.show()

    point_ind = np.random.randint(len(E))
    curr_E = E[point_ind]
    curr_beta = beta[point_ind]

    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(
        (curr_beta * np.exp(1j * theta)).real,
        (curr_beta * np.exp(1j * theta)).imag
    )

    all_beta_theory = np.asarray(get_all_non_Bloch_theoretical(curr_E, N))
    plt.plot(
        all_beta_theory.real,
        all_beta_theory.imag,
        'x'
    )
    all_roots = get_non_Bloch_solutions_numerical(curr_E, N)
    print(len(all_roots), len(all_beta_theory))
    plt.plot(all_roots.real, all_roots.imag, '.')
    plt.show()

    beta2_distance = []
    for curr_root in all_roots:
        # print(curr_root)
        beta2 = get_minor_non_Bloch(curr_E, curr_root)
        beta2_distance.append(np.log(beta2[1] / beta2[0]).imag) 
        # print(np.abs(beta2))
    beta2_distance = np.abs(beta2_distance)
    beta2_distance = np.sort(beta2_distance)

    beta2_distance_th = []
    for curr_root in all_beta_theory:
        # print(curr_root)
        beta2 = get_minor_non_Bloch(curr_E, curr_root)
        beta2_distance_th.append(np.log(beta2[1] / beta2[0]).imag) 
        # print(np.abs(beta2))
    beta2_distance_th = np.abs(beta2_distance_th)
    beta2_distance_th = np.sort(beta2_distance_th)
    plt.plot(beta2_distance / np.pi, '.')
    plt.plot(beta2_distance_th / np.pi, 'x')
    plt.show()


def get_minor_non_Bloch(E, beta1):
    ''' Get minor non-Bloch solutions '''
    beta2_sol = np.roots([
        J_X1 / beta1 + J_Y2,
        -E,
        J_X2 * beta1 + J_Y1
    ])
    return beta2_sol
    

def generate_non_Bloch_basis(
    beta_list: list[tuple[complex]],
    basis_vec: np.ndarray[2, int],
    coords: np.ndarray[2, float],
    p0: tuple[float],
    p0_global: tuple[float] = None,
    r_vec: tuple[float] = (1.0, 1.0)    
):
    ''' Generate the basis of the eigenvectors defined by 'beta_list' '''
    non_Bloch_basis = np.zeros(
        (coords.shape[0], len(beta_list)),
        dtype=complex
    )

    if p0_global is None:
        p0_global = basis_vec @ np.asarray(p0)

    for beta_ind in range(len(beta_list)):
        beta1, beta2 = beta_list[beta_ind]
        non_Bloch_basis[:,beta_ind] = generate_non_Bloch_states(
            beta1, beta2, basis_vec, coords, p0, p0_global, r_vec
        )
    
    return non_Bloch_basis


def decompose_under_quasi_1D_direct(E, aspect_or_N1, N_or_N2):
    ''' Decompose the eigenstate under the quasi-1D basis, directly from the eigenvalue'''
    if isinstance(aspect_or_N1, float):
        aspect = aspect_or_N1
        N = N_or_N2
        with open(
            "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
            'rb'
        ) as fp:
            eigv, eigvec, point_vec, params, shape = pickle.load(fp)
    else:
        N1 = aspect_or_N1
        N2 = N_or_N2
        with open(
            "data/paper-HN-OBC-2-N1_%d-N2_%d.pkl" % (N1, N2),
            'rb'
        ) as fp:
            eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    p0 = (shape[0] / 2, shape[1] / 2)

    eigv_ind = np.argmin(np.abs(eigv - E))
    print("E index:", eigv_ind)
    print("E value:", eigv[eigv_ind])

    # plot eigenstates
    # curr_eigvec = scale_eigenstates(eigvec[:, eigv_ind], point_vec, p0)
    curr_eigvec = eigvec[:, eigv_ind]
   
    # 1. Get the eigen basis
    beta_list = calculate_beta_given_E(eigv[eigv_ind])

    plt.figure()
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, '.', markersize=8)
    plt.xlabel("$E$ real")
    plt.ylabel("$E$ imag")
    plt.show()

    # 2. Decompose
    n_picked = 4
    non_Bloch_basis = generate_non_Bloch_basis(
        beta_list[:n_picked],
        np.array([
            [1,0],
            [1,1]
        ]), point_vec, p0, r_vec=R_VEC
    )

    # 3. Calculate
    curr_eigvec_mesh = curr_eigvec.reshape(shape, order='F')
    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    lower_slice = curr_eigvec_mesh[:,0]
    upper_slice = curr_eigvec_mesh[:,-1]

    lower_point_max = np.argmax(np.abs(lower_slice))
    upper_point_max = np.argmax(np.abs(upper_slice))

    # plt.pcolor(
    #     x_mesh[:,:ny_boundary],
    #     y_mesh[:,:ny_boundary],
    #     np.abs(lower_slice)
    # )
    # plt.plot(
    #     x_mesh[lower_point_max, 0],
    #     y_mesh[lower_point_max, 0],
    #     'rx'
    # )
    # plt.pcolor(
    #     x_mesh[:,-ny_boundary:],
    #     y_mesh[:,-ny_boundary:],
    #     np.abs(upper_slice)
    # )
    # plt.plot(
    #     x_mesh[upper_point_max, -1],
    #     y_mesh[upper_point_max, -1],
    #     'rx'
    # )
    # plt.show()
    # return
    
    nx_selected = 2000
    ny_boundary = 10
    upper_range = ((point_vec[:,0] > (upper_point_max - nx_selected / 2)) 
                   & (point_vec[:,0] < (upper_point_max + nx_selected / 2))
                   & (point_vec[:,1] > (point_vec[:,0] + shape[1] - ny_boundary)))
    lower_range = ((point_vec[:,0] > (lower_point_max - nx_selected / 2)) 
                   & (point_vec[:,0] < (lower_point_max + nx_selected / 2))
                   & (point_vec[:,1] < (point_vec[:,0] + ny_boundary)))

    selected_range =  upper_range | lower_range
    plt.plot(point_vec[:,0], point_vec[:,1], '.')
    plt.plot(point_vec[selected_range,0], point_vec[selected_range,1], '.')
    plt.show()
    res = la.lstsq(
        non_Bloch_basis[selected_range,:],
        curr_eigvec[selected_range]
    )
    coeffs = res[0]
    print(res)

    # scaling to the maximum
    if abs(beta_list[0][1]) > 1:
        upper_scaling_fac = abs(curr_eigvec_mesh[upper_point_max, -1]) / abs(
            (non_Bloch_basis[:, :2] @ coeffs[:2]).reshape(shape, order='F')[upper_point_max,-1]
        )
        lower_scaling_fac = abs(curr_eigvec_mesh[lower_point_max, 0]) / abs(
            (non_Bloch_basis[:, 2:] @ coeffs[2:]).reshape(shape, order='F')[lower_point_max,0]
        )
        coeffs[:2] *= upper_scaling_fac
        coeffs[2:] *= lower_scaling_fac
    else:
        upper_scaling_fac = abs(curr_eigvec_mesh[upper_point_max, -1]) / abs(
            (non_Bloch_basis[:, 2:] @ coeffs[2:]).reshape(shape, order='F')[upper_point_max,-1]
        )
        lower_scaling_fac = abs(curr_eigvec_mesh[lower_point_max, 0]) / abs(
            (non_Bloch_basis[:, :2] @ coeffs[:2]).reshape(shape, order='F')[lower_point_max,0]
        )
        coeffs[2:] *= upper_scaling_fac
        coeffs[:2] *= lower_scaling_fac

    fig = plt.figure()
    plt.axis("equal")
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(curr_eigvec)),
        vmin=-max(np.abs(curr_eigvec))
    )
    for j in range(n_picked):
        fig = plt.figure()
        plt.axis("equal")
        plt.pcolor(
            x_mesh,
            y_mesh,
            # np.abs(non_Bloch_basis[:,j]).reshape(shape, order='F'),
            (non_Bloch_basis[:,j].real).reshape(shape, order='F'),
            cmap="bwr",
            vmax=max(np.abs(non_Bloch_basis[:,j])),
            vmin=-max(np.abs(non_Bloch_basis[:,j]))
        )
    
    plt.figure()
    plt.axis("equal")
    # approx_eigvec = non_Bloch_basis[:,:4] @ coeffs[:4]
    approx_eigvec = non_Bloch_basis @ coeffs
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(approx_eigvec).reshape(shape, order='F'),
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(approx_eigvec)),
        vmin=-max(np.abs(approx_eigvec))
    )

    plt.figure()
    plt.axis("equal")
    diff_vector = curr_eigvec - approx_eigvec
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(approx_eigvec).reshape(shape, order='F'),
        (diff_vector.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(approx_eigvec)),
        vmin=-max(np.abs(approx_eigvec))
    )

    plt.show()

    plt.figure()
    plt.plot(x_mesh[:,0], y_mesh[:,0])
    plt.plot(x_mesh[:,-1], y_mesh[:,-1])
    plt.figure()
    plt.plot(curr_eigvec.reshape(shape, order='F')[:,0].real)
    plt.plot(approx_eigvec.reshape(shape, order='F')[:,0].real)
    plt.figure()
    plt.plot(curr_eigvec.reshape(shape, order='F')[:,-1].real)
    plt.plot(approx_eigvec.reshape(shape, order='F')[:,-1].real)
    plt.show()

    with open("data/paper-eigenstate-compare-direct-%d-%d.pkl"%(shape[0], shape[1]), "wb") as fp:
        pickle.dump(
            (beta_list, eigv_ind, non_Bloch_basis, coeffs, aspect_or_N1, N_or_N2),
            fp
        )


def decompose_under_quasi_1D(E, aspect_or_N1, N_or_N2):
    ''' Decompose the eigenstate under the quasi-1D basis '''
    if isinstance(aspect_or_N1, float):
        aspect = aspect_or_N1
        N = N_or_N2
        with open(
            "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
            'rb'
        ) as fp:
            eigv, eigvec, point_vec, params, shape = pickle.load(fp)
    else:
        N1 = aspect_or_N1
        N2 = N_or_N2
        with open(
            "data/paper-HN-OBC-2-N1_%d-N2_%d.pkl" % (N1, N2),
            'rb'
        ) as fp:
            eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    p0 = (shape[0] / 2, shape[1] / 2)

    eigv_ind = np.argmin(np.abs(eigv - E))
    GBZ_pair = get_nearest_theoretical_solution(eigv[eigv_ind], shape)
    GBZ_energy, GBZ_beta1, GBZ_beta2 = GBZ_pair
    print("E index:", eigv_ind)
    print("E value:", eigv[eigv_ind])
    print("E GBZ:", GBZ_energy)

    # plot eigenstates
    # curr_eigvec = scale_eigenstates(eigvec[:, eigv_ind], point_vec, p0)
    curr_eigvec = eigvec[:, eigv_ind]
   
    # 1. Get the eigen basis
    all_beta1 = get_all_non_Bloch_theoretical(GBZ_energy, shape[1])

    beta1_order = np.argsort(np.abs(np.abs(all_beta1) - R))
    new_beta1 = all_beta1[beta1_order]

    plt.figure()
    plt.plot(new_beta1.real, new_beta1.imag, '.')
    theta = np.linspace(0, 2 * np.pi, 500)
    plt.plot(R * np.cos(theta), R * np.sin(theta))
    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("$\\beta_1$ real")
    plt.ylabel("$\\beta_1$ imag")

    plt.figure()
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, '.', markersize=8)
    plt.xlabel("$E$ real")
    plt.ylabel("$E$ imag")
    plt.show()

    beta_list = []
    for curr_beta1 in new_beta1:
        beta2_list = get_minor_non_Bloch(GBZ_energy, curr_beta1)
        beta_list += [(curr_beta1, curr_beta2) for curr_beta2 in beta2_list]
        
    #     print(np.abs(beta2_list), 
    #           np.sqrt(np.abs((J_X2 * curr_beta1 + J_Y1) / (J_X1 / curr_beta1 + J_Y2))))
    #     input()

    # 2. Decompose
    n_picked = 4
    non_Bloch_basis = generate_non_Bloch_basis(
        beta_list[:n_picked],
        np.array([
            [1,0],
            [1,1]
        ]), point_vec, p0, r_vec=R_VEC
    )

    # 3. Calculate
    curr_eigvec_mesh = curr_eigvec.reshape(shape, order='F')
    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    lower_slice = curr_eigvec_mesh[:,0]
    upper_slice = curr_eigvec_mesh[:,-1]

    lower_point_max = np.argmax(np.abs(lower_slice))
    upper_point_max = np.argmax(np.abs(upper_slice))

    # plt.pcolor(
    #     x_mesh[:,:ny_boundary],
    #     y_mesh[:,:ny_boundary],
    #     np.abs(lower_slice)
    # )
    # plt.plot(
    #     x_mesh[lower_point_max, 0],
    #     y_mesh[lower_point_max, 0],
    #     'rx'
    # )
    # plt.pcolor(
    #     x_mesh[:,-ny_boundary:],
    #     y_mesh[:,-ny_boundary:],
    #     np.abs(upper_slice)
    # )
    # plt.plot(
    #     x_mesh[upper_point_max, -1],
    #     y_mesh[upper_point_max, -1],
    #     'rx'
    # )
    # plt.show()
    # return
    
    nx_selected = 2000
    ny_boundary = 10
    upper_range = ((point_vec[:,0] > (upper_point_max - nx_selected / 2)) 
                   & (point_vec[:,0] < (upper_point_max + nx_selected / 2))
                   & (point_vec[:,1] > (point_vec[:,0] + shape[1] - ny_boundary)))
    lower_range = ((point_vec[:,0] > (lower_point_max - nx_selected / 2)) 
                   & (point_vec[:,0] < (lower_point_max + nx_selected / 2))
                   & (point_vec[:,1] < (point_vec[:,0] + ny_boundary)))

    selected_range =  upper_range | lower_range
    plt.plot(point_vec[:,0], point_vec[:,1], '.')
    plt.plot(point_vec[selected_range,0], point_vec[selected_range,1], '.')
    plt.show()
    res = la.lstsq(
        non_Bloch_basis[selected_range,:],
        curr_eigvec[selected_range]
    )
    coeffs = res[0]
    print(res)

    # scaling to the maximum
    if abs(beta_list[0][1]) > 1:
        upper_scaling_fac = abs(curr_eigvec_mesh[upper_point_max, -1]) / abs(
            (non_Bloch_basis[:, :2] @ coeffs[:2]).reshape(shape, order='F')[upper_point_max,-1]
        )
        lower_scaling_fac = abs(curr_eigvec_mesh[lower_point_max, 0]) / abs(
            (non_Bloch_basis[:, 2:] @ coeffs[2:]).reshape(shape, order='F')[lower_point_max,0]
        )
        coeffs[:2] *= upper_scaling_fac
        coeffs[2:] *= lower_scaling_fac
    else:
        upper_scaling_fac = abs(curr_eigvec_mesh[upper_point_max, -1]) / abs(
            (non_Bloch_basis[:, 2:] @ coeffs[2:]).reshape(shape, order='F')[upper_point_max,-1]
        )
        lower_scaling_fac = abs(curr_eigvec_mesh[lower_point_max, 0]) / abs(
            (non_Bloch_basis[:, :2] @ coeffs[:2]).reshape(shape, order='F')[lower_point_max,0]
        )
        coeffs[2:] *= upper_scaling_fac
        coeffs[:2] *= lower_scaling_fac

    fig = plt.figure()
    plt.axis("equal")
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(curr_eigvec)),
        vmin=-max(np.abs(curr_eigvec))
    )
    for j in range(n_picked):
        fig = plt.figure()
        plt.axis("equal")
        plt.pcolor(
            x_mesh,
            y_mesh,
            # np.abs(non_Bloch_basis[:,j]).reshape(shape, order='F'),
            (non_Bloch_basis[:,j].real).reshape(shape, order='F'),
            cmap="bwr",
            vmax=max(np.abs(non_Bloch_basis[:,j])),
            vmin=-max(np.abs(non_Bloch_basis[:,j]))
        )
    
    plt.figure()
    plt.axis("equal")
    # approx_eigvec = non_Bloch_basis[:,:4] @ coeffs[:4]
    approx_eigvec = non_Bloch_basis @ coeffs
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(approx_eigvec).reshape(shape, order='F'),
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(approx_eigvec)),
        vmin=-max(np.abs(approx_eigvec))
    )

    plt.figure()
    plt.axis("equal")
    diff_vector = curr_eigvec - approx_eigvec
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(approx_eigvec).reshape(shape, order='F'),
        (diff_vector.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(approx_eigvec)),
        vmin=-max(np.abs(approx_eigvec))
    )

    plt.show()

    plt.figure()
    plt.plot(x_mesh[:,0], y_mesh[:,0])
    plt.plot(x_mesh[:,-1], y_mesh[:,-1])
    plt.figure()
    plt.plot(curr_eigvec.reshape(shape, order='F')[:,0].real)
    plt.plot(approx_eigvec.reshape(shape, order='F')[:,0].real)
    plt.figure()
    plt.plot(curr_eigvec.reshape(shape, order='F')[:,-1].real)
    plt.plot(approx_eigvec.reshape(shape, order='F')[:,-1].real)
    plt.show()

    with open("data/paper-eigenstate-compare-%d-%d.pkl"%(shape[0], shape[1]), "wb") as fp:
        pickle.dump(
            (beta_list, eigv_ind, GBZ_energy, non_Bloch_basis, coeffs, aspect_or_N1, N_or_N2),
            fp
        )


def decompose_under_quasi_1D_fft(E, aspect, N):
    ''' Decompose the eigenstate under the quasi-1D basis, where beta1 is calculated by fft '''
    with open(
        "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    p0 = (shape[0] / 2, shape[1] / 2)

    eigv_ind = np.argmin(np.abs(eigv - E))
    curr_eigvec = eigvec[:, eigv_ind]

    # fft
    upper_edge_ind = (point_vec[:,1] - point_vec[:,0] > shape[1] - 1 - 1e-6)
    lower_edge_ind = (point_vec[:,1] - point_vec[:,0] < 1e-6)

    def get_fft(x_mode: np.ndarray):
        mode_edge_fft = fft.fft(x_mode)
        max_ind = np.argmax(np.abs(mode_edge_fft))
        k_max = 2 * np.pi * max_ind / len(mode_edge_fft)
        return k_max
    
    k_lower = get_fft(curr_eigvec[lower_edge_ind])
    k_upper = get_fft(curr_eigvec[upper_edge_ind])

    GBZ_pair1 = get_nearest_theoretical_with_beta1(E, [R * np.exp(1j * k_lower)], shape)
    GBZ_pair2 = get_nearest_theoretical_with_beta1(E, [R * np.exp(1j * k_upper)], shape)
    # print(GBZ_pair1)
    # print(GBZ_pair2)
    # GBZ_energy, GBZ_beta1, GBZ_beta2 = GBZ_pair
    GBZ_energy = (GBZ_pair1[0] + GBZ_pair2[0]) / 2

    print("E index:", eigv_ind)
    print("E value:", eigv[eigv_ind])
    print("E GBZ:", GBZ_energy)
   
    # 1. Get the eigen basis
    all_beta1 = get_all_non_Bloch_theoretical(GBZ_energy, shape[1])

    beta1_order = np.argsort(np.abs(np.abs(all_beta1) - R))
    new_beta1 = all_beta1[beta1_order]

    plt.figure()
    plt.plot(new_beta1.real, new_beta1.imag, '.')
    theta = np.linspace(0, 2 * np.pi, 500)
    plt.plot(R * np.cos(theta), R * np.sin(theta))
    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("$\\beta_1$ real")
    plt.ylabel("$\\beta_1$ imag")

    plt.figure()
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, '.', markersize=8)
    plt.xlabel("$E$ real")
    plt.ylabel("$E$ imag")
    plt.show()

    beta_list = []
    for curr_beta1 in new_beta1:
        beta2_list = get_minor_non_Bloch(GBZ_energy, curr_beta1)
        beta_list += [(curr_beta1, curr_beta2) for curr_beta2 in beta2_list]
        
    #     print(np.abs(beta2_list), 
    #           np.sqrt(np.abs((J_X2 * curr_beta1 + J_Y1) / (J_X1 / curr_beta1 + J_Y2))))
    #     input()

    # 2. Decompose
    n_picked = 4
    non_Bloch_basis = generate_non_Bloch_basis(
        beta_list[:n_picked],
        np.array([
            [1,0],
            [1,1]
        ]), point_vec, p0, r_vec=R_VEC
    )

    # 3. Calculate
    n_boundary = 50
    selected_range = (point_vec[:,0] > n_boundary) & (point_vec[:,0] < shape[0] - n_boundary) 
    plt.plot(point_vec[:,0], point_vec[:,1], '.')
    plt.plot(point_vec[selected_range,0], point_vec[selected_range,1], '.')
    plt.show()
    res = la.lstsq(
        non_Bloch_basis[selected_range,:],
        curr_eigvec[selected_range]
    )
    coeffs = res[0]
    print(res)

    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    fig = plt.figure()
    plt.axis("equal")
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(curr_eigvec)),
        vmin=-max(np.abs(curr_eigvec))
    )
    for j in range(n_picked):
        fig = plt.figure()
        plt.axis("equal")
        plt.pcolor(
            x_mesh,
            y_mesh,
            # np.abs(non_Bloch_basis[:,j]).reshape(shape, order='F'),
            (non_Bloch_basis[:,j].real).reshape(shape, order='F'),
            cmap="bwr",
            vmax=max(np.abs(non_Bloch_basis[:,j])),
            vmin=-max(np.abs(non_Bloch_basis[:,j]))
        )
    
    plt.figure()
    plt.axis("equal")
    # approx_eigvec = non_Bloch_basis[:,:4] @ coeffs[:4]
    approx_eigvec = non_Bloch_basis @ coeffs
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(approx_eigvec).reshape(shape, order='F'),
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(approx_eigvec)),
        vmin=-max(np.abs(approx_eigvec))
    )

    plt.figure()
    plt.axis("equal")
    diff_vector = curr_eigvec - approx_eigvec
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(approx_eigvec).reshape(shape, order='F'),
        (diff_vector.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(approx_eigvec)),
        vmin=-max(np.abs(approx_eigvec))
    )

    plt.show()


def decompose_under_quasi_1D_fft_full_fit(E, aspect, N):
    ''' Decompose the eigenstate under the quasi-1D basis, where beta1 is calculated by fft,
        and GBZ energy is estimated by the nearest point when beta2 runs from -pi to pi 
    '''
    with open(
        "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    p0 = (shape[0] / 2, shape[1] / 2)

    eigv_ind = np.argmin(np.abs(eigv - E))
    curr_eigvec = eigvec[:, eigv_ind]

    # fft
    upper_edge_ind = (point_vec[:,1] - point_vec[:,0] > shape[1] - 1 - 1e-6)
    lower_edge_ind = (point_vec[:,1] - point_vec[:,0] < 1e-6)

    def get_fft(x_mode: np.ndarray):
        mode_edge_fft = fft.fft(x_mode)
        max_ind = np.argmax(np.abs(mode_edge_fft))
        k_max = 2 * np.pi * max_ind / len(mode_edge_fft)
        return k_max
    
    k_lower = get_fft(curr_eigvec[lower_edge_ind])
    k_upper = get_fft(curr_eigvec[upper_edge_ind])

    # get upper GBZ energy
    beta1 = R * np.exp(1j * k_upper)
    spec_bound = 2 * np.sqrt((J_X1 / beta1 + J_Y2) * (J_X2 * beta1 + J_Y1))
    GBZ_energy_1 = (eigv[eigv_ind].real * spec_bound.real + eigv[eigv_ind].imag * spec_bound.imag) / (abs(spec_bound) ** 2) * spec_bound

    # get lower GBZ energy
    beta1 = R * np.exp(1j * k_lower)
    spec_bound = 2 * np.sqrt((J_X1 / beta1 + J_Y2) * (J_X2 * beta1 + J_Y1))
    GBZ_energy_2 = (eigv[eigv_ind].real * spec_bound.real + eigv[eigv_ind].imag * spec_bound.imag) / (abs(spec_bound) ** 2) * spec_bound

    GBZ_energy = (GBZ_energy_1 + GBZ_energy_2) / 2

    print("E index:", eigv_ind)
    print("E value:", eigv[eigv_ind])
    print("E GBZ:", GBZ_energy)
   
    # 1. Get the eigen basis
    all_beta1 = get_all_non_Bloch_theoretical(GBZ_energy, shape[1])

    beta1_order = np.argsort(np.abs(np.abs(all_beta1) - R))
    new_beta1 = all_beta1[beta1_order]

    plt.figure()
    plt.plot(new_beta1.real, new_beta1.imag, '.')
    theta = np.linspace(0, 2 * np.pi, 500)
    plt.plot(R * np.cos(theta), R * np.sin(theta))
    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("$\\beta_1$ real")
    plt.ylabel("$\\beta_1$ imag")

    plt.figure()
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, '.', markersize=8)
    plt.xlabel("$E$ real")
    plt.ylabel("$E$ imag")
    plt.show()

    # beta_list = []
    # for curr_beta1 in new_beta1:
    #     beta2_list = get_minor_non_Bloch(GBZ_energy, curr_beta1)
    #     beta_list += [(curr_beta1, curr_beta2) for curr_beta2 in beta2_list]
    beta_list = calculate_beta_given_E(GBZ_energy)
        
    #     print(np.abs(beta2_list), 
    #           np.sqrt(np.abs((J_X2 * curr_beta1 + J_Y1) / (J_X1 / curr_beta1 + J_Y2))))
    #     input()

    # 2. Decompose
    n_picked = 4
    non_Bloch_basis = generate_non_Bloch_basis(
        beta_list[:n_picked],
        np.array([
            [1,0],
            [1,1]
        ]), point_vec, p0, r_vec=R_VEC
    )

    # 3. Calculate
    n_boundary = 120
    selected_range = (point_vec[:,0] > n_boundary) & (point_vec[:,0] < shape[0] - n_boundary) 
    plt.plot(point_vec[:,0], point_vec[:,1], '.')
    plt.plot(point_vec[selected_range,0], point_vec[selected_range,1], '.')
    plt.show()
    res = la.lstsq(
        non_Bloch_basis[selected_range,:],
        curr_eigvec[selected_range]
    )
    coeffs = res[0]
    print(res)

    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    fig = plt.figure()
    plt.axis("equal")
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(curr_eigvec)),
        vmin=-max(np.abs(curr_eigvec))
    )
    for j in range(n_picked):
        fig = plt.figure()
        plt.axis("equal")
        plt.pcolor(
            x_mesh,
            y_mesh,
            # np.abs(non_Bloch_basis[:,j]).reshape(shape, order='F'),
            (non_Bloch_basis[:,j].real).reshape(shape, order='F'),
            cmap="bwr",
            vmax=max(np.abs(non_Bloch_basis[:,j])),
            vmin=-max(np.abs(non_Bloch_basis[:,j]))
        )
    
    plt.figure()
    plt.axis("equal")
    # approx_eigvec = non_Bloch_basis[:,:4] @ coeffs[:4]
    approx_eigvec = non_Bloch_basis @ coeffs
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(approx_eigvec).reshape(shape, order='F'),
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(approx_eigvec)),
        vmin=-max(np.abs(approx_eigvec))
    )

    plt.figure()
    plt.axis("equal")
    diff_vector = curr_eigvec - approx_eigvec
    plt.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(approx_eigvec).reshape(shape, order='F'),
        (diff_vector.real).reshape(shape, order='F'),
        cmap="bwr",
        vmax=max(np.abs(approx_eigvec)),
        vmin=-max(np.abs(approx_eigvec))
    )

    # plt.show()

    # print(x_mesh)
    # print(y_mesh)
    plt.figure()
    plt.plot(x_mesh[:,0], y_mesh[:,0])
    plt.plot(x_mesh[:,-1], y_mesh[:,-1])
    plt.figure()
    plt.plot(curr_eigvec.reshape(shape, order='F')[:,0].real)
    plt.plot(approx_eigvec.reshape(shape, order='F')[:,0].real)
    plt.figure()
    plt.plot(curr_eigvec.reshape(shape, order='F')[:,-1].real)
    plt.plot(approx_eigvec.reshape(shape, order='F')[:,-1].real)
    plt.show()


def get_beta1(E, phi):
    ''' Get beta1 for given relative phase phi '''
    poly_eq = [
        J_X2 * J_Y2,
        J_X1 * J_X2 + J_Y1 * J_Y2 - (E ** 2) / (4 * np.cos(phi/2) ** 2),
        J_X1 * J_Y1
    ]
    return np.roots(poly_eq)


def test_beta1_solution():
    E = np.random.randn() + 1j * np.random.randn()
    for phi in np.linspace(0, 2 * np.pi, 30):
        print("phi = ", phi / np.pi, "pi")
        beta1_list = get_beta1(E, phi)
        # print(np.sqrt(np.abs((J_X2 * beta1_list + J_Y1) / (J_X1 / beta1_list + J_Y2))) / R_VEC[1])
        # print(np.abs(beta1_list), R)
        # print(np.prod(np.sqrt(np.abs((J_X2 * beta1_list + J_Y1) / (J_X1 / beta1_list + J_Y2))) / R_VEC[1]))
        # input()
        # continue
        for curr_beta1 in beta1_list:
            beta2_list = get_minor_non_Bloch(E, curr_beta1)
            print("beta2 phi = ", np.log(beta2_list[1] / beta2_list[0]).imag / np.pi, "pi")
            print("beta2 norm = ", np.abs(beta2_list))
            print(np.sqrt(np.abs((J_X2 * curr_beta1 + J_Y1) / (J_X1 / curr_beta1 + J_Y2))))
            for curr_beta2 in beta2_list:
                print(E - J_X1 * curr_beta2 / curr_beta1 - J_X2 * curr_beta1 / curr_beta2 
                      - J_Y1 / curr_beta2 - J_Y2 * curr_beta2)
        input()


def test_beta2_norm():
    k1 = np.random.rand() * 2 * np.pi
    beta1 = np.array([
        R * np.exp(1j * (k1 - THETA_2)),
        R * np.exp(1j * (- k1 - THETA_2))
    ])
    print(np.sqrt(np.abs((J_X2 * beta1 + J_Y1) / (J_X1 / beta1 + J_Y2))) / R_VEC[1])
    print(np.abs(beta1), R)
    print(np.prod(np.sqrt(np.abs((J_X2 * beta1 + J_Y1) / (J_X1 / beta1 + J_Y2))) / R_VEC[1]))


def get_all_non_Bloch_theoretical(E, N):
    ''' Get theoretical non-Bloch solutions for E '''
    # 1. Get all phi
    # phi = 2nπ/(N+1), n = 1, 2, 3, ..., N // 2
    all_beta1 = []
    for n in range(1, (N + 2)//2):
        curr_phi = (n * 2 * np.pi / (N + 1)) % (np.pi)
        all_beta1 += list(get_beta1(E, curr_phi))
    
    return np.asarray(all_beta1)


def compare_quasi_1D():
    ''' Get quasi 1D spectrum and compare with the numerical solution '''
    theta = np.linspace(0, 2 * np.pi, 300)
    beta1 = R * np.exp(1j * theta)
    beta_ellipse = 4 * (J_X1 / beta1 + J_Y2) * (J_X2 * beta1 + J_Y1)
    N = 15
    for n in range(1, N + 1):
        kappa = (n * np.pi / (N + 1))
        E = np.sqrt(beta_ellipse) * np.cos(kappa)
        # plt.plot(E.real, E.imag, 'b-')

    with open("data/paper-quasi-1D-y-N_%d.pkl"%(N), "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)
        # E, beta, params, N = pickle.load(fp)
    # plt.plot(E.real, E.imag, '.')
    plt.plot(beta.real, beta.imag, '.')
    plt.show()


def get_nearest_theoretical_solution(E_target, shape):
    ''' Get theoretical solution for a region with given shape 
        with nearest energy E 
    '''
    # 1. Generate the energy mesh
    all_strip_GBZ = []
    for n1 in range(1, shape[0] + 1):
        for beta1 in [
            R * np.exp(-1j * THETA_2) * np.exp(1j * n1 * np.pi / (shape[0] + 1)),
            R * np.exp(-1j * THETA_2) * np.exp(- 1j * n1 * np.pi / (shape[0] + 1))
        ]:
            for n2 in range(1, shape[1] + 1):
                kappa = (n2 * np.pi / (shape[1] + 1))
                beta_ellipse = 4 * (J_X1 / beta1 + J_Y2) * (J_X2 * beta1 + J_Y1)
                E = np.sqrt(beta_ellipse) * np.cos(kappa)
                beta2_list = get_minor_non_Bloch(E, beta1)
                for beta2 in beta2_list:
                    all_strip_GBZ.append((E, beta1, beta2))
    
    E_list = np.array([pair[0] for pair in all_strip_GBZ])

    nearest_ind = np.argmin(np.abs(E_list - E_target))
    return all_strip_GBZ[nearest_ind]


def get_nearest_theoretical_with_beta1(E_target, beta1_list, shape):
    all_strip_GBZ = []
    for beta1 in beta1_list:
        for n2 in range(1, shape[1] + 1):
            kappa = (n2 * np.pi / (shape[1] + 1))
            beta_ellipse = 4 * (J_X1 / beta1 + J_Y2) * (J_X2 * beta1 + J_Y1)
            E = np.sqrt(beta_ellipse) * np.cos(kappa)
            beta2_list = get_minor_non_Bloch(E, beta1)
            for beta2 in beta2_list:
                all_strip_GBZ.append((E, beta1, beta2))
    E_list = np.array([pair[0] for pair in all_strip_GBZ])

    nearest_ind = np.argmin(np.abs(E_list - E_target))
    return all_strip_GBZ[nearest_ind]
    

def compare_norm_diff_major_beta():
    ''' Calculate the derivative of beta1 and the difference of the abs of quasi-1D beta '''
    target_energy = 1 + 0.8j
    l2 = 300
    all_beta_theory = np.asarray(get_all_non_Bloch_theoretical(target_energy, l2))
    # 1. Find GBZ beta
    min_ind = np.argmin(np.abs(np.abs(all_beta_theory) - R))
    GBZ_beta1 = all_beta_theory[min_ind]
    minor_beta = get_minor_non_Bloch(target_energy, GBZ_beta1)

    # 2. Find nearest several points
    nearest_ind = np.argsort(np.abs(all_beta_theory - GBZ_beta1))
    nearest_points = all_beta_theory[nearest_ind]

    # 3. Calculate derivative
    alpha = np.abs(1 / calculate_characteristic_aspect(target_energy, GBZ_beta1))

    # 5. plot
    plt.plot(np.abs(
        (np.abs(nearest_points[:10]) - np.abs(GBZ_beta1)) / np.abs(GBZ_beta1)
    ), '.')
    plt.plot([0, 10], alpha / l2 * np.ones(2))
    plt.show()


def calculate_beta1_derivative(reference_energy: complex, beta1: complex):
    ''' Calculate the derivative of beta1 against phi when E is fixed '''
    char_poly_info = pGs.CharPolyInfo(CHP11)
    minor_beta = get_minor_non_Bloch(reference_energy, beta1)
    dE, dbeta1 = char_poly_info.get_E_and_beta_derivative(
        reference_energy, beta1, minor_beta[0], minor_beta[1]
    )
    return dbeta1


def calculate_characteristic_aspect(
    reference_energy: complex, beta1: complex
):
    ''' Calculate 1/α, where α = 2π Re(β* dβ/dφ) / |β|^2'''
    dbeta = calculate_beta1_derivative(reference_energy, beta1)
    alpha = (
        2 * np.pi / (beta1 * beta1.conjugate()).real 
        * (beta1.conjugate() * dbeta).real
    )
    return 1 / alpha


def test_derivative():
    target_energy = 1 + 1j

    coeffs, degs = MODEL11.get_characteristic_polynomial_data()
    f = pt.CLaurent(3)
    f.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    char_poly_info = pGs.CharPolyInfo(f)

    phi = np.random.rand() * np.pi
    GBZ_beta1 = get_beta1(target_energy, phi)[0]
    minor_beta = get_minor_non_Bloch(target_energy, GBZ_beta1)
    print(phi / np.pi)
    print(np.log(minor_beta[1] / minor_beta[0]) / np.pi)

    dphi = np.random.randn() * 1e-2
    GBZ_beta2 = get_beta1(target_energy, phi + dphi)[0]
    dE, dbeta = char_poly_info.get_E_and_beta_derivative(
        target_energy, GBZ_beta1, minor_beta[0], minor_beta[1]
    )

    print(GBZ_beta2, GBZ_beta1)
    print(GBZ_beta2 - GBZ_beta1)
    print(dbeta * dphi)


def plot_aspect_distribution():
    ''' 越靠近那个月牙的边缘，beta之差越下降 '''
    k1 = np.linspace(-np.pi, np.pi, 100)
    k2 = np.linspace(-np.pi, np.pi, 100)
    k1_mesh, k2_mesh = np.meshgrid(k1, k2)
    beta1_mesh = R * np.exp(1j * k1_mesh)
    beta2_norm = np.sqrt(
        np.abs(J_X2 * beta1_mesh + J_Y1) 
        /  np.abs(J_X1 / beta1_mesh + J_Y2)
    )
    beta2_mesh = beta2_norm * np.exp(1j * k2_mesh)

    eigen_energy = (
        J_X1 * beta2_mesh / beta1_mesh
        + J_X2 * beta1_mesh / beta2_mesh
        + J_Y1 / beta2_mesh
        + J_Y2 * beta2_mesh
    )

    # Calculate minimal aspect
    aspect_mesh = np.zeros_like(k1_mesh)
    for row_ind in range(k1_mesh.shape[0]):
        for col_ind in range(k1_mesh.shape[1]):
            aspect_mesh[row_ind,col_ind] = np.abs(
                calculate_characteristic_aspect(
                    eigen_energy[row_ind,col_ind], beta1_mesh[row_ind,col_ind]
                )
            )
    
    # cmap = plt.cm.viridis
    # cmap_norm = plt.Normalize(0, 10)
    # get_color = lambda v: cmap(cmap_norm(v))
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for row_ind in range(k1_mesh.shape[0]):
    #     for col_ind in range(k1_mesh.shape[1]):
    #         ax.plot(
    #             eigen_energy[row_ind,col_ind].real,
    #             eigen_energy[row_ind,col_ind].imag,
    #             aspect_mesh[row_ind,col_ind],
    #             '.',
    #             color=get_color(aspect_mesh[row_ind,col_ind])
    #         )
    
    # ax.set_zlim([0,10])

    # # pcl = plt.pcolor(eigen_energy.real, eigen_energy.imag, aspect_mesh,
    # #            shading='interp', vmin=0, vmax=16)
    # # plt.colorbar(pcl)
    # plt.show()

    # use pyvista
    pl = pv.Plotter()
    all_points = np.column_stack([
        eigen_energy.real.flatten(),
        eigen_energy.imag.flatten(),
        aspect_mesh.flatten()
    ])
    selected_points = all_points[:,2] < 10
    all_points = all_points[selected_points,:]
    pts = pl.add_points(
        all_points,
        scalars=all_points[:,2],
        clim=[0, 10]
    )
    pl.show_grid(color='black')
    pl.show()


def calculate_modes_at_edge():
    ''' Calculate the modes at edge to fit the wavevector
        the mode is calculated by averaging the fields at the edge
    '''
    aspect = 1/8
    N = 40
    with open(
        "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)
    
    E_target = 1 + 1j
    # eigv_ind = np.random.randint(len(eigv))
    eigv_ind = np.argmin(np.abs(eigv - E_target))
    eigvec_selected = eigvec[:,eigv_ind]

    # plot selected eigenenergy
    plt.figure()
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, 'x')
    
    # plot eigvec
    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    plt.figure()
    plt.pcolor(
        x_mesh,
        y_mesh,
        (eigvec_selected.real).reshape(shape, order='F'),
        cmap='bwr',
        vmax=np.max(np.abs(eigvec_selected)),
        vmin=-np.max(np.abs(eigvec_selected))
    )

    # Get modes at edge
    x_range = 0
    x_filter = (point_vec[:,0] > x_range) & (point_vec[:,0] < shape[0] - x_range)
    upper_edge_ind = (point_vec[:,1] - point_vec[:,0] > shape[1] - 1 - 1e-6) & x_filter
    lower_edge_ind = (point_vec[:,1] - point_vec[:,0] < 1e-6) & x_filter
    edge_ind = upper_edge_ind
    x_line = point_vec[edge_ind,0]
    mode_edge = eigvec_selected[edge_ind]
    plt.figure()
    plt.plot(x_line)
    plt.show()

    # fft
    plt.figure()
    mode_edge_fft = fft.fft(mode_edge)
    plt.plot(np.arange(len(mode_edge_fft)) * 2 * np.pi / len(mode_edge_fft), np.abs(mode_edge_fft))
    max_ind = np.argmax(np.abs(mode_edge_fft))
    k_max = 2 * np.pi * max_ind / len(mode_edge_fft)
    f_max = mode_edge_fft[max_ind] / len(mode_edge_fft)

    # plot fft data
    plt.figure()
    plt.plot(x_line, mode_edge.real)
    plt.plot(x_line, (f_max * np.exp(1j * k_max * x_line)).real)

    plt.show()

    # plot E on the Strip GBZ spectrum
    beta1 = R * np.exp(1j * k_max)
    spec_bound = 2 * np.sqrt((J_X1 / beta1 + J_Y2) * (J_X2 * beta1 + J_Y1))
    plt.plot([-spec_bound.real, spec_bound.real], [-spec_bound.imag, spec_bound.imag])
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, '.')
    plt.show()


def check_fft_and_spectrum():
    aspect = 1/8
    N = 40
    with open(
        "data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (aspect, N),
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)
    
    # eigv_ind = np.random.randint(len(eigv))
    E_target = (1 + 1j) * 1
    eigv_ind = np.argmin(np.abs(eigv - E_target))
    eigvec_selected = eigvec[:,eigv_ind]

    # plot selected eigenenergy
    plt.figure()
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, 'x')
    
    # plot eigvec
    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    plt.figure()
    plt.pcolor(
        x_mesh,
        y_mesh,
        (eigvec_selected.real).reshape(shape, order='F'),
        cmap='bwr',
        vmax=np.max(np.abs(eigvec_selected)),
        vmin=-np.max(np.abs(eigvec_selected))
    )

    # Get modes at edge
    upper_edge_ind = (point_vec[:,1] - point_vec[:,0] > shape[1] - 1 - 1e-6)
    lower_edge_ind = (point_vec[:,1] - point_vec[:,0] < 1e-6)

    def get_fft(x_mode: np.ndarray):
        mode_edge_fft = fft.fft(x_mode)
        max_ind = np.argmax(np.abs(mode_edge_fft))
        k_max = 2 * np.pi * max_ind / len(mode_edge_fft)
        return k_max

    plt.figure()
    # plot E on the Strip GBZ spectrum
    beta1 = R * np.exp(1j * get_fft(eigvec_selected[upper_edge_ind]))
    spec_bound = 2 * np.sqrt((J_X1 / beta1 + J_Y2) * (J_X2 * beta1 + J_Y1))
    E1_approx = (eigv[eigv_ind].real * spec_bound.real + eigv[eigv_ind].imag * spec_bound.imag) / (abs(spec_bound) ** 2) * spec_bound
    plt.plot([-spec_bound.real, spec_bound.real], [-spec_bound.imag, spec_bound.imag])
    plt.plot(E1_approx.real, E1_approx.imag, '.')
    beta1 = R * np.exp(1j * get_fft(eigvec_selected[lower_edge_ind]))
    spec_bound = 2 * np.sqrt((J_X1 / beta1 + J_Y2) * (J_X2 * beta1 + J_Y1))
    E1_approx = (eigv[eigv_ind].real * spec_bound.real + eigv[eigv_ind].imag * spec_bound.imag) / (abs(spec_bound) ** 2) * spec_bound
    plt.plot([-spec_bound.real, spec_bound.real], [-spec_bound.imag, spec_bound.imag])
    plt.plot(E1_approx.real, E1_approx.imag, '.')
    plt.plot(eigv[eigv_ind].real, eigv[eigv_ind].imag, 'x')
    plt.show()


if __name__ == '__main__':
    # calculate_beta_given_E(1 + 1j)
    # compare_eigenstates(1.35 + 1j, 1, 20)
    # test_strip_GBZ_spectrum()
    # select_best_parameters()
    # test_eigenstates_scale(1.35 + 1j, 1, 20)
    # fit_eigenstate(5.10 + 1.67j, 1/8, 40)
    # plot_all_eigenstates(1/8, 40)
    # check_eigenenergy()
    # check_quasi_1D_roots()
    # for target_energy in np.linspace(5.10 + 1.67j, 4.5 + 1.5j, 5):
    #     decompose_under_quasi_1D(target_energy, 1/4, 40)
    # decompose_under_quasi_1D(5.10 + 1.67j, 1/4, 40)
    # decompose_under_quasi_1D(4.74 + 1.55j, 1/4, 40)
    # decompose_under_quasi_1D(4.5 + 1.5j, 1/8, 40)
    # decompose_under_quasi_1D(1.2 + 1.2j, 1/8, 40)
    # decompose_under_quasi_1D(1.2 + 1.2j, 320, 30)
    # decompose_under_quasi_1D_direct(1.2 + 1.2j, 320, 20)
    # decompose_under_quasi_1D_direct(1.2 + 1.2j, 1/8, 40)
    decompose_under_quasi_1D_direct(1.2 + 1.2j, 320, 50)
    # decompose_under_quasi_1D_fft_full_fit((1 + 1j), 1/8, 40)
    # decompose_under_quasi_1D_fft((5.10 + 1.67j) * 0.9, 1/8, 40)
    # test_beta1_solution()
    # get_all_non_Bloch(-4.5 - 1.5j, 10)
    # compare_quasi_1D()
    # get_nearest_theoretical_solution(0, (320, 40))
    # test_beta2_norm()
    # compare_norm_diff_major_beta()
    # test_derivative()
    # plot_aspect_distribution()
    # calculate_modes_at_edge()
    # check_fft_and_spectrum()
