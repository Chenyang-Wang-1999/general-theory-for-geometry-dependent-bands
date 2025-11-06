'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-01-02
Copyright © Department of Physics, Tsinghua University. All rights reserved

Test winding numbers for different models
'''

from model_test_common import *
import multiprocessing
from math import sin, cos, sqrt, pi
from cmath import exp
import sys
sys.path.append("../paper/HN-model")
import HN_model_common as HN

N_PROCESS = 10

def minimal_nontrivial_model()->tb.TightBindingModel:
    intra_cell = []
    inter_cell = [
        [0, 0, 1, (1,0)],
        [0, 0, 1, (-1,0)],
        [0, 0, 1, (-1, 1)],
        [0, 0, 1, (2, -1)]
    ]

    return tb.TightBindingModel(2, 1,[[1,0], [0,1]], intra_cell, inter_cell, [[0,0]])


def SSH_2D(ux1, ux2, vx1, vx2, uy1, uy2, vy1, vy2):
    dim = 2
    site_num = 2
    lat_vec = np.array([
        [1, 0],
        [0, 1]
    ])
    intra_cell = [
        [1, 0, ux1],
        [0, 1, ux2],
    ]
    inter_cell = [
        [0, 1, vx1, (1, 0)],
        [1, 0, vx2, (-1, 0)],
        [0, 0, uy1, (0, 1)],
        [0, 0, uy2, (0, -1)],
        [1, 1, vy1, (0, 1)],
        [1, 1, vy2, (0, -1)]
    ]

    return tb.TightBindingModel(
        dim, site_num, lat_vec, intra_cell, inter_cell
    )


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
        [1, 1, v1 * exp(1j * phi), (1, 0)],
        [1, 1, v1 * exp(1j * phi), (0, 1)],
        [1, 1, v1 * exp(1j * phi), (-1,-1)]
    ]
    return tb.TightBindingModel(dim, site_num, lattice_vec, intra_cell, inter_cell)


def Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma):
    # v1 = v2 = t2 * exp(i gamma)
    return non_Hermitian_Haldane_H(t1, t1, t2 * exp(1j * gamma), t2 * exp(1j * gamma), phi, M)


def batch_task(vars):
    compare_winding_number(*vars)


def test_minimal_nontrivial_model():
    fname = "data/minimal-nontrivial-test"
    model = minimal_nontrivial_model()
    r_list = np.linspace(0.3, 3, 200)
    n_minor = 40
    n_theta1 = 400
    n_theta1_num = 400
    task_list = [
        (
            fname, model,
            np.random.randn() + 1j * np.random.randn(),
            n_minor,
            n_theta1,
            n_theta1_num,
            r_list,
            None
        )
        for _ in range(10)
    ]

    pool = multiprocessing.Pool(N_PROCESS)
    pool.map(batch_task, task_list)
    # for task_ind, task in enumerate(task_list):
    #     print(task_ind)
    #     batch_task(task)


def test_2D_SSH_model():
    ux1, ux2, vx1, vx2, uy1, uy2, vy1, vy2 = np.random.randn(8)
    # with open("data/2D-SSH-params.pkl", "wb") as fp:
    #     pickle.dump((ux1, ux2, vx1, vx2, uy1, uy2, vy1, vy2), fp)
    fname = "data/2D-SSH-test"
    model = SSH_2D(ux1, ux2, vx1, vx2, uy1, uy2, vy1, vy2)
    r_list = np.linspace(0.3, 3, 200)
    n_minor = 20
    n_theta1 = 400
    n_theta1_num = 400
    task_list = [
        (
            fname, model,
            np.random.randn() + 1j * np.random.randn(),
            n_minor,
            n_theta1,
            n_theta1_num,
            r_list,
            (ux1, ux2, vx1, vx2, uy1, uy2, vy1, vy2)
        )
        for _ in range(10)
    ]

    pool = multiprocessing.Pool(N_PROCESS)
    pool.map(batch_task, task_list)
    # for task_ind, task in enumerate(task_list):
    #     print(task_ind)
    #     batch_task(task)


def test_nH_Haldane():
    t1, t2, gamma, phi, M = np.random.randn(5)
    # with open("data/nH-Haldane-params.pkl", "wb") as fp:
    #     pickle.dump((t1, t2, gamma, phi, M), fp)
    fname = "data/nH-Haldane-test"
    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    r_list = np.linspace(0.3, 3, 200)
    n_minor = 20
    n_theta1 = 400
    n_theta1_num = 400
    task_list = [
        (
            fname, model,
            np.random.randn() + 1j * np.random.randn(),
            n_minor,
            n_theta1,
            n_theta1_num,
            r_list,
            (t1, t2, gamma, phi, M)
        )
        for _ in range(10)
    ]

    pool = multiprocessing.Pool(N_PROCESS)
    pool.map(batch_task, task_list)
    # for task_ind, task in enumerate(task_list):
    #     print(task_ind)
    #     batch_task(task)


def test_2D_HN():
    # with open("data/nH-Haldane-params.pkl", "wb") as fp:
    #     pickle.dump((t1, t2, gamma, phi, M), fp)
    fname = "data/2D-HN-test"
    model = HN.get_HN_model(*HN.Default_Model_Params)
    model = model.get_supercell(
        [(0,0)],
        np.array(
            [[1,0],
             [1,1]],
            dtype=int
        )
    )
    r_list = np.linspace(0.3, 3, 200)
    n_minor = 40
    n_theta1 = 400
    n_theta1_num = 400
    task_list = [
        (
            fname, model,
            np.random.randn() + 1j * np.random.randn(),
            n_minor,
            n_theta1,
            n_theta1_num,
            r_list,
            HN.Default_Model_Params
        )
        for _ in range(10)
    ]

    pool = multiprocessing.Pool(N_PROCESS)
    pool.map(batch_task, task_list)


def test_2D_HN_alpha2():
    fname = "data/2D-HN-test-alpha2"
    alpha = 2
    Jx, Jy = [(-0.59477-1.09826j), (1.18704-0.07472j)]
    gamma_x, gamma_y = [1.97069, 0.52332]
    delta = 1.79515
    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, delta, delta)
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1, 0],
            [alpha, 1]
        ])
    )
    kx = 0.3 * pi
    ky = 0.3 * pi
    new_glue_tol = [0.01, 0.02, 1e-4, 0.03]
    beta_x = gamma_x * np.exp(1j * kx)
    beta_y = gamma_y * np.exp(1j * ky)
    E_ref = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y

    r_list = np.linspace(0.3, 0.8, 200)
    n_minor = 40
    n_theta1 = 400
    n_theta1_num = 400
    task_list = [
        (
            fname, model,
            # np.random.randn() + 1j * np.random.randn(),
            E_ref,
            n_minor,
            n_theta1,
            n_theta1_num,
            r_list,
            HN.Default_Model_Params
        )
        # for _ in range(10)
    ]

    pool = multiprocessing.Pool(N_PROCESS)
    pool.map(batch_task, task_list)


if __name__ == "__main__":
    # test_minimal_nontrivial_model()
    # test_2D_SSH_model()
    # test_nH_Haldane()
    # test_2D_HN()
    test_2D_HN_alpha2()
