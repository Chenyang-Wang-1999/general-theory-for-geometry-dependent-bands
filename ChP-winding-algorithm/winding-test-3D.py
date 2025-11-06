'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-07-17
Copyright © Department of Physics, Tsinghua University. All rights reserved

Calculate winding numbers for 3D HN model
'''

import matplotlib.pyplot as plt
import BerryPy.TightBinding as tb
import pickle
import numpy as np
from quasi_1D_winding import calculate_quasi_1D_winding
import os
import multiprocessing


def get_HN_model_3D(
    Jx1, Jx2, Jy1, Jy2, Jz1, Jz2
):
    '''
    Generate 3D HN model
    '''
    dim = 3
    site_num = 1
    intra_cell = []
    inter_cell = [
        [0, 0, Jx1, (1, 0, 0)],
        [0, 0, Jx2, (-1, 0, 0)],
        [0, 0, Jy1, (0, 1, 0)],
        [0, 0, Jy2, (0, -1, 0)],
        [0, 0, Jz1, (0, 0, 1)],
        [0, 0, Jz2, (0, 0, -1)]
    ]
    lattice_vec = np.eye(3)
    return tb.TightBindingModel(
        dim, site_num, lattice_vec, intra_cell, inter_cell
    )


def calculate_winding_number(
    model: tb.TightBindingModel,
    E_ref: complex,
    r1: float,
    minor_size: tuple[int, int],
    n_points: int
):
    '''
    Calculate winding number for a given 3D model
    '''
    # 1. Expand along a2
    model_2d = model.get_supercell(
        [(0, j, 0) for j in range(minor_size[0])],
        np.array([
            [1, 0, 0],
            [0, minor_size[0], 0],
            [0, 0, 1]
        ], dtype=int)
    ).truncate(1)

    model_1d = model_2d.get_supercell(
        [(0, j) for j in range(minor_size[1])],
        np.array([
            [1, 0],
            [0, minor_size[1]]
        ], dtype=int)
    ).truncate(1)

    w = calculate_quasi_1D_winding(
        model_1d,
        E_ref,
        n_points,
        r1
    )
    return w


def main():
    Jx1, Jx2, Jy1, Jy2, Jz1, Jz2 = (
        1.0, 1.0j, 2.0, 2.0j, 3.0, 3.0j
    )
    model = get_HN_model_3D(
        Jx1, Jx2, Jy1, Jy2, Jz1, Jz2
    )
    n_processing = 12

    r_list = np.linspace(0.5, 1.5, 100)
    # E_ref = (np.random.randn() + 1j * np.random.randn()) * 0.1
    theta1, theta2, theta3 = np.random.rand(3) * 2 * np.pi
    E_ref = (
        Jx1 * np.exp(-1j * theta1) + Jx2 * np.exp(1j * theta1)
        + Jy1 * np.exp(-1j * theta2) + Jy2 * np.exp(1j * theta2)
        + Jz1 * np.exp(-1j * theta3) + Jz2 * np.exp(1j * theta3)
    )
    minor_size = (10, 30)
    n_points = 400
    w_list = []

    print("E_ref:", E_ref)
    print("minor_size:", minor_size)
    print("n_points:", n_points)

    with multiprocessing.Pool(n_processing) as pool:
        arg_list = [
            (model, E_ref, r, minor_size, n_points)
            for r in r_list
        ]
        w_list = pool.starmap(
            calculate_winding_number,
            arg_list
        )

    # save
    fname_prefix = "data/HN-model-3D"
    task_id = 0
    while os.path.exists("%s-%d.pkl" % (fname_prefix, task_id)):
        task_id += 1
    
    with open("%s-%d.pkl" % (fname_prefix, task_id), "wb") as fp:
        pickle.dump((
            r_list,
            w_list,
            E_ref,
            minor_size,
            n_points,
            (Jx1, Jx2, Jy1, Jy2, Jz1, Jz2)
        ), fp)


def main_plot():
    fname_prefix = "data/HN-model-3D"
    # task_id = 8
    for task_id in [8, 9, 10]:
        with open("%s-%d.pkl" % (fname_prefix, task_id), "rb") as fp:
            (
                r_list,
                w_list,
                E_ref,
                minor_size,
                n_points,
                (Jx1, Jx2, Jy1, Jy2, Jz1, Jz2)
            ) = pickle.load(fp)
    
        plt.plot(r_list, w_list / np.prod(minor_size))
        plt.show()



if __name__ == "__main__":
    # for _ in range(3):
        # main()
    main_plot()
