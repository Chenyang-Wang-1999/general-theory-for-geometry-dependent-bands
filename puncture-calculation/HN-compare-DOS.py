'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-05-24
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from scipy import linalg as la
import sys
import pickle
sys.path.append("../paper/HN-model")
import HN_model_common as HN

Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
Jx, Jy, gammax, gammay, deltax, deltay = HN.get_new_param_from_coupling_terms(Jx1, Jx2, Jy1, Jy2)

def pre_image_xy(E_ref):
    jx_cosphix, jy_cosphiy = 0.5 * la.solve(
        np.array([
            [np.cos(deltax), np.cos(deltay)],
            [np.sin(deltax), np.sin(deltay)],
        ]),
        np.array([E_ref.real, E_ref.imag])
    )
    if abs(jx_cosphix) > abs(Jx) or abs(jy_cosphiy) > abs(Jy):
        return []
    else:
        phix = np.arccos(jx_cosphix / abs(Jx))
        phiy = np.arccos(jy_cosphiy / abs(Jy))
        return [
            (phix, phiy),
            (phix, -phiy),
            (-phix, phiy),
            (-phix, -phiy)
        ]


def calculate_DOS(eigv: np.ndarray, real_grids: np.ndarray, imag_grids: np.ndarray):
    ''' Indexing: ij '''
    point_num = np.zeros((real_grids.shape[0] - 1, imag_grids.shape[0] - 1))
    for row_ind in range(real_grids.shape[0] - 1):
        for col_ind in range(imag_grids.shape[0] - 1):
            point_num[row_ind, col_ind] = np.sum(
                (eigv.real >= real_grids[row_ind]) & 
                (eigv.real < real_grids[row_ind + 1]) &
                (eigv.imag >= imag_grids[col_ind]) & 
                (eigv.imag < imag_grids[col_ind + 1]))
    return point_num


def main_calculate_DOS_xy():
    aspect = 8
    N = 40
    with open("data/paper-HN-OBC-2-xy-aspect_%.6f-N_%d.pkl" % (aspect, N), "rb") as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)
    E_real_grid = np.linspace(-6, 6, 200)
    E_imag_grid = np.linspace(-3, 3, 200)

    state_count = calculate_DOS(eigv, E_real_grid, E_imag_grid)
    print(state_count.shape)
    result_fname = "data/paper-HN-OBC-2-xy-DOS-aspect_%.6f-N_%d.pkl" % (aspect, N)
    with open(result_fname, "wb") as f:
        pickle.dump((state_count, E_real_grid, E_imag_grid), f)


def main_DOS_theory():
    from dos_area import compute_dos_grid
    E_real_grid = np.linspace(-6, 6, 200)
    E_imag_grid = np.linspace(-3, 3, 200)

    dos_theory = compute_dos_grid(
        E_real_grid,
        E_imag_grid,
        pre_image_xy,
    )
    # print(dos_theory.shape)
    with open("data/DOS-xy-theory.pkl", "wb") as fp:
        pickle.dump(
            (E_real_grid, E_imag_grid, dos_theory), fp
        )


def check_DOS_calculation():
    import matplotlib.pyplot as plt
    aspect = 8
    N = 40
    with open("data/paper-HN-OBC-2-xy-DOS-aspect_%.6f-N_%d.pkl" % (aspect, N), "rb") as fp:
        state_count, E_real_grid, E_imag_grid = pickle.load(fp)
    
    with open("data/DOS-xy-theory.pkl", "rb") as fp:
        E_real_grid, E_imag_grid, dos_theory = pickle.load(fp)

    print(np.sum(state_count), np.mean(state_count), np.max(state_count), np.min(state_count))
    print(np.sum(dos_theory / (2 * np.pi) ** 2 * 12800))
    diff_mat = dos_theory / (2 * np.pi) ** 2 * 12800 - state_count.T
    # print(np.mean(np.abs(diff_mat)) * (len(E_real_grid) - 1) * (len(E_imag_grid) - 1) / 12800)
    print(np.sum(np.abs(diff_mat)) / 12800)
    dos_max = np.max(state_count)
    plt.figure()
    plt.pcolormesh(E_real_grid, E_imag_grid, state_count.T, vmin=0, vmax=dos_max)
    plt.colorbar()
    plt.figure()
    plt.pcolormesh(E_real_grid, E_imag_grid, dos_theory / (2 * np.pi) ** 2 * 12800, vmin=0, vmax=dos_max)
    plt.colorbar()
    plt.figure()
    plt.pcolormesh(E_real_grid, E_imag_grid, diff_mat, vmin=-dos_max, vmax=dos_max, cmap="bwr")
    plt.colorbar()
    plt.show()



if __name__ == "__main__":
    # main_calculate_DOS_xy()
    # main_DOS_theory()
    check_DOS_calculation()
