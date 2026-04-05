'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-03-04
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from model import get_long_range_OBC
import numpy as np
from scipy import linalg as la
import pickle
import matplotlib.pyplot as plt


def coupling_fun(coupling_range: tuple[int, int]) -> complex:
    if coupling_range[0] == 0:
        if coupling_range[1] != 0:
            return 1

    elif coupling_range[1] == 0:
        return 1

    return None


def test_OBC_divergent_coupling(nx: int, ny: int):
    x = np.arange(nx)
    y = np.arange(ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    coord_list = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

    H_mat = get_long_range_OBC(coupling_fun, coord_list)
    assert H_mat.shape == (nx * ny, nx * ny)

    eigv, eigvec = la.eigh(H_mat.todense())

    with open(f"data/OBC-divergent-nx{nx}-ny{ny}.pkl", "wb") as f:
        pickle.dump((coord_list, eigv, eigvec), f)


def plot_OBC_divergent_spectrum(nx: int, ny: int):
    with open(f"data/OBC-divergent-nx{nx}-ny{ny}.pkl", "rb") as f:
        coord_list, eigv, eigvec = pickle.load(f)
    plt.plot(np.arange(len(eigv)) / len(eigv), np.sort(eigv), label=f"nx={nx}, ny={ny}")


if __name__ == "__main__":
    # test_OBC_divergent_coupling(10, 10)
    # test_OBC_divergent_coupling(20, 20)
    # test_OBC_divergent_coupling(30, 30)
    # test_OBC_divergent_coupling(40, 40)
    # test_OBC_divergent_coupling(50, 50)
    for nx in [10, 20, 30, 40]:
        # plt.figure()
        plot_OBC_divergent_spectrum(nx, nx)
    plt.legend()
    plt.show()