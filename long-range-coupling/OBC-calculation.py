'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-02-09
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from model import LongRangeModel, DEFAULT_PARAMS
from math import sin, cos, sqrt, pi
from cmath import exp, log
import numpy as np
from scipy import linalg as la
import pickle
import matplotlib.pyplot as plt

def get_OBC(nx: int, ny: int):
    """
    Get the OBC Hamiltonian of a 2D system with nx * ny sites.
    """
    params = DEFAULT_PARAMS.copy()
    params["tx1"] = params["tx"] * exp(params["gammax"])
    params["tx2"] = params["tx"] * exp(-params["gammax"])
    params["ty1"] = params["ty"] * exp(params["gammay"])
    params["ty2"] = params["ty"] * exp(-params["gammay"])
    params.pop("tx")
    params.pop("ty")
    params.pop("gammax")
    params.pop("gammay")
    model = LongRangeModel(**params)

    x = np.arange(nx)
    y = np.arange(ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    coord_list = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

    H_mat = model.get_OBC_Hamiltonian(coord_list)
    eigv, eigvec = la.eig(H_mat.todense())
    with open(f"data/OBC-default-nx{nx}-ny{ny}.pkl", "wb") as f:
        pickle.dump((params, coord_list, eigv, eigvec), f)


def plot_OBC_spectrum(nx: int, ny: int, type=""):
    with open(f"data/OBC-default{type}-nx{nx}-ny{ny}.pkl", "rb") as f:
        params, coord_list, eigv, eigvec = pickle.load(f)
    plt.plot(eigv.real, eigv.imag, '.')


def get_OBC_11(nx: int, ny: int):
    params = DEFAULT_PARAMS.copy()
    params["tx1"] = params["tx"] * exp(params["gammax"])
    params["tx2"] = params["tx"] * exp(-params["gammax"])
    params["ty1"] = params["ty"] * exp(params["gammay"])
    params["ty2"] = params["ty"] * exp(-params["gammay"])
    params.pop("tx")
    params.pop("ty")
    params.pop("gammax")
    params.pop("gammay")
    model = LongRangeModel(**params)

    x = np.arange(nx)
    y = np.arange(ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    coord_list = np.column_stack((x_mesh.ravel(), (x_mesh + y_mesh).ravel()))

    H_mat = model.get_OBC_Hamiltonian(coord_list)
    eigv, eigvec = la.eig(H_mat.todense())
    with open(f"data/OBC-default-11-nx{nx}-ny{ny}.pkl", "wb") as f:
        pickle.dump((params, coord_list, eigv, eigvec), f)


def plot_BZ_bands():
    params = DEFAULT_PARAMS.copy()
    params["tx1"] = params["tx"] * exp(params["gammax"])
    params["tx2"] = params["tx"] * exp(-params["gammax"])
    params["ty1"] = params["ty"] * exp(params["gammay"])
    params["ty2"] = params["ty"] * exp(-params["gammay"])
    params.pop("tx")
    params.pop("ty")
    params.pop("gammax")
    params.pop("gammay")
    model = LongRangeModel(**params)

    kx = np.linspace(-pi, pi, 51)
    ky = np.linspace(-pi, pi, 51)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    E_mesh = np.zeros_like(kx_mesh, dtype=complex)
    for i in range(kx_mesh.shape[0]):
        for j in range(kx_mesh.shape[1]):
            E_mesh[i, j] = model.char_poly(exp(1j * kx_mesh[i, j] + DEFAULT_PARAMS["gammax"]), exp(1j * ky_mesh[i, j]))
    plt.plot(E_mesh.real, E_mesh.imag, '.')


if __name__ == "__main__":
    nx, ny = 77, 77
    get_OBC(nx, ny)
    # get_OBC_11(50, 50)
    # plt.figure()
    # plot_OBC_spectrum(50, 50, type="")
    plt.figure()
    plot_OBC_spectrum(nx, ny, type="")
    # plt.figure()
    # plot_OBC_spectrum(50, 50, type="-11")
    # plt.figure()
    # plot_BZ_bands()
    plt.show()