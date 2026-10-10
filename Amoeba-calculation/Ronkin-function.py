'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-08-20 16:39:45
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from scipy import integrate
from scipy import linalg as la
import numpy as np
from BerryPy import TightBinding as tb
from math import sin, cos, sqrt, pi
from cmath import exp, log
import pyvista as pv
import matplotlib.pyplot as plt
import pickle

# The color palette is shared by all the plotting scripts of this repository,
# and lives in colors.py at its root.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from colors import COLORS

def get_HN_model(Jx1, Jx2, Jy1, Jy2):
    inter_cell = [
        [0, 0, Jx1, (1, 0)],
        [0, 0, Jx2, (-1, 0)],
        [0, 0, Jy1, (0, 1)],
        [0, 0, Jy2, (0, -1)]
    ]
    return tb.TightBindingModel(2, 1, np.array([[1, 0], [0, 1]]), [], inter_cell, [[0, 0]])

def eval_ChP(theta_1, theta_2, model: tb.TightBindingModel, mu_vec: np.ndarray, E: complex):
    mu1 = mu_vec[0]
    mu2 = mu_vec[1]
    beta1 = np.exp(mu1 + 1j * theta_1)
    beta2 = np.exp(mu2 + 1j * theta_2)
    H = model.get_bulk_Hamiltonian_complex((beta1, beta2))
    chp = la.det(np.eye(model.dim) * E - H.todense())
    return np.log(abs(chp)) / ((2 * pi) ** (model.dim))

def calculate_Ronkin_func(
    model: tb.TightBindingModel,
    mu_vec: np.ndarray,
    E: complex
):
    theta_range = [[-pi, pi], [-pi, pi]]
    return integrate.nquad(
        eval_ChP,
        theta_range,
        args=(model, mu_vec, E),
        opts={'epsabs': 1e-4, 'epsrel': 1e-6}
    )

def generate_HN_data(
    old_data: tuple[float] = None
):
    if old_data is None:
        Jx_re, Jx_im, Jy_re, Jy_im, gammax, gammay, deltax, deltay = np.random.randn(8) // 1e-4 * 1e-4
        Jx = Jx_re + 1j * Jx_im
        Jy = Jy_re + 1j * Jy_im
    else:
        Jx, Jy, gammax, gammay, deltax, deltay = old_data
    print(Jx, Jy, gammax, gammay, deltax, deltay)
    Jx1 = exp(gammax + 1j * deltax) * Jx
    Jx2 = exp(-gammax + 1j * deltax) * Jx.conjugate()
    Jy1 = exp(gammay + 1j * deltay) * Jy
    Jy2 = exp(-gammay + 1j * deltay) * Jy.conjugate()
    model = get_HN_model(Jx1, Jx2, Jy1, Jy2)
    return model, (Jx, Jy, gammax, gammay, deltax, deltay)

def sweep_Ronkin_func(
    model: tb.TightBindingModel,
    E: complex,
    mu1_arr: np.ndarray,
    mu2_arr: np.ndarray
):
    Ronkin_func_arr = np.zeros_like(mu1_arr, dtype=float)
    for point_ind in range(len(mu1_arr)):
        mu_vec = (mu1_arr[point_ind], mu2_arr[point_ind])
        res = calculate_Ronkin_func(model, mu_vec, E)
        print((mu1_arr[point_ind], mu2_arr[point_ind]), res)
        Ronkin_func_arr[point_ind] = res[0]
    return Ronkin_func_arr

def main_select_param():
    params = [(1.4757-0.0246j), (0.1585-1.0847j), 0.5009, -1.3837, -0.2759, 0.2602]
    # params[-2] = params[-1]
    model, params = generate_HN_data(params)
    Jx, Jy, gammax, gammay, deltax, deltay = params

    p1 = 2 * abs(Jx) * np.exp(1j * deltax)
    p2 = 2 * abs(Jy) * np.exp(1j * deltay)

    E = p1 / 2

    all_points = np.array(
        [p1 + p2, p1 - p2, -p1 - p2, -p1 + p2, p1 + p2]
    )
    plt.plot(all_points.real, all_points.imag)
    plt.plot(E.real, E.imag, 'x')
    print(E)
    plt.savefig("data/spectrum.png")
    plt.close()

def main_sweep_GDSE():
    params = [(1.4757-0.0246j), (0.1585-1.0847j), 0.5009, -1.3837, -0.2759, 0.2602]
    model, params = generate_HN_data(params)
    Jx, Jy, gammax, gammay, deltax, deltay = params
    Nx = 21
    Ny = 21
    dmu1 = 1e-1
    dmu2 = 1e-1
    mu1 = gammax + np.linspace(-dmu1, dmu1, Nx)
    mu2 = gammay + np.linspace(-dmu2, dmu2, Ny)

    mu1_arr, mu2_arr = np.meshgrid(mu1, mu2)
    mu1_arr_flat = mu1_arr.flatten()
    mu2_arr_flat = mu2_arr.flatten()
    p1 = 2 * abs(Jx) * np.exp(1j * deltax)
    p2 = 2 * abs(Jy) * np.exp(1j * deltay)

    E = p1 / 2
    Ronkin_func_arr = sweep_Ronkin_func(model, E, mu1_arr_flat, mu2_arr_flat)

    with open("data/Ronkin_func_sweep_GDSE_dmu_%.2f_%.2f.pkl" % (np.log10(dmu1), np.log10(dmu2)), "wb") as fp:
        pickle.dump((Ronkin_func_arr, mu1_arr, mu2_arr, params), fp)

def main_sweep_uniform():
    params = [(1.4757-0.0246j), (0.1585-1.0847j), 0.5009, -1.3837, -0.2759, -0.2759]
    model, params = generate_HN_data(params)
    Jx, Jy, gammax, gammay, deltax, deltay = params
    Nx = 21
    Ny = 21
    dmu1 = 1e-1
    dmu2 = 1e-1
    mu1 = gammax + np.linspace(-dmu1, dmu1, Nx)
    mu2 = gammay + np.linspace(-dmu2, dmu2, Ny)

    mu1_arr, mu2_arr = np.meshgrid(mu1, mu2)
    mu1_arr_flat = mu1_arr.flatten()
    mu2_arr_flat = mu2_arr.flatten()
    p1 = 2 * abs(Jx) * np.exp(1j * deltax)
    p2 = 2 * abs(Jy) * np.exp(1j * deltay)

    E = p1 / 2
    Ronkin_func_arr = sweep_Ronkin_func(model, E, mu1_arr_flat, mu2_arr_flat)

    with open("data/Ronkin_func_sweep_uniform_dmu_%.2f_%.2f.pkl" % (np.log10(dmu1), np.log10(dmu2)), "wb") as fp:
        pickle.dump((Ronkin_func_arr, mu1_arr, mu2_arr, params), fp)

def plot_ronkin_3d():
    def load_and_plot(pkl_path, title):
        with open(pkl_path, 'rb') as fp:
            Ronkin_func_arr, mu1_arr, mu2_arr, params = pickle.load(fp)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        Ronkin_func_mesh = Ronkin_func_arr.reshape(mu1_arr.shape)
        ax.plot_surface(mu1_arr, mu2_arr, Ronkin_func_mesh, 
            color=COLORS["bf"], edgecolor="k")
        plt.show()
        # Ronkin_grid = Ronkin_func_arr.reshape(mu1_arr.shape)
        # x_scale = np.max(mu1_arr) - np.min(mu1_arr)
        # y_scale = np.max(mu2_arr) - np.min(mu2_arr)
        # z_scale = np.max(Ronkin_grid) - np.min(Ronkin_grid)
        # grid = pv.StructuredGrid(mu1_arr / x_scale, mu2_arr / y_scale, Ronkin_grid / z_scale)
        # plotter = pv.Plotter()
        # plotter.add_mesh(grid, show_edges=True)
        # plotter.add_title(title)
        # plotter.show()
    dmu1 = 1e-1
    dmu2 = 1e-1
    # load_and_plot('data/Ronkin_func_sweep_GDSE.pkl', 'Ronkin Function - GDSE')
    # load_and_plot('data/Ronkin_func_sweep_uniform.pkl', 'Ronkin Function - Uniform')
    load_and_plot('data/Ronkin_func_sweep_GDSE_dmu_%.2f_%.2f.pkl' % (np.log10(dmu1), np.log10(dmu2)), 'Ronkin Function - GDSE')
    load_and_plot('data/Ronkin_func_sweep_uniform_dmu_%.2f_%.2f.pkl' % (np.log10(dmu1), np.log10(dmu2)), 'Ronkin Function - Uniform')

if __name__ == "__main__":
    # main_select_param()
    # main_sweep_GDSE()
    # main_sweep_uniform()
    plot_ronkin_3d()
