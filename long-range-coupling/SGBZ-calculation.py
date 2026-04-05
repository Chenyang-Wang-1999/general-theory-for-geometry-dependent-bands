'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-02-09
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from model import DEFAULT_PARAMS, LongRangeModel
from math import sin, cos, sqrt, pi
from cmath import exp, log
import numpy as np
from long_range_GBZ_1D import sweep_GBZ_points_for_mu
import multiprocessing as mp
import pickle
import os
import matplotlib.pyplot as plt
from scipy import interpolate


def calculate_y_SGBZ_for_mu(
    ky: float,
    mu: float,
    N_points: int = 501
):
    print(ky, mu)
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
    betay = exp(1j * ky)
    gbz_1D_func = lambda z: complex(model.char_poly(z, betay))
    GBZ_points, flag = sweep_GBZ_points_for_mu(gbz_1D_func, mu, N_points)
    if isinstance(flag, int):
        return {
            "success": True,
            "GBZ_points": GBZ_points,
            "ky": ky,
            "mu": mu,
        }
    else:
        return {
            "success": False,
            "mean-winding": np.mean(flag),
            "ky": ky,
            "mu": mu,
        }


def batch_sweep_y_SGBZ(
    ky_list: np.ndarray[float, 1],
    mu_list: np.ndarray[float, 1],
    n_process: int = mp.cpu_count()
):
    data_package = zip(ky_list, mu_list)
    if n_process == 1:
        results = [calculate_y_SGBZ_for_mu(ky, mu) for ky, mu in data_package]
    else:
        with mp.Pool(processes=n_process) as pool:
            results = pool.starmap(calculate_y_SGBZ_for_mu, data_package)

    fname_prefix = "data/y-SGBZ-results"
    file_ind = 0
    while os.path.exists(fname_prefix + f"-{file_ind}.pkl"):
        file_ind += 1

    with open(fname_prefix + f"-{file_ind}.pkl", "wb") as f:
        pickle.dump(results, f)


def main_sweep_y_SGBZ():
    ky_list = np.linspace(-pi, pi, 51)
    mu_list = np.linspace(0, 0.4, 11)
    batch_sweep_y_SGBZ(ky_list, mu_list)


def main_continue_sweep_y_SGBZ():
    N_mu = 201
    N_ky = 201
    ky_unique, mu_min, mu_max = get_mu_boundary_from_result(0, 3)
    # ky_filter = (np.abs(ky_unique) > 2.35) & (np.abs(ky_unique) < 2.55)
    # mu_min = 0.8 * np.ones_like(ky_unique[ky_filter])
    # mu_max = 1.2 * np.ones_like(ky_unique[ky_filter])

    mu_min_fun = interpolate.CubicSpline(ky_unique, mu_min, bc_type="periodic")
    mu_max_fun = interpolate.CubicSpline(ky_unique, mu_max, bc_type="periodic")

    ky_new = np.linspace(-pi, pi, N_ky)

    ky_list = []
    mu_list = []
    for ind, ky in enumerate(ky_new):
        ky_list.append(ky * np.ones(N_mu))
        mu_list.append(np.linspace(mu_min_fun(ky), mu_max_fun(ky), N_mu))
    ky_list = np.concatenate(ky_list)
    mu_list = np.concatenate(mu_list)
    # plt.plot(ky_list, mu_list, ".")
    # plt.show()
    batch_sweep_y_SGBZ(ky_list, mu_list)


def plot_sweeping_results(fid_start: int, fid_end: int):
    results = []
    for fid in range(fid_start, fid_end + 1):
        with open(f"data/y-SGBZ-results-{fid}.pkl", "rb") as f:
            results.extend(pickle.load(f))
    
    ky_list = np.array([r["ky"] for r in results])
    mu_list = np.array([r["mu"] for r in results])
    success_list = np.array([r["success"] for r in results])
    flag_list = np.zeros_like(mu_list, dtype=float)
    for i, r in enumerate(results):
        if not r["success"]:
            flag_list[i] = r["mean-winding"]

    ky_unique, mu_min, mu_max = get_mu_boundary_from_result(fid_start, fid_end)
    plt.plot(ky_list[flag_list == 0], mu_list[flag_list == 0], ".")
    plt.plot(ky_list[flag_list < 0], mu_list[flag_list < 0], ".")
    plt.plot(ky_list[flag_list > 0], mu_list[flag_list > 0], ".")
    plt.plot(ky_unique, mu_min, "k--")
    plt.plot(ky_unique, mu_max, "k--")

    plt.show()


def get_mu_boundary_from_result(fid_start: int, fid_end: int):
    result = []
    for fid in range(fid_start, fid_end + 1):
        with open(f"data/y-SGBZ-results-{fid}.pkl", "rb") as f:
            results = pickle.load(f)
        result.extend(results)

    ky_list = np.array([r["ky"] for r in result])
    mu_list = np.array([r["mu"] for r in result])
    flag_list = np.zeros_like(ky_list, dtype=float)
    for i, r in enumerate(result):
        if not r["success"]:
            flag_list[i] = r["mean-winding"]

    ky_unique = np.unique(ky_list)
    mu_min = np.zeros_like(ky_unique, dtype=float)
    mu_max = np.zeros_like(ky_unique, dtype=float)
    for ind, ky in enumerate(ky_unique):
        list_filter = np.abs(ky_list - ky) < 1e-6
        mu_filtered = mu_list[list_filter]
        flag_filtered = flag_list[list_filter]
        if np.any(flag_filtered < 0):
            mu_min[ind] = np.max(mu_filtered[flag_filtered < 0])
        else:
            mu_min[ind] = np.min(mu_filtered) - 0.2

        if np.any(flag_filtered > 0):
            mu_max[ind] = np.min(mu_filtered[flag_filtered > 0])
        else:
            mu_max[ind] = np.max(mu_filtered) + 0.2

    return ky_unique, mu_min, mu_max


def plot_SGBZ_energy(fid_start: int, fid_end: int):
    results = []
    for fid in range(fid_start, fid_end + 1):
        with open(f"data/y-SGBZ-results-{fid}.pkl", "rb") as f:
            results.extend(pickle.load(f))

    E_list = []
    for r in results:
        if r["success"]:
            for E_val, mu, theta1, theta2 in r["GBZ_points"]:
                E_list.append(E_val)
    E_list = np.array(E_list)
    # plt.plot(E_list.real, E_list.imag, '.', markeredgecolor="none")
    plt.plot(E_list.real, E_list.imag, 'x')


def plot_OBC_spectrum(nx: int, ny: int, type=""):
    with open(f"data/OBC-default{type}-nx{nx}-ny{ny}.pkl", "rb") as f:
        params, coord_list, eigv, eigvec = pickle.load(f)
    plt.plot(eigv.real, eigv.imag, '.')


if __name__ == "__main__":
    # main_sweep_y_SGBZ()
    # plot_sweeping_results(0, 4)
    # main_continue_sweep_y_SGBZ()
    # plt.figure()
    plot_SGBZ_energy(0, 4)
    # plt.figure()
    plot_OBC_spectrum(20, 300)
    plt.show()