'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-04-01
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import palette_manager as pm
from math import pi

plt.style.use("../settings-and-materials/paper_plot.mplstyle")
CM = 1/2.54

COLORS = pm.get_palette("matplotlib")

DATA_FOLDER = "../../ChP-winding-algorithm/data/"

def plot_winding_3D(alpha: int):
    with open(DATA_FOLDER + f"HN-alpha-{alpha}.pkl", "rb") as f:
        data = pickle.load(f)

    # 解包数据
    (Jx, Jy, gamma_x, gamma_y, delta), (kx, ky), result = data
    _, _, w_left_list, w_right_list = zip(*result)

    # 重塑数据为网格格式
    w_left_list = np.array(w_left_list).reshape(len(kx), len(ky))
    w_right_list = np.array(w_right_list).reshape(len(kx), len(ky))
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)

    # 创建pyvista绘图器
    plotter = pv.Plotter(off_screen=True, line_smoothing=True, window_size=(2000, 2200))
    # plotter = pv.Plotter()

    # 添加左极限曲面
    left_mesh = pv.StructuredGrid(kx_mesh / pi, ky_mesh / pi, w_left_list / 2)
    plotter.add_mesh(left_mesh, 
                    color=COLORS["rl"],
                    opacity=0.7,
                    show_edges=True,
                    label="Left limit")

    # 添加右极限曲面
    right_mesh = pv.StructuredGrid(kx_mesh / pi, ky_mesh / pi, w_right_list / 2)
    plotter.add_mesh(right_mesh,
                    color=COLORS["bl"],
                    opacity=0.7,
                    label="Right limit",
                    show_edges=True
                    )

    # 添加零平面
    zero_mesh = pv.StructuredGrid(kx_mesh / pi, ky_mesh / pi, np.zeros_like(kx_mesh))
    plotter.add_mesh(zero_mesh, 
                    color="grey",
                    opacity=1,
                    label="Zero plane")

    # 设置标题和坐标轴
    plotter.show_grid(
        xtitle="x",
        ytitle="y",
        ztitle="z",
        font_size=45,
        font_family="Arial",
        fmt="%.0f",
        n_xlabels = 3,
        n_ylabels = 3,
        n_zlabels = 5,
        bounds=[-1, 1, -1, 1, -1.5, 1.5],
    )
    plotter.camera.elevation = -25
    plotter.camera.azimuth = -80
    # plotter.add_title(f'HN Model Winding Number (α={alpha})')
    # plotter.add_axes()
    # plotter.add_legend()

    # plotter.show()
    plotter.screenshot(f"Figures/winding-3D-alpha-{alpha}.png")
    # plotter.export_gif(f"Figures/winding-3D-alpha-{alpha}.gif", fps=5, quality=10)


def plot_Wstrip():
    fig = plt.figure(figsize=(4*CM, 4*CM))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])

    # 1. load data
    with open(DATA_FOLDER + "paper-W_strip-no-GDSE.pkl", "rb") as f:
        params, k_list, E_ref, (r_list, w_list), mGBZ_sols = pickle.load(f)
    # 2. plot
    ax.plot(r_list, w_list, '-')
    ax.set_xlim([0, 1.2])
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$W_\mathrm{strip}(r)$")
    fig.savefig("Figures/W-strip-no-GDSE.pdf")


def plot_E():
    fig = plt.figure(figsize=(1*CM, 5*CM))
    ax = fig.gca()
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    # 1. load data
    with open(DATA_FOLDER + "paper-W_strip-no-GDSE.pkl", "rb") as f:
        params, k_list, E_ref, (r_list, w_list), mGBZ_sols = pickle.load(f)
    (Jx, Jy, gamma_x, gamma_y, delta) = params
    # 2. plot
    E_extreme = 2 * (abs(Jx) + abs(Jy)) * np.exp(1j * delta)
    E_list = np.array([-E_extreme, E_extreme])
    ax.plot(
        E_list.real,
        E_list.imag
    )
    ax.plot(E_ref.real, E_ref.imag, 'x', markersize=4)
    ax.set_aspect(1)
    ax.set_xticks([-1, 1], minor=False)
    ax.set_yticks([-5, 5], minor=False)
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    fig.savefig("Figures/no-NHSE-spectrum.pdf")


def plot_mGBZ():
    fig = plt.figure(figsize=(4*CM, 4*CM))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    # 1. load data
    with open(DATA_FOLDER + "paper-W_strip-no-GDSE.pkl", "rb") as f:
        params, k_list, E_ref, (r_list, w_list), mGBZ_sols = pickle.load(f)

    for loop in mGBZ_sols:
        ax.plot(loop[:,0].real, loop[:,0].imag, '-', color=COLORS["bl"])
    ax.set_xlabel("Re$(\\beta_1)$")
    ax.set_ylabel("Im$(\\beta_1)$")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xticks([-1, 0, 1], minor=False)
    ax.set_yticks([-1, 0, 1], minor=False)
    fig.savefig("Figures/no-GDSE-mGBZ.pdf")


def plot_left_and_right():
    alpha_list = range(-3, 4)
    fig = plt.figure(figsize=(4*CM, 4*CM))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("Limits of $W_\mathrm{strip}(r)$")
    ax.set_xlim([-3.2, 3.2])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3], minor=False)
    ax.set_xticks([], minor=True)

    for alpha in alpha_list:
        with open(DATA_FOLDER + f"HN-alpha-{alpha}.pkl", "rb") as f:
            data = pickle.load(f)
        # 解包数据
        (Jx, Jy, gamma_x, gamma_y, delta), (kx, ky), result = data
        _, _, w_left_list, w_right_list = zip(*result)

        # plot
        pl_left = ax.plot(
            alpha * np.ones_like(w_left_list),
            w_left_list,
            '.',
            color=COLORS["rl"],
            label="Left limit"
        )
        pl_right = ax.plot(
            alpha * np.ones_like(w_right_list),
            w_right_list,
            '.',
            color=COLORS["bl"],
            label="Right limit"
        )
    ax.plot([-3.2, 3.2], [0, 0], '--', color="grey")
    ax.legend(handles=[pl_left[0], pl_right[0]])
    fig.savefig("Figures/no-GDSE-left-and-right-limit.pdf")


if __name__ == "__main__":
    # plot_winding_3D(2)
    # plot_Wstrip()
    # plot_E()
    # plot_mGBZ()
    plot_left_and_right()
