'''
author:        Wang Chenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-06-25 00:00:00
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from cmath import exp, sqrt, pi
import matplotlib.pyplot as plt
from color_set import create_white_to_color_cmap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

import sys
sys.path.append("../HN-model")
import HN_model_common as HN

# The color palette is shared by all the plotting scripts of this repository,
# and lives in colors.py at its root.
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from colors import COLORS

CM_IN_INCH = 1 / 2.54
MY_CMAP = lambda lbl: LinearSegmentedColormap.from_list(f"{lbl}-cmap", [COLORS[lbl], COLORS[lbl]])

class HNModel:
    Jx1: complex
    Jx2: complex
    Jy1: complex
    Jy2: complex
    Jx: complex
    Jy: complex
    gammax: float
    gammay: float
    deltax: float
    deltay: float

    def __init__(self, gammax, gammay, deltax, deltay, Jx, Jy) -> None:
        self.Jx = Jx
        self.Jy = Jy
        self.gammax = gammax
        self.gammay = gammay
        self.deltax = deltax
        self.deltay = deltay
        self.Jx1 = exp(gammax + 1j * deltax) * Jx
        self.Jx2 = exp(-gammax + 1j * deltax) * Jx.conjugate()
        self.Jy1 = exp(gammay + 1j * deltay) * Jy
        self.Jy2 = exp(-gammay + 1j * deltay) * Jy.conjugate()

def plot_11_SGBZ_spectrum(model: HNModel):
    bound, loop = HN.get_11_DGBZ_spectrum(
        model.Jx1, model.Jx2, model.Jy1, model.Jy2
    )
    plt.plot(loop.real, loop.imag)
    plt.show()

def plot_11_SGBZ_spectrum_uniform_Jnorm(model: HNModel):
    assert(abs(abs(model.Jx) - abs(model.Jy)) < 1e-6)
    Jx1 = model.Jx1
    Jx2 = model.Jx2
    Jy1 = model.Jy1
    Jy2 = model.Jy2
    bound = 2 * np.sqrt(
            2 * sqrt(Jx1 * Jx2 * Jy1 * Jy2) * np.array([-1, 1])
            + (Jx1 * Jx2)
            + (Jy1 * Jy2)
    )

    for j in range(2):
        plt.plot(
            [bound[j].real, -bound[j].real],
            [bound[j].imag, -bound[j].imag]
        )
    plt.show()

def get_11_SGBZ_EFC(omega0: float, model: HNModel):
    theta_11 = np.linspace(-pi, pi, 2000)
    r = model.gammax * model.gammay
    f1 = model.Jx1 / r * np.exp(-1j * theta_11) + model.Jy2
    f2 = model.Jx2 * r * np.exp(1j * theta_11) + model.Jy1
    varphi_1 = np.log(f1).imag
    varphi_2 = np.log(f2).imag
    varphi_bar = (varphi_1 + varphi_2) / 2
    delta_varphi = varphi_2 - varphi_1

    v11 = f1 * f2
    sqrt_v11 = np.exp(1j * varphi_bar) * np.sqrt(np.abs(v11))

    # find valid value of cos(theta_y)
    cos_theta_y = omega0 / (2 * sqrt_v11.real)
    selected_ind = np.abs(cos_theta_y) <= 1
    cos_theta_y_selected = cos_theta_y[selected_ind]
    theta_y_offset = np.arccos(cos_theta_y_selected)
    theta_y_total = np.hstack(
        [theta_y_offset + delta_varphi[selected_ind] / 2, 
        -theta_y_offset[::-1] + delta_varphi[selected_ind][::-1] / 2]
    )
    theta_11_total = np.hstack(
        [theta_11[selected_ind], theta_11[selected_ind][::-1]]
    )

    theta_11_total = np.hstack(
        [theta_11_total, theta_11_total[0]]
    )
    theta_y_total = np.hstack(
        [theta_y_total, theta_y_total[0]]
    )

    return (theta_11_total, theta_y_total)

def get_SGBZ_eigenenergy_by_definition(model: HNModel, theta_11: np.ndarray, theta_y: np.ndarray):
    r'''
        Calculate eigenenergies with given theta_11 and theta_y, according to the following formulas
        \begin{align}
            \beta_{[11]}\left(\theta_{[11]},\theta_{y}\right) & =\gamma_{x}\gamma_{y}e^{i\theta_{[11]}},\label{eq:SGBZ-11-1}\\
            \beta_{y^{\prime}}\left(\theta_{[11]},\theta_{y}\right) & =\gamma_{y}\sqrt{\left|\frac{e^{i\Delta_{xy}}J_{x}^{*}e^{i\theta_{[11]}}+J_{y}}{e^{i\Delta_{xy}}J_{x}e^{-i\theta_{[11]}}+J_{y}^{*}}\right|}e^{i\theta_{y}},\label{eq:SGBZ-11-2}
        \end{align}
        where $\Delta_{xy} \equiv \delta_x - \delta_y$, and 
        $$ E=J_{x1}\beta_{[11]}^{-1}\beta_{y}+J_{x2}\beta_{[11]}\beta_{y}^{-1}+J_{y1}\beta_{y}^{-1}+J_{y2}\beta_{y} $$
    '''
    r = model.gammax * model.gammay
    beta_11 = r * np.exp(1j * theta_11)
    Delta_xy = model.deltax - model.deltay
    beta_y = model.gammay * np.sqrt(
        np.abs(
            (exp(1j * Delta_xy) * model.Jx.conjugate() * np.exp(1j * theta_11) + model.Jy)
            / (exp(1j * Delta_xy) * model.Jx * np.exp(-1j * theta_11) + model.Jy.conjugate())
        )
    ) * np.exp(1j * theta_y)
    E = model.Jx1 / beta_11 * beta_y + model.Jx2 * beta_11 / beta_y + model.Jy1 / beta_y + model.Jy2 * beta_y
    return E, beta_11, beta_y

def paper_plot(model: HNModel, omega0: float):
    ''' Plot the 11-SGBZ spectrum for a given HN model
        same as "plot_11_SGBZ_spectrum_uniform_Jnorm"
    '''
    assert(abs(abs(model.Jx) - abs(model.Jy)) < 1e-6)
    bound = 2 * np.sqrt(
            2 * sqrt(model.Jx1 * model.Jx2 * model.Jy1 * model.Jy2) * np.array([-1, 1])
            + (model.Jx1 * model.Jx2)
            + (model.Jy1 * model.Jy2)
    )

    plt.style.use("../settings-and-materials/paper_plot.mplstyle")

    # 1. Plot spectrum
    fig = plt.figure(
        figsize=(3 * CM_IN_INCH, 2.5 * CM_IN_INCH)
    )
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_xticks(range(-4, 6, 2), minor=False)
    ax.set_yticks(range(-4, 6, 2), minor=False)
    ax.set_xticks(range(-3, 4, 2), minor=True)
    ax.set_yticks(range(-3, 4, 2), minor=True)
    for p in bound:
        ax.plot(
            [p.real, -p.real],
            [p.imag, -p.imag],
            color=COLORS["bl"]
        )
    
    ax.plot([2, 2], [-6, 6], "--", color=COLORS["ol"])
    ax.plot([2], [2], ".", color=COLORS["ol"])
    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")
    fig.savefig("Figures/SGBZ-no-spectral-area.pdf")

    # 2. Plot EFC
    theta_11, theta_y = get_11_SGBZ_EFC(omega0, model)
    E, beta_11, beta_y = get_SGBZ_eigenenergy_by_definition(model, theta_11, theta_y)
    fig = plt.figure(
        figsize=(3 * CM_IN_INCH, 2.5 * CM_IN_INCH)
    )
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    # 设置颜色映射
    cmap = create_white_to_color_cmap(np.array([0xff, 0x7f, 0x0e]) / 255, gray=0.9)
    
    # 准备线段数据
    points = np.array([theta_11/pi, theta_y/pi]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 创建带颜色映射的线段集合
    lc = LineCollection(
        segments, 
        cmap=cmap, 
        norm=plt.Normalize(abs(beta_y).min(), abs(beta_y).max())
    )
    lc.set_array(abs(beta_y))
    
    # 添加到坐标轴并设置属性
    line = ax.add_collection(lc)
    # ax.set_xlim(theta_11.min() - 0.1, theta_11.max() + 0.1)
    # ax.set_ylim(theta_y.min() - 0.1, theta_y.max() + 0.1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    # ax.set_aspect('equal')
    
    # 设置坐标轴标签
    ax.set_xlabel(r'$\theta_{[11]}$ ($\pi$)')
    ax.set_ylabel(r'$\theta_y$ ($\pi$)')
    
    fig.savefig("Figures/SGBZ-EFC-color-withoutclb.pdf")
    # 添加颜色条
    clb = plt.colorbar(line, ax=ax)
    clb.set_label('$|β_y|$')
    clb.set_ticks([abs(beta_y).min(), abs(beta_y).max()])
    clb.set_ticks([], minor=True)
    
    # 保存图像
    fig.savefig("Figures/SGBZ-EFC-color.pdf")

def plot_mu_spectrum(model: HNModel):
    Delta_xy = model.deltax - model.deltay
    # theta_11 = np.linspace(-pi, pi, 301)
    offset = np.logspace(-8, -3, 51)
    theta_11 = np.concatenate([
        np.log(model.Jx * model.Jy).imag - abs(Delta_xy) - pi - offset[::-1],
        np.log(model.Jx * model.Jy).imag - abs(Delta_xy) - pi + offset,
        np.linspace(np.log(model.Jx * model.Jy).imag - abs(Delta_xy) - 0.999 * pi, np.log(model.Jx * model.Jy).imag + abs(Delta_xy) - 1.001 * pi, 101),
        np.log(model.Jx * model.Jy).imag + abs(Delta_xy) - pi - offset[::-1],
        np.log(model.Jx * model.Jy).imag + abs(Delta_xy) - pi + offset,
        np.linspace(np.log(model.Jx * model.Jy).imag + abs(Delta_xy) - 0.999 * pi, np.log(model.Jx * model.Jy).imag - abs(Delta_xy) + 0.999 * pi, 101)
    ])
    theta_y = np.linspace(-pi, pi, 301)
    theta_11_mesh, theta_y_mesh = np.meshgrid(theta_11, theta_y)

    # for idx in range(len(theta_11)):
    #     curr_theta_11 = theta_11[idx]
    #     phase_shift = np.log((model.Jx.conjugate() * np.exp(1j * Delta_xy + 1j * curr_theta_11) + model.Jy)
    #         / (model.Jx * np.exp(1j * Delta_xy - 1j * curr_theta_11) + model.Jy.conjugate())).imag / 2
    #     theta_y_mesh[:, idx] += phase_shift

    beta_11 = np.exp(model.gammax + model.gammay + 1j * theta_11_mesh)
    beta_y = np.exp(model.gammay + 1j * theta_y_mesh) * np.sqrt(
        np.abs((model.Jx.conjugate() * np.exp(1j * Delta_xy + 1j * theta_11_mesh) + model.Jy)
            / (model.Jx * np.exp(1j * Delta_xy - 1j * theta_11_mesh) + model.Jy.conjugate()))
    )
    E_mesh = model.Jx1 * beta_y / beta_11 + model.Jx2 * beta_11 / beta_y + model.Jy1 / beta_y + model.Jy2 * beta_y

    # plot all
    plt.style.use("../settings-and-materials/paper_plot.mplstyle")
    fig = plt.figure(figsize=(3 / 0.8 * CM_IN_INCH, 2.5 / 0.8 * CM_IN_INCH))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlim([-4.2, 4.2])
    ax.set_ylim([-4.2, 4.2])
    ax.plot(E_mesh.real, E_mesh.imag, color=COLORS["ol"])
    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")
    fig.savefig("Figures/SGBZ-HN-spectrum.pdf")

    fig = plt.figure(figsize=(3 / 0.8 * CM_IN_INCH, 2.5 / 0.8 * CM_IN_INCH))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlim([-4.2, 4.2])
    ax.set_ylim([-0.61, -0.39])
    ax.plot(E_mesh.real, np.log(beta_11).real, color=COLORS["ol"])
    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("$\\mu_{[11]}$")
    fig.savefig("Figures/SGBZ-HN-mu11.pdf")

    fig = plt.figure(figsize=(3 / 0.8 * CM_IN_INCH, 2.5 / 0.8 * CM_IN_INCH))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlim([-4.2, 4.2])
    ax.set_ylim([-11.6, 7.6])
    # plt.plot(E_mesh.real, np.log(beta_y).real, 'b.')
    pcl = ax.pcolor(E_mesh.real, np.log(beta_y).real, np.zeros_like(E_mesh.real), cmap=MY_CMAP("ol"))
    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("$\\mu_y$")

    ax.set_axis_off()
    fig.savefig("Figures/SGBZ-HN-muy.png", dpi=1200, transparent=True)
    ax.set_axis_on()

    pcl.remove()
    fig.savefig("Figures/SGBZ-HN-muy.pdf")

if __name__ == "__main__":
    model = HNModel(
        1.5, -2, pi/3, pi/6, sqrt(2), 1 + 1j
    )
    # omega0 = 2

    # plot_11_SGBZ_spectrum_uniform_Jnorm(model)

    # theta_11, theta_y = get_11_SGBZ_EFC(2, model)
    # E = get_SGBZ_eigenenergy_by_definition(model, theta_11, theta_y)
    # plt.plot(E.real, E.imag, '.')
    # plt.show()

    # paper_plot(model, omega0)
    plot_mu_spectrum(model)
