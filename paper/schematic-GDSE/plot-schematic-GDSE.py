'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-12-10
Copyright © Department of Physics, Tsinghua University. All rights reserved

Calculate OBC spectrum and eigenstates for schematic diagrams.
'''

import BerryPy.TightBinding as tb
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
import pickle

DEFAULT_PARAMS = (1, 1, 2j, 2j)
plt.style.use("../settings-and-materials/paper_plot.mplstyle")

CM = 1/2.54

def HN_model(Jx1, Jx2, Jy1, Jy2) -> tb.TightBindingModel:
    inter_cell = [
        [0, 0, Jx1, (1,0)],
        [0, 0, Jx2, (-1,0)],
        [0, 0, Jy1, (0,1)],
        [0, 0, Jy2, (0,-1)]
    ]
    return tb.TightBindingModel(2, 1, np.array([[1,0],[0,1]]), [], inter_cell, np.array([[0,0]]))


def calculate_square():
    nx = 20
    ny = 16
    model = HN_model(*DEFAULT_PARAMS)
    model_2d = model.get_supercell(
        [(0, j) for j in range(ny)],
        np.array(
            [[1, 0],
             [0, ny]],
            dtype=int)
    ).get_supercell(
        [(i, 0) for i in range(nx)],
        np.array(
            [[nx, 0],
             [0, 1]],
            dtype=int)
    )

    all_coords = model_2d.lattice2cart(model_2d.SiteCoord.transpose()).transpose()

    H_mat = model_2d.get_bulk_Hamiltonian_complex((None, None)).todense()
    eigv, eigvec = la.eig(H_mat)
    with open("data/square-geometry.pkl", "wb") as fp:
        pickle.dump((nx, ny, all_coords, eigv, eigvec), fp)


def 计算斜几何能谱():
    vec_list = [
        np.array([-5.5, -5.5]),
        np.array([5.5, 5.5]),
        np.array([-4.5, 4.5]),
        np.array([4.5, -4.5])
    ]
    point_list = [
        (i, j) 
        for i in range(-20, 20)
        for j in range(-20, 20)
    ]

    斜几何点集 = []
    for point in point_list:
        在几何内 = True
        for vec in vec_list:
            vec = np.asarray(vec)
            vec_dot = np.dot(vec, point) / (la.norm(vec) ** 2)
            if vec_dot >= 1:
                在几何内 = False
                break
        if 在几何内:
            斜几何点集.append(point)
    model = HN_model(*DEFAULT_PARAMS)
    H_mat = model.generate_OBC_bulk(斜几何点集).todense()

    斜几何点集 = np.asarray(斜几何点集)

    # 画几何
    plt.plot(斜几何点集[:,0], 斜几何点集[:,1], '.')
    plt.savefig("data/斜矩形几何.pdf")
    
    eigv, eigvec = la.eig(H_mat)
    with open("data/slanted-geometry.pkl", "wb") as fp:
        pickle.dump((斜几何点集, eigv, eigvec), fp)


def 画能谱():
    with open("data/square-geometry.pkl", "rb") as fp:
        nx, ny, all_coords, eigv, eigvec = pickle.load(fp)
    plt.figure()
    plt.plot(eigv.real, eigv.imag, '.')
    plt.savefig("data/正矩形谱.pdf")

    with open("data/slanted-geometry.pkl", "rb") as fp:
        已择点集, 本征值, 本征态 = pickle.load(fp)
    plt.figure()
    已择点集 = np.asarray(已择点集)
    plt.plot(已择点集[:,0], 已择点集[:,1], '.')
    plt.savefig("data/斜矩形几何.pdf")
    plt.figure()
    plt.plot(本征值.real, 本征值.imag, '.')
    plt.savefig("data/斜矩形谱.pdf")


def draw_eigenstate_distribution_square():
    with open("data/square-geometry.pkl", "rb") as fp:
        nx, ny, all_coords, eigv, eigvec = pickle.load(fp)

    目标能量 = 0.5 + 1j
    目标ind = np.argmin(np.abs(eigv - 目标能量))
    本征态 = eigvec[:, 目标ind]
    fig = plt.figure(figsize=(4 * CM, 4 * CM))
    ax = fig.gca()
    ax.scatter(
        all_coords[:,0],
        all_coords[:,1],
        c=np.abs(本征态.flatten()),
        cmap="Blues",
        edgecolors=(0.8, 0.8, 0.8),
        s=8,
        linewidths=0.01
    )
    ax.set_aspect(1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig("Figures/eigenstate-square.pdf")


def draw_eigenstate_distribution_slanted():
    with open("data/slanted-geometry.pkl", "rb") as fp:
        all_coords, eigv, eigvec = pickle.load(fp)

    目标能量 = 0.5 + 1j
    目标ind = np.argmin(np.abs(eigv - 目标能量))
    本征态 = eigvec[:, 目标ind]
    fig = plt.figure(figsize=(4 * CM, 4 * CM))
    ax = fig.gca()
    ax.scatter(
        all_coords[:,0],
        all_coords[:,1],
        c=np.abs(本征态.flatten()),
        cmap="Oranges",
        edgecolors=(0.8, 0.8, 0.8),
        s=8,
        linewidths=0.01
    )
    ax.set_aspect(1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig("Figures/eigenstate-slanted.pdf")


if __name__ == '__main__':
    import os
    if not os.path.exists("data"):
        os.makedirs("data")
    calculate_square()
    计算斜几何能谱()
    draw_eigenstate_distribution_square()
    draw_eigenstate_distribution_slanted()
