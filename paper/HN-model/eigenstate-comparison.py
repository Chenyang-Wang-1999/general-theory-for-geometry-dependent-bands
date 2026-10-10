'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-11-09
Copyright © Department of Physics, Tsinghua University. All rights reserved

Compair the eigenstates and non-Bloch states
'''


import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
DATA_DIR = "../../puncture-calculation"
import sys
sys.path.append('..')
import figure_settings_common as fs


# cmap = cm.get_cmap('seismic')
# print(cmap)
# print(cmap.N)
# print(cmap(0))
# print(cmap(1))
# print(1 / cmap.N)
# print(cmap(cmap.N))
cmap = "coolwarm"


def print_eigenenergy():
    n1 = 320
    n2 = 40
    with open("%s/data/paper-eigenstate-compare-direct-%d-%d.pkl"%(DATA_DIR, n1, n2), "rb") as fp:
        beta_list, eigv_ind, non_Bloch_basis, coeffs, aspect, n_feature = pickle.load(fp)

    if n2 == 40:
        fname = "%s/data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (DATA_DIR, aspect, n_feature)
    else:
        fname = "%s/data/paper-HN-OBC-2-N1_%d-N2_%d.pkl" % (DATA_DIR, aspect, n_feature)

    with open(
        fname,
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)

    print("Eigenenergy:", eigv[eigv_ind])    


def plot_eigenstate_distribution():
    ''' load data and plot '''
    plt.style.use("../settings-and-materials/paper_plot.mplstyle")

    n1 = 320
    n2 = 40
    # with open("%s/data/paper-eigenstate-compare-%d-%d.pkl"%(DATA_DIR, n1, n2), "rb") as fp:
    #     beta_list, eigv_ind, GBZ_energy, non_Bloch_basis, coeffs, aspect, n_feature = pickle.load(fp)

    with open("%s/data/paper-eigenstate-compare-direct-%d-%d.pkl"%(DATA_DIR, n1, n2), "rb") as fp:
        beta_list, eigv_ind, non_Bloch_basis, coeffs, aspect, n_feature = pickle.load(fp)

    if n2 == 40:
        fname = "%s/data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (DATA_DIR, aspect, n_feature)
    else:
        fname = "%s/data/paper-HN-OBC-2-N1_%d-N2_%d.pkl" % (DATA_DIR, aspect, n_feature)

    with open(
        fname,
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)
    print(shape)
    # rotate point_vec by -pi/4
    # point_vec = point_vec @ np.array([
    #     [np.cos(np.pi/4), -np.sin(np.pi/4)],
    #     [np.sin(np.pi/4), np.cos(np.pi/4)]
    # ])
    point_vec = point_vec @ np.array([
        [1, -1],
        [0, 1]
    ])
    # get the boundary of the geometric region
    boundary_loop = [
        [0, 0],
        [shape[0], shape[0]],
        [shape[0], shape[0] + shape[1]],
        [0, shape[1]],
        [0, 0]
    ] @ np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4)],
        [np.sin(np.pi/4), np.cos(np.pi/4)]
    ])

    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    curr_eigvec = eigvec[:,eigv_ind]
    n_picked = 4
    
    # Figure settings
    vmax = max(np.abs(curr_eigvec))
    # vmax = 0.06
    print("vmax = ", vmax)

    full_view_xlim = [-10, shape[0] + 10]
    full_view_ylim = [-10, shape[1] + 10]
    full_view_x_ticks = np.linspace(0, shape[0], 5)
    full_view_y_ticks = [0, shape[1]]

    # plot selected eigenstate
    fig = plt.figure(figsize=(7.5 * fs.cm, 2 * fs.cm))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_xlim(full_view_xlim)
    ax.set_ylim(full_view_ylim)
    ax.set_xticks(full_view_x_ticks)
    ax.set_yticks(full_view_y_ticks)
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax
    )
    fig.savefig("Figures/eigenstate-distribution-%d-%d.pdf"%(shape[0], shape[1]))
    plt.colorbar(pcl)
    fig.savefig("Figures/eigenstate-distribution-colorbar-%d-%d.pdf"%(shape[0], shape[1]))

    fig = plt.figure(figsize=(7.5 * fs.cm, 2 * fs.cm))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_xlim(full_view_xlim)
    ax.set_ylim(full_view_ylim)
    ax.set_xticks(full_view_x_ticks)
    ax.set_yticks(full_view_y_ticks)
    approx_eigvec = non_Bloch_basis @ coeffs
    ax.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(approx_eigvec).reshape(shape, order='F'),
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax
    )
    fig.savefig("Figures/approx-eigenstate-distribution-%d-%d.pdf"%(shape[0], shape[1]))

    fig = plt.figure(figsize=(7.5 * fs.cm, 2 * fs.cm))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_xlim(full_view_xlim)
    ax.set_ylim(full_view_ylim)
    ax.set_xticks(full_view_x_ticks)
    ax.set_yticks(full_view_y_ticks)
    diff_vector = curr_eigvec - approx_eigvec
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        # (diff_vector.real).reshape(shape, order='F'),
        np.abs(diff_vector).reshape(shape, order='F'),
        cmap='Reds',
        vmax=max(np.abs(diff_vector)),
        vmin=0
    )
    fig.savefig("Figures/diff-eigenstate-distribution-%d-%d.pdf"%(shape[0],shape[1]))
    plt.colorbar(pcl)
    fig.savefig("Figures/diff-eigenstate-distribution-colorbar-%d-%d.pdf"%(shape[0],shape[1]))

    fig = plt.figure(figsize=(4.5 * fs.cm, 4 * fs.cm))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_ylim([-2,shape[1] + 2])
    ax.set_xlim([200,240])
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax
    )
    fig.savefig("Figures/eigenstate-distribution-zoomed-in-%d-%d.pdf"%(shape[0], shape[1]))

    fig = plt.figure(figsize=(4.5 * fs.cm, 4 * fs.cm))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_ylim([-2,shape[1] + 2])
    ax.set_xlim([200,240])
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax
    )
    fig.savefig("Figures/approx-eigenstate-distribution-zoomed-in-%d-%d.pdf"%(shape[0], shape[1]))


    # plt.figure()
    # plt.plot(curr_eigvec.reshape(shape, order='F')[:,0].real)
    # plt.plot(approx_eigvec.reshape(shape, order='F')[:,0].real)
    # plt.figure()
    # plt.plot(curr_eigvec.reshape(shape, order='F')[:,-1].real)
    # plt.plot(approx_eigvec.reshape(shape, order='F')[:,-1].real)
    # plt.show()


def plot_eigenstate_zoomed_in():
    plt.style.use("../settings-and-materials/paper_plot.mplstyle")

    n1 = 320
    n2 = 40
    # with open("%s/data/paper-eigenstate-compare-%d-%d.pkl"%(DATA_DIR, n1, n2), "rb") as fp:
    #     beta_list, eigv_ind, GBZ_energy, non_Bloch_basis, coeffs, aspect, n_feature = pickle.load(fp)

    with open("%s/data/paper-eigenstate-compare-direct-%d-%d.pkl"%(DATA_DIR, n1, n2), "rb") as fp:
        beta_list, eigv_ind, non_Bloch_basis, coeffs, aspect, n_feature = pickle.load(fp)

    if n2 == 40:
        fname = "%s/data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (DATA_DIR, aspect, n_feature)
    else:
        fname = "%s/data/paper-HN-OBC-2-N1_%d-N2_%d.pkl" % (DATA_DIR, aspect, n_feature)

    with open(
        fname,
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)
    print(shape)
    # rotate point_vec by -pi/4
    # point_vec = point_vec @ np.array([
    #     [np.cos(np.pi/4), -np.sin(np.pi/4)],
    #     [np.sin(np.pi/4), np.cos(np.pi/4)]
    # ])
    point_vec = point_vec @ np.array([
        [1, -1],
        [0, 1]
    ])
    # get the boundary of the geometric region
    boundary_loop = [
        [0, 0],
        [shape[0], shape[0]],
        [shape[0], shape[0] + shape[1]],
        [0, shape[1]],
        [0, 0]
    ] @ np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4)],
        [np.sin(np.pi/4), np.cos(np.pi/4)]
    ])

    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    curr_eigvec = eigvec[:,eigv_ind]
    n_picked = 4
    
    # Figure settings
    vmax = max(np.abs(curr_eigvec))
    # vmax = 0.06
    print("vmax = ", vmax)

    # full_view_xlim = [-10, shape[0] + 10]
    full_view_x_ticks = np.linspace(0, shape[0], 5)
    full_view_xlim = [full_view_x_ticks[1], full_view_x_ticks[-2]]
    full_view_x_ticks = full_view_x_ticks[1:-1]
    upper_slice_ylim = [shape[1] - 6, shape[1]]
    upper_slice_yticks = np.linspace(shape[1] - 6, shape[1], 2)
    lower_slice_ylim = [-1, 6]
    lower_slice_yticks = np.linspace(0, 6, 2)

    fig = plt.figure(figsize=(7.5 * fs.cm, 1 * fs.cm))
    ax = fig.gca()
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_ylim(lower_slice_ylim)
    ax.set_yticks(lower_slice_yticks)
    ax.set_xlim(full_view_xlim)
    ax.set_xticks(full_view_x_ticks)
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax
    )
    fig.savefig("Figures/eigenstate-distribution-sliced-lower-%d-%d.pdf"%(shape[0], shape[1]))

    fig = plt.figure(figsize=(7.5 * fs.cm, 1 * fs.cm))
    ax = fig.gca()
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_ylim(upper_slice_ylim)
    ax.set_yticks(upper_slice_yticks)
    ax.set_xlim(full_view_xlim)
    ax.set_xticks(full_view_x_ticks)
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax
    )
    fig.savefig("Figures/eigenstate-distribution-sliced-upper-%d-%d.pdf"%(shape[0], shape[1]))


    approx_eigvec = non_Bloch_basis @ coeffs
    fig = plt.figure(figsize=(7.5 * fs.cm, 1 * fs.cm))
    ax = fig.gca()
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_ylim(lower_slice_ylim)
    ax.set_yticks(lower_slice_yticks)
    ax.set_xlim(full_view_xlim)
    ax.set_xticks(full_view_x_ticks)
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax
    )
    fig.savefig("Figures/approx-eigenstate-distribution-sliced-lower-%d-%d.pdf"%(shape[0], shape[1]))

    fig = plt.figure(figsize=(7.5 * fs.cm, 1 * fs.cm))
    ax = fig.gca()
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_ylim(upper_slice_ylim)
    ax.set_yticks(upper_slice_yticks)
    ax.set_xlim(full_view_xlim)
    ax.set_xticks(full_view_x_ticks)
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        # np.abs(curr_eigvec).reshape(shape, order='F')
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax
    )
    fig.savefig("Figures/approx-eigenstate-distribution-sliced-upper-%d-%d.pdf"%(shape[0], shape[1]))


def get_eigenstate_difference(n1, n2):
    ''' Plot the average of the absolute value of the difference of eigenstates'''
    # 1. load data
    # with open("%s/data/paper-eigenstate-compare-%d-%d.pkl"%(DATA_DIR, n1, n2), "rb") as fp:
    #     beta_list, eigv_ind, GBZ_energy, non_Bloch_basis, coeffs, aspect, n_feature = pickle.load(fp)
    
    with open("%s/data/paper-eigenstate-compare-direct-%d-%d.pkl"%(DATA_DIR, n1, n2), "rb") as fp:
        beta_list, eigv_ind, non_Bloch_basis, coeffs, aspect, n_feature = pickle.load(fp)
    
    if n2 == 40:
        fname = "%s/data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (DATA_DIR, aspect, n_feature)
    else:
        fname = "%s/data/paper-HN-OBC-2-N1_%d-N2_%d.pkl" % (DATA_DIR, aspect, n_feature)

    with open(
        fname,
        'rb'
    ) as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)
    
    # 2. get difference of the eigenstate and approximate eigenstate
    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')
    curr_eigvec = eigvec[:,eigv_ind]
    approx_eigvec = non_Bloch_basis @ coeffs
    diff_vector = curr_eigvec - approx_eigvec

    approx_eigvec_mesh = approx_eigvec.reshape(shape, order='F')
    mean_approx_upper = np.mean(np.sum(np.abs(approx_eigvec_mesh[:,shape[1]//2:]) ** 2, axis=1))
    mean_approx_lower = np.mean(np.sum(np.abs(approx_eigvec_mesh[:,:shape[1]//2]) ** 2, axis=1))
    diff_vector_mesh = (np.abs(diff_vector)**2).reshape(shape, order='F')
    diff_vector_lower = np.sum(diff_vector_mesh[:,:shape[1]//2] / mean_approx_lower, axis=1)
    diff_vector_upper = np.sum(diff_vector_mesh[:,shape[1]//2:] / mean_approx_upper, axis=1)
    return x_mesh[:,0], diff_vector_lower, diff_vector_upper


def plot_eigenstate_difference():
    x_30, diff_lower_30, diff_upper_30 = get_eigenstate_difference(320, 30)
    x_40, diff_lower_40, diff_upper_40 = get_eigenstate_difference(320, 40)
    x_50, diff_lower_50, diff_upper_50 = get_eigenstate_difference(320, 50)

    plt.style.use("../settings-and-materials/paper_plot.mplstyle")
    xlim = [0, 320]
    xticks = np.linspace(0, 320, 5)

    fig = plt.figure(figsize=(4 * fs.cm, 4 * fs.cm))
    ax = fig.gca()
    ax.set_xlabel("n1")
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_ylim([0, 2])
    ax.plot(x_30, diff_lower_30, '--', label="$L_2 = 30$")
    ax.plot(x_40, diff_lower_40, '-', label="$L_2 = 40$")
    ax.plot(x_50, diff_lower_50, ':', label="$L_2 = 50$")
    ax.legend()
    fig.savefig("Figures/eigenstate-difference-lower.pdf")

    fig = plt.figure(figsize=(4 * fs.cm, 4 * fs.cm))
    ax = fig.gca()
    ax.set_xlabel("n1")
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_ylim([0, 2])
    plt.plot(x_30, diff_upper_30, '--', label="$L_2 = 30$")
    plt.plot(x_40, diff_upper_40, '-', label="$L_2 = 40$")
    plt.plot(x_50, diff_upper_50, ':', label="$L_2 = 50$")
    ax.legend()
    fig.savefig("Figures/eigenstate-difference-upper.pdf")


def plot_average_eigenstate_distribution():
    n_feature = 20
    aspect = 8.0
    rotation_angle = 0

    with open("%s/data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl" % (DATA_DIR, aspect, n_feature), "rb") as fp:
        eigv, eigvec, point_vec, params, shape = pickle.load(fp)
    
    # Normalize eigen vectors
    eigvec = eigvec / np.linalg.norm(eigvec, axis=0)

    # Calculate average of the absolute value of the eigenvectors
    avg_eigvec = np.mean(np.abs(eigvec), axis=1)

    # rotate point_vec
    point_vec = point_vec @ np.array([
        [np.cos(rotation_angle), np.sin(rotation_angle)],
        [-np.sin(rotation_angle), np.cos(rotation_angle)]
    ])

    x_mesh = point_vec[:,0].reshape(shape, order='F')
    y_mesh = point_vec[:,1].reshape(shape, order='F')

    # plot
    pcl = plt.pcolor(x_mesh, y_mesh, avg_eigvec.reshape(shape, order='F'),
               vmin=0, vmax=0.03)
    plt.colorbar(pcl)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    # plot_eigenstate_distribution()
    # plot_eigenstate_zoomed_in()
    # get_eigenstate_difference(320, 30)
    # plot_eigenstate_difference()
    # print_eigenenergy()
    plot_average_eigenstate_distribution()
