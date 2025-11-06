'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-02-01
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

'''
    Check whether the density of states is a constant under the 
    coordinate of phi
'''

import BerryPy.TightBinding as tb
import poly_tools as pt
import partial_GBZ_solver as pGs
import GBZ_manifold as Gm
from scipy import linalg as la
import pickle
import matplotlib.pyplot as plt
import numpy as np

def check_OBC_DOS():
    ''' model: HN model with next-nearest hopping '''
    t1 = 1
    t2 = 1/2
    gamma = 0.1

    # generate tightbinding model
    dim = 1
    site_num = 1
    lattice_vec = [[1]]
    intra_cell = []
    inter_cell = [
        [0, 0, t1, (1,)],
        [0, 0, t1, (-1,)], 
        [0, 0, t2 + gamma, (2,)], 
        [0, 0, t2 - gamma, (-2,)]
    ]
    model = tb.TightBindingModel(dim, site_num, lattice_vec, intra_cell, inter_cell)

    # # calculate OBC
    # N_points = 100
    # model_super = model.get_supercell( 
    #     [(j,) for j in range(N_points)],
    #     [[N_points]]
    # )
    # H_mat = model_super.get_bulk_Hamiltonian_dense((None,))
    # eigv, eigvec = la.eig(H_mat)
    # with open("data/DOS-check-OBC.pkl", "wb") as fp:
    #     pickle.dump({
    #         'eigv': eigv, 
    #         'eigvec': eigvec
    #     }, fp)

    # generate characteristic polynomial
    coeffs = pt.CScalarVec([
        1, -t1, -t1, -t2 - gamma, -t2 + gamma
    ])
    degs = pt.CLaurentIndexVec([
        # E, beta
        1, 0,  
        0, -1,
        0, 1,
        0, -2, 
        0, 2
    ])
    char_poly = pt.CLaurent(2)
    char_poly.set_Laurent_by_terms(coeffs, degs)

    GBZ_loops, phi_loops = pGs.solve_GBZ(char_poly)
    with open("data/DOS-check-GBZ.pkl", "wb") as fp: 
        pickle.dump({
            "GBZ_loops": GBZ_loops,
            "phi_loops": phi_loops
        }, fp)

def plot_GBZ():
    with open("data/DOS-check-GBZ.pkl", "rb") as fp: 
        GBZ_data = pickle.load(fp)
    GBZ_loops = GBZ_data["GBZ_loops"]
    phi_loops = GBZ_data['phi_loops']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = fig.gca()
    # ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)), 
    #          np.sin(np.linspace(0, 2 * np.pi, 100)), 'r--')
    for j, loop in enumerate(GBZ_loops): 
        # ax.plot(loop[:,1].real, loop[:,1].imag, loop[:,0])
        # ax.plot(loop[:,1].real, loop[:,1].imag)
        ax.plot(loop[:,1].real, loop[:,1].imag, phi_loops[j][:,0]/np.pi)
    ax.set_aspect('equalxy')
    ax.set_xlabel("$\\mathrm{Re}(\\beta)$")
    ax.set_ylabel("$\\mathrm{Im}(\\beta)$")
    ax.set_zlabel("$\\phi(\pi)$")
    plt.show()

def compare_spectrum():
    fig = plt.figure()
    ax = fig.gca()
    with open("data/DOS-check-GBZ.pkl", "rb") as fp: 
        GBZ_data = pickle.load(fp)
    with open("data/DOS-check-OBC.pkl", "rb") as fp: 
        OBC_data = pickle.load(fp)
    
    GBZ_loops = GBZ_data['GBZ_loops']
    eigv = OBC_data["eigv"]

    print(len(GBZ_loops))
    loop = GBZ_loops[0]
    GBZ_plot = ax.plot(loop[:,0].real, loop[:,0].imag)
    
    OBC_plot = ax.plot(eigv.real, eigv.imag, '.')
    ax.set_xlabel("$\\mathrm{Re}E$")
    ax.set_ylabel("$\\mathrm{Im}E$")
    ax.legend(GBZ_plot + OBC_plot, ['GBZ', 'OBC'])
    plt.show()

def plot_DOS():
    with open("data/DOS-check-GBZ.pkl", "rb") as fp: 
        GBZ_data = pickle.load(fp)
    with open("data/DOS-check-OBC.pkl", "rb") as fp: 
        OBC_data = pickle.load(fp)

    GBZ_loop = GBZ_data['GBZ_loops'][0]
    phi_loop = GBZ_data['phi_loops'][0]
    eigv = OBC_data["eigv"]

    index_list = []
    k_list = []
    for curr_E in eigv:
        min_id = np.argmin(np.abs(GBZ_loop[:,0] - curr_E))
        print(np.abs(GBZ_loop[min_id, 0] - curr_E))
        index_list.append(min_id)
        k_list.append(np.log(GBZ_loop[min_id, 1]).imag)

        loop_without_min = np.delete(GBZ_loop[:,0], min_id)
        loop_without_min = GBZ_loop[:,0].copy()
        loop_without_min[min_id] += 1e8
        min_id = np.argmin(np.abs(loop_without_min - curr_E))
        print(np.abs(loop_without_min[min_id] - curr_E))
        index_list.append(min_id)
        k_list.append(np.log(GBZ_loop[min_id, 1]).imag)


    plt.plot(np.sort(index_list)/len(GBZ_loop[:,0]), '.')
    plt.xlabel("Index of the eigenstates")
    plt.ylabel("$\\phi$ linear coordinate")
    plt.show()
    plt.plot(np.sort(k_list), '.')
    plt.xlabel("Index of the eigenstates")
    plt.ylabel("$k = \\mathrm{Im}(\\ln(\\beta))$")
    plt.show()

def ssh_model(u1, u2, v1, v2, w):
    dim = 1
    site_num = 2
    intra_cell = [
        [0, 1, u1],
        [1, 0, u2]
    ]
    inter_cell = [
        [1, 0, v1, (1,)],
        [0, 1, v2, (-1,)],
        [0, 0, 0.2*w, (2,)],
        [0, 0, 0.2*w, (-2,)],
        [1, 1, 0.2*w, (2,)],
        [1, 1, 0.2*w, (-2,)],
    ]
    return tb.TightBindingModel(
        dim, site_num, [[1]], intra_cell, inter_cell
    )

def find_point_in_GBZ(
        E: complex, 
        GBZ_E: list[np.ndarray],
        GBZ_params: list[np.ndarray]):
    ''' Return the GBZ index and param '''

    min_dist = [0] * len(GBZ_E)
    min_coord = [(0,0)] * len(GBZ_E)
    for j in range(len(GBZ_E)):
        min_id = np.argmin(np.abs(GBZ_E[j] - E))
        min_dist[j] = abs(GBZ_E[j][min_id] - E)
        coord1 = GBZ_params[j][min_id]

        loop_without_min = GBZ_E[j].copy()
        loop_without_min[min_id] += 1e8
        min_id = np.argmin(np.abs(loop_without_min - E))
        coord2 = GBZ_params[j][min_id]
        min_coord[j] = (coord1, coord2)
    
    loop_id = np.argmin(min_dist)
    return loop_id, min_coord[loop_id]


def check_DOS_distribution():
    params = np.random.rand(5)
    print(params)
    model = ssh_model(*params)
    coeffs, degs = model.get_characteristic_polynomial_data()
    f = pt.CLaurent(2)
    f.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))

    glue_tol = 0.05
    GBZ_data, GBZ_param, aGBZ_data = pGs.solve_GBZ(
        'GBZ', f, glue_tol=glue_tol
    )
    GBZ_data, GBZ_param, aGBZ_data = pGs.expand_solution(
        GBZ_data, aGBZ_data, GBZ_param)
    all_segs = pGs.get_GBZ_segments(GBZ_data, GBZ_param, 10 * glue_tol, 
            pGs.aGBZEquation1DGBZCP1(f))
    all_loops = pGs.GBZ_segments_to_loops(all_segs)
    all_loops = pGs.batch_get_global(all_loops, pGs.aGBZEquation1DGBZCP1(f))
    for loop in all_loops:
        print(loop.param_list[-1][0] - loop.param_list[0][0])

    GBZ_E = []
    GBZ_coord = []
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for loop in all_loops:
        all_points = np.zeros((len(loop.point_vec), 2), dtype=complex)
        all_params = np.zeros(len(loop.point_vec))
        for j, point in enumerate(loop.point_vec):
            Gm.GBZ_to_chart(point, [0])
            all_points[j,:] = point.coords
            all_params[j] = loop.param_list[j][0]
        # ax.plot(all_points[:,1].real, all_points[:,1].imag, all_points[:,0].real, '.-')
        ax.plot(all_points[:,1].real, all_points[:,1].imag, all_params, '.-')
        GBZ_E.append(all_points[:,0])
        GBZ_coord.append(all_params)
    plt.show()

    N = 50
    model_1d = model.get_supercell(
        [(j,) for j in range(N)],
        [[N]]
    )
    H = model_1d.get_bulk_Hamiltonian_complex((None,)).todense()

    eigv, eigvec = la.eig(H)
    plt.plot(eigv.real, eigv.imag, 'x')
    for curr_E in GBZ_E:
        plt.plot(curr_E.real, curr_E.imag)
    plt.show()

    OBC_coords = [[] for _ in range(len(GBZ_E))]
    for E in eigv:
        loop_id, loop_coord = find_point_in_GBZ(E, GBZ_E, GBZ_coord)
        for x in loop_coord:
            OBC_coords[loop_id].append(x)
    
    for curr_coord in OBC_coords:
        plt.plot(np.sort(curr_coord),'x')
        A = np.polyfit(np.arange(len(curr_coord)), np.sort(curr_coord), 1)
        print(A)
        plt.plot([0, len(curr_coord)], np.polyval(A, [0, len(curr_coord)]))
    plt.show()



if __name__ == '__main__':
    # check_OBC_DOS()        
    # plot_GBZ()
    # compare_spectrum()
    # plot_DOS()
    check_DOS_distribution()
