'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-02-28
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' Check GBZ condition for the minimal non-trivial model '''

import poly_tools as pt
import partial_GBZ_solver as pGs
import GBZ_manifold as Gm
import numpy as np
import matplotlib.pyplot as plt
# import pyvista as pv
import BerryPy.TightBinding as tb
import pickle
import os
import sys
import slepc_solver

from scipy import linalg as la
from scipy import optimize
from scipy.sparse import linalg as spla
from scipy import sparse

import feast

from math import pi
from multiprocessing import Pool
import pickle

####### pickle util #######
def pickle_dump(fname:str, data):
    with open(fname, "wb") as fp:
        pickle.dump(data, fp)

def pickle_load(fname:str):
    with open(fname, "rb") as fp:
        data = pickle.load(fp)
    return data

####### define model #######
coeffs = pt.CScalarVec([
    1, -1, -1, -1, -1
])
degs = pt.CLaurentIndexVec([
  # E, beta1, beta2
    1, 0, 0,
    0, 1, 0,
    0, -1, 0,
    0, 1, -1,
    0, -2, 1
])
f = pt.CLaurent(3)
f.set_Laurent_by_terms(coeffs, degs)

####### BerryPy model #######
def minimal_nontrivial_model()->tb.TightBindingModel:
    intra_cell = []
    inter_cell = [
        [0, 0, 1, (1,0)],
        [0, 0, 1, (-1,0)],
        [0, 0, 1, (-1, 1)],
        [0, 0, 1, (2, -1)]
    ]

    return tb.TightBindingModel(2, 1,[[1,0], [0,1]], intra_cell, inter_cell, [[0,0]])


####### xmajor #######
def winding_x_major(E: complex, beta_loop: np.ndarray, s_loop: np.ndarray):
    ''' get the winding number of x major partial solution around given E and given loop '''
    minor_beta = (np.abs(beta_loop) ** 1.5) * np.exp(1j * s_loop)
    fun_val = np.zeros_like(beta_loop, dtype=complex)
    for j in range(len(fun_val)):
        fun_val[j] = f.eval(pt.CVarScalarVec([E, beta_loop[j], minor_beta[j]]))
    
    return fun_val

def get_x_major_OBC():
    ''' Direct calculation of the OBC spectrum with major x '''
    Nx = 100
    Ny = 50
    model = minimal_nontrivial_model()

    # 1. get point list
    point_list = []
    for nx in range(Nx):
        for ny in range(Ny):
            point_list.append((nx, ny))
    
    # 2. get the Hamiltionian and spectrum
    H = model.generate_OBC_bulk(point_list)

    all_eigvs = []
    for j1, E_real in enumerate(np.linspace(-2.5, 4.5, 50)):
        for j2, E_imag in enumerate(np.linspace(-2.5, 2.5, 50)):
            print(j1, j2)
            curr_E = E_real + 1j * E_imag
            eigv = spla.eigs(H - curr_E * sparse.eye(H.shape[0]), k=1, which='SM', 
                             return_eigenvectors=False)
            all_eigvs.append(eigv + curr_E)
    
    all_eigvs = np.hstack(all_eigvs)
    pickle_dump("data/minimal-nontrivial-OBC-Nx_%d-Ny_%d-sparse-sweep.pkl"%(Nx, Ny), 
                (all_eigvs))

            

    # eigv, eigvec = la.eig(H.todense())

    # # 3. save
    # pickle_dump("data/minimal-nontrivial-OBC-Nx_%d-Ny_%d.pkl"%(Nx, Ny), 
    #             (point_list, eigv, eigvec))


def get_x_major_punctures(E: complex, N_points: int) -> np.ndarray:
    ''' get punctures with given E for x major '''
    s_list = np.linspace(0, pi, N_points)
    cos_s_list = np.cos(s_list)
    all_sols = np.zeros((N_points, 4), dtype=complex)

    # solve equations
    for j in range(N_points):
        all_sols[j,:] = np.roots(
            [1, 0, -E, 2 * cos_s_list[j], 1]
        )
    
    beta_major = all_sols ** 2
    beta_minor = np.zeros_like(beta_major, dtype=complex)
    for j in range(4):
        beta_minor[:,j] = all_sols[:,j] ** 3 * np.exp(-1j * s_list)

    return beta_major, beta_minor


def calculate_x_major_supercell_GBZ():
    ''' calculate the supercell 1D GBZ by BerryPy '''
    N_points = 15
    model = minimal_nontrivial_model()
    model_1d = model.get_supercell(
        [(0,j) for j in range(N_points)],
        [
            [1,0],
            [0,N_points]
        ]
    )
    model_1d = model_1d.truncate(1)

    _, E, beta = model_1d.get_GBZ(N_process=6)
    pickle_dump("data/x-major-supercell-GBZ-N_%d.pkl"%(N_points), (E, beta))

def calculate_y_major_supercell_GBZ():
    ''' calculate the supercell 1D GBZ by BerryPy '''
    N_points = 15
    model = minimal_nontrivial_model()
    model_1d = model.get_supercell(
        [(j,0) for j in range(N_points)],
        [
            [N_points,0],
            [0,1]
        ]
    )
    model_1d = model_1d.truncate(0)

    _, E, beta = model_1d.get_GBZ(N_process=6)
    pickle_dump("data/y-major-supercell-GBZ-N_%d.pkl"%(N_points), (E, beta))



def main_plot_x_major_punctures():
    ''' scripts for get_x_major_punctures '''
    E = 2.38002801621341
    beta_major_num = 0.8562533298501848+1.092401856758521j
    print("beta norm", abs(beta_major_num))
    N_points = 200
    beta_major, beta_minor = get_x_major_punctures(E, N_points)

    exp_is = beta_minor/(beta_major ** 1.5)
    beta_minor_another = beta_major ** 1.5 * np.conjugate(exp_is)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    char_poly_test = pGs.CharPolyInfo(f)
    for j in range(4):
        for point_id in range((beta_major.shape[0])):
            detE, dbeta = char_poly_test.get_det_and_dbeta(E, beta_major[point_id,j], beta_minor[point_id,j], beta_minor_another[point_id,j])
            if(detE > 0):
                color = 'r'
            else:
                color = 'b'
            ax.plot(beta_major[point_id,j].real, 
                    beta_major[point_id,j].imag, 
                    np.log(beta_minor[point_id,j]).imag / pi, color + '.')
    ax.set_xlabel("$\\beta_1$ real")
    ax.set_ylabel("$\\beta_1$ imag")
    ax.set_zlabel("$\\beta_2$ argument")
    
    fig = plt.figure()
    ax = fig.gca()
    for j in range(4):
        ax.plot(beta_major[:,j].real, beta_major[:,j].imag, '.')
    
    num_circle = beta_major_num * np.exp(1j * np.linspace(0, 2 * pi, 500))
    ax.plot(num_circle.real, num_circle.imag, '--')
    num_circle = beta_major_num * 1.025 * np.exp(1j * np.linspace(0, 2 * pi, 500))
    ax.plot(num_circle.real, num_circle.imag, '--')
    ax.set_xlabel("$\\beta_1$ real")
    ax.set_ylabel("$\\beta_1$ imag")

    # plt.figure()
    # plt.plot(f_loop.real, f_loop.imag)

    fig = plt.figure()
    ax = fig.gca()

    # beta1_list = [0.8562533298501848+1.092401856758521j, 0.8562533298501848-1.092401856758521j, 1.3879] # |beta|
    # print(np.abs(beta1_list))
    # new_beta = 0.83529 + 1.14301j # 1.02 |beta|
    new_beta = 0.83009 + 1.15536j # 1.025 |beta|
    beta1_list = [new_beta, new_beta.conjugate(), abs(new_beta)] 
    print(np.abs(beta1_list))

    beta1 = beta1_list[0]
    x = np.sqrt(beta1)
    s = np.arccos((x**4 - E * x ** 2 + 1)/(-2 * x)).real
    beta2_1 = np.exp(-1j * s) * x ** 3
    beta2_2 = np.exp(1j * s) * x ** 3
    print(np.abs(np.log(beta2_2).imag - np.log(beta2_1).imag))
    ax.plot(np.log(beta1).imag, np.log(beta2_1).imag, 'bx')
    ax.plot([-np.pi,np.pi], [np.log(beta2_1).imag, np.log(beta2_1).imag], 'b--')
    ax.plot(np.log(beta1).imag, np.log(beta2_2).imag, 'rx')
    ax.plot([-np.pi,np.pi], [np.log(beta2_2).imag, np.log(beta2_2).imag], 'r--')
    ax.set_xlabel("$\\beta_1$ argument")
    ax.set_ylabel("$\\beta_2$ argument")


    beta1 = beta1_list[1]
    x = np.sqrt(beta1)
    s = np.arccos((x**4 - E * x ** 2 + 1)/(-2 * x)).real
    beta2_1 = np.exp(-1j * s) * x ** 3
    beta2_2 = np.exp(1j * s) * x ** 3
    print(np.abs(np.log(beta2_2).imag - np.log(beta2_1).imag))
    ax.plot(np.log(beta1).imag, np.log(beta2_1).imag, 'bx')
    ax.plot([-np.pi,np.pi], [np.log(beta2_1).imag, np.log(beta2_1).imag], 'b--')
    ax.plot(np.log(beta1).imag, np.log(beta2_2).imag, 'rx')
    ax.plot([-np.pi,np.pi], [np.log(beta2_2).imag, np.log(beta2_2).imag], 'r--')

    beta1 = beta1_list[2]
    x = np.sqrt(beta1)
    s = np.arccos((x**4 - E * x ** 2 + 1)/(-2 * x)).real
    beta2_1 = np.exp(-1j * s) * x ** 3
    beta2_2 = np.exp(1j * s) * x ** 3
    print(np.abs(np.log(beta2_2).imag - np.log(beta2_1).imag))
    ax.plot(np.log(beta1).imag, np.log(beta2_1).imag, 'rx')
    ax.plot([-np.pi,np.pi], [np.log(beta2_1).imag, np.log(beta2_1).imag], 'r--')
    ax.plot(np.log(beta1).imag, np.log(beta2_2).imag, 'bx')
    ax.plot([-np.pi,np.pi], [np.log(beta2_2).imag, np.log(beta2_2).imag], 'b--')

    s_list = [-2.3, -1.7, -1, 0, 1, 1.7, 2.3]
    for s in s_list:
        f_loop = winding_x_major(E, abs(beta1)*np.exp(1j * np.linspace(0, 2*np.pi, 100)), s * np.ones(100))
        w = (pGs.get_winding_number(f_loop))
        ax.text(-0.8 * np.pi, s, str(int(np.round(w))))
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])

    plt.show()

def compare_condition_number():
    point_list, eigv, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_200-Ny_50.pkl")
    new_eigv, new_eigvec, picked_eigv, curr_r = pickle_load("data/minimal-OBC-full-solve-Nx200-Ny50-%d.pkl"%(0))
    print(curr_r)
    # neighbor_id = np.argsort(np.abs(eigv - picked_eigv))[:100]
    neighbor_id = (np.abs(eigv-picked_eigv) < 0.08)
    neighbor_states = eigvec[:,neighbor_id]
    print("Original:", np.linalg.cond(neighbor_states))
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(eigv[neighbor_id].real, eigv[neighbor_id].imag, '.')
    plt.plot(picked_eigv.real, picked_eigv.imag, 'x')
    plt.show()

    # neighbor_id = np.argsort(np.abs(new_eigv - picked_eigv))[:100]
    neighbor_id = (np.abs(new_eigv-picked_eigv) < 0.08)
    neighbor_states = new_eigvec[:,neighbor_id]
    print("Scaled:", np.linalg.cond(neighbor_states))
    plt.plot(new_eigv.real, new_eigv.imag, '.')
    plt.plot(new_eigv[neighbor_id].real, new_eigv[neighbor_id].imag, '.')
    plt.plot(picked_eigv.real, picked_eigv.imag, 'x')
    plt.show()


def plot_x_major_OBC_spectrum():
    ''' plot OBC spectrum '''
    point_list, eigv, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_200-Ny_50.pkl")


    selected_id = 100
    selected_state = eigvec[:,selected_id]
    selected_eigv = eigv[selected_id]

    eigv_diff = np.abs(eigv - selected_eigv)
    neighbor_id = np.argsort(eigv_diff)[:100]
    neighbor_eigv = eigv[neighbor_id]
    neighbor_eigvec = eigvec[:,neighbor_id]
    
    print(np.linalg.cond(neighbor_eigvec))
    plt.plot(eigv.real, eigv.imag, '.')
    # plt.plot(neighbor_eigv.real, neighbor_eigv.imag, 'x')
    plt.show()

    point_arr = np.asarray(point_list)
    sca = plt.scatter(point_arr[:,0], point_arr[:,1], c=np.abs(selected_state))
    clb = plt.colorbar(sca)
    print(selected_eigv)
    plt.show()


def plot_x_major_supercell():
    ''' plot the data calculated by calculate_x_major '''
    N_points = 15
    E, beta = pickle_load("data/x-major-supercell-GBZ-N_%d.pkl"%(N_points))
    point_list, eigv, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_200-Ny_50.pkl")
    point_list, eigv2, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_200-Ny_50.pkl")

    point_selected = (np.abs(E.imag) < 1e-5) & (E.real > 1.6)
    print("E = ", E[point_selected][3])
    print("beta = ", beta[point_selected][3])

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(E.real, E.imag, '.')
    ax.plot(E[point_selected][3].real, E[point_selected][3].imag, 'x')
    ax.set_xlabel("$E$ real")
    ax.set_ylabel("$E$ imag")
    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(beta.real, beta.imag, E.real, '.')
    ax.set_xlabel("$\\beta$ real")
    ax.set_ylabel("$\\beta$ imag")
    ax.set_zlabel("$E$ real")

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(eigv.real, eigv.imag, '.')
    ax.plot(eigv2.real, eigv2.imag, '.')
    ax.plot(E.real, E.imag, 'x')
    ax.set_xlabel("$E$ real")
    ax.set_ylabel("$E$ imag")
    ax.legend(["OBC spectrum 200 x 50", "Supercell GBZ along x"])

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(beta.real, beta.imag, '.')
    ax.set_xlabel("$\\beta$ real")
    ax.set_ylabel("$\\beta$ imag")

    plt.show()

def plot_supercell_spectrum():
    N_points = 15
    E1, beta1 = pickle_load("data/x-major-supercell-GBZ-N_%d.pkl"%(N_points))
    E2, beta2 = pickle_load("data/y-major-supercell-GBZ-N_%d.pkl"%(N_points))
    point_list, eigv, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_200-Ny_50.pkl")

    plt.figure(1)
    plt.plot(E1.real, E1.imag, '.')
    plt.plot(E2.real, E2.imag, '.')
    fig = plt.figure(2)
    plt.plot(eigv.real, eigv.imag, '.', picker=True)
    plt.show()

def calculate_GBZ_distance_x_major(major_beta: complex, E: complex, poly_info: pGs.CharPolyInfo):
    ''' Given a major beta, calculate the GBZ distance '''
    x = np.sqrt(major_beta)
    s = np.arccos((x**4 - E * x ** 2 + 1)/(-2 * x)).real
    beta2_1 = np.exp(-1j * s) * x ** 3
    beta2_2 = np.exp(1j * s) * x ** 3

    detE, dbeta1 = poly_info.get_det_and_dbeta(E, major_beta, beta2_1, beta2_2)
    if(detE > 0):
        topo_charge = 1
    else:
        topo_charge = -1

    return beta2_1, beta2_2, topo_charge

def calculate_total_winding(E: complex, beta2_list: list[complex], beta1_norm: float, topo_charge_list: list):
    # 0. check existence of critical points
    if(len(beta2_list) == 0):
        f_loop = winding_x_major(E, beta1_norm*np.exp(1j * np.linspace(0, 2*np.pi, 100)), 0 * np.ones(100))
        return (pGs.get_winding_number(f_loop)) * 2 * np.pi

    # 1. sort all the beta by arg
    beta2_args = np.log(beta2_list).imag
    beta2_seq = np.argsort(beta2_args)
    beta2_args = beta2_args[beta2_seq]
    beta2_list = [beta2_list[j] for j in beta2_seq]
    topo_charge_list = [topo_charge_list[j] for j in beta2_seq]

    # 2. find largest diff
    beta2_arg_diff = np.hstack([2 * np.pi + (beta2_args[0] - beta2_args[-1]), beta2_args[1:] - beta2_args[:-1]])
    max_id = np.argmax(beta2_arg_diff)

    # # 2.5 check winding
    # winding_numerical = np.zeros_like(beta2_arg_diff)
    # for curr_id in range(len(winding_numerical)):
    #     if(curr_id == 0):
    #         initial_phase = (2 * np.pi + beta2_args[0] + beta2_args[-1]) / 2
    #     else:
    #         initial_phase = (beta2_args[curr_id] + beta2_args[curr_id - 1]) / 2
    #     f_loop = winding_x_major(E, beta1_norm*np.exp(1j * np.linspace(0, 2*np.pi, 100)), initial_phase * np.ones(100))
    #     # print("initial phase:", initial_phase, "min value:",min(np.abs(f_loop)))
    #     winding_numerical[curr_id] = (pGs.get_winding_number(f_loop))
        


    # 3. get initial winding
    if(max_id == 0):
        initial_phase = (2 * np.pi + beta2_args[0] + beta2_args[-1]) / 2
    else:
        initial_phase = (beta2_args[max_id] + beta2_args[max_id - 1]) / 2

    f_loop = winding_x_major(E, beta1_norm*np.exp(1j * np.linspace(0, 2*np.pi, 1000)), initial_phase * np.ones(1000))
    w = (pGs.get_winding_number(f_loop))

    # 4. get all winding
    all_windings = np.zeros_like(beta2_arg_diff)
    curr_id = max_id
    curr_winding = w

    all_windings[curr_id] = curr_winding
    curr_winding += topo_charge_list[curr_id]
    curr_id = (curr_id + 1) % (len(topo_charge_list))
    while(curr_id != max_id):
        all_windings[curr_id] = curr_winding
        curr_winding += topo_charge_list[curr_id]
        curr_id = (curr_id + 1) % (len(topo_charge_list))
   
    return all_windings.dot(beta2_arg_diff)
        

def main_calculate_puncture_distance():
    E = 2.38002801621341
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1,2])
    )
    poly_info = pGs.CharPolyInfo(f)

    beta_loop, phi_loop = pGs.solve_aGBZ("CP1", f_partial, N_points=200, glue_tol=0.05)
    GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DCP1(f_partial), beta_loop, 1e-4, 'CP1')
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.5)
    all_segments = pGs.expand_segments(all_segments)
    all_loops = pGs.GBZ_segments_to_loops(all_segments)
    all_loops = pGs.batch_glue(all_loops)

    beta_norm = abs(0.83009 + 1.15536j)

    beta_list = []
    for loop in all_loops:
        beta_list += pGs.find_critical_beta(beta_norm, loop)
    
    beta2_list = []
    topo_charge_list = []
    for beta in beta_list:
        beta2_1, beta2_2, topo_charge = calculate_GBZ_distance_x_major(beta, E, poly_info)
        beta2_list += [beta2_1]
        topo_charge_list += [topo_charge]
        # beta2_list += [beta2_1, beta2_2]
        # topo_charge_list += [topo_charge, -topo_charge]
    
    print(calculate_total_winding(E, beta2_list, beta_norm, topo_charge_list))
    
    # print(abs(0.83009 + 1.15536j))
    
    # plt.figure(2)
    # plt.plot(beta_norm, distance, '.-')
    # plt.plot(beta_norm, 0 * beta_norm)
    # plt.show()

def main_sweep_beta_norm():
    ''' Sweep beta norm and calculate winding number '''

    E = 2.38002801621341
    # E = 6
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1,2])
    )
    poly_info = pGs.CharPolyInfo(f)

    beta_loop, phi_loop = pGs.solve_aGBZ("CP1", f_partial, N_points=200, glue_tol=0.05)
    GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DCP1(f_partial), beta_loop, 1e-4, 'CP1')
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.5)
    all_segments = pGs.expand_segments(all_segments)
    all_loops = pGs.GBZ_segments_to_loops(all_segments)
    all_loops = pGs.batch_glue(all_loops)

    # 1. calculate holes
    all_holes, all_intervals = pGs.get_holes_from_loops(all_loops)
    fig = plt.figure()
    ax = fig.gca()
    for loop in all_loops:
        beta_list = np.zeros((len(loop.point_vec), 2), dtype=complex)
        for j, point in enumerate(loop.point_vec):
            for k in range(2):
                if(point.chart_labels[k] == 0):
                    beta_list[j,k] = point.coords[k]
                else:
                    beta_list[j,k] = 1/point.coords[k]
        ax.plot(beta_list[:,0].real, beta_list[:,0].imag)
    
    for curr_hole in all_holes:
        beta_norm = np.sqrt(
            (1 + curr_hole) / (1 - curr_hole)
        )
        theta = np.linspace(0, 2 * np.pi, 500)
        ax.plot(beta_norm * np.cos(theta), 
                beta_norm * np.sin(theta), '--')
        curr_winding = winding_x_major(E, beta_norm * np.exp(1j * theta), np.zeros_like(theta))
        print(beta_norm, int(np.round(pGs.get_winding_number(curr_winding))))

    # 2. calculate total winding number
    N_theta = 1000
    theta_list = np.linspace(np.pi/(N_theta + 1), np.pi - np.pi/(N_theta + 1), N_theta)
    total_winding = np.zeros_like(theta_list)
    for theta_id, theta in enumerate(theta_list):
        norm_cos_theta = np.cos(theta)
        beta_norm = 1/np.tan(theta/2)
        # 1. find all critical segments
        all_segs = []
        all_params = []
        for loop in all_loops:
            curr_segs, curr_params = pGs.find_critical_segments(norm_cos_theta, loop)
            all_segs += curr_segs
            all_params += curr_params
        
        # print(beta_norm, len(all_segs))
        # 2. label these segments
        labeled_segs = pGs.label_critical_segments(
            poly_info, E, all_segs, all_params
        )

        # 3. pair all segments
        paired_segs = pGs.pair_critical_segments(labeled_segs)
        if(paired_segs is None):
            print("None detected")
            total_winding[theta_id] = np.nan
            raise ValueError()

        # 3.5 check the norms of beta2
        # for curr_index in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        #     for curr_seg in paired_segs[curr_index]:
                # print(abs(curr_seg[0].coords[1]), abs(curr_seg[1].coords[1]))

        # 4. Iteratively calculate critical beta
        all_pairs = pGs.calculate_all_critical_points(
            poly_info, pGs.aGBZEquation1DCP1(f_partial), paired_segs, E, norm_cos_theta, 1e-5
        )
        if(all_pairs is None):
            print("None detected")
            total_winding[theta_id] = np.nan
            raise ValueError()

        # 5. Calculate total winding
        beta2_list = []
        topo_charge_list = []
        for curr_pair in all_pairs:
            for j in range(2):
                if(curr_pair[j].chart_labels[1] == 0):
                    beta2_list.append(curr_pair[j].coords[1])
                else:
                    beta2_list.append(1/curr_pair[j].coords[1])
                topo_charge_list.append(curr_pair[j].topo_charge)
        total_winding[theta_id] = (calculate_total_winding(E, beta2_list, beta_norm, topo_charge_list))
    
    # Visualization
    print(abs(0.83009 + 1.15536j))
    plt.figure()
    plt.plot(1/np.tan(theta_list), total_winding, '.-')
    plt.show()

def get_x_major_GBZ_2D(E:float):
    ''' For given E, get x major 2D GBZ. If GBZ do not exists, return empty list '''

    # 1. solve GBZ loops    
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1,2])
    )
    poly_info = pGs.CharPolyInfo(f)

    GBZ_loops, phi_loop, _ = pGs.solve_GBZ("CP1", f_partial, N_points=200, glue_tol=0.05)
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.06)
    all_segments = pGs.expand_segments(all_segments)
    all_loops = pGs.GBZ_segments_to_loops(all_segments)
    all_loops = pGs.batch_glue(all_loops)

    # 2. find zero holes
    all_holes, all_intervals = pGs.get_holes_from_loops(all_loops)
    negative_list = []
    positive_list = []
    for j, curr_hole in enumerate(all_holes):
        beta_norm = np.sqrt(
            (1 + curr_hole) / (1 - curr_hole)
        )
        theta = np.linspace(0, 2 * np.pi, 500)
        curr_winding = winding_x_major(E, beta_norm * np.exp(1j * theta), np.zeros_like(theta))
        topo_charge = np.round(pGs.get_winding_number(curr_winding))
        if(topo_charge == 0):
            return []
        elif(topo_charge < 0):
            negative_list.append(curr_hole)
        else:
            positive_list.append(curr_hole)
    
    # 3. bisect between two holes
    def calculate_total_winding_given_costheta(norm_cos_theta: float):
        beta_norm = np.sqrt(
            (1 + norm_cos_theta) / (1 - norm_cos_theta)
        )
        # 1. find all critical segments
        all_segs = []
        all_params = []
        for loop in all_loops:
            curr_segs, curr_params = pGs.find_critical_segments(norm_cos_theta, loop)
            all_segs += curr_segs
            all_params += curr_params
        
        # print(beta_norm, len(all_segs))
        # 2. label these segments
        labeled_segs = pGs.label_critical_segments(
            poly_info, E, all_segs, all_params
        )

        # 3. pair all segments
        paired_segs = pGs.pair_critical_segments(labeled_segs)
        if(paired_segs is None):
            raise ValueError("paired_segs is None")

        # 4. Iteratively calculate critical beta
        all_pairs = pGs.calculate_all_critical_points(
            poly_info, pGs.aGBZEquation1DCP1(f_partial), paired_segs, E, norm_cos_theta, 1e-5
        )
        if(all_pairs is None):
            raise ValueError("all_pairs is None")

        # 5. Calculate total winding
        beta2_list = []
        topo_charge_list = []
        for curr_pair in all_pairs:
            for j in range(2):
                if(curr_pair[j].chart_labels[1] == 0):
                    beta2_list.append(curr_pair[j].coords[1])
                else:
                    beta2_list.append(1/curr_pair[j].coords[1])
                topo_charge_list.append(curr_pair[j].topo_charge)
        w = (calculate_total_winding(E, beta2_list, beta_norm, topo_charge_list))
        return w
    
    def get_2D_GBZ_given_beta(norm_cos_theta:float):
        # 1. find all critical segments
        all_segs = []
        all_params = []
        for loop in all_loops:
            curr_segs, curr_params = pGs.find_critical_segments(norm_cos_theta, loop)
            all_segs += curr_segs
            all_params += curr_params
        
        # 2. label these segments
        labeled_segs = pGs.label_critical_segments(
            poly_info, E, all_segs, all_params
        )

        # 3. pair all segments
        paired_segs = pGs.pair_critical_segments(labeled_segs)
        if(paired_segs is None):
            raise ValueError("paired_segs is None")

        # 4. Iteratively calculate critical beta
        all_pairs = pGs.calculate_all_critical_points(
            poly_info, pGs.aGBZEquation1DCP1(f_partial), paired_segs, E, norm_cos_theta, 1e-5
        )
        if(all_pairs is None):
            raise ValueError("all_pairs is None")

        return all_pairs

   
    class FunAvoidNone:
        detour_distance: float
        def __init__(self, detour_distance):
            self.detour_distance = detour_distance
        def __call__(self, norm_cos_theta: float):
            curr_val = calculate_total_winding_given_costheta(norm_cos_theta)
            if(curr_val is None):
                theta = np.arccos(norm_cos_theta)
                val1 = calculate_total_winding_given_costheta(np.cos(theta + self.detour_distance))
                val2 = calculate_total_winding_given_costheta(np.cos(theta - self.detour_distance))
                return (val1 + val2)/2
            else:
                return curr_val

    x_tol = 1e-6
    fun_call = FunAvoidNone(x_tol)
    x_critical = optimize.brentq(fun_call, negative_list[-1], positive_list[0], xtol=x_tol,maxiter=100)
    return get_2D_GBZ_given_beta(x_critical)

def get_x_major_spectrum(E: complex):
    # 1. solve GBZ loops    
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1,2])
    )

    beta_loop, phi_loop = pGs.solve_aGBZ("CP1", f_partial, N_points=200, glue_tol=0.05)
    GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DCP1(f_partial), beta_loop, 1e-4, 'CP1')
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.5)
    all_segments = pGs.expand_segments(all_segments)
    all_loops = pGs.GBZ_segments_to_loops(all_segments)
    all_loops = pGs.batch_glue(all_loops)

    # 2. find zero holes
    all_holes, all_intervals = pGs.get_holes_from_loops(all_loops)
    for j, curr_hole in enumerate(all_holes):
        beta_norm = np.sqrt(
            (1 + curr_hole) / (1 - curr_hole)
        )
        theta = np.linspace(0, 2 * np.pi, 500)
        curr_winding = winding_x_major(E, beta_norm * np.exp(1j * theta), np.zeros_like(theta))
        topo_charge = np.round(pGs.get_winding_number(curr_winding))
        if(topo_charge == 0):
            return False
    return True
    
 
def get_y_major_spectrum(E: complex):
    f_y_major = f.partial_eval(
        pt.CScalarVec([]), pt.CIndexVec([]),
        pt.CIndexVec([0,2,1])
    )
    # 1. solve GBZ loops    
    solver = pGs.GBZMainSolver(f_y_major)
    err_code = solver.get_beta_GBZ(E, no_glue=True)
    all_holes = solver.GBZ_holes

    print(all_holes)
    for j, curr_hole in enumerate(all_holes[1:-1]):
        beta_norm = np.sqrt(
            (1 + curr_hole) / (1 - curr_hole)
        )
        try:
            E_loop = pGs.calculate_hole_winding_loop(
                f_y_major, beta_norm
            )
        except:
            print("Warning: solver failed for E = %s"%(str(E)), file=sys.stderr)
            return False
        topo_charge = int(np.round(pGs.get_winding_number(E-E_loop)))
        print(curr_hole)
        print(topo_charge)
        if(topo_charge == 0):
            return False
    return True
    
class XMajorMainSolver(pGs.GBZMainSolver):
    def get_beta_GBZ(self, E: complex, N_points: int = 200) -> pGs.GBZSolverErrorCode:
        err_code =  super().get_beta_GBZ(E, N_points)
        # calculate winding number
        if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
            self.no_zero_hole = True
            max_neg_id = 0
            for j, curr_hole in enumerate(self.GBZ_holes):
                beta_norm = np.sqrt(
                    (1 + curr_hole) / (1 - curr_hole)
                )
                theta = np.linspace(0, 2 * np.pi, 500)
                f_loop = winding_x_major(E, beta_norm * np.exp(1j * theta), np.zeros_like(theta))
                w = int(np.round(pGs.get_winding_number(f_loop)))
                if(w == 0):
                    self.no_zero_hole = False
                    return err_code
                elif(w < 0):
                    max_neg_id = j
            if(self.no_zero_hole):
                self.central_interval = self.GBZ_intervals[max_neg_id]

        return err_code


    def get_total_winding(self, E: complex, norm_cos_theta):
        all_beta2 = []
        all_charge = []
        positive_beta, negative_beta = self.critical_beta_pair
        for j in range(len(positive_beta)):
            Gm.CP1_to_chart(positive_beta[j][0], [0,0])
            Gm.CP1_to_chart(negative_beta[j][0], [0,0])
            all_beta2.append(positive_beta[j][0].coords[1])
            all_charge.append(1)
            all_beta2.append(negative_beta[j][0].coords[1])
            all_charge.append(-1)
        beta_norm = np.sqrt(
            (1 + norm_cos_theta) / (1 - norm_cos_theta)
        )
        self.curr_total_winding = calculate_total_winding(E, all_beta2, beta_norm, all_charge)
        return pGs.GBZSolverErrorCode.SOLVER_SUCCESS
    
    def total_winding_brentq_fun(self, norm_cos_theta, E: complex):
        ''' Assuming the beta_GBZ is calculated '''
        err_code = self.get_critical_beta(E, norm_cos_theta)
        self.err_state = err_code
        if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
            err_code = self.get_total_winding(E, norm_cos_theta)
            return self.curr_total_winding
        else:
            self.dumped_data = (norm_cos_theta, E)
            return 0
    
    def robust_get_critical_betanorm(self, E: complex, N_retry: int = 30):
        def post_process_solver_success(E, res):
            self.critical_beta_norm = res
            err_code = self.get_critical_beta(E, res)
            for perturb_dist in [-self.beta_perturb_dist, self.beta_perturb_dist]:
                if(err_code != pGs.GBZSolverErrorCode.CRITICAL_BETA_NOT_CONVERGED):
                    break
                err_code = self.get_critical_beta(E, res + perturb_dist)
                
            if(err_code == pGs.GBZSolverErrorCode.CRITICAL_BETA_NOT_CONVERGED):
                return err_code
            else:
                all_GBZ_points = []
                positive_points, negative_points = self.critical_beta_pair
                for curr_pair in positive_points:
                    Gm.CP1_to_chart(curr_pair[0], [0,0])
                    all_GBZ_points.append((E, curr_pair[0].coords[0], curr_pair[0].coords[1], curr_pair[1], +1))
                for curr_pair in negative_points:
                    Gm.CP1_to_chart(curr_pair[0], [0,0])
                    all_GBZ_points.append((E, curr_pair[0].coords[0], curr_pair[0].coords[1], curr_pair[1], -1))
                self.all_GBZ_points = all_GBZ_points
                return err_code

        critical_interval = self.central_interval
        for _ in range(N_retry):
            res = optimize.brentq(self.total_winding_brentq_fun, 
                        critical_interval[0], critical_interval[1], args=(E,),
                        xtol=self.critical_beta_tol)
            if(self.err_state == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
                return post_process_solver_success(E, res)
            else:
                # critical points
                critical_norm_beta = self.dumped_data[0]
                pGs.raise_warning("Warning: critical norm beta detected for %f"%(critical_norm_beta))
                curr_perturb = critical_norm_beta
                for _ in range(3):
                    curr_perturb += self.beta_perturb_dist
                    err_code = self.get_critical_beta(E, curr_perturb)
                    if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
                        break

                if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
                    err_code = self.get_total_winding(E, curr_perturb)
                    wp = self.curr_total_winding
                else:
                    pGs.raise_warning("Warning: brentq solver fails after perturbation outside.")
                    return err_code
                
                if(wp < 0):
                    critical_interval[0] = curr_perturb
                else:
                    curr_perturb = critical_norm_beta
                    for _ in range(3):
                        curr_perturb -= self.beta_perturb_dist
                        err_code = self.get_critical_beta(E, curr_perturb)
                        if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
                            break

                    if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
                        err_code = self.get_total_winding(E, curr_perturb)
                        wm = self.curr_total_winding
                    else:
                        pGs.raise_warning("Warning: brentq solver fails after perturbation inside.")
                        return err_code
                    
                    if(wm > 0):
                        critical_interval[1] = curr_perturb
                    else:
                        return post_process_solver_success(E, critical_norm_beta)

        return pGs.GBZSolverErrorCode.E_CRITICAL
 
####### tests #######
# def check_partial_GBZ_continuity():
#     ''' check the continuity of the partial GBZ in beta1 '''
#     beta2_list = np.exp(1j * np.linspace(0, 2 * np.pi, 50))
#     for beta2 in beta2_list:
#         pl = pv.Plotter()
#         f_eff = f.partial_eval(
#             pt.CScalarVec([beta2]), 
#             pt.CIndexVec([2]),
#             pt.CIndexVec([0,1])
#         )
#         GBZ_loops, phi_loops = pGs.solve_GBZ(f_eff, rel_tol=1e-4)
#         print(len(GBZ_loops))
#         all_phis = np.vstack(phi_loops).flatten()
#         print("max phi = ", np.max(all_phis), ", min phi = ", np.min(all_phis))
#         loop = np.vstack(GBZ_loops)
#         print(loop.shape[0])

#         pl.add_points(np.column_stack([loop[:,1].real, loop[:,1].imag, all_phis]))
    
#         pl.show()

def test_segment_selection_and_pairing():
    E = 2.38002801621341
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1,2])
    )
    poly_info = pGs.CharPolyInfo(f)

    beta_loop, phi_loop = pGs.solve_aGBZ("CP1", f_partial, N_points=200, glue_tol=0.05)
    GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DCP1(f_partial), beta_loop, 1e-4, 'CP1')
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.5)
    all_segments = pGs.expand_segments(all_segments)
    all_loops = pGs.GBZ_segments_to_loops(all_segments)
    all_loops = pGs.batch_glue(all_loops)

    # beta_norm = abs(0.83009 + 1.15536j)
    beta_norm = 1.0975
    norm_cos_theta = (beta_norm ** 2 - 1) / (beta_norm ** 2 + 1)
    # 1. find all critical segments
    all_segs = []
    all_params = []
    for loop in all_loops:
        curr_segs, curr_params = pGs.find_critical_segments(norm_cos_theta, loop)
        all_segs += curr_segs
        all_params += curr_params
    
    # 2. label these segments
    labeled_segs = pGs.label_critical_segments(
        poly_info, E, all_segs, all_params
    )

    # 3. pair all segments
    paired_segs = pGs.pair_critical_segments(labeled_segs)

    # # output
    # if(True):
    #     for index in [(1,1), (-1,-1), (1,-1), (-1,1)]:
    #         print(index)
    #         for j in range(len(paired_segs[index])):
    #             for l in range(2):
    #                 for k in range(2):
    #                     if(paired_segs[index][j][l].chart_labels[k] == 0):
    #                         print(paired_segs[index][j][l].coords[k], end=',')
    #                     else:
    #                         print(1/paired_segs[index][j][l].coords[k], end=',')
    #                 print("||||", end='')
    #             print()

    # 4. Iteratively calculate critical beta
    all_pairs = pGs.calculate_all_critical_points(
        poly_info, pGs.aGBZEquation1DCP1(f_partial), paired_segs, E, norm_cos_theta, 1e-5
    )
                
    # 5. Calculate total winding
    beta2_list = []
    topo_charge_list = []
    for curr_pair in all_pairs:
        for j in range(2):
            if(curr_pair[j].chart_labels[1] == 0):
                beta2_list.append(curr_pair[j].coords[1])
            else:
                beta2_list.append(1/curr_pair[j].coords[1])
            topo_charge_list.append(curr_pair[j].topo_charge)
    print(calculate_total_winding(E, beta2_list, beta_norm, topo_charge_list))

    
    # visualization
    fig = plt.figure()
    ax = fig.gca()
    for loop in all_loops:
        beta = np.zeros((len(loop.point_vec), 2), dtype=complex)
        for point_id, point in enumerate(loop.point_vec):
            for j in range(2):
                if(point.chart_labels[j] == 0):
                    beta[point_id, j] = point.coords[j]
                else:
                    beta[point_id, j] = 1.0 / point.coords[j]
        ax.plot(beta[:,0].real, beta[:,0].imag, '.-')
    
    # for seg in all_segs:
    #     curr_beta = [0,0]
    #     for j in range(2):
    #         if(seg[j].chart_labels[0] == 0):
    #             curr_beta[j] = seg[j].coords[0]
    #         else:
    #             curr_beta[j] = 1/seg[j].coords[0]
    #     ax.plot(curr_beta[0].real, curr_beta[0].imag, 'x')
    #     ax.plot(curr_beta[1].real, curr_beta[1].imag, '+')

    for pair in all_pairs:
        curr_beta = [0,0]
        for j in range(2):
            if(pair[j].chart_labels[0] == 0):
                curr_beta[j] = pair[j].coords[0]
            else:
                curr_beta[j] = 1/pair[j].coords[0]
        ax.plot(curr_beta[0].real, curr_beta[0].imag, 'x')
        ax.plot(curr_beta[1].real, curr_beta[1].imag, '+')
    
    theta = np.linspace(0, 2*np.pi, 500)
    ax.plot(beta_norm * np.cos(theta), beta_norm * np.sin(theta))
    
    plt.show()

def debug_critical_point_calculation():
    E = -2
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1,2])
    )
    poly_info = pGs.CharPolyInfo(f)
    char_poly_obj = pGs.aGBZEquation1DCP1(f_partial)

    GBZ_loops, phi_loop, _ = pGs.solve_GBZ("CP1", f_partial, N_points=200, glue_tol=0.05)

    # for loop in beta_loop:
    #     curr_plt = []
    #     for point in loop:
    #         if(point.chart_labels[0] == 0):
    #             plt.plot(point.coords[0].real, point.coords[0].imag, 'b.')
    #             curr_plt += plt.plot(point.coords[0].real, point.coords[0].imag, 'ro')
    #         else:
    #             plt.plot((1/point.coords[0]).real, (1/point.coords[0]).imag, 'b.')
    #             curr_plt += plt.plot((1/point.coords[0]).real, (1/point.coords[0]).imag, 'ro')
    #     plt.pause(0.1)
    #     for item in curr_plt:
    #         item.remove()
    # plt.show()

    # GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DCP1(f_partial), beta_loop, 1e-4, 'CP1')
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.06)
    all_segments = pGs.expand_segments(all_segments)
    all_loops = pGs.GBZ_segments_to_loops(all_segments)
    for loop in all_loops:
        pGs.glue_loop(loop, char_poly_obj, glue_tol=1e-6)
    


    
    norm_cos_theta = -5.000000002220446e-07
    # norm_cos_theta = -5.000000002220446e-01
    beta_norm = np.sqrt((1 + norm_cos_theta) / (1 - norm_cos_theta))

    # 1. find all critical segments
    all_segs = []
    for loop in all_loops:
        for seg in loop:
            all_segs += pGs.find_critical_segments(norm_cos_theta, seg)

    # 2. Iteratively calculate critical beta
    all_points = pGs.batch_calculate_critical_points(
        char_poly_obj, all_segs, norm_cos_theta, 1e-5
    )

    # plot the loops 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for loop in all_loops:
        for seg in loop:
            curr_beta = np.zeros((len(seg.point_vec), 2), dtype=complex)
            for j, point in (enumerate(seg.point_vec)):
                Gm.CP1_to_chart(point, [0,0])
                curr_beta[j,:] = point.coords
            ax.plot(curr_beta[:,1].real, curr_beta[:,1].imag, np.abs(curr_beta[:,0]), '.-')
            ax.plot(curr_beta[0,1].real, curr_beta[0,1].imag, np.abs(curr_beta[0,0]), 'x')
            ax.plot(curr_beta[-1,1].real, curr_beta[-1,1].imag, np.abs(curr_beta[-1,0]), '+')
    
    for point_pair in all_points:
        Gm.CP1_to_chart(point_pair[0], [0,0])
        ax.plot(point_pair[0].coords[1].real, point_pair[0].coords[1].imag, np.abs(point_pair[0].coords[0]), 'd')

    # plt.plot(beta_norm * np.cos(np.linspace(0, 2 * pi, 1000)),
    #          beta_norm * np.sin(np.linspace(0, 2 * pi, 1000)), 'r--')
    plt.show()

def test_GBZ_derivative():
    E = 2
    # 1. solve GBZ loops    
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1,2])
    )

    beta_loop, phi_loop = pGs.solve_aGBZ("CP1", f_partial, N_points=10000, glue_tol=0.05)
    char_poly_obj = pGs.aGBZEquation1DCP1(f_partial)
    GBZ_loops = pGs.pick_GBZ(char_poly_obj, beta_loop, 1e-4, 'CP1')
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.5)
    all_segments = pGs.expand_segments(all_segments)

    # for segs in all_segments:
    #     for j in range(len(segs.point_vec) - 1):
    #         curr_point = Gm.CP1PointData(
    #             segs.point_vec[j].coords, segs.point_vec[j].chart_labels
    #         )
    #         next_point = Gm.CP1PointData(
    #             segs.point_vec[j + 1].coords, segs.point_vec[j + 1].chart_labels
    #         )
    #         Gm.CP1_to_chart(curr_point, [0,0])
    #         Gm.CP1_to_chart(next_point, [0,0])
    #         curr_s = segs.param_list[j][0]
    #         next_s = segs.param_list[j + 1][0]
    #         ds = next_s - curr_s
    #         dp = pGs.get_GBZ_diff(f_partial, curr_point.coords, curr_s)
    #         if(dp is not None):
    #             print(np.array(next_point.coords) - np.array(curr_point.coords))
    #             print(dp * ds)
    #             print(np.linalg.norm(dp * ds))
    #             print(np.linalg.norm(np.array(next_point.coords) - np.array(curr_point.coords) - dp * ds))
    #             input()

    for segs in all_segments:
        for j in range(len(segs.point_vec) - 1):
            curr_point = segs.point_vec[j]
            curr_s = segs.param_list[j][0]
            ds = segs.param_list[j + 1][0] - curr_s
            dp = pGs.get_GBZ_diff_CP1(char_poly_obj, curr_point, curr_s)
            if(not dp is None):
                new_point = Gm.CP1PointData(
                    [curr_point.coords[k] + dp[k] * ds for k in range(2)],
                    curr_point.chart_labels
                )
                print("current chart labels:", curr_point.chart_labels)
                print("Point distance:", Gm.get_distance(curr_point, segs.point_vec[j+1]))
                print("ds = ", ds)
                print(Gm.get_distance(new_point, segs.point_vec[j+1]))
                input()
 
def test_segment_join():
    E = -2
    f_partial = f.partial_eval(
        pt.CVarScalarVec([E]), pt.CIndexVec([0]), pt.CIndexVec([1,2])
    )
    GBZ_loops, phi_loop, aGBZ_loop = pGs.solve_GBZ("CP1", f_partial, N_points=200, glue_tol=0.05)
    # GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DGBZCP1(f_partial), beta_loop, 1e-4, 'GBZ')
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.1)
    all_segments = pGs.expand_segments(all_segments)
    all_loops = pGs.GBZ_segments_to_loops(all_segments)
    with open("data/temp_segments.pkl", "wb") as fp:
        pickle.dump(all_loops, fp)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.gca()
    print(len(all_loops))
    for loop in all_loops:
        print(len(loop))
        for seg in loop:
            beta_list = np.zeros((len(seg.point_vec), 2), dtype=complex)
            for j, point in enumerate(seg.point_vec):
                # Gm.GBZ_to_chart(point, [0])
                Gm.CP1_to_chart(point, [0,0])
                beta_list[j,:] = point.coords
            # ax.plot(beta_list[:,1].real, beta_list[:,1].imag, beta_list[:,0].real)
            ax.plot(beta_list[:,0].real, beta_list[:,0].imag)

    plt.show()

def test_segment_join2():
    E = -2
    f_partial = f.partial_eval(
        pt.CVarScalarVec([E]), pt.CIndexVec([0]), pt.CIndexVec([1,2])
    )
    char_poly_obj = pGs.aGBZEquation1DCP1(f_partial)

    # load segments
    with open("data/temp_segments.pkl", "rb") as fp:
        all_loops = pickle.load(fp)
   
    for loop in all_loops:
        pGs.glue_loop(loop, char_poly_obj)

    fig = plt.figure()
    ax = fig.gca()
    for loop in all_loops:
        for seg in loop:
            beta_list = np.zeros((len(seg.point_vec), 2), dtype=complex)
            for j, point in enumerate(seg.point_vec):
                Gm.CP1_to_chart(point, [0,0])
                beta_list[j,:] = point.coords
            ax.plot(beta_list[:,0].real, beta_list[:,0].imag, '.-')
            ax.plot(beta_list[0,0].real, beta_list[0,0].imag, '+')
            ax.plot(beta_list[-1,0].real, beta_list[-1,0].imag, 'x')
    plt.show()
                
def test_new_api():
    E = -2
    solver = pGs.GBZMainSolver(f)
    for _ in range(10):
        err_code = solver.get_beta_GBZ(E)
        if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
            break
        pGs.raise_warning("Warning: solver failed, recalculating ...")
    
    all_loops = solver.beta_GBZ
    for loop in all_loops:
        for seg in loop:
            curr_beta = np.zeros((len(seg.point_vec), 2), dtype=complex)
            for j, point in enumerate(seg.point_vec):
                new_point = Gm.CP1PointData(point.coords, point.chart_labels)
                Gm.CP1_to_chart(new_point, [0,0])
                curr_beta[j,:] = new_point.coords
            plt.plot(curr_beta[:,0].real, curr_beta[:,0].imag, '.-')
    plt.show()

    err_code = solver.get_critical_beta(E, -1e-5)
    print(err_code)
    if(err_code == 0 or err_code == 3):
        positive_points, negative_points = solver.critical_beta_pair
        for j in range(len(positive_points)):
            print(positive_points[j][0].coords, negative_points[j][0].coords)

def test_partial_GBZ_calcualtion():
    E = 5
    solver = pGs.GBZMainSolver(f.partial_eval(
        pt.CVarScalarVec([]), pt.CIndexVec([]), pt.CIndexVec([0,2,1])
    ))
    solver.get_beta_GBZ(E, no_glue=True)

    all_holes = np.asarray(solver.GBZ_holes)
    all_beta_norm = np.sqrt((1 + all_holes) / (1 - all_holes))

    # 1. plot
    all_loops = solver.beta_GBZ
    for loop in all_loops:
        for seg in loop:
            curr_beta = np.zeros((len(seg.point_vec), 2), dtype=complex)
            for point_id, point in enumerate(seg.point_vec):
                Gm.CP1_to_chart(point, [0,0])
                curr_beta[point_id,:] = point.coords
            plt.plot(curr_beta[:,0].real, curr_beta[:,0].imag)
    for curr_beta_norm in all_beta_norm[1:-1]:
        theta = np.linspace(0, 2 * pi, 500)
        plt.plot(
            curr_beta_norm * np.cos(theta),
            curr_beta_norm * np.sin(theta),
            'r--'
        )
    plt.show()

    # 2. calculate winding loop of each hole
    for hole_id in range(1, len(all_beta_norm)-1):
        E_loop = pGs.calculate_hole_winding_loop(
            f, all_beta_norm[hole_id], glue_tol=1
        )
        print(pGs.get_winding_number(E_loop))

    # GBZ_loops, params, _ = pGs.solve_partial_GBZ(
    #     f, all_beta_norm[hole_id], glue_tol=1
    # )
    # # print(all_beta_norm[hole_id] ** 1.5)

    # for curr_points in GBZ_loops:
    #     for point in curr_points:
    #         Gm.GBZ_to_chart(point, [0])
    #         plt.plot(point.coords[0].real, point.coords[0].imag, 'b.')
    # plt.show()

    # all_segments = pGs.get_GBZ_segments(GBZ_loops, params, 1)

    # for seg in all_segments:
    #     curr_data = np.zeros((len(seg.point_vec), 2), dtype=complex)
    #     for point_id, point in enumerate(seg.point_vec):
    #         Gm.GBZ_to_chart(point, [0])
    #         curr_data[point_id,:] = point.coords
    #     plt.plot(curr_data[:,0].real, curr_data[:,0].imag)
    # plt.show()

    # 3. plot winding number
    # fig1 = plt.figure(1)
    # ax1 = fig1.gca()
    # fig2 = plt.figure(2)
    # ax2 = fig2.gca()
    # fig3 = plt.figure(3)
    # ax3 = fig3.gca()
    # # print(len(params))
    # for seg in all_segments:
    #     curr_data = np.zeros((len(seg.point_vec), 2), dtype=complex)
    #     curr_params = np.zeros(len(seg.point_vec))
    #     for point_id, point in enumerate(seg.point_vec):
    #         Gm.GBZ_to_chart(point, [0])
    #         curr_data[point_id,:] = point.coords
    #         ax3.plot(all_beta_norm[hole_id] * np.cos(seg.param_list[point_id][0]),
    #                  all_beta_norm[hole_id] * np.sin(seg.param_list[point_id][0]), '.')
    #         # print(f.eval(pt.CVarScalarVec([point.coords[0], all_beta_norm[hole_id] * np.exp(1j * seg.param_list[point_id][0]), point.coords[1]])))
    #         curr_params[point_id] = seg.param_list[point_id][0]
    #     ax1.plot(curr_data[:,0].real, curr_data[:,0].imag)
    #     ax2.plot(curr_data[:,1].real, curr_data[:,1].imag)
    #     # ax3.plot(curr_params)
    # ax1.plot(E.real, E.imag, 'x')

    # for phi in (np.linspace(0, 2 * pi, 100)):
    #     print(phi)
    #     f_partial = f.partial_eval(
    #         pt.CScalarVec([all_beta_norm[hole_id]*np.exp(1j * phi)]), pt.CIndexVec([1]), pt.CIndexVec([0,2])
    #     )
    #     partial_GBZ, partial_GBZ_param, _ = pGs.solve_GBZ(
    #         "GBZ", f_partial, glue_tol=1
    #     )
    #     partial_GBZ_segments = pGs.get_GBZ_segments(partial_GBZ, partial_GBZ_param, 1)
    #     partial_GBZ_segments = pGs.expand_segments(partial_GBZ_segments)
    #     partial_GBZ_loops = pGs.GBZ_segments_to_loops(partial_GBZ_segments)
    #     partial_GBZ_loops = pGs.batch_get_global(partial_GBZ_loops)
    #     print("Number of partial GBZ:", len(partial_GBZ_loops))
    #     for loop in partial_GBZ_loops:
    #         curr_data = np.zeros((len(loop.point_vec), 2), dtype=complex)
    #         for point_id, point in enumerate(loop.point_vec):
    #             Gm.GBZ_to_chart(point, [0])
    #             curr_data[point_id,:] = point.coords
    #         ax1.plot(curr_data[:,0].real, curr_data[:,0].imag, ':')
    

    # plt.show()

    # for phi in np.linspace(0, 2*np.pi, 100):
    
    # beta2 = all_beta_norm[hole_id] ** 1.5 * np.exp(1j * phi)
    # beta1 = all_beta_norm[hole_id] * np.exp(1j * np.linspace(0, 2*pi, 500))
    # beta2 = npe.continuous_sqrt(beta1) ** 3
    # f_val = beta1 + 1/beta1 + beta1/beta2 + beta2/(beta1**2)
    # plt.plot(f_val.real, f_val.imag)
    # plt.show()

def check_y_major_GBZ():
    E_super, beta_super = pickle_load("data/y-major-supercell-GBZ-N_15.pkl")
    E = -0.02 + 0.06j

    solver = pGs.GBZMainSolver(f.partial_eval(
        pt.CVarScalarVec([]), pt.CIndexVec([]), pt.CIndexVec([0,2,1])
    ))
    # solver = pGs.GBZMainSolver(f)
    solver.get_beta_GBZ(E, no_glue=True)

    all_holes = np.asarray(solver.GBZ_holes)
    all_beta_norm = np.sqrt((1 + all_holes) / (1 - all_holes))

    # 1. plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    all_loops = solver.beta_GBZ
    for loop in all_loops:
        for seg in loop:
            curr_beta = np.zeros((len(seg.point_vec), 2), dtype=complex)
            z_coord = np.zeros(len(seg.point_vec))
            for point_id, point in enumerate(seg.point_vec):
                Gm.CP1_to_chart(point, [0,0])
                z_coord[point_id] = Gm.get_z_coord(point, 0)
                curr_beta[point_id,:] = point.coords
            ax.plot(curr_beta[:,1].real, curr_beta[:,1].imag, z_coord)
    print(all_holes)
    # for curr_beta_norm in all_beta_norm[1:-1]:
    #     theta = np.linspace(0, 2 * pi, 500)
    #     plt.plot(
    #         curr_beta_norm * np.cos(theta),
    #         curr_beta_norm * np.sin(theta),
    #         'r--'
    #     )
    plt.show()

    # 2. calculate winding loop of each hole
    for hole_id in range(1, len(all_beta_norm)-1):
        E_loop = pGs.calculate_hole_winding_loop(
            f, all_beta_norm[hole_id], glue_tol=1
        )
        print(pGs.get_winding_number(E_loop))
    
    norm_cos_theta = np.linspace(-0.003, 0.001, 21)
    GBZ_distance_list = np.zeros_like(norm_cos_theta)
    for j, curr_theta in enumerate(norm_cos_theta):
        err_code = solver.get_critical_beta(E, curr_theta)
        if(err_code == 0):
            err_code = solver.get_partial_GBZ_distance(E)
            if(err_code == 0):
                print(solver.partial_GBZ_distance)
                GBZ_distance_list[j] = np.sum(solver.partial_GBZ_distance) % 1
            else:
                print("GBZ calculation error code:", err_code)
                GBZ_distance_list[j] = np.nan
        else:
            print("beta finding error code:", err_code)
            GBZ_distance_list[j] = np.nan

    plt.plot(norm_cos_theta, GBZ_distance_list, '.-')        
    plt.show()



def check_y_major_partial_GBZ():

    E = -0.08 + 0.06j

    solver = pGs.GBZMainSolver(f.partial_eval(
        pt.CVarScalarVec([]), pt.CIndexVec([]), pt.CIndexVec([0,2,1])
    ))
    # solver = pGs.GBZMainSolver(f)
    solver.get_beta_GBZ(E, no_glue=True)

    all_holes = np.asarray(solver.GBZ_holes)
    all_beta_norm = np.sqrt((1 + all_holes) / (1 - all_holes))

    norm_cos_theta =  0.26
    solver.get_critical_beta(E, norm_cos_theta)
    positive_points, negative_points = solver.critical_beta_pair
    curr_positive = positive_points[0][0]
    curr_negative = negative_points[0][0]
    Gm.CP1_to_chart(curr_positive, [0,0])
    Gm.CP1_to_chart(curr_negative, [0,0])

    major_beta = curr_positive.coords[0]
    solver.get_partial_GBZ(major_beta)
    partial_GBZ = solver.partial_GBZ

    # plot GBZ
    GBZ_coords = np.zeros((len(partial_GBZ[0].point_vec), 2), dtype=complex)
    for j, point in enumerate(partial_GBZ[0].point_vec):
        Gm.GBZ_to_chart(point, [0])
        GBZ_coords[j,:] = point.coords

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(GBZ_coords[:,1].real, GBZ_coords[:,1].imag, GBZ_coords[:,0].real)
    ax.plot(curr_positive.coords[1].real, curr_positive.coords[1].imag, E.real, 'rx')
    ax.plot(curr_negative.coords[1].real, curr_negative.coords[1].imag, E.real, 'bx')

    # plot coordinate
    GBZ_param = np.asarray(partial_GBZ[0].param_list).flatten()
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(GBZ_param, np.abs(GBZ_coords[:,0] - E))

    # calculate GBZ distance
    solver.get_partial_GBZ_distance(E)
    print(solver.partial_GBZ_distance)
    print((np.log(curr_negative.coords[1] / curr_positive.coords[1]).imag / (2 * np.pi)) % 1)

    plt.show()


def check_x_major_OBC_calculation():
    Nx = 50
    Ny = 50
    model = minimal_nontrivial_model()

    # 1. get point list
    point_list = []
    for nx in range(Nx):
        for ny in range(Ny):
            point_list.append((nx, ny))
    
    # 2. get the Hamiltionian and spectrum
    H = model.generate_OBC_bulk(point_list)

    point_list, eigv, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_%d-Ny_%d.pkl"%(Nx, Ny))
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(eigv.real, eigv.imag, '.', picker=True)
    
    def print_error(event):
        curr_ind = event.ind[0]
        print(la.norm(H @ eigvec[:,curr_ind] - eigv[curr_ind] * eigvec[:,curr_ind]))
    fig.canvas.mpl_connect('pick_event', print_error)
    plt.show()

def debug_winding_error():
    E = 2 + 0.2j
    beta_norm = 0.9068

    beta = beta_norm * np.exp(1j * np.linspace(0, 2 * pi, 1000))
    for s in np.linspace(-pi, pi, 100):
        f_val = winding_x_major(E, beta, s * np.ones_like(beta))
        plt.plot(f_val.real, f_val.imag)
        plt.plot(0,0,'r.')
        plt.title(s)
        plt.pause(1)
    plt.show()

    return 
    norm_cos_theta = (beta_norm ** 2 - 1) / (beta_norm ** 2 + 1)
    solver = pGs.GBZMainSolver(f)
    err_code = solver.robust_get_beta_GBZ(E)

    all_loops = solver.beta_GBZ
    # 1. find all critical segments
    all_segs = []
    for loop in all_loops:
        for seg in loop:
            all_segs += pGs.find_critical_segments(norm_cos_theta, seg)

    # 3. Label topological charge
    all_topo_charges = pGs.label_critical_segments(
        solver.char_poly_info, E, all_segs
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for loop in all_loops:
        for seg in loop:
            curr_beta = np.zeros((len(seg.point_vec), 2), dtype=complex) 
            for point_id, point in enumerate(seg.point_vec):
                Gm.CP1_to_chart(point, [0,0])
                curr_beta[point_id, :] =  point.coords
            ax.plot(curr_beta[:,1].real, curr_beta[:,1].imag, np.abs(curr_beta[:,0]), 'g-')
    
    beta2_list = []
    topo_charge_list = []
    for seg_id, seg in enumerate(all_segs):
        for j in range(2):
            Gm.CP1_to_chart(seg.point_vec[j], [0,0])
        point = np.array([
            seg.point_vec[0].coords,
            seg.point_vec[1].coords
        ])
        p = pGs.iteratively_solve_critical_points(
            solver.char_poly_CP1,
            all_segs[seg_id],
            norm_cos_theta,
            solver.critical_beta_tol,
            solver.zero_tol
        )
        Gm.CP1_to_chart(p[0], [0,0])
        beta1 = p[0].coords[0]
        beta2 = p[0].coords[1]
        dE = np.array([
            -1j * beta1/ beta2 + 1j * beta2 / (beta1**2), 
            1j * beta1 - 1j / beta1 + 1j * beta1 / beta2 - 2j * beta2 / (beta1**2)
        ])
        dE_mat = np.array([
            dE.real,
            dE.imag
        ])
        print(all_topo_charges[seg_id])
        print(np.log(beta1).imag, np.log(beta2).imag)
        print(solver.char_poly_info.get_det_and_dbeta(E, p[0].coords[0], p[0].coords[1], 
                        p[0].coords[1]* np.exp(-2j*p[1])))
        print(la.det(dE_mat))

        topo_charge_list.append(all_topo_charges[seg_id])
        beta2_list.append(beta2)

        if(all_topo_charges[seg_id] == 1):
            ax.plot(point[:,1].real, point[:,1].imag, np.abs(point[:,0]), 'r.')
            ax.plot(p[0].coords[1].real, p[0].coords[1].imag, np.abs(p[0].coords[0]), 'rx')
        else:
            ax.plot(point[:,1].real, point[:,1].imag, np.abs(point[:,0]), 'b.')
            ax.plot(p[0].coords[1].real, p[0].coords[1].imag, np.abs(p[0].coords[0]), 'bx')
    
    plt.show()

    print(calculate_total_winding(E, beta2_list, beta_norm, topo_charge_list))

def debug_eigenvalue_calculation():
    ''' Direct calculation of the OBC spectrum with major x '''
    Nx = 100
    Ny = 100
    model = minimal_nontrivial_model()

    # 1. get point list
    point_list = []
    for nx in range(Nx):
        for ny in range(Ny):
            point_list.append((nx, ny))
    
    dtype = complex
    method = 'arpack'
    # 2. get the Hamiltionian and spectrum
    H = model.generate_OBC_bulk(point_list)
    H = sparse.csr_matrix(H, dtype=dtype)


    E_real_list = np.linspace(-2.5, 4, 21)
    E_imag_list = np.linspace(-2, 2, 21)
    E_real_mat, E_imag_mat = np.meshgrid(E_real_list, E_imag_list)
    E_mat = E_real_mat + 1j * E_imag_mat
    s_mat = np.zeros_like(E_mat)
    for row_id in range(E_mat.shape[0]):
        for col_id in range(E_mat.shape[1]):
            curr_E = E_mat[row_id, col_id]
            curr_mat = curr_E * sparse.eye(H.shape[0], dtype=dtype) - H
            s = spla.svds(curr_mat, 4, which='SM', return_singular_vectors=False, solver=method)
            s_mat[row_id, col_id] = np.min(s)
            print("E:", curr_E, "Singular value:", s_mat[row_id, col_id])

    pickle_dump("data/minimal-nontrivial-OBC-Nx_%d-Ny_%d-svd-sweep-%s.pkl"%(Nx, Ny, method), 
                (E_mat, s_mat, point_list))

def debug_eigenvalue_calculation_feast():
    ''' Direct calculation of the OBC spectrum with major x '''
    Nx = 200
    Ny = 200
    model = minimal_nontrivial_model()

    # 1. get point list
    point_list = []
    print("Assembling...")
    for nx in range(Nx):
        for ny in range(Ny):
            point_list.append((nx, ny))
    
    dtype = complex
    method = 'arpack'
    # 2. get the Hamiltionian and spectrum
    H = model.generate_OBC_bulk(point_list)
    H = sparse.csr_matrix(H, dtype=dtype)
    print("Assembling finished")

    obj=feast.GSolver(H,M0=1000,Emid=3.0,r=5.0) 
    obj.fpm[3] = 200
    obj.fpm[17] = 30
    obj.fpm[18] = 90
    eigv,eigvec,M,info=obj.eigs()
    print(info)
    print(M)
    print(len(eigv))
    plt.plot(eigv.real, eigv.imag, '.')
    plt.show()
    pickle_dump("data/temp-data.pkl", (eigv, eigvec, M, info))

def test_feast():
    dtype = complex
    
    A = np.random.randn(100,100) + 1j * np.random.randn(100,100)
    A = sparse.csr_matrix(A)
    solver = feast.GSolver(A, M0 = 100, Emid=0.0, r=10)
    eigv, eigvec, M, info = solver.eigs()
    print(info)
    print(eigv)
    print(M)

####### batch run #######

def calculate_x_major_2D_GBZ_pythontype(E:complex):
    print(E)
    curr_GBZ_point_list = []
    # E = E_real_mat[j] + 1j * E_imag_mat[j]
    curr_GBZ_points = get_x_major_GBZ_2D(E)
    for curr_pair in curr_GBZ_points:
        for point in curr_pair:
            if(point.chart_labels[0] == 0):
                beta1 = point.coords[0]
            else:
                beta1 = 1 / point.coords[0]
            if(point.chart_labels[1] == 0):
                beta2 = point.coords[1]
            else:
                beta2 = 1 / point.coords[1]
            curr_GBZ_point_list.append((E, beta1, beta2))
    return curr_GBZ_point_list

def x_major_GBZ_sweep_fun(input_args):
    E, fname = input_args
    print(fname)
    solver = XMajorMainSolver(f)
    err_code = solver.robust_get_beta_GBZ(E)
    if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
        if(solver.no_zero_hole):
            err_code = solver.robust_get_critical_betanorm(E)
            if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
                with open(fname, "wb") as fp:
                    pickle.dump(solver.all_GBZ_points, fp)
            else:
                return [(E, err_code)]
    else:
        return [(E, err_code)]

    return []

def print_nothing(*args):
    pass

def sweep_x_major_GBZ(N_processes, verbose=True):
    if(not verbose):
        pGs.raise_warning = print_nothing
    if(not os.path.exists("data/2D-x-major")):
        os.mkdir("data/2D-x-major")
    E_real = np.linspace(-5, 5, 21)
    E_imag = np.linspace(-0.5, 0.5, 3)
    E_list = [(E_real[row_id] + 1j * E_imag[col_id], "data/2D-x-major/E_data_%d_%d.pkl"%(row_id, col_id)) 
              for row_id in range(len(E_real)) for col_id in range(len(E_imag))]

    mul_pool = Pool(N_processes)
    failed_E_list = mul_pool.map(x_major_GBZ_sweep_fun, E_list)
    failed_E = []
    for curr_critical in failed_E_list:
        failed_E += curr_critical
    all_data = {
        "search-range": (E_real, E_imag),
        "failed-E": failed_E
    }
    with open("data/2D-x-major/info.pkl", "wb") as fp:
        pickle.dump(all_data, fp)

def sweep_spectrum(GBZ_type: str, n_processes: int):
    E_real = np.linspace(-2.5, 4.5, 201)
    E_imag = np.linspace(-2.5, 2.5, 201)
    E_real_mat, E_imag_mat = np.meshgrid(E_real, E_imag)
    E_real_mat = E_real_mat.flatten()
    E_imag_mat = E_imag_mat.flatten()
    E_list = [E_real_mat[j] + 1j * E_imag_mat[j] for j in range(len(E_imag_mat))]
    # E_list = [4.5 + 2.5j]

    # for j in range(len(E_real_mat)):
    mul_pool = Pool(n_processes)
    if(GBZ_type == 'x'):
        fun_call = get_x_major_spectrum
    else:
        fun_call = get_y_major_spectrum
    GBZ_flags = mul_pool.map(fun_call, E_list)
    GBZ_spectrum = np.asarray(E_list)[GBZ_flags]
    all_data = {
        "search-range": (E_real, E_imag),
        "E": GBZ_spectrum
    }

    file_id = 0
    while(os.path.exists("data/2D-GBZ-spectrum-%s-major-%d.pkl"%(GBZ_type, file_id))):
        file_id += 1
    with open("data/2D-GBZ-spectrum-%s-major-%d.pkl"%(GBZ_type, file_id), "wb") as fp:
        pickle.dump(all_data, fp)

####### plots #######

def plot_2D_GBZ_spectrum(GBZ_type:str, N_points: int, file_id_range: tuple[int, int] = (0,100)):
    E_list = []
    file_id = file_id_range[0]
    while(os.path.exists("data/2D-GBZ-spectrum-%s-major-%d.pkl"%(GBZ_type, file_id))):
        with open("data/2D-GBZ-spectrum-%s-major-%d.pkl"%(GBZ_type, file_id), "rb") as fp:
            data = pickle.load(fp)
        E_list.append(data['E'])
        E_real, E_imag = data['search-range']
        print(min(E_real), max(E_real), min(E_imag), max(E_imag))
        file_id += 1
        if(file_id >= file_id_range[1]):
            break
    E_list = np.hstack(E_list)
    plt.plot(E_list.real, E_list.imag, '.')
    # with open("data/minimal-nontrivial-OBC-Nx_200-Ny_10.pkl", "rb") as fp:
    #     point_list, eigv, eigvec = pickle.load(fp)
    # plt.plot(eigv.real, eigv.imag, 'x')
    E, beta = pickle_load("data/%s-major-supercell-GBZ-N_%d.pkl"%(GBZ_type, N_points))
    plt.plot(E.real, E.imag, 'x')
    plt.legend(['2D GBZ', 'supercell 1D GBZ (N=15)'])

    # point_list, eigv, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_100-Ny_50.pkl")
    # eigv, eigvec, picked_eigv, curr_r = pickle_load("data/minimal-OBC-full-solve-Nx200-Ny50-%d.pkl"%(0))
    # plt.plot(eigv.real, eigv.imag, '.')
    # plt.legend(['2D GBZ', 'OBC spectrum (100 x 50)'])
    plt.show()


def plot_2D_GBZ():
    all_GBZ_points = []
    with open("data/2D-x-major/info.pkl", "rb") as fp:
        data = pickle.load(fp)

    E_real, E_imag = data['search-range']
    for row_id in range(len(E_real)):
        for col_id in range(len(E_imag)):
            if(os.path.exists("data/2D-x-major/E_data_%d_%d.pkl"%(row_id, col_id))):
                with open("data/2D-x-major/E_data_%d_%d.pkl"%(row_id, col_id), "rb") as fp:
                    all_GBZ_points += pickle.load(fp)
    

    # eigv, eigvec, picked_eigv, curr_r = pickle_load("data/minimal-OBC-full-solve-Nx200-Ny50-%d.pkl"%(2))
    # eigv, eigvec, picked_eigv, curr_r = pickle_load("data/minimal-OBC-full-solve-Nx100-Ny100-%d.pkl"%(0))

    # plt.plot(eigv.real, eigv.imag, '.')

    all_E = np.asarray([point[0] for point in all_GBZ_points])
    all_beta1 = np.asarray([point[1] for point in all_GBZ_points])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(all_beta1.real, all_beta1.imag, all_E.real, '.')
    # plt.plot(all_E.real, all_E.imag, 'x')

    E, beta = pickle_load("data/x-major-supercell-GBZ-N_15.pkl")
    ax.plot(beta.real, beta.imag, E.real, 'x')
    plt.xlabel("Re(E)")
    plt.ylabel("Im(E)")
    plt.show()

def plot_punctures_for_E():
    E = 2
    solver = XMajorMainSolver(f)
    err_code = solver.robust_get_beta_GBZ(E)
    all_loops = solver.beta_GBZ_glued
    fig = plt.figure()
    ax = fig.gca()
    # ax.plot(0,0,'rx')
    for loop in all_loops:
        curr_beta = np.zeros((len(loop.point_vec), 2), dtype=complex)
        for point_id, point in enumerate(loop.point_vec):
            Gm.CP1_to_chart(point, [0,0])
            curr_beta[point_id,:] = point.coords
        ax.plot(curr_beta[:,0].real, curr_beta[:,0].imag, 'b-')
    theta = np.linspace(0, 2*pi, 500)
    for curr_hole in solver.GBZ_holes[:-1]:
        beta_norm = np.sqrt((1+curr_hole)/(1-curr_hole))
        ax.plot(beta_norm * np.cos(theta), beta_norm * np.sin(theta), 'r:')
    # ax.plot(1.1 * np.cos(theta), 1.1 * np.sin(theta), 'g:')
    ax.plot(5 * np.cos(theta), 5 * np.sin(theta), 'r:')
    ax.set_xlabel("Re$(\\beta_1)$")
    ax.set_ylabel("Im$(\\beta_1)$")
    ax.set_aspect(1)
    # ax.set_xlim([-2,5.5])
    # ax.set_ylim([-2,2])
    plt.show()
        
def plot_total_winding_for_E():
    E = 2 + 0.2j
    solver = XMajorMainSolver(f)
    err_code = solver.robust_get_beta_GBZ(E)
    all_loops = solver.beta_GBZ_glued

    # beta_list = np.linspace(1.1, 1.4, 4)
    beta_list = np.linspace(1e-3, 5, 100)
    w_list = np.zeros_like(beta_list)
    for curr_id in range(len(beta_list)):
        curr_beta = beta_list[curr_id]
        print(curr_beta)
        z_coord = (curr_beta ** 2 - 1) / (curr_beta ** 2 + 1)
        solver.get_critical_beta(E, z_coord)
        solver.get_total_winding(E, z_coord)
        w_list[curr_id] = solver.curr_total_winding / (2 * pi)
    
    plt.plot(beta_list, w_list, '.-')
    plt.plot(beta_list, 0 * beta_list, 'r--')
    plt.xlabel("$|\\beta_1|$")
    plt.ylabel("Total winding number")
    plt.show()

############ accurate OBC spectrum ###########
def get_initial_scaling():
    Nx = 200
    Ny = 50
    point_list, eigv, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_200-Ny_50.pkl")
    expo_fitter = pGs.ExponentsFitting(point_list, [Nx, Ny], 1)
    expo_fitter.pick_boundary_coordinates(3)

    selected_id = 100
    eigv_selected = eigv[selected_id]
    eigvec_selected = eigvec[:, selected_id]
    r = expo_fitter.fit_exponents(np.abs(eigvec_selected))
    print(eigv_selected, r)
    point_list = np.asarray(point_list)
    plt.scatter(point_list[:,0], point_list[:,1], c=np.abs(eigvec_selected))
    plt.show()

def coarse_scaling(picked_eigv, curr_r):
    ''' r: |beta| = exp(curr_r) '''
    Nx = 100
    Ny = 100
    model = minimal_nontrivial_model()

    scaled_model = model.get_exponentially_scaled_system(np.exp(curr_r))
    scaled_model_x_1d = scaled_model.get_supercell(
        [(j,0) for j in range(Nx)],
        [[Nx,0],
         [0,1]]
    )
    scaled_model_OBC = scaled_model_x_1d.get_supercell(
        [(0,j) for j in range(Ny)],
        [[1,0],
         [0,Ny]]
    )
    new_H = scaled_model_OBC.get_bulk_Hamiltonian_complex((None,None))
    M = new_H - picked_eigv * np.eye(new_H.shape[0], dtype=complex)
    M = sparse.coo_matrix(M)

    slepc_solver.mat_to_file(M, "data/temp-mat-info.dat")

    ncv = 50
    nev = 1
    max_it = 4000
    os.system("mpirun -n 11 ./slepc-solve-coo.exe  -eps_smallest_magnitude -eps_ncv %d -eps_nev %d -eps_max_it %d"%(ncv, nev, max_it))

    new_eigv, new_eigvec = slepc_solver.get_eigensystem_h5("data/temp-eigen-system.h5", M.shape[0])
    new_eigv = new_eigv[0]
    new_eigvec = new_eigvec[:,0]

    expo_fitter = pGs.ExponentsFitting(
        scaled_model_OBC.lattice2cart(np.asarray(scaled_model_OBC.SiteCoord).transpose()).transpose(),
        [Nx, Ny], 1
    )
    expo_fitter.pick_boundary_coordinates(3)
    new_r = np.asarray(expo_fitter.fit_exponents(np.abs(new_eigvec)))
    print(new_eigv + picked_eigv)
    print("new_r:", new_r[0], new_r[1])
    print("total_r:", (curr_r + new_r)[0], (curr_r + new_r)[1])

    plt.scatter(expo_fitter.site_coords[:,0], expo_fitter.site_coords[:,1], 
                c=np.abs(new_eigvec))
    plt.show()
    return (new_eigv + picked_eigv), (curr_r + new_r)
    
def batch_solve(picked_eigv, curr_r, fname_prefix):
    ''' r: |beta| = exp(curr_r) '''
    Nx = 200
    Ny = 50
    model = minimal_nontrivial_model()

    scaled_model = model.get_exponentially_scaled_system(np.exp(curr_r))
    scaled_model_x_1d = scaled_model.get_supercell(
        [(j,0) for j in range(Nx)],
        [[Nx,0],
         [0,1]]
    )
    scaled_model_OBC = scaled_model_x_1d.get_supercell(
        [(0,j) for j in range(Ny)],
        [[1,0],
         [0,Ny]]
    )
    new_H = scaled_model_OBC.get_bulk_Hamiltonian_complex((None,None))
    M = new_H - picked_eigv * np.eye(new_H.shape[0], dtype=complex)
    M = sparse.coo_matrix(M)

    slepc_solver.mat_to_file(M, "data/temp-mat-info.dat")

    mpd = 60
    nev = 240
    ncv = 280
    max_it = 4000
    os.system("mpirun -n 11 ./slepc-solve-coo.exe  -eps_smallest_magnitude -eps_mpd %d -eps_nev %d -eps_ncv %d -eps_max_it %d"%(
        mpd, nev, ncv, max_it))

    new_eigv, new_eigvec = slepc_solver.get_eigensystem_h5("data/temp-eigen-system.h5", M.shape[0])
    new_eigv += picked_eigv

    fname = fname_prefix + "-Nx%d-Ny%d"%(Nx, Ny)
    file_id = 0
    while(os.path.exists("%s-%d.pkl"%(fname, file_id))):
        file_id += 1
    pickle_dump(
        "%s-%d.pkl"%(fname, file_id), (new_eigv, new_eigvec, picked_eigv, curr_r)
    )

def batch_solve_full_solver(picked_eigv, curr_r, fname_prefix):
    ''' r: |beta| = exp(curr_r) '''
    Nx = 100
    Ny = 100
    model = minimal_nontrivial_model()

    scaled_model = model.get_exponentially_scaled_system(np.exp(curr_r))
    scaled_model_x_1d = scaled_model.get_supercell(
        [(j,0) for j in range(Nx)],
        [[Nx,0],
         [0,1]]
    )
    scaled_model_OBC = scaled_model_x_1d.get_supercell(
        [(0,j) for j in range(Ny)],
        [[1,0],
         [0,Ny]]
    )
    new_H = scaled_model_OBC.get_bulk_Hamiltonian_complex((None,None))
    M = new_H - picked_eigv * np.eye(new_H.shape[0], dtype=complex)
    M = sparse.coo_matrix(M)

    new_eigv, new_eigvec = la.eig(M.todense())
    new_eigv += picked_eigv

    fname = fname_prefix + "-Nx%d-Ny%d"%(Nx, Ny)
    file_id = 0
    while(os.path.exists("%s-%d.pkl"%(fname, file_id))):
        file_id += 1
    pickle_dump(
        "%s-%d.pkl"%(fname, file_id), (new_eigv, new_eigvec, picked_eigv, curr_r)
    )


def plot_solved(N):
    point_list, eigv, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_200-Ny_50.pkl")
    plt.plot(eigv.real, eigv.imag, '.')
    for j in range(N):
        eigv, eigvec, picked_eigv, curr_r = pickle_load("data/minimal-OBC-slepc-solve-Nx200-Ny50-%d.pkl"%(j))
        print(j,np.linalg.cond(eigvec))
        plt.plot(eigv.real, eigv.imag, 'r.')
        plt.plot(picked_eigv.real, picked_eigv.imag, 'gx')
        max_rad = np.max(np.abs(eigv - picked_eigv))
        theta = np.linspace(0, 2 * pi, 100)
        plt.plot(picked_eigv.real + max_rad * np.cos(theta), picked_eigv.imag + max_rad * np.sin(theta))
    plt.show()

def plot_full_solved(N):
    point_list, eigv, eigvec = pickle_load("data/minimal-nontrivial-OBC-Nx_200-Ny_50.pkl")
    plt.plot(eigv.real, eigv.imag, '.')
    for j in range(N):
        eigv, eigvec, picked_eigv, curr_r = pickle_load("data/minimal-OBC-full-solve-Nx200-Ny50-%d.pkl"%(j))
        plt.plot(eigv.real, eigv.imag, '.')
        plt.plot(picked_eigv.real, picked_eigv.imag, 'gx')
        # max_rad = np.max(np.abs(eigv - picked_eigv))
        # theta = np.linspace(0, 2 * pi, 100)
        # plt.plot(picked_eigv.real + max_rad * np.cos(theta), picked_eigv.imag + max_rad * np.sin(theta))
    plt.show()



def batch_adjust_scaling(initial_E, initial_r):
    new_E = initial_E
    new_r = initial_r
    for j in range(3):
        new_E, new_r = coarse_scaling(initial_E, new_r)


def check_scaling():
    r = np.random.randn(2)
    Nx = 50
    Ny = 50
    model = minimal_nontrivial_model()
    model_x = model.get_supercell(
        [(j,0) for j in range(Nx)],
        [[Nx, 0], 
         [0, 1]]
    )
    model_2d = model_x.get_supercell(
        [(0,j) for j in range(Ny)],
        [[1,0],
         [0,Ny]]
    )
    H0 = model_2d.get_bulk_Hamiltonian_complex((None,None))

    model_scaled = model.get_exponentially_scaled_system(np.exp(r))
    model_scaled_x = model_scaled.get_supercell(
        [(j,0) for j in range(Nx)],
        [[Nx,0],
         [0,1]]
    )
    model_scaled_2d = model_scaled_x.get_supercell(
        [(0,j) for j in range(Ny)],
        [[1,0],
         [0,Ny]]
    )
    H = model_scaled_2d.get_bulk_Hamiltonian_complex((None, None))

    # transformation
    coord_list = model_scaled_2d.lattice2cart(np.asarray(model_scaled_2d.SiteCoord).transpose())
    rj_list = np.zeros(H.shape[0])
    for j in range(len(rj_list)):
        rj_list[j] = np.exp(r[0] * (coord_list[0,j] - Nx/2) + r[1] * (coord_list[1,j] - Ny/2))
    plt.plot(rj_list,'.')
    plt.show()
    r_mat = sparse.dia_matrix((rj_list,0), shape=H.shape)
    H_trans = spla.inv(r_mat) @ H0 @ r_mat
    print(spla.norm(H_trans - H))
    print(spla.norm(H), spla.norm(H0))


if __name__ == '__main__':
    ##### tests ######
    # check_partial_GBZ_continuity()
    # get_x_major_OBC()
    # main_plot_x_major_punctures()
    # calculate_x_major_supercell_GBZ()
    # plot_x_major_supercell()
    # main_calculate_puncture_distance()
    # test_segment_selection_and_pairing()
    # main_sweep_beta_norm()
    # print(get_x_major_GBZ_2D(2.38))
    # test_GBZ_derivative()
    # test_segment_join()
    # test_segment_join2()
    # test_partial_GBZ_calcualtion()

    # Debug
    # debug_critical_point_calculation()
    # get_x_major_GBZ_2D(-2)
    # calculate_x_major_2D_GBZ_pythontype(2)
    # get_x_major_GBZ_2D(5 + 5j)
    # print(get_x_major_spectrum((3.6366666666666667+0j)))
    # test_new_api()
    # check_x_major_OBC_calculation()
    # debug_winding_error()
    # print(get_y_major_spectrum(-1.5))
    # debug_eigenvalue_calculation()
    # debug_eigenvalue_calculation_feast()
    # test_feast()
    # check_y_major_GBZ()
    # check_y_major_partial_GBZ()

    # sweep_x_major_GBZ(6)
    # sweep_spectrum('y', 11)
    # plot_2D_GBZ_spectrum('x')
    # plt.show()

    # plot
    plot_2D_GBZ()
    # plot_punctures_for_E()
    # plot_total_winding_for_E()
    # calculate_y_major_supercell_GBZ()
    # plot_2D_GBZ_spectrum('x', 15, (5,10))
    # plot_2D_GBZ_spectrum('y', 15, (1,19))
    # eigv, eigvec, picked_eigv, curr_r = pickle_load("data/minimal-OBC-full-solve-Nx200-Ny50-%d.pkl"%(0))
    # plt.plot(eigv.real, eigv.imag, '.')
    # plt.plot(picked_eigv.real, picked_eigv.imag, 'x')
    # plt.show()

    # print(np.linalg.cond(eigvec))
    # plot_supercell_spectrum()
    # plot_x_major_OBC_spectrum()
    # get_initial_scaling()

    # batch_adjust_scaling((0 - 1j), np.array([0.18,0.27]))

    # batch_solve((3.6310824891857294+0.003626243400592687j), np.array([0.39250422453278794, 0.5819689256545028]),
    #             "data/minimal-OBC-slepc-solve") # 0

    # batch_solve((2.574697292417477+0.11294583021587129j), np.array([0.3467571481247496, 0.5513424237187435]),
    #             "data/minimal-OBC-slepc-solve") # 1

    # batch_solve((2.7610995936378027+0.06429096786320682j), np.array([0.3499717055398372, 0.5849384648806722]),
    #             "data/minimal-OBC-slepc-solve") # 2


    # batch_solve((2.68), np.array([0.3491438646722855, 0.5497469956777155]),
    #             "data/minimal-OBC-slepc-solve") # 3

    # batch_solve((2.68), np.array([0.40686731507701834, 0.5923002094790505]),
    #             "data/minimal-OBC-slepc-solve") # 4

    # batch_solve_full_solver((0+0j), np.array([0,0]), "data/minimal-OBC-full-solve")

    # plot_solved(5)
    # plot_full_solved(1)
    # compare_condition_number()
    # check_scaling()
    