'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-04-13
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import BerryPy.TightBinding as tb
import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
import pickle
import matplotlib.pyplot as plt
import feast
import partial_GBZ_solver as pGs
import slepc_solver
import os

def ssh_model(t1, t2, gamma):
    dim = 1
    site_num = 2
    intra_cell = [
        [0, 1, t1 - gamma/2],
        [1, 0, t1 + gamma/2]
    ]
    inter_cell = [
        [1, 0, t2, (1,)],
        [0, 1, t2, (-1,)]
    ]
    coords = [[0,], [0.5,]]
    return tb.TightBindingModel(
        dim, site_num, [[1]], intra_cell, inter_cell, coords
    )

def solve_spectrum():
    N = 100
    t1 = 1
    t2 = 0.5
    gamma = 0.2
    model = ssh_model(t1, t2, gamma)

    model_super = model.get_supercell(
        [(j,) for j in range(N)],
        [[N]]
    )
    

    dtype = np.complex256
    H = model_super.get_bulk_Hamiltonian_complex((None,))
    H = sparse.coo_matrix(H, dtype=dtype)
    H_dense = H.todense()
    eigv, eigvec = la.eig(H_dense)
    with open("data/eigen-system-N%d-%s.pkl"%(N, str(dtype)), "wb") as fp:
        pickle.dump((eigv, eigvec), fp)

def plot_spectrum():
    N = 5000
    with open("data/eigen-system-N%d-18.pkl"%(N), "rb") as fp:
        eigv, eigvec = pickle.load(fp)
    
    t1 = 1
    t2 = 0.5
    gamma = 0.2
    model = ssh_model(t1, t2, gamma)
    model_super = model.get_supercell(
        [(j,) for j in range(N)],
        [[N]]
    )

    r = np.sqrt((t1 - gamma) / (t1 + gamma))

    H = model_super.get_bulk_Hamiltonian_complex((None,))
    H = sparse.csr_matrix(H, dtype=np.complex256)
    H_dag = H.transpose().conjugate()
    site_coords = np.asarray([model_super.lattice2cart(x) for x in model_super.SiteCoord])
    for _ in range(100):
        random_id = np.random.randint(N)
        curr_E = eigv[random_id]
        plt.plot(np.abs(eigvec[:,random_id]), '.-')
        plt.show()
        R = H - curr_E * sparse.eye(H.shape[0])
        R_dag = H_dag - curr_E.conjugate() * sparse.eye(H.shape[0])
        RdagR = R_dag @ R
        curr_eigvec = eigvec[:,random_id].copy()
        for j in range(N):
            curr_eigvec[2 * j] /= r ** j
            curr_eigvec[2 * j + 1] /= r ** (j + 1) 
        print(la.norm(RdagR @ eigvec[:,random_id]), la.norm(eigvec[:,random_id]), curr_E.imag)

        # solver = feast.HSolver(RdagR, M0 = 100, Em=-1, Ex=1)
        # s_list, _, M, info = solver.eigsh()
        # print(info, M, s_list)

        plt.plot(site_coords[:,0], np.abs(curr_eigvec))
        plt.show()

        # s, _ = spla.lobpcg(RdagR, eigvec[:,random_id].reshape((-1,1)), tol=1e-16)
        # print(s[0])

        input("press enter to continue")
    flag, E, beta = model.get_GBZ()
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(E.real, E.imag, 'x')
    plt.show()

def compare_different_dtype():
    N = 5000
    # with open("data/eigen-system-N%d.pkl"%(N), "rb") as fp:
    #     eigv, eigvec = pickle.load(fp)
    # plt.plot(eigv.real, eigv.imag, '.')
    with open("data/eigen-system-N%d-18-perfect.pkl"%(N), "rb") as fp:
        eigv, eigvec = pickle.load(fp)
    plt.plot(eigv.real, eigv.imag, '.')
    plt.show()

def recalculate_by_scaling():
    N = 5000
    with open("data/eigen-system-N%d-18.pkl"%(N), "rb") as fp:
        eigv, eigvec = pickle.load(fp)
    
    t1 = 1
    t2 = 0.5
    gamma = 0.2
    model = ssh_model(t1, t2, gamma)
    model_super = model.get_supercell(
        [(j,) for j in range(N)],
        [[N]]
    )
    expo_fitter = pGs.ExponentsFitting(
        [model_super.lattice2cart(p) for p in model_super.SiteCoord],
        [N], 2
    )
    expo_fitter.pick_boundary_coordinates(5)

    plt.plot(eigv.real, eigv.imag, '.')
    plt.show()

    random_id = np.random.randint(N)
    print(random_id, eigv[random_id])
    r = expo_fitter.fit_exponents(np.abs(eigvec[:,random_id]))
    print(np.exp(r), np.sqrt((t1 - gamma/2) / (t1 + gamma/2)))

    # recalculate
    scaled_model = model.get_exponentially_scaled_system(np.exp(r))
    scaled_model_super = scaled_model.get_supercell(
        [(j,) for j in range(N)],
        [[N]]
    )

    new_H = scaled_model_super.get_bulk_Hamiltonian_complex((None,))
    M = new_H - eigv[random_id] * np.eye(new_H.shape[0], dtype=complex)
    new_eigv, new_eigvec = spla.eigs(M, k = 4, which = 'SM')
    new_eigv += eigv[random_id]
    print(np.linalg.cond(new_eigvec.transpose().conjugate() @ new_eigvec))
    plt.plot(new_eigv.real, new_eigv.imag, '.')
    plt.show()

def coarse_scaling(picked_eigv, curr_r):
    ''' r: |beta| = exp(curr_r) '''
    t1 = 1
    t2 = 0.5
    gamma = 0.2
    N = 5000
    model = ssh_model(t1, t2, gamma)

    scaled_model = model.get_exponentially_scaled_system(np.exp(curr_r))
    scaled_model_super = scaled_model.get_supercell(
        [(j,) for j in range(N)],
        [[N]]
    )
    new_H = scaled_model_super.get_bulk_Hamiltonian_complex((None,))
    M = new_H - picked_eigv * np.eye(new_H.shape[0], dtype=complex)
    M = sparse.coo_matrix(M)

    slepc_solver.mat_to_file(M, "data/temp-mat-info.dat")

    ncv = 50
    nev = 1
    max_it = 2000
    os.system("mpirun -n 11 ./slepc-solve-coo.exe  -eps_smallest_magnitude -eps_ncv %d -eps_nev %d -eps_max_it %d"%(ncv, nev, max_it))
    input()

    new_eigv, new_eigvec = slepc_solver.get_eigensystem_h5("data/temp-eigen-system.h5", M.shape[0])
    new_eigv = new_eigv[0]
    new_eigvec = new_eigvec[:,0]

    expo_fitter = pGs.ExponentsFitting(
        [scaled_model_super.lattice2cart(p) for p in scaled_model_super.SiteCoord],
        [N], 2
    )
    expo_fitter.pick_boundary_coordinates(5)
    new_r = np.asarray(expo_fitter.fit_exponents(np.abs(new_eigvec)))
    print(new_eigv + picked_eigv)
    print("new_r:", new_r[0])
    print("total_r:", (curr_r + new_r)[0])
    plt.plot(np.abs(new_eigvec))
    plt.show()
    
def get_initial_r():
    N = 5000
    with open("data/eigen-system-N%d.pkl"%(N), "rb") as fp:
        eigv, eigvec = pickle.load(fp)
    
    # plt.plot(eigv.real, eigv.imag, '.')
    # plt.show()
    
    N_picked = 4536
    print(N_picked, eigv[N_picked])

    curr_eigv = eigv[N_picked]
    curr_eigvec = eigvec[:, N_picked]

    t1 = 1
    t2 = 0.5
    gamma = 0.2
    N = 5000
    model = ssh_model(t1, t2, gamma)
    model_super = model.get_supercell(
        [(j,) for j in range(N)],
        [[N]]
    )

    expo_fitter = pGs.ExponentsFitting(
        [model_super.lattice2cart(p) for p in model_super.SiteCoord],
        [N], 1
    )
    eigv_norm = abs(curr_eigvec.reshape((-1,2))) 
    eigv_norm = np.mean(eigv_norm, axis=1)
    expo_fitter.pick_boundary_coordinates(5)
    plt.plot(np.abs(eigv_norm))
    plt.show()
    new_r = np.asarray(expo_fitter.fit_exponents(eigv_norm))
    print(new_r)
    print(np.sqrt((t1 - gamma/2) / (t1 + gamma/2)), np.exp(new_r))
    


if __name__ == '__main__':
    # solve_spectrum()
    # plot_spectrum()
    # compare_different_dtype()
    recalculate_by_scaling()
    # main_coarse_scale()
    # get_initial_r()
    # coarse_scaling((-1.2090414235956848+0.01813483335279828j), np.array([-0.0785]))
    # coarse_scaling((-1.2091211591769317+0.005572082023259773j), np.array([-0.08730618882282924]))
    # coarse_scaling((-1.2093074691622239+0.0029310542793249076j), np.array([-0.09371319502673212]))
    # coarse_scaling((-1.209204920049886+0.0007467730242007868j), np.array([-0.09909929302670481]))