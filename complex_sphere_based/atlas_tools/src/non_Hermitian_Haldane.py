'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-09-12
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import BerryPy.TightBinding as tb
import BerryPy as bp
import numpy as np
from math import sin, cos, sqrt, pi
from cmath import exp, log
from scipy import io as sio
import pickle

def non_Hermitian_Haldane_H(u1, u2, v1, v2, phi, M):
    dim = 2
    site_num = 2

    lattice_vec = np.array(
        [[-cos(pi/3), -cos(pi/3)],
         [-sin(pi/3), sin(pi/3)]]
    )

    intra_cell = [
        [0, 0, M],
        [1, 1, -M],
        [1, 0, u1],
        [0, 1, u2]
    ]
    inter_cell = [
        [0, 0, v2 * exp(1j*phi), (-1,0)],
        [0, 0, v2 * exp(1j*phi), (0, -1)],
        [0, 0, v2 * exp(1j*phi), (1,1)],
        [0, 0, v1 * exp(-1j*phi), (1,0)],
        [0, 0, v1 * exp(-1j*phi), (0,1)],
        [0, 0, v1 * exp(-1j*phi), (-1,-1)],
        [1, 0, u1, (0, -1)],
        [1, 0, u1, (1, 0)],
        [0, 1, u2, (0, 1)],
        [0, 1, u2, (-1, 0)],
        [1, 1, v2 * exp(-1j * phi), (-1, 0)],
        [1, 1, v2 * exp(-1j*phi), (0, -1)],
        [1, 1, v2 * exp(-1j*phi), (1, 1)],
        [1, 1, v1 * exp(1j * phi), (1, 0)],
        [1, 1, v1 * exp(1j * phi), (0, 1)],
        [1, 1, v1 * exp(1j * phi), (-1,-1)]
    ]
    return tb.TightBindingModel(dim, site_num, lattice_vec, intra_cell, inter_cell)

def Hermitian_Haldane(t1, t2, phi, M):
    return non_Hermitian_Haldane_H(t1, t1, t2, t2, phi, M)

def Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma):
    # v1 = v2 = t2 * exp(i gamma)
    return non_Hermitian_Haldane_H(t1, t1, t2 * exp(1j * gamma), t2 * exp(1j * gamma), phi, M)

def Hermitian_Chern(model:tb.TightBindingModel):
    solver = bp.MatrixSolver(model.get_bulk_Hamiltonian_dense, True, 2, True)
    chern_solver = bp.BerrySolver(solver, '')
    N = 20
    mesh_grid = np.zeros((2,N,N))
    k1 = np.linspace(-0.5, 0.5, N)
    k2 = np.linspace(-0.5, 0.5, N)
    K2, K1 = np.meshgrid(k2, k1)
    mesh_grid[0,:,:] = K1
    mesh_grid[1,:,:] = K2
    flux, area = chern_solver.Curvature(mesh_grid, 0)
    return np.sum(flux)/(2*np.pi)

def Hermitian_check():
    t1 = 1
    t2 = 1/3
    M = 3*sqrt(3)* t2
    phi = pi/3
    C = Hermitian_Chern(Hermitian_Haldane(t1, t2, phi, M))
    print(C)

def generate_model_and_char_poly():
    solution_index = 7

    t1 = 1
    t2 = 1/3
    M = 0
    phi = pi/3
    gamma = 0.2*pi

    model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)
    
    # model_square = model.get_supercell_and_eliminate([(0,1), (0,0), (1,0)],
    #                                                  [
    #                                                      [-1, -1],
    #                                                      [1, -1]
    #                                                  ], 
    #                                                  [((0,1), 0), ((1,0),1)])
    # model_square.save_to_file("test.txt")
    # char_poly = model_square.get_characteristic_polynomial_ctype()
    model_clinic = model.get_supercell([(0,0)], [
        [1, -1],
        [0, 1]
    ])
    char_poly = model_clinic.get_characteristic_polynomial_ctype()
    num_coeff, num_orders = char_poly.f_num.batch_get_data()
    denom_coeff, denom_orders = char_poly.f_denom.batch_get_data()
    char_poly_orders = num_orders.copy()
    for j in range(char_poly_orders.shape[0]):
        for k in range(3):
            char_poly_orders[j, k] -= denom_orders[0, k]
    char_poly_data = {"coeffs":num_coeff, "orders":char_poly_orders}

    model_info = {
        "solution_index":solution_index,
        't1': t1,
        't2': t2,
        'M' : M,
        'phi' : phi,
        'gamma' : gamma, 
        "lattice_vec":model_clinic.LatticeVec,
        'info' : "Non-Hermitian Haldane model with non-Hermitian phase. non_Hermitian_Haldane.py"
    }
    sio.savemat("data/GBZ_info_%d.mat"%(solution_index), model_info)
    sio.savemat("data/GBZ_char_poly_%d.mat"%(solution_index), char_poly_data)
    with open("data/GBZ_model_%d.pkl"%(solution_index), "wb") as fp:
        pickle.dump(model, fp)

if __name__ == '__main__':
    # Hermitian_check()
    generate_model_and_char_poly()