'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-01-14
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from BerryPy import TightBinding as tb
import matplotlib.pyplot as plt

####### model definition #######

def 米_model(coupling_coefficients:dict):
    '''
        a HN model with 米-shaped couplings
        coupling_coefficients: coefficients for (1,0), (0,1), ...
    '''

    site_num = 1
    dim = 2
    intra_cell = []
    inter_cell = [
        [0, 0, coupling_coefficients[direction], direction] for direction in [(1,0), (-1,0), (0,1), (0, -1), (1,1), (1,-1), (-1,1), (-1,-1)]
    ]

    return tb.TightBindingModel(dim, site_num, [[1,0],[0,1]], intra_cell, inter_cell)


###### main functions #######
def test_米_shaped_GBZ():
    nearest = 1
    next_nearest = 0.5
    gamma1 = 0.0
    gamma2 = 0.1
    coupling_coefficients = {
        (1,0):nearest, 
        (-1,0):nearest,
        (0,1):nearest,
        (0,-1):nearest,
        (1,1) : next_nearest + gamma1/2,
        (-1,-1) : next_nearest - gamma1/2,
        (1, -1) : next_nearest + gamma2/2,
        (-1,1) : next_nearest - gamma2/2
    }

    model = 米_model(coupling_coefficients)
    N = 15
    model_1d = model.get_supercell([(j,0) for j in range(N)], [[N,0],[0,1]])
    model_1d = model_1d.truncate(0)
    flag,E,beta = model_1d.get_GBZ(N_process=6)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(beta.real, beta.imag, E.real, '.')

    fig2 = plt.figure(2)
    ax = fig2.gca()
    ax.plot(E.real, E.imag, '.')
    plt.show()

def 米_shaped_along_11():
    nearest = 1
    next_nearest = 0.5
    gamma1 = 0.0
    gamma2 = 0.1
    coupling_coefficients = {
        (1,0):nearest + gamma1, 
        (-1,0):nearest - gamma1,
        (0,1):nearest + gamma2,
        (0,-1):nearest - gamma2,
        (1,1) : next_nearest,
        (-1,-1) : next_nearest,
        (1, -1) : next_nearest,
        (-1,1) : next_nearest
    }

    model = 米_model(coupling_coefficients)
    N = 15
    model2 = model.get_supercell([(0,0)],
                                 [[1,0], 
                                  [1,1]])
    model_1d = model2.get_supercell([
        (0,j) for j in range(N)
        ], 
        [[1,0],
         [0,N]])
    model_1d = model_1d.truncate(1)

    flag,E,beta = model_1d.get_GBZ(N_process=6)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(beta.real, beta.imag, E.real, '.')

    fig2 = plt.figure(2)
    ax = fig2.gca()
    ax.plot(E.real, E.imag, '.')
    plt.show()


if __name__ == '__main__':
    # test_米_shaped_GBZ()
    米_shaped_along_11()