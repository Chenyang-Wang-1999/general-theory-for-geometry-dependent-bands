'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-03-20
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import partial_GBZ_solver as pGs
import poly_tools as pt
import BerryPy.TightBinding as tb
from scipy import linalg as la
import matplotlib.pyplot as plt

def test_model(t1, t2, gamma):
    ''' Single band model with unidirectional coupling 
            f = E - t1 beta - t1 beta^{-1} - (t2 + gamma) beta **2 - (t2 - gamma) beta ** {-2} 
    '''
    char_poly_coeffs = pt.CScalarVec([
        1, -t1, -t1, -(t2 + gamma), -(t2 - gamma)
    ])
    char_poly_degs = pt.CLaurentIndexVec([
        # E, beta
        1, 0,
        0, 1, 
        0, -1,
        0, 2,
        0, -2
    ])
    char_poly = pt.CLaurent(2)
    char_poly.set_Laurent_by_terms(char_poly_coeffs, char_poly_degs)

    dim = 1
    site_num = 1
    inter_cell = [
        [0, 0, t1, (1,)],
        [0, 0, t1, (-1,)],
        [0, 0, t2 + gamma, (-2,)],
        [0, 0, t2 - gamma, (2,)]
    ]
    model = tb.TightBindingModel(
        dim, site_num, [[1]], [], inter_cell
    )
    return model, char_poly

def test_GBZ():
    t1 = np.random.rand()
    t2 = np.random.rand()
    gamma = np.random.randn()
    model, char_poly = test_model(t1, t2, gamma)

    # 1. Calculate GBZ
    all_GBZ_points = pGs.solve_GBZ_temp(char_poly, rel_tol=1e-4)
    E_list = np.zeros(len(all_GBZ_points), dtype=complex)
    fig = plt.figure(1)
    ax1 = fig.gca()
    num_chart0 = 0
    for j in range(len(all_GBZ_points)):
        E_list[j] = all_GBZ_points[j].coords[0]
        if(abs(all_GBZ_points[j].coords[1]) > 1):
            print(all_GBZ_points[j].chart_labels, all_GBZ_points[j].coords)

        if(all_GBZ_points[j].chart_labels[0] == 0):
            num_chart0 += 1
            ax1.plot(all_GBZ_points[j].coords[1].real,
                     all_GBZ_points[j].coords[1].imag,
                     'b.')
        else:
            new_point = 1/all_GBZ_points[j].coords[1]
            ax1.plot(new_point.real,
                     new_point.imag,
                     'b.')
    
    print(num_chart0, len(all_GBZ_points))
    input()

    flag, E_old, beta_old = model.get_GBZ(N_phi=50)

    ax1.plot(beta_old.real, beta_old.imag)

    # 2. Calculate OBC spectrum
    N = 100
    model_1d = model.get_supercell(
        [(j,) for j in range(N)], 
        [[N]])
    
    H = model_1d.get_bulk_Hamiltonian_complex((None,)).todense()
    eigv, eigvec = la.eig(H)

    # 3. visuallization
    plt.figure(2)
    plt.plot(E_list.real, E_list.imag, '.')
    plt.plot(E_old.real, E_old.imag, '.')
    plt.plot(eigv.real, eigv.imag, '.')
    plt.legend(['E new', 'E old', 'OBC'])
    plt.show()

if __name__ == '__main__':
    test_GBZ()