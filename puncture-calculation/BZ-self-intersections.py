'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-08-02
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' Check the properties of the self intersections of BZ spectrum '''

import BerryPy.TightBinding as tb
import partial_GBZ_solver as pGs
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import poly_tools as pt
import GBZ_manifold as Gm

def test_model(t1, t2, gamma1, gamma2):
    ''' Model proposed by https://arxiv.org/pdf/2209.12534 '''
    inter_cell = [
        [0, 0, t1 - gamma1, (1,)],
        [0, 0, t2 - gamma2, (2,)],
        [0, 0, t1 + gamma1, (-1,)], 
        [0, 0, t2 + gamma2, (-2,)]
    ]
    return tb.TightBindingModel(1, 1, [[1]], [], inter_cell)

def get_BZ_spectrum(model: tb.TightBindingModel):
    ''' Get spectrum on BZ '''
    k = np.linspace(-0.5, 0.5, 100)
    E = np.zeros_like(k, dtype=complex)
    for curr_id, curr_k in enumerate(k):
        E[curr_id] = model.get_bulk_Hamiltonian_dense((curr_k,))[0,0]
    
    return E

def get_GBZ(model: tb.TightBindingModel):
    coeffs, degs = model.get_characteristic_polynomial_data()
    poly = pt.CLaurent(2)
    poly.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))

    _, params, aGBZ_loops = pGs.solve_GBZ(
        'GBZ', poly
    )

    aGBZ_points = []
    index_of_solution = []

    for param_id, curr_points in enumerate(aGBZ_loops):
        for point in curr_points:
            Gm.GBZ_to_chart(point, [0])
            aGBZ_points.append(point.coords)
            new_coords = list(point.coords)
            new_coords[1] *= np.exp(1j * params[param_id,0])
            aGBZ_points.append(new_coords)

            # get index of the solution
            E = point.coords[0]
            beta_eq = poly.partial_eval(
                pt.CScalarVec([E]), 
                pt.CIndexVec([0]),
                pt.CIndexVec([1]))
            coeffs = pt.CScalarVec([])
            degs = pt.CIndexVec([])
            beta_eq.num.batch_get_data(coeffs, degs)
            max_deg = max(degs)
            beta_eq_arr = np.zeros(max_deg + 1, dtype=complex)
            for term_id, curr_coeff in enumerate(coeffs):
                beta_eq_arr[max_deg - degs[term_id]] = curr_coeff
            
            # solve
            all_sols = np.roots(beta_eq_arr)
            all_sols_norm = np.abs(all_sols)
            sol_seq = np.argsort(all_sols_norm)
            all_sols = all_sols[sol_seq]
            all_sols_norm = all_sols_norm[sol_seq]

            # find index of current point in 'all_sols'
            sol_diff = all_sols - point.coords[1]
            min_id = np.argmin(np.abs(sol_diff))

            while(min_id != 0 and abs(all_sols_norm[min_id - 1] - all_sols_norm[min_id]) < 1e-6):
                min_id -= 1

            index_of_solution.append(min_id - beta_eq.denom_orders[0] + 1)
            index_of_solution.append(min_id - beta_eq.denom_orders[0] + 1)
    
    return np.array(aGBZ_points), index_of_solution

def main_plot_E():
    t1 = 1
    gamma1 = 0.0
    t2 = 1
    gamma2 = 0.5
    model = test_model(t1, t2, gamma1, gamma2)
    E = get_BZ_spectrum(model)
    plt.plot(E.real, E.imag, '.-')
    plt.savefig("data/temp.pdf")

def main_plot_aGBZ_spectrum():
    t1 = 1
    gamma1 = 0.0
    t2 = 1
    gamma2 = 0.5
    model = test_model(t1, t2, gamma1, gamma2)
    E = get_BZ_spectrum(model)
    hd = plt.plot(E.real, E.imag, '-')

    aGBZ_points, index_of_solution = get_GBZ(model)
    marker_list = []
    for curr_sol_index in set(index_of_solution):
        curr_indices = (np.asarray(index_of_solution) == curr_sol_index)
        curr_E = aGBZ_points[curr_indices,0]
        hd += plt.plot(curr_E.real, curr_E.imag, '.')
        marker_list.append(curr_sol_index)    
    
    plt.legend(hd, ['GBZ'] + ['aGBZ(%d)'%(n) for n in marker_list])
    plt.xlim([-3,3])
    plt.savefig("data/temp.pdf")


    plt.savefig("data/temp.pdf")

def main_plot_aGBZ():
    t1 = 1
    t2 = 1
    gamma2 = 0.5
    gamma1 = 0
    # gamma1 = np.random.randn()
    # t2 = np.random.randn()
    # gamma2 = np.random.randn()
    # print(t2, gamma1, gamma2)
    model = test_model(t1, t2, gamma1, gamma2)

    aGBZ_points, index_of_solution = get_GBZ(model)
    marker_list = []
    marker_list2 = []

    plt.figure(1)
    hd = plt.plot(np.cos(np.linspace(0, 2*np.pi, 100)),
                  np.sin(np.linspace(0, 2*np.pi, 100)))
    plt.figure(2)
    E = get_BZ_spectrum(model)
    hd2 = plt.plot(E.real, E.imag, '-')
    plt.quiver(E[0].real, E[0].imag, E[1].real - E[0].real, E[1].imag - E[0].imag)

    for curr_sol_index in set(index_of_solution):
        curr_indices = (np.asarray(index_of_solution) == curr_sol_index)
        curr_beta = aGBZ_points[curr_indices,1]
        curr_E = aGBZ_points[curr_indices, 0]

        plt.figure(1)
        hd += plt.plot(curr_beta.real, curr_beta.imag, '.')
        marker_list.append(curr_sol_index)    

        plt.figure(2)
        BZ_points = np.abs(np.abs(curr_beta) - 1) < 1e-2
        if(any(BZ_points)):
            BZ_E = curr_E[BZ_points]
            hd2 += plt.plot(BZ_E.real, BZ_E.imag, '.')
            marker_list2.append(curr_sol_index)

    plt.figure(1)
    plt.legend(hd, ['BZ'] + ['aGBZ(%d)'%(n) for n in marker_list])
    plt.ylim([-2,2])
    # plt.savefig('data/temp1.pdf')
    plt.figure(2)
    plt.legend(hd, ['BZ'] + ['aGBZ(%d)'%(n) for n in marker_list2])
    # plt.savefig('data/temp2.pdf')

    plt.show()
 


if __name__ == '__main__':
    # main_plot_E()
    main_plot_aGBZ()