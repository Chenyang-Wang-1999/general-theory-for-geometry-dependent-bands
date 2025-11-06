'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-10-01
Copyright © Department of Physics, Tsinghua University. All rights reserved

Calculate 1D GBZ by brute-force method
'''

import numpy as np
import phcpy
import poly_tools as pt
import winding_GBZ_solver as pGs
import sys
from multiprocessing import Pool

class CharPolyObj:
    ''' Wrapper for characteristic polynomial '''
    char_poly_num: pt.CPolyLinkedList   # Numerator of ChP
    M: int     # Maximal negative degree of beta
    df: list[pt.CPolyLinkedList]    # Derivative of ChP


def solve_aGBZ(
    char_poly: pt.CPolyLinkedList,
    phi: float,
    n_processes: int
):
    ''' Solve aGBZ points 
        i.e. f(E, beta) = 0, df/dbeta (E, beta) = 0 
    '''
    # 1. phcpy solve
    if abs(phi) < 1e-6:
        eq_system = [char_poly, char_poly.derivative(1)]
    else:
        eq_system = [char_poly, char_poly.scale_var(
            pt.CScalarVec([1.0, np.exp(1j * phi)])
        )]

    eq_str = [
        f.to_str(pt.CStrVec(['x', 'y']))
        for f in eq_system
    ]

    all_sols = phcpy.solver.solve(eq_str, dictionary_output=True, tasks=n_processes,
                                  precision='qd')

    # 2. scipy solve
    def aGBZ_eq_numerical(E_and_beta: np.ndarray[1, complex]):
        var_val = np.zeros(2, dtype=complex)
        for eq_ind in range(2):
            var_val[eq_ind] = eq_system[eq_ind].eval(pt.CScalarVec(E_and_beta))

        jac = np.zeros((2,2), dtype=complex)
        for row_ind in range(2):
            for col_ind in range(2):
                diff_order = pt.CIndexVec([0,0])
                diff_order[col_ind] += 1
                jac[row_ind, col_ind] = (
                    eq_system[row_ind].eval_diff(
                        diff_order,
                        pt.CScalarVec(E_and_beta)
                    )
                )
        
        return var_val, jac
    
    aGBZ_sols = []
    for curr_sol in all_sols:
        curr_x = np.array([curr_sol['x'], curr_sol['y']])
        res = pGs.complex_root(aGBZ_eq_numerical, curr_x, jac=True)
        aGBZ_sols.append(res.x)
    
    return aGBZ_sols


def solve_poly_eq_1D(poly_eq: pt.CLaurent):
    ''' Solve polynomial equation with numpy.root '''
    # 1. Get coefficients and degrees
    if isinstance(poly_eq, pt.CLaurent):
        # Laurent polynomial
        coeffs = pt.CScalarVec([])
        degs = pt.CIndexVec([])
        poly_eq.num.batch_get_data(coeffs, degs)
    else:
        # polynomial
        coeffs, degs = poly_eq.batch_get_data()
    
    if len(degs) == 0:
        return []

    # 2. Generate 1D array
    max_deg = max(degs)
    equation_1d_nparray = np.zeros(max_deg + 1, dtype=complex)

    for term_id in range(len(coeffs)):
        equation_1d_nparray[max_deg - degs[term_id]] = coeffs[term_id]
    
    all_roots = np.roots(equation_1d_nparray)

    return all_roots


def check_GBZ_condition(char_poly: pt.CPolyLinkedList, M: int, sol: np.ndarray[1, complex], 
                        tol_check=1e-3, tol_warning=1e-6):
    ''' Check GBZ condition '''
    # 1. Get 1D equation
    eq_1D = char_poly.partial_eval(
        pt.CScalarVec([sol[0]]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1])
    )
    all_roots = solve_poly_eq_1D(eq_1D)
    if len(all_roots) == 0:
        return True

    # 2. sort roots
    all_norms = np.sort(np.abs(all_roots))

    # 3. Check GBZ condition
    if abs(abs(sol[1]) - all_norms[M - 1]) < tol_check:
        if abs(abs(sol[1]) - all_norms[M]) < tol_check:
            if abs(all_norms[M - 1] - all_norms[M]) > tol_warning:
                print("Warning: large error detected, err = %e" % (abs(all_norms[M - 1] - all_norms[M])),
                      file=sys.stderr)
            return True
    return False


def batch_run(input_args):
    phi, coeffs, degs, M = input_args
    char_poly_num = pt.CPolyLinkedList(2)
    char_poly_num.init_with_data(
        pt.CScalarVec(coeffs), 
        pt.CIndexVec(degs)
    )

    curr_prog = int(100 * phi / np.pi)
    print("#" * curr_prog + "-" * (100 - curr_prog))
    # 2.1 Get aGBZ 
    all_aGBZ_points = solve_aGBZ(
        char_poly_num, phi, 0
    )
    # 2.2 Check GBZ condition
    all_sols = []
    for curr_point in all_aGBZ_points:
        if check_GBZ_condition(
            char_poly_num, M, curr_point
        ):
            all_sols.append(curr_point)
            new_point = curr_point.copy()
            new_point[1] *= np.exp(1j * phi)
            all_sols.append(new_point)
    
    return all_sols

def solve_GBZ(char_poly: pt.CLaurent, n_phi=50, n_processes=1):
    ''' Solve GBZ for 1D characteristic polynomial '''
    # 1. Get numerator and M
    char_poly_num = pt.CPolyLinkedList(2)
    char_poly.num.copy_to(char_poly_num)
    M = - char_poly.denom_orders[1]

    # 2. Prepare data package
    coeffs, degs = char_poly_num.batch_get_data()
    degs = degs.flatten()
    all_args = [
        (phi, coeffs, degs, M) for phi in np.linspace(0, np.pi, n_phi)
    ]

    # 2. Sweep all phi
    if n_processes == 1:
        all_sols = [batch_run(curr_args) for curr_args in all_args]
    else:
        pool = Pool(processes=n_processes)
        all_sols = pool.map(batch_run, all_args)

    # 3. Collect the solutions
    final_sols = []
    for curr_sol in all_sols:
        final_sols += curr_sol
    
    return np.asarray(final_sols)
