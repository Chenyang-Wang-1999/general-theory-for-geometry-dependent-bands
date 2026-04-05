'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-18
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

'''
Calcualtions related to root solving of polynomials
'''

import numpy as np
# import poly_tools as pt
import c_solver_2D as pt
from scipy import optimize


class Event:
    history_level: int   # number of historical solutions passed to the function
    def check_solutions(self, sols, **args):
        ''' Return True if the solver need to be stopped. '''
        assert(sols.shape[0] <= self.history_level + 1)
        return False


class NormReversionEvent(Event):
    ''' An event that checks if the norm of the solutions reverted. '''
    M: int
    N: int

    def __init__(self, M: int, N: int, zero_tol: float = 1e-6):
        self.M = M
        self.N = N
        self.history_level = 0
        self.zero_tol = zero_tol

    def check_solutions(self, sols):
        ''' Return True if the solver need to be stopped. '''
        max_lowest_norm = np.max(np.abs(sols[0, :self.M]))
        min_highest_norm = np.min(np.abs(sols[0, self.M:]))
        mean_norm = (max_lowest_norm + min_highest_norm) / 2
        if max_lowest_norm > min_highest_norm - self.zero_tol * mean_norm:
            self.diff_norm = min_highest_norm - max_lowest_norm
            return True
        return False


class ContinuumZeroEndEvent(Event):
    ''' An event that checks if the continuum zero ends. '''
    M: int
    N: int
    zero_tol: float

    def __init__(self, M: int, N: int, zero_tol: float = 1e-6):
        self.M = M
        self.N = N
        self.history_level = 0
        self.zero_tol = zero_tol

    def check_solutions(self, sols):
        ''' Return True if the solver need to be stopped. '''
        sorted_sols = np.sort(np.abs(sols[0]))
        mean_norm = (sorted_sols[self.M - 1] + sorted_sols[self.M]) / 2
        if sorted_sols[self.M - 1] < sorted_sols[self.M] - self.zero_tol * mean_norm:
            return True
        else:
            return False


class ComplexEqConverter:
    def __init__(self, fun, cdim) -> None:
        self.fun = fun
        self.cdim = cdim
    
    def __call__(self, x_re, *args):
        x = x_re[:self.cdim] + 1j * x_re[self.cdim:]
        eq_LHS, eq_jac = self.fun(x, *args)

        # convert to real equation
        eq_LHS_re = np.zeros(2*self.cdim)
        eq_LHS_re[:self.cdim] = eq_LHS.real
        eq_LHS_re[self.cdim:] = eq_LHS.imag

        eq_jac_re = np.zeros((2*self.cdim, 2*self.cdim))
        eq_jac_re[:self.cdim, :self.cdim] = eq_jac.real 
        eq_jac_re[:self.cdim, self.cdim:] = - eq_jac.imag
        eq_jac_re[self.cdim:, :self.cdim] = eq_jac.imag
        eq_jac_re[self.cdim:, self.cdim:] = eq_jac.real
        return eq_LHS_re, eq_jac_re


def complex_root(complex_fun, x0, **options):
    fun = ComplexEqConverter(complex_fun, len(x0))
    x0_re = np.zeros(2*fun.cdim)
    x0_re[:fun.cdim] = x0.real
    x0_re[fun.cdim:] = x0.imag
    res= optimize.root(fun, x0_re, **options)
    x_re = res.x
    # jac_re = res.jac
    res.x = x_re[:fun.cdim] + 1j * x_re[fun.cdim:]
    if('args' in options.keys()):
        val, jac = complex_fun(x0,*options['args'])
    else:
        val, jac = complex_fun(x0)
    res.jac = jac
    return res


def poly_to_np_coefficients(coeffs: list[complex], degs: list[int]) -> np.ndarray:
    '''
    Convert a polynomial to a numpy array of coefficients.

    Parameters:
    coeffs (list[complex]): The coefficients of the polynomial.
    degs (list[int]): The degrees of the coefficients.

    Returns:
    np.ndarray: The numpy array of coefficients.
    '''
    max_deg = max(degs)
    np_coeffs = np.zeros(max_deg + 1, dtype=complex)
    for coeff, deg in zip(coeffs, degs):
        np_coeffs[max_deg - deg] = coeff
    return np_coeffs


def match_new_to_old(
    old_sols: np.ndarray[1, complex],
    new_sols: list[complex],
) -> np.ndarray[1, complex]:
    '''
    Match the new solutions to the old solutions.

    Parameters:
    old_sols (np.ndarray[1, complex]): The old solutions.
    new_sols (list[complex]): The new solutions.

    Returns:
    np.ndarray[1, complex]: The matched solutions.
    '''

    assert(len(old_sols) == len(new_sols))
    matched_sols = np.zeros_like(old_sols)

    inf_ind = -1
    if np.inf in old_sols:
        inf_ind = np.where(old_sols == np.inf)[0][0]
        curr_root_ind = np.argmax(np.abs(new_sols))
        matched_sols[inf_ind] = new_sols[curr_root_ind]
        new_sols.pop(curr_root_ind)

    elif np.inf in new_sols:
        inf_ind = np.argmax(np.abs(old_sols))
        curr_root_ind = new_sols.index(np.inf)
        matched_sols[inf_ind] = new_sols[curr_root_ind]
        new_sols.pop(curr_root_ind)

    for old_root_ind in range(len(old_sols)):
        if old_root_ind == inf_ind:
            continue
        curr_root_ind = np.argmin(np.abs(np.array(new_sols) - old_sols[old_root_ind]))
        matched_sols[old_root_ind] = new_sols[curr_root_ind]
        new_sols.pop(curr_root_ind)
    
    return matched_sols


def calculate_point_roots(
    char_poly: pt.CLaurent,
    param_ind_ctype: pt.CIndexVec,
    param_val: np.ndarray,
    var_ind_ctype: pt.CIndexVec,
    M_max: int,
    N_max: int
):
    poly_1d = char_poly.partial_eval(
        pt.CScalarVec(param_val),
        param_ind_ctype,
        var_ind_ctype
    )
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    poly_1d.num.batch_get_data(coeffs, degs)
    deg_M = poly_1d.denom_orders[0]
    np_coeffs = poly_to_np_coefficients(coeffs, degs)
    curr_roots = list(np.roots(np_coeffs))
    new_root_len = len(curr_roots)
    if new_root_len < M_max + N_max:
        if deg_M < M_max:
            curr_roots.append(0)
        if new_root_len - deg_M < N_max:
            curr_roots.append(np.inf)
    return curr_roots


def brute_force_roots(
    char_poly: pt.CLaurent,  # characteristic polynomial
    param_ind: tuple[int],  # indices of the parameters to be solved
    param_list: np.ndarray, # list of parameters to be solved. size = (num_of_params, len(param_ind))
    var_ind: int,  # index of the variable to be solved
    M_max: int,
    N_max: int,
    initial_sol: np.ndarray = None,
    events: list[Event] = []
):
    '''
    Solve the roots of a polynomial with respect to a variable.

    Parameters:
    char_poly (pt.CLaurent): The characteristic polynomial.
    param_ind (tuple[int]): The indices of the parameters to be solved.
    param_list (np.ndarray): The list of parameters to be solved. size = (num_of_params, len(param_ind))
    var_ind (int): The index of the variable to be solved.
    M_max (int): The maximum number of roots to be solved.
    N_max (int): The maximum number of additional roots to be solved.
    initial_sol (np.ndarray, optional): The initial guess of the roots for param_list[0,:]. size = (M_max + N_max)

    Returns:
    np.ndarray: The array of roots, ordered by the norm. 
    '''
    num_of_params = param_list.shape[0]
    roots = np.zeros((num_of_params, M_max + N_max), dtype=complex)
    param_ind_ctype = pt.CIndexVec(param_ind)
    var_ind_ctype = pt.CIndexVec([var_ind])
    for i in range(num_of_params):
        curr_roots = calculate_point_roots(
            char_poly,
            param_ind_ctype,
            param_list[i,:],
            var_ind_ctype,
            M_max,
            N_max
        )

        if i == 0 and initial_sol is None:
            roots[i,:] = sorted(curr_roots, key=lambda x: abs(x))
        else:
            if i == 0:
                old_roots = initial_sol
            else:
                old_roots = roots[i-1,:]
            
            roots[i,:] = match_new_to_old(old_roots, curr_roots)

        for event_ind, event in enumerate(events):
            left_bound = max([0, i - event.history_level])
            if event.check_solutions(roots[left_bound:i+1,:]):
                return event_ind, roots[:i+1,:]

    return None, roots
