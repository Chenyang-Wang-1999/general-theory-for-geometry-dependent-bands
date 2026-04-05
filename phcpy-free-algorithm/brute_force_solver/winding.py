'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-18
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' Functions for winding number calculations '''

# import poly_tools as pt
import c_solver_2D as pt
import numpy as np

from scipy import integrate
from math import pi
from scipy import sparse
from scipy.sparse import linalg as spla
import BerryPy.TightBinding as tb

class WindingFun:
    char_poly: pt.CLaurent
    loop_fun: callable  # Input: t. Output: param, dparam/dt
    dchar_poly: list[pt.CLaurent]
    loop_range: tuple[float, float]

    def __init__(self, char_poly: pt.CLaurent, loop_fun: callable, loop_range: tuple[float, float]):
        self.char_poly = char_poly
        self.loop_fun = loop_fun
        self.dchar_poly = [self.char_poly.derivative(var_ind) for var_ind in range(char_poly.dim)]
        self.loop_range = loop_range

    def __call__(self, t: float) -> complex:
        param, dparam_dt = self.loop_fun(t)
        param_ctype = pt.CScalarVec(param)
        val = self.char_poly.eval(param_ctype)
        dval_dt = sum([dchar.eval(param_ctype) * dparam_dt[param_ind] for param_ind, dchar in enumerate(self.dchar_poly)])
        return (dval_dt / val).imag
    
    def eval_dmu2(self, var: tuple[complex]):
        '''Reture (dmu2/dmu1, dmu2/dtheta1)'''
        var_ctype = pt.CScalarVec(var)
        complex_diff = var[1] * self.dchar_poly[1].eval(var_ctype) / (var[2] * self.dchar_poly[2].eval(var_ctype))
        return (-complex_diff.real, complex_diff.imag)


class MatWindingFun(WindingFun):
    mat_fun: callable  # foo(beta)
    dmat_fun: callable  # foo(beta, diff_orders)
    param_fun: callable # t -> beta(t), dbeta(t)
    E_ref: complex
    is_sparse: bool

    def __init__(self, mat_fun: callable, dmat_fun: callable, param_fun, loop_range, E_ref: complex, is_sparse: bool = True):
        self.mat_fun = mat_fun
        self.dmat_fun = dmat_fun
        self.param_fun = param_fun
        self.is_sparse = is_sparse
        self.loop_range = loop_range
        self.E_ref = E_ref

    def call_sparse(self, t: float) -> complex:
        beta, dbeta_dt = self.param_fun(t)
        mat = self.mat_fun(beta)
        dmat_dt = sparse.csc_matrix(mat.shape, dtype=complex)
        for dim_ind in range(len(beta)):
            diff_orders = np.zeros(len(beta), dtype=int)
            diff_orders[dim_ind] = 1
            dmat_dt += self.dmat_fun(beta, diff_orders) * dbeta_dt[dim_ind]

        try:
            return spla.spsolve(self.E_ref * sparse.eye(mat.shape[0]) - mat, -dmat_dt).trace().imag
        except RuntimeError:
            return -np.inf
    
    def __call__(self, t: float) -> complex:
        if self.is_sparse:
            return self.call_sparse(t)
        else:
            raise NotImplementedError("Dense matrix method is not implemented yet.")


def get_winding_number(winding_fun: WindingFun, N_seg=1) -> float:
    '''
    Calculate winding number.
    '''
    int_tot = 0.0
    intervals = np.linspace(winding_fun.loop_range[0], winding_fun.loop_range[1], N_seg + 1)
    for i in range(N_seg):
        int_tot += integrate.quad(winding_fun, intervals[i], intervals[i+1], epsabs=1e-2, epsrel=1e-2, limit=500)[0]

    return int_tot / (2 * pi)
