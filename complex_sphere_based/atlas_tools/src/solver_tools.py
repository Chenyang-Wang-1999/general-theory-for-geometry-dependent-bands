'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-07-20
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''
import numpy as np
from scipy import optimize

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