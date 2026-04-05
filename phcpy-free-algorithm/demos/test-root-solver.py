'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-19 10:38:12
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


class FunExcludedRoots:
    ''' A wrapper for univariate functions with excluded roots '''
    excluded_roots: np.ndarray
    original_fun: callable

    def __init__(self, original_fun, excluded_roots = np.array([])):
        self.original_fun = original_fun
        self.excluded_roots = excluded_roots

    def __call__(self, z) -> tuple[complex, complex]:
        val, der = self.original_fun(z)
        diff_inv = 1 / (z - self.excluded_roots)
        diff_prod = np.prod(diff_inv)
        new_val = val * diff_prod
        new_der = der * diff_prod - np.sum(diff_inv) * new_val
        return new_val, new_der
    
    def add_root(self, z0):
        self.excluded_roots = np.hstack([self.excluded_roots, [z0]])


def test_derivative():
    def test_fun(z):
        return z ** 2, 2 * z
    
    new_fun = FunExcludedRoots(test_fun, np.array([0]))
    z_list = np.linspace(-1, 1, 1000)
    val_list = np.zeros(1000)
    diff_list = np.zeros(1000)
    for ind, z in enumerate(z_list):
        val_list[ind], diff_list[ind] = new_fun(z)

    plt.plot(z_list, val_list)
    plt.plot(z_list, diff_list)
    plt.savefig("data/temp.png")


def test_root():
    def test_fun(z):
            return z ** 2 + 0.1 * z, 2 * z + 0.1
    new_fun = FunExcludedRoots(test_fun)

    # solve
    res = optimize.root_scalar(new_fun, x0=1j, fprime=True)
    print(res)
    new_fun.add_root(res["root"])
    res = optimize.root_scalar(new_fun, x0=1, fprime=True)
    print(res)

if __name__ == "__main__":
    # test_derivative()
    test_root()
