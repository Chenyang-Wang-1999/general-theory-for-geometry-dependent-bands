'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-02-10
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import scipy.linalg as la
from math import sin, cos, sqrt, pi
import BerryPy.TightBinding as tb
from fractions import Fraction as frac


#### log det ####
class BIT:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index, value=1):
        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def count_inversions(arr):
    # 离散化处理（如果数组元素不是连续的整数）
    sorted_arr = sorted(arr)
    rank = {num: i+1 for i, num in enumerate(sorted_arr)}

    bit = BIT(len(arr))
    inversion_count = 0

    for num in reversed(arr):
        r = rank[num]
        inversion_count += bit.query(r - 1)
        bit.update(r)

    return inversion_count


def sparse_logdet(A: sparse.csc_matrix):
    ''' Calculate log(det(A)) '''
    A_lu = spla.splu(A)
    total_perm = count_inversions(A_lu.perm_r) + count_inversions(A_lu.perm_c)
    L_diag = A_lu.L.diagonal()
    U_diag = A_lu.U.diagonal()
    L_diag = np.asarray(L_diag, dtype=complex)
    U_diag = np.asarray(U_diag, dtype=complex)
    logdet_val = np.sum(np.log(L_diag)) + np.sum(np.log(U_diag))
    if total_perm % 2:  # odd inversion number
        logdet_val += 1j * pi

    # normalization
    logdet_imag = logdet_val.imag
    logdet_imag = ((logdet_imag + pi) % (2 * pi)) - pi
    return logdet_val.real + 1j * logdet_imag

#### END log det ####


def calculate_quasi_1D_winding(
        model:tb.TightBindingModel,
        E_ref: complex,
        n_points: int,
        radius: float,
        max_incre: float = pi / 6,
        debug_mode=False):
    ''' Calculate quasi-1D winding number with self-adapted algorithm '''
    total_winding = 0.0

    def eval_logdetimag_for_theta(theta):
        beta = radius * np.exp(1j * theta)
        curr_H = model.get_bulk_Hamiltonian_complex((beta,))
        E_minus_H = E_ref * sparse.eye(curr_H.shape[0], dtype=complex) - curr_H
        return sparse_logdet(E_minus_H).imag

    theta_stack = np.linspace(0, 2 * pi, n_points + 1)
    theta_stack = list(theta_stack[::-1])
    if debug_mode:
        all_points = set(theta_stack)
    curr_theta = theta_stack.pop()
    curr_det = eval_logdetimag_for_theta(curr_theta)
    while theta_stack:
        next_data = theta_stack.pop()
        if isinstance(next_data, tuple):
            next_theta, next_det = next_data
        else:
            next_theta = next_data
            next_det = eval_logdetimag_for_theta(next_theta)

        # calculate
        det_incre = ((next_det - curr_det + pi) % (2 * pi)) - pi
        if abs(det_incre) < max_incre:
            total_winding += det_incre
            curr_theta = next_theta
            curr_det = next_det
        else:
            theta_stack.append((next_theta, next_det))
            theta_stack.append((next_theta + curr_theta) / 2)
            if debug_mode:
                all_points.append((next_theta + curr_theta) / 2)

    total_winding /= 2 * pi

    if debug_mode:
        return total_winding, all_points
    else:
        return total_winding


if __name__ == "__main__":
    A = np.random.randn(100, 100) + 1j * np.random.randn(100, 100)
    A_sp = sparse.csc_matrix(A)
    print(np.log(la.det(A)))
    print(sparse_logdet(A_sp))
