'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-06-01
Copyright © Department of Physics, Tsinghua University. All rights reserved

Ultilities related to HN model for the paper plot
'''

import BerryPy.TightBinding as tb
import numpy as np
from cmath import sqrt
import matplotlib.pyplot as plt
from scipy import linalg as la

Default_Model_Params = (
    1 + 1j,
    1.5 + 1.2j,
    -1 + 1j,
    -1.2 - 0.5j
)


def continuous_sqrt(x: np.ndarray) -> np.ndarray:
    '''
    Continuously calculate the square root of an array of complex numbers.

    The first point takes the principal square root.  For every later point the
    sign of the previous root is kept when that root is closer to the
    difference of the two candidate square roots, is flipped when it is closer
    to their sum, and is reset to the negated root when the two distances are
    equal (the tie rule of the original implementation).  The sign walk is done
    with cumulative sums, so there is no Python loop over the array.

    This is a stand-alone replacement of ``numpy_extended.continuous_sqrt``,
    the pybind11 wrapper around a C++ implementation that the scripts used to
    import; it returns the same values bit for bit.  Its argument is a 1-D array
    of complex numbers; real input is cast to complex.  The same function is
    also defined in quasi-1D-spectrum.py.
    '''
    root = np.sqrt(np.asarray(x, dtype=complex))
    if root.ndim != 1:
        raise ValueError("continuous_sqrt expects a 1-D array of complex numbers")
    if root.size < 2:
        return root

    # squared distances from the previous root to the two candidate roots
    dist_diff = root[:-1] - root[1:]
    dist_sum = root[:-1] + root[1:]
    n_diff = dist_diff.real * dist_diff.real + dist_diff.imag * dist_diff.imag
    n_sum = dist_sum.real * dist_sum.real + dist_sum.imag * dist_sum.imag
    keep = n_diff < n_sum
    flip = n_sum < n_diff
    tie = ~(keep | flip)

    # The sign is kept on `keep`, flipped on `flip` and reset to -1 on `tie`,
    # hence the sign at j is (-1) ** (number of flips after the last tie up to
    # j), negated as soon as a tie has happened.
    last_tie = np.maximum.accumulate(np.where(tie, np.arange(flip.size), -1))
    flips = np.concatenate(([0], np.cumsum(flip)))
    odd = (flips[1:] - flips[last_tie + 1]) % 2 == 1
    negative = np.where(last_tie < 0, odd, ~odd)

    out = root.copy()
    out[1:] = np.where(negative, -root[1:], root[1:])
    return out


def get_HN_model(Jx1, Jx2, Jy1, Jy2):
    inter_cell = [
        [0, 0, Jx1, (1, 0)],
        [0, 0, Jx2, (-1, 0)],
        [0, 0, Jy1, (0, 1)],
        [0, 0, Jy2, (0, -1)]
    ]
    return tb.TightBindingModel(2, 1, np.array([[1, 0], [0, 1]]), [], inter_cell, [[0, 0]])


def get_11_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2):
    bound = 2 * continuous_sqrt(
            2 * sqrt(Jx1 * Jx2 * Jy1 * Jy2) * np.linspace(-1, 1, 500)
            + (Jx1 * Jx2)
            + (Jy1 * Jy2)
    )
    loop = np.hstack(
        [bound, -bound[::-1], bound[0]]
    )
    return bound, loop


def get_x_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2):
    all_points = [
        2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
        2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
        -2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
        -2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
    ]
    return all_points, np.asarray(all_points + [all_points[0]])


def get_coupling_terms_from_new_param(
    Jx, Jy, gammax, gammay, deltax, deltay
):
    Jx1 = np.exp(gammax) * Jx * np.exp(1j * deltax)
    Jx2 = np.exp(-gammax) * (Jx.conjugate()) * np.exp(1j * deltax)
    Jy1 = np.exp(gammay) * Jy * np.exp(1j * deltay)
    Jy2 = np.exp(-gammay) * (Jy.conjugate()) * np.exp(1j * deltay)
    return Jx1, Jx2, Jy1, Jy2


def get_new_param_from_coupling_terms(
    Jx1, Jx2, Jy1, Jy2
):
    gammax = np.log(abs(Jx1 / Jx2)) / 2
    gammay = np.log(abs(Jy1 / Jy2)) / 2
    deltax = np.angle(Jx1 * Jx2) / 2
    deltay = np.angle(Jy1 * Jy2) / 2
    Jx = Jx1 / np.exp(gammax + 1j * deltax)
    Jy = Jy1 / np.exp(gammay + 1j * deltay)
    return Jx, Jy, gammax, gammay, deltax, deltay


##### tests #####
def test_11_DGBZ_spectrum():
    bound, loop = get_11_DGBZ_spectrum(*Default_Model_Params)
    plt.plot(loop.real, loop.imag, '.-')
    plt.show()


def test_x_DGBZ_spectrum():
    bound, loop = get_x_DGBZ_spectrum(*Default_Model_Params)
    plt.plot(loop.real, loop.imag, '.-')
    plt.show()


def test_new_parameters():
    # Jx = np.random.randint(-20, 21) / 10 + 1j * np.random.randint(-20, 21) / 10
    # Jy = np.random.randint(-20, 21) / 10 + 1j * np.random.randint(-20, 21) / 10
    Jx = 1 + 1j
    Jy = 1.5 + 1.2j
    deltax = 0.1 * np.pi
    deltay = 0.3 * np.pi
    gammax = 0.2
    gammay = 0.1

    print(Jx, Jy)
    Jx1, Jx2, Jy1, Jy2 = get_coupling_terms_from_new_param(Jx, Jy, gammax, gammay, deltax, deltay)

    bound, loop = get_11_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2)
    plt.plot(loop.real, loop.imag, '.-')

    bound, loop = get_x_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2)
    plt.plot(loop.real, loop.imag, '.-')
    plt.show()


def test_param_conversion():
    old_Js = np.random.randn(4) + 1j * np.random.randn(4)
    new_Js = get_coupling_terms_from_new_param(
        *get_new_param_from_coupling_terms(*old_Js)
    )
    print(old_Js - new_Js)
    print(la.norm(old_Js - new_Js))


if __name__ == '__main__':
    # test_11_DGBZ_spectrum()
    # test_new_parameters()
    test_param_conversion()
