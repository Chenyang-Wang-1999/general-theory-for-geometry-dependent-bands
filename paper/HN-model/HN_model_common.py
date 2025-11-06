'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-06-01
Copyright © Department of Physics, Tsinghua University. All rights reserved

Ultilities related to HN model for the paper plot
'''

import BerryPy.TightBinding as tb
import numpy as np
from cmath import sqrt
import numpy_extended as npe
import matplotlib.pyplot as plt

Default_Model_Params = (
    1 + 1j,
    1.5 + 1.2j,
    -1 + 1j,
    -1.2 - 0.5j
)


def get_HN_model(Jx1, Jx2, Jy1, Jy2):
    inter_cell = [
        [0, 0, Jx1, (1, 0)],
        [0, 0, Jx2, (-1, 0)],
        [0, 0, Jy1, (0, 1)],
        [0, 0, Jy2, (0, -1)]
    ]
    return tb.TightBindingModel(2, 1, np.array([[1, 0], [0, 1]]), [], inter_cell, [[0, 0]])


def get_11_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2):
    bound = 2 * npe.continuous_sqrt(
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
    Jx1 = gammax * Jx * np.exp(1j * deltax)
    Jx2 = 1 / gammax * (Jx.conjugate()) * np.exp(1j * deltax)
    Jy1 = gammay * Jy * np.exp(1j * deltay)
    Jy2 = 1 / gammay * (Jy.conjugate()) * np.exp(1j * deltay)
    return Jx1, Jx2, Jy1, Jy2


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


if __name__ == '__main__':
    test_11_DGBZ_spectrum()
    # test_new_parameters()
