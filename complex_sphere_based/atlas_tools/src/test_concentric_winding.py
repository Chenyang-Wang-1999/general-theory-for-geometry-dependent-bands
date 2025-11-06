'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-08-28
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import c_solver_2D as csolver
import numpy as np
from math import sin, cos, sqrt, pi
import matplotlib.pyplot as plt

def get_char_poly(t1, t2, t3):
    coeffs = csolver.CScalarVec([
        1, - t2 * t3, - t2 * t3, 
        - (t1*t2 + t1 * t3), -(t1*t2 + t1 * t3),
        -(t1**2 + t2 ** 2 + t3 ** 2)
    ])
    orders = csolver.CIntVec([
        # E, beta
        2, 0, 
        0, -2,
        0, 2,
        0, 1,
        0, -1,
        0, 0
    ])

    char_poly = csolver.CLaurant(2)
    char_poly.set_Laurant_by_terms(coeffs, orders)
    return char_poly

def get_spectrum(beta, t1, t2, t3):
    Esq = (t1 + t2/beta + t3 * beta) * (t1 + t2 * beta + t3/beta)
    return np.sqrt(Esq)

def test_winding():
    t1 = 1
    t2 = 0.2
    t3 = 0.2
    char_poly = get_char_poly(t1, t2, t3)

    k0 = pi*np.random.randn()
    E0 = get_spectrum(np.exp(1j*k0), t1, t2, t3)
    x0 = csolver.CScalarVec([E0, np.exp(1j*k0)])

    # evaluate 
    phi_list = np.linspace(0, 2*pi, 200)
    val_list = np.zeros_like(phi_list, dtype=complex)
    for j, phi in enumerate(phi_list):
        x0[1] = np.exp(1j*k0) * np.exp(1j*phi)
        val_list[j] = char_poly.eval(x0)
    
    # show
    plt.plot(phi_list/pi, np.abs(val_list)**2)
    plt.plot(phi_list/pi, phi_list*0, 'k-')
    plt.plot(phi_list/pi, np.ones_like(phi_list) * np.max(np.abs(val_list)**2)*1e-3)
    plt.show()

if __name__ == '__main__':
    test_winding()

