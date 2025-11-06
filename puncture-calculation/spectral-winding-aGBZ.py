'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-04-29
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import poly_tools as pt
import partial_GBZ_solver as pGs
import numpy as np
import matplotlib.pyplot as plt
import GBZ_manifold as Gm

def test_char_poly():
    t1 = 1
    t2 = 0.5
    t3 = 0.2
    coeffs = pt.CScalarVec([
        1, -t2, -t1, -t1, -t2, -t3, -t3 
    ])
    degs = pt.CLaurentIndexVec([
        # E, beta
        1, 0,
        0, -2,
        0, -1,
        0, 1,
        0, 2,
        0, -3,
        0, 3
    ])   
    f = pt.CLaurent(2)
    f.set_Laurent_by_terms(coeffs, degs)
    return f

def draw_aGBZ_winding():
    f = test_char_poly()
    GBZ_data, phi_data, aGBZ_data = pGs.solve_GBZ("GBZ", f, 800)

    E_list = []
    beta_list = []
    beta2_list = []
    for j in range(len(aGBZ_data)):
        for point in aGBZ_data[j]:
            Gm.GBZ_to_chart(point,[0])
            if(pGs.check_GBZ_condition_with_offset(
                f, point.coords[1], point.coords[0], 1e-5, -2
            )):
                E_list.append(point.coords[0])
                beta_list.append(point.coords[1])
                beta2_list.append(point.coords[1] * np.exp(1j * phi_data[j][0]))
    
    E_list = np.asarray(E_list)
    beta_list = np.asarray(beta_list)
    beta2_list = np.asarray(beta2_list)

    plt.figure(1)
    plt.plot(E_list.real, E_list.imag,'.')
    plt.figure(2)
    plt.plot(beta_list.real, beta_list.imag, '.')
    plt.plot(beta2_list.real, beta2_list.imag, '.')
    theta = np.linspace(0, 2 * np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), '--')
    plt.show()

if __name__ == '__main__':
    draw_aGBZ_winding()


