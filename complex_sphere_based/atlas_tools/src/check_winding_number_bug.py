'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-08-24
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
from complex_sphere_based.atlas_tools.src.GBZ_solve_2d_test import *

def load_data():
    with open("data/bug_points.pkl", "rb") as fp:
        bug_points = pickle.load(fp)
    with open("data/bug_params.pkl", 'rb') as fp:
        bug_params = pickle.load(fp)
    
    new_bug_points = []
    new_bug_params = []
    for j in range(len(bug_points)):
        if(abs(abs(bug_points[j][2]) - 1) < 1e-6):
            new_bug_points.append(bug_points[j])
            new_bug_params.append(bug_params[j])
    return new_bug_points, new_bug_params

def main_check_bug_points():
    vx1 = 1.0 + np.random.randn() * 0
    vx2 = 1.0 + np.random.randn() * 0
    ux1 = 3.0 + np.random.randn() * 0
    ux2 = 3.0 + np.random.randn() * 0
    uy1 = 4.0 + np.random.randn() * 0
    uy2 = 4.0 + np.random.randn() * 0
    coeffs = csolver.CScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0,
    ])

    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    bug_points, bug_params = load_data()
    for p in bug_points:
        ax.plot(p[2].real, p[2].imag, p[0].real, '.')
    plt.show()
    for j in range(len(bug_params)):
        csolver.check_GBZ_condition(char_poly, bug_points[j][:3], bug_params[j], 2)
    
if __name__ == '__main__':
    main_check_bug_points()