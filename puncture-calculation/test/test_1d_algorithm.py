'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-01-25
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''
import unittest
import matplotlib.pyplot as plt
import numpy as np
import poly_tools as pt
import partial_GBZ_solver as pGs


class TestAlgorithm1D(unittest.TestCase):
    def test_main_SSH(self):
        t1 = 1
        t2 = 1
        t3 = 0.5
        gamma = 0.2
        f_num = pt.CPolyLinkedList(2)
        f_num_coeffs = pt.CScalarVec([
            -t2*t3, - (t1*(t2 + t3) + gamma * (t3 - t2)/2), 
            -(-gamma**2/4 + t1**2 + t2**2 + t3**2), 
            -(t1*(t2 + t3) + gamma * (t2 - t3)/2), 
            -t2 * t3, 1
        ])
        f_num_orders = pt.CLaurentIndexVec([
            0, -2,  
            0, -1,
            0, 0,
            0, 1,
            0, 2,
            2, 0
        ])

        f = pt.CLaurent(2)
        f.set_Laurent_by_terms(f_num_coeffs, f_num_orders)

        aGBZ, exp_iphi = pGs.solve_aGBZ(f)
        GBZ_loops, phi_loops = pGs.pick_GBZ(f, aGBZ, np.log(exp_iphi).imag, 1e-3)
        print(GBZ_loops)
        print(phi_loops)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for j,loop in enumerate(GBZ_loops):
            ax.plot(loop[:,1].real, loop[:,1].imag, phi_loops[j][:,0])
        plt.show()
