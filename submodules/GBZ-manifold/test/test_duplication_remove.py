'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-03-30
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import unittest
import matplotlib.pyplot as plt
import numpy as np
import GBZ_manifold as Gm

class TestDuplicationRemove(unittest.TestCase):
    def test_circle(self):
        '''
            Generate two arcs of an circle, 
            one is from 0 to pi, and the other is from 0.5pi to 1.5pi.
        '''
        r = 1.0
        theta1 = np.linspace(0, np.pi, 100)
        theta2 = np.linspace(0.5*np.pi, 1.5*np.pi, 100)
        glue_tol = np.pi/100
        z1 = r * np.exp(1j * theta1)
        z2 = r * np.exp(1j * theta2)
        point_vec1 = Gm.CP1PointVec()
        point_vec2 = Gm.CP1PointVec()
        for idx in range(len(z1)):
            point_vec1.append(Gm.CP1PointData([0, z1[idx]], [0, 0]))
            point_vec2.append(Gm.CP1PointData([0, z2[idx]], [0, 0]))
        Gm.batch_to_standard(point_vec1)
        Gm.batch_to_standard(point_vec2)
        results = Gm.remove_duplicated_points(point_vec1, point_vec2, glue_tol)
        print(results)
        plt.plot(z1.real, z1.imag, '.', label='z1')
        plt.plot(z2.real, z2.imag, 'x', label='z2')
        for idx_range in results[1]:
            plt.plot(
                z1[idx_range[0]:idx_range[1]].real, 
                z1[idx_range[0]:idx_range[1]].imag, 
                '+', 
                label='z2-selected')
        plt.show()
        # for point in point_vec1:
        #     print(Gm.calculate_min_distance(point, point_vec2), glue_tol)
