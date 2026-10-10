'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-03-18
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import unittest
from GBZ_manifold import * 
import numpy as np
import random

class TestManifoldBasics(unittest.TestCase):
    def test_batch_to_standard(self):
        N = 10
        A = np.random.randn(N, 2) + 1j * np.random.randn(N, 2)
        B = np.random.randint(2, size=(N))
        print(A, B)

        point_vec = GBZPointVec()
        for j in range(N):
            point_vec.append(
                GBZPointData(list(A[j,:]), [B[j]])
            )
        for j in range(N):
            if(abs(A[j,1]) > 1):
                B[j] = not B[j]
                A[j,1] = 1/A[j,1]
        for j in range(N):
            print(point_vec[j].chart_labels, point_vec[j].coords)
        print()
        batch_to_standard(point_vec)
        for j in range(N):
            print(point_vec[j].chart_labels, point_vec[j].coords)
        print(A)
        print(B)
        
    def test_get_real_coord(self):
        N = 10
        A = np.random.randn(N, 2) + 1j * np.random.randn(N, 2)
        B = np.random.randint(2, size=(N))
        print(A, B)

        point_vec = GBZPointVec()
        for j in range(N):
            point_vec.append(
                GBZPointData(list(A[j,:]), [B[j]])
            )
        
        real_points = Point3DVec()
        batch_get_point_3D_coord(point_vec, real_points)
        real_points_theory = np.zeros((len(point_vec), 3))
        for j in range(len(point_vec)):
            if(point_vec[j].chart_labels[0] == 0):
                # north
                beta = point_vec[j].coords[1]
                C = np.abs(beta) ** 2 + 1
                real_points_theory[j,0] = 2 * beta.real / C
                real_points_theory[j,1] = 2 * beta.imag / C
                real_points_theory[j,2] = (abs(beta) ** 2 - 1) / C
            else:
                # south
                gamma = point_vec[j].coords[1]
                C = abs(gamma) ** 2 + 1
                real_points_theory[j,0] = 2 * gamma.real / C
                real_points_theory[j,1] = - 2 * gamma.imag / C
                real_points_theory[j,2] = (1 - abs(gamma) ** 2) / C
            
        for j in range(len(point_vec)):
            print(real_points[j], 
                  np.linalg.norm([
                      real_points[j][k] - real_points_theory[j,k] for k in range(3)
                  ]))

    def test_point_to_numpy_array(self):
        point_list = Point3DVec()
        for j in range(100):
            point_list.append(np.random.randn(3))
        print(point_list)
        print(np.asarray(point_list))
    
    def test_point_match(self):
        N = 10
        A = np.random.randn(N, 2) + 1j * np.random.randn(N, 2)
        B = np.random.randint(2, size=(N))
        print(A, B)

        point_vec = GBZPointVec()
        for j in range(N):
            point_vec.append(
                GBZPointData(list(A[j,:]), [B[j]])
            )

        for j in range(N):
            print(point_vec[j].chart_labels, point_vec[j].coords)
        print()

        seq = np.arange(N)
        random.shuffle(seq)
        new_point_vec = GBZPointVec()
        for j in range(N):
            new_point_vec.append(point_vec[seq[j]])
            new_point_vec[-1].coords = [
                x + (np.random.randn() + 1j * np.random.randn()) * 1e-3 for x in new_point_vec[-1].coords]
        
        print(seq)
        for j in range(N):
            print(new_point_vec[j].chart_labels, new_point_vec[j].coords)

        results = GBZPointVec()
        match_point_list(point_vec, new_point_vec, results)
        for j in range(N):
            print("next", results[j].chart_labels, results[j].coords)
            print("previous", point_vec[j].chart_labels, point_vec[j].coords)

        for j in range(N):
            print("distance", get_distance(point_vec[j], results[j]))

