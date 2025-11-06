'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-01-22
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import unittest
import partial_GBZ_solver as pGs
import poly_tools as pt
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

coeffs = pt.CScalarVec([
    1, -1.15, -0.3, -0.3, -0.3, -0.3, 0.02, 0.02, 0.02, 0.02, 
    -0.5, -0.4, 0.04, 0.04
])
degs = pt.CLaurentIndexVec([
    2, 0, 0, 
    0, 0, 0,
    1, 1, 0,
    1, -1, 0,
    1, 0, -1,
    1, 0, 1,
    0, 2, 0,
    0, -2, 0,
    0, 0, 2,
    0, 0, -2,
    0, 1, 1,
    0, -1, -1,
    0, 1, -1,
    0, -1, 1 
])

f = pt.CLaurent(3)
f.set_Laurent_by_terms(coeffs, degs)

class TestPunctureSolver(unittest.TestCase):
    # def test_1d_equation_generation(self):
    #     char_poly = pGs.CharPolyND(f)
    #     target_fun = pGs.PunctureTargetFun(char_poly, 0)
    #     target_fun.set_E(0)
    #     target_fun.calculate_minor_beta(1)

    def test_derivative(self):
        char_poly = pGs.CharPolyND(f)
        target_fun = pGs.PunctureTargetFun(char_poly, 0)
        target_fun.set_E(0)
        
        # generate two beta
        beta = np.random.randn() + 1j * np.random.randn()
        lower_beta, upper_beta, dlower_beta, dupper_beta = target_fun.calculate_minor_beta(beta)
        delta_beta = (np.random.randn() + 1j * np.random.randn()) * 1e-5
        lower_beta1, upper_beta1, _, _ = target_fun.calculate_minor_beta(beta + delta_beta)

        # compare the two derivative
        print(lower_beta1 - lower_beta)
        print(delta_beta * dlower_beta)
        print(abs(lower_beta1 - lower_beta-delta_beta*dlower_beta)/abs(delta_beta*dlower_beta))
        print(upper_beta1 - upper_beta)
        print(delta_beta * dupper_beta)
        print(abs(upper_beta1 - upper_beta-delta_beta*dupper_beta)/abs(delta_beta*dupper_beta))
    
    def test_norm_diff(self):
        char_poly = pGs.CharPolyND(f)
        target_fun = pGs.PunctureTargetFun(char_poly, 0)
        target_fun.set_E(0)
        
        # generate two beta
        beta = np.random.randn() + 1j * np.random.randn()
        norm_diff_1, _ = target_fun.get_betanorm_difference(beta)
        delta_k = np.random.randn() * 1e-3
        _, dnorm = target_fun.get_betanorm_difference(beta*np.exp(1j*delta_k/2))
        norm_diff_2, _ = target_fun.get_betanorm_difference(beta*np.exp(1j*delta_k))

        # compare
        print(norm_diff_1, norm_diff_2)
        print(norm_diff_2 - norm_diff_1)
        print(dnorm * delta_k)
        print(abs(norm_diff_2 - norm_diff_1 - dnorm * delta_k)/abs(dnorm*delta_k))

    def test_minor_beta(self):    
        char_poly = pGs.CharPolyND(f)
        target_fun = pGs.PunctureTargetFun(char_poly, 0)
        target_fun.set_E(0.7)
        r = 1
        k = np.linspace(0, 2 * np.pi, 100)
        beta_diff = np.zeros_like(k)
        for j, curr_k in enumerate(k):
            beta_diff[j] = target_fun.calculate_minor_beta(r* np.exp(1j*curr_k))
        
        plt.plot(k, beta_diff)
        plt.show()
    
    def test_brentq(self):
        char_poly = pGs.CharPolyND(f)
        target_fun = pGs.PunctureTargetFun(char_poly, 0)
        target_fun.set_E(0.7)
        r = 1
        cost_fun = lambda k: target_fun.calculate_minor_beta(r * np.exp(1j * k))
        res = optimize.minimize_scalar(cost_fun, (2,3,4))
        print(res)
        # k = np.linspace(0, 2 * np.pi, 100)
        # beta_diff = np.zeros_like(k)
        # for j, curr_k in enumerate(k):
        #     beta_diff[j] = target_fun.calculate_minor_beta(r* np.exp(1j*curr_k))
        
        # plt.plot(k, beta_diff)
        # plt.show()

    def test_minima_solving(self):
        char_poly = pGs.CharPolyND(f)
        target_fun = pGs.PunctureTargetFun(char_poly, 0)
        target_fun.set_E(0.7)
        r = 1
        k0 = np.pi*98/100
        
        cost_fun = lambda k: target_fun.get_betanorm_difference(r * np.exp(1j*k))

        # k1 = np.random.randn() 
        # dk = np.random.randn() * 1e-4

        # f1, _ = cost_fun(k1)
        # f2, _ = cost_fun(k1 + dk)
        # _, df = cost_fun(k1 + dk/2)
        # print(f2- f1)
        # print(df*dk)
        # print(abs(f2-f1-df*dk)/abs(df*dk))

        res = optimize.minimize(cost_fun, k0, jac=True, method='TNC', options={'xtol':1e-10})
        print(res)

        # k = np.linspace(0, 2 * np.pi, 100)
        # beta_diff = np.zeros_like(k)
        # for j, curr_k in enumerate(k):
        #     beta_diff[j] = target_fun.get_betanorm_difference(r* np.exp(1j*curr_k))[0]
        
        # plt.plot(k, beta_diff)
        # plt.show()
        
    def test_interval_merge(self):
        N = np.random.randint(1,20) # number of intervals
        start_points = np.random.randn(N) * 100 
        interval_lengths = np.random.rand(N) * 80
        intervals = [[start_points[j], start_points[j] + interval_lengths[j]] for j in range(N)]

        merged_intervals = pGs.get_disconnect_intervals(intervals)
        print(len(merged_intervals))

        fig = plt.figure()
        for j, curr_inter in enumerate(intervals):
            plt.plot(curr_inter, [j,j], 'b-')
        
        for curr_inter in merged_intervals:
            plt.plot(curr_inter, [-1,-1], 'r-')
        plt.show()

    def test_derivative_at_zero(self):
        a, b, c = np.random.randn(3)
        coeffs = pt.CScalarVec([
            a, -b, -c
        ])
        degs = pt.CLaurentIndexVec([
            # E, beta
            1, 0,
            0, 1,
            0, -1
        ])
        f = pt.CLaurent(2)
        f.set_Laurent_by_terms(coeffs, degs)
        char_poly_fun = pGs.aGBZEquation1D(f)
        char_poly_fun.set_params([0,])

        x = np.random.randn(2) + 1j * np.random.randn(2)
        f_th = np.array([
            a * x[0] - b * x[1] - c / x[1],
            -b + c / (x[1]**2)
        ])
        J_th = np.array([
            [a, -b + c / (x[1]**2)],
            [0, -2 * c / (x[1]**3)]
        ])

        f_num, J_num = char_poly_fun(x)
        print(np.linalg.norm(f_num - f_th))
        print(np.linalg.norm(J_num - J_th))