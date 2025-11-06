'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-09-04
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import test_winding
import numpy as np
import matplotlib.pyplot as plt

def generate_bubble():
    beta_center = 1.5 * np.exp(2j*np.pi/3)
    outer_path = test_winding.CComplexVec([])
    inner_path = test_winding.CComplexVec([])
    
    remaining_phase = 1e-2
    rel_r_detour = 2 * np.sin(remaining_phase)
    test_winding.bubble_gen(beta_center, 
            rel_r_detour, remaining_phase, 100, 
            outer_path, inner_path)
    
    circle = beta_center * np.exp(1j * np.linspace(0, 2 * np.pi, 1000))

    plt.plot(np.asarray(outer_path).real, np.asarray(outer_path).imag)
    plt.plot(np.asarray(inner_path).real, np.asarray(inner_path).imag)
    plt.plot(circle.real, circle.imag)
    plt.show()

def test_long_bubble():
    beta1 = 1.5 * np.exp(2j*np.pi/3)
    beta2 = beta1 * np.exp(5e-3j)
    outer_begin = test_winding.CComplexVec([])
    inner_begin = test_winding.CComplexVec([])
    outer_end = test_winding.CComplexVec([])
    inner_end = test_winding.CComplexVec([])
    outer_arc = test_winding.CComplexVec([])
    inner_arc = test_winding.CComplexVec([])
    
    remaining_phase = 1e-2
    rel_r_detour = 2 * np.sin(remaining_phase)
    test_winding.long_bubble_gen([beta1, beta2], 
            rel_r_detour, remaining_phase, 100, 100,
            outer_begin, inner_begin, outer_arc, inner_arc, outer_end, inner_end)
    
    circle = beta1 * np.exp(1j * np.linspace(0, 2 * np.pi, 1000))

    plt.plot(np.asarray(outer_begin).real, np.asarray(outer_begin).imag)
    plt.plot(np.asarray(inner_begin).real, np.asarray(inner_begin).imag)
    plt.plot(np.asarray(outer_arc).real, np.asarray(outer_arc).imag)
    plt.plot(np.asarray(inner_arc).real, np.asarray(inner_arc).imag)
    plt.plot(np.asarray(outer_end).real, np.asarray(outer_end).imag)
    plt.plot(np.asarray(inner_end).real, np.asarray(inner_end).imag)
    plt.plot(circle.real, circle.imag)
    plt.show()
 
def test_winding_loop():
    outer_winding_real = np.loadtxt("data/test_outer_loop_real.csv", delimiter=',')
    outer_winding_imag = np.loadtxt("data/test_outer_loop_imag.csv", delimiter=',')
    inner_winding_real = np.loadtxt("data/test_inner_loop_real.csv", delimiter=',')
    inner_winding_imag = np.loadtxt("data/test_inner_loop_imag.csv", delimiter=',')
    plt.plot(outer_winding_real, outer_winding_imag, '.-')
    plt.plot(inner_winding_real, inner_winding_imag, '.-')
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    # generate_bubble()
    # test_long_bubble()
    test_winding_loop()
