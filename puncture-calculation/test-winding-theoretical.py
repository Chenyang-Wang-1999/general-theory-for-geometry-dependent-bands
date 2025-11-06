'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-08-29
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from math import sin, cos, pi,sqrt
from cmath import exp, log
import partial_GBZ_solver as pGs
import matplotlib.pyplot as plt

# Jx1, Jy1, Jx2, Jy2 = [ 0.08956388, -0.96803433, -0.20686002,  0.98633881]
Jx1, Jy1, Jx2, Jy2 = np.random.randn(4) + 1j * np.random.randn(4)

phi_x1 = log(Jx1).imag
phi_x2 = log(Jx2).imag
phi_y1 = log(Jy1).imag
phi_y2 = log(Jy2).imag
phibar_x = (phi_x1 + phi_x2) / 2
phibar_y = (phi_y1 + phi_y2) / 2
r = 1
A1 = abs(Jx2) * r + abs(Jx1) / r
B1 = abs(Jx2) * r - abs(Jx1) / r
theta0 = (phi_x2 - phi_x1) / 2
print("A1:", A1)
print("B1:", B1)

def get_E_xy(theta, s):
    E = (exp(1j * phibar_x) * (
            A1 * np.cos(theta + theta0) + 1j * B1 * np.sin(theta + theta0))
        + 2 * exp(1j * phibar_y) * sqrt(abs(Jy1) * abs(Jy2)) * np.cos(s))
    return E

def get_winding_number(E_fun, theta_center, s_center, rho=1e-3):
    # get winding number around theta_center, s_center
    u = np.linspace(0, 2*pi, 100)
    E_center = E_fun(theta_center, s_center)
    E_loop = E_fun(theta_center + rho * np.sin(u),
                    s_center + rho * np.cos(u))
    return pGs.get_winding_number(E_loop - E_center)

def main_plot_winding_xy():
    theta = np.linspace(0, 2*np.pi, 100)
    s = np.random.rand() * pi + pi
    print("s=",s/pi,"pi")
    positive_theta = []
    negative_theta = []
    other_theta = []
    for curr_theta in theta:
        if(abs(get_winding_number(get_E_xy, curr_theta, s) - 1) < 1e-3):
            positive_theta.append(curr_theta)
        elif(abs(get_winding_number(get_E_xy, curr_theta, s) + 1) < 1e-3):
            negative_theta.append(curr_theta)
        else:
            print(curr_theta/pi,"pi: winding number = ", get_winding_number(get_E_xy, curr_theta, s))
            other_theta.append(curr_theta)
    positive_theta = np.asarray(positive_theta)
    negative_theta = np.asarray(negative_theta)
    other_theta = np.asarray(other_theta)

    # plot
    def get_vxy(theta):
        return exp(1j * (phibar_x - phibar_y)) * (A1 * np.cos(theta + theta0) + 1j * B1 * np.sin(theta + theta0))
    vxy = get_vxy(positive_theta)
    plt.plot(vxy.real, vxy.imag, 'r.')
    vxy = get_vxy(negative_theta)
    plt.plot(vxy.real, vxy.imag, 'b.')
    vxy = get_vxy(other_theta)
    plt.plot(vxy.real, vxy.imag, 'g.')
    plt.show()

if __name__ == '__main__':
    main_plot_winding_xy()