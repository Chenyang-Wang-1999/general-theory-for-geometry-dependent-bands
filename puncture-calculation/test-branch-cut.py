'''
author:        Wang Chenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-09-01
Copyright © Department of Physics, Tsinghua University. All rights reserved

Test branch cut in the calculation of (1,1) direction
'''

import numpy as np
from cmath import exp, sqrt, log, pi
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize
from partial_GBZ_solver.util import get_winding_number


J_X1, J_X2, J_Y1, J_Y2 = (
    (1.218098989965905+2.218766496018679j),
    (0.5993724714661576-0.4281903947158586j),
    (-0.11693934094831741+0.3031403873104943j),
    (1.4555387777891486+0.9992129698174818j)
)
J_X1, J_X2, J_Y1, J_Y2 = np.random.randn(4) + 1j * np.random.randn(4)

print("Variables:", J_X1, J_X2, J_Y1, J_Y2)


def get_critical_angle():
    ''' Critical angle for branch cut '''
    p1 = (J_X1 * J_X2 + J_Y1 * J_Y2
          + 2 * sqrt(J_X1 * J_X2 * J_Y1 * J_Y2))
    p2 = (J_X1 * J_X2 + J_Y1 * J_Y2
          - 2 * sqrt(J_X1 * J_X2 * J_Y1 * J_Y2))
    return log(p1).imag, log(p2).imag


def get_expresions(
        beta: np.ndarray[complex]
) -> tuple[np.ndarray[complex], np.ndarray[complex]]:
    ''' Get J_{x2}\\beta_{1}+J_{y1} and J_{x1}\\beta_{1}^{-1}+J_{y2} '''
    expr1 = J_X2 * beta + J_Y1
    expr2 = J_X1 / beta + J_Y2
    return expr1, expr2


def calculate_branch_cut(t: np.ndarray[float], branch_cut_angle):
    ''' Calculate branch cut for z1 and z2 '''
    def get_beta_solution(curr_t: float):
        a = J_X2 * J_Y2
        b = - curr_t * exp(1j * branch_cut_angle) + J_X1 * J_X2 + J_Y1 * J_Y2
        c = J_X1 * J_Y1
        Delta = b**2 - 4 * a * c
        return (-b + sqrt(Delta)) / (2 * a), (-b - sqrt(Delta)) / (2 * a)

    def batch_get_beta_solution(beta_init: list[complex], t_list: np.ndarray[float]):
        beta1 = [beta_init[0]]
        beta2 = [beta_init[1]]
        for curr_t in t_list:
            sol1, sol2 = get_beta_solution(curr_t)
            if abs(sol1 - beta1[-1]) > abs(sol1 - beta2[-1]):
                beta2.append(sol1)
                beta1.append(sol2)
            else:
                beta2.append(sol2)
                beta1.append(sol1)
        beta1.pop(0)
        beta2.pop(0)
        return np.asarray(beta1), np.asarray(beta2)

    return batch_get_beta_solution([- J_X1 / J_Y2, - J_Y1 / J_X2], t)


def select_correct_ray(branch_cut_angle, t_max: float = 10):
    ''' Calculate branch cut and select correct side '''
    t_negative = np.linspace(0, -t_max, 100000)
    beta1, beta2 = calculate_branch_cut(t_negative, branch_cut_angle)
    if abs(beta1[-1]) < abs(beta2[-1]):
        return t_negative, beta1, beta2
    else:
        beta1, beta2 = calculate_branch_cut(-t_negative, branch_cut_angle)
        return -t_negative, beta1, beta2


def main_check_winding():
    r = np.linspace(0.05, 2, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    beta = r_mesh.flatten() * np.exp(1j * theta_mesh.flatten())
    expr1, expr2 = get_expresions(beta)
    mul = expr1 * expr2     # Multiplication of expr1 and expr2

    # get sqrt of expr1 and expr2
    sqrt_expr1 = np.sqrt(expr1)
    sqrt_expr2 = np.sqrt(expr2)
    sqrt_mul = np.sqrt(expr1 * expr2)

    # plot
    plt.figure(1)
    plt.plot(mul.real, mul.imag, '.')

    plt.figure(2)
    plt.plot(expr1.real, expr1.imag, '.')
    plt.plot(expr2.real, expr2.imag, '.')

    plt.figure(3)
    plt.plot(
        (sqrt_expr1 * sqrt_expr2).real,
        (sqrt_expr1 * sqrt_expr2).imag,
        'r.'
    )

    plt.figure(4)
    plt.plot(
        sqrt_mul.real, sqrt_mul.imag, 'b.'
    )
    plt.show()


def main_check_branch_cut():
    ''' Check branch cut for z1 and z2 '''
    branch_cut_angle = np.log(J_X1 * J_X2 + J_Y1 * J_Y2).imag
    t, beta1, beta2 = select_correct_ray(branch_cut_angle)
    # t = np.linspace(0, -100, 10000)
    # beta1, beta2 = calculate_branch_cut(t)

    # plt.figure(1)
    # plt.plot(beta1.real, beta1.imag, 'b')
    # plt.plot(beta2.real, beta2.imag, 'r')

    # t = np.linspace(0, 100, 10000)
    # beta1, beta2 = calculate_branch_cut(t)

    # plt.plot(beta1.real, beta1.imag, 'b')
    # plt.plot(beta2.real, beta2.imag, 'r')

    # plt.plot(0, 0, 'x')
    # plt.plot(beta1[0].real, beta1[0].imag, '+')
    # plt.plot(beta2[0].real, beta2[0].imag, '+')

    z11 = J_X1 / beta1 + J_Y2
    z21 = J_X2 * beta1 + J_Y1
    z12 = J_X1 / beta2 + J_Y2
    z22 = J_X2 * beta2 + J_Y1

    # plt.figure(1)
    # plt.plot(z11.real, z11.imag)
    # plt.figure(2)
    # plt.plot(z22.real, z22.imag)
    plt.figure(3)
    plt.plot(t, np.abs(z11))
    plt.plot(t, np.abs(z22))
    plt.plot(t, np.abs(z12), '--')
    plt.plot(t, np.abs(z21), '--')
    plt.legend(["z11", "z22", "z12", "z21"])

    # fig = plt.figure(4)
    # ax = fig.add_subplot(1, 2, 1)
    # plt.plot(
    #     (z11 * z21).real,
    #     (z11 * z21).imag
    # )
    # ax = fig.add_subplot(1, 2, 2)
    # plt.plot(
    #     (z12 * z22).real,
    #     (z12 * z22).imag
    # )
    plt.show()


def main_check_critical_branch_cut():
    ''' Check critical branch cut angle '''
    critical_branch_cut = get_critical_angle()

    ### 1. plot critical points ###
    plt.figure(1)
    for j in range(2):
        plt.plot(
            np.array([-10, 10]) * np.cos(critical_branch_cut[j]),
            np.array([-10, 10]) * np.sin(critical_branch_cut[j]),
        )
    p1 = (J_X1 * J_X2 + J_Y1 * J_Y2
          + 2 * sqrt(J_X1 * J_X2 * J_Y1 * J_Y2))
    p2 = (J_X1 * J_X2 + J_Y1 * J_Y2
          - 2 * sqrt(J_X1 * J_X2 * J_Y1 * J_Y2))
    plt.plot([p1.real, p2.real], [p1.imag, p2.imag], '--')

    ### 2. Check branch cut
    delta_theta = 1e-2
    branch_cut_angle1 = critical_branch_cut[1] + delta_theta
    branch_cut_angle2 = critical_branch_cut[1] - delta_theta

    def plot_branch_cut(branch_cut_angle, ax):
        t = np.linspace(0, -10, 10000)
        beta1, beta2 = calculate_branch_cut(t, branch_cut_angle)

        ax.plot(beta1.real, beta1.imag, 'b')
        ax.plot(beta2.real, beta2.imag, 'r')

        t = np.linspace(0, 10, 10000)
        beta1, beta2 = calculate_branch_cut(t, branch_cut_angle)

        ax.plot(beta1.real, beta1.imag, 'b')
        ax.plot(beta2.real, beta2.imag, 'r')

        ax.plot(0, 0, 'x')
        ax.plot(beta1[0].real, beta1[0].imag, '+')
        ax.plot(beta2[0].real, beta2[0].imag, '+')

    fig = plt.figure(2)
    ax = fig.gca()
    plot_branch_cut(branch_cut_angle1, ax)

    fig = plt.figure(3)
    ax = fig.gca()
    plot_branch_cut(branch_cut_angle2, ax)

    plt.show()


def get_branch_cut_intersections(branch_cut: np.ndarray[complex]):
    ''' Get all the intersections with real axis '''
    branch_cut_interp = interpolate.CubicSpline(
        np.abs(branch_cut),
        branch_cut,
        extrapolate=True
    )

    ascending_intersections = (branch_cut[:-1].imag < 0) & (branch_cut[1:].imag >= 0) & (branch_cut[:-1].real > 1e-6)
    descending_intersections = (branch_cut[:-1].imag >= 0) & (branch_cut[1:].imag < 0) & (branch_cut[:-1].real > 1e-6)
    ascending_intersections_left = branch_cut[:-1][ascending_intersections]
    ascending_intersections_right = branch_cut[1:][ascending_intersections]
    descending_intersections_left = branch_cut[:-1][descending_intersections]
    descending_intersections_right = branch_cut[1:][descending_intersections]
    ascending_intersections = np.zeros_like(
        ascending_intersections_left, dtype=float
    )
    descending_intersections = np.zeros_like(
        descending_intersections_left, dtype=float
    )
    for point_ind in range(len(ascending_intersections)):
        curr_point = optimize.brentq(
            lambda r: branch_cut_interp(r).imag,
            abs(ascending_intersections_left[point_ind]),
            abs(ascending_intersections_right[point_ind])
        )
        ascending_intersections[point_ind] = curr_point

    for point_ind in range(len(descending_intersections)):
        curr_point = optimize.brentq(
            lambda r: branch_cut_interp(r).imag,
            abs(descending_intersections_left[point_ind]),
            abs(descending_intersections_right[point_ind])
        )

    return branch_cut_interp, ascending_intersections, descending_intersections


def get_arg(z: np.ndarray[complex],
            branch_cut_interp: interpolate.CubicSpline,
            ascend_intersect: np.ndarray[float],
            descend_intersect: np.ndarray[float]):
    ''' Get the argument of z with given branch cut '''
    z_abs = np.abs(z)
    ascend_num_list = np.array([sum(ascend_intersect < curr_r) for curr_r in z_abs])
    descend_num_list = np.array([sum(descend_intersect < curr_r) for curr_r in z_abs])
    z_branch_cut = branch_cut_interp(z_abs)
    z_branch_cut = z_branch_cut / np.abs(z_branch_cut)

    # Calculate the angle
    arg_beta_list = (np.log(z * (-1 / z_branch_cut)).imag - np.log(-1 / z_branch_cut).imag
                     + (ascend_num_list - descend_num_list) * 2 * pi)  # real axis crosses the branch cut

    return arg_beta_list


def test_arbitrary_branch_cut():
    ''' Test how to set the branch cut '''
    branch_cut_angle = - 0.4 * pi
    branch_cut_point = exp(1j * branch_cut_angle)

    beta_list = np.exp(1j * np.linspace(0, 2 * pi, 100))

    arg_beta_list = np.log(beta_list * (-1 / branch_cut_point)).imag - log(-1 / branch_cut_point).imag

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(beta_list.real, beta_list.imag, arg_beta_list / pi)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    print(max(np.abs(beta_list - np.exp(1j * arg_beta_list))))
    plt.show()


def test_branch_cut_function():
    branch_cut_angle = np.log(J_X1 * J_X2 + J_Y1 * J_Y2).imag
    t, beta1, beta2 = select_correct_ray(branch_cut_angle)
    z1_branch_cut = J_X1 / beta1 + J_Y2

    # test z
    z_abs = np.linspace(1e-2, max(np.abs(z1_branch_cut)) - 1e-2, 50)
    z_arg = np.linspace(0, 2 * pi, 50)
    z_abs_mesh, z_arg_mesh = np.meshgrid(z_abs, z_arg)
    z = z_abs_mesh * np.exp(1j * z_arg_mesh)

    z = z.flatten()
    z_arg_branch_cut = get_arg(z, z1_branch_cut)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.plot(z.real, z.imag, z_arg_branch_cut / pi, '.')
    ax.plot(z1_branch_cut.real, z1_branch_cut.imag, np.zeros_like(z1_branch_cut).real, 'r-')
    plt.show()


def test_definition_z1_z2():
    ''' test the definition of z1 and z2 '''
    branch_cut_angle = np.log(J_X1 * J_X2 + J_Y1 * J_Y2).imag
    t, beta1, beta2 = select_correct_ray(branch_cut_angle, 1000)
    z1_branch_cut = J_X1 / beta1 + J_Y2
    z1_interp, z1_ascend, z1_descend = get_branch_cut_intersections(z1_branch_cut)
    print(z1_ascend, z1_descend)
    z2_branch_cut = J_X2 * beta2 + J_Y1
    z2_interp, z2_ascend, z2_descend = get_branch_cut_intersections(z2_branch_cut)
    print(z2_ascend, z2_descend)

    beta_abs = np.linspace(0.1, 10, 100)
    beta_arg = np.linspace(0, 2 * pi, 100)
    abs_mesh, arg_mesh = np.meshgrid(beta_abs, beta_arg)
    beta = abs_mesh * np.exp(1j * arg_mesh)
    beta = beta.flatten()

    z1 = J_X1 / beta + J_Y2
    z2 = J_X2 * beta + J_Y1
    varphi_1 = get_arg(z1, z1_interp, z1_ascend, z1_descend)
    varphi_2 = get_arg(z2, z2_interp, z2_ascend, z2_descend)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_xlim([min(z1.real), max(z1.real)])
    ax.set_ylim([min(z1.imag), max(z1.imag)])
    ax.plot(z1.real, z1.imag, varphi_1 / pi, '.')
    ax.plot(z1_branch_cut.real, z1_branch_cut.imag, np.zeros_like(z1_branch_cut).real, 'r-')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_xlim([min(z2.real), max(z2.real)])
    ax.set_ylim([min(z2.imag), max(z2.imag)])
    ax.plot(z2.real, z2.imag, varphi_2 / pi, '.')
    ax.plot(z2_branch_cut.real, z2_branch_cut.imag, np.zeros_like(z2_branch_cut).real, 'r-')

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(
        np.sqrt(np.abs(z1 * z2)) * np.cos((varphi_1 + varphi_2) / 2),
        np.sqrt(np.abs(z1 * z2)) * np.sin((varphi_1 + varphi_2) / 2),
        '.'
    )
    ax.plot([0, 10 * np.cos(branch_cut_angle)], [0, 10 * np.sin(branch_cut_angle)], 'r-')
    plt.show()


def check_branch_cut_ellipse():
    ''' Check whether the ellipse lies in single branch cut '''
    # 1. Get branch cut
    branch_cut_angle = np.log(-J_X1 * J_X2 + J_Y1 * J_Y2).imag
    t, beta1, beta2 = select_correct_ray(branch_cut_angle, 1000)
    z1_branch_cut = J_X1 / beta1 + J_Y2
    z2_branch_cut = J_X2 * beta2 + J_Y1

    # 2. Calculate z1 and z2
    r = sqrt(abs(J_X1 * J_Y1 / J_X2 / J_Y2))
    theta = np.linspace(0, 2 * pi, 100)
    beta = r * np.exp(1j * theta)
    z1 = J_X1 / beta + J_Y2
    z2 = J_X2 * beta + J_Y1
    varphi_1 = get_arg(z1, z1_branch_cut)
    varphi_2 = get_arg(z2, z2_branch_cut)
    z1z2 = z1 * z2
    sqrt_z1z2 = np.sqrt(np.abs(z1 * z2)) * np.exp(1j * (varphi_1 + varphi_2) / 2)

    # 3. Plot
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(z1.real, z1.imag)
    plt.plot(z1_branch_cut.real, z1_branch_cut.imag)
    plt.axis([
        min(z1.real) - 1,
        max(z1.real) + 1,
        min(z1.imag) - 1,
        max(z1.imag) + 1
    ])

    plt.subplot(2, 2, 2)
    plt.plot(z2.real, z2.imag)
    plt.plot(z2_branch_cut.real, z2_branch_cut.imag)
    plt.axis([
        min(z2.real) - 1,
        max(z2.real) + 1,
        min(z2.imag) - 1,
        max(z2.imag) + 1
    ])

    plt.subplot(2, 2, 3)
    plt.plot(z1z2.real, z1z2.imag)
    plt.plot(0, 0, 'x')

    plt.subplot(2, 2, 4)
    plt.plot(sqrt_z1z2.real, sqrt_z1z2.imag)

    plt.figure(2)
    s0 = varphi_2 - varphi_1
    plt.plot(theta, s0 / pi)

    plt.show()


def check_energy_continuity():
    # Check whether energy is continuous against s and theta
    # 1. Get branch cut
    branch_cut_angle = np.log(-J_X1 * J_X2 + J_Y1 * J_Y2).imag
    t, beta1, beta2 = select_correct_ray(branch_cut_angle, 1000)
    z1_branch_cut = J_X1 / beta1 + J_Y2
    z1_interp, z1_ascend, z1_descend = get_branch_cut_intersections(z1_branch_cut)
    z2_branch_cut = J_X2 * beta2 + J_Y1
    z2_interp, z2_ascend, z2_descend = get_branch_cut_intersections(z2_branch_cut)

    # 2. Calculate z1 and z2
    r = sqrt(abs(J_X1 * J_Y1 / J_X2 / J_Y2))
    theta = np.linspace(0, 2 * pi, 100)
    s = np.linspace(0, 2 * pi, 100)
    theta_mesh, s_mesh = np.meshgrid(theta, s)

    beta = r * np.exp(1j * theta)
    z1 = J_X1 / beta + J_Y2
    z2 = J_X2 * beta + J_Y1
    varphi_1 = get_arg(z1, z1_interp, z1_ascend, z1_descend)
    varphi_2 = get_arg(z2, z2_interp, z2_ascend, z2_descend)
    sqrt_z1z2 = np.sqrt(np.abs(z1 * z2)) * np.exp(
        1j * (varphi_1 + varphi_2) / 2
    )

    s0 = (varphi_2 - varphi_1)
    s0_mesh = np.asarray([s0 for _ in s])

    sqrt_z1z2_mesh = np.asarray([sqrt_z1z2 for _ in s])
    E = 2 * sqrt_z1z2_mesh * np.cos(s_mesh - s0_mesh / 2)

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(theta_mesh, s_mesh, E.real)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(theta_mesh, s_mesh, E.imag)

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(theta, s0)
    plt.subplot(1, 2, 2)
    plt.plot(E.flatten().real, E.flatten().imag, '.')

    plt.show()


def get_eigen_energy(
        beta: np.ndarray[complex],
        s: np.ndarray[float]
):
    # Get eigen energy for given theta and s
    z1 = J_X1 / beta + J_Y2
    z2 = J_X2 * beta + J_Y1
    exp_varphi1 = z1 / np.abs(z1)
    exp_varphi2 = z2 / np.abs(z2)
    return np.sqrt(np.abs(z1 * z2) * (
        exp_varphi2 * np.exp(-1j * s)
        + exp_varphi1 * np.exp(1j * s)
    ))


def calculate_winding_given_point(
        beta: complex,
        s: float,
        winding_radius: float = 1e-4
):
    ''' Given beta and s, calculate winding number '''
    # 1. get center energy
    E_center = get_eigen_energy(beta, s)

    # 2. get winding loop
    loop_angle = np.linspace(0, 2 * pi, 100)
    dtheta = winding_radius * np.sin(loop_angle)
    ds = winding_radius * np.cos(loop_angle)
    beta_winding = beta * np.exp(1j * dtheta)
    s_winding = s + ds

    # 3. get winding number
    E_winding = get_eigen_energy(beta_winding, s_winding)
    return get_winding_number(E_winding - E_center)


def test_11_winding():
    ''' Test the winding number of 11 stripe GBZ '''
    # 1. Get beta loop
    theta = np.linspace(0, 2 * pi, 100)
    u = pi / 3
    r = sqrt(abs(J_X1 * J_Y1 / J_X2 / J_Y2)) * 1.1
    theta_positive = []
    theta_negative = []
    for curr_theta in theta:
        beta = r * np.exp(1j * curr_theta)
        z1 = J_X1 / beta + J_Y2
        z2 = J_X2 * beta + J_Y1
        exp_s0 = z2/z1 / abs(z2/z1)
        s = u + log(exp_s0).imag / 2 + pi
        if calculate_winding_given_point(beta, s) > 0:
            theta_positive.append(curr_theta)
        else:
            theta_negative.append(curr_theta)
    theta_positive = np.asarray(theta_positive)
    theta_negative = np.asarray(theta_negative)

    # 3. plot
    plt.plot(0, 0, 'kx')
    beta_positive = r * np.exp(1j * theta_positive)
    beta_negative = r * np.exp(1j * theta_negative)
    ellp = (
        J_X1 * J_Y1 / beta_positive
        + J_X2 * J_Y2 * beta_positive
        + J_X1 * J_X2 + J_Y1 * J_Y2
    )
    plt.plot(ellp.real, ellp.imag, 'r.')
    ellp = (
        J_X1 * J_Y1 / beta_negative
        + J_X2 * J_Y2 * beta_negative
        + J_X1 * J_X2 + J_Y1 * J_Y2
    )
    plt.plot(ellp.real, ellp.imag, 'b.')
    plt.show()


def test_11_winding_beta2_GBZ():
    ''' test the winding number distribution on beta2 GBZ '''
    # 1. Get beta loop
    theta = pi / 3
    r = sqrt(abs(J_X1 * J_Y1 / J_X2 / J_Y2)) * 0.9
    beta = r * np.exp(1j * theta)
    z1 = J_X1 / beta + J_Y2
    z2 = J_X2 * beta + J_Y1
    exp_s0 = z2/z1 / abs(z2/z1)
    s = np.linspace(0, 2*pi, 100)
    s_positive = []
    s_negative = []
    for curr_s in s:
        if calculate_winding_given_point(beta, curr_s) > 0:
            s_positive.append(curr_s)
        else:
            s_negative.append(curr_s)
    s_positive = np.asarray(s_positive)
    s_negative = np.asarray(s_negative)

    # 3. plot
    plt.plot(np.cos(s_positive), np.sin(s_positive), 'r.')
    plt.plot(np.cos(s_negative), np.sin(s_negative), 'b.')
    # plt.plot(s_positive, np.cos(s_positive) ** 2, 'r.')
    # plt.plot(s_negative, np.cos(s_negative) ** 2, 'b.')
    plt.plot(sqrt(exp_s0).real, sqrt(exp_s0).imag, 'kx')
    plt.plot(-sqrt(exp_s0).real, -sqrt(exp_s0).imag, 'kx')
    plt.show()


if __name__ == '__main__':
    # main_check_winding()
    # main_check_branch_cut()
    # main_check_critical_branch_cut()
    # test_arbitrary_branch_cut()
    # test_branch_cut_function()
    # test_definition_z1_z2()
    # check_branch_cut_ellipse()
    # check_energy_continuity()
    test_11_winding()
    # test_11_winding_beta2_GBZ()
