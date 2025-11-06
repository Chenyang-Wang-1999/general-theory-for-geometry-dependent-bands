'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-02-17 00:00:00
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''
import poly_tools as pt
import numpy as np
from scipy import linalg as la
from math import pi
import GBZ_manifold as Gm
import sys
from .intersections import *
from .util import get_winding_number


def calculate_topological_charge(
    dchar_poly: list[pt.CLaurent],
    E_ref: complex,
    point_pair: tuple[np.ndarray[1,complex]]
):
    ''' Calculate the topological charge around a point
        Args:
            dchar_poly: List of derivatives of the characteristic polynomial
                with respect to E_ref, beta1 and beta2
            E_ref: Reference energy
            point_pair: a pair of points with format [beta1, beta2]
        Return:
            topo_charge: topological charge of the first point'''

    # 1. Get beta2 derivative at the two points
    beta2_diff = [
        get_beta2_derivative(
            dchar_poly[1],
            dchar_poly[2],
            E_ref, point[0], point[1]
        )
        for point in point_pair
    ]
    beta2_norm_diff = np.real([
        point_pair[j][1].conjugate() * 1j * point_pair[j][0] * beta2_diff[j]
        for j in range(2)
    ])

    # Compare two diff
    if beta2_norm_diff[0] > beta2_norm_diff[1]:
        topo_charge = 1
    else:
        topo_charge = -1
    return topo_charge


def get_beta2_norm_around_loop(
    char_poly: pt.CLaurent,
    E_ref: complex,
    beta1_loop: np.ndarray[1,complex]
):
    ''' Get the norm of beta2 '''
    beta2_norm = np.zeros(len(beta1_loop), dtype=float)
    for ind in range(len(beta1_loop)):
        beta2_list = solve_minor_beta(
            char_poly, E_ref, beta1_loop[ind], [0, 1]
        )
        beta2_norm[ind] = (abs(beta2_list[0]) + abs(beta2_list[1])) / 2

    return beta2_norm


def calculate_winding_around_loop(
    char_poly: pt.CLaurent,
    E_ref: complex,
    radius_Cr: float,
    theta1_list: np.ndarray[1,float],
    theta2_list: np.ndarray[1,float]
):
    ''' Calculate the winding number around a loop
        Args:
            char_poly: Characteristic polynomial of the non-Bloch Hamiltonian
            radius_Cr: Radius of Cr
            theta1_list: List of theta1
            theta2_list: List of theta2
        Return:
            winding_number: Winding number of the loop
    '''
    beta1_list = radius_Cr * np.exp(1j * theta1_list)
    beta2_norm = get_beta2_norm_around_loop(
        char_poly, E_ref, beta1_list
    )
    beta2_list = beta2_norm * np.exp(1j * theta2_list)

    f_vals = np.zeros(len(theta1_list), dtype=complex)
    for ind in range(len(beta2_list)):
        f_vals[ind] = char_poly.eval(
            pt.CScalarVec([E_ref, beta1_list[ind], beta2_list[ind]])
        )
    return np.round(get_winding_number(f_vals))


def calculate_loop_winding_self_adapted(
    char_poly: pt.CLaurent,
    E_ref: complex,
    radius_Cr: float,
    theta1_list: np.ndarray[1,float],
    theta2: float,
    max_incre: float = pi / 12,
    debug_mode=False
):
    ''' Calculate the winding number around a loop with self-adapted algorithm
        Args:
            char_poly: Characteristic polynomial of the non-Bloch Hamiltonian
            radius_Cr: Radius of Cr
            theta1_list: Initial list of theta1
            theta2: Fixed theta2 value
            max_incre: Maximum allowed phase increment between adjacent points
            debug_mode: If True, return additional debug information
        Return:
            winding_number: Winding number of the loop
    '''
    total_winding = 0.0

    def eval_fval_for_theta1(theta1):
        beta1 = radius_Cr * np.exp(1j * theta1)
        beta2_norm = get_beta2_norm_around_loop(char_poly, E_ref, np.array([beta1]))[0]
        beta2 = beta2_norm * np.exp(1j * theta2)
        f_val = char_poly.eval(pt.CScalarVec([E_ref, beta1, beta2]))
        return np.angle(f_val)

    theta_stack = list(theta1_list[::-1])
    if debug_mode:
        all_points = set(theta_stack)
    curr_theta = theta_stack.pop()
    curr_phase = eval_fval_for_theta1(curr_theta)

    while theta_stack:
        next_data = theta_stack.pop()
        if isinstance(next_data, tuple):
            next_theta, next_phase = next_data
        else:
            next_theta = next_data
            next_phase = eval_fval_for_theta1(next_theta)

        # Calculate phase increment
        phase_incre = ((next_phase - curr_phase + pi) % (2 * pi)) - pi
        if abs(phase_incre) < max_incre:
            total_winding += phase_incre
            curr_theta = next_theta
            curr_phase = next_phase
        else:
            theta_stack.append((next_theta, next_phase))
            mid_theta = (next_theta + curr_theta) / 2
            theta_stack.append(mid_theta)
            if debug_mode:
                all_points.add(mid_theta)

    total_winding /= 2 * pi

    if debug_mode:
        return np.round(total_winding), all_points
    else:
        return np.round(total_winding)


def pair_points(point_list: list[np.ndarray[1,complex]]):
    ''' Pair points according to the beta1 part, this process will destroy the original list'''
    point_pairs = []
    assert(len(point_list) % 2 == 0)
    while point_list:
        curr_point = point_list.pop(0)
        point_distance = [
            abs(curr_point[0] - point[0]) for point in point_list
        ]
        min_ind = np.argmin(point_distance)
        if False:
        # if point_distance[min_ind] > 1e-2:
            print("Warning: point pairing failed!", file=sys.stderr)
            print(point_list, curr_point)
            print(point_distance[min_ind])
            assert(0)
        new_point = point_list.pop(min_ind)
        point_pairs.append((curr_point, new_point))
    return point_pairs


def calculate_total_winding_number(
    char_poly: pt.CLaurent,
    E_ref: complex,
    radius_Cr: float,
    n_theta1: int,
    positive_points: list,
    negative_points: list,
    max_incre: float = pi / 12
):
    ''' Given pairs of positive and negative charges, calculate the total winding number '''
    # 1. Find best winding loop and calculate the winding number
    if positive_points:
        beta2_args = np.array(
            [np.log(p[1]).imag for p in positive_points]
            + [np.log(p[1]).imag for p in negative_points]
        )
        beta2_args_sorted = np.sort(beta2_args)
        beta2_args_diff = np.concatenate([
            beta2_args_sorted[1:] - beta2_args_sorted[:-1],
            [2 * np.pi - beta2_args_sorted[-1] + beta2_args_sorted[0]]
        ])
        max_segment = np.argmax(beta2_args_diff)
        # arg for best loop: max_segment to max_segment + 1
        best_loop_theta2 = (
            beta2_args_sorted[max_segment] + beta2_args_diff[max_segment] / 2
        )
        if best_loop_theta2 > np.pi:
            best_loop_theta2 -= 2 * np.pi
    else:
        best_loop_theta2 = 0

    theta1_loop = np.linspace(-np.pi, np.pi, n_theta1)
    # theta2_loop = best_loop_theta2 * np.ones(n_theta1)
    # loop_winding = calculate_winding_around_loop(
    #     char_poly, E_ref, radius_Cr, theta1_loop, theta2_loop
    # )
    loop_winding = calculate_loop_winding_self_adapted(
        char_poly, E_ref, radius_Cr, theta1_loop, best_loop_theta2, max_incre
    )

    total_winding = 2 * np.pi * loop_winding

    # 2. For each pair, determine the relative position of the winding loop and the non-Bloch solutions
    for charge_ind in range(len(positive_points)):
        curr_positive_arg = np.log(positive_points[charge_ind][1]).imag
        curr_negative_arg = np.log(negative_points[charge_ind][1]).imag
        # 2.1. Whether the winding loop is between the pair or outside the pair
        if (best_loop_theta2 - curr_positive_arg) * (best_loop_theta2 - curr_negative_arg) < 0:
            # 2.2. Which one is the positive charge
            d = abs(curr_negative_arg - curr_positive_arg)
            if curr_negative_arg > curr_positive_arg:
                total_winding += - 2 * np.pi + d
            else:
                total_winding += 2 * np.pi - d
        else:
            total_winding += curr_negative_arg - curr_positive_arg

    return total_winding


if __name__ == '__main__':
    pass