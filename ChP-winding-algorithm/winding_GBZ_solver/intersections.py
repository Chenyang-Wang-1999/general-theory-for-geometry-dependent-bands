'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-12-30
Copyright © Department of Physics, Tsinghua University. All rights reserved

Calculate intersections of non-Bloch solutions and Cr
'''

import numpy as np
import poly_tools as pt
from scipy import optimize


def solve_minor_beta(
    char_poly: pt.CLaurent,
    E_ref: complex,
    beta1: complex,
    offset_list: list[int]
):
    ''' Solve the 1D equation with respect to beta2 for given E_ref and beta1
        Args:
            char_poly: Characteristic polynomial of the non-Bloch Hamiltonian
            E_ref: Reference energy
            beta1: beta1
            offset_list: List of offsets of the beta2 solutions, defined as
                solution_ind = M + offset

        Return:
            beta2_list: List of beta2 solutions
    '''
    # 1. Get equation
    poly_1d = char_poly.partial_eval(
        pt.CScalarVec([E_ref, beta1]),
        pt.CIndexVec([0, 1]),
        pt.CIndexVec([2])
    )
    max_denom = poly_1d.denom_orders[0]
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    poly_1d.num.batch_get_data(coeffs, degs)

    # 2. Convert to numpy polynomial format
    max_deg = max(degs)
    np_coeffs = np.zeros(max_deg + 1, dtype=complex)
    for ind in range(len(coeffs)):
        np_coeffs[max_deg - degs[ind]] = coeffs[ind]

    # 3. Solve
    all_beta2 = np.roots(np_coeffs)
    all_beta2 = sorted(all_beta2, key=lambda x: np.abs(x))

    # 4. Pick solutions
    beta2_list = np.zeros(len(offset_list), dtype=complex)
    for ind, offset in enumerate(offset_list):
        beta2_list[ind] = all_beta2[max_denom - 1 + offset]

    return beta2_list


def get_beta2_derivative(
    df_dbeta1: pt.CLaurent,
    df_dbeta2: pt.CLaurent,
    E_ref: complex,
    beta1: complex,
    beta2: complex
):
    ''' Get derivative of beta2 against beta1 '''
    return (
        - df_dbeta1.eval(pt.CScalarVec([E_ref, beta1, beta2]))
        / df_dbeta2.eval(pt.CScalarVec([E_ref, beta1, beta2]))
    )


def get_critical_segments(
    GBZ_loops: list[np.ndarray[2, complex]],
    radius_Cr: float
):
    ''' Given a list of GBZ loops, find the segments that intersect with Cr
        Args:
            GBZ_loops: List of GBZ loops, each loop must be a numpy array of shape (N, 2)
            radius_Cr: Radius of Cr
        Return:
            critical_seg: List of critical segments
    '''
    critical_seg = []
    for loop in GBZ_loops:
        beta1_abs = np.abs(loop[:,0])
        selected = (
            (beta1_abs[1:] - radius_Cr) * (beta1_abs[:-1] - radius_Cr) <= 0
        )
        selected_ind = np.arange(len(beta1_abs) - 1)[selected]
        for curr_ind in selected_ind:
            critical_seg.append((
                loop[curr_ind, :],
                loop[curr_ind + 1, :]
            ))

    return critical_seg


def calculate_beta2_radius_diff(
    radius_Cr: float,
    theta1: float,
    char_poly: pt.CLaurent,
    dchar_poly: list[pt.CLaurent],
    E_ref: complex
):
    ''' Calculate the difference between |beta2_{M+1}|^2 - |beta2_{M}|^2
        Args:
            radius_Cr: Radius of Cr
            theta1: Angle of beta1
            char_poly: Characteristic polynomial of the non-Bloch Hamiltonian
            dchar_poly: List of derivatives of the characteristic polynomial
                with respect to E_ref, beta1 and beta2
            E_ref: Reference energy
        Return:
            diff_val: Difference between |beta2_{M+1}|^2 - |beta2_{M}|^2
            diff_der: Derivative of diff_val with respect to theta1
    '''

    beta1 = radius_Cr * np.exp(1j * theta1)
    beta2_sols = solve_minor_beta(
        char_poly,
        E_ref,
        beta1,
        [0, 1]
    )
    beta2_der = [
        get_beta2_derivative(
            dchar_poly[1],
            dchar_poly[2],
            E_ref, beta1, beta2
        )
        for beta2 in beta2_sols
    ]

    # evaluate difference between |beta2_{M+1}|^2 - |beta2_{M}|^2
    diff_val = (
        (beta2_sols[1] * beta2_sols[1].conjugate()).real
        - (beta2_sols[0] * beta2_sols[0].conjugate()).real
    )

    diff_der = 2 * (
        (1j * beta1 * beta2_sols[1].conjugate() * beta2_der[1]).real
        - (1j * beta1 * beta2_sols[0].conjugate() * beta2_der[0]).real
    )

    return diff_val, diff_der


def bisect_critical_segments(
    char_poly: pt.CLaurent,
    dchar_poly: list[pt.CLaurent],
    seg_start: np.ndarray[1, complex],
    seg_end: np.ndarray[1, complex],
    E_ref: complex,
    radius_Cr: float
):
    ''' Bisect the critical segments to find the intersection point. 
        Args:
            char_poly: Characteristic polynomial of the non-Bloch Hamiltonian
            dchar_poly: List of derivatives of the characteristic polynomial
                with respect to E_ref, beta1 and beta2
            seg_start: Start point of the segment
            seg_end: End point of the segment
            E_ref: Reference energy
            radius_Cr: Radius of Cr
        Return:
    '''
    pass


def interp_to_radius_r(
    seg_start: np.ndarray[1, complex],
    seg_end: np.ndarray[1, complex],
    radius_Cr: float
):
    ''' Get the intersection point of the segment with Cr by direct interpolation '''
    def abs_diff(t):
        curr_point = seg_start[0] * (1 - t) + seg_end[0] * t
        return abs(curr_point) - radius_Cr
    t = optimize.bisect(abs_diff, 0, 1)
    return seg_start * (1 - t) + seg_end * t
