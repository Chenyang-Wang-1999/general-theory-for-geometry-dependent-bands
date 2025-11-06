'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-03-21
Copyright © Department of Physics, Tsinghua University. All rights reserved

Investigate the HN model without the GDSE.
'''

import numpy as np
from numpy import pi
import pyvista as pv
import poly_tools as pt
import matplotlib.pyplot as plt
from palette_manager import get_palette
import sys
import pickle
sys.path.append("../paper/HN-model")
import HN_model_common as HN

COLOR = get_palette("matplotlib")

def get_ChP(gammax, gammay, Jx, Jy, alpha):
    coeffs = pt.CScalarVec(
        [1, -gammax * Jx, 
         -Jx.conjugate() / gammax, 
         -gammay * Jy, 
         -Jy.conjugate() / gammay]
    )
    degs = pt.CLaurentIndexVec([
        # E, beta1, beta2
        1, 0, 0,
        0, -1, alpha,
        0, 1, -alpha,
        0, 0, -1,
        0, 0, 1
    ])
    chp = pt.CLaurent(3)
    chp.set_Laurent_by_terms(coeffs, degs)
    return chp


def get_ChP_eq(chp: pt.CLaurent, E, beta1):
    chp_1d = chp.partial_eval(
        pt.CScalarVec([E, beta1]),
        pt.CIndexVec([0, 1]),
        pt.CIndexVec([2])
    )
    M = chp_1d.denom_orders[0]
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    chp_1d.num.batch_get_data(coeffs, degs)

    # Get numpy array
    max_deg = max(degs)
    eq_np = np.zeros(max_deg + 1, dtype=complex)
    for i in range(len(coeffs)):
        eq_np[max_deg - degs[i]] = coeffs[i]
    return eq_np, M


def plot_mGBZ():
    Jx = 1
    Jy = 0.5 + 0.2j
    alpha = 2
    gammax = 0.8
    gammay = 0.7

    r = gammax * (gammay ** alpha) * 1.0

    chp = get_ChP(gammax, gammay, Jx, Jy, alpha)
    E = 0.2
    theta1 = np.linspace(-pi, pi, 1000)
    beta1 = r * np.exp(1j * theta1)
    beta2 = np.zeros((2, len(beta1)), dtype=complex)
    picked_roots = []
    for idx, curr_beta1 in enumerate(beta1):
        eq_np, M = get_ChP_eq(chp, E, curr_beta1)
        all_roots = np.roots(eq_np)
        all_roots = sorted(all_roots, key=lambda x: abs(x))
        beta2[0, idx] = all_roots[M - 1]
        beta2[1, idx] = all_roots[M]
        picked_roots += [(curr_beta1, beta2) for beta2 in all_roots if abs(abs(beta2) - gammay) / gammay < 0.4]
    # Plot
    plt.figure()
    plt.plot(theta1, np.abs(beta2[0,:]))
    plt.plot(theta1, np.abs(beta2[1,:]))
    plt.plot([-pi, pi], [gammay, gammay], '--')
    plt.xlim([-pi, pi])

    picked_roots = np.asarray(picked_roots)
    plt.figure()
    plt.plot(np.log(picked_roots[:, 1]).imag, np.log(picked_roots[:, 0]).imag, '.')
    plt.xlim([-pi, pi])
    plt.ylim([-pi, pi])
    plt.show()

    pl = pv.Plotter()
    pl.add_points(
        np.column_stack([
            picked_roots[:, 1].real,
            picked_roots[:, 1].imag,
            np.log(picked_roots[:, 0]).imag
        ]),
        color=COLOR['ol']
    )
    cld = pv.Cylinder(radius=gammay, height=2*pi, direction=(0, 0, 1))
    pl.add_mesh(cld, color=COLOR['bf'], opacity=0.5)
    pl.show()


def get_zeros_for_E(
    E,
    Jx, Jy, deltax
):
    t = (E * np.exp(-1j * deltax) / 2).real
    if (abs((E * np.exp(-1j * deltax) / 2).imag) > 1e-10):
        print((E * np.exp(-1j * deltax) / 2).imag)
        assert(0)
    Jx_norm = abs(Jx)
    Jy_norm = abs(Jy)
    Jx_arg = np.log(Jx).imag
    Jy_arg = np.log(Jy).imag

    # Get range for ux = cos(theta_x) and uy = cos(theta_y)
    # Jx_norm * ux + Jy_norm * uy = t, -1 <= ux <= 1, -1 <= uy <= 1

    min_ux = max(-1 + 1e-6, (t - Jy_norm) / Jx_norm + 1e-6)
    max_ux = min(1 - 1e-6, (t + Jy_norm) / Jx_norm - 1e-6) 
    ux = np.linspace(min_ux, max_ux, 1000)
    uy = (t - Jx_norm * ux) / Jy_norm
    Deltax = np.arccos(ux)
    Deltay = np.arccos(uy)
    return [
        np.column_stack([Jx_arg + Deltax, Jy_arg + Deltay]),
        np.column_stack([Jx_arg - Deltax, Jy_arg + Deltay]),
        np.column_stack([Jx_arg + Deltax, Jy_arg - Deltay]),
        np.column_stack([Jx_arg - Deltax, Jy_arg - Deltay]),
    ]


def concatenate_lines(lines: list[np.ndarray]):
    ''' Concatenate the lines (line[j].shape = (n, 2)) in the list by the following rules:
        1. Match the tail of each array with the head or tail of other arrays, including the head of the array itself,
        2. Find the minimum distance
        3. If the minimum distance is between the head and tail of the same array, save it directly to the result list
        4. Else, concatenate the arrays with the minimum distance:
            - head to head: reverse the second array and concatenate
            - tail to head: concatenate directly
    '''
    if not lines:
        return []
    
    # Convert to list for easier manipulation
    lines_list = [line.copy() for line in lines]
    result = []
    
    while lines_list:
        if len(lines_list) == 1:
            # Only one line left, add to result
            result.append(lines_list.pop())
            continue
            
        # Find all possible connections between lines
        min_distance = float('inf')
        best_i = -1
        best_j = -1
        best_connection_type = None  # 'tail_head', 'tail_tail', 'head_head'
        
        for i in range(len(lines_list)):
            for j in range(len(lines_list)):
                if i == j:
                    # Check connection between head and tail of same line
                    head_i = lines_list[i][0]
                    tail_i = lines_list[i][-1]
                    distance = np.linalg.norm(np.exp(1j * tail_i) - np.exp(1j * head_i))
                    if distance < min_distance:
                        min_distance = distance
                        best_i = i
                        best_j = j
                        best_connection_type = 'self'
                else:
                    # Check different connections between lines i and j
                    head_i = lines_list[i][0]
                    tail_i = lines_list[i][-1]
                    head_j = lines_list[j][0]
                    tail_j = lines_list[j][-1]
                    
                    # Tail to head connection
                    distance_th = np.linalg.norm(np.exp(1j * tail_i) - np.exp(1j * head_j))
                    if distance_th < min_distance:
                        min_distance = distance_th
                        best_i = i
                        best_j = j
                        best_connection_type = 'tail_head'
                    
                    # Tail to tail connection (need to reverse j)
                    distance_tt = np.linalg.norm(np.exp(1j * tail_i) - np.exp(1j * tail_j))
                    if distance_tt < min_distance:
                        min_distance = distance_tt
                        best_i = i
                        best_j = j
                        best_connection_type = 'tail_tail'
                    
                    # Head to head connection (need to reverse i)
                    distance_hh = np.linalg.norm(np.exp(1j * head_i) - np.exp(1j * head_j))
                    if distance_hh < min_distance:
                        min_distance = distance_hh
                        best_i = i
                        best_j = j
                        best_connection_type = 'head_head'
        
        # Handle the best connection
        if best_connection_type == 'self':
            # Save the line as is
            result.append(lines_list.pop(best_i))
        elif best_connection_type == 'tail_head':
            # Concatenate line i and line j (tail to head)
            concatenated = np.vstack([lines_list[best_i], lines_list[best_j]])
            # Remove both lines and add the concatenated one
            line_i = lines_list.pop(max(best_i, best_j))
            line_j = lines_list.pop(min(best_i, best_j))
            lines_list.append(concatenated)
        elif best_connection_type == 'tail_tail':
            # Reverse line j and concatenate (tail to tail)
            reversed_j = lines_list[best_j][::-1]
            concatenated = np.vstack([lines_list[best_i], reversed_j])
            # Remove both lines and add the concatenated one
            line_i = lines_list.pop(max(best_i, best_j))
            line_j = lines_list.pop(min(best_i, best_j))
            lines_list.append(concatenated)
        elif best_connection_type == 'head_head':
            # Reverse line i and concatenate (head to head)
            reversed_i = lines_list[best_i][::-1]
            concatenated = np.vstack([reversed_i, lines_list[best_j]])
            # Remove both lines and add the concatenated one
            line_i = lines_list.pop(max(best_i, best_j))
            line_j = lines_list.pop(min(best_i, best_j))
            lines_list.append(concatenated)
    
    return result


def theoretical_winding_step(Jx, Jy, deltax, alpha, E):
    zeros = get_zeros_for_E(E, Jx, Jy, deltax)
    zeros = concatenate_lines(zeros)
    width = 0
    for zero in zeros:
        projected_zero = zero[:, 0] + alpha * zero[:, 1]
        diff_zero = np.log(np.exp(1j * projected_zero[1:]) / np.exp(1j * projected_zero[:-1])).imag
        # Find positive ones
        positive_ones = diff_zero[diff_zero >= 0]
        width += np.sum(positive_ones)
    return width


def main_plot_zeros_for_E():
    # Jx = 1
    # Jy = 0.5 + 0.2j
    # deltax = 0
    Jx, Jy = [(-0.59477-1.09826j), (1.18704-0.07472j)]
    gamma_x, gamma_y = [1.97069, 0.52332]
    deltax = 1.79515
    kx = 0.8 * pi
    ky = 0.3 * pi

    Jx1, Jx2, Jy1, Jy2 = HN.get_coupling_terms_from_new_param(Jx, Jy, gamma_x, gamma_y, deltax, deltax)
    beta_x = gamma_x * np.exp(1j * kx)
    beta_y = gamma_y * np.exp(1j * ky)
    E_ref = Jx1 / beta_x + Jx2 * beta_x + Jy1 / beta_y + Jy2 * beta_y
    zeros = get_zeros_for_E(E_ref, Jx, Jy, deltax)
    zeros = concatenate_lines(zeros)
    alpha = 1

    for zero in zeros:
        projected_zero = zero[:, 0] + alpha * zero[:, 1]
        diff_zero = np.log(np.exp(1j * projected_zero[1:]) / np.exp(1j * projected_zero[:-1])).imag
        # Find positive ones
        positive_ones = diff_zero[diff_zero >= 0]
        width = np.sum(positive_ones)
        print(width / 2 / pi)

    plt.figure()
    for zero in zeros:
        plt.plot((zero[:, 0] + alpha * zero[:, 1]) % (2 * pi), zero[:, 1] % (2 * pi))
    plt.xlim([0, 2 * pi])
    plt.ylim([0, 2 * pi])
    plt.show()


def compare_numerical():
    alpha = 3
    with open(f"data/HN-alpha-{alpha}.pkl", "rb") as f:
        data = pickle.load(f)

    # 解包数据
    (Jx, Jy, gamma_x, gamma_y, delta), (kx, ky), result = data
    _, _, w_left_list, w_right_list = zip(*result)

    # 重塑数据为网格格式
    w_left_list = np.array(w_left_list).reshape(len(kx), len(ky))
    w_right_list = np.array(w_right_list).reshape(len(kx), len(ky))
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)

    print(np.mean(np.abs(w_left_list + w_right_list)))

    # for row_ind in range(kx_mesh.shape[0]):
    #     for col_ind in range(kx_mesh.shape[1]):
    #         kx = kx_mesh[row_ind, col_ind]
    #         ky = ky_mesh[row_ind, col_ind]
    #         E = 2 * np.exp(1j * delta) * (
    #             (Jx.conjugate() * np.exp(1j * kx)).real + 
    #             (Jy.conjugate() * np.exp(1j * ky)).real)
    #         w = theoretical_winding_step(Jx, Jy, delta, alpha, E)
    #         # print("For E = ", E, "kx, ky = ", kx / pi, ky / pi)
    #         # print(w / 2 / pi, w_left_list[row_ind, col_ind] / 2 / pi, w_right_list[row_ind, col_ind] / 2 / pi)
    #         # print("ratio", w / w_right_list[row_ind, col_ind])


if __name__ == '__main__':
    # plot_mGBZ() 
    # main_plot_zeros_for_E()
    compare_numerical()
