'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-08-21 10:52:37
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import HN_model_common as HN
import numpy as np
import sys
sys.path.append("../../ChP-winding-algorithm")
import poly_tools as pt
import matplotlib.pyplot as plt
CM = 1/2.54

PARAMS = [(1.4757-0.0246j), (0.1585-1.0847j), 0.5009, -1.3837, -0.2759, 0.2602]
J_X, J_Y, GAMMA_X, GAMMA_Y, DELTA_X, DELTA_Y = PARAMS
E = abs(J_X) * np.exp(1j * DELTA_X)

def q1D_zeros(model, ny):
    model_1D = model.get_supercell(
        [(0, j) for j in range(ny)],
        np.array([
            [1, 0],
            [0, ny]
        ], dtype=int)
    )
    model_1D = model_1D.truncate(1)

    coeffs, degs = model_1D.get_characteristic_polynomial_data()
    chp = pt.CLaurent(2)
    chp.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    chp_1d = chp.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([1])
    )
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    chp_1d.num.batch_get_data(coeffs, degs)

    # numpy array
    max_deg = max(degs)
    eq_np = np.zeros(max_deg + 1, dtype=complex)
    for term_idx in range(len(degs)):
        eq_np[max_deg - degs[term_idx]] = coeffs[term_idx]

    # solve
    return np.roots(eq_np)


def main_q1D_zero_GDSE():
    Jx1 = np.exp(GAMMA_X + 1j * DELTA_X) * J_X
    Jx2 = np.exp(-GAMMA_X + 1j * DELTA_X) * J_X.conjugate()
    Jy1 = np.exp(GAMMA_Y + 1j * DELTA_Y) * J_Y
    Jy2 = np.exp(-GAMMA_Y + 1j * DELTA_Y) * J_Y.conjugate()
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)

    ny = 20
    zeros = q1D_zeros(model, ny)
    log_beta = np.log(zeros)

    plt.plot(log_beta.real, log_beta.imag, '.')
    plt.plot([GAMMA_X, GAMMA_X], [min(log_beta.imag), max(log_beta.imag)])
    plt.show()


def main_q1D_zero_uniform():
    plt.style.use('../settings-and-materials/paper_plot.mplstyle')
    Jx1 = np.exp(GAMMA_X + 1j * DELTA_X) * J_X
    Jx2 = np.exp(-GAMMA_X + 1j * DELTA_X) * J_X.conjugate()
    Jy1 = np.exp(GAMMA_Y + 1j * DELTA_X) * J_Y
    Jy2 = np.exp(-GAMMA_Y + 1j * DELTA_X) * J_Y.conjugate()
    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)

    ny = 20
    zeros = q1D_zeros(model, ny)
    log_beta = np.log(zeros)

    fig = plt.figure(
        figsize=(2/0.8 * CM, 2/0.8 * CM)
    )
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    plt.plot(log_beta.real, log_beta.imag, '.')
    plt.plot([GAMMA_X, GAMMA_X], [min(log_beta.imag), max(log_beta.imag)])
    ax.set_xlabel(r"ln$|\beta_1|$")
    ax.set_ylabel(r"Arg$(\beta_1)$")
    ax.set_ylim([-np.pi - 1e-3, np.pi + 1e-3])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    fig.savefig("Figures/q1D-beta.pdf")


if __name__ == '__main__':
    # main_q1D_zero_GDSE()
    main_q1D_zero_uniform()
