'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-02-20
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import poly_tools as pt
import numpy as np
import partial_GBZ_solver as pGs
import GBZ_manifold as Gm
from scipy import optimize

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

char_poly_info = pGs.CharPolyInfo(f)

solver = pGs.GBZMainSolver(f)

def test_puncture_derivative():
    E = np.random.randn() + 1j * np.random.randn()
    
    err_code = solver.get_beta_GBZ(E)
    beta_GBZ = solver.beta_GBZ

    selected_loop = np.random.randint(len(beta_GBZ))
    selected_seg = np.random.randint(len(beta_GBZ[selected_loop]))
    selected_point = np.random.randint(len(beta_GBZ[selected_loop][selected_seg].point_vec) - 1)
    curr_point = beta_GBZ[selected_loop][selected_seg].point_vec[selected_point]
    curr_phi = beta_GBZ[selected_loop][selected_seg].param_list[selected_point][0] * (-2)

    next_point = beta_GBZ[selected_loop][selected_seg].point_vec[selected_point + 1]
    next_phi = beta_GBZ[selected_loop][selected_seg].param_list[selected_point + 1][0] * (-2)

    Gm.CP1_to_chart(curr_point, [0,0])
    Gm.CP1_to_chart(next_point, [0,0])

    curr_beta_minor_m = curr_point.coords[1]
    curr_beta_minor_p = curr_beta_minor_m * np.exp(1j * curr_phi)
    curr_beta_major = curr_point.coords[0]

    dE, dbeta = char_poly_info.get_E_and_beta_derivative(E, curr_beta_major, curr_beta_minor_p, curr_beta_minor_m)

    # check dbeta
    next_beta_major = next_point.coords[0]
    dbeta_num = next_beta_major - curr_beta_major
    dphi = next_phi - curr_phi

    # print(dbeta_num)
    # print(dbeta * dphi)
    # print(abs(dbeta * dphi - dbeta_num))

    # test beta norm
    dabs_beta_num = abs(next_beta_major) - abs(curr_beta_major)
    dabs_beta = (curr_beta_major.conjugate() * dbeta).real / abs(curr_beta_major)
    print(dabs_beta_num)
    print(dabs_beta * dphi)
    print(abs(dabs_beta * dphi - dabs_beta_num))

    
def test_E_derivative():
    beta1 = np.random.randn() + 1j * np.random.randn()
    dtheta = np.random.randn() * 1e-3
    beta2 = beta1 * np.exp(1j * dtheta)

    # calculate GBZ
    err_code = solver.get_partial_GBZ(beta1, False)
    partial_GBZ = solver.partial_GBZ

    selected_loop = np.random.randint(len(partial_GBZ))
    # selected_seg = np.random.randint(len(partial_GBZ[selected_loop]))
    selected_point = np.random.randint(len(partial_GBZ[selected_loop].point_vec) - 1)
    curr_point = partial_GBZ[selected_loop].point_vec[selected_point]
    curr_phi = partial_GBZ[selected_loop].param_list[selected_point][0] * (-2)
    next_point = partial_GBZ[selected_loop].point_vec[selected_point + 1]
    next_phi = partial_GBZ[selected_loop].param_list[selected_point + 1][0] * (-2)
    next_E = next_point.coords[0]

    Gm.GBZ_to_chart(curr_point, [0])

    beta_minor_m = curr_point.coords[1]
    beta_minor_p = beta_minor_m * np.exp(1j * curr_phi)
    E = curr_point.coords[0]

    dE, dbeta1 = char_poly_info.get_E_and_beta_derivative(E, beta1, beta_minor_p, beta_minor_m)
    
    dE_num = next_E - E
    dphi = next_phi - curr_phi
    print(dE_num)
    print(dphi * dE[0])
    print(abs(dE_num - dphi * dE[0]))

    # derivative against beta1
    f_beta2 = f.partial_eval(
        pt.CVarScalarVec([beta2]), pt.CIndexVec([1]), pt.CIndexVec([0,2])
    )
    new_char_poly_eq = pGs.aGBZEquation1D(f_beta2)
    new_char_poly_eq.set_params([curr_phi,])
    res = pGs.complex_root(new_char_poly_eq, np.asarray(curr_point.coords), jac=True)
    new_point = res.x

    print(new_point[0] - E)
    print(dE[1] * dtheta)
    print(abs(new_point[0] - E - (dE[1] * dtheta)))

def test_winding():
    beta1 = np.random.randn() + 1j * np.random.randn()

    # calculate GBZ
    err_code = solver.get_partial_GBZ(beta1, False)
    partial_GBZ = solver.partial_GBZ

    selected_loop = np.random.randint(len(partial_GBZ))
    # selected_seg = np.random.randint(len(partial_GBZ[selected_loop]))
    selected_point = np.random.randint(len(partial_GBZ[selected_loop].point_vec))
    curr_point = partial_GBZ[selected_loop].point_vec[selected_point]
    curr_phi = partial_GBZ[selected_loop].param_list[selected_point][0] * (-2)

    Gm.GBZ_to_chart(curr_point, [0])

    beta_minor_m = curr_point.coords[1]
    beta_minor_p = beta_minor_m * np.exp(1j * curr_phi)
    E = curr_point.coords[0]

    dE_det, dbeta1 = char_poly_info.get_det_and_dbeta(E, beta1, beta_minor_p, beta_minor_m)
    print(dE_det)

    dE_det, dbeta1 = char_poly_info.get_det_and_dbeta(E, beta1, beta_minor_m, beta_minor_p)
    print(dE_det)

    # wind around given point
    dphi_loop = np.cos(np.linspace(0, 2 * np.pi, 100)) * 1e-3
    dtheta_loop = np.sin(np.linspace(0, 2 * np.pi, 100)) * 1e-3
    E_loop = np.zeros_like(dphi_loop, dtype=complex)

    for j in range(len(dphi_loop)):
        curr_dphi = dphi_loop[j]
        curr_dtheta = dtheta_loop[j]
        f_beta2 = f.partial_eval(pt.CVarScalarVec([beta1 * np.exp(1j * curr_dtheta)]), 
                                 pt.CIndexVec([1]), 
                                 pt.CIndexVec([0,2]))
        new_char_poly_eq = pGs.aGBZEquation1D(f_beta2)
        new_char_poly_eq.set_params([curr_phi + curr_dphi])
        res = pGs.complex_root(new_char_poly_eq, np.asarray(curr_point.coords), jac=True)
        E_loop[j] = res.x[0]
    
    print(pGs.get_winding_number(E - E_loop))
    print(pGs.get_winding_number(E_loop - E))
        

if __name__ == '__main__':
    test_puncture_derivative()
    # test_E_derivative()
    # test_winding()