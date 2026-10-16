'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-06-02
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import pickle
from HN_model_common import *

##### DGBZ tools #####
# ``poly_tools`` / ``partial_GBZ_solver`` / ``GBZ_manifold`` are only used by the
# routines that actually run the punctured-GBZ solver (``get_GBZ_candidates``,
# ``plot_GBZ_candidates`` and the [11] winding calculation).  They are imported
# lazily inside those routines so that ``plot_11_DGBZ_spectrum`` - the only
# routine needed for the figure - runs without building the C++ extensions and
# without PHCpy, since the [11]-SGBZ it draws is analytic.
import sys
sys.path.append("../../puncture-calculation")

##### Figure plot #####
import matplotlib.pyplot as plt
plt.style.use("../settings-and-materials/paper_plot.mplstyle")
sys.path.append("..")
from figure_settings_common import *

E1 = -1 + 1j
E2 = 1 + 1j

def plot_11_DGBZ_spectrum():
    fig = plt.figure()
    ax = fig.gca()

    formal_plot = True
    #### axis settings ####
    if(formal_plot):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    #### get DGBZ spectrum profile ####
    bound, loop = get_11_DGBZ_spectrum(*Default_Model_Params)
    ax.plot(loop.real, loop.imag)
    ax.plot(E1.real, E1.imag, '.')
    ax.plot(E2.real, E2.imag, '.')
    
    #### save ####
    if(formal_plot):
        fig.savefig("Figures/DGBZ-spectrum-11-direction.pdf")
    else:
        plt.show()

def get_GBZ_candidates():
    ''' Calculate GBZ candidates for E1 and E2 '''

    import poly_tools as pt
    import partial_GBZ_solver as pGs

    ### get model ChP ###
    f = get_HN_model(*Default_Model_Params)
    f_11 = f.get_supercell(
        [(0,0)],
        np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )
    coeffs, degs = f_11.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))

    ### calculate punctures ###
    solver = pGs.GBZMainSolver(char_poly)
    err_code = solver.get_beta_GBZ(E1)
    if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
        beta_GBZ = solver.beta_GBZ_glued
        GBZ_holes = solver.GBZ_holes
        with open("data/GBZ-candidates-E1.pkl", "wb") as fp:
            pickle.dump((beta_GBZ, GBZ_holes), fp)
    else:
        raise ValueError("E1 fails")
    
    err_code = solver.get_beta_GBZ(E2)
    if(err_code == pGs.GBZSolverErrorCode.SOLVER_SUCCESS):
        beta_GBZ = solver.beta_GBZ_glued
        GBZ_holes = solver.GBZ_holes
        with open("data/GBZ-candidates-E2.pkl", "wb") as fp:
            pickle.dump((beta_GBZ, GBZ_holes), fp)
    else:
        raise ValueError("E2 fails")

def plot_GBZ_candidates():
    ''' Plot GBZ candidates '''
    import GBZ_manifold as Gm

    which = "E1"
    formal_plot = True

    fig = plt.figure(figsize=(4.3*cm, 4*cm))
    ax = fig.gca()
    ax.set_xlabel("Re $\\beta_1$")
    ax.set_ylabel("Im $\\beta_1$")

    ### load GBZ segments ###
    with open("data/GBZ-candidates-%s.pkl"%(which), "rb") as fp:
        beta_GBZ, GBZ_holes = pickle.load(fp)
    
    GBZ_holes = np.asarray(GBZ_holes)
    GBZ_hole_beta = np.sqrt((1 + GBZ_holes) / (1 - GBZ_holes))
    print(GBZ_hole_beta)

    ### plot GBZ candidates ###
    for seg in beta_GBZ:
        beta_coords = np.zeros((len(seg.point_vec), 2), dtype=complex)
        for point_id, point in enumerate(seg.point_vec):
            Gm.CP1_to_chart(point, [0,0])
            beta_coords[point_id, :] = point.coords
        ax.plot(beta_coords[:,0].real, beta_coords[:,0].imag)


    if(which == 'E1'):
        ax.set_xlim([-2,6])

        ### plot shells ###
        curr_beta = GBZ_hole_beta[1]
        theta = np.linspace(0, 2 * pi, 500)
        ax.plot(curr_beta * np.cos(theta), curr_beta * np.sin(theta), '--')
    else:
        ax.set_xlim([-6,2])
        curr_beta = abs(0.873 + 0.426j)
        theta = np.linspace(0, 2 * pi, 500)
        ax.plot(curr_beta * np.cos(theta), curr_beta * np.sin(theta), '--')

    ax.set_ylim([-4,4])
    ax.set_position((0.2,0.2,0.8,0.8*4/4.3*8/8))
    ax.set_aspect(1)
    
    if formal_plot:
        fig.savefig("Figures/GBZ-candidates-%s.pdf"%(which))
    else:
        plt.show()


def calculate_x_total_winding_for_r(E, Jx1, Jx2, Jy1, Jy2, r):
    ''' Calculate all the GBZ candidates for given r '''
    a = abs(Jx2) * r + abs(Jx1) / r
    b = abs(Jx2) * r - abs(Jx1) / r
    theta_x1 = np.log(Jx1).imag
    theta_x2 = np.log(Jx2).imag
    theta_y1 = np.log(Jy1).imag
    theta_y2 = np.log(Jy2).imag
    sqrt_Jy1 = np.sqrt(abs(Jy1)) * np.exp(theta_y1 / 2)
    sqrt_Jy2 = np.sqrt(abs(Jy2)) * np.exp(theta_y2 / 2)
    sqrt_Jx1 = np.sqrt(abs(Jx1)) * np.exp(theta_x1 / 2)
    sqrt_Jx2 = np.sqrt(abs(Jx2)) * np.exp(theta_x2 / 2)
    p0 = E * np.exp(-1j * (theta_x1 + theta_x2) / 2)
    v = (-2 * np.exp(-1j * (theta_x1 + theta_x2) / 2) 
         * sqrt_Jy1 * sqrt_Jy2)

    v_vec = np.array([v.real, v.imag])
    p0_vec = np.array([p0.real, p0.imag])
    M = np.diag([1/(a**2), 1/(b**2)])
    poly_eq = [
        v_vec.dot(M @ v_vec),
        2 * p0_vec.dot(M @ v_vec),
        p0_vec.dot(M @ p0_vec) - 1
    ]
    all_roots = np.roots(poly_eq)

    ### get u = cos(k2) ###
    u_list = []
    for curr_root in all_roots:
        if(abs(curr_root.imag) < 1e-10):
            if(abs(curr_root.real) <= 1):
                u_list.append(curr_root.real)
    
    ### get winding for cos(k2) = 0 or \pm 1 ###
    if(b > 0):
        winding_charge = 1
    else:
        winding_charge = -1
    
    test_point_list = np.array([p0, p0 - v, p0 + v]) # u = 0, 1, -1
    test_point_u = [0, -1, 1]
    test_determinant = test_point_list.real ** 2 / (a ** 2) + test_point_list.imag ** 2 / (b ** 2) 
    max_point = np.argmax(np.abs(test_determinant - 1))
    if(test_determinant[max_point] > 1):
        winding_number = 0
    else:
        winding_number = winding_charge
    
    ### calculate topological charge ###
    def get_topo_charge(u: float):
        exp_phase = np.exp(1j * (
            theta_x1 + theta_x2 - theta_y1 - theta_y2) / 2)
        curr_p = u * v + p0
        dEdphi = exp_phase * (
            -a * (curr_p.imag) / b + 1j * b * (curr_p.real) / a)
        if(dEdphi.imag > 0):
            return -1
        else:
            return 1
    
    ### calculate total winding number
    all_topo_charges = [get_topo_charge(u) for u in u_list]
    total_winding = 2 * np.pi * winding_number
    for u_id in range(len(u_list)):
        curr_u = u_list[u_id]
        cand_dist = np.arccos(curr_u) * 2
        total_winding -= all_topo_charges[u_id] * cand_dist
        if(curr_u <= test_point_u[max_point]):
            total_winding += 2 * np.pi * all_topo_charges[u_id]
    
    return total_winding

def calculate_11_total_winding_for_r(E, Jx1, Jx2, Jy1, Jy2, r):
    ''' Calculate all the GBZ candidates for given r '''
    import poly_tools as pt
    import partial_GBZ_solver as pGs

    a = abs(Jx2 * Jy2) * r + abs(Jx1 * Jy1) / r
    b = abs(Jx2 * Jy2) * r - abs(Jx1 * Jy1) / r
    theta_x1 = np.log(Jx1).imag
    theta_x2 = np.log(Jx2).imag
    theta_y1 = np.log(Jy1).imag
    theta_y2 = np.log(Jy2).imag
    sqrt_Jy1 = np.sqrt(abs(Jy1)) * np.exp(theta_y1 / 2)
    sqrt_Jy2 = np.sqrt(abs(Jy2)) * np.exp(theta_y2 / 2)
    sqrt_Jx1 = np.sqrt(abs(Jx1)) * np.exp(theta_x1 / 2)
    sqrt_Jx2 = np.sqrt(abs(Jx2)) * np.exp(theta_x2 / 2)

    phase_factor = np.exp(-1j * (
        theta_x1 + theta_x2 + theta_y1 + theta_y2) / 2)
    p0 = -(Jx1 * Jx2 + Jy1 * Jy2) * phase_factor
    v = E**2 * phase_factor / 4

    v_vec = np.array([v.real, v.imag])
    p0_vec = np.array([p0.real, p0.imag])
    M = np.diag([1/(a**2), 1/(b**2)])
    poly_eq = [
        v_vec.dot(M @ v_vec),
        2 * p0_vec.dot(M @ v_vec),
        p0_vec.dot(M @ p0_vec) - 1
    ]
    all_roots = np.roots(poly_eq)

    ### get u_inv_sq = 1/cos^2(k2) ###
    u_inv_sq_list = []
    for curr_root in all_roots:
        if(abs(curr_root.imag) < 1e-10):
            if(curr_root.real >= 1):
                u_inv_sq_list.append(curr_root.real)
    u_list = []
    GBZ_candidates = []
    for u_inv_sq in u_inv_sq_list:
        ellp_point = p0 + v * u_inv_sq
        beta1 = (ellp_point.real / a * r
                + 1j * ellp_point.imag / b * r)
        beta1 /= np.exp(1j * (
            theta_x2 + theta_y2 - theta_x1 - theta_y1
        ) / 2)
        for sigma in [1, -1]:
            u = sigma/sqrt(u_inv_sq)
            E_theo = 2 * u * np.sqrt(Jx2 * beta1 + Jy1) * np.sqrt(Jx1 / beta1 + Jy2)
            if(abs(E_theo - E) < 1e-10):
                u_list.append(u.real)
                exp_ik2 = u + 1j * np.sqrt(1-u**2)
                beta2_p = exp_ik2 * np.sqrt(Jx2 * beta1 + Jy1) / np.sqrt(Jx1 / beta1 + Jy2)
                beta2_m = (exp_ik2.conjugate()) * np.sqrt(Jx2 * beta1 + Jy1) / np.sqrt(Jx1 / beta1 + Jy2)
                GBZ_candidates.append((beta1, beta2_p, beta2_m))
    
    ### calculate topological charges ###
    model = get_HN_model(*Default_Model_Params)
    model_11 = model.get_supercell(
        [(0,0)],
        np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )
    coeffs, degs = model_11.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    topo_charge_list = np.zeros_like(u_list)
    char_poly_info = pGs.CharPolyInfo(char_poly)
    for u_id in range(len(u_list)):
        detE, dbeta = char_poly_info.get_det_and_dbeta(
            E, *(GBZ_candidates[u_id])
        )
        if(detE > 0):
            topo_charge_list[u_id] = 1
        else:
            topo_charge_list[u_id] = -1
    
    ### winding for cos(k2) = 0 is 0###
   
    ### calculate total winding number
    total_winding = 0
    for u_id in range(len(u_list)):
        curr_u = u_list[u_id]
        cand_dist = np.arccos(curr_u) * 2
        total_winding -= topo_charge_list[u_id] * cand_dist
        if(curr_u < 0):
            total_winding += 2 * np.pi * topo_charge_list[u_id]
    
    return total_winding



def plot_total_winding_x():
    Jx1, Jx2, Jy1, Jy2 = Default_Model_Params
    print(np.sqrt(abs(Jx1/Jx2)))
    r = np.linspace(0.1, 3, 200)
    w = np.array([calculate_x_total_winding_for_r(
        E1, Jx1, Jx2, Jy1, Jy2, curr_r
    ) for curr_r in r])
    plt.plot(r, w/(2*np.pi), '.-')
    plt.show()

def plot_total_winding_11():
    which_one = "E2"
    if(which_one == "E1"):
        curr_E = E1
    else:
        curr_E = E2

    Jx1, Jx2, Jy1, Jy2 = Default_Model_Params
    print(np.sqrt(abs(Jx1 * Jy1 / Jx2 / Jy2)))
    r = np.linspace(0.1, 3, 300)
    w = np.array([calculate_11_total_winding_for_r(
        curr_E, Jx1, Jx2, Jy1, Jy2, curr_r
    ) for curr_r in r])


    fig = plt.figure(figsize=(4.3*cm, 4*cm))
    ax = fig.gca()
    ax.set_position((0.2,0.2,0.8,0.8*4/4.3*8/8))
    ax.set_xlabel("$r$")
    ax.set_ylabel("$W_{\\mathrm{bulk}}/\\rho_0$")
    ax.plot(r, w/(2*np.pi), '-')
    ax.plot(r, 0*r, '--')
    ax.set_xlim([0, max(r)])

    fig.savefig("Figures/total-winding-11-direction-%s.pdf"%(which_one))
    

if __name__ == '__main__':
    # Figure 3: [11]-SGBZ spectrum.  Analytic; needs only NumPy/SciPy/Matplotlib
    # (plus BerryPy, through HN_model_common).
    plot_11_DGBZ_spectrum()

    # The routines below run the punctured-GBZ solver, so they additionally need the
    # compiled ``poly_tools`` / ``GBZ_manifold`` extensions and PHCpy - see README.
    # get_GBZ_candidates()      # -> data/GBZ-candidates-{E1,E2}.pkl
    # plot_GBZ_candidates()
    # plot_total_winding_11()
    # plot_total_winding_x()    # pure NumPy, kept for reference
