'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-09-19
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import c_solver_2D as csolver
import numpy as np
import numpy_extended as npe
import phcpy
from c_solver_2D import complex_root
from scipy import io as sio
import pickle
import BerryPy.TightBinding as tb
from scipy import linalg as la
import matplotlib.pyplot as plt
import pyvista as pv

def minimal_winding_char_poly():
    coeffs = csolver.CScalarVec([
        1, -1, -1, -1, -1
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        1, 0, 0, 
        0, 1, 0,
        0, -1, 0,
        0, 1, -1,
        0, -2, 1
    ])
    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)
    return char_poly

def minimal_winding_model()->tb.TightBindingModel:
    intra_cell = []
    inter_cell = [
        [0, 0, 1, (1,0)],
        [0, 0, 1, (-1,0)],
        # [0, 0, 2, (0,1)],
        # [0, 0, 2, (0,-1)],
        [0, 0, 1, (-1, 1)],
        [0, 0, 1, (2, -1)]
        # [0, 0, 1, (1, -1)]
        # [0, 0, 0.5, (1, -1)],
        # [0, 0, 0.5, (-2, 1)]
    ]

    return tb.TightBindingModel(2, 1,[[1,0], [0,1]], intra_cell, inter_cell)

def main_get_aGBZ():
    char_poly = minimal_winding_char_poly()
    major_dim = 1
    critical_distance = 0.1
    N_loop_points = 200
    N_detour_points = 50
    theta_detour = 1e-3 * np.pi
    zero_val_th = 1e-3

    N_params = 50
    N_phi2 = 50
    phi2_list = np.linspace(np.pi/(N_phi2+1), np.pi - np.pi/(N_phi2+1), N_phi2)
    phi2 = phi2_list[36]
    params = np.zeros((N_params, 2), dtype=complex)
    if(major_dim == 1):
        params[:,1] = np.exp(1j*np.linspace(np.pi/(N_params+1), np.pi - np.pi/(N_params+1), N_params))
        params[:,0] = np.exp(1j*phi2)
    else:
        params[:,0] = np.exp(1j*np.linspace(np.pi/(N_params+1), np.pi - np.pi/(N_params+1), N_params))
        params[:,1] = np.exp(1j*phi2)

    solver = csolver.aGBZ2DSolver(major_dim, critical_distance, N_loop_points, N_detour_points, theta_detour, zero_val_th)
    solver.set_char_poly(char_poly)
    solver.set_param(params)

    solver_flag = True
    full_solver_flag_list = []
    while(solver_flag):
        if(solver.full_solver_flag):
            print("full solver")
            # full solver
            full_solver_next_flag = True
            while(full_solver_next_flag):
                all_sols = phcpy.solver.solve(list(solver.curr_poly_string_list), dictionary_output=True, verbose=False)
                sols_complex = []
                for sol in all_sols:
                    if(abs(sol['y'])<1.5):
                        if(abs(sol['z'])<1.5):
                            if(abs(sol['w'])<1.5):
                                res = complex_root(solver.continuation_solver_fun, np.array([sol['x'], sol['y'], sol['z'], sol['w']]), jac=True)
                                sols_complex.append(res.x)
                                if(not res.success):
                                    print(res.message)
                                    input()
                if(len(sols_complex) == 0):
                    full_solver_next_flag = solver.insert_full_solver_results(np.asarray([[0,0,0,0]], dtype=complex), True)
                else:
                    full_solver_next_flag = solver.insert_full_solver_results(np.asarray(sols_complex), False)
        else:
            # continuation solver
            print("continuation solver")
            continuation_solver_next_flag = True
            while(continuation_solver_next_flag):
                res = complex_root(solver.continuation_solver_fun, np.array(solver.curr_initial_values), jac=True)
                if(not res.success):
                    solver.suspend_solver()
                
                continuation_solver_next_flag = solver.insert_continuation_solver_results(res.x.reshape((1,-1)))

        print(solver.get_curr_param())
        full_solver_flag_list.append(solver.full_solver_flag)
        solver_flag = solver.next()
    
    GBZ_points = []
    GBZ_params = []
    non_GBZ_points = []
    non_GBZ_params = []
    solver_flag = solver.sol_to_segments_init()
    N_calls = 0
    while(solver_flag):
        GBZ_condition_check_result = csolver.check_GBZ_condition(
            char_poly, solver.GBZ_check_curr_point[:3], params[solver.GBZ_check_curr_ptr,:], solver.major_dim + 1,
            theta_detour, N_loop_points, N_detour_points, theta_detour, 
            N_loop_points, N_detour_points, zero_val_th
        )
        # print("current point:", solver.GBZ_check_curr_point)
        # print("current ptr:", solver.GBZ_check_curr_ptr)
        print(N_calls, "finished, ", GBZ_condition_check_result )
        if(GBZ_condition_check_result):
            GBZ_points.append(solver.GBZ_check_curr_point)
            GBZ_params.append(params[solver.GBZ_check_curr_ptr,:])
        else:
            non_GBZ_points.append(solver.GBZ_check_curr_point)
            non_GBZ_params.append(params[solver.GBZ_check_curr_ptr,:])
        solver_flag = solver.GBZ_check_main_loop(GBZ_condition_check_result)
        N_calls += 1
    
    print(solver.N_segments())

    all_segments = []

    for j in range(solver.N_segments()):
        coords, chart_labels = solver.segment_readout(j)
        for k in range(2):        
            chart1_flags = (chart_labels[:,k]==1)
            coords[chart1_flags, k + 1] = 1/coords[chart1_flags, k + 1]
        all_segments.append(coords)
    
    with open("data/minimal_model_test.pkl", "wb") as fp:
        pickle.dump((all_segments, GBZ_points, GBZ_params, non_GBZ_points, non_GBZ_params), fp)
    
    # all_points = []
    # for j in range(solver.N_points()):
    #     coords, chart_labels = solver.raw_sol_readout(j)
    #     for k in range(3):
    #         chart1_flags = (chart_labels[:,k] == 1)
    #         coords[chart1_flags, k+1] = 1/coords[chart1_flags, k + 1]
    #     all_points.append(coords)

    # data = {"params":params}
    # sio.savemat("data/minimal_winding_aGBZ.mat", data)
    # with open("data/minimal_winding_aGBZ.pkl", "wb") as fp:
    #     pickle.dump(all_points, fp)

def save_model_data():
    model = minimal_winding_model()
    # char_poly = model.get_characteristic_polynomial_ctype()
    char_poly = minimal_winding_char_poly()
    
    num_coeff = csolver.CScalarVec([])
    num_orders = csolver.CIndexVec([])
    char_poly.num.batch_get_data(num_coeff, num_orders)
    # num_coeff, num_orders = char_poly.f_num.batch_get_data()
    print(num_orders)
    denom_orders = char_poly.denom_orders
    # denom_coeffs, denom_orders = char_poly.f_denom.batch_get_data()
    # denom_orders = denom_orders.flatten()
    char_poly_orders = np.reshape(num_orders, (-1, 3))
    for j in range(char_poly_orders.shape[0]):
        for k in range(3):
            char_poly_orders[j, k] -= denom_orders[k]
    print(char_poly_orders)
    print(num_coeff)
    sio.savemat("data/GBZ_char_poly_8.mat", {'coeffs':np.array(num_coeff), "orders":char_poly_orders})
    sio.savemat("data/GBZ_info_8.mat", {
        "solution_index":8,
        'info' : "Non-Hermitian HN model with non-trivial winding. minimal_winding_model.py"
    })
    with open("data/GBZ_model_8.pkl", "wb") as fp:
        pickle.dump(model, fp)

def check_model_data():
    data = sio.loadmat("data/GBZ_char_poly_6.mat")
    coeffs = data['coeffs'].flatten()
    orders = data['orders'].flatten()
    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(csolver.CScalarVec(coeffs), csolver.CIntVec(orders))
    print(char_poly.num.to_str(csolver.CStrVec(['E','x','y'])))
    print(char_poly.denom_orders)

def calculate_OBC_spectrum():
    # intra_cell = []
    # inter_cell = [
    #     [0, 0, 1, (1,0)],
    #     [0, 0, 1, (-1,0)],
    #     [0, 0, 1, (-1, 1)],
    #     [0, 0, 1, (2, -1)]
    # ]

    # model = tb.TightBindingModel(2, 1,[[1,0], [0,1]], intra_cell, inter_cell)

    model = minimal_winding_model()

    # for sys_size in [(100, 100), (50, 200), (200, 50), ]:
    for sys_size in [(400, 3), (450, 3), (500, 3)]:
        print(sys_size)
        Nx, Ny = sys_size
        model_expand1 = model.get_supercell(
            [(j,0) for j in range(Nx)],
            [
                [Nx, 0],
                [0, 1]
            ]
        ) 
        model_expand2 = model_expand1.get_supercell(
            [(0,j) for j in range(Ny)],
            [
                [1, 0],
                [0, Ny]
            ]
        )
        eigv, eigvec = la.eig(model_expand2.get_bulk_Hamiltonian_dense((None, None)))
        sio.savemat("data/minimal_model_OBC_spectrum_Nx_%d_Ny_%d.mat"%(Nx, Ny), {'eigv':eigv, "eigvec":eigvec})
        plt.plot(eigv.real, eigv.imag, '.')
    plt.show()

def test_new_bubble_gen():
    char_poly = minimal_winding_char_poly()
    which_var = 2
    with open("data/minimal_model_test.pkl", "rb") as fp:
        data = pickle.load(fp)
    all_segments, GBZ_points, GBZ_params, non_GBZ_points, non_GBZ_params = data

    new_solver = csolver.HomotopyGBZChecker(char_poly, which_var)
    raw_sols = new_solver.get_aGBZ_solutions(GBZ_points[0], GBZ_params[0])
    outer_vals, inner_vals = new_solver.GBZ_check(GBZ_points[0], GBZ_params[0], N_points_detour=50, N_points_loop=100, 
                                                        theta_detour=1e-3 * np.pi)
    outer_vals = np.array(outer_vals)
    inner_vals = np.array(inner_vals)
    print(outer_vals[0], outer_vals[-1])
    print(inner_vals[0], inner_vals[-1])
    print(csolver.winding_number(outer_vals), csolver.winding_number(inner_vals))
    plt.plot(outer_vals.real, outer_vals.imag, 'r.')
    plt.plot(inner_vals.real, inner_vals.imag, 'b.')
    plt.plot(0,0,'x')
    plt.show()
    # print(len(outer_paths), len(inner_paths), len(links))
    # for j in range(len(outer_paths)):
    #     print("outer:", outer_paths[j][0], "to", outer_paths[j][-1])
    #     print("inner:", inner_paths[j][0], "to", inner_paths[j][-1])
    #     print("link:", links[j][0], "to", links[j][-1])
    
    # plt.plot(raw_sols[:,1].real, raw_sols[:,1].imag, 'bx')
    # for p in outer_paths:
    #     plt.plot(p.real, p.imag, 'r-')
    # for p in inner_paths:
    #     plt.plot(p.real, p.imag, 'g-')
    # for p in links:
    #     plt.plot(p.real, p.imag, 'b-')
    # plt.show()

def test_groebner():
    char_poly = minimal_winding_char_poly()
    coeffs = csolver.CScalarVec([])
    orders = csolver.CIndexVec([])
    char_poly.num.batch_get_data(coeffs, orders)

    raw_orders = np.reshape(orders, (-1,3))
    new_orders1 = np.zeros((raw_orders.shape[0], 4), dtype=int)
    new_orders2 = np.zeros((raw_orders.shape[0], 4), dtype=int)
    new_orders1[:,0] = raw_orders[:,2]
    new_orders1[:,1] = raw_orders[:,1]
    new_orders1[:,2] = 0
    new_orders1[:,3] = raw_orders[:,0]
    new_orders2[:,0] = raw_orders[:,2]
    new_orders2[:,1] = raw_orders[:,1]
    new_orders2[:,2] = raw_orders[:,2]
    new_orders2[:,3] = raw_orders[:,0]

    eff_poly_1 = csolver.CPolyLinkedList(4)
    eff_poly_2 = csolver.CPolyLinkedList(4)
    eff_poly_1.batch_add_elements(coeffs, new_orders1.flatten())
    eff_poly_2.batch_add_elements(coeffs, new_orders2.flatten())

    k = np.exp(1j * np.pi/3)
    E = 1 + 2j
    # print(eff_poly_1.to_str(['x', 'y', 'k', 'E']))
    # print(eff_poly_2.to_str(['x', 'y', 'k', 'E']))

    poly_basis = csolver.CPolyDeque()
    poly_basis.push_back(eff_poly_1.partial_eval(csolver.CScalarVec([k, E]), csolver.CIndexVec([2, 3]), csolver.CIndexVec([0,1])))
    poly_basis.push_back(eff_poly_2.partial_eval(csolver.CScalarVec([k, E]), csolver.CIndexVec([2, 3]), csolver.CIndexVec([0,1])))

    print("Basis:", poly_basis.size())
    eq_strs = []
    for j in range(poly_basis.size()):
        curr_poly = csolver.CPolyLinkedList(4)
        poly_basis.get_poly(j, curr_poly)
        print(curr_poly.to_str(csolver.CStrVec(["y", "x", "k", "E"])))
        eq_strs.append(curr_poly.to_str(csolver.CStrVec(["y", "x", "k", "E"])))
    
    all_sols = phcpy.solver.solve(eq_strs, dictionary_output=True)
    for sol in all_sols:
        print(sol)
    csolver.buchberger(poly_basis, 1e-15)
    # poly_basis = csolver.non_zero_Groebner(poly_basis, 1e-10)
    print("Groebner:", poly_basis.size())
    for j in range(poly_basis.size()):
        curr_poly = csolver.CPolyLinkedList(4)
        poly_basis.get_poly(j, curr_poly)
        print(curr_poly.to_str(csolver.CStrVec(["y", "x", "k", "E"])))

def check_1d_GBZ_direct_solver():
    char_poly = minimal_winding_char_poly()
    which_var = 2
    with open("data/minimal_model_test.pkl", "rb") as fp:
        data = pickle.load(fp)
    all_segments, GBZ_points, GBZ_params, non_GBZ_points, non_GBZ_params = data
    new_solver = csolver.DirectGBZConditionChecker(char_poly, 2)
    curr_j = 0
    for phi in np.linspace(np.pi/101, np.pi * 100/101, 100):
        all_sols = new_solver.get_aGBZ_solutions(GBZ_points[curr_j][:3], 
                    # non_GBZ_params[0], 
                    [np.exp(1j * phi), 1])
        print(len(all_sols))
        plt.plot(all_sols[:,1].real, all_sols[:,1].imag, '.')

def solve_1d_equation(char_poly:csolver.CLaurant, curr_E, curr_beta, which_var):
    if(which_var == 1):
        other_var = 2
    else:
        other_var = 1
    eff_poly_1d = csolver.CPolyLinkedList(1)
    char_poly.num.partial_eval(csolver.CScalarVec([curr_E, curr_beta]), csolver.CIndexVec([0, which_var]),
                            csolver.CIndexVec([other_var]), eff_poly_1d)
    coeffs, degrees = eff_poly_1d.batch_get_data()
    degrees = degrees.flatten()
    max_deg = np.max(degrees)


    all_coeffs = np.zeros(max_deg + 1, dtype=complex)
    for j in range(len(degrees)):
        all_coeffs[max_deg - degrees[j]] = coeffs[j]

    all_roots = np.roots(all_coeffs)
    seq = np.argsort(np.abs(all_roots))
    return all_roots[seq]

def complex_to_sphere(beta):
    denom = (np.abs(beta) ** 2) + 1
    X = 2 * beta.real / denom
    Y = 2 * beta.imag / denom
    Z = (np.abs(beta) ** 2 - 1) / denom
    return X, Y, Z

def test_roots():
    r_arr = np.linspace(1e-3, 1, 100)
    phi_arr = np.linspace(0, 2*np.pi, 100)
    r_mesh, phi_mesh = np.meshgrid(r_arr, phi_arr)
    beta_mesh = r_mesh * np.exp(1j * phi_mesh)

    curr_E = 5
    which_var = 2

    if (which_var == 1):
        all_roots1 = np.zeros((2,) + beta_mesh.shape, dtype=complex)
        all_roots2 = np.zeros((2,) + beta_mesh.shape, dtype=complex)
    else:
        all_roots1 = np.zeros((3,) + beta_mesh.shape, dtype=complex)
        all_roots2 = np.zeros((3,) + beta_mesh.shape, dtype=complex)

    char_poly = minimal_winding_char_poly()

    for i in range(beta_mesh.shape[0]):
        for j in range(beta_mesh.shape[1]):
            all_roots1[:,i,j] = solve_1d_equation(char_poly, curr_E, beta_mesh[i,j], which_var)
            all_roots2[:,i,j] = solve_1d_equation(char_poly, curr_E, 1.0/beta_mesh[i,j], which_var)

    X, Y, Z = complex_to_sphere(beta_mesh)
    mesh1 = pv.StructuredGrid(X, Y, Z)
    # mesh1 = pv.StructuredGrid(beta_mesh.real, beta_mesh.imag, 0 * beta_mesh.real)
    X, Y, Z = complex_to_sphere(1/beta_mesh)
    mesh2 = pv.StructuredGrid(X, Y, Z)
    # mesh_list = []
    # for j in range(all_roots1.shape[0]):
    #     mesh_list.append(
    #         pv.StructuredGrid(beta_mesh.real, beta_mesh.imag, np.abs(all_roots[j,:,:]))
    #     )

    plotter = pv.Plotter()
    # color = [(255,0,0), (0,0,255), (0, 255, 0), (255,255,0)]
    # for j in range(all_roots.shape[0]):
    #     plotter.add_mesh(mesh_list[j], color=color[j], opacity=0.95)
    j = 0
    max_color = 0.2
    curr_scalar = np.abs(all_roots1[j+1,:,:]) - np.abs(all_roots1[j,:,:])
    plotter.add_mesh(mesh1, scalars=(curr_scalar.transpose()), cmap="hot", clim=[0,max_color])
    curr_scalar = np.abs(all_roots2[j+1,:,:]) - np.abs(all_roots2[j,:,:])
    plotter.add_mesh(mesh2, scalars=(curr_scalar.transpose()), cmap="hot", clim=[0,max_color])
    plotter.show()

def check_supercell_1D_GBZ(N_supercell:int=10):
    model = minimal_winding_model()
    model_x_super = model.get_supercell(
        [(0,j) for j in range(N_supercell)],
        [
            [1, 0],
            [0, N_supercell]
        ]
    )
    model_x_super_1d = model_x_super.truncate(1)
    # GBZ_loops = model_x_super_1d.get_GBZ(segments_only=True, N_phi=50)
    GBZ_loops = model_x_super_1d.get_GBZ(N_phi=50)
    with open("data/minimal_winding_GBZ_loops_N%d.pkl"%(N_supercell), "wb") as fp:
        pickle.dump(GBZ_loops, fp)
    
    # for curr_loop in GBZ_loops:
    #     plt.plot(curr_loop[:,0].real, curr_loop[:,0].imag)
    plt.plot(GBZ_loops[1].real, GBZ_loops[1].imag, '.')
    plt.xlabel("$\\mathrm{Re}(E)$")
    plt.ylabel("$\\mathrm{Im}(E)$")
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # for curr_loop in GBZ_loops:
    #     ax.plot(curr_loop[:,1].real, curr_loop[:,1].imag, curr_loop[:,0].real)
    ax.plot(GBZ_loops[2].real, GBZ_loops[2].imag, GBZ_loops[1].real, '.')
    ax.set_xlabel("$\\mathrm{Re}(\\beta)$")
    ax.set_ylabel("$\\mathrm{Im}(\\beta)$")
    ax.set_zlabel("$\\mathrm{Re}(E)$")
    plt.show()

def check_supercell_1D_GBZ_y_major(N_supercell:int = 10):
    model = minimal_winding_model()
    model_y_super = model.get_supercell(
        [(j,0) for j in range(N_supercell)],
        [
            [N_supercell, 0],
            [0, 1]
        ]
    )
    model_y_super_1d = model_y_super.truncate(0)
    GBZ_loops = model_y_super_1d.get_GBZ(N_phi=50)
    with open("data/minimal_winding_GBZ_loops_Ny%d.pkl"%(N_supercell), "wb") as fp:
        pickle.dump(GBZ_loops, fp)
    
    # for curr_loop in GBZ_loops:
    #     plt.plot(curr_loop[:,0].real, curr_loop[:,0].imag)
    plt.plot(GBZ_loops[1].real, GBZ_loops[1].imag, '.')
    plt.xlabel("$\\mathrm{Re}(E)$")
    plt.ylabel("$\\mathrm{Im}(E)$")
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # for curr_loop in GBZ_loops:
    #     ax.plot(curr_loop[:,1].real, curr_loop[:,1].imag, curr_loop[:,0].real)
    ax.plot(GBZ_loops[2].real, GBZ_loops[2].imag, GBZ_loops[1].real, '.')
    ax.set_xlabel("$\\mathrm{Re}(\\beta)$")
    ax.set_ylabel("$\\mathrm{Im}(\\beta)$")
    ax.set_zlabel("$\\mathrm{Re}(E)$")
    plt.show()

def plot_OBC():
    with open("data/minimal_winding_GBZ_loops_N8.pkl", "rb") as fp:
        all_segs = pickle.load(fp)
    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.gca()
    
    # color = ['m.', 'g.', 'b.', 'c.']
    for j, Nx in enumerate(range(250, 501, 50)):
        obc_data = sio.loadmat("data/minimal_model_OBC_spectrum_Nx_%d_Ny_%d.mat"%(Nx, 8))
        eigv = obc_data['eigv'].flatten()
        ax.plot(eigv.real, eigv.imag, '.')

    for seg in all_segs:
        # ax.plot(seg[:,1].real, seg[:,1].imag, seg[:,0].real)
        ax.plot(seg[:,0].real, seg[:,0].imag, 'r-',linewidth=1.5)

    plt.show()


if __name__ == '__main__':
    # main_get_aGBZ()
    # save_model_data()
    # check_model_data()
    # calculate_OBC_spectrum()
    # test_new_bubble_gen()
    # test_groebner()
    # check_1d_GBZ_direct_solver()
    # test_roots()
    # for N in [11,12,13,14]:
    check_supercell_1D_GBZ(20)
    # check_supercell_1D_GBZ_y_major()
    # plot_OBC()