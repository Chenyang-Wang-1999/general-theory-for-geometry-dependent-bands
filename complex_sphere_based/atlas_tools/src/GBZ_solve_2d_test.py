'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-08-16
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from solver_tools import *
import c_solver_2D as csolver
import phcpy
from math import sin, cos, sqrt, pi
from cmath import exp, log
import matplotlib.pyplot as plt
import pickle
import sys
from scipy import io as sio
from scipy import linalg as la
import os
import cProfile
from minimal_winding_model import minimal_winding_char_poly

def test_diff():
    pass

def char_poly_sample(E, beta1, beta2, params):
    ux1, ux2, uy1, uy2, vx1, vx2 = params
    polyval = (E**2 - (vx1/beta2 + vx2*beta2 + uy1/beta2 + uy2*beta2) * E + 
                vx2*uy2 * (beta2**2) + uy1 * vx1/(beta2**2) - ux1 * vx1/beta1 - ux2 * vx2 * beta1 + 
                vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2)
    return polyval

def test_char_poly(x_arr, expiphi, chart_labels, params):
    E = x_arr[0]
    beta1 = x_arr[1]
    beta2 = x_arr[2]
    beta_another = x_arr[3]
    if(chart_labels[0] == 1):
        beta1 = 1/beta1
    if(chart_labels[1] == 1):
        beta2 = 1/beta2
    if(chart_labels[2] == 1):
        beta_another = 1/beta_another
    results = np.zeros(4,dtype=complex)
    results[0] = char_poly_sample(E, beta1, beta2, params)
    results[1] = char_poly_sample(E, beta1, beta2*expiphi[1], params)
    results[2] = char_poly_sample(E, beta1*expiphi[0], beta_another, params)
    results[3] = char_poly_sample(E, beta1*expiphi[0], beta_another*expiphi[1], params)

    return results

def test_char_poly_y_major(x_arr, expiphi, chart_labels, params):
    E = x_arr[0]
    beta1 = x_arr[1]
    beta2 = x_arr[2]
    beta_another = x_arr[3]
    if(chart_labels[0] == 1):
        beta1 = 1/beta1
    if(chart_labels[1] == 1):
        beta2 = 1/beta2
    if(chart_labels[2] == 1):
        beta_another = 1/beta_another
    results = np.zeros(4,dtype=complex)
    results[0] = char_poly_sample(E, beta1, beta2, params)
    results[1] = char_poly_sample(E, beta1*expiphi[0], beta2, params)
    results[2] = char_poly_sample(E, beta_another, beta2*expiphi[1], params)
    results[3] = char_poly_sample(E, beta_another*expiphi[0], beta2*expiphi[1], params)

    return results

def test_char_poly_Laurant():
    vx1 = 1.0
    vx2 = 1.0
    ux1 = 3.0
    ux2 = 3.0
    uy1 = 4.0
    uy2 = 4.0

    coeffs = csolver.ScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.IntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0,
    ])

    my_Laurant = csolver.CLaurant(3)
    my_Laurant.set_Laurant_by_terms(coeffs, orders)
    x = np.random.randn(3) + 1j * np.random.randn(3)
    val_theo = char_poly_sample(x[0], x[1], x[2], (ux1, ux2, uy1, uy2, vx1, vx2 ))
    val_Laurant = my_Laurant.eval(csolver.ScalarVec(x))
    print(val_theo)
    print(val_Laurant)
    print(val_theo - val_Laurant)

def solve_GBZ(char_poly:csolver.CLaurant, major_dim:int, params:np.ndarray, critical_distance:float=0.1, 
               N_loop_points:int = 100, rel_r_detour:float = 1e-3, 
               outer_N_loop_points:int=50, outer_rel_r_detour:float=1e-3, dphase_th = 0.1*np.pi):
    solver = csolver.aGBZ2DSolver(major_dim, critical_distance, N_loop_points, 
                                  rel_r_detour, dphase_th)
    # solver.set_char_poly_by_coeffs(coeffs, orders)
    solver.set_char_poly(char_poly)
    solver.set_param(params)

    solver_flag = True
    full_solver_flag_list = []
    while(solver_flag):
        if(solver.full_solver_flag):
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
            continuation_solver_next_flag = True
            while(continuation_solver_next_flag):
                res = complex_root(solver.continuation_solver_fun, np.array(solver.curr_initial_values), jac=True)
                if(not res.success):
                    solver.suspend_solver()
                
                continuation_solver_next_flag = solver.insert_continuation_solver_results(res.x.reshape((1,-1)))

        print(" "*100, end='\r')
        print(solver.get_curr_param(), file=sys.stdout, end='\r')
        full_solver_flag_list.append(solver.full_solver_flag)
        solver_flag = solver.next()
    
    solver_flag = solver.sol_to_segments_init()
    N_calls = 0
    while(solver_flag):
        GBZ_condition_check_result = csolver.check_GBZ_condition(
            char_poly, solver.GBZ_check_curr_point[:3], params[solver.GBZ_check_curr_ptr,:], solver.major_dim + 1,
            outer_N_points_loop=outer_N_loop_points,  rel_r_detour=rel_r_detour, N_points_loop = N_loop_points,outer_rel_r_detour=outer_rel_r_detour,
            dphase_th = dphase_th
        )
        print(" "*100, end='\r')
        print(N_calls, "finished, ", GBZ_condition_check_result, end='\r')
        solver_flag = solver.GBZ_check_main_loop(GBZ_condition_check_result)
        N_calls += 1
    print(" "*100, end='\r')

    all_segments = []

    for j in range(solver.N_segments()):
        coords, chart_labels = solver.segment_readout(j)
        for k in range(2):        
            chart1_flags = (chart_labels[:,k]==1)
            coords[chart1_flags, k + 1] = 1/coords[chart1_flags, k + 1]
        all_segments.append(coords)
    
    return all_segments

def test_y_major():
    vx1 = 1.0 + np.random.randn() * 0.5
    vx2 = 1.0 + np.random.randn() * 0.5
    ux1 = 3.0 + np.random.randn() * 0.5
    ux2 = 3.0 + np.random.randn() * 0.5
    uy1 = 4.0 + np.random.randn() * 0.5
    uy2 = 4.0 + np.random.randn() * 0.5
    coeffs = csolver.CScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0
    ])

    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)

    N_phi1 = 100
    N_phi2 = 100
    phi1_list = np.linspace(np.pi/(N_phi1+1), np.pi - np.pi/(N_phi1+1), N_phi1)
    phi2_list = np.linspace(np.pi/(N_phi2+1), np.pi - np.pi/(N_phi2+1), N_phi2)

    all_segments = []
    # for j, phi2 in enumerate(phi2_list):
    # for j, phi2 in enumerate([phi2_list[50]]):
    for phi1 in [pi/3]:
        # print("current phi2 = %.2f pi"%(phi2/pi), "%.2f%%"%(j/N_phi2*100))
        params = np.zeros((N_phi1, 2), dtype=complex)
        params[:,0] = np.exp(1j*phi1)
        params[:,1] = np.exp(1j*phi2_list)
        curr_segments = solve_GBZ(char_poly, 1, params)
        all_segments.append(curr_segments)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for seg in curr_segments:
            ax.plot(seg[:,1].real, seg[:,1].imag, seg[:,0].real)
        plt.show()

def sweep_2D_GBZ(data_id:float):
    vx1 = 1.0 + np.random.randn() * 0
    vx2 = 1.0 + np.random.randn() * 0
    ux1 = 3.0 + np.random.randn() * 0
    ux2 = 3.0 + np.random.randn() * 0
    uy1 = 4.0 + np.random.randn() * 0
    uy2 = 4.0 + np.random.randn() * 0
    coeffs = csolver.CScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0,
    ])

    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)

    N_phi1 = 50
    N_phi2 = 50
    phi1_list = np.linspace(np.pi/(N_phi1+1), np.pi - np.pi/(N_phi1+1), N_phi1)
    phi2_list = np.linspace(np.pi/(N_phi2+1), np.pi - np.pi/(N_phi2+1), N_phi2)

    data = {}
    data['model_params'] = (ux1, ux2, uy1, uy2, vx1, vx2)
    data['phi1_list'] = phi1_list
    data['phi2_list'] = phi2_list
    sio.savemat("data/GBZ_info_%d.mat"%(data_id), data)
    if(not os.path.exists("data/GBZ_data_%d"%(data_id))):
        os.mkdir("data/GBZ_data_%d"%(data_id))

    all_segments = []
    for j, phi2 in enumerate(phi2_list):
    # for j, phi2 in enumerate([phi2_list[50]]):
        print("current phi2 = %.2f pi"%(phi2/pi), "%.2f%%"%(j/N_phi2*100))
        params = np.zeros((N_phi1, 2), dtype=complex)
        params[:,0] = np.exp(1j*phi1_list)
        params[:,1] = np.exp(1j*phi2)
        try:
            curr_segments = solve_GBZ(char_poly, 0, params)
        except:
            curr_segments = solve_GBZ(char_poly, 0, params, N_loop_points=100, outer_N_loop_points=200, dphase_th=0.05*np.pi, outer_rel_r_detour=0.01)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # for seg in curr_segments:
        #     ax.plot(seg[:,1].real, seg[:,1].imag, seg[:,0].real)
        # plt.show()
        with open("data/GBZ_data_%d/%d.pkl"%(data_id, j), "wb") as fp:
            pickle.dump(curr_segments, fp)
    
def sweep_2D_GBZ_y_major(data_id:float):
    vx1 = 1.0 + np.random.randn() * 0
    vx2 = 1.0 + np.random.randn() * 0
    ux1 = 3.0 + np.random.randn() * 0
    ux2 = 3.0 + np.random.randn() * 0
    uy1 = 4.0 + np.random.randn() * 0
    uy2 = 4.0 + np.random.randn() * 0
    coeffs = csolver.CScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0,
    ])

    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)

    N_phi1 = 50
    N_phi2 = 50
    phi1_list = np.linspace(np.pi/(N_phi1+1), np.pi - np.pi/(N_phi1+1), N_phi1)
    phi2_list = np.linspace(np.pi/(N_phi2+1), np.pi - np.pi/(N_phi2+1), N_phi2)

    data = {}
    data['model_params'] = (ux1, ux2, uy1, uy2, vx1, vx2)
    data['phi1_list'] = phi1_list
    data['phi2_list'] = phi2_list
    sio.savemat("data/GBZ_info_y_major_%d.mat"%(data_id), data)
    if(not os.path.exists("data/GBZ_data_y_major_%d"%(data_id))):
        os.mkdir("data/GBZ_data_y_major_%d"%(data_id))

    all_segments = []
    for j, phi1 in enumerate(phi1_list):
    # for j, phi2 in enumerate([phi2_list[50]]):
        print("current phi1 = %.2f pi"%(phi1/pi), "%.2f%%"%(j/N_phi1*100))
        params = np.zeros((N_phi2, 2), dtype=complex)
        params[:,0] = np.exp(1j*phi1)
        params[:,1] = np.exp(1j*phi2_list)
        try:
            curr_segments = solve_GBZ(char_poly, 1, params)
        except:
            curr_segments = solve_GBZ(char_poly, 1, params, N_loop_points=100, outer_N_loop_points=100, dphase_th=0.05*np.pi)

        all_segments.append(curr_segments)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # for seg in curr_segments:
        #     ax.plot(seg[:,1].real, seg[:,1].imag, seg[:,0].real)
        # plt.show()
        with open("data/GBZ_data_y_major_%d/%d.pkl"%(data_id, j), "wb") as fp:
            pickle.dump(curr_segments, fp)
 

def Hamiltonian_matrix(k1, k2, ux, uy, vx):
    H = np.array([
        [2 * uy * np.cos(k2), ux + vx * np.exp(1j*k1)],
        [ux + vx * np.exp(-1j*k1), 2 * vx * np.cos(k2)]
    ])
    return H
        
def test_aGBZ_x_major(char_poly:csolver.CLaurant, x_arr, expiphi, chart_labels):
    E = x_arr[0]
    beta1 = x_arr[1]
    beta2 = x_arr[2]
    beta_another = x_arr[3]
    if(chart_labels[0] == 1):
        beta1 = 1/beta1
    if(chart_labels[1] == 1):
        beta2 = 1/beta2
    if(chart_labels[2] == 1):
        beta_another = 1/beta_another
    results = np.zeros(4,dtype=complex)
    results[0] = char_poly.num.eval(csolver.CScalarVec([E, beta1, beta2]))
    results[1] = char_poly.num.eval(csolver.CScalarVec([E, beta1, beta2*expiphi[1]]))
    results[2] = char_poly.num.eval(csolver.CScalarVec([E, beta1*expiphi[0], beta_another]))
    results[3] = char_poly.num.eval(csolver.CScalarVec([E, beta1*expiphi[0], beta_another*expiphi[1]]))
    return results

def test_aGBZ_y_major(char_poly:csolver.CLaurant, x_arr, expiphi, chart_labels):
    E = x_arr[0]
    beta1 = x_arr[1]
    beta2 = x_arr[2]
    beta_another = x_arr[3]
    if(chart_labels[0] == 1):
        beta1 = 1/beta1
    if(chart_labels[1] == 1):
        beta2 = 1/beta2
    if(chart_labels[2] == 1):
        beta_another = 1/beta_another
    results = np.zeros(4,dtype=complex)
    results[0] = char_poly.num.eval(csolver.CScalarVec([E, beta1, beta2]))
    results[1] = char_poly.num.eval(csolver.CScalarVec([E, beta1*expiphi[0], beta2]))
    results[2] = char_poly.num.eval(csolver.CScalarVec([E, beta_another, beta2*expiphi[1]]))
    results[3] = char_poly.num.eval(csolver.CScalarVec([E, beta_another*expiphi[0], beta2*expiphi[1]]))

    return results

def test_basics():
    vx1 = 1.0 + np.random.randn() * 0
    vx2 = 1.0 + np.random.randn() * 0
    ux1 = 3.0 + np.random.randn() * 0
    ux2 = 3.0 + np.random.randn() * 0
    uy1 = 4.0 + np.random.randn() * 0
    uy2 = 4.0 + np.random.randn() * 0
    print(vx1, vx2, ux1, ux2, uy1, uy2)
    # input()

    # coeffs = [
    #     1, -1, -t2*t3, -(t1*t2 + t1 * t3 + gamma * t2 / 2 - gamma * t3/2),
    #     - (t1 * t2 + t1 * t3 - gamma * t2/2 + gamma*t3/2),
    #     -(-gamma**2/4 + t1**2 + t2**2 + t3**2)
    # ]
    # orders = [
    #     2, 0,
    #     0, -2,
    #     0, 2,
    #     0, 1,
    #     0, -1,
    #     0, 0
    # ]
    # coeffs = csolver.CScalarVec([
    #     1, - vx1, -vx2, -uy1, -uy2,
    #     vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
    #     vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    # ])
    # orders = csolver.CIntVec([
    #     # E, beta1, beta2
    #     2, 0, 0,
    #     1, 0, -1,
    #     1, 0, 1,
    #     1, 0, -1,
    #     1, 0, 1,
    #     0, 0, 2,
    #     0, 0, -2,
    #     0, -1, 0,
    #     0, 1, 0,
    #     0, 0, 0
    # ])

    # char_poly_data = sio.loadmat("data/GBZ_char_poly_5.mat")
    
    # coeffs = csolver.CScalarVec(char_poly_data['coeffs'].flatten())
    # orders = csolver.CIntVec(char_poly_data['orders'].flatten())

    # char_poly = csolver.CLaurant(3)
    # char_poly.set_Laurant_by_terms(coeffs, orders)
    char_poly = minimal_winding_char_poly()

    # print(char_poly.num.to_str(csolver.CStrVec(['x','y','z','w'])))
    # print(char_poly.denom_orders)
    # return

    theta_detour = 5e-3*np.pi
    solver = csolver.aGBZ2DSolver(1, 1e-1, 200, 50, theta_detour, 1e-3)
    solver.set_char_poly(char_poly)
    # solver.set_char_poly_by_coeffs(coeffs, orders)
    N_params = 50
    # params = np.exp(1j*np.linspace(np.pi/201, np.pi - np.pi/201, 200)).reshape((-1,1))
    N_phi2 = 50
    phi2_list = np.linspace(np.pi/(N_phi2+1), np.pi - np.pi/(N_phi2+1), N_phi2)
    phi2 = phi2_list[36]
    params = np.zeros((N_params, 2), dtype=complex)
    if(solver.major_dim == 1):
        params[:,1] = np.exp(1j*np.linspace(np.pi/(N_params+1), np.pi - np.pi/(N_params+1), N_params))
        # params[:,1] = np.exp(1j*np.linspace(np.pi/2, np.pi - np.pi/(N_params+1), N_params))
        params[:,0] = np.exp(1j*phi2)
    else:
        params[:,0] = np.exp(1j*np.linspace(np.pi/(N_params+1), np.pi - np.pi/(N_params+1), N_params))
        # params[:,0] = np.exp(1j*np.linspace(np.pi/2, np.pi - np.pi/(N_params+1), N_params))
        params[:,1] = np.exp(1j*phi2)

    solver.set_param(params)

    solver_flag = True
    full_solver_flag_list = []
    while(solver_flag):
        if(solver.full_solver_flag):
            print("Full solver")
            # full solver
            full_solver_next_flag = True
            N_solver_times = 0
            while(full_solver_next_flag):
                # print(solver.curr_poly_string_list, file=sys.stdout)
                all_sols = phcpy.solver.solve(list(solver.curr_poly_string_list), dictionary_output=True, verbose=False)
                # print(len(all_sols))
                # input()
                sols_complex = []
                # print(N_solver_times, full_solver_next_flag)
                N_solver_times += 1
                for sol in all_sols:
                    if(abs(sol['y'])<1.5):
                        if(abs(sol['z'])<1.5):
                            if(abs(sol['w'])<1.5):
                    # if(True):
                                res = complex_root(solver.continuation_solver_fun, np.array([sol['x'], sol['y'], sol['z'], sol['w']]), jac=True)
                                sols_complex.append(res.x)
                                if(not res.success):
                                    print(res.message)
                                    # input()
                                    raise ValueError()
                                # print(solver.continuation_solver_fun(res.x))
                # print(np.array(sols_complex))
                # print(1/np.array(sols_complex)[:,1])
                if(len(sols_complex) == 0):
                    full_solver_next_flag = solver.insert_full_solver_results(np.asarray([[0,0,0,0]], dtype=complex), True)
                else:
                    full_solver_next_flag = solver.insert_full_solver_results(np.asarray(sols_complex), False)

                # input()
                # plt.plot(np.array(sols_complex)[:,1].real, np.array(sols_complex)[:,1].imag, 'b.')
        else:
            print("Continuation solver")
            # continuation solver
            continuation_solver_next_flag = True
            while(continuation_solver_next_flag):
                # print("Continuation solver:", solver.curr_initial_values)
                # plt.plot(solver.curr_initial_values[1].real, solver.curr_initial_values[1].imag, 'b.')
                res = complex_root(solver.continuation_solver_fun, np.array(solver.curr_initial_values), jac=True)
                if(not res.success):
                    # print(solver.continuation_solver_fun(res.x))
                    # print(res.message)
                    solver.suspend_solver()
                # print(solver.continuation_solver_fun(res.x))
                
                # x0 = np.random.randn(2) + 1j*np.random.randn(2)
                # dx0 = (np.random.randn(2) + 1j*np.random.randn(2)) * 1e-3
                # x1, jac = solver.continuation_solver_fun(x0)
                # print(jac)
                # x2 = solver.continuation_solver_fun(x0 + dx0)[0]
                # print(x2 - x1)
                # print(jac@dx0)

                # print(res.success)
                # print(res.message)
                # print(solver.continuation_solver_fun(res.x)[0])
                continuation_solver_next_flag = solver.insert_continuation_solver_results(res.x.reshape((1,-1)))
                # input()

        # if(solver.solver_suspended):
        #     print("solver suspended")
        #     input()
        # print("solving finishes")
        print(solver.get_curr_param(), file=sys.stdout)
        full_solver_flag_list.append(solver.full_solver_flag)
        solver_flag = solver.next()
        # input()
        # plt.pause(0.5)
    # plt.show()

    # test raw solutions
    # sols, chart_labels = solver.raw_sol_readout(0)
    # for j in range(sols.shape[0]):
    #     for k in range(3):
    #         if(chart_labels[j,k]):
    #             sols[j, k+1] = 1.0/sols[j,k+1]
    # order = np.argsort(sols[:,0].real)
    # print(sols[order,:])
    # print(chart_labels)
    # return
    
    GBZ_condition_checker = csolver.DirectGBZConditionChecker(
        char_poly, solver.major_dim + 1
    )

    solver_flag = solver.sol_to_segments_init()
    N_calls = 0
    bug_points = []
    bug_params = []
    while(solver_flag):
        # print(solver.GBZ_check_curr_point)
        # print(params[solver.GBZ_check_curr_ptr,:])
        # print(solver.major_dim)
        # GBZ_condition_check_result = csolver.check_GBZ_condition(
        #     char_poly, solver.GBZ_check_curr_point[:3], params[solver.GBZ_check_curr_ptr,:], solver.major_dim + 1, 
        #     outer_N_points_detour = 20, N_points_detour = 20, outer_theta_detour = theta_detour, theta_detour = theta_detour
        # )
        GBZ_condition_check_result = GBZ_condition_checker.check_GBZ_condition(
            solver.GBZ_check_curr_point[:3], params[solver.GBZ_check_curr_ptr, :]
        )

        N_calls += 1
        # if(not GBZ_condition_check_result):
        # if(True):
        #     # if(abs(abs(solver.GBZ_check_curr_point[solver.major_dim + 1])- 1) < 1e-6):
        #     bug_points.append(solver.GBZ_check_curr_point.copy())
        #     bug_params.append(params[solver.GBZ_check_curr_ptr,:])
        #     print("Saving bug points:", solver.GBZ_check_curr_point)
        print(N_calls, "finished, ", GBZ_condition_check_result)
        # print(char_poly.eval(csolver.solver.GBZ_check_curr_point[:3]))
        solver_flag = solver.GBZ_check_main_loop(GBZ_condition_check_result)

    #     # if(solver.major_dim == 0):
    #     #     print("Polynomial values:", np.abs(test_char_poly(solver.GBZ_check_curr_point, params[solver.GBZ_check_curr_ptr,:], [0,0,0], (ux1, ux2, uy1, uy2, vx1, vx2))))
    #     # else:
    #     #     print("Polynomial values:", np.abs(test_char_poly_y_major(solver.GBZ_check_curr_point, params[solver.GBZ_check_curr_ptr,:], [0,0,0], (ux1, ux2, uy1, uy2, vx1, vx2))))
        # solver_flag = solver.GBZ_check_main_loop(True)
    #     N_calls += 1
    # with open("data/bug_points_theta%.6fpi.pkl"%(theta_detour/np.pi), "wb") as fp:
    #     pickle.dump(bug_points, fp)
    # with open("data/bug_params_theta%.6fpi.pkl"%(theta_detour/np.pi), "wb") as fp:
    #     pickle.dump(bug_params, fp)
    # return
    # print(solver.N_loops())
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-5,5])
    ax.set_ylim([-5,5])
    ax.set_zlim([-5,5])
    all_colors = "rgbcmy"
    N_points = 0
    # for j in range(solver.N_points()):
    for j in range(solver.N_segments()):
    # for j in range(solver.N_loops()):
    # for j in range(1):
        # coords, chart_labels = solver.raw_sol_readout(j)
        coords, chart_labels = solver.segment_readout(j)
        N_points += coords.shape[0]
        # print(j, full_solver_flag_list[j])
        # for k in range(coords.shape[0]):
        #     if(solver.major_dim == 0):
        #         test_aGBZ_fun = test_aGBZ_x_major
        #     else:
        #         test_aGBZ_fun = test_aGBZ_y_major
        #     if(any(np.abs(np.abs(test_aGBZ_fun(char_poly, coords[k,:], params[j,:], chart_labels[k,:]))) > 1e-5)):
        #         print("params:", params[j,:])
        #         print("Coord:", coords[k,:])
        #         print("chart labels:", chart_labels[k])
        #         print("Polynomial values:", np.abs(test_aGBZ_fun(char_poly, coords[k,:], params[j,:], chart_labels[k,:])))
                # input()
            # (chart_labels[k] == 0):
            #     plt.plot(coords[k,1].real, coords[k,1].imag, '.')
            #     plt.plot((coords[k,1]*params[j]).real, (coords[k,1]*params[j]).imag, 'b.')
            # else:
            #     plt.plot((1/coords[k,1]).real, (1/coords[k,1]).imag, 'r.')
            #     plt.plot((params[j]/coords[k,1]).real, (params[j]/coords[k,1]).imag, 'r.')

            


        # print(coords.shape, chart_labels.shape)
        # for k in range(coords.shape[0]):
        #     # chart1_flags = (chart_labels[k,0]==1)
        #     # if(chart1_flags):
        #     #     coords[k, 1] = 1/coords[k, 1]
        #     if(chart_labels[k,0] == 1):
        #         coords[k, 1] = 1/coords[k, 1]
        #     if(chart_labels[k,1] == 1):
        #         coords[k, 2] = 1/coords[k, 2]
            
        #     # GBZ_check_result = csolver.check_GBZ_condition(char_poly, coords[k,:], 
        #     #                                                )
        #     ax.plot(coords[k,2].real, coords[k,2].imag, coords[k,0].real, all_colors[k%6] + '.')
        #     ax.plot((coords[k,2] * params[j,1]).real, (coords[k,2] * params[j,1]).imag, coords[k,0].real, all_colors[k%6] + '.')
        # plt.pause(0.5)
        chart1_flags = (chart_labels[:,solver.major_dim]==1)
        coords[chart1_flags, solver.major_dim + 1] = 1/coords[chart1_flags, solver.major_dim + 1]
        ax.plot(coords[:,solver.major_dim + 1].real, coords[:,solver.major_dim + 1].imag, coords[:,0].real, '.')
        # ax.plot(coords[:,solver.major_dim + 1].real, coords[:,solver.major_dim + 1].imag, coords[:,0].real, '.')
        # ax.plot(coords[0,solver.major_dim + 1].real, coords[0,solver.major_dim + 1].imag, coords[0,0].real, 'x')
        # ax.plot((coords[:,2]).real, (coords[:,2]).imag, coords[:,0].real, '.')
        # ax.plot((coords[:,2] * params[j,1]).real, (coords[:,2] * params[j,1]).imag, coords[:,0].real, '.')
        # plt.pause(0.01)
        # input()
    
    # k1 = np.linspace(0, 2 * np.pi, 100)
    # k2 = phi2/2
    # E = np.zeros((2, len(k1)))
    # for j in range(len(k1)):
    #     E[:,j] = la.eigh(Hamiltonian_matrix(k1[j], k2, ux1, uy1, vx1))[0]
    # for j in range(2):
    #     ax.plot(np.cos(k1), np.sin(k1), E[j, :], '--')


    plt.show()
    # print(coords)
    # print(chart_labels)
    # print(N_calls)
    # print(N_points)

def test_zero_point_selection():
    vx1 = 1.0 + np.random.randn() * 0
    vx2 = 1.0 + np.random.randn() * 0
    ux1 = 3.0 + np.random.randn() * 0
    ux2 = 3.0 + np.random.randn() * 0
    uy1 = 4.0 + np.random.randn() * 0
    uy2 = 4.0 + np.random.randn() * 0
    print(vx1, vx2, ux1, ux2, uy1, uy2)

    coeffs = csolver.CScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0
    ])

    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)

    major_dim = 0

    N_params = 50
    # params = np.exp(1j*np.linspace(np.pi/201, np.pi - np.pi/201, 200)).reshape((-1,1))
    N_phi2 = 50
    phi2_list = np.linspace(np.pi/(N_phi2+1), np.pi - np.pi/(N_phi2+1), N_phi2)
    phi2 = phi2_list[10]
    params = np.zeros((N_params, 2), dtype=complex)
    if(major_dim == 1):
        params[:,1] = np.exp(1j*np.linspace(np.pi/(N_params+1), np.pi - np.pi/(N_params+1), N_params))
        params[:,0] = np.exp(1j*phi2)
    else:
        params[:,0] = np.exp(1j*np.linspace(np.pi/(N_params+1), np.pi - np.pi/(N_params+1), N_params))
        params[:,1] = np.exp(1j*phi2)



    # E_and_beta =[2.15724-0.332099j,(-0.273308-0.0295373j),(0.772133+0.370849j),(0.943154+0.332355j)]
    # charts = [0, 0, 1]

    E_and_beta = [(0.183515+3.98468e-17j),(0.999526-0.0307951j),(-0.943154-0.332355j),(-0.943154+0.332355j)]
    charts = [0,1,0]
    for j in range(3):
        if(charts[j]):
            E_and_beta[j+1] = 1.0/E_and_beta[j+1]
    curr_ptr = 0 
    curr_param = params[curr_ptr,:]
    print(char_poly.eval(csolver.CScalarVec([E_and_beta[0], E_and_beta[1], E_and_beta[2]])))
    print(char_poly.eval(csolver.CScalarVec([E_and_beta[0], E_and_beta[1], E_and_beta[2]* curr_param[1]] )))
    print(char_poly.eval(csolver.CScalarVec([E_and_beta[0], E_and_beta[1]*curr_param[0], E_and_beta[3]])))
    print(char_poly.eval(csolver.CScalarVec([E_and_beta[0], E_and_beta[1]*curr_param[0], E_and_beta[3] * curr_param[1]])))
    
    phase_list = np.linspace(0, 2 * np.pi, 200)
    beta_loop = E_and_beta[2] * np.exp(1j * phase_list)
    vals = np.zeros_like(phase_list, dtype=complex)
    for j in range(len(phase_list)):
        vals[j] = char_poly.eval(csolver.CScalarVec([E_and_beta[0], E_and_beta[1], beta_loop[j]]))
    plt.plot(phase_list, np.abs(vals)**2)
    new_zero_points = csolver.csolver.AtlasZeroPointSegmentDeque()
    success = csolver.csolver.atlas_collect_zero_point(csolver.CScalarVec(vals), 
                                                       200, 1, 2, new_zero_points)
    for j in range(len(new_zero_points)):
        print(new_zero_points[j].N_points)
        print(new_zero_points[j].left_point)
        print(new_zero_points[j].right_point)
        print(new_zero_points[j].left_index)
        print(new_zero_points[j].right_index)
    plt.show()

    beta_loop = E_and_beta[3] * np.exp(1j * phase_list)
    for j in range(len(phase_list)):
        vals[j] = char_poly.eval(csolver.CScalarVec([E_and_beta[0], E_and_beta[1] * curr_param[0], beta_loop[j]]))
    plt.plot(phase_list, np.abs(vals)**2)
    new_zero_points = csolver.csolver.AtlasZeroPointSegmentDeque()
    success = csolver.csolver.atlas_collect_zero_point(csolver.CScalarVec(vals), 
                                                       200, 1, 2, new_zero_points)
    for j in range(len(new_zero_points)):
        print(new_zero_points[j].N_points)
        print(new_zero_points[j].left_point)
        print(new_zero_points[j].right_point)
        print(new_zero_points[j].left_index)
        print(new_zero_points[j].right_index)
    plt.show()
 
def plot_zero_point_error():
    vals_real = np.loadtxt("data/winding_vals_real.csv", delimiter=',')
    vals_imag = np.loadtxt("data/winding_vals_imag.csv", delimiter=',')
    vals = vals_real + vals_imag * 1j
    new_zero_points = csolver.csolver.AtlasZeroPointSegmentDeque()
    success = csolver.csolver.atlas_collect_zero_point(csolver.CScalarVec(vals), 
                                                       len(vals), 1, 1.99, new_zero_points)
    print("N total points:", len(new_zero_points))
    for j in range(len(new_zero_points)):
        print(new_zero_points[j].N_points)
        print(new_zero_points[j].left_point)
        print(new_zero_points[j].right_point)
        print(new_zero_points[j].left_index)
        print(new_zero_points[j].right_index)
    plt.plot(np.abs(vals) ** 2)
    plt.show()

def test_continuity():
    vx1 = 1.0 + np.random.randn() * 0.5
    vx2 = 1.0 + np.random.randn() * 0.5
    ux1 = 3.0 + np.random.randn() * 0.5
    ux2 = 3.0 + np.random.randn() * 0.5
    uy1 = 4.0 + np.random.randn() * 0.5
    uy2 = 4.0 + np.random.randn() * 0.5
    print(vx1, vx2, ux1, ux2, uy1, uy2)
    input()

    # coeffs = [
    #     1, -1, -t2*t3, -(t1*t2 + t1 * t3 + gamma * t2 / 2 - gamma * t3/2),
    #     - (t1 * t2 + t1 * t3 - gamma * t2/2 + gamma*t3/2),
    #     -(-gamma**2/4 + t1**2 + t2**2 + t3**2)
    # ]
    # orders = [
    #     2, 0,
    #     0, -2,
    #     0, 2,
    #     0, 1,
    #     0, -1,
    #     0, 0
    # ]
    coeffs = csolver.CScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0,
    ])

    solver = csolver.aGBZ2DSolver(1, 1e-1, 100, 100, 0.01)
    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)
    solver.set_char_poly_by_coeffs(coeffs, orders)
    N_params = 50
    # params = np.exp(1j*np.linspace(np.pi/201, np.pi - np.pi/201, 200)).reshape((-1,1))
    params = np.zeros((N_params, 2), dtype=complex)
    params[:,1] = np.exp(1j*np.linspace(np.pi/(N_params+1), np.pi - np.pi/(N_params+1), N_params))
    params[:,0] = np.exp(1j*pi/3)
    solver.set_param(params)

    solver_flag = True
    full_solver_flag_list = []
    while(solver_flag):
        if(solver.full_solver_flag):
            print("Full solver")
            # full solver
            full_solver_next_flag = True
            while(full_solver_next_flag):
                # print(solver.curr_poly_string_list, file=sys.stdout)
                all_sols = phcpy.solver.solve(list(solver.curr_poly_string_list), dictionary_output=True, verbose=False)
                sols_complex = []
                for sol in all_sols:
                    if(abs(sol['y'])<1.5):
                        if(abs(sol['z'])<1.5):
                            if(abs(sol['w'])<1.5):
                    # if(True):
                                res = complex_root(solver.continuation_solver_fun, np.array([sol['x'], sol['y'], sol['z'], sol['w']]), jac=True)
                                sols_complex.append(res.x)
                                if(not res.success):
                                    print(res.message)
                                    input()
                                # print(solver.continuation_solver_fun(res.x))
                # print(np.array(sols_complex))
                # print(1/np.array(sols_complex)[:,1])
                if(len(sols_complex) == 0):
                    full_solver_next_flag = solver.insert_full_solver_results(np.asarray([[0,0,0,0]], dtype=complex), True)
                else:
                    full_solver_next_flag = solver.insert_full_solver_results(np.asarray(sols_complex), False)

                # input()
                # plt.plot(np.array(sols_complex)[:,1].real, np.array(sols_complex)[:,1].imag, 'b.')
        else:
            print("Continuation solver")
            # continuation solver
            continuation_solver_next_flag = True
            while(continuation_solver_next_flag):
                # print("Continuation solver:", solver.curr_initial_values)
                # plt.plot(solver.curr_initial_values[1].real, solver.curr_initial_values[1].imag, 'b.')
                res = complex_root(solver.continuation_solver_fun, np.array(solver.curr_initial_values), jac=True)
                if(not res.success):
                    # print(solver.continuation_solver_fun(res.x))
                    # print(res.message)
                    solver.suspend_solver()
                # print(solver.continuation_solver_fun(res.x))
                
                # x0 = np.random.randn(2) + 1j*np.random.randn(2)
                # dx0 = (np.random.randn(2) + 1j*np.random.randn(2)) * 1e-3
                # x1, jac = solver.continuation_solver_fun(x0)
                # print(jac)
                # x2 = solver.continuation_solver_fun(x0 + dx0)[0]
                # print(x2 - x1)
                # print(jac@dx0)

                # print(res.success)
                # print(res.message)
                # print(solver.continuation_solver_fun(res.x)[0])
                continuation_solver_next_flag = solver.insert_continuation_solver_results(res.x.reshape((1,-1)))
                # input()

        # if(solver.solver_suspended):
        #     print("solver suspended")
        #     input()
        # print("solving finishes")
        print(solver.get_curr_param(), file=sys.stdout)
        full_solver_flag_list.append(solver.full_solver_flag)
        solver_flag = solver.next()
        # input()
        # plt.pause(0.5)
    # plt.show()
    
    # solver.post_process(0.1)
    # solver.sol_to_segments()
    # solver_flag = solver.sol_to_segments_init()
    # N_calls = 0
    # while(solver_flag):
    #     print(N_calls)
    #     GBZ_condition_check_result = csolver.check_GBZ_condition(
    #         char_poly, solver.GBZ_check_curr_point[:3], params[solver.GBZ_check_curr_ptr,:], solver.major_dim + 1,
    #         outer_N_points_loop=20, outer_N_points_detour=10, N_points_loop = 20, N_points_detour = 10
    #     )
    #     print(N_calls, "finished, ", GBZ_condition_check_result)
    #     solver_flag = solver.GBZ_check_main_loop(GBZ_condition_check_result)
    #     # solver_flag = solver.GBZ_check_main_loop(True)
    #     N_calls += 1
    # print(solver.N_loops())
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.set_xlim([-5,5])
    # ax.set_ylim([-5,5])
    # ax.set_zlim([-5,5])
    all_colors = "rgbcmy"
    N_points = 0
    for j in range(solver.N_points() - 1):
    # for j in range(solver.N_segments()):
    # for j in range(solver.N_loops()):
    # for j in range(1):
        coords, chart_labels = solver.raw_sol_readout(j)
        chart1_flags = (chart_labels[:,0]==1)
        coords[chart1_flags, 1] = 1/coords[chart1_flags, 1]
        coords_j = coords

        coords, chart_labels = solver.raw_sol_readout(j+1)
        chart1_flags = (chart_labels[:,0]==1)
        coords[chart1_flags, 1] = 1/coords[chart1_flags, 1]
        coords_j1 = coords

        print(np.column_stack([coords_j1[:,0], coords_j[:,0]]))
        print()
        print(np.column_stack([coords_j1[:,1], coords_j[:,1]]))
        print()
        print(np.column_stack([coords_j1[:,2], coords_j[:,2]]))
        print()
        print(np.column_stack([coords_j1[:,3], coords_j[:,3]]))
        print()
        input()
    # print(coords)
    # print(chart_labels)
    # print(N_calls)
    # print(N_points)

def test_winding_check():
    vx1 = 1.0 + np.random.randn() * 0
    vx2 = 1.0 + np.random.randn() * 0
    ux1 = 3.0 + np.random.randn() * 0
    ux2 = 3.0 + np.random.randn() * 0
    uy1 = 4.0 + np.random.randn() * 0
    uy2 = 4.0 + np.random.randn() * 0
    print(vx1, vx2, ux1, ux2, uy1, uy2)

    coeffs = csolver.CScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0
    ])


    GBZ_point = np.array([ 0.79325164+7.93359151e-11j,  0.09226836-9.95734176e-01j,
        -0.57777383+8.16196912e-01j, -0.57777383+8.16196912e-01j])
    params = [-0.9829731+0.18374952j, -0.3323548+0.94315443j]
    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)
    # result = csolver.check_GBZ_condition(char_poly, GBZ_point, params, 1,
    #             rel_r_detour=1e-3, outer_rel_r_detour=1e-3, N_points_loop=500)
    result = csolver.check_GBZ_condition(char_poly, GBZ_point, params, 1, outer_N_points_loop = 100,
                outer_N_points_detour=20, N_points_detour=20)
    print(result)
    input("Press any key to continue")

def test_1d_checker():
    vx1 = 1.0 + np.random.randn() * 0
    vx2 = 1.0 + np.random.randn() * 0
    ux1 = 3.0 + np.random.randn() * 0
    ux2 = 3.0 + np.random.randn() * 0
    uy1 = 4.0 + np.random.randn() * 0
    uy2 = 4.0 + np.random.randn() * 0
    print(vx1, vx2, ux1, ux2, uy1, uy2)

    coeffs = csolver.CScalarVec([
        1, - vx1, -vx2, -uy1, -uy2,
        vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
        vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    ])
    orders = csolver.CIntVec([
        # E, beta1, beta2
        2, 0, 0,
        1, 0, -1,
        1, 0, 1,
        1, 0, -1,
        1, 0, 1,
        0, 0, 2,
        0, 0, -2,
        0, -1, 0,
        0, 1, 0,
        0, 0, 0
    ])


    GBZ_point = np.array([-0.12772926-1.38426957e-10j,  0.75940492-6.50618300e-01j,
                  0.91420745+4.05246505e-01j,  0.91420745+4.05246505e-01j])
    # print(np.abs(GBZ_point))
    # input()
    params = [0.15339165+0.98816547j, -0.952942  +0.30315267j]
    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)
    
    # E_val = -0.12772926 -1.38426957e-10j
    # major_beta = 0.75940492-6.50618300e-01j
    # beta  = 0.91420745+4.05246505e-01j
    x = [0.11263782+2.52569303e-10j, 0.99699551-7.74593996e-02j]
    beta_var = (0.798948993178733-0.6013987954043352j)
    checker = csolver.csolver.GBZCondition1DChecker(char_poly, 1)
    # for j in range(500):
    print(checker.GBZ_condition_check(x[0], beta_var, x[1], 1e-2*np.pi, 200, 50, 1e-3))

def test_winding_loops():
    beta1 = 3*exp(1j*pi/2)
    expiphi = exp(1j*pi/3)
    rel_r_detour = 0.05
    path_12 = csolver.ScalarVec([])
    path_21 = csolver.ScalarVec([])
    circ_1 = csolver.ScalarVec([])
    circ_2 = csolver.ScalarVec([])
    inner_circ_1 = csolver.ScalarVec([])
    inner_circ_2 = csolver.ScalarVec([])
    csolver.winding_loop_gen(beta1, expiphi, rel_r_detour, 100, 20, 
                            path_12, circ_2, inner_circ_2, path_21, circ_1, inner_circ_1)

    plt.plot(np.array(path_12[:]).real, np.array(path_12[:]).imag)
    plt.plot(np.array(path_21[:]).real, np.array(path_21[:]).imag)
    plt.plot(np.array(circ_1[:]).real, np.array(circ_1[:]).imag)
    plt.plot(np.array(circ_2[:]).real, np.array(circ_2[:]).imag)
    plt.plot(np.array(inner_circ_1[:]).real, np.array(inner_circ_1[:]).imag)
    plt.plot(np.array(inner_circ_2[:]).real, np.array(inner_circ_2[:]).imag)
    plt.show()

def test_winding_number():
    theta = np.linspace(0, -4*pi, 1000)
    complex_winding = csolver.ScalarVec(np.exp(1j*theta)*(np.cos(theta)+1.01))
    print(csolver.winding_number(complex_winding))

if __name__ == "__main__":
    # test_continuity()
    test_basics()
    # test_char_poly_Laurant()
    # test_winding_loops()
    # test_winding_number()
    # sweep_2D_GBZ(0)
    # sweep_2D_GBZ_y_major(0)
    # test_y_major()
    # cProfile.run("test_basics()")
    # test_winding_check()
    # test_1d_checker()
    # test_zero_point_selection()
    # plot_zero_point_error()

    