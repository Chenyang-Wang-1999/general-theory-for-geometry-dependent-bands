'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-08-29
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from solver_tools import *
import c_solver_2D as csolver
import phcpy
from math import sin, cos, sqrt, pi
from cmath import exp, log
import pickle
import sys
from scipy import io as sio
from scipy import linalg as la
import os

def solve_GBZ(char_poly:csolver.CLaurant, major_dim:int, params:np.ndarray, critical_distance:float=0.1, 
               N_loop_points:int = 200, N_detour_points=50, theta_detour:float = 1e-2*np.pi, 
               outer_N_loop_points:int=200, outer_N_detour_points=50, outer_theta_detour:float=1e-2*np.pi, zero_val_th = 1e-3):
    solver = csolver.aGBZ2DSolver(major_dim, critical_distance, N_loop_points, N_detour_points, theta_detour, zero_val_th)
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
                                if(not res.success):
                                    print(res.message)
                                    # input()
                                    # assert(0)
                                    sols_complex.append(np.array([sol['x'], sol['y'], sol['z'], sol['w']]))
                                else:
                                    sols_complex.append(res.x)
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
            outer_theta_detour, outer_N_loop_points, outer_N_detour_points, theta_detour, 
            N_loop_points, N_detour_points, zero_val_th
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
    
    del solver
    return all_segments


def sweep_2D_GBZ_x_major(data_folder, char_poly, phi1_list, phi2_list, offset:int, N_points:int):
    print("x major, %d to %d"%(offset, offset + N_points))
    if(not os.path.exists(data_folder)):
        os.mkdir(data_folder)

    for j, phi2 in enumerate(phi2_list[offset:(offset + N_points)]):
        print("current phi2 = %.2f pi"%(phi2/pi), "%.2f%%"%(j/N_points*100))
        params = np.zeros((len(phi1_list), 2), dtype=complex)
        params[:,0] = np.exp(1j*phi1_list)
        params[:,1] = np.exp(1j*phi2)
        curr_segments = solve_GBZ(char_poly, 0, params)

        with open("%s/x_major_%d.pkl"%(data_folder, j+offset), "wb") as fp:
            pickle.dump(curr_segments, fp)
    
def sweep_2D_GBZ_y_major(data_folder, char_poly, phi1_list, phi2_list, offset:int, N_points:int):
    print("y major, %d to %d"%(offset, offset + N_points))
    if(not os.path.exists(data_folder)):
        os.mkdir(data_folder)

    for j, phi1 in enumerate(phi1_list[offset:(offset + N_points)]):
        print("current phi1 = %.2f pi"%(phi1/pi), "%.2f%%"%(j/N_points*100))
        params = np.zeros((len(phi2_list), 2), dtype=complex)
        params[:,0] = np.exp(1j*phi1)
        params[:,1] = np.exp(1j*phi2_list)
        curr_segments = solve_GBZ(char_poly, 1, params)

        with open("%s/y_major_%d.pkl"%(data_folder, j+offset), "wb") as fp:
            pickle.dump(curr_segments, fp)

def main_sweep_2D_GBZ(data_id:int, offset:int, N_points):
    # vx1 = 1.0 + np.random.randn() * 0
    # vx2 = 1.0 + np.random.randn() * 0
    # ux1 = 3.0 + np.random.randn() * 0
    # ux2 = 3.0 + np.random.randn() * 0
    # uy1 = 4.0 + np.random.randn() * 0
    # uy2 = 4.0 + np.random.randn() * 0
    # coeffs = csolver.CScalarVec([
    #     1, - vx1, -vx2, -uy1, -uy2,
    #     vx2 * uy2, uy1 * vx1, - ux1 * vx1, - ux2 * vx2,
    #     vx2 * uy1 + vx1 * uy2 - ux1 * ux2 - vx1 * vx2
    # ])
    # orders = csolver.CIntVec([
        # E, beta1, beta2
    #     2, 0, 0,
    #     1, 0, -1,
    #     1, 0, 1,
    #     1, 0, -1,
    #     1, 0, 1,
    #     0, 0, 2,
    #     0, 0, -2,
    #     0, -1, 0,
    #     0, 1, 0,
    #     0, 0, 0,
    # ])


    char_poly_data = sio.loadmat("data/GBZ_char_poly_%d.mat"%(data_id))
    coeffs = csolver.CScalarVec(char_poly_data['coeffs'].flatten())
    orders = csolver.CIntVec(char_poly_data['orders'].flatten())
    char_poly = csolver.CLaurant(3)
    char_poly.set_Laurant_by_terms(coeffs, orders)

    N_phi1 = 50
    N_phi2 = 50
    phi1_list = np.linspace(np.pi/(N_phi1+1), np.pi - np.pi/(N_phi1+1), N_phi1)
    phi2_list = np.linspace(np.pi/(N_phi2+1), np.pi - np.pi/(N_phi2+1), N_phi2)

    if(offset == 0):
        data = sio.loadmat("data/GBZ_info_%d.mat"%(data_id))
        # data['model_params'] = (ux1, ux2, uy1, uy2, vx1, vx2)
        data['phi1_list'] = phi1_list
        data['phi2_list'] = phi2_list
        sio.savemat("data/GBZ_info_%d.mat"%(data_id), data)
    
    sweep_2D_GBZ_x_major("data/GBZ_data_%d"%(data_id), char_poly, phi1_list, phi2_list, offset, N_points)
    sweep_2D_GBZ_y_major("data/GBZ_data_%d"%(data_id), char_poly, phi1_list, phi2_list, offset, N_points)
 
import sys
if __name__ == "__main__":
    # print(sys.argv[1])
    offset = int(sys.argv[1])
    main_sweep_2D_GBZ(data_id=7, offset=offset, N_points=1)
