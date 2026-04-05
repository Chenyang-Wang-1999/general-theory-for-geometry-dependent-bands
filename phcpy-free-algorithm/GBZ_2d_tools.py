'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-08-29 00:00:00
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import c_solver_2D as csolver
import brute_force_solver as bfs
from brute_force_solver import complex_root
import phcpy
from math import sin, cos, sqrt, pi
from cmath import exp, log
import pickle
import sys
from scipy import io as sio
from scipy import linalg as la
import os

def solve_GBZ(char_poly:csolver.CLaurent, major_dim:int, params:np.ndarray, critical_distance:float=0.1, 
               N_loop_points:int = 200, N_detour_points=50, theta_detour:float = 1e-2*np.pi, 
               zero_val_th = 1e-3):
    solver = csolver.aGBZ2DSolver(major_dim, critical_distance, N_loop_points, N_detour_points, theta_detour, zero_val_th)
    checker = bfs.SGBZChecker(char_poly)
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
                all_sols = phcpy.solver.solve(list(solver.curr_poly_string_list), dictionary_output=True)
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
        curr_E = solver.GBZ_check_curr_point[0]
        curr_mu1 = log(abs(solver.GBZ_check_curr_point[1]))
        GBZ_condition_check_result = checker.check_for_E_and_mu1(curr_E, curr_mu1, N_points=N_loop_points)
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
        try:
            curr_segments = solve_GBZ(char_poly, 0, params)

            with open("%s/x_major_%d.pkl"%(data_folder, j+offset), "wb") as fp:
                pickle.dump(curr_segments, fp)
        except Exception as e:
            with open("%s/x_major_%d.log"%(data_folder, j+offset), "w") as fp:
                print(e, file=fp)
