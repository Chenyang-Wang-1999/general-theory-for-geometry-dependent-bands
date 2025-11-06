'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-07-20
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from solver_tools import *
import c_aGBZ_1D_solver as csolver
import phcpy
from math import sin, cos, sqrt, pi
from cmath import exp, log
import matplotlib.pyplot as plt
import sys

class aGBZ1DSolver(csolver.aGBZ1DSolver):
    def __init__(self, critical_distance:float, N_loop_points:int,
                N_detour:int, rel_r_detour:float):
        super().__init__(critical_distance, N_loop_points, N_detour, rel_r_detour)

    def set_char_poly_by_coeffs(self, coeffs:list[complex], orders:list[int]):
        super().set_char_poly_by_coeffs(csolver.ScalarVec(coeffs), csolver.IntVec(orders))
    
    def continuation_solver_fun(self, x0):
        all_data = super().continuation_solver_fun(x0)
        return all_data[0,:], all_data[1:,:]

    def segment_readout(self, segment_id:int):
        loop_coords = csolver.ScalarVec([])
        loop_chart_labels = csolver.ChartIdVec([])
        super().segment_readout(segment_id, loop_coords, loop_chart_labels)
        return np.asarray(loop_coords).reshape((-1,2)), np.asarray(loop_chart_labels)

    def raw_sol_readout(self, sol_id:int):
        loop_coords = csolver.ScalarVec([])
        loop_chart_labels = csolver.ChartIdVec([])
        super().raw_sol_readout(sol_id, loop_coords, loop_chart_labels)
        # print(loop_coords, loop_chart_labels)
        return np.asarray(loop_coords).reshape((-1,2)), np.asarray(loop_chart_labels)

    def loop_readout(self, loop_id:int):
        loop_coords = csolver.ScalarVec([])
        loop_chart_labels = csolver.ChartIdVec([])
        super().loop_readout(loop_id, loop_coords, loop_chart_labels)
        return np.asarray(loop_coords).reshape((-1,2)), np.asarray(loop_chart_labels)

def test_diff():
    pass

def test_char_poly(x_arr, expiphi, chart_labels):
    x = x_arr[0]
    if(chart_labels == 0):
        y = x_arr[1]
    else:
        y = 1/x_arr[1]
    results = np.zeros(2,dtype=complex)
    results[0] = +(-0.5+0*1j)+(-1.44999999999999996+0*1j)*y+(-2.24000000000000021+0*1j)*(y**2)+(-1.55000000000000004+0*1j)*(y**3)+(-0.5+0*1j)*(y**4)+(1+0*1j)*(x**2)*(y**2)
    y *= expiphi
    results[1] = +(-0.5+0*1j)+(-1.44999999999999996+0*1j)*y+(-2.24000000000000021+0*1j)*(y**2)+(-1.55000000000000004+0*1j)*(y**3)+(-0.5+0*1j)*(y**4)+(1+0*1j)*(x**2)*(y**2)
    return results

def test_basics():
    t1 = 1
    t2 = 1
    t3 = 0.5
    gamma = 0.2

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
    coeffs = [
        -t2*t3, - (t1*(t2 + t3) + gamma * (t3 - t2)/2), 
        -(-gamma**2/4 + t1**2 + t2**2 + t3**2), 
        -(t1*(t2 + t3) + gamma * (t2 - t3)/2), 
        -t2 * t3, 1
    ]
    orders = [
        0, -2,  
        0, -1,
        0, 0,
        0, 1,
        0, 2,
        2, 0
    ]

    solver = aGBZ1DSolver(1e-2, 100, 100, 0.01)
    solver.set_char_poly_by_coeffs(coeffs, orders)
    params = np.exp(1j*np.linspace(np.pi/201, np.pi - np.pi/201, 200)).reshape((-1,1))
    plt.plot(params.real, params.imag)
    plt.show()
    solver.set_param(params)

    solver_flag = True
    full_solver_flag_list = []
    while(solver_flag):
        if(solver.full_solver_flag):
            # print("Full solver")
            # full solver
            full_solver_next_flag = True
            while(full_solver_next_flag):
                print(solver.curr_poly_string_list, file=sys.stdout)
                all_sols = phcpy.solver.solve(list(solver.curr_poly_string_list), dictionary_output=True, verbose=False)
                sols_complex = []
                for sol in all_sols:
                    if(abs(sol['y'])<1.5):
                    # if(True):
                        res = complex_root(solver.continuation_solver_fun, np.array([sol['x'], sol['y']]), jac=True)
                        sols_complex.append(res.x)
                print(np.array(sols_complex))
                # print(1/np.array(sols_complex)[:,1])
                full_solver_next_flag = solver.insert_full_solver_results(np.array(sols_complex))
                # input()
                # plt.plot(np.array(sols_complex)[:,1].real, np.array(sols_complex)[:,1].imag, 'b.')
        else:
            # print("Continuation solver")
            # continuation solver
            continuation_solver_next_flag = True
            while(continuation_solver_next_flag):
                # print("Continuation solver:", solver.curr_initial_values)
                # plt.plot(solver.curr_initial_values[1].real, solver.curr_initial_values[1].imag, 'b.')
                res = complex_root(solver.continuation_solver_fun, np.array(solver.curr_initial_values), jac=True)
                
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

        if(solver.solver_suspended):
            print("solver suspended")
            input()
        # print("solving finishes")
        print(solver.get_curr_param()[0], file=sys.stdout)
        full_solver_flag_list.append(solver.full_solver_flag)
        solver_flag = solver.next()
        # input()
        # plt.pause(0.5)
    # plt.show()
    
    solver.post_process(0.1)
    # # solver.sol_to_segments()
    # print(solver.N_loops())
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # for j in range(solver.N_points()):
    # for j in range(solver.N_segments()):
    for j in range(2):
        coords, chart_labels = solver.loop_readout(j)
        # coords, chart_labels = solver.raw_sol_readout(j)
        # print(j, full_solver_flag_list[j])
        # for k in range(coords.shape[0]):
            # print("Coord:", coords[k,:])
            # print("chart labels:", chart_labels[k])
            # print("Polynomial values:", test_char_poly(coords[k,:], params[j], chart_labels[k]))
            # if(chart_labels[k] == 0):
            #     plt.plot(coords[k,1].real, coords[k,1].imag, '.')
            #     plt.plot((coords[k,1]*params[j]).real, (coords[k,1]*params[j]).imag, 'b.')
            # else:
            #     plt.plot((1/coords[k,1]).real, (1/coords[k,1]).imag, 'r.')
            #     plt.plot((params[j]/coords[k,1]).real, (params[j]/coords[k,1]).imag, 'r.')

            


        # print(coords.shape, chart_labels.shape)
        chart1_flags = (chart_labels==1)
        coords[chart1_flags] = 1/coords[chart1_flags]
        ax.plot(coords[:,1].real, coords[:,1].imag, coords[:,0].real)
        plt.pause(0.5)
        # plt.plot((coords[:,1]*params[j]).real, (coords[:,1]*params[j]).imag, '.')
    plt.show()
    # print(coords)
    # print(chart_labels)

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
    test_basics()
    # test_winding_loops()
    # test_winding_number()