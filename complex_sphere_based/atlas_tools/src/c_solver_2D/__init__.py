'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-08-18
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import numpy_extended as npe
from scipy import io as sio
from scipy import optimize
from scipy import linalg as la
import c_solver_2D.c_aGBZ_2D_solver as csolver
import sys
import phcpy.solver as phcpy_solver
import matplotlib.pyplot as plt
import sympy as sym


class ComplexEqConverter:
    def __init__(self, fun, cdim) -> None:
        self.fun = fun
        self.cdim = cdim
    
    def __call__(self, x_re, *args):
        x = x_re[:self.cdim] + 1j * x_re[self.cdim:]
        eq_LHS, eq_jac = self.fun(x, *args)

        # convert to real equation
        eq_LHS_re = np.zeros(2*self.cdim)
        eq_LHS_re[:self.cdim] = eq_LHS.real
        eq_LHS_re[self.cdim:] = eq_LHS.imag

        eq_jac_re = np.zeros((2*self.cdim, 2*self.cdim))
        eq_jac_re[:self.cdim, :self.cdim] = eq_jac.real 
        eq_jac_re[:self.cdim, self.cdim:] = - eq_jac.imag
        eq_jac_re[self.cdim:, :self.cdim] = eq_jac.imag
        eq_jac_re[self.cdim:, self.cdim:] = eq_jac.real
        return eq_LHS_re, eq_jac_re

def complex_root(complex_fun, x0, **options):
    fun = ComplexEqConverter(complex_fun, len(x0))
    x0_re = np.zeros(2*fun.cdim)
    x0_re[:fun.cdim] = x0.real
    x0_re[fun.cdim:] = x0.imag
    res= optimize.root(fun, x0_re, **options)
    x_re = res.x
    # jac_re = res.jac
    res.x = x_re[:fun.cdim] + 1j * x_re[fun.cdim:]
    if('args' in options.keys()):
        val, jac = complex_fun(x0,*options['args'])
    else:
        val, jac = complex_fun(x0)
    res.jac = jac
    return res

def winding_number(x):
    '''
        calculate complex winding number of x
    '''
    x = np.asarray(x)
    phase_diff = np.zeros_like(x, dtype=float)
    phase_diff[:-1] = np.log(x[1:]/x[:-1]).imag
    phase_diff[-1] = np.log(x[0]/x[-1]).imag
    # plt.plot(np.abs(phase_diff)/np.pi)
    # plt.plot(np.abs(phase_diff)/np.pi, '.')
    # plt.show()
    return int(np.round((np.sum(phase_diff)/(2*np.pi))))

def winding_phase(start_phase:float, end_phase:float, 
                  radius:float, N_points:int, dphase_th:float, 
                  fun):
    if(end_phase <= start_phase):
        end_phase += 2 * np.pi
    
    # 1. generate the points
    phase_vec = np.linspace(start_phase, end_phase, N_points)
    beta_path = np.exp(1j* phase_vec) * radius

    # 2. evaluate along the points and params
    vals = fun(beta_path)

    # 3. calculate winding phase
    total_phase = 0.0
    for point_id in range(len(beta_path) - 1):
        dphase = np.log((vals[point_id + 1])/ vals[point_id]).imag
        if(abs(dphase) < dphase_th):
            total_phase += dphase
        else:
            N_split = int(np.round(abs(dphase)/dphase_th)) + 3
            total_phase += winding_phase(phase_vec[point_id], phase_vec[point_id + 1], radius, N_split, 
                                         dphase_th, fun)
    
    return total_phase

def winding_phase_with_param(start_phase:float, end_phase:float, 
                  radius:float, N_points:int, dphase_th:float, 
                  start_param, fun, recursion_depth:int = 0):
    if(end_phase <= start_phase):
        end_phase += 2 * np.pi
    
    # print("From:", start_phase, ", to:", end_phase, ", N_points:", N_points)
    # 1. generate the points
    phase_vec = np.linspace(start_phase, end_phase, N_points)
    beta_path = np.exp(1j* phase_vec) * radius

    # 2. evaluate along the points and params
    return_result = fun(beta_path, start_param)
    if(return_result is None):
        return 0, 0, False
    vals, param_list = return_result
    # print(vals)
    # for item in param_list:
    #     print(item)
    #     print(np.abs(item))
    # input()
    if(recursion_depth > 20):
        return 0, 0, False

    # 3. calculate winding phase
    total_phase = 0.0
    for point_id in range(len(beta_path) - 1):
        dphase = np.log((vals[point_id + 1])/ vals[point_id]).imag
        if(abs(dphase) < dphase_th):
            total_phase += dphase
        else:
            N_split = int(np.round(abs(dphase)/dphase_th)) + 3
            next_result = winding_phase_with_param(phase_vec[point_id], phase_vec[point_id + 1], radius, N_split, 
                                         dphase_th, param_list[point_id], fun, recursion_depth + 1)
            if(not next_result[2]):
                return 0.0, 0.0, False
            else:
                total_phase += next_result[0]
    
    return total_phase, param_list[-1], True


def eval_for_point_path_check(beta:complex, zero_point:csolver.AtlasZeroPointSegment, 
                   points_outer_loop:list, points_inner_loop:list, N_detour:int, rel_r_detour:float,
                   remaining_phase:float, N_loop:int, initial_param, fun):
    if(zero_point.N_points == 1):
        beta_center = beta * np.exp(2.0j * np.pi * (zero_point.left_point)/(N_loop - 1))
        outer_path = CScalarVec([])
        inner_path = CScalarVec([])
        csolver.atlas_bubble_gen(beta_center, rel_r_detour, remaining_phase, N_detour, outer_path, inner_path)

        points_outer_loop += list(outer_path)
        points_inner_loop += list(inner_path)
    else:
        beta_pair = [
            beta * np.exp(2.0j * np.pi * (zero_point.left_point)/(N_loop - 1)),
            beta * np.exp(2.0j * np.pi * (zero_point.right_point)/(N_loop - 1))
        ]
        outer_begin = CScalarVec([])
        inner_begin = CScalarVec([])
        outer_arc = CScalarVec([])
        inner_arc = CScalarVec([])
        outer_end = CScalarVec([])
        inner_end = CScalarVec([])
        csolver.atlas_long_bubble_gen(beta_pair, rel_r_detour, remaining_phase, N_detour//2, 
            N_detour * (zero_point.N_points-1), outer_begin, inner_begin,
            outer_arc, inner_arc, outer_end, inner_end)
        
        # get results
        points_outer_loop += list(outer_begin)
        points_outer_loop += list(outer_arc)
        points_outer_loop += list(outer_end)

        points_inner_loop += list(inner_begin)
        points_inner_loop += list(inner_arc)
        points_inner_loop += list(inner_end)

    return True

def winding_phase_auto_detour_path_check(
        beta0:complex, theta_detour:float, zero_eps:float,
        N_points_loop:int, N_points_detour:int, fun, 
        initial_param, continuation_check_tol:float):
    # 0. parameters
    rel_r_detour = 2 * np.sin(theta_detour)
    indices_detour = theta_detour/(np.pi/(N_points_loop-1))

    # 1. generate circular winding loop
    circular_loop = beta0 * np.exp(1j * np.linspace(0, 2 * np.pi, N_points_loop))
    circular_data = fun(circular_loop, initial_param)
    if(circular_data is None):
        return False
    circular_vals, circular_params = circular_data

    # 1.2 Closedness check
    MAX_WINDING_LOOPS = 1000
    N_winding_loops = 1
    for j in range(MAX_WINDING_LOOPS):
        print(j)
        if(la.norm(circular_params[0] - circular_params[-1]) > continuation_check_tol * la.norm(circular_params[0])):
            circular_data = fun(circular_loop, circular_params[-1])
            print(j, "finishes")
            if(circular_data is None):
                assert(0)
            curr_vals, curr_params = circular_data
            circular_vals += curr_vals[1:]
            circular_params += curr_params[1:]
            N_winding_loops += 1
        else:
            break

        if(j == MAX_WINDING_LOOPS - 1):
            print("Warning: Closedness check failed!", file=sys.stderr)
            return False

    # 2. collect zero points
    N_points_loop_full = (N_points_loop - 1) * N_winding_loops + 1
    circular_loop_full = beta0 * np.exp(1j * np.linspace(0, 2 * np.pi * N_winding_loops, N_points_loop_full))
    new_zero_points = csolver.AtlasZeroPointSegmentDeque()
    success = csolver.atlas_collect_zero_point(CScalarVec(circular_vals), N_points_loop_full, zero_eps, indices_detour, new_zero_points)
    if(not success):
        return False

    # 3. get winding values
    outer_points = []
    inner_points = []
    head_point = new_zero_points[0]
    if(head_point.left_index < 0):
        begin_ptr = head_point.right_index
        end_ptr = head_point.left_index + N_points_loop_full
        result = eval_for_point_path_check(beta0, head_point, outer_points, inner_points, N_points_detour, 
                    rel_r_detour, theta_detour, N_points_loop,
                    circular_params[head_point.left_index + N_points_loop_full],
                    fun)
        new_zero_points.pop_front()
    else:
        tail_point = new_zero_points[len(new_zero_points) - 1]
        if(tail_point.right_point > (N_points_loop_full - 1)):
            end_ptr = tail_point.left_index + 1
            begin_ptr = tail_point.right_index + 1 - N_points_loop
            result = eval_for_point_path_check(beta0, tail_point, outer_points, inner_points, N_points_detour,
                rel_r_detour, theta_detour, N_points_loop_full, circular_params[tail_point.left_index], fun)
        else:
            end_ptr = N_points_loop_full
            begin_ptr = 0
    
    curr_left_ptr = begin_ptr
    if(len(new_zero_points)):
        for j in range(len(new_zero_points)):
            point = new_zero_points[j]
            curr_right_ptr = point.left_index
            outer_points += list(circular_loop_full[curr_left_ptr:curr_right_ptr + 1])
            inner_points += list(circular_loop_full[curr_left_ptr:curr_right_ptr + 1])
            result = eval_for_point_path_check(beta0, point, outer_points, inner_points, N_points_detour,
                rel_r_detour, theta_detour, N_points_loop, circular_params[point.left_index], fun)
            curr_left_ptr = point.right_index
    if(curr_left_ptr < end_ptr):
        outer_points += list(circular_loop_full[curr_left_ptr:end_ptr])
        inner_points += list(circular_loop_full[curr_left_ptr:end_ptr])
    
    return outer_points, inner_points


def eval_for_point(beta:complex, zero_point:csolver.AtlasZeroPointSegment, 
                   vals_outer_loop:list, vals_inner_loop:list, N_detour:int, rel_r_detour:float,
                   remaining_phase:float, N_loop:int, initial_param, fun):
    if(zero_point.N_points == 1):
        beta_center = beta * np.exp(2.0j * np.pi * (zero_point.left_point)/(N_loop - 1))
        outer_path = CScalarVec([])
        inner_path = CScalarVec([])
        csolver.atlas_bubble_gen(beta_center, rel_r_detour, remaining_phase, N_detour, outer_path, inner_path)
        outer_result = fun(outer_path, initial_param)
        if(outer_result is None):
            return False
        outer_vals, outer_params = outer_result
        vals_outer_loop += outer_vals

        inner_result = fun(inner_path, initial_param)
        if(inner_result is None):
            return False
        inner_vals, inner_params = inner_result
        vals_inner_loop += inner_vals
    else:
        beta_pair = [
            beta * np.exp(2.0j * np.pi * (zero_point.left_point)/(N_loop - 1)),
            beta * np.exp(2.0j * np.pi * (zero_point.right_point)/(N_loop - 1))
        ]
        outer_begin = CScalarVec([])
        inner_begin = CScalarVec([])
        outer_arc = CScalarVec([])
        inner_arc = CScalarVec([])
        outer_end = CScalarVec([])
        inner_end = CScalarVec([])
        csolver.atlas_long_bubble_gen(beta_pair, rel_r_detour, remaining_phase, N_detour//2, 
            N_detour * (zero_point.N_points-1), outer_begin, inner_begin,
            outer_arc, inner_arc, outer_end, inner_end)
        
        # get results
        result = fun(outer_begin, initial_param)
        if(result is None):
            return False
        vals, params = result
        vals_outer_loop += vals

        result = fun(outer_arc, params[-1])
        if(result is None):
            return False
        vals, params = result
        vals_outer_loop += vals

        result = fun(outer_end, params[-1])
        if(result is None):
            return False
        vals, params = result
        vals_outer_loop += vals
    
        result = fun(inner_begin, initial_param)
        if(result is None):
            return False
        vals, params = result
        vals_inner_loop += vals

        result = fun(inner_arc, params[-1])
        if(result is None):
            return False
        vals, params = result
        vals_inner_loop += vals

        result = fun(inner_end, params[-1])
        if(result is None):
            return False
        vals, params = result
        vals_inner_loop += vals


    return True

def winding_phase_auto_detour(
        beta0:complex, theta_detour:float, zero_eps:float,
        N_points_loop:int, N_points_detour:int, fun, 
        initial_param, continuation_check_tol:float):
    # 0. parameters
    rel_r_detour = 2 * np.sin(theta_detour)
    indices_detour = theta_detour/(np.pi/(N_points_loop-1))

    # 1. generate circular winding loop
    circular_loop = beta0 * np.exp(1j * np.linspace(0, 2 * np.pi, N_points_loop))
    circular_data = fun(circular_loop, initial_param)
    if(circular_data is None):
        return False
    circular_vals, circular_params = circular_data

    # 1.2 Closedness check
    MAX_WINDING_LOOPS = 30
    N_winding_loops = 1
    for j in range(MAX_WINDING_LOOPS):
        if(la.norm(circular_params[0] - circular_params[-1]) > continuation_check_tol * la.norm(circular_params[0])):
            # print(j)
            # print("difference:", la.norm(circular_params[0] - circular_params[-1]))
            # print("norm of params", la.norm(circular_params[0]))
            circular_data = fun(circular_loop, circular_params[-1])
            if(circular_data is None):
                return False
            curr_vals, curr_params = circular_data
            circular_vals += curr_vals[1:]
            circular_params += curr_params[1:]
            N_winding_loops += 1
        else:
            break

        if(j == MAX_WINDING_LOOPS - 1):
            print("Warning: Closedness check failed!", file=sys.stderr)
            return False

    # 2. collect zero points
    N_points_loop_full = (N_points_loop - 1) * N_winding_loops + 1
    new_zero_points = csolver.AtlasZeroPointSegmentDeque()
    success = csolver.atlas_collect_zero_point(CScalarVec(circular_vals), N_points_loop_full, zero_eps, indices_detour, new_zero_points)
    if(not success):
        return False

    # 3. get winding values
    outer_vals = []
    inner_vals = []
    head_point = new_zero_points[0]
    if(head_point.left_index < 0):
        begin_ptr = head_point.right_index
        end_ptr = head_point.left_index + N_points_loop_full
        result = eval_for_point(beta0, head_point, outer_vals, inner_vals, N_points_detour, 
                    rel_r_detour, theta_detour, N_points_loop,
                    circular_params[head_point.left_index + N_points_loop_full],
                    fun)
        if(not result):
            return False
        new_zero_points.pop_front()
    else:
        tail_point = new_zero_points[len(new_zero_points) - 1]
        if(tail_point.right_point > (N_points_loop_full - 1)):
            end_ptr = tail_point.left_index + 1
            begin_ptr = tail_point.right_index + 1 - N_points_loop_full
            result = eval_for_point(beta0, tail_point, outer_vals, inner_vals, N_points_detour,
                rel_r_detour, theta_detour, N_points_loop, circular_params[tail_point.left_index], fun)
            if(not result):
                return False
        else:
            end_ptr = N_points_loop_full
            begin_ptr = 0
    
    curr_left_ptr = begin_ptr
    if(len(new_zero_points)):
        for j in range(len(new_zero_points)):
            point = new_zero_points[j]
            curr_right_ptr = point.left_index
            outer_vals += circular_vals[curr_left_ptr:curr_right_ptr + 1]
            inner_vals += circular_vals[curr_left_ptr:curr_right_ptr + 1]
            result = eval_for_point(beta0, point, outer_vals, inner_vals, N_points_detour,
                rel_r_detour, theta_detour, N_points_loop, circular_params[point.left_index], fun)
            if(not result):
                return False
            curr_left_ptr = point.right_index
    if(curr_left_ptr < end_ptr):
        outer_vals += circular_vals[curr_left_ptr:end_ptr]
        inner_vals += circular_vals[curr_left_ptr:end_ptr]

    outer_winding = winding_number(outer_vals)
    inner_winding = winding_number(inner_vals)
    # print(outer_winding, inner_winding)
    if(outer_winding > 0 and inner_winding < 0):
        return True
    else:
        return False


class aGBZ2DSolver(csolver.aGBZ2DSolver):
    def __init__(self, major_dim:int, critical_distance:float, N_loop_points:int,
                 N_detour_points, detour_phase_change, zero_val_th = 1e-3):
        super().__init__(major_dim, critical_distance, N_loop_points, N_detour_points, detour_phase_change, zero_val_th)

    def set_char_poly_by_coeffs(self, coeffs:list[complex], orders:list[int]):
        super().set_char_poly_by_coeffs(CScalarVec(coeffs), CIntVec(orders))
    
    def continuation_solver_fun(self, x0):
        all_data = super().continuation_solver_fun(x0)
        return all_data[0,:], all_data[1:,:]

    def segment_readout(self, segment_id:int):
        loop_coords = CScalarVec([])
        loop_chart_labels = CChartIdVec([])
        super().segment_readout(segment_id, loop_coords, loop_chart_labels)
        return np.asarray(loop_coords).reshape((-1,3)), np.asarray(loop_chart_labels).reshape((-1,2))

    def raw_sol_readout(self, sol_id:int):
        loop_coords = CScalarVec([])
        loop_chart_labels = CChartIdVec([])
        super().raw_sol_readout(sol_id, loop_coords, loop_chart_labels)
        return np.asarray(loop_coords).reshape((-1,4)), np.asarray(loop_chart_labels).reshape((-1,3))

    def loop_readout(self, loop_id:int):
        loop_coords = CScalarVec([])
        loop_chart_labels = CChartIdVec([])
        super().loop_readout(loop_id, loop_coords, loop_chart_labels)
        return np.asarray(loop_coords).reshape((-1,3)), np.asarray(loop_chart_labels).reshape((-1,2))

'''
    check 2D GBZ conditions
    given: point to check, parameters(expiphi1, expiphi2), characteristic polynomial
    algorithm:
        1. calculate the GBZ around a given segment
        2. evaluate the polynomial along the segment
'''

class CScalarVec(csolver.aGBZ_2D_solver_ScalarVec):
    def __init__(self, vec):
        super().__init__(vec)

class CComplexArrVec(csolver.aGBZ_2D_solver_ComplexArrVec):
    def __init__(self, vec):
        super().__init__(vec)

class CIntVec(csolver.aGBZ_2D_solver_IntVec):
    def __init__(self, vec):
        super().__init__(vec)

class CStrVec(csolver.aGBZ_2D_solver_StrVec):
    def __init__(self, vec):
        super().__init__(vec)

class CChartIdVec(csolver.aGBZ_2D_solver_ChartIdVec):
    def __init__(self, vec):
        super().__init__(vec)

class CIndexVec(csolver.aGBZ_2D_solver_IndexVec):
    def __init__(self, vec):
        super().__init__(vec)

class CPolyLinkedList(csolver.aGBZ_2D_solver_PolyLinkedList):
    def __init__(self, dim):
        super().__init__(dim)

    def reinit(self, dim):
        super().reinit(dim)
    
    def remove_zeros(self):
        super().remove_zeros()

    def copy_to(self, another):
        super().copy_to(another)

    def print_info(self):
        super().print_info()
    
    def destructive_add_self(self, another):
        super().destructive_add_self(another)

    def destructive_add(self, another):
        result = CPolyLinkedList(self.dim)
        super().destructive_add(another, result)
        return result

    def add_self(self, another):
        super().add_self(another)

    def add(self, another):
        result = CPolyLinkedList(self.dim)
        super().add(another, result)
        return result

    def scalar_mul_self(self, k):
        super().scalar_mul_self(k)
    
    def scalar_mul(self, k):
        result = CPolyLinkedList(self.dim)
        super().scalar_mul(k, result)
        return result

    def scale_var_self(self, k_list):
        if(not isinstance(k_list, CScalarVec)):
            k_list = CScalarVec(k_list)
        super().scale_var_self(k_list)

    def scale_var(self, k_list):
        if(not isinstance(k_list, CScalarVec)):
            k_list = CScalarVec(k_list)
        new_poly = CPolyLinkedList(self.dim)
        super().scale_var(k_list, new_poly)
        return new_poly

    def neg_self(self):
        super().neg_self()

    def neg(self):
        result = CPolyLinkedList(self.dim)
        super().neg(result)
        return result

    def destructive_subs_self(self, another):
        super().destructive_subs_self(another)

    def subs_self(self, another):
        super().subs_self(another)

    def destructive_subs(self, another):
        result = CPolyLinkedList(self.dim)
        super().destructive_subs(another, result)
        return result
    
    def subs(self, another):
        result = CPolyLinkedList(self.dim)
        super().subs(another, result)
        return result

    def derivative(self, var_id:int):
        result = CPolyLinkedList(self.dim)
        super().derivative(var_id, result)
        return result

    def eval(self, x):
        if(not isinstance(x, CScalarVec)):
            x = CScalarVec(x)
        return super().eval(x)

    def eval_diff(self, diff_order, x):
        if(not isinstance(diff_order, CIndexVec)):
            diff_order = CIndexVec(diff_order)

        if(not isinstance(x, CScalarVec)):
            x = CScalarVec(x)

        return super().eval_diff(diff_order, x)
    
    def partial_eval(self, var_vals:CScalarVec, eval_id_list:CIndexVec, new_dof_map:CIndexVec):
        new_poly = CPolyLinkedList(1)
        super().partial_eval(var_vals, eval_id_list, new_dof_map, new_poly)
        return new_poly

    def batch_eval(self, diff_order, x_arr):
        if(not isinstance(diff_order, CIndexVec)):
            diff_order = CIndexVec(diff_order)
        return super().batch_eval(diff_order, x_arr)

    def batch_add_elements(self, coeffs:CScalarVec, orders:CIndexVec):
        if(not isinstance(coeffs, CScalarVec)):
            coeffs = CScalarVec(coeffs)
        if(not isinstance(orders, CIndexVec)):
            orders = CIndexVec(orders)
        super().batch_add_elements(coeffs, orders)

    def batch_get_data(self):
        _coeffs = CScalarVec([])
        _orders = CIndexVec([])
        super().batch_get_data(_coeffs, _orders)
        coeffs = np.array(_coeffs)
        orders = np.array(_orders, dtype=int).reshape((-1,self.dim))
        return (coeffs, orders)

    def init_with_data(self, coeff, orders):
        _coeffs = CScalarVec(coeff)
        _orders = CIndexVec(orders)
        super().init_with_data(_coeffs, _orders)

    # format: coeffs: np.array(complex), orders: np.array(int)
    def save_to_file(self, fname):
        coeffs, orders = self.batch_get_data()
        data = {}
        data['coeffs'] = coeffs
        data['orders'] = orders
        data['dim'] = self.dim
        sio.savemat(fname, data)

    def to_str(self, varname:list):
        varname_cstr = CStrVec(varname)
        return super().to_str(varname_cstr)

class CLaurant(csolver.aGBZ_2D_solver_Laurant):
    dim: int 
    num: CPolyLinkedList
    num_max_orders: CIndexVec
    denom_orders: CIndexVec

    def __init__(self, dim):
        super().__init__(dim)
    
    def reinit(self):
        super().reinit()

    def reduction(self):
        super().reduction()

    def set_Laurant_by_terms(self, coeffs:CScalarVec , orders:CIntVec):
        super().set_Laurant_by_terms(coeffs, orders)
    
    def eval(self, x_arr:CScalarVec):
        return super().eval(x_arr)

    def flip_variable(self, var_id:int):
        new_laurant = CLaurant(self.dim)
        super().flip_variable(var_id, new_laurant)
        return new_laurant

class CPolyDeque(csolver.aGBZ_2D_solver_PolyDeque):
    def __init__(self):
        super().__init__()
    
    def size(self):
        return super().size()
    
    def push_front(self, poly:CPolyLinkedList):
        super().push_front(poly)

    def push_back(self, poly:CPolyLinkedList):
        super().push_back(poly)
    
    def get_poly(self, n:int, poly:CPolyLinkedList):
        super().get_poly(n, poly)

def buchberger(poly_basis:CPolyDeque, tol:float):
    csolver.aGBZ_2D_buchberger(poly_basis, tol)

def remove_zeros(poly:CPolyLinkedList):
    non_zero_flag = True
    coeffs, orders = poly.batch_get_data()
    new_orders = orders.reshape((-1, poly.dim))
    for j in range(poly.dim):
        curr_min_index = min(new_orders[:,j])
        if(curr_min_index > 0):
            non_zero_flag = False
            new_orders[:,j] -= curr_min_index
    poly.reinit(poly.dim)
    # poly.batch_add_elements(coeffs, new_orders.flatten())
    poly.init_with_data(CScalarVec(coeffs.flatten()), CIndexVec(new_orders.flatten()))
    
    return non_zero_flag

def non_zero_Groebner(poly_basis:CPolyDeque, tol:float):
    # 1. eliminate zeros
    new_basis = CPolyDeque()
    for j in range(poly_basis.size()):
        curr_poly = CPolyLinkedList(1)
        poly_basis.get_poly(j, curr_poly)
        non_zero_flag = remove_zeros(curr_poly)
        new_basis.push_back(curr_poly)
    poly_basis = new_basis
    
    # 2. calculate Groebner basis and get non zero terms
    non_zero_flag = False
    while(not non_zero_flag):
        non_zero_flag = True
        buchberger(poly_basis, tol)
        new_basis = CPolyDeque()
        for j in range(poly_basis.size()):
            curr_poly = CPolyLinkedList(1)
            poly_basis.get_poly(j, curr_poly)
            curr_non_zero_flag = remove_zeros(curr_poly)
            curr_poly_coeffs, curr_poly_orders = curr_poly.batch_get_data()
            if((curr_poly_orders == 0).all()):
                pass
            else:
                new_basis.push_back(curr_poly)
            if(not curr_non_zero_flag):
                non_zero_flag = False
        poly_basis = new_basis

    return poly_basis


class GBZConditionChecker2D:
    continuation_check_tol = 0.5
    curr_point:np.ndarray
    expiphi:complex
    char_poly:CLaurant
    eff_poly:CPolyLinkedList
    which_var: int # the index of the variate beta in "curr_point"
    other_vars: CIndexVec
    param_list:np.ndarray
    initial_val:np.ndarray
    full_solver_success:bool
    checker_1d: csolver.GBZCondition1DChecker

    def __init__(self, curr_point:np.ndarray, expiphi:np.ndarray, char_poly:CLaurant, which_var:int, 
                    continuation_check_tol = 0.5) -> None:
        self.curr_point = curr_point
        self.curr_point_cscalar = CScalarVec(curr_point)
        self.expiphi = expiphi
        self.char_poly = char_poly
        self.which_var = which_var
        self.full_solver_success = True
        self.continuation_check_tol = continuation_check_tol
        if(which_var == 1):
            self.other_vars = CIndexVec([0,2])
        elif(which_var == 2):
            self.other_vars = CIndexVec([0,1])
        else:
            print("GBZConditionChecker2D.__init__: which_var must be 1 or 2", file=sys.stderr)
            assert(0)
        
        self.checker_1d = csolver.GBZCondition1DChecker(char_poly, which_var)

    
    def set_effective_poly(self, var_val):
        self.eff_poly = CPolyLinkedList(3)
        self.char_poly.num.partial_eval(CScalarVec([var_val]),
                                                        CIndexVec([self.which_var]),
                                                        self.other_vars,
                                                        self.eff_poly)

    def continuation_solver_fun(self, x):
        '''
            f(E, beta) = 0
            f(E, beta * expiphi) = 0
        '''
        f_val = np.zeros(2, dtype=complex)
        jac = np.zeros((2,2), dtype=complex)
        f_val[0] = self.eff_poly.eval(x)
        f_val[1] = self.eff_poly.eval([x[0], x[1] * self.expiphi])
        jac[0,0] = self.eff_poly.eval_diff(CIndexVec([1,0]), x)
        jac[0,1] = self.eff_poly.eval_diff(CIndexVec([0,1]), x)
        jac[1,0] = self.eff_poly.eval_diff(CIndexVec([1,0]), [x[0], x[1] * self.expiphi])
        jac[1,1] = self.eff_poly.eval_diff(CIndexVec([0,1]), [x[0], x[1] * self.expiphi]) * self.expiphi
        return f_val, jac

    def full_solve_str(self):
        '''
            f(E, beta) = 0
            f(E, beta * expiphi) = 0
        '''
        str_list = [""]*2
        str_list[0] = self.eff_poly.to_str(['x','y'])
        str_list[1] = self.eff_poly.scale_var(CScalarVec([1.0, self.expiphi])).to_str(['x','y'])
        return str_list
    
    def eval(self, x, beta_var):
        self.curr_point_cscalar[self.which_var] = beta_var
        self.curr_point_cscalar[self.other_vars[1]] = x[1]
        return self.char_poly.eval(self.curr_point_cscalar)

    # def eval_on_segment(self, x, beta_var, segment:CScalarVec):
    #     all_vals = np.zeros(len(segment), dtype=complex)
    #     for j, curr_beta in enumerate(segment):
    #         full_coords = CScalarVec([0.0]*3)
    #         full_coords[self.other_vars[0]] = x[0]
    #         full_coords[self.other_vars[1]] = curr_beta
    #         full_coords[self.which_var] = beta_var
    #         all_vals[j] = self.char_poly.eval(full_coords)
    #     return all_vals
    
    # def eval_on_segment(self, segment:CScalarVec, vals:CScalarVec):
    #     print(vals)
    #     for j, curr_beta in enumerate(segment):
    #         full_coords = CScalarVec([0.0]*3)
    #         full_coords[self.other_vars[0]] = self.x[0]
    #         full_coords[self.other_vars[1]] = curr_beta
    #         full_coords[self.which_var] = self.beta_var
    #         vals[j] = self.char_poly.eval(full_coords)
    #     print(segment)
    #     print(vals)
    #     input()


    def GBZ_condition_1d(self, x, beta_var, theta_detour:float=0.01 * np.pi,
                         N_points_loop:int=200, N_points_detour = 50, zero_eps=1e-3) -> bool:
        return self.checker_1d.GBZ_condition_check(x[0], beta_var, x[1], theta_detour, N_points_loop, N_points_detour, zero_eps)
        # # 1. get winding loop
        # outer_radius = abs(x[1]) * (1 + rel_r_detour)
        # inner_radius = abs(x[1]) * (1 - rel_r_detour)
        
        # # 2. eval on each segment
        # self.x = x
        # self.beta_var = beta_var
        # # eval_fun = lambda loop_path, vals:self.eval_on_segment(x, beta_var, loop_path, vals)
        # segment = CScalarVec(np.linspace(0,2*np.pi, 100))
        # vals = CScalarVec([0.0]*100)
        # self.eval_on_segment(segment, vals)
        # print(vals)
        # input("eval_on_segment test end")
        # outer_winding = int(np.round(csolver.python_winding_phase(0, 2*np.pi, outer_radius, N_points_loop, dphase_th, self.eval_on_segment)/(2*np.pi)))
        # inner_winding = int(np.round(csolver.python_winding_phase(0, 2*np.pi, inner_radius, N_points_loop, dphase_th, self.eval_on_segment)/(2*np.pi)))
    
        # if(outer_winding - inner_winding < 2):
        #     print("Polynomial value:", self.continuation_solver_fun(x), file=sys.stderr)
        #     print("outer winding: %d, inner winding: %d"%(outer_winding, inner_winding), file=sys.stderr)
        #     raise ValueError("GBZConditionChecker2D.GBZ_condition_1d: difference of winding number less than 2!")
        # return ((inner_winding < 0) and (outer_winding > 0))

    def get_continuation_solution(self, initial_val):
        ''' use continuation solver to get the next solution '''
        res = complex_root(self.continuation_solver_fun, initial_val, jac=True)
        return res

    def get_full_solution(self, beta_var, **args):
        ''' use full solver to get all the solutions '''
        eq_str = self.full_solve_str()
        # print(eq_str)
        all_sols = phcpy_solver.solve(eq_str, verbose=False, dictionary_output=True)
        sol_list = []
        # print(self.initial_val)
        for sol in all_sols:
            x = np.array([sol['x'], sol['y']])
            res = self.get_continuation_solution(x)
            # print("1")
            # print(res.x)
            # print(beta_var)
            if(self.GBZ_condition_1d(res.x, beta_var, **args)):
                if(res.success):
                    sol_list.append(res.x)
                else:
                    raise ValueError("GBZConditionChecker2D.get_full_solution: continuation solving failed!")
        
        # 2. find nearest solution
        if(len(sol_list) == 0):
            self.full_solver_success = False
            print("Warning:GBZConditionChecker2D.get_full_solution: GBZ solution number = 0", file=sys.stderr)
            return None
        else:
            self.full_solver_success = True
        all_distances = [la.norm(sol_list[j] - self.initial_val) for j in range(len(sol_list))]
        min_id = np.argmin(all_distances)
        print("Distances:", all_distances, file=sys.stderr)
        if(all_distances[min_id]> la.norm(self.initial_val) * self.continuation_check_tol):
            print("Warning:GBZConditionChecker2D.get_full_solution: continuation check failed", file=sys.stderr)
            # print("Distances:", all_distances, file=sys.stderr)
            self.full_solver_success = False
            return None

        # print(all_distances)
        return sol_list[min_id]

    def eval_param_path(self, param_path, initial_val, **args):
        self.initial_val = initial_val

        curr_param_ptr = 0
        full_solver_flag = False
        all_sols = [0] * len(param_path)
        # char_poly_val = np.zeros((param_path.shape[0]), dtype=complex)
        char_poly_val = [0.0j] * len(param_path)
        sol_success = False
        while(curr_param_ptr < len(param_path)):
            # 1. solve and save solution
            self.set_effective_poly(param_path[curr_param_ptr])
            if(full_solver_flag):
                # print("full solver")
                curr_sol = self.get_full_solution(param_path[curr_param_ptr], **args)
                # print(self.initial_val)
                # print(param_path[curr_param_ptr,0])
                # print(curr_sol)
                if(not self.full_solver_success):
                    return None
                all_sols[curr_param_ptr] = curr_sol
                char_poly_val[curr_param_ptr] = self.eval(curr_sol, param_path[curr_param_ptr])
                sol_success = True
            else:
                # print("continuation solver")
                res = self.get_continuation_solution(self.initial_val)
                sol_success = False
                if(res.success):
                    # print(self.initial_val)
                    # print(res.x)
                    # print(param_path[curr_param_ptr,0])
                    # print(np.abs(res.x - self.initial_val))
                    # print(np.abs(self.initial_val))
                    if(la.norm(res.x - self.initial_val)< la.norm(self.initial_val) * self.continuation_check_tol):
                        if(self.GBZ_condition_1d(res.x, param_path[curr_param_ptr], **args)):
                            all_sols[curr_param_ptr] = res.x
                            char_poly_val[curr_param_ptr] = self.eval(res.x, param_path[curr_param_ptr])
                            sol_success = True

            # print(sol_success)
            # 2. move next
            if(sol_success):
                self.initial_val = all_sols[curr_param_ptr]
                curr_param_ptr += 1
                full_solver_flag = False
            else:
                full_solver_flag = True
        return char_poly_val, all_sols
    
class DirectGBZConditionChecker:
    char_poly:CLaurant
    which_var: int # the index of the variate beta in "curr_point"
    other_var: int
    checker_1d: csolver.GBZCondition1DChecker

    M_major: int
    M_other: int
    N_major: int
    N_other: int

    theta_detour: float
    N_points_loop: int
    N_points_detour: int
    zero_eps: float

    eff_poly: CPolyLinkedList
    another_poly: CPolyLinkedList

    def __init__(self, char_poly:CLaurant, which_var:int, 
                theta_detour:float = 1e-3*np.pi, 
                N_points_loop: int = 200,
                N_points_detour: int = 50,
                zero_eps: float = 1e-3) -> None:
        self.char_poly = char_poly
        self.which_var = which_var

        self.theta_detour = theta_detour
        self.N_points_loop = N_points_loop
        self.N_points_detour = N_points_detour
        self.zero_eps = zero_eps

        if(which_var == 1):
            self.other_var = 2
        elif(which_var == 2):
            self.other_var = 1
        else:
            print("GBZConditionChecker2D.__init__: which_var must be 1 or 2", file=sys.stderr)
            assert(0)
        
        self.checker_1d = csolver.GBZCondition1DChecker(char_poly, which_var)

        self.M_major = self.char_poly.denom_orders[self.which_var]
        self.M_other = self.char_poly.denom_orders[self.other_var]
        coeffs = CScalarVec([])
        orders = CIndexVec([])
        self.char_poly.num.batch_get_data(coeffs, orders)
        orders_arr = np.reshape(orders, (-1,3))
        self.N_major = max(orders_arr[:,self.which_var])
        self.N_other = max(orders_arr[:,self.other_var])

    def newtow_solver_equation(self, x):
        # equation for newton solver

        # equation
        fun = np.zeros(2, dtype=complex)
        fun[0] = self.eff_poly.eval(CScalarVec(x))
        fun[1] = self.another_poly.eval(CScalarVec(x))

        # jacobian
        jac = np.zeros((2,2), dtype=complex)
        jac[0,0] = self.eff_poly.eval_diff(CIndexVec([1,0]), CScalarVec(x))
        jac[0,1] = self.eff_poly.eval_diff(CIndexVec([0,1]), CScalarVec(x))
        jac[1,0] = self.another_poly.eval_diff(CIndexVec([1,0]), CScalarVec(x))
        jac[1,1] = self.another_poly.eval_diff(CIndexVec([0,1]), CScalarVec(x))
        return fun, jac
    
    def get_aGBZ_solutions(self, curr_point, curr_params):
        # 1. set characteristic polynomial
        self.eff_poly = CPolyLinkedList(2)
        self.char_poly.num.partial_eval(CScalarVec([curr_point[0]]),
                                                        CIndexVec([0]),
                                                        CIndexVec([1,2]),
                                                        self.eff_poly)
        expiphi = curr_params[self.other_var - 1]
        scale_factor = CScalarVec([1.0, 1.0])
        scale_factor[self.other_var - 1] = expiphi
        self.another_poly = self.eff_poly.scale_var(scale_factor)
        eff_eq = [
            self.eff_poly.to_str(['x','y']),
            self.another_poly.to_str(['x', 'y'])
        ]

        # 2. get aGBZ solution
        all_sols = phcpy_solver.solve(eff_eq, verbose=False, dictionary_output= True)
        all_checked_sols = []
        for sol in all_sols:
            res = complex_root(self.newtow_solver_equation, np.array([sol['x'], sol['y']]), jac=True)
            print(sol['x'], sol['y'], sol['m'])
            if(not res.success):
                print("Newton solver fails, error solution:", sol['x'], sol['y'], file=sys.stderr)
                print(res.message, file=sys.stderr)
            # if(self.checker_1d.GBZ_condition_check(curr_point[0], res.x[self.which_var - 1],
            #                             res.x[self.other_var-1], self.theta_detour, self.N_points_loop, self.N_points_detour, self.zero_eps)):
            if(self.GBZ_condition_1d(curr_point, res.x[self.which_var-1])):
                for j in range(sol['m']):
                    all_checked_sols.append(res.x)
        
        return np.asarray(all_checked_sols)
    
    def GBZ_condition_1d(self, curr_point, curr_beta, tol = 1e-6):
        # get 1D equation and then GBZ condition

        # 1. get effective 1D equation
        poly_1d = CPolyLinkedList(1)
        self.char_poly.num.partial_eval(CScalarVec([curr_point[0], curr_beta]), CIndexVec([0, self.which_var]),
                                    CIndexVec([self.other_var]), poly_1d)
        coeffs, orders = poly_1d.batch_get_data()
        if(abs(curr_beta) < tol):
            return False
        if(max(orders) != self.N_other):
            return False

        eff_1D_eq = np.zeros((self.N_other + 1,), dtype=complex)
        for j in range(len(orders)):
            eff_1D_eq[self.N_other - orders[j]] = coeffs[j]
        
        # 2. solve
        sols = np.roots(eff_1D_eq)
        sols_seq = np.argsort(np.abs(sols))
        sols = sols[sols_seq]
        print(abs(sols))

        # 3. check solution
        return (abs(abs(sols[self.M_other - 1]) - abs(sols[self.M_other])) < tol) 
    
    def check_GBZ_condition(self, curr_point, curr_params):
        # 1. get all solutions
        all_sols = self.get_aGBZ_solutions(curr_point, curr_params)

        # 2. sort which_var
        major_var_order = np.argsort(np.abs(all_sols[:, self.which_var - 1]))
        all_sols = all_sols[major_var_order, :]
        major_var_abs = np.abs(all_sols[:, self.which_var - 1])

        # 3. check the GBZ condition
        beta_denom_order = self.char_poly.denom_orders[self.which_var]
        print(self.char_poly.denom_orders)
        print(beta_denom_order)
        print(len(major_var_abs))
        print(major_var_abs)
        if(len(major_var_abs) <= beta_denom_order):
            print("DirectGBZConditionChecker.check_GBZ_condition: solution number < beta order")
            return False
        if(abs(major_var_abs[beta_denom_order - 1] - major_var_abs[beta_denom_order]) < 1e-6 * abs(major_var_abs[beta_denom_order - 1])):
            all_distances = [la.norm(all_sols[j, :] - curr_point[1:]) for j in range(all_sols.shape[0])]
            min_id = np.argmin(all_distances)
            if(min_id == beta_denom_order or min_id == (beta_denom_order - 1)):
                return True
        return False
        
        
    

def check_GBZ_condition(char_poly, aGBZ_point, expiphi, which_var,
                        outer_theta_detour:float=0.01*np.pi,
                        outer_N_points_loop:int=200,
                        outer_N_points_detour:int = 50,
                        theta_detour:float=0.01*np.pi,
                        N_points_loop:int=200,
                        N_points_detour:int = 50,
                        zero_val_th:float = 1e-3,
                        continuation_check_tol:float=0.1
                        ):
    
    # 1. generate beta path
    beta = aGBZ_point[which_var]

    # 2. evaluate the values on each path
    if(which_var == 1):
        winding_solver = GBZConditionChecker2D(aGBZ_point, expiphi[1], char_poly, which_var, continuation_check_tol)
    else:
        winding_solver = GBZConditionChecker2D(aGBZ_point, expiphi[0], char_poly, which_var, continuation_check_tol)


    eval_fun = lambda beta_path, start_param: winding_solver.eval_param_path(
        beta_path, start_param, 
        theta_detour = theta_detour, N_points_loop=N_points_loop, N_points_detour = N_points_detour, 
        zero_eps=zero_val_th
    )

    return winding_phase_auto_detour(beta, outer_theta_detour, zero_val_th, outer_N_points_loop,
                                     outer_N_points_detour, eval_fun, aGBZ_point[winding_solver.other_vars],
                                     continuation_check_tol)

    # phase_diff = np.log(expiphi[which_var-1]).imag
    # if(phase_diff < 0):
    #     phase_diff += 2 * np.pi
    # if(phase_diff > 2 * np.pi):
    #     phase_diff -= 2 * np.pi
    


    # curr_r_detour = outer_rel_r_detour
    # outer_phase1, outer_end_param1, zero_point_flag = winding_phase_with_param(start_phase, start_phase + phase_diff, beta_norm*(1 + curr_r_detour), outer_N_points_loop, 
    #                                 dphase_th, aGBZ_point[winding_solver.other_vars], eval_fun)
    # if(not winding_solver.full_solver_success):
    #     return False
    # while(not zero_point_flag):
    #     curr_r_detour/= 2                        
    #     outer_phase1, outer_end_param1, zero_point_flag = winding_phase_with_param(start_phase, start_phase + phase_diff, beta_norm*(1 + curr_r_detour), outer_N_points_loop, 
    #                                     dphase_th, aGBZ_point[winding_solver.other_vars], eval_fun)
    #     if(not winding_solver.full_solver_success):
    #         return False
    
    # curr_r_detour = outer_rel_r_detour
    # inner_phase1, inner_end_param1, zero_point_flag = winding_phase_with_param(start_phase, start_phase + phase_diff, beta_norm*(1-curr_r_detour), outer_N_points_loop, 
    #                                 dphase_th, aGBZ_point[winding_solver.other_vars], eval_fun)
    # if(not winding_solver.full_solver_success):
    #     return False
    # while(not zero_point_flag):
    #     curr_r_detour /= 2
    #     inner_phase1, inner_end_param1, zero_point_flag = winding_phase_with_param(start_phase, start_phase + phase_diff, beta_norm*(1-curr_r_detour), outer_N_points_loop, 
    #                                     dphase_th, aGBZ_point[winding_solver.other_vars], eval_fun)
    #     if(not winding_solver.full_solver_success):
    #         return False
                                    
    # curr_r_detour = outer_rel_r_detour
    # outer_phase2, outer_end_param2, zero_point_flag = winding_phase_with_param(start_phase + phase_diff, start_phase + 2 * np.pi, beta_norm*(1+curr_r_detour), outer_N_points_loop, 
    #                                 dphase_th, outer_end_param1, eval_fun)
    # if(not winding_solver.full_solver_success):
    #     return False
    # while(not zero_point_flag):
    #     curr_r_detour /= 2
    #     outer_phase2, outer_end_param2, zero_point_flag = winding_phase_with_param(start_phase + phase_diff, start_phase + 2 * np.pi, beta_norm*(1+curr_r_detour), outer_N_points_loop, 
    #                                     dphase_th, outer_end_param1, eval_fun)
    #     if(not winding_solver.full_solver_success):
    #         return False

    # curr_r_detour = outer_rel_r_detour
    # inner_phase2, inner_end_param2, zero_point_flag = winding_phase_with_param(start_phase + phase_diff, start_phase + 2 * np.pi, beta_norm*(1-curr_r_detour), outer_N_points_loop, 
    #                                 dphase_th, inner_end_param1, eval_fun)
    # if(not winding_solver.full_solver_success):
    #     return False
    # while(not zero_point_flag):
    #     curr_r_detour /= 2
    #     inner_phase2, inner_end_param2, zero_point_flag = winding_phase_with_param(start_phase + phase_diff, start_phase + 2 * np.pi, beta_norm*(1-curr_r_detour), outer_N_points_loop, 
    #                                     dphase_th, inner_end_param1, eval_fun)
    #     if(not winding_solver.full_solver_success):
    #         return False
   
    # outer_winding = int(np.round((outer_phase1 + outer_phase2)/(2*np.pi)))
    # inner_winding = int(np.round((inner_phase1 + inner_phase2)/(2*np.pi)))

    # if((outer_winding > 0) and (inner_winding < 0)):
    #     return True
    # else:
    #     return False
class BubbleType:
    left_point: complex
    right_point: complex
    N_points: int

    def __init__(self, left) -> None:
        self.left_point = left
        self.right_point = left
        self.N_points = 1
    
    def add_right(self, right):
        self.right_point = right
        self.N_points += 1
    
    def add_left(self, left):
        self.left_point = left
        self.N_points += 1
        

class HomotopyGBZChecker:
    char_poly:CLaurant
    which_var: int # the index of the variate beta in "curr_point"
    other_var: int

    eff_poly: CPolyLinkedList
    another_poly: CPolyLinkedList

    def __init__(self, char_poly:CLaurant, which_var:int) -> None:
        self.char_poly = char_poly
        self.which_var = which_var

        if(which_var == 1):
            self.other_var = 2
        elif(which_var == 2):
            self.other_var = 1
        else:
            print("GBZConditionChecker2D.__init__: which_var must be 1 or 2", file=sys.stderr)
            assert(0)

    def newtow_solver_equation(self, x):
        # equation for newton solver

        # equation
        fun = np.zeros(2, dtype=complex)
        fun[0] = self.eff_poly.eval(CScalarVec(x))
        fun[1] = self.another_poly.eval(CScalarVec(x))

        # jacobian
        jac = np.zeros((2,2), dtype=complex)
        jac[0,0] = self.eff_poly.eval_diff(CIndexVec([1,0]), CScalarVec(x))
        jac[0,1] = self.eff_poly.eval_diff(CIndexVec([0,1]), CScalarVec(x))
        jac[1,0] = self.another_poly.eval_diff(CIndexVec([1,0]), CScalarVec(x))
        jac[1,1] = self.another_poly.eval_diff(CIndexVec([0,1]), CScalarVec(x))
        return fun, jac
    
    def set_current_polys(self, curr_point, curr_params):
        # 1. set characteristic polynomial
        self.eff_poly = CPolyLinkedList(2)
        self.char_poly.num.partial_eval(CScalarVec([curr_point[0]]),
                                                        CIndexVec([0]),
                                                        CIndexVec([1,2]),
                                                        self.eff_poly)
        expiphi = curr_params[self.other_var - 1]
        scale_factor = CScalarVec([1.0, 1.0])
        scale_factor[self.other_var - 1] = expiphi
        self.another_poly = self.eff_poly.scale_var(scale_factor)

    def get_aGBZ_solutions(self, curr_point, curr_params):
        # 1. set characteristic polynomial
        self.set_current_polys(curr_point, curr_params)
        eff_eq = [
            self.eff_poly.to_str(['x','y']),
            self.another_poly.to_str(['x', 'y'])
        ]

        # 2. get aGBZ solution
        all_sols = phcpy_solver.solve(eff_eq, verbose=False, dictionary_output= True)
        all_checked_sols = []
        for sol in all_sols:
            res = complex_root(self.newtow_solver_equation, np.array([sol['x'], sol['y']]), jac=True)
            if(not res.success):
                print("Newton solver fails", file=sys.stderr)
            all_checked_sols.append(res.x)
        
        return np.asarray(all_checked_sols)
    

    def get_full_1d_solutions(self, curr_beta):
        # 1. convert current polynomial to np.array
        eff_1d_poly = self.eff_poly.partial_eval(CScalarVec([curr_beta]), CIndexVec([self.which_var - 1]), CIndexVec([self.other_var-1]))
        coeffs, orders = eff_1d_poly.batch_get_data()
        max_order = np.max(orders)
        eff_1d_poly_arr = np.zeros((max_order + 1,), dtype=complex)
        for j in range(len(orders)):
            eff_1d_poly_arr[max_order - orders[j]] = coeffs[j]

        # 2. solve
        all_roots = np.roots(eff_1d_poly_arr)
        seq = np.argsort(np.abs(all_roots))
        return all_roots[seq]
    
    def get_poly_vals(self, curr_point, beta_path):
        all_vals = np.zeros(len(beta_path), dtype=complex)
        other_beta_sq = np.zeros_like(beta_path, dtype=complex)
        for j, curr_beta in enumerate(beta_path):
            curr_sol = self.get_full_1d_solutions(curr_beta)
            denom_order = self.char_poly.denom_orders[self.other_var]
            beta_for_loop = np.mean(np.abs(curr_sol[[denom_order-1, denom_order]]))

            point_loop = beta_for_loop * np.exp(1j * np.linspace(0, 2 * np.pi, 500))
            curr_vals = np.zeros_like(point_loop, dtype=complex)
            curr_point_to_eval = CScalarVec(curr_point)
            # curr_point_to_eval[self.other_var] = beta_for_loop * curr_point[self.other_var] / np.abs(curr_point[self.other_var])
            curr_point_to_eval[self.which_var] = curr_beta
            for k in range(len(curr_vals)):
                curr_point_to_eval[self.other_var] = point_loop[k]
                curr_vals[k] = self.char_poly.eval(curr_point_to_eval) 
            print(winding_number(curr_vals))
            if(winding_number(curr_vals) != 0):
                print(np.abs(curr_sol))
                print(denom_order)
            other_beta_sq[j] = np.prod(curr_sol[[denom_order - 1, denom_order]])
        
        other_beta = npe.continuous_sqrt(other_beta_sq)
        for j, curr_beta in enumerate(beta_path):
            curr_point_to_eval = CScalarVec(curr_point)
            # curr_point_to_eval[self.other_var] = beta_for_loop * curr_point[self.other_var] / np.abs(curr_point[self.other_var])
            curr_point_to_eval[self.other_var] = other_beta[j]
            curr_point_to_eval[self.which_var] = curr_beta
            all_vals[j] = self.char_poly.eval(curr_point_to_eval)
        return all_vals
    
    def GBZ_check(self, curr_point, curr_params, N_points_detour, N_points_loop, theta_detour, norm_tol = 1e-3):

        # 1. get all aGBZ solutions
        all_sols = self.get_aGBZ_solutions(curr_point, curr_params)

        # 2. collect the solutions with same norms
        curr_beta = curr_point[self.which_var]
        aGBZ_beta = all_sols[:,self.which_var - 1]
        near_points_indices = np.abs(np.abs(aGBZ_beta) - np.abs(curr_beta)) < norm_tol
        all_near_points = aGBZ_beta[near_points_indices]

        # 3. generate loop path and evaluate
        outer_vals = []
        inner_vals = []
        rel_r_detour = 2 * np.sin(theta_detour)
        # 3.1 generate bubble data
        # 3.1.1 sort points
        all_angles = np.log(all_near_points).imag
        point_seq = np.argsort(all_angles)
        all_near_points = list(all_near_points[point_seq])

        # 3.1.2 collect bubbles
        bubble_list = []
        curr_bubble = BubbleType(all_near_points.pop(0))
        if(len(all_near_points)):
            while(abs(all_near_points[0] - curr_bubble.right_point) < 2 * rel_r_detour):
                curr_bubble.add_right(all_near_points.pop(0))
                if(not len(all_near_points)):
                    break
        if(len(all_near_points)):
            while(abs(all_near_points[-1] - curr_bubble.left_point) < 2 * rel_r_detour):
                curr_bubble.add_left(all_near_points.pop(-1))
                if(not len(all_near_points)):
                    break
        
        bubble_list.append(curr_bubble)
        if(len(all_near_points)):
            new_bubble = BubbleType(all_near_points.pop(0))
            while(len(all_near_points)):
                if(abs(all_near_points[0] - new_bubble.right_point) < 2 * rel_r_detour):
                    curr_bubble.add_right(all_near_points.pop(0))
                else:
                    bubble_list.append(new_bubble)
                    new_bubble = BubbleType(all_near_points.pop(0))
        bubble_list.append(new_bubble)

        # 3.2 for each bubble, generate bubble paths
        outer_bubble_paths = []
        inner_bubble_paths = []
        for bubble in bubble_list:
            if(bubble.N_points == 1):
                curr_outer_path = CScalarVec([])
                curr_inner_path = CScalarVec([])
                csolver.atlas_bubble_gen(bubble.left_point, rel_r_detour, theta_detour, N_points_detour,
                        curr_outer_path, curr_inner_path)
                outer_bubble_paths.append(np.array(curr_outer_path))
                inner_bubble_paths.append(np.array(curr_inner_path))
            else:
                outer_begin = CScalarVec([])
                outer_arc = CScalarVec([])
                outer_end = CScalarVec([])
                inner_begin = CScalarVec([])
                inner_arc = CScalarVec([])
                inner_end = CScalarVec([])
                csolver.atlas_long_bubble_gen([bubble.left_point, bubble.right_point],
                            rel_r_detour, theta_detour, N_points_detour//2, N_points_detour * (bubble.N_points - 1),
                            outer_begin, inner_begin, outer_arc, inner_arc, outer_end, inner_end)
                outer_bubble_paths.append(np.hstack([outer_begin, outer_arc, outer_end]))
                inner_bubble_paths.append(np.hstack([inner_begin, inner_arc, inner_end]))

        # 3.3 generate links between them
        link_list = []
        for j in range(len(bubble_list) - 1):
            curr_start_point = outer_bubble_paths[j][-1]
            curr_end_point = outer_bubble_paths[j + 1][0]
            diff_phase = np.log(curr_end_point/curr_start_point).imag
            if(diff_phase < 0):
                diff_phase += 2 * np.pi
            link_list.append(curr_start_point * np.exp(1j * np.linspace(0, diff_phase, N_points_loop)))
        curr_start_point = outer_bubble_paths[-1][-1]
        curr_end_point = outer_bubble_paths[0][0]
        diff_phase = np.log(curr_end_point/curr_start_point).imag
        if(diff_phase < 0):
            diff_phase += 2 * np.pi
        link_list.append(curr_start_point * np.exp(1j * np.linspace(0, diff_phase, N_points_loop)))
        
        # 4. calculate winding number
        outer_paths = []
        inner_paths = []
        for j in range(len(bubble_list)):
            outer_paths.append(outer_bubble_paths[j])
            inner_paths.append(inner_bubble_paths[j])
            outer_paths.append(link_list[j])
            inner_paths.append(link_list[j])
        outer_paths = np.hstack(outer_paths)
        inner_paths = np.hstack(inner_paths)
        # return outer_paths, inner_paths
            
            # curr_outer_vals = list(self.get_poly_vals(curr_point, outer_bubble_paths[j]))
            # curr_inner_vals = list(self.get_poly_vals(curr_point, inner_bubble_paths[j]))
            # curr_link_vals = list(self.get_poly_vals(curr_point, link_list[j]))
            # outer_vals += curr_outer_vals + curr_link_vals
            # inner_vals += curr_inner_vals + curr_link_vals
        
        outer_vals = self.get_poly_vals(curr_point, outer_paths)
        inner_vals = self.get_poly_vals(curr_point, inner_paths)
        outer_winding = winding_number(outer_vals)
        inner_winding = winding_number(inner_vals)
        return outer_vals, inner_vals

class GBZ1DSolver:
    char_poly:CLaurant
    which_var:int
    other_var:int
    min_deg:int
    max_deg:int

    def __init__(self, char_poly:CLaurant, which_var:int) -> None:
        self.char_poly = char_poly
        self.which_var = which_var
        if(which_var == 1):
            self.other_var = 2
        else:
            self.other_var = 1
        
        self.min_deg = self.char_poly.denom_orders[self.other_var]
        degrees = CIndexVec([])
        self.char_poly.num.batch_get_data(CScalarVec([]), degrees)
        self.max_deg = np.max(degrees)
    
    def get_all_solutions(self, E, beta_loop):
        all_sols = []
        for beta in beta_loop:
            curr_eq = CPolyLinkedList(1)
            self.char_poly.num.partial_eval(
                CScalarVec([E, beta]),
                CIndexVec([0, self.which_var]),
                CIndexVec([self.other_var]),
                curr_eq
            )
            curr_eq_arr = np.zeros(self.max_deg + 1, dtype=complex)
            coeffs, degrees = curr_eq.batch_get_data()
            for j in range(len(coeffs)):
                curr_eq_arr[self.max_deg - degrees[j]] = coeffs[j]
            
            curr_roots = np.roots(curr_eq_arr)
            curr_roots_seq = np.argsort(np.abs(curr_roots))
            all_sols.append(curr_roots[curr_roots_seq])
        return np.array(all_sols)
    

class GBZFullSolver:
    char_poly:CLaurant
    which_var:int
    other_var:int
    min_deg:int
    max_deg:int

    def __init__(self, char_poly:CLaurant, which_var:int) -> None:
        self.char_poly = char_poly
        self.which_var = which_var
        if(which_var == 1):
            self.other_var = 2
        else:
            self.other_var = 1
        
        self.min_deg = self.char_poly.denom_orders[self.other_var]
        degrees = CIndexVec([])
        self.char_poly.num.batch_get_data(CScalarVec([]), degrees)
        self.max_deg = np.max(degrees)
    
    def get_1D_GBZ(self, beta_loop: np.ndarray, tol:float=1e-6):
        pass

def segments_to_loops(segment_list:list, repairing_tolerence:float = 1e-3, loop_critical_length:int=3):
    all_loops = []
    fragments = []

    # sweep each loops until the segment_list become empty
    curr_seg = segment_list.pop(0)
    while(len(segment_list)):
        all_heads = np.vstack([curr_seg[0,:]]+[seg[0,:] for seg in segment_list])

        # calculate distances
        all_distances = np.zeros(all_heads.shape[0])
        for j in range(all_heads.shape[0]):
            all_distances[j] = la.norm(all_heads[j,:] - curr_seg[-1,:])
        
        # select the nearest head
        min_id = np.argmin(all_distances)

        if(all_distances[min_id] < repairing_tolerence):
            if(min_id == 0):
                if(curr_seg.shape[0] < loop_critical_length):
                    fragments.append(curr_seg)
                else:
                    all_loops.append(curr_seg)
                curr_seg = segment_list.pop(0)
            else:
                new_seg =segment_list.pop(min_id - 1)
                curr_seg = np.vstack([curr_seg, new_seg])
        else:
            fragments.append(curr_seg)
            curr_seg = segment_list.pop(0)
    
    if(la.norm(curr_seg[0,:] - curr_seg[-1,:]) < repairing_tolerence):
        if(curr_seg.shape[0] < loop_critical_length):
            fragments.append(curr_seg)
        else:
            all_loops.append(curr_seg)
    else:
        fragments.append(curr_seg)

    return all_loops, fragments

def get_min_distance(point:np.ndarray, point_set:np.ndarray):
    point_dim = len(point)
    diff_vec = point_set.copy()
    for j in range(point_dim):
        diff_vec[:,j] -= point[j]
    diff_norm = la.norm(diff_vec, axis=1)
    min_id = np.argmin(diff_norm)
    return diff_norm[min_id], min_id
    
def loop_distances(loop1:np.ndarray, loop2:np.ndarray, N_points:int):
    # find several points of loop1, calculate the mean distance between these points and loop2
    N_intervals = loop1.shape[0]//N_points
    point_lists = loop1[::N_intervals,:]
    mean_distances = 0.0
    for j in range(point_lists.shape[0]):
        mean_distances += get_min_distance(point_lists[j,:], loop2)[0]
    mean_distances/=(point_lists.shape[0])
    return mean_distances

# given a list of loops, get a T2
def loops_to_loop_chain(loop_list:list):
    N_chains = len(loop_list[0])
    loop_chains = [[loop] for loop in loop_list[0]]
    for curr_index in range(len(loop_list) - 1):
        if(len(loop_list[curr_index + 1]) != N_chains):
            raise ValueError("Number of loops does not match")
        for chain_index in range(N_chains):
            all_distances = np.zeros(N_chains)
            for k in range(N_chains):
                all_distances[k] = loop_distances(loop_chains[chain_index][-1], loop_list[curr_index + 1][k], 5)
            # print(all_distances)
            min_id = np.argmin(all_distances)
            loop_chains[chain_index].append(loop_list[curr_index + 1][min_id])
    return loop_chains

def loop_chain_to_torus(loop_chains):
    torus_list = []
    curr_direction = True
    curr_torus = loop_chains.pop(0)
    while(len(loop_chains)):
        if(curr_direction):
            next_heads = [chain[-1] for chain in loop_chains]
            all_distances = np.zeros(len(loop_chains) + 1)
            all_distances[0] = loop_distances(curr_torus[-1], curr_torus[0], 5)
            for k in range(len(loop_chains)):
                all_distances[k + 1] = loop_distances(curr_torus[-1], next_heads[k], 5)
            min_id = np.argmin(all_distances)

            if(min_id == 0):
                torus_list.append(curr_torus)
                curr_torus = loop_chains.pop(0)
            else:
                curr_torus += loop_chains.pop(min_id - 1)[::-1]
        else:
            next_heads = [chain[0] for chain in loop_chains]
            all_distances = np.zeros(len(loop_chains) + 1)
            all_distances[0] = loop_distances(curr_torus[-1], curr_torus[0], 5)
            for k in range(len(loop_chains)):
                all_distances[k + 1] = loop_distances(curr_torus[-1], next_heads[k], 5)
            min_id = np.argmin(all_distances)

            if(min_id == 0):
                torus_list.append(curr_torus)
                curr_torus = loop_chains.pop(0)
            else:
                curr_torus += loop_chains.pop(min_id - 1)

        curr_direction = not curr_direction
    torus_list.append(curr_torus)
    return torus_list