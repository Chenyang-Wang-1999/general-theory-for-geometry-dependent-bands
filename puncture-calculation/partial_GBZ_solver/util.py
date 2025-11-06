'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-01-25
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from numpy.linalg import cond

import matplotlib.pyplot as plt

from scipy import optimize
from scipy import linalg as LA

import sys

def raise_warning(*warning_str):
    print(*warning_str, file=sys.stderr)

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

class PointSetDistanceChecker:
    min_dist:float
    max_range:float
    def __init__(self, min_dist, max_range) -> None:
        self.min_dist = min_dist
        self.max_range = max_range

    def __call__(self, point_list) -> bool:
        for j in range(point_list.shape[0]):
            if(np.linalg.norm(point_list[j,:])>self.max_range):
                return False
            for k in range(j+1, point_list.shape[0]):
                curr_dist = np.linalg.norm(point_list[j,:] - point_list[k,:])
                if(curr_dist < self.min_dist):
                    return False
        return True

class BatchComplexRoot:
    root_options:dict
    def __init__(self, **root_options):
        self.root_options = root_options
    
    def __call__(self, fun, all_x0) -> np.ndarray:
        all_sols = np.zeros_like(all_x0, dtype=complex)
        for j in range(all_sols.shape[0]):
            if(any(np.isnan(all_x0[j,:]))):
                all_sols[j,:] = np.nan
            else:
                res = complex_root(fun, all_x0[j,:], **self.root_options)
                if(res.success):
                    all_sols[j,:] = res.x
                else:
                    all_sols[j,:] = np.nan
        return all_sols

def segments_to_loop(segments, tol = 1e-3):
    # glue a list of segments to loops
    loop_list = []
    loop_indices = []
    all_indices = list(range(len(segments)))
    segment_dim = segments[0].shape[1]
    while(len(segments)):
        curr_loop = segments.pop(0)
        curr_indices = [all_indices.pop(0)]
        while(LA.norm(curr_loop[0,:] - curr_loop[-1,:]) > tol):
            if(len(segments) == 0):
                raise ValueError("All segments exhausted")
            # collect all off the start point
            new_heads = np.zeros((len(segments), segment_dim), dtype=complex)
            for j in range(len(segments)):
                new_heads[j,:] = segments[j][0,:]
            diff_vec = new_heads - curr_loop[-1,:].reshape((1,-1))
            diff_norm = LA.norm(diff_vec, axis=1)
            min_id = np.argmin(diff_norm)
            if(diff_norm[min_id]>tol):
                raise_warning(diff_norm[min_id])
                # raise ValueError("GBZ not continuous")
                raise_warning("Warning: GBZ not continuous")
                break
            curr_loop = np.vstack([curr_loop, segments.pop(min_id)])
            curr_indices.append(all_indices.pop(min_id))
        loop_list.append(curr_loop)
        loop_indices.append(curr_indices)
    return loop_list, loop_indices

class CriticalChecker:
    using_raw: bool = True
    def __call__(self, curr_param:np.ndarray, curr_sol:np.ndarray, next_sol) -> bool:
        return False

class ContinuityCriticalChecker(CriticalChecker):
    continuity_tol:float
    def __init__(self, continuity_tol) -> None:
        self.continuity_tol = continuity_tol

    def __call__(self, curr_param: np.ndarray, curr_sol: np.ndarray, next_sol) -> bool:
        if(any(np.isinf(next_sol.flatten()))):
            return True
        if(any(np.isnan(next_sol.flatten()))):
            # print(next_sol)
            return True
        dist = LA.norm(next_sol - curr_sol, axis=1)
        sol_norm = LA.norm(curr_sol, axis=1)
        flag_list = (dist > self.continuity_tol*sol_norm)
        if(any(flag_list)):
            # print(curr_param)
            # print(flag_list)
            # print(sol_norm)
            # print(dist)
            # input()
            return True
        else:
            return False


def match_next_point(curr_points, sol_data):
    # give current points, match next points

    if(len(sol_data) == 0):
        raise ValueError("Full solver fails!")
    elif(len(sol_data) == curr_points.shape[0]):
        next_indices = list(range(len(sol_data)))
        next_points = np.zeros_like(curr_points, dtype=curr_points.dtype)
        for j in range(curr_points.shape[0]):
            next_candidates = sol_data[next_indices,:]
            diff_vec = next_candidates - curr_points[j,:].reshape((1,-1))
            min_id = np.argmin(np.linalg.norm(diff_vec, axis=1))
            next_point_id = next_indices.pop(min_id)
            next_points[j,:] = sol_data[next_point_id,:]
        return next_points
    else:
        # 1. for each src, map to dest
        dest_ptr_list = [[] for j in range(len(sol_data))]
        src_to_dest = [0 for j in range(curr_points.shape[0])]
        for j in range(curr_points.shape[0]):
            # get the index of the minimum of dest
            diff_vec = sol_data - curr_points[j,:].reshape((1,-1))
            min_id = np.argmin(np.linalg.norm(diff_vec, axis=1))
            src_to_dest[j] = min_id
            dest_ptr_list[min_id].append(j)
        
        # 2. for empty dest, find nearest src, and move its ptr to curr dest
        for j in range(len(sol_data)):
            if(len(dest_ptr_list[j]) == 0):
                # this dest is not pointed
                diff_vec = curr_points - sol_data[j,:].reshape((1,-1))
                min_id = np.argmin(np.linalg.norm(diff_vec, axis=1))

                # find the original dest which "min_id" points to
                old_dest_id = src_to_dest[min_id]

                # check old id
                if(len(dest_ptr_list[old_dest_id]) < 2):
                    print("intersection points:",len(dest_ptr_list[old_dest_id]))
                    raise ValueError("Intersection point error")
                
                # move the ptr to new dest
                dest_ptr_list[old_dest_id].remove(min_id)
                src_to_dest[min_id] = j
                dest_ptr_list[j].append(min_id)

        # 3. assemble new point list
        next_points = np.zeros_like(curr_points, dtype=curr_points.dtype)
        for j in range(curr_points.shape[0]):
            next_points[j,:] = sol_data[src_to_dest[j],:]

        return next_points


class ParamFun:
    params: np.ndarray
    def set_params(self, params):
        self.params = params
    def __call__(self, x):
        pass

class FullSolverCode:
    NORMAL = 0
    REVERSE = 1
    SHAKE = 2

def check_full_solver_default(curr_sol, next_sol):
    if(len(curr_sol) == len(next_sol)):
        return FullSolverCode.NORMAL
    elif(len(curr_sol) > len(next_sol)):
        return FullSolverCode.SHAKE
    else:
        return FullSolverCode.REVERSE


class DefaultPostProcessor():
    def __call__(self, sol_data):
        return sol_data


class BatchSolverController:
    param_fun: ParamFun
    param_path:np.ndarray
    path_ptr:int
    value_list: list
    result_list: list
    curr_sol: np.ndarray # sol dim x n_sols
    critical_checker: CriticalChecker
    backward_flag: bool = False
    force_full_solver: bool = False
    # continuation_solver(fun, curr_sols) -> next_sols
    # full_solver(fun) -> sols
    # sol_checker(sol) -> True(continuation)/False(full)
    def __init__(self, full_solver, continuation_solver, param_fun, 
                 sol_checker, critical_checker:CriticalChecker, post_processor: DefaultPostProcessor,
                 match_fun, full_sol_checker) -> None:
        self.param_fun = param_fun
        self.full_solver = full_solver
        self.continuation_solver = continuation_solver
        self.sol_checker = sol_checker
        self.critical_checker = critical_checker
        self.match_fun = match_fun
        self.post_processor = post_processor
        self.full_sol_checker = full_sol_checker
    
    def set_path(self, param_path):
        self.param_path = param_path
        self.path_ptr = 0
        self.backward_flag = False
        self.force_full_solver = False
        self.value_list = [[] for _ in range(param_path.shape[0])]
        self.result_list = [[] for _ in range(param_path.shape[0])]

        # initial solutions
        self.param_fun.set_params(param_path[self.path_ptr,:].copy())
        self.curr_sol = self.full_solver(self.param_fun)
        self.value_list[0] = self.curr_sol
        self.result_list[0] = self.post_processor(self.curr_sol)
        self.path_ptr += 1
    
    def run_full_solve(self):
        self.force_full_solver = False
        sol_data = self.full_solver(self.param_fun)
        sol_err_code = self.full_sol_checker(self.curr_sol, sol_data)
        
        if(sol_err_code == FullSolverCode.REVERSE):
            raise_warning("Warning: solver reversed")
            self.dumped_ptr = self.path_ptr
            self.backward_flag = True
        elif(sol_err_code == FullSolverCode.SHAKE):
            raise_warning("Warning: not all points solved. %d, %s"%(self.path_ptr, self.param_path[self.path_ptr,:]))
            self.force_full_solver = True
        return sol_data, sol_err_code
        
    
    def evolve_next(self):
        self.param_fun.set_params(self.param_path[self.path_ptr,:].copy())
        if(self.force_full_solver):
            full_solver_flag = False
        else:
            full_solver_flag = self.sol_checker(self.curr_sol)
        if(full_solver_flag):
            # continuation
            # print("Parameters %s: continuation solver"%(str(self.param_path[self.path_ptr,:])))
            new_sol = self.continuation_solver(self.param_fun, self.curr_sol)
            sol_err_code = FullSolverCode.NORMAL
            if(new_sol is not None):
                if(not self.sol_checker(new_sol)):
                    new_sol = None
        else:
            # full
            # print("Parameters %s: full solver"%(str(self.param_path[self.path_ptr,:])))
            new_sol, sol_err_code = self.run_full_solve()
            
        if(new_sol is None):
            self.force_full_solver = True
        else:
            if(self.critical_checker.using_raw):
                critical_flag = self.critical_checker(self.param_path[self.path_ptr,:], 
                        self.value_list[self.path_ptr-1], new_sol)
            else:
                critical_flag = self.critical_checker(self.param_path[self.path_ptr,:], 
                        self.result_list[self.path_ptr-1], 
                        self.post_processor(new_sol))
            if(critical_flag):
                # if current parameter is critical
                self.insert_midpoints()
            else:
                if(sol_err_code != FullSolverCode.SHAKE):
                    self.curr_sol = new_sol
                self.value_list[self.path_ptr] = new_sol
                self.result_list[self.path_ptr] = self.post_processor(new_sol)
                if(self.backward_flag):
                    if(self.path_ptr == 0):
                        self.curr_ptr = self.dumped_ptr + 1
                        self.curr_sol = self.value_list[self.dumped_ptr]
                        self.backward_flag = False
                    else:
                        self.path_ptr -= 1
                else:
                    self.path_ptr += 1


    def evolve_to_dest(self):
        while(self.path_ptr < self.param_path.shape[0]):
            self.evolve_next()

    def insert_midpoints(self):
        '''
            When the continuity check fails, a midpoint will be inserted between "param_ptr" and "param_ptr - 1"
        '''
        mid_point = (self.param_path[self.path_ptr - 1, :] + self.param_path[self.path_ptr,:]).reshape((1,-1))/2
        self.param_path = np.vstack([self.param_path[:self.path_ptr,:], mid_point, self.param_path[self.path_ptr:,:]])
        self.value_list.insert(self.path_ptr, [])
        self.result_list.insert(self.path_ptr, [])
        self.force_full_solver = True


    # def insert_midpoints(self):
    #     '''
    #         When a critical point is detected, add two midpoint of the adjacent intervals
    #         to avoid critical points

    #         When the insertion finished, self.path_ptr is pointed to the first changed point
    #     '''
    #     if(self.path_ptr == 0 ):
    #         # left endpoint: change the current parameters to the midpoint
    #         self.param_path[self.path_ptr,:] = (self.param_path[self.path_ptr, :] + self.param_path[self.path_ptr + 1,:])/2
    #     elif(self.path_ptr == self.param_path.shape[0] - 1):
    #         # right endpoint: similar to the left endpoint
    #         self.param_path[self.path_ptr,:] = (self.param_path[self.path_ptr, :] + self.param_path[self.path_ptr - 1, :])/2
    #     else:
    #         # get the two midpoints
    #         midpoint_left = (self.param_path[self.path_ptr-1,:] + self.param_path[self.path_ptr,:]).reshape((1,-1))/2
    #         midpoint_right = (self.param_path[self.path_ptr,:] + self.param_path[self.path_ptr + 1,:]).reshape((1,-1))/2
    #         self.param_path = np.vstack([self.param_path[:self.path_ptr,:], midpoint_left, midpoint_right, self.param_path[self.path_ptr+1:,:]])


class ParamEvolutionSolver:
    param_fun: ParamFun
    param_path: np.ndarray
    path_ptr: int
    
    def __init__(self, param_fun, root_solver) -> None:
        self.param_fun = param_fun
        self.root_solver = root_solver
    
    def set_path(self, param_path):
        self.param_path = param_path
        self.path_ptr = 0
    
    def evolve_to_dest(self, x0, **options):
        x = x0
        if('n_print' in options.keys()):
            n_print = options['n_print']
            del options['n_print']
            print_while_solve = True
        else:
            print_while_solve = False
        print_counter = n_print
        all_sols = []
        all_dets = []
        
        while(self.path_ptr < self.param_path.shape[0]):
            self.param_fun.set_params(self.param_path[self.path_ptr,:].copy())
            res = self.root_solver(self.param_fun, x, **options)
            x = res.x
            val, jac = self.param_fun(x)
            success = res.success
            if(not success):
                # critical point find
                self.insert_midpoints()
                continue
                # print("Warning: solution failed, cond(jac)=%f"%(cond(jac)))
                # print("det(jac) = %s"%(str(LA.det(jac))))
                # print("message:", res.message)
                # return (all_sols, success, all_dets)

            all_sols.append(x)
            all_dets.append(cond(jac))
           
            if(print_while_solve):
                if(print_counter):
                    print_counter -= 1
                else:
                    print_counter = n_print
                    print('param:',self.param_path[self.path_ptr,:])
                    print('Res:',val)
                    print('x:',x)
                    print('cond(jac):', cond(jac))
                    print()
            self.path_ptr += 1
        return (all_sols, success, all_dets)

    def insert_midpoints(self):
        '''
            When a critical point is detected, add two midpoint of the adjacent intervals
            to avoid critical points

            When the insertion finished, self.path_ptr is pointed to the first changed point
        '''
        if(self.path_ptr == 0 ):
            # left endpoint: change the current parameters to the midpoint
            self.param_path[self.path_ptr,:] = (self.param_path[self.path_ptr, :] + self.param_path[self.path_ptr + 1,:])/2
        elif(self.path_ptr == self.param_path.shape[0] - 1):
            # right endpoint: similar to the left endpoint
            self.param_path[self.path_ptr,:] = (self.param_path[self.path_ptr, :] + self.param_path[self.path_ptr - 1, :])/2
        else:
            # get the two midpoints
            midpoint_left = (self.param_path[self.path_ptr-1,:] + self.param_path[self.path_ptr,:]).reshape((1,-1))/2
            midpoint_right = (self.param_path[self.path_ptr,:] + self.param_path[self.path_ptr + 1,:]).reshape((1,-1))/2
            self.param_path = np.vstack([self.param_path[:self.path_ptr,:], midpoint_left, midpoint_right, self.param_path[self.path_ptr+1:,:]])


def find_loops_from_edges(edges: list[tuple[int]]) -> list[list[int]]:
    '''
        给定一系列边，用tuple[int]表示，返回所有的环.每个环用组成环的边在'edges'中的序号按顺序组成的list表示. 
        input: 
            edges: list[tuple[int]] 由(起点, 终点)表示的边组成的list
        output:
            all_loops: list[list[int]] 所有环的list
    '''
    all_loops = []
    visited = set()  # 记录已访问的节点
    # 创建一个字典，用于快速查找从某个起点出发的边
    edge_dict = {start: (end, idx) for idx, (start, end) in enumerate(edges)}

    for i in range(len(edges)):
        if i not in visited:
            # 开始追踪一个新的环
            current_loop = []
            current_node = i
            # 循环直到回到起始点
            while current_node not in visited:
                visited.add(current_node)
                next_node, edge_idx = edge_dict[current_node]
                current_loop.append(edge_idx)
                current_node = next_node
            all_loops.append(current_loop)

    return all_loops


def get_winding_number(x):
    '''
        calculate complex winding number of x
    '''

    phase_diff = np.zeros_like(x, dtype=float)
    phase_diff[:-1] = np.log(x[1:]/x[:-1]).imag
    phase_diff[-1] = np.log(x[0]/x[-1]).imag
    return (np.sum(phase_diff)/(2*np.pi))


class ExponentsFittingFun:
    direction_index: tuple[int]  # (b1, b2, b3, ...), bj = 0 for left, 1 for right
    x0: np.ndarray
    def __init__(self, direction_index, x0) -> None:
        self.direction_index = direction_index
        self.x0 = x0
    
    def __call__(self, x, A, *r) -> np.ndarray:
        expo = np.zeros(x.shape[0])
        for curr_dim in range(len(r)):
            expo += (x[:,curr_dim] - self.x0[curr_dim]) * r[curr_dim]
        f_val = A * np.exp(expo)
        return f_val
    
    def jacobian(self, x, A, *r) -> np.ndarray:
        f_jac = np.zeros((x.shape[0], len(r) + 1))
        expo = np.zeros(x.shape[0])
        for curr_dim in range(len(r)):
            expo += (x[:,curr_dim] - self.x0[curr_dim]) * r[curr_dim]
        
        f_jac[:,0] = np.exp(expo)
        for curr_dim in range(len(r)):
            f_jac[:,1+curr_dim] = A * f_jac[:,0] * (x[:,curr_dim] - self.x0[curr_dim])

        return f_jac


class ExponentsFitting:
    site_coords: list[list]
    N_list: list[int]
    site_num: int
    boundary_coord: list[list[int]]
    boundary_coord_id_list: list[int]

    def __init__(self, site_coords: list[list], N_list: list[int], site_num: int) -> None:
        self.site_num = site_num
        self.dim = len(site_coords[0])
        self.site_coords = site_coords
        self.N_list = N_list
        self.boundary_coord = [[] for _ in range(0x1 << self.dim)]
    
    def pick_boundary_coordinates(self, boundary_dist: int):
        '''
            Encoding: (b3,b2,b1), bi = 0 for boundary_list, = 1 for N_list[i] - boundary_list - 1
        '''

        # 1. Get boundary coordinates
        for curr_index in range(0x1 << self.dim):
            curr_index_copy = curr_index
            curr_coord = [0 for _ in range(self.dim)]
            for curr_dim in range(self.dim):
                curr_bit = curr_index_copy % 2
                if(curr_bit == 1):
                    curr_coord[curr_dim] = (self.N_list[curr_dim] - boundary_dist - 1)
                else:
                    curr_coord[curr_dim] = boundary_dist
                curr_index_copy >>= 1
            self.boundary_coord[curr_index] = curr_coord
        
        # 2. Find the indices in the coordinate list
        self.boundary_coord_id_list = [0 for _ in range(0x1 << self.dim)]
        for coord_id in range(len(self.site_coords)):
            curr_coord = self.site_coords[coord_id]
            for curr_index in range(0x1 << self.dim):
                curr_boundary_coord = self.boundary_coord[curr_index]
                if(np.allclose(curr_boundary_coord, curr_coord, atol=1e-10)):
                    self.boundary_coord_id_list[curr_index] = coord_id

    def get_max_corner(self, psi_mag, site_id):
        corner_mag = psi_mag[[j + site_id for j in self.boundary_coord_id_list]]

        max_index = np.argmax(corner_mag)
        x0 = np.asarray(self.boundary_coord[max_index])
        direction_flags = []
        new_index = max_index
        for _ in range(self.dim):
            direction_flags.append(max_index % 2)
            new_index >>= 1
        return max_index, direction_flags, x0
    
    def fit_exponents(self, psi_mag):
        r_list = []
        for curr_site_id in range(self.site_num):
            max_index, direction_flags, x0 = self.get_max_corner(psi_mag, curr_site_id)
            fit_fun = ExponentsFittingFun(direction_flags, x0)
            point_list = np.asarray(self.site_coords[curr_site_id::self.site_num])
            value_list = psi_mag[curr_site_id::self.site_num]
            res = optimize.curve_fit(fit_fun, point_list, value_list, 
                    [psi_mag[self.boundary_coord_id_list[max_index]] + curr_site_id] + [0] * self.dim, jac=fit_fun.jacobian)
            r_list.append(res[0][1:])
        r_list = np.mean(r_list, axis=0)
        return r_list