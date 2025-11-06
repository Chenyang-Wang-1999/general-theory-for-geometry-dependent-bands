'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-01-25
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import poly_tools as pt
import numpy as np
from partial_GBZ_solver.util import *
from partial_GBZ_solver import aGBZ
import GBZ_manifold as Gm
from .GBZ_diff import *
from .GBZ_diff import *

def check_GBZ_condition(char_poly:pt.CLaurent, beta:complex, E:complex, rel_tol:float):
    '''
        check whether the data (beta, E) satisfies the GBZ condition
    '''

    # 1. solve equation by np.root
    equation_1d = char_poly.partial_eval(
        pt.CVarScalarVec([E]), pt.CIndexVec([0]), pt.CIndexVec([1])
    )
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    equation_1d.num.batch_get_data(coeffs, degs)

    max_deg = max(degs)
    equation_1d_nparray = np.zeros(max_deg + 1, dtype=complex)

    for term_id in range(len(coeffs)):
        equation_1d_nparray[max_deg - degs[term_id]] = coeffs[term_id]
    
    all_roots = np.roots(equation_1d_nparray)
    all_roots_norm = np.sort(np.abs(all_roots))

    # 2. compare the norms
    denom_deg = equation_1d.denom_orders[0]
    if(abs(abs(beta)-all_roots_norm[denom_deg - 1]) > rel_tol * abs(beta)):
        return False
    if(abs(abs(beta)-all_roots_norm[denom_deg]) > rel_tol * abs(beta)):
        return False
    
    return True

def check_GBZ_condition_with_offset(char_poly:pt.CLaurent, beta:complex, E:complex, rel_tol:float, offset: int):
    '''
        check whether the data (beta, E) satisfies the aGBZ condition
    '''

    # 1. solve equation by np.root
    equation_1d = char_poly.partial_eval(
        pt.CVarScalarVec([E]), pt.CIndexVec([0]), pt.CIndexVec([1])
    )
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    equation_1d.num.batch_get_data(coeffs, degs)

    max_deg = max(degs)
    equation_1d_nparray = np.zeros(max_deg + 1, dtype=complex)

    for term_id in range(len(coeffs)):
        equation_1d_nparray[max_deg - degs[term_id]] = coeffs[term_id]
    
    all_roots = np.roots(equation_1d_nparray)
    all_roots_norm = np.sort(np.abs(all_roots))

    # 2. compare the norms
    denom_deg = equation_1d.denom_orders[0] + offset
    if(abs(abs(beta)-all_roots_norm[denom_deg - 1]) > rel_tol * abs(beta)):
        return False
    if(abs(abs(beta)-all_roots_norm[denom_deg]) > rel_tol * abs(beta)):
        return False
    
    return True


def pick_GBZ(char_poly_obj:aGBZ.aGBZEquation1DCP1, aGBZ_data: list, rel_tol: float, data_type:str='GBZ'):
    if(data_type == 'GBZ'):
        dtype = Gm.GBZPointVec
    elif(data_type == 'CP1'):
        dtype = Gm.CP1PointVec
    else:
        raise ValueError("unsupported type")

    GBZ_data = [dtype() for _ in range(len(aGBZ_data))]
    for j, curr_vec in enumerate(aGBZ_data):
        for point in curr_vec:
            GBZ_flag = check_GBZ_condition(
                char_poly_obj.get_callable(point.chart_labels).char_poly,
                point.coords[1],
                point.coords[0],
                rel_tol
            )
            if(GBZ_flag):
                GBZ_data[j].append(point)
    return GBZ_data

class GBZConditionChecker:
    def __init__(self, char_poly_obj, rel_tol):
        if(isinstance(char_poly_obj, aGBZ.aGBZEquation1DCP1)):
            self.data_type = 'CP1'
            self.dtype = Gm.CP1PointData
            self.vec_type = Gm.CP1PointVec
        else:
            self.data_type = 'GBZ'
            self.dtype = Gm.GBZPointData
            self.vec_type = Gm.GBZPointVec
        self.char_poly_obj = char_poly_obj
        self.rel_tol = rel_tol
    
    def __call__(self, raw_point_vec: Gm.CP1PointVec) -> Gm.CP1PointVec:
        result_point_vec = self.vec_type()
        for point in raw_point_vec:
            GBZ_flag = check_GBZ_condition(
                self.char_poly_obj.get_callable(point.chart_labels).char_poly,
                point.coords[1],
                point.coords[0],
                self.rel_tol
            )
            if(GBZ_flag):
                result_point_vec.append(point)
        return result_point_vec

def match_point_list(point_vec1, point_vec2, vec_type, glue_tol: float):
    ''' Match point_vec1 to point_vec2, assuming that len(point_vec1) <= len(point_vec2) '''
    index_stack1 = [j for j in range(len(point_vec1))]
    index_stack2 = [j for j in range(len(point_vec2))]
    all_pairs = []
    unpaired_stack1 = []

    while(index_stack1):
        curr_points2 = vec_type()
        for curr_id2 in index_stack2:
            curr_points2.append(point_vec2[curr_id2])
        curr_id1 = index_stack1.pop(0)
        min_dist, min_id = Gm.calculate_min_distance(point_vec1[curr_id1], curr_points2)
        if(min_dist < glue_tol):
            curr_id2 = index_stack2.pop(min_id)
            all_pairs.append((curr_id1, curr_id2))
        else:
            unpaired_stack1.append(curr_id1)
    return all_pairs, unpaired_stack1, index_stack2

def solve_GBZ(data_type:str, char_poly: pt.CLaurent, N_points: int = 200, 
               glue_tol: float = 0.1, distance_tol: float = 1e-1, GBZ_checker_tol: float = 1e-6):
    distance_checker = aGBZ.CP1DistanceChecker(tol=distance_tol)
    critical_checker = aGBZ.GBZContinuityChecker(tol=glue_tol)
    # critical_checker = lambda x,y,z: False
    if(data_type == 'GBZ'):
        ''' solve aGBZ in GBZ space '''
        aGBZ_eq = aGBZEquation1DGBZCP1(char_poly)
        post_processer = GBZConditionChecker(aGBZ_eq, GBZ_checker_tol)
        root_solver = aGBZ.BatchCP1Root(dtype=Gm.GBZPointData, jac=True)
        solver_control = BatchSolverController(
            aGBZ.phcpy_GBZCP1_full_solve, 
            root_solver,
            aGBZ_eq,
            distance_checker,
            critical_checker,
            post_processer,
            aGBZ.match_CP1_points,
            check_full_solver_default
        )
    elif(data_type == 'CP1'):
        ''' solve aGBZ in CP1 product space'''
        aGBZ_eq = aGBZEquation1DCP1(char_poly)
        post_processer = GBZConditionChecker(aGBZ_eq, GBZ_checker_tol)
        root_solver = aGBZ.BatchCP1Root(dtype=Gm.CP1PointData, jac=True)
        solver_control = BatchSolverController(
            aGBZ.phcpy_CP1_full_solve, 
            root_solver,
            aGBZ_eq,
            distance_checker,
            critical_checker,
            post_processer,
            aGBZ.match_CP1_points,
            check_full_solver_default
        )

    # phi_loop = np.linspace(np.pi/(N_points+1), np.pi - np.pi/(N_points+1), N_points).reshape((-1,1))
    phi_loop = np.linspace(0, np.pi - 1e-12, N_points).reshape((-1,1))

    solver_control.set_path(phi_loop)
    solver_control.evolve_to_dest()

    phi_loop = solver_control.param_path
    return solver_control.result_list, phi_loop, solver_control.value_list
 

def solve_aGBZ(data_type:str, char_poly: pt.CLaurent, N_points: int = 200, 
               glue_tol: float = 0.1, distance_tol: float = 1e-1):
    distance_checker = aGBZ.CP1DistanceChecker(tol=distance_tol)
    critical_checker = aGBZ.GBZContinuityChecker(tol=glue_tol)
    # critical_checker = lambda x,y,z: False
    post_processer = aGBZ.DefaultPostProcessor()
    if(data_type == 'GBZ'):
        ''' solve aGBZ in GBZ space '''
        aGBZ_eq = aGBZEquation1DGBZCP1(char_poly)
        root_solver = aGBZ.BatchCP1Root(dtype=Gm.GBZPointData, jac=True)
        solver_control = BatchSolverController(
            aGBZ.phcpy_GBZCP1_full_solve, 
            root_solver,
            aGBZ_eq,
            distance_checker,
            critical_checker,
            post_processer,
            aGBZ.match_CP1_points,
            check_full_solver_default
        )
    elif(data_type == 'CP1'):
        ''' solve aGBZ in CP1 product space'''
        aGBZ_eq = aGBZEquation1DCP1(char_poly)
        root_solver = aGBZ.BatchCP1Root(dtype=Gm.CP1PointData, jac=True)
        solver_control = BatchSolverController(
            aGBZ.phcpy_CP1_full_solve, 
            root_solver,
            aGBZ_eq,
            distance_checker,
            critical_checker,
            post_processer,
            aGBZ.match_CP1_points,
            check_full_solver_default
        )

    # phi_loop = np.linspace(np.pi/(N_points+1), np.pi - np.pi/(N_points+1), N_points).reshape((-1,1))
    phi_loop = np.linspace(0, np.pi - 1e-12, N_points).reshape((-1,1))

    solver_control.set_path(phi_loop)
    solver_control.evolve_to_dest()

    phi_loop = solver_control.param_path
    return solver_control.result_list, phi_loop, solver_control.value_list
 

def get_GBZ_segments(GBZ_data: list, params: np.ndarray, glue_tol: float,
                     char_poly_obj: aGBZEquation1DCP1 = None):
    ''' Glue the GBZ points picked by 'pick_GBZ' to segments
        Algorithm: 
            1. Create several segments to a temp stack according to the first point list
            2. For a set of new points, match each points to the segments
            3. If none of the new points match a segment, then move the segment to a new stack
            4. If a new point cannot match to either of the segments, create a new segment
    '''
    if(isinstance(GBZ_data[0], Gm.GBZPointVec)):
        vec_type = Gm.GBZPointVec
        data_type = 'GBZ'
    elif(isinstance(GBZ_data[0], Gm.CP1PointVec)):
        vec_type = Gm.CP1PointVec
        data_type = 'CP1'
    else:
        raise ValueError("unsupported type")

    temp_segment_stack = []
    staged_segment_stack = []

    for j in range(len(GBZ_data)):
        temp_seg_tails = vec_type()
        for seg in temp_segment_stack:
            temp_point = seg.point_vec[-1]
            temp_param = seg.param_list[-1][0]
            if(char_poly_obj is None):
                new_point = temp_point
            else:
                ds = params[j][0] - temp_param
                dp = get_GBZ_diff_CP1(char_poly_obj, temp_point, temp_param)
                if(dp is None):
                    new_point = temp_point
                else:
                    new_coords = [temp_point.coords[j] + dp[j]*ds for j in range(2)]
                    new_point = temp_point.__class__()
                    new_point.coords = new_coords
                    new_point.chart_labels = temp_point.chart_labels
            temp_seg_tails.append(new_point)
        if(len(temp_segment_stack) <= len(GBZ_data[j])):
            all_pairs, unpaired1, unpaired2 = match_point_list(temp_seg_tails, GBZ_data[j], vec_type, glue_tol)
            new_segment_stack = []
            for curr_pair in all_pairs:
                temp_segment_stack[curr_pair[0]].append_point(GBZ_data[j][curr_pair[1]], params[j,:])
                new_segment_stack.append(temp_segment_stack[curr_pair[0]])
        else:
            all_pairs, unpaired2, unpaired1 = match_point_list(GBZ_data[j], temp_seg_tails, vec_type, glue_tol)
            new_segment_stack = []
            for curr_pair in all_pairs:
                temp_segment_stack[curr_pair[1]].append_point(GBZ_data[j][curr_pair[0]], params[j,:])
                new_segment_stack.append(temp_segment_stack[curr_pair[1]])
        for curr_id1 in unpaired1:
            staged_segment_stack.append(temp_segment_stack[curr_id1])
        for curr_id2 in unpaired2:
            new_seg = Gm.GBZSegment(data_type)
            new_seg.append_point(GBZ_data[j][curr_id2], params[j,:])
            new_segment_stack.append(new_seg)
        
        temp_segment_stack = new_segment_stack
    
    staged_segment_stack += temp_segment_stack
    return staged_segment_stack
    
def expand_solution(GBZ_data: list[Gm.GBZPointVec], aGBZ_data: list[Gm.GBZPointVec], param_list: np.ndarray):
    # Expand the GBZ and aGBZ data and change the parameter to phi/2
    N_points = len(GBZ_data)
    vec_type = aGBZ_data[0].__class__
    point_type = aGBZ_data[0][0].__class__

    new_GBZ_data = []
    new_aGBZ_data = []
    new_param = []
    for param_id in range(N_points-1, 0, -1):
        curr_s = -param_list[param_id]/2
        new_GBZ_data.append(GBZ_data[param_id])
        new_aGBZ_data.append(aGBZ_data[param_id])
        new_param.append(curr_s)

    for param_id in range(N_points):
        curr_s = param_list[param_id]/2
        # GBZ
        curr_GBZ_vec = vec_type()
        for point in GBZ_data[param_id]:
            new_coords = point.coords
            if(point.chart_labels[-1] == 0):
                new_coords[-1] *= np.exp(1j * param_list[param_id,0])
            else:
                new_coords[-1] *= np.exp(-1j * param_list[param_id,0])
            new_point = point_type()
            new_point.coords = new_coords
            new_point.chart_labels = point.chart_labels
            curr_GBZ_vec.append(
                new_point
            )
        
        # aGBZ
        curr_aGBZ_vec = vec_type()
        for point in aGBZ_data[param_id]:
            new_coords = point.coords
            if(point.chart_labels[-1] == 0):
                new_coords[-1] *= np.exp(1j * param_list[param_id])
            else:
                new_coords[-1] *= np.exp(-1j * param_list[param_id])
            new_point = point_type()
            new_point.coords = new_coords
            new_point.chart_labels = point.chart_labels
            curr_aGBZ_vec.append(
                new_point
            )
        
        new_GBZ_data.append(curr_GBZ_vec)
        new_aGBZ_data.append(curr_aGBZ_vec)
        new_param.append(curr_s)
    
    return new_GBZ_data, np.asarray(new_param), new_aGBZ_data

def expand_segments(all_segs: list[Gm.GBZSegment]):
    new_segs = []
    for seg in all_segs:
        curr_new_seg1 = Gm.GBZSegment(seg.data_type)
        curr_new_seg2 = Gm.GBZSegment(seg.data_type)
        if(seg.data_type == 'GBZ'):
            curr_point_type = Gm.GBZPointData
        else:
            curr_point_type = Gm.CP1PointData
        for j in range(len(seg.point_vec)):
            curr_new_seg1.append_point(seg.point_vec[-(j+1)], -seg.param_list[-(j+1)]/2)

            new_coords = seg.point_vec[j].coords
            new_labels = seg.point_vec[j].chart_labels
            if(new_labels[-1] == 1):
                new_coords[-1] *= np.exp(-1j * seg.param_list[j][0])
            else:
                new_coords[-1] *= np.exp(1j * seg.param_list[j][0])
            new_point = curr_point_type(new_coords, new_labels)
            curr_new_seg2.append_point(new_point, seg.param_list[j]/2)
        if(abs(curr_new_seg2.param_list[0][0]) < 1e-15):
            for point_id in range(1, len(curr_new_seg2.point_vec)):
                curr_new_seg1.append_point(curr_new_seg2.point_vec[point_id], 
                                           curr_new_seg2.param_list[point_id])
            new_segs.append(curr_new_seg1)
        else:
            new_segs.append(curr_new_seg1)
            new_segs.append(curr_new_seg2)
    return new_segs
        
def GBZ_segments_to_loops(all_segments: list[Gm.GBZSegment]):
    ''' Glue segments to loops, this function will destroy the list 'all_segments' '''
    all_loops = []
    curr_loop = [all_segments.pop(0)]
    if(curr_loop[0].data_type == 'GBZ'):
        curr_point_type = Gm.GBZPointData
        curr_vec_type = Gm.GBZPointVec
    else:
        curr_point_type = Gm.CP1PointData
        curr_vec_type = Gm.CP1PointVec
 
    while(all_segments):
        curr_tail = curr_loop[-1].point_vec[-1]
        all_heads = curr_vec_type()
        all_heads.append(curr_loop[0].point_vec[0])
        for seg in all_segments:
            all_heads.append(seg.point_vec[0])

        _, min_id = Gm.calculate_min_distance(curr_tail, all_heads)
        if(min_id == 0):
            all_loops.append(curr_loop)
            curr_loop = [all_segments.pop(0)]
        else:
            next_seg = all_segments.pop(min_id - 1)
            curr_loop.append(next_seg)
    all_loops.append(curr_loop)
    return all_loops

def insert_intersection(tail: Gm.CP1PointData, head: Gm.CP1PointData,
            s_tail: float, s_head: float,
            char_poly_obj: aGBZ.aGBZEquation1DCP1, glue_tol: float, zero_tol: float):
    ''' For a given tail and a given head, insert a new point 
        by linear interpolation 
    '''
    if(Gm.get_distance(tail, head) < glue_tol):
        return [],[]
    
    # 1. convert head to the chart of tail
    if(isinstance(char_poly_obj, aGBZ.aGBZEquation1DCP1)):
        dtype = Gm.CP1PointData
        Gm.CP1_to_chart(head, tail.chart_labels)
    else:
        dtype = Gm.GBZPointData
        Gm.GBZ_to_chart(head, tail.chart_labels)
    
    # 2. calculate derivative
    dp1 = get_GBZ_diff_CP1(char_poly_obj, tail, s_tail, zero_tol)
    dp2 = get_GBZ_diff_CP1(char_poly_obj, head, s_head, zero_tol)
    p1 = np.asarray(tail.coords)
    p2 = np.asarray(head.coords)
    p1_re = np.hstack([p1.real, p1.imag]).reshape((-1,1))
    p2_re = np.hstack([p2.real, p2.imag]).reshape((-1,1))

    if(dp1 is not None):
        dp1_re = np.hstack([dp1.real, dp1.imag]).reshape((-1,1))
    if(dp2 is not None):
        dp2_re = np.hstack([dp2.real, dp2.imag]).reshape((-1,1))

    if(dp1 is None):
        if(dp2 is None):
            raise ValueError("Both ends are critical ")
        ds2 = la.lstsq(dp2_re, p1_re - p2_re)[0].flatten()[0]
        ds1 = 0
        new_points = (
            dtype(tail.coords, tail.chart_labels), 
            dtype([p2[j] + dp2[j] * ds2 for j in range(2)], tail.chart_labels)
        )
    elif(dp2 is None):
        ds1 = la.lstsq(dp1_re, p2_re - p1_re)[0].flatten()[0]
        ds2 = 0
        new_points = (
            dtype([p1[j] + dp1[j] * ds1 for j in range(2)], tail.chart_labels),
            dtype(head.coords, head.chart_labels)
        )
    else:
        ds = la.lstsq(
            np.column_stack([dp1_re.flatten(), -dp2_re.flatten()]),
            p2_re - p1_re
        )[0].flatten()
        ds1 = ds[0]
        ds2 = ds[1]
        new_points = (
            dtype([p1[j] + dp1[j] * ds1 for j in range(2)], tail.chart_labels),
            dtype([p2[j] + dp2[j] * ds2 for j in range(2)], tail.chart_labels)
        )
    return (ds1, ds2), new_points
    
def iterative_get_intersection(tail: Gm.CP1PointData, head: Gm.CP1PointData,
            s_tail: float, s_head: float,
            char_poly_obj: aGBZ.aGBZEquation1DCP1, glue_tol: float, zero_tol: float,
            max_iter: int = 30):
    ''' Iterative solve the intersections '''
    if(isinstance(char_poly_obj, aGBZ.aGBZEquation1DCP1)):
        dtype = Gm.CP1PointData
    else:
        dtype = Gm.GBZPointData
    
    root_solver = aGBZ.BatchCP1Root(dtype, jac=True)

    for _ in range(max_iter):
        ds_vec, new_point = insert_intersection(
            tail, head, s_tail, s_head, char_poly_obj, glue_tol, zero_tol)
        if(not ds_vec):
            break

        s_tail += ds_vec[0]
        s_head += ds_vec[1]

        char_poly_obj.set_params([-2 * s_tail,])
        tail = root_solver.solve_one(char_poly_obj, new_point[0])

        char_poly_obj.set_params([-2 * s_head,])
        head = root_solver.solve_one(char_poly_obj, new_point[1])

    return (s_tail, s_head), (tail, head)
    
def glue_loop(loop: list[Gm.GBZSegment], char_poly_obj: aGBZ.aGBZEquation1DCP1, 
              glue_tol: float = 1e-10, zero_tol: float = 1e-14, max_dist: float = 0.05):
    ''' Glue the loop by modify the tail and head to the intersection '''
    raise DeprecationWarning("The function glue_loop is not stable")
    for loop_id in range(len(loop)):
        curr_id = loop_id
        next_id = (loop_id + 1) % (len(loop))
        tail = loop[curr_id].point_vec[-1]
        s_tail = loop[curr_id].param_list[-1][0]
        head = loop[next_id].point_vec[0]
        s_head = loop[next_id].param_list[0][0]

        print("Current pair:", s_tail, s_head)
        print("Current distance:", Gm.get_distance(tail, head))
        # s_list, new_points = iterative_get_intersection(
        #     tail, head, s_tail, s_head, char_poly_obj, glue_tol, zero_tol     
        # )
        ds_list, new_points = insert_intersection(
            tail, head, s_tail, s_head, char_poly_obj, glue_tol, zero_tol)
        if(not ds_list):
            s_list = [s_tail, s_head]
            new_points = (tail, head)
        else:
            s_list = [s_tail + ds_list[0], s_head + ds_list[1]]
        if(Gm.get_distance(new_points[0], new_points[1]) > max_dist):
            return -1
        if(Gm.get_distance(new_points[0], loop[curr_id].point_vec[-1]) > max_dist):
            return -2
        if(Gm.get_distance(new_points[1], loop[next_id].point_vec[0]) > max_dist):
            return -3
        loop[curr_id].point_vec[-1] = new_points[0]
        loop[curr_id].param_list[-1] = [s_list[0],]
        loop[next_id].point_vec[0] = new_points[1]
        loop[next_id].param_list[0] = [s_list[1],]
    return 0

def get_global_coords(loop: list[Gm.GBZSegment], char_poly_obj: aGBZ.aGBZEquation1DCP1,
                      glue_tol=1e-8, zero_tol=1e-14):
    new_seg = Gm.GBZSegment(loop[0].data_type)
    for seg in loop:
        if(len(new_seg.point_vec)>0):
            ds, _ = insert_intersection(
                new_seg.point_vec[-1], 
                seg.point_vec[0], 
                curr_s_tail, 
                seg.param_list[0][0], char_poly_obj, glue_tol, zero_tol)
            if(ds):
                curr_offset = new_seg.param_list[-1][0] + ds[0] - seg.param_list[0][0] - ds[1]
            else:
                curr_offset = new_seg.param_list[-1][0] - seg.param_list[0][0] 
        else:
            curr_offset = 0 * seg.param_list[0][0]
        for j in range(len(seg.point_vec)):
            new_seg.append_point(seg.point_vec[j], [seg.param_list[j][0] + curr_offset])
        curr_s_tail = seg.param_list[-1][0]

    ds, _ = insert_intersection(
        new_seg.point_vec[-1], 
        new_seg.point_vec[0], 
        curr_s_tail, 
        loop[0].param_list[0][0], char_poly_obj, glue_tol, zero_tol)
    if(ds):
        new_seg.append_point(new_seg.point_vec[0], new_seg.param_list[-1] + ds[0] - ds[1])

    return new_seg

def normalize_segment_coords(loop: Gm.GBZSegment):
    ''' Normalize the coordinates, '''
    x0 = loop.param_list[0][0]
    for j in range(len(loop.param_list)):
        loop.param_list[j][0] -= x0
    
    M = loop.param_list[-1][0]
    for j in range(len(loop.param_list)):
        loop.param_list[j][0] /= M

def glue_with_local_coords(loop: list[Gm.GBZSegment]):
    new_seg = Gm.GBZSegment(loop[0].data_type)
    for seg in loop:
        for j in range(len(seg.point_vec)):
            new_seg.append_point(seg.point_vec[j], seg.param_list[j])
    return new_seg


def batch_get_global(all_loops: list[list[Gm.GBZSegment]],
                     char_poly_obj: aGBZ.aGBZEquation1DCP1):
    new_loops = []
    for loop in all_loops:
        new_loops.append(get_global_coords(loop, char_poly_obj))
    return new_loops
    
def batch_glue(all_loops: list[list[Gm.GBZSegment]]):
    new_loops = []
    for loop in all_loops:
        new_loops.append(glue_with_local_coords(loop))
    return new_loops
    
def get_another_point(point: Gm.CP1PointData, s_param: float):
    point_class = point.__class__
    new_coords = point.coords
    if(point.chart_labels[-1] == 0):
        new_coords[-1] *= np.exp(-2j * s_param)
    else:
        new_coords[-1] *= np.exp(2j * s_param)
    new_point = point_class()
    new_point.coords = new_coords
    new_point.chart_labels = point.chart_labels
    return new_point


def get_solution_index(
    poly: pt.CLaurent,
    E: complex,
    beta: complex
):
    beta_eq = poly.partial_eval(
        pt.CScalarVec([E]), 
        pt.CIndexVec([0]),
        pt.CIndexVec([1]))
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    beta_eq.num.batch_get_data(coeffs, degs)
    max_deg = max(degs)
    beta_eq_arr = np.zeros(max_deg + 1, dtype=complex)
    for term_id, curr_coeff in enumerate(coeffs):
        beta_eq_arr[max_deg - degs[term_id]] = curr_coeff
    
    # solve
    all_sols = np.roots(beta_eq_arr)
    all_sols_norm = np.abs(all_sols)
    sol_seq = np.argsort(all_sols_norm)
    all_sols = all_sols[sol_seq]
    all_sols_norm = all_sols_norm[sol_seq]

    # find index of current point in 'all_sols'
    sol_diff = all_sols - beta
    min_id = np.argmin(np.abs(sol_diff))

    while(min_id != 0 and abs(all_sols_norm[min_id - 1] - all_sols_norm[min_id]) < 1e-6):
        min_id -= 1

    return (min_id - beta_eq.denom_orders[0] + 1)
