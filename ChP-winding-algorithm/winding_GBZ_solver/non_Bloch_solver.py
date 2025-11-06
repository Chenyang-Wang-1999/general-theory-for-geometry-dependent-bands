'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-12-30
Copyright © Department of Physics, Tsinghua University. All rights reserved

Get non-Bloch solutions for a given energy E
'''

from .util import *
import poly_tools as pt
import GBZ_manifold as Gm


def match_CP1_points(curr_sol, next_sol):
    if(isinstance(curr_sol, Gm.GBZPointVec)):
        result = Gm.GBZPointVec()
    elif(isinstance(curr_sol, Gm.CP1PointVec)):
        result = Gm.CP1PointVec()
    else:
        raise ValueError("unsupported type")
    Gm.match_point_list(curr_sol, next_sol, result)
    return result


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


class GBZConditionChecker:
    def __init__(self, char_poly_obj, rel_tol):
        if(isinstance(char_poly_obj, aGBZEquation1DCP1)):
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


def solve_GBZ(data_type:str, char_poly: pt.CLaurent, N_points: int = 200, 
               glue_tol: float = 0.1, distance_tol: float = 1e-1, GBZ_checker_tol: float = 1e-6):
    distance_checker = CP1DistanceChecker(tol=distance_tol)
    critical_checker = GBZContinuityChecker(tol=glue_tol)
    # critical_checker = lambda x,y,z: False
    if(data_type == 'GBZ'):
        ''' solve aGBZ in GBZ space '''
        aGBZ_eq = aGBZEquation1DGBZCP1(char_poly)
        post_processer = GBZConditionChecker(aGBZ_eq, GBZ_checker_tol)
        root_solver = BatchCP1Root(dtype=Gm.GBZPointData, jac=True)
        solver_control = BatchSolverController(
            phcpy_GBZCP1_full_solve, 
            root_solver,
            aGBZ_eq,
            distance_checker,
            critical_checker,
            post_processer,
            match_CP1_points,
            check_full_solver_default
        )
    elif(data_type == 'CP1'):
        ''' solve aGBZ in CP1 product space'''
        aGBZ_eq = aGBZEquation1DCP1(char_poly)
        post_processer = GBZConditionChecker(aGBZ_eq, GBZ_checker_tol)
        root_solver = BatchCP1Root(dtype=Gm.CP1PointData, jac=True)
        solver_control = BatchSolverController(
            phcpy_CP1_full_solve, 
            root_solver,
            aGBZ_eq,
            distance_checker,
            critical_checker,
            post_processer,
            match_CP1_points,
            check_full_solver_default
        )

    # phi_loop = np.linspace(np.pi/(N_points+1), np.pi - np.pi/(N_points+1), N_points).reshape((-1,1))
    phi_loop = np.linspace(0, np.pi - 1e-12, N_points).reshape((-1,1))

    solver_control.set_path(phi_loop)
    solver_control.evolve_to_dest()

    phi_loop = solver_control.param_path
    return solver_control.result_list, phi_loop, solver_control.value_list


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


def expand_solution(GBZ_data: list[Gm.GBZPointVec], aGBZ_data: list[Gm.GBZPointVec], param_list: np.ndarray):
    # Expand the GBZ and aGBZ data and change the parameter to phi/2
    N_points = len(GBZ_data)
    vec_type = aGBZ_data[0].__class__
    for i in range(N_points):
        if len(aGBZ_data[i]) != 0:
            point_type = aGBZ_data[i][0].__class__
            break

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


def get_GBZ_segments(GBZ_data: list, glue_tol: float):
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
            temp_point = seg[-1]
            new_point = temp_point

            temp_seg_tails.append(new_point)
        if(len(temp_segment_stack) <= len(GBZ_data[j])):
            all_pairs, unpaired1, unpaired2 = match_point_list(temp_seg_tails, GBZ_data[j], vec_type, glue_tol)
            new_segment_stack = []
            for curr_pair in all_pairs:
                temp_segment_stack[curr_pair[0]].append(GBZ_data[j][curr_pair[1]])
                new_segment_stack.append(temp_segment_stack[curr_pair[0]])
        else:
            all_pairs, unpaired2, unpaired1 = match_point_list(GBZ_data[j], temp_seg_tails, vec_type, glue_tol)
            new_segment_stack = []
            for curr_pair in all_pairs:
                temp_segment_stack[curr_pair[1]].append(GBZ_data[j][curr_pair[0]])
                new_segment_stack.append(temp_segment_stack[curr_pair[1]])
        for curr_id1 in unpaired1:
            staged_segment_stack.append(temp_segment_stack[curr_id1])
        for curr_id2 in unpaired2:
            new_seg = vec_type()
            new_seg.append(GBZ_data[j][curr_id2])
            new_segment_stack.append(new_seg)

        temp_segment_stack = new_segment_stack

    staged_segment_stack += temp_segment_stack
    return staged_segment_stack


def GBZ_segments_to_loops(
    all_segments: list[Gm.GBZPointVec],
    glue_tol: float
):
    ''' Glue segments to loops, this function will destroy the list 'all_segments' '''
    all_segments = sorted(all_segments, key=lambda x: -len(x))

    has_broken_segs = False
    all_loops = []
    curr_loop = [all_segments.pop(0)]

    # Check type
    if isinstance(curr_loop[0], Gm.GBZPointVec):
        curr_point_type = Gm.GBZPointData
        curr_vec_type = Gm.GBZPointVec
    elif isinstance(curr_loop[0], Gm.CP1PointVec):
        curr_point_type = Gm.CP1PointData
        curr_vec_type = Gm.CP1PointVec
    else:
        raise ValueError("unsupported type")
 
    while(all_segments):
        curr_tail = curr_loop[-1][-1]
        all_heads = curr_vec_type()
        all_heads.append(curr_loop[0][0])
        for seg in all_segments:
            all_heads.append(seg[0])

        min_dist, min_id = Gm.calculate_min_distance(curr_tail, all_heads)
        if min_dist > glue_tol:
            ''' It is a segment '''
            has_broken_segs = True
            all_loops.append(curr_loop)
            curr_loop = [all_segments.pop(0)]
        else:
            if(min_id == 0):
                curr_loop[-1].append(curr_loop[0][0])
                all_loops.append(curr_loop)
                curr_loop = [all_segments.pop(0)]
            else:
                next_seg = all_segments.pop(min_id - 1)
                curr_loop.append(next_seg)
    all_loops.append(curr_loop)

    # Glue loops
    glued_loops = []
    for loop in all_loops:
        glued_loop = curr_vec_type()
        for seg in loop:
            for point in seg:
                glued_loop.append(point)
        glued_loops.append(glued_loop)

    return glued_loops, has_broken_segs


def remove_duplicated_points(
    seg_list: list[Gm.CP1PointVec],
    glue_tol: float
):
    ''' Remove duplicated points in a list of segments '''
    if isinstance(seg_list[0], Gm.GBZPointVec):
        vec_type = Gm.GBZPointVec
    elif isinstance(seg_list[0], Gm.CP1PointVec):
        vec_type = Gm.CP1PointVec
    else:
        raise ValueError("unsupported type")
    
    seg_list = sorted(seg_list, key=lambda x: len(x))

    new_seg_list = []
    while seg_list:
        curr_seg = seg_list.pop(0)
        for seg in seg_list:
            has_duplicates, sliced_indices = Gm.remove_duplicated_points(
                curr_seg, seg, glue_tol
            )
            if has_duplicates:
                temp_seg = curr_seg
                curr_seg = vec_type()
                if sliced_indices:
                    for point_idx in range(sliced_indices[0][0], sliced_indices[0][1]):
                        curr_seg.append(temp_seg[point_idx])
                    for slice_range in sliced_indices[1:]:
                        new_seg = vec_type()
                        for point_idx in range(slice_range[0], slice_range[1]):
                            new_seg.append(temp_seg[point_idx])
                        seg_list.insert(0, new_seg)
                else:
                    break

        if curr_seg:
            new_seg_list.append(curr_seg)

    return new_seg_list


def solve_non_Bloch_loops(
    f: pt.CLaurent,
    E_ref: complex,
    N_points: int = 200, 
    glue_tol: list[float] = 0.1, 
    distance_tol: float = 1e-1, 
    GBZ_checker_tol: float = 1e-6
):
    ''' Main function for non-Bloch solutions.'''
    char_poly = f.partial_eval(
        pt.CScalarVec([E_ref]), pt.CIndexVec([0]), pt.CIndexVec([1,2])
    )

    if isinstance(glue_tol, float):
        glue_tol = [glue_tol, 1.1 * glue_tol, 0.5 * glue_tol, 10 * glue_tol]

    # Solve non-Bloch loops
    GBZ_loops, phi_loop, aGBZ_loops = solve_GBZ(
        "CP1", char_poly, N_points, glue_tol[0],
        distance_tol, GBZ_checker_tol
    )
    GBZ_loops, phi_loop, aGBZ_loops = expand_solution(
        GBZ_loops, aGBZ_loops, phi_loop)
    all_segments = get_GBZ_segments(GBZ_loops, glue_tol[1])
    all_loops = remove_duplicated_points(all_segments, glue_tol[2])
    all_loops, has_broken_segs = GBZ_segments_to_loops(all_segments, glue_tol[3])
    # for _ in range(10):
    #     if not has_broken_segs:
    #         break
    #     all_loops = remove_duplicated_points(all_loops, glue_tol[2])
    #     all_loops, has_broken_segs = GBZ_segments_to_loops(all_loops, glue_tol[3])

    return all_loops, has_broken_segs


def convert_to_complex_numbers(data):
    ''' Convert the data under the format of complex numbers '''
    if isinstance(data, Gm.cpp_GBZPointData):
        Gm.GBZ_to_chart(data, [0])
        return data.coords
    elif isinstance(data, Gm.cpp_CP1PointData):
        Gm.CP1_to_chart(data, [0,0])
        return data.coords
    else:
        return [convert_to_complex_numbers(item) for item in data]


def get_disconnect_intervals(beta_intervals: list[list[float]]):
    ''' Given several intervals, return the disconnected part '''    
    # 1. Sort intervals
    lower_bound_list = [x[0] for x in beta_intervals]
    seq = np.argsort(lower_bound_list)
    sorted_beta_intervals = [beta_intervals[j] for j in seq]

    # 2. Merge overlaped intervals
    merged_intervals = []
    curr_interval = sorted_beta_intervals.pop(0)
    while(sorted_beta_intervals):
        if(curr_interval[1] >= sorted_beta_intervals[0][0]):
            new_interval = sorted_beta_intervals.pop(0)
            curr_interval[1] = max(curr_interval[1], new_interval[1])
        else:
            merged_intervals.append(curr_interval)
            curr_interval = sorted_beta_intervals.pop(0)
    merged_intervals.append(curr_interval)
    return merged_intervals


def get_holes_from_loops(GBZ_loops: list[list[list[complex]]]):
    ''' Get the position of holes for winding number calculation '''
    beta_norm_intervals = []
    for loop in GBZ_loops:
        curr_norm_list = [abs(point[0]) for point in loop]
        beta_norm_intervals.append(
            [min(curr_norm_list), max(curr_norm_list)]
        )

    disconnected_intervals = get_disconnect_intervals(beta_norm_intervals)
    all_holes = []
    for j in range(len(disconnected_intervals) - 1):
        all_holes.append(
            (disconnected_intervals[j][1] + disconnected_intervals[j+1][0])/2
        )
    return all_holes, disconnected_intervals
