'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-04-06
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' Module for calculating partial GBZ '''

import GBZ_manifold as Gm
import poly_tools as pt
from .aGBZ import *
from .GBZ_solver import *


def find_point_in_subGBZ(partial_GBZ_loop: Gm.GBZSegment, critical_point: Gm.GBZPointData):
    ''' Given a point, find the position of the critical point and minimum index '''
    min_dist, min_id = Gm.calculate_min_distance(critical_point, partial_GBZ_loop.point_vec)
    return min_dist, min_id


def find_point_in_GBZ(partial_GBZ_list: list[Gm.GBZSegment], critical_point: Gm.GBZPointData):
    ''' Given a point, find its GBZ index and coordinate in the GBZ_list '''
    dist_list = np.zeros(len(partial_GBZ_list))
    min_ind_list = np.zeros(len(partial_GBZ_list), dtype=int)
    for j in range(len(partial_GBZ_list)):
        min_dist, min_id = find_point_in_subGBZ(partial_GBZ_list[j], critical_point)
        dist_list[j] = min_dist
        min_ind_list[j] = min_id
    min_loop = np.argmin(dist_list)
    return min_loop, min_ind_list[min_loop]


def extract_energy_seg(minor_subGBZ: Gm.GBZSegment, start_ind: int, end_ind: int):
    ''' Given a GBZ loop, extract the energy segment between start_ind and end_ind '''
    energy_seg = np.zeros(abs(end_ind - start_ind), dtype=complex)
    if start_ind < end_ind:
        minor_subGBZ_range = range(start_ind, end_ind)
    else:
        minor_subGBZ_range = range(start_ind, end_ind, -1)
    
    for E_ind, point_ind in enumerate(minor_subGBZ_range):
        energy_seg[E_ind] = minor_subGBZ.point_vec[point_ind].coords[0]
    
    return energy_seg


def find_point_in_GBZ_list(partial_GBZ_list: list[Gm.GBZSegment], critical_point: Gm.GBZPointData):
    ''' Given a point, find its GBZ index and coordinate in the GBZ_list '''
    all_dists = np.zeros(len(partial_GBZ_list))
    all_coordinates = np.zeros(len(partial_GBZ_list))
    for j in range(len(partial_GBZ_list)):
        min_dist, min_id = find_point_in_GBZ(partial_GBZ_list[j], critical_point)
        all_dists[j] = min_dist
        all_coordinates[j] = partial_GBZ_list[j].param_list[min_id][0]
    
    min_loop = np.argmin(min_dist)
    return min_loop, all_coordinates[min_loop]

def get_distance_in_partial_GBZ(
        point_begin: Gm.GBZPointData, 
        point_end: Gm.GBZPointData, 
        GBZ_list: list[Gm.GBZSegment]):
    loop_id, coord_begin = find_point_in_GBZ_list(GBZ_list, point_begin)
    loop_id2, coord_end = find_point_in_GBZ_list(GBZ_list, point_end)
    if(loop_id != loop_id2):
        return None

    GBZ_length = GBZ_list[loop_id].param_list[-1][0] - GBZ_list[loop_id].param_list[0][0]
    if(coord_end >= coord_begin):
        return coord_end - coord_begin
    else:
        return coord_end - coord_begin + GBZ_length

# def solve_partial_GBZ(
#         beta_pair: list[(Gm.CP1PointData, float)], 
#         char_poly: pt.CLaurent, 
#         N_points: int = 200, 
#         glue_tol: float = 0.1, 
#         distance_tol: float = 1e-1, 
#         GBZ_checker_tol: float = 1e-6):

#     # 1. Get major beta
#     beta1, s1 = beta_pair[0]
#     beta2, s2 = beta_pair[1]
#     Gm.CP1_to_chart(beta1, [0,0])
#     Gm.CP1_to_chart(beta2, [0,0])
#     beta_major = (beta1.coords[0] + beta2.coords[0])/2

#     distance_checker = CP1DistanceChecker(tol=distance_tol)
#     critical_checker = GBZContinuityChecker(tol=glue_tol)

#     aGBZ_eq = aGBZEquation1DGBZCP1(char_poly)
#     post_processer = GBZConditionChecker(aGBZ_eq, GBZ_checker_tol)
#     root_solver = BatchCP1Root(dtype=Gm.GBZPointData, jac=True)
#     solver_control = BatchSolverController(
#         phcpy_GBZCP1_full_solve, 
#         root_solver,
#         aGBZ_eq,
#         distance_checker,
#         critical_checker,
#         post_processer,
#         match_CP1_points,
#         check_full_solver_default
#     )

#     phi_loop = np.linspace(0, np.pi - 1e-12, N_points).reshape((-1,1))

#     solver_control.set_path(phi_loop)
#     solver_control.evolve_to_dest()

#     phi_loop = solver_control.param_path
#     return solver_control.result_list, phi_loop, solver_control.value_list
 

class PartialGBZWindingFun(ParamFun):
    beta_norm: float
    relative_phase: float
    char_poly: pt.CLaurent
    char_poly_obj: aGBZEquation1DGBZCP1

    def __init__(self, char_poly: pt.CLaurent, 
                 beta_norm: float,
                 relative_phase: float) -> None:
        self.char_poly = char_poly
        self.beta_norm = beta_norm
        self.relative_phase = relative_phase
    
    def set_params(self, params):
        # params: argument of beta
        beta = self.beta_norm * np.exp(1j * params[0])
        f_partial = self.char_poly.partial_eval(
            pt.CScalarVec([beta]), 
            pt.CIndexVec([1]),
            pt.CIndexVec([0,2])
        )
        self.char_poly_obj = aGBZEquation1DGBZCP1(f_partial)
        self.char_poly_obj.set_params([self.relative_phase,])
        super().set_params(params)
    
    def __call__(self, x: Gm.GBZPointData):
        return self.char_poly_obj(x)
    
    def get_callable(self, chart_labels: list[int]):
        return self.char_poly_obj.get_callable(chart_labels)

def solve_partial_GBZ(char_poly: pt.CLaurent, beta_norm: float, N_points: int = 200, 
               glue_tol: float = 0.1, distance_tol: float = 1e-1, GBZ_checker_tol: float = 1e-6):
    distance_checker = CP1DistanceChecker(tol=distance_tol)
    critical_checker = GBZContinuityChecker(tol=glue_tol)

    aGBZ_eq = PartialGBZWindingFun(char_poly, beta_norm, 0)
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

    phi_loop = np.linspace(0, 2 * np.pi, N_points).reshape((-1,1))

    solver_control.set_path(phi_loop)
    solver_control.evolve_to_dest()

    phi_loop = solver_control.param_path
    return solver_control.result_list, phi_loop, solver_control.value_list

# def glue_partial_GBZ_winding(GBZ_segments: list[Gm.GBZSegment], start_phase: float, 
#                              glue_tol: float, zero_tol: float = 1e-12):
#     glued_segments = []
#     head_segments = []
#     other_segments = []
#     for seg in GBZ_segments:
#         if(abs(seg.param_list[0][0] - start_phase) < zero_tol):
#             if(abs(seg.param_list[-1][0] - start_phase - 2*np.pi) < zero_tol):
#                 glued_segments.append(seg)
#             else:
#                 head_segments.append(seg)
#         else:
#             other_segments.append(seg)

#     for seg in head_segments:
#         seg_succeed = True
#         while(abs(seg.param_list[-1][0] - start_phase - 2*np.pi) >= zero_tol):
#             dist_list = np.zeros(len(other_segments))
#             for j in range(len(other_segments)):
#                 dist_list[j] = (Gm.get_distance(seg.point_vec[-1], other_segments[j].point_vec[0])
#                                 + abs(seg.param_list[0][0] - other_segments[j].param_list[0][0]))
#             min_id = np.argmin(dist_list)
#             if dist_list[min_id] > glue_tol:
#                 seg_succeed = False
#                 break
#             new_seg = other_segments.pop(min_id)
#             for j in range(len(new_seg.point_vec)):
#                 seg.append_point(new_seg.point_vec[j], new_seg.param_list[j])
#         if seg_succeed:
#             glued_segments.append(seg)
        
#     # assert(len(other_segments) == 0)
#     return glued_segments


def match_partial_GBZ_winding(GBZ_data):
    '''
    Match the winding loops in the GBZ data.
    '''
    GBZ_chains = []
    for point in (GBZ_data[0]):
        new_chain = Gm.GBZPointVec()
        new_chain.append(point)
        GBZ_chains.append(new_chain)

    for sol_ind in range(1, len(GBZ_data)):
        # 1. Get the tails of current chain
        tails = Gm.GBZPointVec()
        for curr_chain in GBZ_chains:
            tails.append(curr_chain[-1])
        
        # 2. match points
        all_pairs = Gm.match_points(tails, GBZ_data[sol_ind])

        # 3. update chains
        temp_chains = []
        for pair in all_pairs:
            new_chain = GBZ_chains[pair[0]]
            new_chain.append(GBZ_data[sol_ind][pair[1]])
            temp_chains.append(new_chain)
        GBZ_chains = temp_chains
    
    return GBZ_chains


def calculate_hole_winding_loop(char_poly: pt.CLaurent, beta_norm: float, N_points: int = 200, 
               glue_tol: float = 0.1, distance_tol: float = 1e-1, GBZ_checker_tol: float = 1e-6):
    GBZ_data, param_data, _ = solve_partial_GBZ(
        char_poly, beta_norm, N_points, glue_tol, distance_tol, GBZ_checker_tol
    )
    winding_loop_list = match_partial_GBZ_winding(GBZ_data)

    f_partial = char_poly.partial_eval(
        pt.CVarScalarVec([beta_norm]), pt.CIndexVec([1]), pt.CIndexVec([0,2])
    )

    partial_GBZ_data, partial_GBZ_param, _ = solve_GBZ(
        'GBZ', f_partial, N_points, glue_tol, distance_tol, GBZ_checker_tol
    )
    partial_GBZ_segments = get_GBZ_segments(partial_GBZ_data, partial_GBZ_param, 10 * glue_tol, 
            aGBZEquation1DGBZCP1(f_partial))
    partial_GBZ_segments = expand_segments(partial_GBZ_segments)
    partial_GBZ_loops = GBZ_segments_to_loops(partial_GBZ_segments)
    partial_GBZ_loops = batch_glue(partial_GBZ_loops)

    # copy the list of winding loops
    start_point_info = [
        find_point_in_GBZ(partial_GBZ_loops, curr_loop[0])
            for curr_loop in winding_loop_list
    ]   # subGBZ_ind, start_ind

    start_band = [p[0] for p in start_point_info]

    print(start_band)

    assert(set(start_band) == set(range(len(partial_GBZ_loops))))
    raw_winding_loop_list = winding_loop_list
    winding_loop_list = []
    for band_ind in range(len(partial_GBZ_loops)):
        winding_loop_list.append(
            raw_winding_loop_list[start_band.index(band_ind)]
        )

    start_point_info = [
        find_point_in_GBZ(partial_GBZ_loops, curr_loop[0])
            for curr_loop in winding_loop_list
    ]   # subGBZ_ind, start_ind

    start_band = [p[0] for p in start_point_info]

    if len(set(start_band)) != len(start_band):
        raise ValueError("Start point does not match GBZ bands")
    
    end_point_info = [
        find_point_in_GBZ(partial_GBZ_loops, curr_loop[-1])
            for curr_loop in winding_loop_list
    ]
    end_band = [p[0] for p in end_point_info]
    if set(end_band) != set(start_band):
        raise ValueError("End point does not match start point")
    
    # Get loops
    edges = [(start, end) for start, end in zip(start_band, end_band)]
    all_loops = find_loops_from_edges(edges)
    print(all_loops)

    # Get energies for all winding loops
    all_E_loops = []
    for curr_loop in all_loops:
        E_loop = []
        for step_ind in range(len(curr_loop)):
            next_step_ind = (step_ind + 1) % len(curr_loop)
            edge_ind = curr_loop[step_ind]
            next_ind = curr_loop[next_step_ind]
            assert(end_point_info[edge_ind][0] == start_point_info[next_ind][0])
            # energies in the winding loop
            curr_winding_loop = winding_loop_list[edge_ind]
            winding_loop_energy = np.zeros(len(curr_winding_loop), dtype=complex)
            for point_ind in range(len(curr_winding_loop)):
                winding_loop_energy[point_ind] = curr_winding_loop[point_ind].coords[0]
            E_loop.append(winding_loop_energy)
            # E_loop.append(extract_energy_seg(
            #     partial_GBZ_loops[end_point_info[edge_ind][0]],
            #     end_point_info[edge_ind][1],
            #     start_point_info[next_ind][1]
            # ))
        all_E_loops.append(np.concatenate(E_loop))
    return all_E_loops

    # old single-band code
    curr_GBZ_loop = partial_GBZ_loops[0]
    curr_winding_loop = GBZ_segments[0]

    _, start_id = find_point_in_GBZ(curr_GBZ_loop, curr_winding_loop.point_vec[0])
    _, end_id = find_point_in_GBZ(curr_GBZ_loop, curr_winding_loop.point_vec[-1])

    E_loop = np.zeros(len(curr_winding_loop.point_vec) + abs(end_id - start_id), dtype=complex)
    curr_id = 0
    for point in curr_winding_loop.point_vec:
        E_loop[curr_id] = point.coords[0]
        curr_id += 1
    
    if(start_id > end_id):
        partial_GBZ_range = range(end_id, start_id)
    else:
        partial_GBZ_range = range(end_id, start_id, -1)

    for partial_GBZ_id in partial_GBZ_range:
        E_loop[curr_id] = curr_GBZ_loop.point_vec[partial_GBZ_id].coords[0]
        curr_id += 1
    
    return E_loop
 

def calculate_minor_GBZ_torus(char_poly: pt.CLaurent, beta1_list: np.ndarray[1, complex],
                              n_mGBZ: int, glue_tol: float = 0.1, distance_tol: float = 1e-1, 
                              GBZ_checker_tol: float = 1e-6):
    ''' Calculate the whole minor-GBZ torus for a given beta_norm '''
    n_bands = char_poly.num_max_orders[0]

    def calculate_minor_GBZ(char_poly, beta1):
        f_partial = char_poly.partial_eval(
            pt.CScalarVec([beta1]), pt.CIndexVec([1]), pt.CIndexVec([0,2])
        )
        minor_GBZ_data, minor_GBZ_param, _ = solve_GBZ(
            'GBZ', f_partial, n_mGBZ, glue_tol, distance_tol, GBZ_checker_tol
        )
        minor_GBZ_segments = get_GBZ_segments(minor_GBZ_data, minor_GBZ_param, 20 * glue_tol, 
                aGBZEquation1DGBZCP1(f_partial))
        minor_GBZ_segments = expand_segments(minor_GBZ_segments)
        minor_GBZ_loops = GBZ_segments_to_loops(minor_GBZ_segments)
        minor_GBZ_loops = batch_glue(minor_GBZ_loops)

        # remove bad loops
        if len(minor_GBZ_loops) > n_bands:
            minor_GBZ_loops = sorted(minor_GBZ_loops, key=lambda loop: (-len(loop.point_vec)))
            print("Warning: bad loops detected! length=%d" % (len(minor_GBZ_loops[n_bands].point_vec)), file=sys.stderr)
            minor_GBZ_loops = minor_GBZ_loops[:n_bands]
        return minor_GBZ_loops
    
    minor_GBZ_torus = []
    for beta1 in beta1_list:
        print(np.log(beta1).imag / np.pi, "pi")
        new_GBZ = calculate_minor_GBZ(char_poly, beta1)
        # print(len(new_GBZ))
        if minor_GBZ_torus:
            curr_GBZ = minor_GBZ_torus[-1]
            new_GBZ_sorted = []
            for band_ind in range(n_bands):
                curr_point = curr_GBZ[band_ind].point_vec[0]
                loop_ind, _ = find_point_in_GBZ(new_GBZ, curr_point)
                new_GBZ_sorted.append(new_GBZ.pop(loop_ind))
            minor_GBZ_torus.append(new_GBZ_sorted)
        else:
            minor_GBZ_torus.append(new_GBZ)
        
        # for curr_GBZ in minor_GBZ_torus:
        #     print(len(curr_GBZ), end=',')
        # print()
    
    return minor_GBZ_torus


def calculate_minor_GBZ_no_sort(char_poly: pt.CLaurent, beta1_list: np.ndarray[1, complex],
                              n_mGBZ: int, glue_tol: float = 0.1, distance_tol: float = 1e-1, 
                              GBZ_checker_tol: float = 1e-6):
    ''' Calculate the whole minor-GBZ torus for a given beta_norm '''
    n_bands = char_poly.num_max_orders[0]

    def calculate_minor_GBZ(char_poly, beta1):
        f_partial = char_poly.partial_eval(
            pt.CScalarVec([beta1]), pt.CIndexVec([1]), pt.CIndexVec([0,2])
        )
        minor_GBZ_data, minor_GBZ_param, _ = solve_GBZ(
            'GBZ', f_partial, n_mGBZ, glue_tol, distance_tol, GBZ_checker_tol
        )
        minor_GBZ_segments = get_GBZ_segments(minor_GBZ_data, minor_GBZ_param, 20 * glue_tol, 
                aGBZEquation1DGBZCP1(f_partial))
        minor_GBZ_segments = expand_segments(minor_GBZ_segments)
        minor_GBZ_loops = GBZ_segments_to_loops(minor_GBZ_segments)
        minor_GBZ_loops = batch_glue(minor_GBZ_loops)

        return minor_GBZ_loops
    
    minor_GBZ_torus = []
    for beta1 in beta1_list:
        print(np.log(beta1).imag / np.pi, "pi")
        new_GBZ = calculate_minor_GBZ(char_poly, beta1)
        minor_GBZ_torus.append(new_GBZ)
        
        # for curr_GBZ in minor_GBZ_torus:
        #     print(len(curr_GBZ), end=',')
        # print()
    
    return minor_GBZ_torus

