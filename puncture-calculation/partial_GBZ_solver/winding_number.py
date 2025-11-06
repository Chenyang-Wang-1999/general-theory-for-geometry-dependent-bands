'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-02-17
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''
import poly_tools as pt
import numpy as np
from scipy import linalg as la
import GBZ_manifold as Gm
from .GBZ_solver import *
from .aGBZ import *

''' Calculations for the winding number of a given critical point '''

class CharPolyInfo:
    ''' Wrapper for characteristic polynomial with its derivatives '''
    char_poly: pt.CLaurent
    df: list[pt.CLaurent]

    def __init__(self, char_poly: pt.CLaurent):
        self.char_poly = char_poly
        self.df = [char_poly.derivative(var_id) for var_id in range(3)]
    
    def eval_df(self, E_beta):
        if(not isinstance(E_beta, pt.CVarScalarVec)):
            E_beta = pt.CVarScalarVec(E_beta)
        return [
            self.df[j].eval(E_beta) for j in range(3)
        ]
    
    def eval(self, E_beta):
        if(not isinstance(E_beta, pt.CVarScalarVec)):
            E_beta = pt.CVarScalarVec(E_beta)
        return self.char_poly.eval(E_beta)

    def get_E_and_beta_derivative(self,
                                  E,
                                  beta_major,
                                  beta_minor_p,
                                  beta_minor_m):
        ''' get the derivatives of E (against the argument of phi and theta) and major beta, same sign with 'beta_minor_p' '''

        # 1. Calculate exp(i phi)
        exp_iphi = beta_minor_p/beta_minor_m

        # 2. Calculate df 
        df_p = self.eval_df([
            E, beta_major, beta_minor_p
        ]) # values at p1
        df_m = self.eval_df([
            E, beta_major, beta_minor_m
        ]) # values at p2

        # # debug
        # print(df_p, np.abs(df_p))
        # print(df_m, np.abs(df_m))

        # 3. Calculate dE (against beta1 and theta)
        # AT = np.array([
        #     [df_m[0], df_p[0]],
        #     [df_m[2], df_p[2] * exp_iphi]
        # ])
        # if(abs(la.det(AT)) < 1e-12):
        #     return 0, np.inf
        # X = la.solve(AT, np.array([[1], [0]]))
        # X = X.reshape((1,-1))
        # B = np.array([
        #     [df_m[1], 0],
        #     [df_p[1], 1j * beta_minor_p * df_p[2]]
        # ])
        # dE_beta_phi = - (X @ B)
        # dE_beta_phi = dE_beta_phi.flatten()
        dE_beta_phi = np.array([
            -df_p[2] * exp_iphi * df_m[1] + df_m[2] * df_p[1],
            1j * beta_minor_p * df_p[2] * df_m[2]
        ]) / (df_m[0] * df_p[2] * exp_iphi - df_m[2] * df_p[0])

        dE = np.array([dE_beta_phi[1], dE_beta_phi[0] * 1j * beta_major])
        
        # 4. Calculate dbeta
        AT = np.array([
            [df_m[1], df_p[1]],
            [df_m[2], exp_iphi * df_p[2]]
        ])
        X = la.solve(AT, np.array([[1], [0]]))
        X = X.flatten()
        B = np.array([
            0,
            -1j * beta_minor_p * df_p[2]
        ])
        dbeta = (X.dot(B))

        return dE, dbeta
    
    def get_det_and_dbeta(self,
                            E,
                            beta_major,
                            beta_minor_p,
                            beta_minor_m):
        ''' Get the sign of det(dE) and dbeta, same sign with 'beta_minor_p' '''
        dE, dbeta = self.get_E_and_beta_derivative(E,
                                                   beta_major, 
                                                   beta_minor_p, 
                                                   beta_minor_m)
        E_realmat = np.zeros((2,2))
        E_realmat[0,:] = dE.real
        E_realmat[1,:] = dE.imag
        return la.det(E_realmat), dbeta

def find_critical_segments(norm_cos_theta: float, GBZ_loop: Gm.GBZSegment) -> list[Gm.GBZSegment]:
    ''' Given the norm of the beta, find the critical points '''
    assert(GBZ_loop.data_type == 'CP1')
    critical_segments = []
    
    for j in range(1, len(GBZ_loop.point_vec)):
        if(j == 0):
            point1 = GBZ_loop.point_vec[-1]
            param1 = GBZ_loop.param_list[-1]
            point2 = GBZ_loop.point_vec[0]
            param2 = GBZ_loop.param_list[0]
        else:
            point1 = GBZ_loop.point_vec[j - 1]
            param1 = GBZ_loop.param_list[j - 1]
            point2 = GBZ_loop.point_vec[j]
            param2 = GBZ_loop.param_list[j]
        
        # get major beta
        norm1 = Gm.get_z_coord(point1, 0)
        norm2 = Gm.get_z_coord(point2, 0)

        # check critical condition
        if(norm1 >= norm_cos_theta):
            if(norm2 < norm_cos_theta):
                curr_pair = Gm.GBZSegment('CP1')
                curr_pair.append_point(point2, param2)
                curr_pair.append_point(point1, param1)
                critical_segments.append(curr_pair)
        else:
            if(norm2 >= norm_cos_theta):
                curr_pair = Gm.GBZSegment('CP1')
                curr_pair.append_point(point1, param1)
                curr_pair.append_point(point2, param2)
                critical_segments.append(curr_pair)
        
    return critical_segments 

class CP1PointWithCharge(Gm.CP1PointData):
    topo_charge: int
    def __init__(self, point: Gm.CP1PointData, topo_charge: int):
        self.topo_charge = topo_charge
        super().__init__(point.coords, point.chart_labels)
        
def label_critical_segments(
        poly_info: CharPolyInfo, 
        E: complex, 
        critical_segments: list[Gm.GBZSegment]) -> list[int]:
    ''' Label the topological charge of the points'''
    topo_charge_list = []
    for j in range(len(critical_segments)):
        # get current points and params
        curr_seg = critical_segments[j]

        curr_labeled_pair = []
        detE_pair = [0.0, 0.0]
        for point_id, point in enumerate(curr_seg.point_vec):
            # get major and minor beta
            new_point = Gm.CP1PointData(point.coords, point.chart_labels)
            curr_params = curr_seg.param_list[point_id]
            Gm.CP1_to_chart(new_point, [0,0])
            detE, dbeta = poly_info.get_det_and_dbeta(
                E, new_point.coords[0],
                new_point.coords[1],
                new_point.coords[1] * np.exp(-2j * curr_params[0])
            )
            detE_pair[point_id] = detE
            if(detE > 0):
                topo_charge = 1
            else:
                topo_charge = -1
            curr_labeled_pair.append(
                topo_charge
            )
        if(curr_labeled_pair[0] != curr_labeled_pair[1]):
            if(abs(detE_pair[0]) > abs(detE_pair[1])):
                curr_labeled_pair[1] = curr_labeled_pair[0]
            else:
                curr_labeled_pair[0] = curr_labeled_pair[1]
        topo_charge_list.append(curr_labeled_pair[0])

    return topo_charge_list

# def pair_list(
#         seg_list1: list[list[CP1PointWithCharge]],
#         seg_list2: list[list[CP1PointWithCharge]]):
#     ''' Pair the lists of labeled segments
#         return: sorted seg_list2 according to seg_list1 
#     '''
#     assert(len(seg_list1) == len(seg_list2))
#     new_seg_list2 = []
#     for curr_seg in seg_list1:
#         distance_list = []
#         for curr_seg2 in seg_list2:
#             curr_distance = Gm.get_CP1_distance(curr_seg[0], curr_seg2[0], 0)
#             distance_list.append(curr_distance)
#         min_id = np.argmin(distance_list)
#         curr_new_seg = seg_list2.pop(min_id)
#         new_seg_list2.append(curr_new_seg)
#     return new_seg_list2

# def pair_critical_segments(
#         labeled_segments: list[list[CP1PointWithCharge]]
# ):
#     ''' Pair the labeled segments
#         1. Classify the segments into ++, --, +- and -+
#         2. Pair ++ to --, +- to -+
#     '''

#     # 1. Collect all the segs
#     collected_pairs = {
#         (1,1): [],
#         (1,-1): [],
#         (-1,1): [],
#         (-1,-1): []
#     }
#     for curr_seg in labeled_segments:
#         collected_pairs[(
#             curr_seg[0].topo_charge, curr_seg[1].topo_charge
#         )].append(curr_seg)
    
#     # 2. Check length
#     # assert(len(collected_pairs[(1,1)]) == len(collected_pairs[(-1,-1)]))
#     # assert(len(collected_pairs[(1,-1)]) == len(collected_pairs[(-1,1)]))
#     if(len(collected_pairs[(1,1)]) != len(collected_pairs[(-1,-1)])):
#         # for curr_index in [(1,1), (-1,-1), (1,-1), (-1,1)]:
#         #     print(curr_index, len(collected_pairs[curr_index]))
#         return None

#     # 3. pair segments
#     new_list = pair_list(
#         collected_pairs[(1,1)], collected_pairs[(-1,-1)]
#     )
#     collected_pairs[(-1,-1)] = new_list
#     new_list = pair_list(
#         collected_pairs[(1,-1)], collected_pairs[(-1,1)]
#     )
#     collected_pairs[(-1,1)] = new_list

#     return collected_pairs


# def interp_critical_segments(
#         paired_seg1: list[Gm.CP1PointData], 
#         paired_seg2: list[Gm.CP1PointData],
#         norm_cos_theta: float):

#     s = Gm.get_CP1Product_interp(
#         paired_seg1[0], paired_seg1[1], norm_cos_theta
#     )
#     if(not np.isfinite(s)):
#         return None, None
#     paired_point1 = Gm.linear_interp_CP1(paired_seg1[0], paired_seg1[1], s)

#     s = Gm.get_CP1Product_interp(
#         paired_seg2[0], paired_seg2[1], norm_cos_theta
#     )
#     if(not np.isfinite(s)):
#         return None, None
 
#     paired_point2 = Gm.linear_interp_CP1(paired_seg2[0], paired_seg2[1], s)
        
#     return paired_point1, paired_point2 


# def calculate_aGBZ_from_given_pair(
#     point1: Gm.CP1PointData, point2: Gm.CP1PointData,
#     poly_fun: aGBZEquation1DCP1
# ):
#     ''' Solve the GBZ by the initial guess of point1 and point2 '''
#     # 1. get param
#     if(point1.chart_labels[-1] == 0):
#         beta_minor1 = point1.coords[-1]
#     else:
#         beta_minor1 = 1/point1.coords[-1]
#     if(point2.chart_labels[-1] == 0):
#         beta_minor2 = point2.coords[-1]
#     else:
#         beta_minor2 = 1/point2.coords[-1]
#     param = np.log(beta_minor2/beta_minor1).imag
#     print(param)

#     # 2. solve equation
#     poly_fun.set_params(np.array([param,]))
#     solver = BatchCP1Root(dtype=Gm.CP1PointData, jac=True)
#     new_point_list = solver(poly_fun, [point1])

#     # 2.5 Handle errors
#     # print("current param:", param, "current |beta2|:", abs(beta_minor1), abs(beta_minor2))
#     if(new_point_list is None):
#         # start full solver
#         all_sols = phcpy_CP1_full_solve(poly_fun)
#         min_dist, min_id = Gm.calculate_min_distance(point1, all_sols)
#         print("Warning: continuation solver fails. min_dist = %f"%(min_dist), file=sys.stderr)
#         new_point1 = all_sols[min_id]
#     else:
#         new_point1 = new_point_list[0]


#     # 3. retrieve the results
#     new_point2_coords = new_point1.coords
#     if(new_point1.chart_labels[-1] == 0):
#         new_point2_coords[-1] *= np.exp(1j * param)
#     else:
#         new_point2_coords[-1] /= np.exp(1j * param)
#     new_point2 = Gm.CP1PointData(new_point2_coords, new_point1.chart_labels)
#     return new_point1, new_point2

def get_relative_phase(point1: Gm.GBZPointData, point2: Gm.GBZPointData):
    if(point1.chart_labels[-1] == 0):
        beta2_1 = point1.coords[1]
    else:
        beta2_1 = 1/point1.coords[1]
    
    if(point2.chart_labels[-1] == 0):
        beta2_2 = point2.coords[1]
    else:
        beta2_2 = 1/point2.coords[1]
    
    phase_diff = np.log(beta2_2/beta2_1).imag
    return phase_diff

# using derivative
# def iteratively_solve_critical_points(
#         poly_fun: aGBZEquation1DCP1,
#         initial_seg: Gm.GBZSegment,
#         norm_cos_theta: float,
#         tol: float,
#         zero_tol: float = 1e-14,
#         max_iter: int = 100
# ):
#     ''' Iteratively solve the critical points '''
#     if(isinstance(poly_fun, aGBZ.aGBZEquation1DCP1)):
#         root_solver = aGBZ.BatchCP1Root(dtype=Gm.CP1PointData, jac=True)
#     else:
#         root_solver = aGBZ.BatchCP1Root(dtype=Gm.GBZPointData, jac=True)

#     dp1 = get_GBZ_diff_CP1(poly_fun, initial_seg.point_vec[0], 
#                            initial_seg.param_list[0][0], 
#                            zero_tol)
#     dp2 = get_GBZ_diff_CP1(poly_fun, initial_seg.point_vec[1], 
#                            initial_seg.param_list[1][0], 
#                            zero_tol)
    

#     if(dp1 is None):
#         if(dp2 is None):
#             z1 = Gm.get_z_coord(initial_seg.point_vec[0], 0)
#             z2 = Gm.get_z_coord(initial_seg.point_vec[1], 0)
#             if(abs(z1 - norm_cos_theta) < abs(z2 - norm_cos_theta)):
#                 return initial_seg.point_vec[0], initial_seg.param_list[0][0]
#             else:
#                 return initial_seg.point_vec[1], initial_seg.param_list[1][0]
#         curr_id = 1
#     elif(dp2 is None):
#         curr_id = 0
#     else:
#         t = Gm.get_CP1Product_interp(initial_seg.point_vec[0], 
#                                     initial_seg.point_vec[1],
#                                     norm_cos_theta)
#         if(t < 0.5):
#             curr_id = 0
#         else:
#             curr_id = 1
    
#     curr_point = initial_seg.point_vec[curr_id]
#     curr_s_param = initial_seg.param_list[curr_id][0]
#     curr_z = Gm.get_z_coord(curr_point, 0)
    
#     for _ in range(max_iter):
#         dp = get_GBZ_diff_CP1(poly_fun, curr_point, curr_s_param, zero_tol)
#         dbeta_sq = 2 * (curr_point.coords[0].conjugate() * dp[0]).real
#         dz = Gm.CP1_diff_z(curr_z, dbeta_sq, curr_point.chart_labels[0])

#         # 1. get ds
#         ds = (norm_cos_theta - curr_z) / dz

#         # 2. update points
#         curr_coords = curr_point.coords
#         for j in range(2):
#             curr_coords[j] += dp[j] * ds
#         curr_point.coords = curr_coords
#         curr_s_param += ds

#         # 3. solve
#         poly_fun.set_params([-2 * curr_s_param,])
#         curr_point = root_solver.solve_one(poly_fun, curr_point)
#         curr_z = Gm.get_z_coord(curr_point, 0)

#         if(abs(curr_z - norm_cos_theta) < tol):
#             return (curr_point, curr_s_param)
        
    
#     print("Warning(iteratively_solve_critical_points): iteration does not converge. (N > %d)"%(max_iter))
#     return None

def iteratively_solve_critical_points(
        poly_fun: aGBZEquation1DCP1,
        initial_seg: Gm.GBZSegment,
        norm_cos_theta: float,
        tol: float,
        zero_tol: float = 1e-14,
        max_iter: int = 10000
):
    ''' Iteratively solve the critical points '''
    curr_pair = Gm.GBZSegment(initial_seg.data_type)
    for j in range(2):
        curr_pair.append_point(initial_seg.point_vec[j], initial_seg.param_list[j])

    if(isinstance(poly_fun, aGBZ.aGBZEquation1DCP1)):
        root_solver = aGBZ.BatchCP1Root(dtype=Gm.CP1PointData, jac=True)
    else:
        root_solver = aGBZ.BatchCP1Root(dtype=Gm.GBZPointData, jac=True)

    for _ in range(max_iter):
        t = Gm.get_CP1Product_interp(
            curr_pair.point_vec[0], curr_pair.point_vec[1], norm_cos_theta
        )
        
        if((t < 0) or (t > 1)):
            z1 = Gm.get_z_coord(curr_pair.point_vec[0], 0)
            z2 = Gm.get_z_coord(curr_pair.point_vec[1], 0)
            t = (norm_cos_theta - z1) / (z2 - z1)
            raise_warning("Warning: estimating t by z coord.")
        
        if(t < 0.5):
            new_point = curr_pair.point_vec[0]
        else:
            new_point = curr_pair.point_vec[1]

        new_s_param = curr_pair.param_list[0][0] * (1 - t) + curr_pair.param_list[1][0] * t

        poly_fun.set_params([-2 * new_s_param,])
        new_point = root_solver.solve_one(poly_fun, new_point)
        z_coord = Gm.get_z_coord(new_point, 0)


        if(abs(z_coord - norm_cos_theta) < tol):
            return new_point, new_s_param
        
        if(z_coord > norm_cos_theta):
            curr_pair.point_vec[1] = new_point
            curr_pair.param_list[1][0] = new_s_param
        else:
            curr_pair.point_vec[0] = new_point
            curr_pair.param_list[0][0] = new_s_param
    
    raise_warning("Warning(iteratively_solve_critical_points): iteration does not converge. (N > %d), residual: %s"%(max_iter, z_coord - norm_cos_theta))
    if(abs(z_coord - norm_cos_theta) > abs(Gm.get_z_coord(initial_seg.point_vec[0],0) - norm_cos_theta)):
        return None
    else:
        return new_point, new_s_param

def batch_calculate_critical_points(
        poly_fun: aGBZEquation1DCP1,
        initial_seg_list: list[Gm.GBZSegment],
        norm_cos_theta: float,
        tol: float,
        zero_tol: float = 1e-14,
        max_iter: int = 10000
):
    return [iteratively_solve_critical_points(
        poly_fun, 
        seg,
        norm_cos_theta,
        tol,
        zero_tol,
        max_iter
    ) for seg in initial_seg_list]

def calculate_topological_charge(
    point_list: list[tuple[Gm.CP1PointData, float]], 
    char_poly_info: CharPolyInfo,
    E: complex
):
    positive_points = []
    negative_points = []
    for point_pair in point_list:
        point, param = point_pair
        Gm.CP1_to_chart(point, [0,0])
        detE, dbeta = char_poly_info.get_det_and_dbeta(
            E, point.coords[0], point.coords[1], point.coords[1] * np.exp(-2j * param)
        )
        if(detE > 0):
            positive_points.append((point, param))
        else:
            negative_points.append((point, param))
        
    return positive_points, negative_points

def pair_points(positive_points, negative_points):
    ''' Pair negative points with positive points '''
    if(len(positive_points) != len(negative_points)):
        raise_warning("Warning: beta not in pair. +:%d, -:%d"%(len(positive_points), len(negative_points)))
        return None
    new_negative_points = []
    for point_pair in positive_points:
        all_dist = np.zeros(len(negative_points))
        for j, point_pair2 in enumerate(negative_points):
            all_dist[j] = Gm.get_CP1_distance(point_pair[0], point_pair2[0], 0)
        min_id = np.argmin(all_dist)
        new_negative_points.append(negative_points.pop(min_id))
    return new_negative_points




# def calculate_all_critical_points(
#         poly_info: CharPolyInfo,
#         poly_fun: aGBZEquation1DCP1,
#         all_paired_segs: dict[list[Gm.GBZPointData]],
#         E: complex,
#         norm_cos_theta: float,
#         tol: float
# ):
#     ''' Calculate all segments '''
#     def get_topo_charge(p1: Gm.CP1PointData, p2: Gm.CP1PointData):
#         if(p1.chart_labels[0] == 0):
#             major_beta = p1.coords[0]
#         else:
#             major_beta = 1/p1.coords[0]
#         if(p1.chart_labels[1] == 0):
#             minor_beta1 = p1.coords[1]
#         else:
#             minor_beta1 = 1/p1.coords[1]
#         if(p2.chart_labels[1] == 0):
#             minor_beta2 = p2.coords[1]
#         else:
#             minor_beta2 = 1/p2.coords[1]
#         detE, dbeta = poly_info.get_det_and_dbeta(E, major_beta, minor_beta1, minor_beta2)
#         if(detE > 0):
#             return 1
#         else:
#             return -1

#     all_pairs = []
#     for j in range(len(all_paired_segs[(1,1)])):
#         p1, p2 = iteratively_solve_critical_points(
#                 poly_fun, all_paired_segs[(1,1)][j], all_paired_segs[(-1,-1)][j],
#                 norm_cos_theta, tol
#             )
#         if(p1 is None):
#             return None
#         topo_charge = get_topo_charge(p1, p2)
#         all_pairs.append(
#             (CP1PointWithCharge(p1, topo_charge),
#              CP1PointWithCharge(p2, -topo_charge))
#         )

#     for j in range(len(all_paired_segs[(1,-1)])):
#         p1, p2 = iteratively_solve_critical_points(
#                 poly_fun, all_paired_segs[(1,-1)][j], all_paired_segs[(-1,1)][j],
#                 norm_cos_theta, tol
#             )
#         if(p1 is None):
#             return None
#         topo_charge = get_topo_charge(p1, p2)
#         all_pairs.append(
#             (CP1PointWithCharge(p1, topo_charge),
#              CP1PointWithCharge(p2, -topo_charge))
#         )
#     return all_pairs
    

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


def get_holes_from_loops(GBZ_loops: list[Gm.GBZSegment]):
    ''' Get the position of holes for winding number calculation '''
    beta_norm_intervals = []
    for loop in GBZ_loops:
        curr_norm_list = [Gm.get_z_coord(point, 0) for point in loop.point_vec]
        beta_norm_intervals.append(
            [min(curr_norm_list), max(curr_norm_list)]
        )
    
    disconnected_intervals = get_disconnect_intervals(beta_norm_intervals)
    all_holes = [(-1 + disconnected_intervals[0][0])/2]
    for j in range(len(disconnected_intervals) - 1):
        all_holes.append(
            (disconnected_intervals[j][1] + disconnected_intervals[j+1][0])/2
        )
    all_holes.append((1 + disconnected_intervals[-1][1])/2)
    return all_holes, disconnected_intervals


if __name__ == '__main__':
    pass