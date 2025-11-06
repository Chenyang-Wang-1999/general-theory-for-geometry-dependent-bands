'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-03-18
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' python interface for GBZ manifold '''

from .lib_GBZ_manifold import *
from typing import Literal

class GBZPointData(cpp_GBZPointData):
    chart_labels: list[int]
    coords: list[complex]
    def __init__(self, coords=None, chart_labels=None):
        super().__init__()
        if(coords is not None):
            self.coords = coords
        if(chart_labels is not None):
            self.chart_labels = chart_labels
    
    def __reduce__(self):
        return (self.__class__, (self.coords, self.chart_labels))

class CP1PointData(cpp_CP1PointData):
    chart_labels: list[int]
    coords: list[complex]
    def __init__(self, coords=None, chart_labels=None):
        super().__init__()
        if(coords is not None):
            self.coords = coords
        if(chart_labels is not None):
            self.chart_labels = chart_labels
    def __reduce__(self):
        return (self.__class__, (self.coords, self.chart_labels))

GBZPointVec = cpp_GBZPointVec
Point3DVec = cpp_GBZPoint3DVec
CP1PointVec = cpp_CP1PointVec

def check_min_distance(point_list, tol: float) -> float:
    if(isinstance(point_list, GBZPointVec)):
        return cpp_check_min_distance_GBZ(point_list, tol)
    elif(isinstance(point_list, CP1PointVec)):
        return cpp_check_min_distance_CP1(point_list, tol)
    else:
        raise ValueError("unsupported type")

def batch_to_standard(point_list):
    if(isinstance(point_list, GBZPointVec)):
        cpp_batch_to_standard_GBZ(point_list) 
    elif(isinstance(point_list, CP1PointVec)):
        cpp_batch_to_standard_CP1(point_list) 
    else:
        raise ValueError("unsupported type")

def match_point_list(
        curr_points,
        new_points,
        results) -> bool:
    if(isinstance(curr_points, GBZPointVec)):
        return cpp_match_point_list_GBZ(
            curr_points,
            new_points,
            results
        )
    elif(isinstance(curr_points, CP1PointVec)):
        return cpp_match_point_list_CP1(
            curr_points,
            new_points,
            results
        )
    else:
        raise ValueError("unsupported type")

def get_distance(point1, point2):
    if(isinstance(point1, cpp_GBZPointData)):
        return cpp_get_distance_GBZ(point1, point2)
    elif(isinstance(point1, cpp_CP1PointData)):
        return cpp_get_distance_CP1(point1, point2)
    else:
        raise ValueError("unsupported type", type(point1))

def get_point_3D_coord(GBZ_point: GBZPointData):
    return cpp_get_point_3D_coord(GBZ_point)

def batch_get_point_3D_coord(GBZ_point: GBZPointData,
                             real_point: Point3DVec):
    return cpp_batch_get_point_3D_coord(GBZ_point, real_point)

def calculate_min_distance(point, point_vec):
    if(isinstance(point, cpp_GBZPointData)):
        return cpp_calculate_min_distance_GBZ(point, point_vec)
    elif(isinstance(point, cpp_CP1PointData)):
        return cpp_calculate_min_distance_CP1(point, point_vec)
    else:
        raise ValueError("unsupported type:", type(point))

def get_CP1Product_interp(
    point1: CP1PointData,
    point2: CP1PointData,
    norm_cos_theta: float,
    s_ini: float = 0.5,
    tol: float = 1e-12,
    max_iter: int = 10000       
):
    return cpp_get_CP1Product_interp(
        point1,
        point2,
        norm_cos_theta,
        s_ini,
        tol,
        max_iter
    )

def linear_interp_CP1(point1: CP1PointData, point2: CP1PointData, s: float):
    result = CP1PointData([0,0], [0,0])
    cpp_linear_interp_CP1(point1, point2, result, s)
    return result

def linear_interp_GBZ(point1: GBZPointData, point2: GBZPointData, s: float):
    result = GBZPointData([0,0], [0])
    cpp_linear_interp_GBZ(point1, point2, result, s)
    return result

def get_z_coord(point: GBZPointData, j:int):
    if(isinstance(point, cpp_GBZPointData)):
        return cpp_get_z_coord_GBZ(point, j)
    elif(isinstance(point, cpp_CP1PointData)):
        return cpp_get_z_coord_CP1(point, j)
    else:
        raise ValueError("unsupported type: %s"%(str(type(point))))

def get_CP1_distance(point1: CP1PointData, point2: CP1PointData, which_beta: int):
    if(isinstance(point1, cpp_GBZPointData)):
        return cpp_get_CP1_distance_GBZ(point1, point2, which_beta)
    elif(isinstance(point1, cpp_CP1PointData)):
        return cpp_get_CP1_distance_CP1(point1, point2, which_beta)
    else:
        raise ValueError("unsupported type: %s"%(str(type(point1))))
    
def CP1_to_chart(point: CP1PointData, chart_labels: list[int]):
    cpp_to_chart_CP1(point, chart_labels)

def GBZ_to_chart(point: GBZPointData, chart_labels: list[int]):
    cpp_to_chart_GBZ(point, chart_labels)

class GBZSegment:
    point_vec: list
    param_list: list
    data_type: str
    def __init__(self, data_type: str = 'GBZ'):
        self.data_type = data_type
        if(data_type == 'GBZ'):
            self.point_vec = GBZPointVec()
        elif(data_type == 'CP1'):
            self.point_vec = CP1PointVec()
        else:
            raise ValueError("unsupported type:", data_type)
        self.param_list = []
    
    def append_point(self, point, param):
        self.point_vec.append(point)
        self.param_list.append(param)
    
    @classmethod
    def create_new_one(cls, point_vec, param_list, data_type):
        new_one = cls(data_type)
        new_one.point_vec = point_vec
        new_one.param_list = param_list
        return new_one

    def __reduce__(self):
        return (self.create_new_one, (self.point_vec, self.param_list, self.data_type))

def CP1_diff_z(z_coord: float, beta_sq_diff: float, chart_label: int):
    if(chart_label == 0):
        return ((1 - z_coord) ** 2 * beta_sq_diff) / 2
    else:
        return - ((1 + z_coord) ** 2 * beta_sq_diff) / 2

def match_points(point_list1: GBZPointVec, 
                 point_list2: GBZPointVec) -> list[tuple[int, int]]:
    '''
        Match the points in point_list2 to point_list1,
        return the matched points.
        input:
            point_list1: the old point list
            point_list2: the new point list 
        output:
            matched_ind_pairs: pairs of the index of the matched points in point_list1 and point_list2
    '''
    
    # Determine the smaller and larger sets
    if len(point_list1) <= len(point_list2):
        small_set = point_list1
        large_set = point_list2
        # Flag to indicate swap
        swapped = False
    else:
        small_set = point_list2
        large_set = point_list1
        swapped = True
   
    # Find the closest large_set point for each small_set point
    pairs = []
    for i, point in enumerate(small_set):
        min_dist, min_ind = calculate_min_distance(point, large_set)
        pairs.append( (i, min_ind, min_dist) )
    
    # Sort pairs by distance ascending
    pairs_sorted = sorted(pairs, key=lambda x: x[2])
    
    # Initialize set of matched large_set indices
    remaining_indices = list(range(len(large_set)))
    
    # Initialize list of matched pairs
    matched_pairs = []
    
    # Match pairs starting from the smallest distance
    while pairs_sorted:
        pair = pairs_sorted.pop(0)
        small_ind, large_ind, dist = pair

        # match current pair
        matched_pairs.append( (small_ind, large_ind) )
        remaining_indices.remove(large_ind)

        # remove conflict pairs
        new_pairs = []
        is_adjusted = False
        for curr_pair in pairs_sorted:
            if curr_pair[1] == large_ind:
                is_adjusted = True
                new_point_list = point_list1.__class__()
                for j in remaining_indices:
                    new_point_list.append(large_set[j])
                min_dist, min_ind = calculate_min_distance(small_set[curr_pair[0]], new_point_list)
                new_pairs.append(
                    (curr_pair[0], remaining_indices[min_ind], min_dist)
                )

            else:
                new_pairs.append(curr_pair)
        
        if is_adjusted:
            pairs_sorted = sorted(new_pairs, key=lambda x: x[2])
        else:
            pairs_sorted = new_pairs
    
    # If swapped, adjust the indices
    if swapped:
        # Swap indices and adjust to original point_list1 and point_list2 indices
        adjusted_pairs = [ (large_ind, small_ind) for small_ind, large_ind in matched_pairs ]
        return adjusted_pairs
    else:
        return matched_pairs


def remove_duplicated_points(
    point_vec1: Literal['GBZPointVec', 'CP1PointVec'],
    point_vec2: Literal['GBZPointVec', 'CP1PointVec'],
    glue_tol: float
):
    if isinstance(point_vec1, cpp_GBZPointVec) and isinstance(point_vec2, cpp_GBZPointVec):
        return cpp_remove_duplicated_points_GBZ(point_vec1, point_vec2, glue_tol)
    elif isinstance(point_vec1, cpp_CP1PointVec) and isinstance(point_vec2, cpp_CP1PointVec):
        return cpp_remove_duplicated_points_CP1(point_vec1, point_vec2, glue_tol)
