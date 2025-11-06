'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-04-10
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' Interfaces for user to solve partial GBZ'''

import poly_tools as pt
import GBZ_manifold as Gm
import numpy as np
from .aGBZ import *
from .GBZ_solver import *
from .winding_number import *
from .partial_GBZ import *


####### Solvers ########
class GBZSolverErrorCode:
    SOLVER_SUCCESS: int = 0
    BETA_LOOP_FAIL: int = 1
    CRITICAL_BETA_NOT_CONVERGED: int = 2
    BETA_NOT_IN_PAIR: int = 3
    E_CRITICAL: int = 4
    PARTIAL_GBZ_ERROR: int = 5


class GBZMainSolver:
    ''' Main solver for GBZ '''

    ### polynomials ###
    f: pt.CLaurent
    f_partial: pt.CLaurent
    char_poly_info: CharPolyInfo
    char_poly_CP1: aGBZEquation1DCP1
    char_poly_GBZ: aGBZEquation1DGBZCP1

    ### configurations ###
    critical_beta_tol: float # tol for critical beta searching
    zero_tol: float # tol for float = 0 in numerical calculation
    glue_tol: float # tol for glue segments
    segment_intersect_tol: float # tol for glue different segments
    GBZ_checker_tol: float # tol for GBZ checker
    beta_perturb_dist: float # distance for beta to perturb

    ### results ###
    beta_GBZ: list[list[Gm.GBZSegment]]

    def __init__(self,
        f: pt.CLaurent,
        critical_beta_tol: float = 1e-4, 
        zero_tol: float = 1e-14,
        glue_tol: float = 0.02,
        segment_intersect_tol: float = 1e-6,
        GBZ_checker_tol: float = 1e-6,
        beta_perturb_dist: float = 1e-3
    ) -> None:
        # polynomials
        self.f = f
        self.char_poly_info = CharPolyInfo(f)

        # configuration
        self.critical_beta_tol = critical_beta_tol
        self.zero_tol = zero_tol
        self.glue_tol = glue_tol
        self.segment_intersect_tol = segment_intersect_tol
        self.GBZ_checker_tol = GBZ_checker_tol
        self.beta_perturb_dist = beta_perturb_dist

        # Initialize results
        self.beta_GBZ = None
    
    def get_beta_GBZ(self, E: complex, N_points: int = 200, no_glue: bool = False) -> GBZSolverErrorCode:
        ''' Get beta GBE for given energy E '''
        self.beta_GBZ = []
        self.f_partial = self.f.partial_eval(
            pt.CVarScalarVec([E]), pt.CIndexVec([0]), pt.CIndexVec([1,2])
        )
        self.char_poly_CP1 = aGBZEquation1DCP1(self.f_partial)

        GBZ_loops, phi_loop, aGBZ_loops = solve_GBZ("CP1", self.f_partial, N_points, glue_tol=self.glue_tol,
                                           GBZ_checker_tol=self.GBZ_checker_tol)
        GBZ_loops, phi_loop, aGBZ_loops = expand_solution(
            GBZ_loops, aGBZ_loops, phi_loop)
        all_segments = get_GBZ_segments(GBZ_loops, phi_loop, 10 * self.glue_tol, 
                    self.char_poly_CP1)
        all_loops = GBZ_segments_to_loops(all_segments)
        self.beta_GBZ = all_loops        

        # get holes
        self.beta_GBZ_glued = batch_glue(all_loops)
        GBZ_holes, GBZ_intervals = get_holes_from_loops(self.beta_GBZ_glued)
        self.GBZ_holes = GBZ_holes
        self.GBZ_intervals = GBZ_intervals

        return GBZSolverErrorCode.SOLVER_SUCCESS
    
    def robust_get_beta_GBZ(self, E: complex, N_retry=30):
        for _ in range(N_retry):
            err_code = self.get_beta_GBZ(E)
            if(err_code == GBZSolverErrorCode.SOLVER_SUCCESS):
                return GBZSolverErrorCode.SOLVER_SUCCESS
            raise_warning("Warning: solver failed, recalculating ...")

        return GBZSolverErrorCode.BETA_LOOP_FAIL
    
    def get_critical_beta(self, E: complex, norm_cos_theta: float):
        all_loops = self.beta_GBZ
        # 1. find all critical segments
        all_segs = []
        for loop in all_loops:
            for seg in loop:
                all_segs += find_critical_segments(norm_cos_theta, seg)
 
        # 3. Label topological charge
        all_topo_charges = label_critical_segments(
            self.char_poly_info, E, all_segs
        )
        
        # 3. Iteratively calculate critical beta
        positive_points = []
        negative_points = []
        for seg_id in range(len(all_segs)):
            p = iteratively_solve_critical_points(
                self.char_poly_CP1,
                all_segs[seg_id],
                norm_cos_theta,
                self.critical_beta_tol,
                self.zero_tol
            )
            if(p is None):
                return GBZSolverErrorCode.CRITICAL_BETA_NOT_CONVERGED
            if(all_topo_charges[seg_id] > 0):
                positive_points.append(p)
            else:
                negative_points.append(p)

        negative_points = pair_points(positive_points, negative_points) 
        if(negative_points is None):
            self.critical_beta_pair = (positive_points, negative_points)
            return GBZSolverErrorCode.BETA_NOT_IN_PAIR
        self.critical_beta_pair = (positive_points, negative_points)
        return GBZSolverErrorCode.SOLVER_SUCCESS

    def get_partial_GBZ(self, major_beta: complex, using_global_coordinates: bool = True):
        f_partial = self.f.partial_eval(
            pt.CScalarVec([major_beta]),
            pt.CIndexVec([1]),
            pt.CIndexVec([0,2])
        )

        curr_GBZ, curr_param, curr_aGBZ = solve_GBZ('GBZ', f_partial, glue_tol=self.glue_tol)
        curr_GBZ, curr_param, curr_aGBZ = expand_solution(curr_GBZ, curr_aGBZ, curr_param)
        all_segments = get_GBZ_segments(curr_GBZ, curr_param, 10 * self.glue_tol, 
                    aGBZEquation1DGBZCP1(f_partial))
        all_loops = GBZ_segments_to_loops(all_segments)
        if(using_global_coordinates):
            all_loops = batch_get_global(all_loops, aGBZEquation1DGBZCP1(f_partial))
            for loop in all_loops:
                normalize_segment_coords(loop)
        else:
            all_loops = batch_glue(all_loops)
        self.partial_GBZ = all_loops

        degs = pt.CIndexVec([])
        f_partial.num.batch_get_data(pt.CScalarVec([]), degs)
        degs = np.reshape(degs, (-1,2))
        E_deg = np.max(degs[:,0])
        if(len(all_loops) != E_deg):
            return GBZSolverErrorCode.PARTIAL_GBZ_ERROR

        return GBZSolverErrorCode.SOLVER_SUCCESS



    def get_partial_GBZ_distance(self, E: complex):
        positive_points, negative_points = self.critical_beta_pair

        dist_list = np.zeros(len(positive_points))
        for j in range(len(dist_list)):
            curr_positive = positive_points[j][0]
            curr_negative = negative_points[j][0]
            Gm.CP1_to_chart(curr_positive, [0,0])
            Gm.CP1_to_chart(curr_negative, [0,0])
            major_beta = curr_positive.coords[0]

            # 1. For each major beta, calculate the partial GBZ
            err_code = self.get_partial_GBZ(major_beta)
            if(err_code != 0):
                return err_code
                
            all_loops = self.partial_GBZ


            # 2. Calculate the distance from positive point to negative point
            dist_list[j] = get_distance_in_partial_GBZ(
                Gm.GBZPointData([E, curr_positive.coords[1]], [0]), 
                Gm.GBZPointData([E, curr_negative.coords[1]], [0]),
                all_loops
            )
        
        self.partial_GBZ_distance = dist_list

        return GBZSolverErrorCode.SOLVER_SUCCESS


def solve_GBZ_1D(char_poly, **kwargs):
    minor_GBZ_data, minor_GBZ_param, _ = solve_GBZ(
        'GBZ', char_poly, **kwargs
    )
    minor_GBZ_segments = get_GBZ_segments(minor_GBZ_data, minor_GBZ_param, 2 * kwargs['glue_tol'], 
            aGBZEquation1DGBZCP1(char_poly))
    minor_GBZ_segments = expand_segments(minor_GBZ_segments)
    minor_GBZ_loops = GBZ_segments_to_loops(minor_GBZ_segments)
    minor_GBZ_loops = batch_glue(minor_GBZ_loops)

    return minor_GBZ_loops
