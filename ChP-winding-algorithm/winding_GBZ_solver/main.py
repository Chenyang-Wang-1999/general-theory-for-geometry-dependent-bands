'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-01-01
Copyright © Department of Physics, Tsinghua University. All rights reserved

Provide high-level interface for calculating strip-GBZ
'''

from .non_Bloch_solver import *
from .intersections import *
from .winding_number import *
import poly_tools as pt
import multiprocessing


class SolverErrCode:
    ''' Error code for solver '''
    SUCCESS: int = 0
    MGBZ_LOOP_BROKEN: int = 1
    PAIRING_FAILED: int = 2


class StripGBZMainSolver:
    ''' Main solver for strip-GBZ '''
    ### input parameters ###
    char_poly: pt.CLaurent

    ### derived parameters ###
    dChP: list[pt.CLaurent]

    ### solutions ###
    non_Bloch_holes: list[float]
    non_Bloch_ranges: list[list[float]]
    non_Bloch_sols: list[np.ndarray[2, complex]]

    total_winding: float
    positive_charges: list[np.ndarray[1, complex]]
    negative_charges: list[np.ndarray[1, complex]]

    ### flags ###
    temp_E_ref: complex
    temp_radius_Cr: float

    def __init__(
        self, 
        char_poly: pt.CLaurent  # Characteristic polynomial of the non-Bloch Hamiltonian
    ):
        self.char_poly = char_poly
        self.non_Bloch_sols = []
        self.dChP = [char_poly.derivative(i) for i in range(3)]

    def solve_non_Bloch_solutions(
        self, 
        E_ref, 
        n_points=200, 
        glue_tol=0.1, 
        distance_tol=0.1, 
        GBZ_checker_tol=1e-6
    ):
        self.temp_E_ref = E_ref
        non_Bloch_loops_raw, has_broken_loops = solve_non_Bloch_loops(
            self.char_poly, E_ref, n_points, glue_tol, distance_tol, GBZ_checker_tol
        )
        non_Bloch_loops = convert_to_complex_numbers(non_Bloch_loops_raw)
        self.non_Bloch_holes, self.non_Bloch_ranges = get_holes_from_loops(non_Bloch_loops)
        self.non_Bloch_sols = [np.array(loop) for loop in non_Bloch_loops]
        if has_broken_loops:
            return SolverErrCode.MGBZ_LOOP_BROKEN
        else:
            return SolverErrCode.SUCCESS

    def get_total_winding(
        self,
        radius_Cr: float,
        n_theta1: int,
        max_incre: float = pi/12
    ):
        if not self.non_Bloch_sols:
            raise Exception('Non-Bloch solutions not solved yet')

        # 1. Find intersections
        critical_segs = get_critical_segments(
            self.non_Bloch_sols,
            radius_Cr
        )
        intersection_point_list = []
        for seg in critical_segs:
            # for i in range(seg[0].shape[0]):
            intersection_point_list.append(
                interp_to_radius_r(
                    seg[0],
                    seg[1],
                    radius_Cr
                )
            )

        # 3. Pairing
        try:
            point_pair = pair_points(intersection_point_list)
        except:
            print("Warning: point pairing failed!", file=sys.stderr)
            print(intersection_point_list, file=sys.stderr)
            return SolverErrCode.PAIRING_FAILED

        positive_points = []
        negative_points = []
        for curr_pair in point_pair:
            if calculate_topological_charge(
                self.dChP, self.temp_E_ref, curr_pair
            ) > 0:
                positive_points.append(curr_pair[0])
                negative_points.append(curr_pair[1])
            else:
                positive_points.append(curr_pair[1])
                negative_points.append(curr_pair[0])

        # 4. Calculate total winding number
        self.total_winding = calculate_total_winding_number(
            self.char_poly, self.temp_E_ref, radius_Cr,
            n_theta1, positive_points, negative_points, max_incre
        )
        self.positive_charges = positive_points
        self.negative_charges = negative_points
        self.temp_radius_Cr = radius_Cr

        return SolverErrCode.SUCCESS

    def get_total_winding_at_hole(
        self,
        hole_ind: int,
        n_theta1: int
    ):
        ''' Simplified version of get_total_winding at a hole '''
        if not self.non_Bloch_sols:
            raise Exception('Non-Bloch solutions not solved yet')

        return calculate_total_winding_number(
            self.char_poly, self.temp_E_ref,
            self.non_Bloch_holes[hole_ind], n_theta1, [], []
        )

    def check_GBZ_spectrum(
        self, E_ref: complex,
        n_non_Bloch: int = 50,
        glue_tol: float = 0.1,
        distance_tol: float = 0.1,
        GBZ_checker_tol: float = 1e-6,
        n_theta1: int = 50
    ):
        ''' Check whether E_ref is in the GBZ spectrum'''
        self.solve_non_Bloch_solutions(
            E_ref, n_non_Bloch, glue_tol, distance_tol, GBZ_checker_tol
        )
        for hole_ind in range(len(self.non_Bloch_holes)):
            curr_winding = self.get_total_winding_at_hole(hole_ind, n_theta1)
            if abs(curr_winding / 2 * np.pi) < 0.01:
                return False

        return True


def GBZ_sweep_fun(args):
    coeffs, degs, E_ref, n_non_Bloch, glue_tol, distance_tol, GBZ_checker_tol, n_theta1 = args
    # print(E_ref)
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    try:
        solver = StripGBZMainSolver(char_poly)
        result = solver.check_GBZ_spectrum(
            E_ref, n_non_Bloch, glue_tol, distance_tol, GBZ_checker_tol, n_theta1
        )
    except:
        result = np.nan

    return result


def batch_sweep_strip_GBZ(
    coeffs: list[complex],
    degs: list[int],
    E_ref_list: list[complex],
    n_processes: int,
    n_non_Bloch: int = 50,
    glue_tol: float = 0.1,
    distance_tol: float = 0.1,
    GBZ_checker_tol: float = 1e-6,
    n_theta1: int = 50
):
    if n_processes == 1:
        return [
            GBZ_sweep_fun(
                (coeffs, degs, E_ref, n_non_Bloch, glue_tol, distance_tol, GBZ_checker_tol, n_theta1)
            )
            for E_ref in E_ref_list
        ]
    else:
        pool = multiprocessing.Pool(n_processes)
        data_pack = [
            (coeffs, degs, E_ref, n_non_Bloch, glue_tol, distance_tol, GBZ_checker_tol, n_theta1)
            for E_ref in E_ref_list
        ]
        return pool.map(GBZ_sweep_fun, data_pack)
