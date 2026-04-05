'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-23
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from math import sin, cos, sqrt, pi
from cmath import exp, log
# import poly_tools as pt
import c_solver_2D as pt
from scipy import optimize
from .strip_winding_number import get_strip_winding, get_strip_winding_brute_force
from .winding import WindingFun


class SGBZSolver:
    char_poly: pt.CLaurent
    winding_fun: WindingFun

    def __init__(self, char_poly: pt.CLaurent):
        self.char_poly = char_poly
        self.winding_fun = WindingFun(char_poly, None, (0, 2 * pi))

    def solve_for_E(
        self, 
        E_ref: complex, 
        mu1_guess: tuple[float, float] = (-1, 1), 
        zero_tol: float = 1e-10,
        N_points: int = 101
    ):
        # 1. Determine mu1_left and mu1_right
        # 1.1 mu1_left
        mu1_left = mu1_guess[0]
        mu1_right = None

        while True:
            left_winding, PMGBZ_points = get_strip_winding(self.winding_fun, E_ref, mu1_left, N_points)
            if left_winding < zero_tol:
                break
            else:
                mu1_right = mu1_left
                mu1_left -= 1

        # Check zero
        if left_winding > - zero_tol:
            return mu1_left, PMGBZ_points

        # 1.2 mu1_right
        if mu1_right is None:
            mu1_right = mu1_guess[1]
            while True:
                right_winding, PMGBZ_points = get_strip_winding(self.winding_fun, E_ref, mu1_right, N_points)
                if right_winding > -zero_tol:
                    break
                else:
                    mu1_right += 1

            if right_winding < zero_tol:
                return mu1_right, PMGBZ_points

        # 2. Find zero of the strip winding number between mu1_left and mu1_right
        def strip_winding_fun(mu1: float):
            w = get_strip_winding(self.winding_fun, E_ref, mu1, N_points)[0]
            if isinstance(w, float):
                return w
            else:
                return np.mean(w)
        
        mu1_0 = optimize.brentq(strip_winding_fun, mu1_left, mu1_right)
        _, PMGBZ_points = get_strip_winding(self.winding_fun, E_ref, mu1_0, N_points)

        return mu1_0, PMGBZ_points

class SGBZChecker:
    char_poly: pt.CLaurent
    winding_fun: WindingFun

    def __init__(self, char_poly: pt.CLaurent):
        self.char_poly = char_poly
        self.winding_fun = WindingFun(char_poly, None, (0, 2 * pi))
    
    def check_for_E_and_mu1(self, E: complex, mu1: float, N_points: int = 101, zero_tol: float = 1e-10):
        try:
            winding, PMGBZ_points = get_strip_winding(self.winding_fun, E, mu1, N_points)
        except:
            winding = get_strip_winding_brute_force(self.winding_fun, E, mu1, N_points)
            PMGBZ_points = [None]

        if isinstance(winding, tuple):
            if winding[0] < 0 and winding[1] > 0:
                return True
            else:
                return False
        else:
            if abs(winding) < zero_tol and len(PMGBZ_points) > 0:
                return True
            else:
                return False


def check_SGBZ(
    coeffs: np.ndarray,
    degs: np.ndarray,
    E_ref: complex,
    perc: float,
    debug_mode: bool = False
):
    print("%.2f" % (perc * 100) + r"%")
    coeffs = pt.CScalarVec(coeffs)
    degs = pt.CLaurentIndexVec(degs.flatten())
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(coeffs, degs)
    solver = SGBZSolver(char_poly)
    try:
        mu1, PMGBZ_points = solver.solve_for_E(E_ref, N_points=301)
        success = True
    except Exception as e:
        if debug_mode:
            raise e
        else:
            print("Error: %s" % str(e))
        success = False
        return {"success": success, "error": str(e)}

    if PMGBZ_points:
        return {"success": success, "is_PMGBZ": True, "mu1": mu1, "PMGBZ_points": PMGBZ_points}
    else:
        return {"success": success, "is_PMGBZ": False, "mu1": mu1}