'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-05-30
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from .aGBZ import *
from .GBZ_solver import *
import poly_tools as pt
from multiprocessing import Pool

def brute_force_solve_GBZ_for_phi(args):
    # solve for given phi
    coeffs, degs, phi, GBZ_checker_tol, perc = args
    print("#" * int(perc * 100) + "-" * (100 - int(perc*100)) + " %.2f%%"%(perc * 100) )
    char_poly = pt.CLaurent(2)
    char_poly.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))
    aGBZ_eq = aGBZEquation1D(char_poly)
    aGBZ_eq.set_params([phi])
    point_list = phcpy_full_solve(aGBZ_eq)
    GBZ_points = []
    for j in range(point_list.shape[0]):
        flag = check_GBZ_condition(char_poly, point_list[j,1], point_list[j,0], GBZ_checker_tol)
        if(flag):
            GBZ_points.append(point_list[j,:])
    
    # print progress bar
    return np.asarray(GBZ_points), point_list
    

def batch_brute_force_solve_GBZ(coeffs: np.ndarray, degs: np.ndarray, N_phi: int = 50, GBZ_checker_tol: float = 1e-4,
                    N_process: int = 1):
    # solve GBZ by brute force algorithm
    phi_list = np.linspace(0, np.pi, N_phi)
    input_param_list = [
        (coeffs, degs, phi, GBZ_checker_tol, sol_id/N_phi) for sol_id, phi in enumerate(phi_list)
    ]

    if(N_process == 1):
        all_results = [brute_force_solve_GBZ_for_phi(args) for args in input_param_list]
    else:
        pool = Pool(N_process)
        all_results = pool.map(brute_force_solve_GBZ_for_phi, input_param_list)

    GBZ_points = [res[0] for res in all_results]
    aGBZ_points = [res[1] for res in all_results]
    return np.vstack(GBZ_points), np.vstack(aGBZ_points)