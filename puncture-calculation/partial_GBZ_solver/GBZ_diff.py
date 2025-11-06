'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-04-07
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' The differential structure of GBZ '''
import poly_tools as pt
import numpy as np
import GBZ_manifold as Gm
from .aGBZ import aGBZEquation1DCP1, aGBZEquation1DGBZCP1

from scipy import linalg as la

def get_GBZ_diff(char_poly: pt.CLaurent, 
                 curr_point: np.ndarray, 
                 s_local: float, 
                 zero_tol: float = 1e-6):
    ''' Get the derivative of GBZ against GBZ parameter s '''
    curr_point_ctype = pt.CVarScalarVec(curr_point)
    f_E = char_poly.derivative(0)
    f_beta = char_poly.derivative(1)
    if(abs(s_local) < zero_tol):
        f_Eb = f_E.derivative(1)
        f_bb = f_beta.derivative(1)
        f_bb_val = f_bb.eval(curr_point_ctype)
        A = np.array([
            [f_E.eval(curr_point_ctype), 0], 
            [f_Eb.eval(curr_point_ctype), f_bb_val]
        ])
        b = np.array([
            0, -1j * curr_point[1] * f_bb_val
        ])

    else:
        another_point_ctype = pt.CVarScalarVec(curr_point_ctype) 
        exp_iphi = np.exp(2j * s_local)
        another_point_ctype[1] *= exp_iphi
        f_b_another = f_beta.eval(another_point_ctype) 
        A = np.array([
            [f_E.eval(curr_point_ctype), f_beta.eval(curr_point_ctype)],
            [f_E.eval(another_point_ctype), exp_iphi * f_b_another]
        ])
        b = np.array([
            0, -1j * another_point_ctype[1] * f_b_another
        ])

    if(abs(la.det(A)) < zero_tol):
        return None
    else:
        return -2 * la.solve(A,b)

def get_GBZ_diff_CP1(char_poly: aGBZEquation1DCP1, point: Gm.CP1PointData,
                     s_local: float, zero_tol: float = 1e-6):
    ''' Get the derivative of point.coords against the parameter s '''
    curr_char_poly = char_poly.get_callable(point.chart_labels).char_poly
    if(point.chart_labels[-1] == 0):
        point_der = get_GBZ_diff(curr_char_poly, point.coords, s_local, zero_tol)
        return point_der
    else:
        point_der = get_GBZ_diff(curr_char_poly, point.coords, -s_local, zero_tol)
        if(point_der is None):
            return None
        else:
            return -point_der