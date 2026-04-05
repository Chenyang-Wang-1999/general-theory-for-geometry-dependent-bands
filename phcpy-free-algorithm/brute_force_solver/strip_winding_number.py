'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-11-18
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' Calculations of the strip winding number '''

import numpy as np
# import poly_tools as pt
import c_solver_2D as pt
from .root_solver import brute_force_roots, NormReversionEvent, poly_to_np_coefficients, match_new_to_old, ContinuumZeroEndEvent, calculate_point_roots
from .winding import WindingFun, get_winding_number
from math import sin, cos, sqrt, pi
from cmath import exp, log
from scipy import optimize
from scipy import interpolate
from scipy import integrate


class PMGBZSearchingFun:
    def __init__(
        self, 
        char_poly: pt.CLaurent, 
        E_ref: complex, 
        mu1: float, 
        ref_sols: np.ndarray, 
        M: int,
        with_derivative: bool = False,
        dchar_poly: list[pt.CLaurent] = None
    ):
        self.char_poly = char_poly
        self.E_ref = E_ref
        self.mu1 = mu1
        self.ref_sols = ref_sols
        self.M = M
        self.with_derivative = with_derivative
        if with_derivative:
            if dchar_poly is None:
                dchar_poly = [char_poly.derivative(var_ind) for var_ind in range(char_poly.dim)]
            self.dchar_poly = dchar_poly
    
    def solve_for_theta1(self, theta1: float):
        beta1 = np.exp(self.mu1 + 1j * theta1)
        poly_1d = self.char_poly.partial_eval(
            pt.CScalarVec([self.E_ref, beta1]),
            pt.CIndexVec([0, 1]),
            pt.CIndexVec([2])
        )
        coeffs = pt.CScalarVec([])
        degs = pt.CIndexVec([])
        poly_1d.num.batch_get_data(coeffs, degs)
        coeffs = poly_to_np_coefficients(coeffs, degs)
        sols = list(np.roots(coeffs))
        matched_sols = match_new_to_old(self.ref_sols, sols)
        if self.with_derivative:
            dsols = np.zeros_like(matched_sols, dtype=complex)
            for sol_ind, sol_beta2 in enumerate(matched_sols):
                param_ctype = pt.CScalarVec([self.E_ref, beta1, sol_beta2])
                dsols[sol_ind] = -1j * beta1 * self.dchar_poly[1].eval(param_ctype) / (self.dchar_poly[2].eval(param_ctype))
            return matched_sols, dsols
        else:
            return matched_sols
    
    def __call__(self, theta1: float):
        if self.with_derivative:
            matched_sols, dsols = self.solve_for_theta1(theta1)
            max_lowest_ind = np.argmax(np.abs(matched_sols[:self.M]))
            min_highest_ind = np.argmin(np.abs(matched_sols[self.M:]))
            max_lowest_abs = abs(matched_sols[max_lowest_ind]) 
            min_highest_abs = abs(matched_sols[self.M + min_highest_ind])
            dnorm_sq_high = 2 * (
                matched_sols[self.M + min_highest_ind].conjugate() * dsols[self.M + min_highest_ind]
            ).real
            dnorm_sq_low = 2 * (
                matched_sols[max_lowest_ind].conjugate() * dsols[max_lowest_ind]
            ).real
            return (
                max_lowest_abs ** 2 - min_highest_abs ** 2,
                dnorm_sq_low - dnorm_sq_high
            )
        else:
            matched_sols = self.solve_for_theta1(theta1)
            max_lowest_abs = max(np.abs(matched_sols[:self.M]))
            min_highest_abs = min(np.abs(matched_sols[self.M:]))
            return max_lowest_abs - min_highest_abs


def get_PMGBZ_beta(new_sols, M: int):
    '''
    1. Sort the sols
    2. Get the average of M and M + 1
    3. Get the inverted indices
    4. Pair the points
    '''
    new_args = np.argsort(np.abs(new_sols))
    positive_inds = new_args[M:][new_args[M:] < M]
    negative_inds = new_args[:M][new_args[:M] > M - 1] - M
    assert(len(positive_inds) == len(negative_inds))

    return positive_inds, negative_inds


def check_PMGBZ_point(
    winding_fun: WindingFun,
    E_ref: complex,
    beta1: complex,
    beta2_roots: np.ndarray,  # assuming beta2_roots are sorted by abs(beta2_roots)
    M: int,
    zero_tol: float
):
    '''
        Check the type of PMGBZ.
    '''
    # 1. Check whether the (M-1)-th and M-th roots are close to each other
    if abs(np.max(np.abs(beta2_roots[:M])) - np.min(np.abs(beta2_roots[M:]))) >= zero_tol * abs(beta2_roots[M - 1]):
        return {"PMGBZ_flag": False}

    # 2. Check partial f / partial beta2
    df_dbeta2 = winding_fun.dchar_poly[2].eval(pt.CScalarVec([E_ref, beta1, beta2_roots[M - 1]]))
    if abs(df_dbeta2) < zero_tol * abs(beta2_roots[M - 1]):
        return {"PMGBZ_flag": True, "double_root_flag": True}
    
    # 3. Check partial mu2 / partial theta1
    _, dmu2_dtheta1_a = winding_fun.eval_dmu2((E_ref, beta1, beta2_roots[M - 1]))
    _, dmu2_dtheta1_b = winding_fun.eval_dmu2((E_ref, beta1, beta2_roots[M]))
    return {"PMGBZ_flag": True, "double_root_flag": False, "dmu2_dtheta1": dmu2_dtheta1_b - dmu2_dtheta1_a}


def get_minor_degrees(winding_fun: WindingFun):
    coeffs = pt.CScalarVec([])
    degs = pt.CIndexVec([])
    winding_fun.char_poly.num.batch_get_data(coeffs, degs)
    M_plus_N = max(degs[2::3])
    M = winding_fun.char_poly.denom_orders[2]
    N = M_plus_N - M
    return M, N

def get_roots_and_PMGBZ(
    winding_fun: WindingFun,  # f(E, beta1, beta2)
    E_ref: complex,  # reference energy
    mu1: float,  # mu1 = log(|beta1|)
    N_points: int = 301,  # number of points around the loop |beta1| = exp(mu1)
    zero_tol: float = 1e-10,  # tolerance for the zero of the characteristic polynomial
    theta1_offset_step: float = 0.1
):
    M, N = get_minor_degrees(winding_fun)

    # Data containers
    PMGBZ_points = []
    sols_list = []
    continuum_flag = False

    # events
    PMGBZ_event = NormReversionEvent(M, N, zero_tol=zero_tol)
    continuum_event = ContinuumZeroEndEvent(M, N, zero_tol=zero_tol)

    # Determine initial value of theta1
    theta1_offset = 0
    while True:
        beta1 = exp(mu1 + 1j * theta1_offset)
        initial_sol = calculate_point_roots(
            winding_fun.char_poly,
            pt.CIndexVec((0, 1)),
            (E_ref, beta1),
            pt.CIndexVec([2]),
            M, N
        )
        initial_sol = sorted(initial_sol, key=lambda x: abs(x))
        check_res = check_PMGBZ_point(
            winding_fun,
            E_ref,
            beta1,
            initial_sol,
            M, zero_tol
        )
        if check_res["PMGBZ_flag"]:
            if check_res["double_root_flag"]:
                theta1_offset += theta1_offset_step
            else:
                if abs(check_res["dmu2_dtheta1"]) < zero_tol:
                    curr_event = [continuum_event]
                    continuum_flag = True
                    break
                else:
                    theta1_offset += theta1_offset_step
        else:
            curr_event = [PMGBZ_event]
            break

    # 1. Get the winding loop
    theta1_list = np.linspace(0, 2*pi, N_points) + theta1_offset

    while len(theta1_list):
        beta1 = np.exp(mu1 + 1j * theta1_list)
        E = E_ref * np.ones_like(beta1, dtype=complex)
        param_list = np.column_stack((E, beta1))

        # 2. solve the points
        event_call, sols = brute_force_roots(
            winding_fun.char_poly,
            (0, 1),
            param_list,
            2,
            M,
            N,
            initial_sol=initial_sol,
            events=curr_event
        )
        curr_theta1 = theta1_list[:sols.shape[0]]

        if event_call is not None:
            sols_list.append((curr_theta1[:-1], sols[:-1, :]))
            if isinstance(curr_event[event_call], NormReversionEvent):
                check_res = check_PMGBZ_point(
                    winding_fun,
                    E_ref,
                    exp(mu1 + 1j * curr_theta1[-1]),
                    sols[-1, :],
                    M, zero_tol
                )

                if check_res["PMGBZ_flag"]:
                    if check_res["double_root_flag"]:
                        PMGBZ_point_continuum = True
                    else:
                        if abs(check_res["dmu2_dtheta1"]) < zero_tol:
                            PMGBZ_point_continuum = True
                        else:
                            PMGBZ_point_continuum = False

                    if PMGBZ_point_continuum:
                        PMGBZ_points.append(
                            {
                                "theta1": curr_theta1[-1],
                                "is_start_point": True,
                                "is_continuum": True
                            }
                        )
                        continuum_flag = True
                        curr_event = [continuum_event]
                    else:
                        PMGBZ_points.append(
                            {
                                "theta1": curr_theta1[-1],
                                "beta2_pairs": [(sols[-1, M - 1], sols[-1, M])],
                                "is_continuum": False
                            }
                        )
                        curr_event = [PMGBZ_event]

                    initial_sol = None
                    sols_list.append((
                        (curr_theta1[-1],),
                        sols[-1, :]
                    ))
                    theta1_list = theta1_list[sols.shape[0]:]

                else:
                    # solve PMGBZ point
                    PMGBZ_searching_fun = PMGBZSearchingFun(
                        winding_fun.char_poly,
                        E_ref,
                        mu1,
                        sols[-2, :],
                        M,
                        True,
                        winding_fun.dchar_poly
                    )

                    root_res = optimize.root_scalar(
                        PMGBZ_searching_fun,
                        x0=curr_theta1[-1],
                        fprime=True
                    )
                    PMGBZ_theta1 = root_res.root
                    if PMGBZ_theta1 >= curr_theta1[-1] or PMGBZ_theta1 <= curr_theta1[-2]:
                        PMGBZ_searching_fun.with_derivative = False
                        PMGBZ_theta1 = optimize.brentq(
                            PMGBZ_searching_fun,
                            curr_theta1[-2],
                            curr_theta1[-1]
                        )
                    if PMGBZ_searching_fun.with_derivative:
                        PMGBZ_sols, _ = PMGBZ_searching_fun.solve_for_theta1(PMGBZ_theta1)
                    else:
                        PMGBZ_sols = PMGBZ_searching_fun.solve_for_theta1(PMGBZ_theta1)
                    positive_inds, negative_inds = get_PMGBZ_beta(sols[-1, :], M)
                    beta_points = [
                        (PMGBZ_sols[positive_inds[i]], PMGBZ_sols[M + negative_inds[i]])
                        for i in range(len(positive_inds))
                    ]
                    PMGBZ_points.append(
                        {
                            "theta1": PMGBZ_theta1,
                            "beta2_pairs": beta_points,
                            "is_continuum": False
                        }
                    )

                    sols_list.append((
                        (PMGBZ_theta1,),
                        PMGBZ_sols
                    ))
                    initial_sol = sorted(sols[-1, :], key=lambda x: abs(x))
                    theta1_list = theta1_list[sols.shape[0] - 1:]


            elif isinstance(curr_event[event_call], ContinuumZeroEndEvent):
                PMGBZ_points.append(
                    {
                        "theta1": curr_theta1[-1],
                        "is_start_point": False,
                        "is_continuum": True
                    }
                )
                initial_sol = sorted(sols[-1, :], key=lambda x: abs(x))
                theta1_list = theta1_list[sols.shape[0] - 1:]
                curr_event = [PMGBZ_event]

            else:
                raise ValueError(f"Unknown event type: {event_call}")

        else:
            sols_list.append((curr_theta1, sols))
            theta1_list = theta1_list[sols.shape[0]:]
            initial_sol = sorted(sols[-1, :], key=lambda x: abs(x))

    
    theta1_arr = np.hstack([v[0] for v in sols_list])
    sols_arr = np.vstack([v[1] for v in sols_list])
    return PMGBZ_points, theta1_arr, sols_arr, {"M": M, "N": N, "continuum_flag": continuum_flag}


def get_loop_winding(
    winding_fun: WindingFun,
    E_ref: complex,
    mu1: float,
    mu2_fun: callable,
    theta2: float,
    N_seg: int = 5
) -> float:
    def param_fun(theta1: float):
        mu2_val = mu2_fun(theta1, extrapolate="periodic")
        mu2_prime = mu2_fun(theta1, extrapolate="periodic", nu=1)
        param = np.array([
            E_ref,
            exp(mu1 + 1j * theta1),
            exp(mu2_val + 1j * theta2)
        ])
        dparam = np.array([
            0,
            1j * exp(mu1 + 1j * theta1),
            exp(mu2_val + 1j * theta2) * mu2_prime
        ])
        return param, dparam
    
    winding_fun.loop_fun = param_fun
    winding_fun.loop_range = (0, 2 * pi)
    
    return get_winding_number(winding_fun, N_seg=N_seg)


def get_strip_winding(
    winding_fun: WindingFun,
    E_ref: complex,
    mu1: float,
    N_points: int = 301,
    with_gap_info: bool = False,
    continuum_perturb: float = 1e-2,
    zero_tol: float = 1e-10,
) -> float:
    PMGBZ_points, theta1_arr, sols_arr, info = get_roots_and_PMGBZ(winding_fun, E_ref, mu1, N_points, zero_tol)
    # sort sols_arr by norm
    for row_ind in range(sols_arr.shape[0]):
        sols_arr[row_ind, :] = sols_arr[row_ind, np.argsort(np.abs(sols_arr[row_ind, :]))]

    M = info["M"]
    N = info["N"]

    # print(sols_arr)
    if info["continuum_flag"]:
        # get W_left
        _, _, _, info = get_roots_and_PMGBZ(winding_fun, E_ref, mu1 - continuum_perturb, N_points, zero_tol)
        if info["continuum_flag"]:
            W_left = np.nan
        else:
            W_left = get_strip_winding(
                winding_fun,
                E_ref,
                mu1 - continuum_perturb,
                N_points,
                with_gap_info,
                continuum_perturb,
                zero_tol
            )[0]
        
        # get W_right
        _, _, _, info = get_roots_and_PMGBZ(winding_fun, E_ref, mu1 + continuum_perturb, N_points, zero_tol)
        if info["continuum_flag"]:
            W_right = np.nan
        else:
            W_right = get_strip_winding(
                winding_fun,
                E_ref,
                mu1 + continuum_perturb,
                N_points,
                with_gap_info,
                continuum_perturb,
                zero_tol
            )[0]
        
        if with_gap_info:
            return (W_left, W_right), PMGBZ_points, None
        else:
            return (W_left, W_right), PMGBZ_points
        
        # print("Warning: continuum_flag is True.")


    mu2_max_arr = np.log(np.abs(sols_arr[:, M]))
    mu2_min_arr = np.log(np.abs(sols_arr[:, M - 1]))
    mu2_max_arr[-1] = mu2_max_arr[0]
    mu2_min_arr[-1] = mu2_min_arr[0]
    mu2_arr = (mu2_max_arr + mu2_min_arr) / 2
    mu2_fun = interpolate.CubicSpline(theta1_arr, mu2_arr, bc_type="periodic")

    theta2_min_arr = np.angle(sols_arr[:, M - 1])
    theta2_max_arr = np.angle(sols_arr[:, M])

    def calculate_min_mu2_diff(theta2_target: float):
        # shift theta2_target to [-pi, pi]
        theta2_target = (theta2_target + pi) % (2 * pi) - pi

        # find crossing
        theta2_max_ind = np.arange(len(theta2_max_arr) - 1)[
            (theta2_max_arr[1:] - theta2_target) * (theta2_max_arr[:-1] - theta2_target) <= 0
        ]
        theta2_min_ind = np.arange(len(theta2_min_arr) - 1)[
            (theta2_min_arr[1:] - theta2_target) * (theta2_min_arr[:-1] - theta2_target) <= 0
        ]

        # mu2 list
        mu2_list = []
        for ind in np.concatenate([theta2_max_ind, theta2_min_ind]):
            curr_theta1 = theta1_arr[ind]
            curr_mu2 = mu2_max_arr[ind]
            mu2_list.append(abs(curr_mu2 - mu2_fun(curr_theta1)))
        if len(mu2_list) == 0:
            return np.inf
        else:
            return min(mu2_list)


    # 1. Select best theta2: 将所有PMGBZ_points的theta2部分在数轴上排开，取宽度最大的区间的中位数
    if not PMGBZ_points:
        N_samples = 4
        theta2_samples = np.linspace(-pi + 2 * pi / (N_samples), pi, N_samples)
        mu2_diff_samples = [calculate_min_mu2_diff(theta2) for theta2 in theta2_samples]
        theta2_median = theta2_samples[np.argmax(mu2_diff_samples)]
        W_strip = np.round(get_loop_winding(winding_fun, E_ref, mu1, mu2_fun, theta2_median))
    else:
        PMGBZ_beta2 = []
        for v in PMGBZ_points:
            PMGBZ_beta2 += v["beta2_pairs"]
        PMGBZ_beta2 = np.array(PMGBZ_beta2)
        theta2_arr = np.angle(PMGBZ_beta2.flatten())
        theta2_arr = (theta2_arr + 2 * pi) % (2 * pi)
        theta2_arr.sort()
        theta2_samples = (theta2_arr[1:] + theta2_arr[:-1]) / 2
        theta2_samples = np.hstack((theta2_samples, [(theta2_samples[0] + 2 * pi + theta2_samples[-1]) / 2]))
        mu2_diff_samples = [calculate_min_mu2_diff(theta2) for theta2 in theta2_samples]
        theta2_median = theta2_samples[np.argmax(mu2_diff_samples)]

        w0 = np.round(get_loop_winding(winding_fun, E_ref, mu1, mu2_fun, theta2_median))
        theta2_mat = np.angle(PMGBZ_beta2 / exp(1j * theta2_median)) % (2 * pi) 
        W_strip = w0 + (np.sum(theta2_mat[:, 1]) - np.sum(theta2_mat[:, 0])) / (2 * pi)
    
    if with_gap_info:
        if PMGBZ_points:
            gap_info = None
        else:
            # 1. Get minimum of mu2_max_arr and maximum of mu2_min_arr
            mu2_max_ind = np.argmin(mu2_max_arr)
            mu2_min_ind = np.argmax(mu2_min_arr)

            # 2. Get interpolating functions
            def beta2_solving_fun(beta2, theta1):
                var_ctype = pt.CScalarVec([E_ref, exp(mu1 + 1j * theta1), beta2])
                return winding_fun.char_poly.eval(var_ctype), winding_fun.dchar_poly[2].eval(var_ctype)
            
            def derivative_fun(theta1, beta2_initial):
                res = optimize.root_scalar(
                    beta2_solving_fun,
                    args=(theta1,),
                    x0=beta2_initial,
                    fprime=True
                )
                beta2_val = res.root
                return winding_fun.eval_dmu2([E_ref, exp(mu1 + 1j * theta1), beta2_val])[1]

            theta1_extended = np.hstack(([theta1_arr[-2] - 2 * pi], theta1_arr, [theta1_arr[1] + 2 * pi]))
            theta1_max = optimize.brentq(derivative_fun, theta1_extended[mu2_max_ind], theta1_extended[mu2_max_ind + 2],
                    args=(sols_arr[mu2_max_ind, M],))
            theta1_min = optimize.brentq(derivative_fun, theta1_extended[mu2_min_ind], theta1_extended[mu2_min_ind + 2],
                    args=(sols_arr[mu2_min_ind, M - 1],))

            beta2_max = optimize.root_scalar(
                beta2_solving_fun,
                args=(theta1_max,),
                x0=sols_arr[mu2_max_ind, M],
                fprime=True
            ).root

            beta2_min = optimize.root_scalar(
                beta2_solving_fun,
                args=(theta1_min,),
                x0=sols_arr[mu2_min_ind, M - 1],
                fprime=True
            ).root

            gap_info = {
                "point_max": (E_ref, exp(mu1 + 1j * theta1_max), beta2_max),
                "point_min": (E_ref, exp(mu1 + 1j * theta1_min), beta2_min),
                "mu2_diff": mu2_max_arr[mu2_max_ind] - mu2_min_arr[mu2_min_ind],
                "dmu2_diff": winding_fun.eval_dmu2([E_ref, exp(mu1 + 1j * theta1_max), beta2_max])[0] - winding_fun.eval_dmu2([E_ref, exp(mu1 + 1j * theta1_min), beta2_min])[0]
            }

        return W_strip, PMGBZ_points, gap_info

    else:
        return W_strip, PMGBZ_points


def get_strip_winding_brute_force(
    winding_fun: WindingFun,
    E_ref: complex,
    mu1: float,
    N_points: int = 101
):
    M, N = get_minor_degrees(winding_fun)
    theta1_list = np.linspace(0, 2 * pi, N_points)
    beta1 = np.exp(mu1 + 1j * theta1_list)
    E = E_ref * np.ones_like(beta1, dtype=complex)
    param_list = np.column_stack((E, beta1))
    _, sols = brute_force_roots(
        winding_fun.char_poly,
        (0, 1),
        param_list,
        2,
        M,
        N
    )

    all_abs = np.sort(np.abs(sols), axis=0)
    mu2_max_arr = np.log(all_abs[:, M])
    mu2_min_arr = np.log(all_abs[:, M - 1])
    mu2_max_arr[-1] = mu2_max_arr[0]
    mu2_min_arr[-1] = mu2_min_arr[0]
    mu2_arr = (mu2_max_arr + mu2_min_arr) / 2
    mu2_fun = interpolate.CubicSpline(theta1_list, mu2_arr, bc_type="periodic")

    def param_fun_2d(theta: tuple[float]):
        theta1, theta2 = theta
        mu2_val = mu2_fun(theta1, extrapolate="periodic")
        mu2_prime = mu2_fun(theta1, extrapolate="periodic", nu=1)
        param = np.array([
            E_ref,
            exp(mu1 + 1j * theta1),
            exp(mu2_val + 1j * theta2)
        ])
        dparam = np.array([
            0,
            1j * exp(mu1 + 1j * theta1),
            exp(mu2_val + 1j * theta2) * mu2_prime
        ])
        return param, dparam

    winding_fun.loop_fun = param_fun_2d

    fun_call = lambda t1, t2: winding_fun((t1, t2))

    res = integrate.nquad(fun_call, [[0, 2 * pi], [0, 2 * pi]])
    w_strip = res[0] / ((2 * pi) ** 2)

    return w_strip
