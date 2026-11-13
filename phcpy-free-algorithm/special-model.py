'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-05-12
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''


import BerryPy.TightBinding as tb
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import linalg as la
from cmath import sin, cos, sqrt, pi, exp, log
import gbz_solver as gbz
import multiprocessing as mp
import mpmath
import io
import contextlib


DEFAULT_PARAMETERS = {
    "a0": 0.8,
    "c0": 0.4 * np.exp(1j * np.pi / 12), # b0 in revised manuscript
}


def get_model(a0, c0):
    dim = 2
    site_num = 1
    intra_cell = []
    inter_cell = [
        [0, 0, 2, (0, 1)],
        [0, 0, 2, (0, -1)],
        [0, 0, a0, (-1, -1)],
        [0, 0, a0, (1, -1)],
        [0, 0, c0, (-1, 1)],
        [0, 0, c0, (1, 1)],
    ]
    return tb.TightBindingModel(dim, site_num, np.eye(2), intra_cell, inter_cell)


def check_beta2_zeros(betax: np.ndarray):
    a_arr = 2 + 0.8 * (betax + 1 / betax)
    # b_arr = (betax + 1 / betax) * np.exp(1j * np.pi / 12)
    b_arr = 0 * betax - 2
    c_arr = 2 + 0.4 * (betax + 1 / betax) * np.exp(1j * np.pi / 12)
    sqrt_delta = np.sqrt(b_arr ** 2 - 4 * a_arr * c_arr)
    betay_1 = (-b_arr + sqrt_delta) / (2 * a_arr)
    betay_2 = (-b_arr - sqrt_delta) / (2 * a_arr)
    plt.plot(np.angle(betax), np.abs(betay_1))
    plt.plot(np.angle(betax), np.abs(betay_2))
    plt.show()

    E = 0
    fun_val = (4 * a_arr * c_arr)
    plt.plot(fun_val.real, fun_val.imag, '-')
    plt.plot(0, 0, "x")
    plt.show()


def _get_xy_fname_suffix(Nx: int, Ny: int = None):
    if Ny is None:
        Ny = Nx
    if Nx == Ny:
        return f"xy-{Nx}"
    return f"xy-{Nx}-{Ny}"


def get_OBC_spectrum_xy(Nx: int, Ny: int = None):
    if Ny is None:
        Ny = Nx
    model = get_model(**DEFAULT_PARAMETERS)
    coords = get_xy_obc_coords(Nx, Ny)
    H_mat = model.generate_OBC_bulk(coords).todense()
    eigv, eigvec = la.eig(H_mat)
    fname_suffix = _get_xy_fname_suffix(Nx, Ny)
    with open(f"data/special-test-OBC-spectrum-{fname_suffix}.pkl", "wb") as f:
        pickle.dump((eigv, eigvec, coords), f)


def get_OBC_spectrum_diamond(N_layers: int):
    model = get_model(**DEFAULT_PARAMETERS)
    y_list = np.arange(N_layers)
    coord = []
    for y in y_list:
        for x in range(-y, y + 1):
            coord.append((x, y - N_layers))
    for x in range(-N_layers, N_layers + 1):
        coord.append((x, 0))
    for y in y_list:
        for x in range(-y, y + 1):
            coord.append((x, N_layers - y))
    H_mat = model.generate_OBC_bulk(coord).todense()
    eigv, eigvec = la.eig(H_mat)
    with open(f"data/special-test-OBC-spectrum-diamond-{N_layers}.pkl", "wb") as f:
        pickle.dump((eigv, eigvec, coord), f)


def plot_OBC_spectrum(fname_suffix):
    with open(f"data/special-test-OBC-spectrum-{fname_suffix}.pkl", "rb") as f:
        eigv, eigvec, coord = pickle.load(f)
    print(len(eigv))
    plt.plot(eigv.real, eigv.imag, 'x', label=fname_suffix)


def get_xy_obc_coords(Nx: int, Ny: int = None):
    if Ny is None:
        Ny = Nx
    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    x = x.flatten()
    y = y.flatten()
    return [(int(x[j]), int(y[j])) for j in range(Nx * Ny)]


def _get_mpmath_parameters(a0=None, c0=None):
    if a0 is None:
        a0 = DEFAULT_PARAMETERS["a0"]
    if c0 is None:
        c0 = DEFAULT_PARAMETERS["c0"]
    return mpmath.mpc(a0), mpmath.mpc(c0)


def _get_theoretical_eigenpairs_xy_mpmath(
    Nx: int,
    Ny: int = None,
    dps: int = 80,
    a0=None,
    c0=None,
    include_eigvec: bool = False,
):
    if Ny is None:
        Ny = Nx
    mpmath.mp.dps = dps
    a0_mp, c0_mp = _get_mpmath_parameters(a0=a0, c0=c0)
    eigv_list = []
    for mx in range(1, Nx + 1):
        theta_x = mpmath.pi * mx / (Nx + 1)
        cos_theta_x = mpmath.cos(theta_x)
        t_up = mpmath.mpc(2) + 2 * a0_mp * cos_theta_x
        t_down = mpmath.mpc(2) + 2 * c0_mp * cos_theta_x
        prefactor = 2 * mpmath.sqrt(t_up * t_down)
        rho = mpmath.sqrt(t_down / t_up)
        for my in range(1, Ny + 1):
            theta_y = mpmath.pi * my / (Ny + 1)
            eigv_list.append(
                {
                    "mx": mx,
                    "my": my,
                    "theta_x": theta_x,
                    "theta_y": theta_y,
                    "t_up": t_up,
                    "t_down": t_down,
                    "rho": rho,
                    "eigv_mp": prefactor * mpmath.cos(theta_y),
                }
            )
            if include_eigvec:
                eigv_list[-1]["eigvec_mp"] = get_theoretical_right_eigenvector_xy_mpmath(
                    Nx, Ny, mx, my, dps=dps, a0=a0, c0=c0
                )
    return eigv_list


def _attach_theoretical_eigvec_xy_mpmath(eigenpair: dict, Nx: int, Ny: int = None, dps: int = 80, a0=None, c0=None):
    if Ny is None:
        Ny = Nx
    enriched_pair = dict(eigenpair)
    enriched_pair["eigvec_mp"] = get_theoretical_right_eigenvector_xy_mpmath(
        Nx, Ny, eigenpair["mx"], eigenpair["my"], dps=dps, a0=a0, c0=c0
    )
    return enriched_pair


def get_OBC_spectrum_xy_theory(Nx: int, Ny: int = None, dps: int = 80, return_eigpairs: bool = True):
    if Ny is None:
        Ny = Nx
    eigv_data = _get_theoretical_eigenpairs_xy_mpmath(Nx, Ny, dps=dps)
    eigv = np.array([complex(item["eigv_mp"]) for item in eigv_data], dtype=complex)
    coords = get_xy_obc_coords(Nx, Ny)
    fname_suffix = _get_xy_fname_suffix(Nx, Ny).replace("xy-", "xy-theory-")
    with open(f"data/special-test-OBC-spectrum-{fname_suffix}.pkl", "wb") as f:
        pickle.dump((eigv, None, coords), f)
    if return_eigpairs:
        return eigv_data
    return eigv


def _as_mpmath_column(vec):
    vec = np.asarray(vec).flatten()
    mp_vec = mpmath.matrix(len(vec), 1)
    for idx, val in enumerate(vec):
        mp_vec[idx] = mpmath.mpc(val)
    return mp_vec


def _mpmath_vector_norm(vec):
    norm_sq = mpmath.mpf("0")
    for idx in range(vec.rows):
        norm_sq += abs(vec[idx]) ** 2
    return mpmath.sqrt(norm_sq)


def apply_OBC_hamiltonian_xy_mpmath(vec, Nx: int, Ny: int = None, a0=None, c0=None, dps: int = 80):
    if Ny is None:
        Ny = Nx
    mpmath.mp.dps = dps
    a0_mp, c0_mp = _get_mpmath_parameters(a0=a0, c0=c0)
    psi = _as_mpmath_column(vec)
    H_psi = mpmath.matrix(Nx * Ny, 1)
    for y in range(Ny):
        for x in range(Nx):
            curr = mpmath.mpc(0)
            if y > 0:
                curr += 2 * psi[(y - 1) * Nx + x]
                if x > 0:
                    curr += c0_mp * psi[(y - 1) * Nx + x - 1]
                if x + 1 < Nx:
                    curr += c0_mp * psi[(y - 1) * Nx + x + 1]
            if y + 1 < Ny:
                curr += 2 * psi[(y + 1) * Nx + x]
                if x > 0:
                    curr += a0_mp * psi[(y + 1) * Nx + x - 1]
                if x + 1 < Nx:
                    curr += a0_mp * psi[(y + 1) * Nx + x + 1]
            H_psi[y * Nx + x] = curr
    return H_psi


def get_theoretical_right_eigenvector_xy_mpmath(Nx: int, Ny: int = None, mx: int = 1, my: int = 1, dps: int = 80, a0=None, c0=None):
    if Ny is None:
        Ny = Nx
    mpmath.mp.dps = dps
    a0_mp, c0_mp = _get_mpmath_parameters(a0=a0, c0=c0)
    theta_x = mpmath.pi * mx / (Nx + 1)
    theta_y = mpmath.pi * my / (Ny + 1)
    t_up = mpmath.mpc(2) + 2 * a0_mp * mpmath.cos(theta_x)
    t_down = mpmath.mpc(2) + 2 * c0_mp * mpmath.cos(theta_x)
    rho = mpmath.sqrt(t_down / t_up)
    eigvec = mpmath.matrix(Nx * Ny, 1)
    for y in range(Ny):
        y_factor = (rho ** y) * mpmath.sin((y + 1) * theta_y)
        for x in range(Nx):
            eigvec[y * Nx + x] = mpmath.sin((x + 1) * theta_x) * y_factor
    norm = _mpmath_vector_norm(eigvec)
    if norm != 0:
        eigvec /= norm
    return eigvec


def get_eigenpair_residual_xy_mpmath(eigv, eigvec, Nx: int, Ny: int = None, dps: int = 80, a0=None, c0=None):
    if Ny is None:
        Ny = Nx
    mpmath.mp.dps = dps
    vec_mp = _as_mpmath_column(eigvec)
    H_vec = apply_OBC_hamiltonian_xy_mpmath(vec_mp, Nx, Ny, a0=a0, c0=c0, dps=dps)
    residual = H_vec - mpmath.mpc(eigv) * vec_mp
    scale = max(
        mpmath.mpf("1"),
        _mpmath_vector_norm(H_vec),
        abs(mpmath.mpc(eigv)) * _mpmath_vector_norm(vec_mp),
    )
    return _mpmath_vector_norm(residual) / scale


def get_average_residual_xy(
    Nx: int,
    Ny: int = None,
    dps: int = 80,
):
    if Ny is None:
        Ny = Nx

    fname_suffix = _get_xy_fname_suffix(Nx, Ny)
    spectrum_path = f"data/special-test-OBC-spectrum-{fname_suffix}.pkl"

    with open(spectrum_path, "rb") as f:
        eigv_num, eigvec_num, coords = pickle.load(f)
    theory_data = _get_theoretical_eigenpairs_xy_mpmath(Nx, Ny, dps=dps)

    # theory_residuals = []
    # for theory_item in theory_data:
    #     theory_item = _attach_theoretical_eigvec_xy_mpmath(theory_item, Nx, Ny, dps=dps)
    #     theory_residual = get_eigenpair_residual_xy_mpmath(
    #         theory_item["eigv_mp"], theory_item["eigvec_mp"], Nx, Ny, dps=dps
    #     )
    #     theory_residuals.append(float(theory_residual))
    #     print(theory_residuals)

    numeric_residuals = []
    for eig_ind in range(len(eigv_num)):
        numeric_residual = get_eigenpair_residual_xy_mpmath(
            eigv_num[eig_ind], eigvec_num[:, eig_ind], Nx, Ny, dps=dps
        )
        numeric_residuals.append(float(numeric_residual))
        print(numeric_residuals)

    result = {
        "Nx": Nx,
        "Ny": Ny,
        "dps": dps,
        "n_states": Nx * Ny,
        "coords": coords,
        "theory_mean_residual": float(np.mean(theory_residuals)),
        "theory_max_residual": float(np.max(theory_residuals)),
        "numeric_mean_residual": float(np.mean(numeric_residuals)),
        "numeric_max_residual": float(np.max(numeric_residuals)),
        "theory_residuals": np.array(theory_residuals, dtype=float),
        "numeric_residuals": np.array(numeric_residuals, dtype=float),
    }

    return result


def sweep_amoeba():
    model = get_model(**DEFAULT_PARAMETERS)
    coeffs, degs = model.get_characteristic_polynomial_data()
    
    N_points = 101
    E_re = np.linspace(-6.5, 6.5, N_points)
    E_im = np.linspace(-0.9, 0.9, N_points)
    E_re_mesh, E_im_mesh = np.meshgrid(E_re, E_im)
    E_mesh = E_re_mesh + 1j * E_im_mesh
    E_list = E_mesh.flatten()

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(gbz.check_amoeba, [(coeffs, degs, E, j / len(E_list)) for j, E in enumerate(E_list)])
    pool.close()
    pool.join()

    with open("data/special-test-amoeba.pkl", "wb") as f:
        pickle.dump(
            {
                "E_real": E_re,
                "E_imag": E_im,
                "results": results,
                "params": DEFAULT_PARAMETERS,
                "coeffs": coeffs,
                "degs": degs
            }, 
            f
        )


def sweep_SGBZ_x():
    model = get_model(**DEFAULT_PARAMETERS)
    coeffs, degs = model.get_characteristic_polynomial_data()
    
    N_points = 101
    E_re = np.linspace(-6.5, 6.5, N_points)
    E_im = np.linspace(-0.9, 0.9, N_points)
    E_re_mesh, E_im_mesh = np.meshgrid(E_re, E_im)
    E_mesh = E_re_mesh + 1j * E_im_mesh
    E_list = E_mesh.flatten()

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(gbz.check_SGBZ, [(coeffs, degs, E, j / len(E_list)) for j, E in enumerate(E_list)])
    pool.close()
    pool.join()

    with open("data/special-test-SGBZ-x.pkl", "wb") as f:
        pickle.dump(
            {
                "E_real": E_re,
                "E_imag": E_im,
                "results": results,
                "params": DEFAULT_PARAMETERS,
                "coeffs": coeffs,
                "degs": degs
            }, 
            f
        )


def sweep_SGBZ_y():
    model = get_model(**DEFAULT_PARAMETERS)
    coeffs, degs = model.get_characteristic_polynomial_data()
    degs = degs[:, [0,2,1]]
    
    N_points = 101
    E_re = np.linspace(-6.5, 6.5, N_points)
    E_im = np.linspace(-0.9, 0.9, N_points)
    E_re_mesh, E_im_mesh = np.meshgrid(E_re, E_im)
    E_mesh = E_re_mesh + 1j * E_im_mesh
    E_list = E_mesh.flatten()

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(gbz.check_SGBZ, [(coeffs, degs, E, j / len(E_list)) for j, E in enumerate(E_list)])
    pool.close()
    pool.join()

    with open("data/special-test-SGBZ-y.pkl", "wb") as f:
        pickle.dump(
            {
                "E_real": E_re,
                "E_imag": E_im,
                "results": results,
                "params": DEFAULT_PARAMETERS,
                "coeffs": coeffs,
                "degs": degs
            }, 
            f
        )


def plot_SGBZ(fname_suffix: str = "x"):
    with open("data/special-test-SGBZ-%s.pkl" % fname_suffix, "rb") as fp:
        data = pickle.load(fp)
    E_real, E_imag, results, params = data["E_real"], data["E_imag"], data["results"], data["params"]
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_list = (E_real_mesh + 1j * E_imag_mesh).flatten()
    coeffs, degs = data["coeffs"], data["degs"]

    success_mesh = np.array([res["success"] for res in results]).reshape(E_real_mesh.shape)
    failed_mesh = np.array([not res["success"] for res in results]).reshape(E_real_mesh.shape)
    success_mesh = np.array([res["success"] and res["is_PMGBZ"] for res in results]).reshape(E_real_mesh.shape)
    print(E_real_mesh[failed_mesh] + 1j * E_imag_mesh[failed_mesh])
    plt.plot(E_real_mesh[success_mesh], E_imag_mesh[success_mesh], '.', label="SGBZ")
    plt.plot(E_real_mesh[~success_mesh & ~failed_mesh], E_imag_mesh[~success_mesh & ~failed_mesh], '.', label="Non-SGBZ")
    plt.plot(E_real_mesh[failed_mesh], E_imag_mesh[failed_mesh], '.', label="Failed")
    plt.legend()

    # if which == "a1":
    #     model = Haldane_non_Hermitian_phase(*ALL_PARAMS)
    #     coeffs, degs = model.get_characteristic_polynomial_data()

    # else:
    #     coeffs, degs = data["coeffs"], data["degs"]

    # E_k_list = gbz.results_to_momenta(results, E_list)
    # E_k_list = np.array(E_k_list)
    # plt.plot(E_k_list[:,0].real, E_k_list[:,1].imag, ".")
    # plt.plot(E_k_list[:,0].real, E_k_list[:,2].imag, ".")
    # plt.show()


def plot_amoeba():
    with open("data/special-test-amoeba.pkl", "rb") as fp:
        data = pickle.load(fp)
    E_real, E_imag, results, params = data["E_real"], data["E_imag"], data["results"], data["params"]
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_list = (E_real_mesh + 1j * E_imag_mesh).flatten()
    coeffs, degs = data["coeffs"], data["degs"]

    success_mesh = np.array([res["success"] for res in results]).reshape(E_real_mesh.shape)
    failed_mesh = np.array([not res["success"] for res in results]).reshape(E_real_mesh.shape)
    success_mesh = np.array([res["success"] and res["is_amoeba"] for res in results]).reshape(E_real_mesh.shape)
    print(E_real_mesh[failed_mesh] + 1j * E_imag_mesh[failed_mesh])
    plt.plot(E_real_mesh[success_mesh], E_imag_mesh[success_mesh], '.', label="Amoeba")
    plt.plot(E_real_mesh[~success_mesh & ~failed_mesh], E_imag_mesh[~success_mesh & ~failed_mesh], '.', label="Non-amoeba")
    plt.plot(E_real_mesh[failed_mesh], E_imag_mesh[failed_mesh], '.', label="Failed")
    plt.legend()




def plot_zero_distribution(fname_suffix: str = "x"):
    with open(f"data/special-test-SGBZ-{fname_suffix}.pkl", "rb") as fp:
        data = pickle.load(fp)
    E_real, E_imag, results, params = data["E_real"], data["E_imag"], data["results"], data["params"]
    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    E_list = (E_real_mesh + 1j * E_imag_mesh).flatten()
    coeffs, degs = data["coeffs"], data["degs"]

    for sol_ind, res in enumerate(results):
        if res["success"] and res["is_PMGBZ"]:
            plt.title(f"E = {E_list[sol_ind]}")
            for point in res["PMGBZ_points"]:
                theta1 = point["theta1"]
                if "beta2" in point:                     # record of gbz_solver
                    plt.plot(theta1, np.angle(point["beta2"]), 'r+')
                elif "beta2_sols" in point:              # older record
                    for beta2_arr, marker in zip(point["beta2_sols"],
                                                 ('r+', 'bx', 'ko')):
                        plt.plot(theta1 * np.ones(len(beta2_arr)),
                                 np.angle(beta2_arr), marker)
                else:                                    # older record
                    for pair in point["beta2_pairs"]:
                        for beta2 in pair:
                            plt.plot(theta1, np.angle(beta2), 'r+')
            plt.show()


def plot_theoretical_spectrum(Lx: int, Ly: int):
    m = np.arange(1, Lx + 1)
    a0 = DEFAULT_PARAMETERS["a0"]
    b0 = DEFAULT_PARAMETERS["c0"]
    Am = 2 * (1 + a0 * np.cos(m * np.pi / (Lx + 1)))
    Bm = 2 * (1 + b0 * np.cos(m * np.pi / (Lx + 1)))
    E_mesh = np.zeros((Lx, Ly), dtype=complex)
    sqrt_AB = np.sqrt(Am * Bm)
    for n_ind in range(Ly):
        E_mesh[:, n_ind] = 2 * sqrt_AB * np.cos((n_ind + 1) * np.pi / (Ly + 1))
    E_list = E_mesh.flatten()
    plt.plot(E_list.real, E_list.imag, "+")


def _binary_search_amoeba_boundary_point(
    E_inside: complex,
    E_outside: complex,
    coeffs: np.ndarray,
    degs: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 60,
):
    """
    Binary search the amoeba boundary between an inside point and an outside point.

    Returns the boundary location (complex energy) where the classification flips.
    """
    for _ in range(max_iter):
        if abs(E_inside - E_outside) < tol:
            break
        E_mid = (E_inside + E_outside) / 2
        with contextlib.redirect_stdout(io.StringIO()):
            res = gbz.check_amoeba(coeffs, degs, E_mid, 0)
        if not res["success"]:
            print("Warning: solver failed!")
            break
        if res["is_amoeba"]:
            E_inside = E_mid
        else:
            E_outside = E_mid
    return (E_inside + E_outside) / 2


def _radial_binary_search(
    E_start: complex,
    direction: complex,
    coeffs: np.ndarray,
    degs: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 60,
):
    """Binary search the amoeba boundary along a ray E_start + t * direction.

    t=0 is assumed to be inside amoeba. Adaptively grows t until outside,
    then binary searches the boundary.
    """

    def _is_inside(E: complex) -> bool:
        with contextlib.redirect_stdout(io.StringIO()):
            res = gbz.check_amoeba(coeffs, degs, E, 0)
        if not res["success"]:
            print(f"Warning: solver failed at E = {E}")
        return res["success"] and res["is_amoeba"]

    if not _is_inside(E_start):
        print(f"Warning: E_start = {E_start} is not inside amoeba, ray may miss boundary")

    t_high = 1.0
    while _is_inside(E_start + t_high * direction):
        t_high += 1.0

    t_low = 0.0
    for _ in range(max_iter):
        if t_high - t_low < tol:
            break
        t_mid = (t_low + t_high) / 2
        if _is_inside(E_start + t_mid * direction):
            t_low = t_mid
        else:
            t_high = t_mid

    return E_start + (t_low + t_high) / 2 * direction


def refine_amoeba_corner(
    N: int = 20,
    E0: complex = 6.11 + 0.2161j,
    data_path: str = "data/special-test-amoeba.pkl",
    tol: float = 1e-4,
    max_iter: int = 60,
    x_y_scale: tuple = (3, 1),
):
    """Refine the amoebic spectrum boundary at corners via radial binary search.

    For each angle theta in [0, pi/2], shoots two rays:
      - Ray 1: E =  E0 + t * exp(1j * theta)
      - Ray 2: E = -E0 - t * exp(1j * theta)
    t=0 is inside amoeba; binary search finds the boundary where the
    classification flips.

    Parameters
    ----------
    N : int
        Number of angular sample points in [0, pi/2].
    E0 : complex
        Center point for the first-quadrant ray bundle.
    data_path : str
        Path to the existing amoeba sweep pickle.
    tol : float
        Convergence tolerance on t.
    max_iter : int
        Max binary search iterations per ray.
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    coeffs = data["coeffs"]
    degs = data["degs"]

    thetas = np.linspace(0, np.pi / 2, N)

    # Verify E0 and -E0 are inside amoeba
    def _is_inside(E: complex) -> bool:
        with contextlib.redirect_stdout(io.StringIO()):
            res = gbz.check_amoeba(coeffs, degs, E, 0)
        return res["success"] and res["is_amoeba"]

    if not _is_inside(E0):
        print(f"Warning: E0 = {E0} is not inside amoeba")
    if not _is_inside(-E0):
        print(f"Warning: -E0 = {-E0} is not inside amoeba")

    boundary_points = []

    for theta in thetas:
        # direction = np.exp(1j * theta)
        print(theta / np.pi, "pi")
        direction = np.cos(theta) * x_y_scale[0] + 1j * np.sin(theta) * x_y_scale[1]
        direction /= abs(direction)

        # Quadrant 1: E0 + t * exp(1j * theta)
        pt1 = _radial_binary_search(E0, direction, coeffs, degs, tol, max_iter)
        boundary_points.append(pt1)

        # Quadrant 3: -E0 - t * exp(1j * theta)
        pt2 = _radial_binary_search(-E0, -direction, coeffs, degs, tol, max_iter)
        boundary_points.append(pt2)

    boundary_points = np.array(boundary_points)

    out_path = data_path.replace(".pkl", "-radial-boundary.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "boundary_points": boundary_points,
            "E0": E0,
            "N": N,
            "tol": tol,
        }, f)
    print(f"Saved {len(boundary_points)} boundary points to {out_path}")

    return boundary_points


def refine_amoeba_boundary(
    data_path: str = "data/special-test-amoeba.pkl",
    tol: float = 1e-8,
):
    """
    Refine the amoebic spectrum boundary via binary search on existing grid data.

    Scans all adjacent horizontal and vertical grid point pairs. Where the
    classification differs (True/False), binary search pinpoints the boundary.
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    E_real = data["E_real"]
    E_imag = data["E_imag"]
    results = data["results"]
    coeffs = data["coeffs"]
    degs = data["degs"]

    n_y = len(E_imag)
    n_x = len(E_real)

    is_amoeba = np.array([
        res["success"] and res["is_amoeba"] for res in results
    ]).reshape(n_y, n_x)

    boundary_points = []

    # Horizontal adjacencies (left-right neighbor pairs)
    for i in range(n_y):
        for j in range(n_x - 1):
            if is_amoeba[i, j] != is_amoeba[i, j + 1]:
                E1 = E_real[j] + 1j * E_imag[i]
                E2 = E_real[j + 1] + 1j * E_imag[i]
                print(E1, E2)
                if is_amoeba[i, j]:
                    pt = _binary_search_amoeba_boundary_point(E1, E2, coeffs, degs, tol)
                else:
                    pt = _binary_search_amoeba_boundary_point(E2, E1, coeffs, degs, tol)
                boundary_points.append(pt)

    # Vertical adjacencies (top-bottom neighbor pairs)
    for i in range(n_y - 1):
        for j in range(n_x):
            if is_amoeba[i, j] != is_amoeba[i + 1, j]:
                E1 = E_real[j] + 1j * E_imag[i]
                E2 = E_real[j] + 1j * E_imag[i + 1]
                if is_amoeba[i, j]:
                    pt = _binary_search_amoeba_boundary_point(E1, E2, coeffs, degs, tol)
                else:
                    pt = _binary_search_amoeba_boundary_point(E2, E1, coeffs, degs, tol)
                boundary_points.append(pt)

    boundary_points = np.array(boundary_points)

    if len(boundary_points) > 0:
        center = np.mean(boundary_points)
        angles = np.angle(boundary_points - center)
        boundary_points = boundary_points[np.argsort(angles)]

    out_path = data_path.replace(".pkl", "-boundary.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "boundary_points": boundary_points,
            "tol": tol,
            "E_real": E_real,
            "E_imag": E_imag,
        }, f)
    print(f"Saved {len(boundary_points)} boundary points to {out_path}")

    return boundary_points


def rearrange_amoeba_boundary():
    with open("data/special-test-amoeba-boundary.pkl", "rb") as fp:
        data = pickle.load(fp)
    
    boundary_pts = data["boundary_points"]
    
    with open("data/special-test-amoeba-radial-boundary.pkl", "rb") as fp:
        data = pickle.load(fp)
    boundary_pts = np.hstack([boundary_pts, data["boundary_points"]])

    x_neg = boundary_pts[boundary_pts.real < 0]
    x_pos = boundary_pts[boundary_pts.real >= 0]

    x_cen = -0.2j

    x_pos_arg = np.argsort(np.angle(x_pos - x_cen))
    x_pos = x_pos[x_pos_arg]
    x_neg_arg = np.argsort(np.angle(-x_neg - x_cen))
    x_neg = x_neg[x_neg_arg]

    boundary_pts = np.hstack([x_pos, x_neg])

    plt.plot(boundary_pts.real, boundary_pts.imag, '.-')
    plt.show()
    with open("data/special-test-amoeba-boundary-rearranged.pkl", "wb") as fp:
        pickle.dump(boundary_pts, fp)


def plot_amoeba_boundary(data_path: str = "data/special-test-amoeba.pkl",
                         boundary_path: str = None):
    """Plot the original amoeba grid classification with the refined boundary overlay."""
    if boundary_path is None:
        boundary_path = data_path.replace(".pkl", "-boundary.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    E_real = data["E_real"]
    E_imag = data["E_imag"]
    results = data["results"]

    E_real_mesh, E_imag_mesh = np.meshgrid(E_real, E_imag)
    is_amoeba = np.array([
        res["success"] and res["is_amoeba"] for res in results
    ]).reshape(E_real_mesh.shape)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(E_real_mesh, E_imag_mesh, is_amoeba, cmap="Blues", shading="auto")

    with open(boundary_path, "rb") as f:
        bdata = pickle.load(f)
    boundary = bdata["boundary_points"]
    plt.plot(boundary.real, boundary.imag, "r.-", markersize=2, linewidth=1, label="Refined boundary")

    plt.xlabel("Re(E)")
    plt.ylabel("Im(E)")
    plt.legend()
    plt.tight_layout()


def plot_ameoba_boundary_line():
    with open("data/special-test-amoeba-boundary-rearranged.pkl", "rb") as fp:
        boundary_pts = pickle.load(fp)
    plt.plot(boundary_pts.real, boundary_pts.imag)


if __name__ == "__main__":
    # betax = np.exp(1j * np.linspace(-np.pi, np.pi, 1000))
    # check_beta2_zeros(betax)
    # plt.show()
    plt.figure()
    plt.title("x-SGBZ")
    plot_SGBZ("x")
    plt.figure()
    plt.title("y-SGBZ")
    plot_SGBZ("y")
    plt.show()

    # get_OBC_spectrum_diamond(80)
    # get_OBC_spectrum_xy(114)

    # print(get_average_residual_xy(114, dps=50))
    # plot_theoretical_spectrum(57, 57)
    # plot_OBC_spectrum("xy-57")
    # plt.show()

    # sweep_amoeba()
    # plot_amoeba()
    # plt.show()
    # refine_amoeba_boundary(tol=1e-4)
    # rearrange_amoeba_boundary()
    # plot_ameoba_boundary_line()
    # plt.show()

    # refine_amoeba_corner(N=51)
