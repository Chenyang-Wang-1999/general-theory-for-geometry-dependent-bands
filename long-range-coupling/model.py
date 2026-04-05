'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-02-09
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import mpmath
from cmath import exp, log
from math import sin, cos, sqrt, pi
import io
import sys
from contextlib import redirect_stdout
from scipy import sparse
import pickle
from scipy import linalg as la
import matplotlib.pyplot as plt
import long_range_cpp as cpp
import time
from typing import Callable

CPP_MODE = True

DEFAULT_PARAMS = {
    "gammax": 0.2,
    "gammay": 0.0,
    "tx": 1,
    "ty": 0.8,
    "v": 0.5,
    "cx": 0.1,
    "cy": 0.2,
    "kappax": 1,
    "kappay": 1,
}

def safe_nsum(func, range, method="r", tol=1e-15):
    f = io.StringIO()
    
    # 重定向stdout到缓冲区
    with redirect_stdout(f):
        result = mpmath.nsum(func, range, method=method, tol=tol, verbose=True)

    # 获取所有输出
    output = f.getvalue()

    if "Warning:" in output:
        raise ArithmeticError("nsum failed: \n" + output)
    else:
        return result


class LongRangeModel:
    tx1: complex
    tx2: complex
    ty1: complex
    ty2: complex
    v: float
    cx: float
    cy: float
    kappax: float
    kappay: float
    safe_mode: bool = False

    def __init__(self, tx1: complex, tx2: complex, ty1: complex, ty2: complex, v: float, cx: float, cy: float, kappax: float, kappay: float, safe_mode: bool = False):
        self.tx1 = tx1
        self.tx2 = tx2
        self.ty1 = ty1
        self.ty2 = ty2
        self.v = v
        self.cx = cx
        self.cy = cy
        self.kappax = kappax
        self.kappay = kappay
        self.safe_mode = safe_mode
        self.coupling_dict = {
            (1, 0): tx1,
            (-1, 0): tx2,
            (0, 1): ty1,
            (0, -1): ty2,
            (1, 1): v,
            (-1, -1): v,
            (1, -1): v,
            (-1, 1): v
        }

    def char_poly(self, betax: complex, betay: complex) -> complex:
        if CPP_MODE:
            func_1 = lambda ell: cpp.beta_series(int(ell), self.kappax, betax)
            func_2 = lambda ell: cpp.beta_series(int(ell), self.kappay, betay)
        else:
            func_1 = lambda ell: mpmath.exp(-self.kappax * ell) * (betax ** ell + betax ** (-ell))
            func_2 = lambda ell: mpmath.exp(-self.kappay * ell) * (betay ** ell + betay ** (-ell))

        args_1 = (
            func_1,
            [2, mpmath.inf],
        )

        args_2 = (
            func_2,
            [2, mpmath.inf],
        )

        method = "r"
        tol = 1e-15

        if self.safe_mode:
            series_1 = safe_nsum(*args_1, method=method, tol=tol)
            series_2 = safe_nsum(*args_2, method=method, tol=tol)
        else:
            series_1 = mpmath.nsum(
                *args_1,
                method=method,
                tol=tol,
            )

            series_2 = mpmath.nsum(
                *args_2,
                method=method,
                tol=tol,
            )

        return (
            self.tx1 * betax ** (-1) + self.tx2 * betax + self.ty1 * betay ** (-1) + self.ty2 * betay
            + self.v * (betax * betay + betax / betay + betay / betax + 1 / betax / betay)
            + self.cx * series_1 + self.cy * series_2
        )

    def char_poly_dx(self, betax: complex, betay: complex) -> complex:
        if CPP_MODE:
            func_1 = lambda ell: cpp.dbeta_series(int(ell), self.kappax, betax)
        else:
            func_1 = lambda ell: mpmath.exp(-self.kappax * ell) * (ell * betax ** (ell - 1) - ell * betax ** (-ell - 1))

        args_1 = (
            func_1,
            [2, mpmath.inf]
        )
        method = "r"
        tol = 1e-15
        if self.safe_mode:
            series_1 = safe_nsum(*args_1, method=method, tol=tol)
        else:
            series_1 = mpmath.nsum(
                *args_1,
                method=method,
                tol=tol,
            )

        return (
            - self.tx1 * betax ** (-2) + self.tx2 + self.v * (betay + 1/betay - betax ** (-2) * betay - betax ** (-2) / betay)
            + self.cx * series_1
        )
    
    def char_poly_dy(self, betax: complex, betay: complex) -> complex:
        if CPP_MODE:
            func_2 = lambda ell: cpp.dbeta_series(int(ell), self.kappay, betay)
        else:
            func_2 = lambda ell: mpmath.exp(-self.kappay * ell) * (ell * betay ** (ell - 1) - ell * betay ** (-ell - 1))

        args_2 = (
            func_2,
            [2, mpmath.inf]
        )
        method = "r"
        tol = 1e-15
        if self.safe_mode:
            series_2 = safe_nsum(*args_2, method=method, tol=tol)
        else:
            series_2 = mpmath.nsum(
                *args_2,
                method=method,
                tol=tol,
            )
        return (
            - self.ty1 * betay ** (-2) + self.ty2 + self.v * (betax + 1/betax - betay ** (-2) * betax - betay ** (-2) / betax)
            + self.cy * series_2
        )
    
    def get_OBC_Hamiltonian(self, coord_list: np.ndarray[int, 2]) -> sparse.coo_matrix:
        I_list = []
        J_list = []
        V_list = []
        for ind_i in range(len(coord_list)):
            for ind_j in range(len(coord_list)):
                row_coord = coord_list[ind_i]
                col_coord = coord_list[ind_j]
                coupling_range = tuple(row_coord - col_coord)
                if coupling_range in self.coupling_dict:
                    I_list.append(ind_i)
                    J_list.append(ind_j)
                    V_list.append(self.coupling_dict[coupling_range])
                else:
                    if coupling_range[0] == 0 and abs(coupling_range[1]) > 1:
                        # y long-range
                        I_list.append(ind_i)
                        J_list.append(ind_j)
                        V_list.append(self.cy * exp(-self.kappay * abs(coupling_range[1])))
                    elif coupling_range[1] == 0 and abs(coupling_range[0]) > 1:
                        # x long-range
                        I_list.append(ind_i)
                        J_list.append(ind_j)
                        V_list.append(self.cx * exp(-self.kappax * abs(coupling_range[0])))
        return sparse.coo_matrix((V_list, (I_list, J_list)), shape=(len(coord_list), len(coord_list)))


def get_long_range_OBC(
    coupling_fun: Callable[[tuple[int, int]], complex],
    coords: np.ndarray[int, 2],
):
    I_list = []
    J_list = []
    V_list = []
    for ind_i in range(len(coords)):
        for ind_j in range(len(coords)):
            row_coord = coords[ind_i]
            col_coord = coords[ind_j]
            coupling_range = tuple(row_coord - col_coord)
            coeff = coupling_fun(coupling_range)
            if coeff is not None:
                I_list.append(ind_i)
                J_list.append(ind_j)
                V_list.append(coeff)
    return sparse.coo_matrix((V_list, (I_list, J_list)), shape=(len(coords), len(coords)))


def test_derivative():
    tx1, tx2, ty1, ty2 = np.random.randn(4) + 1j * np.random.randn(4)
    kappax = np.random.rand() * 2 + 2
    kappay = np.random.rand() * 2 + 2
    # kappax = -1
    # kappay = -1
    v, cx, cy = np.random.randn(3)
    model = LongRangeModel(tx1, tx2, ty1, ty2, v, cx, cy, kappax, kappay, safe_mode=True)

    betax, betay = np.random.randn(2) + 1j * np.random.randn(2)
    dl = 1e-4
    dbetax, dbetay = dl * (np.random.randn(2) + 1j * np.random.randn(2))

    v0 = model.char_poly(betax, betay)
    print(np.log(betax).real, np.log(betax).imag)
    print(kappax, kappay)
    dv0 = np.array(
        [model.char_poly_dx(betax, betay), model.char_poly_dy(betax, betay)]
    )
    v1 = model.char_poly(betax + dbetax, betay + dbetay)

    print(v1 - v0)
    print(dv0.dot([dbetax, dbetay]))
    print(abs(v1 - v0 - dv0.dot([dbetax, dbetay])) / abs(v1 - v0))


def test_OBC_Hamiltonian():
    tx1 = 1.1
    tx2 = 1
    ty1 = 2.1
    ty2 = 2
    v = 0.5
    cx = 0.1
    cy = 0.2
    kappax = 1.5
    kappay = 2.5

    x = np.arange(5)
    y = np.arange(5)
    x_mesh, y_mesh = np.meshgrid(x, y)
    coord_list = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    model = LongRangeModel(tx1, tx2, ty1, ty2, v, cx, cy, kappax, kappay)
    H = model.get_OBC_Hamiltonian(coord_list)
    H_dense = H.todense()
    with open("data/H.txt", "w") as f:
        for i, j in np.column_stack(H.nonzero()):
            print(coord_list[i,:], coord_list[j,:], H_dense[i, j], file=f)


def test_Hermitian():
    tx1 = DEFAULT_PARAMS["tx"]
    tx2 = DEFAULT_PARAMS["tx"]
    ty1 = DEFAULT_PARAMS["ty"]
    ty2 = DEFAULT_PARAMS["ty"]
    v = DEFAULT_PARAMS["v"]
    cx = DEFAULT_PARAMS["cx"]
    cy = DEFAULT_PARAMS["cy"]
    kappax = DEFAULT_PARAMS["kappax"]
    kappay = DEFAULT_PARAMS["kappay"]
    model = LongRangeModel(tx1, tx2, ty1, ty2, v, cx, cy, kappax, kappay, safe_mode=True)

    # Get BZ
    print("BZ start")
    kx = np.linspace(-pi, pi, 51)
    ky = np.linspace(-pi, pi, 51)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    E_mesh = np.zeros_like(kx_mesh, dtype=complex)
    for i in range(len(kx)):
        for j in range(len(ky)):
            E_mesh[i, j] = model.char_poly(exp(1j * kx_mesh[i, j]), exp(1j * ky_mesh[i, j]))
    with open("data/Hermitian-BZ.pkl", "wb") as f:
        pickle.dump(E_mesh, f)
    
    # OBC
    print("OBC start")
    Nx, Ny = 50, 50
    x = np.arange(Nx)
    y = np.arange(Ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    coord_list = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    H = model.get_OBC_Hamiltonian(coord_list)
    H_dense = H.todense()
    print("OBC assemble done")
    eigv, eigvec = la.eig(H_dense)
    with open("data/Hermitian-OBC.pkl", "wb") as f:
        pickle.dump((coord_list, eigv, eigvec), f)


def plot_bands():
    with open("data/Hermitian-BZ.pkl", "rb") as f:
        E_mesh = pickle.load(f)
    plt.plot(E_mesh.real, E_mesh.imag + 0.1, '.')
    with open("data/Hermitian-OBC.pkl", "rb") as f:
        coord_list, eigv, eigvec = pickle.load(f)
    plt.plot(eigv.real, eigv.imag, 'x')
    plt.show()


def test_cpp_acceleration():
    tx1 = DEFAULT_PARAMS["tx"]
    tx2 = DEFAULT_PARAMS["tx"]
    ty1 = DEFAULT_PARAMS["ty"]
    ty2 = DEFAULT_PARAMS["ty"]
    v = DEFAULT_PARAMS["v"]
    cx = DEFAULT_PARAMS["cx"]
    cy = DEFAULT_PARAMS["cy"]
    kappax = DEFAULT_PARAMS["kappax"]
    kappay = DEFAULT_PARAMS["kappay"]
    model = LongRangeModel(tx1, tx2, ty1, ty2, v, cx, cy, kappax, kappay, safe_mode=True)

    kx = np.linspace(-pi, pi, 21)
    ky = np.linspace(-pi, pi, 21)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    E_mesh = np.zeros_like(kx_mesh, dtype=complex)
    # Test
    start_time = time.time()
    for i in range(len(kx)):
        for j in range(len(ky)):
            E_mesh[i, j] = model.char_poly(exp(1j * kx_mesh[i, j]), exp(1j * ky_mesh[i, j]))
    end_time = time.time()
    print("Time:", end_time - start_time)


if __name__ == '__main__':
    # test_derivative()
    # test_OBC_Hamiltonian()
    test_Hermitian()
    plot_bands()
    # test_cpp_acceleration()
