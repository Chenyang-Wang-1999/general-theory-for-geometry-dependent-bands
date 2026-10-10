'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-05-14
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import poly_tools as pt
import partial_GBZ_solver as pGs
import GBZ_manifold as Gm
import BerryPy.TightBinding as tb
from math import sin, cos, pi
from cmath import exp, log
import matplotlib.pyplot as plt
import os
import pickle
from scipy import linalg as la
import shapely
from scipy.sparse import linalg as spla
from scipy import sparse
from scipy import optimize
import brute_force_GBZ as bfG

import sys
sys.path.append('../paper/HN-model')
import HN_model_common as HN


def HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2) -> tb.TightBindingModel:
    inter_cell = [
        [0, 0, Jx1, (1,0)],
        [0, 0, Jx2, (-1,0)],
        [0, 0, Jy1, (0,1)],
        [0, 0, Jy2, (0,-1)]
    ]
    return tb.TightBindingModel(2, 1, np.array([[1,0],[0,1]]), [], inter_cell, [[0,0]])


def HN_model(Jx1, Jx2, Jy1, Jy2):
    coeffs = pt.CScalarVec(
        [1, -Jx1, -Jx2, -Jy1, -Jy2]
    )
    degs = pt.CLaurentIndexVec([
        # E, betax, betay
        1, 0, 0,
        0, -1, 0,
        0, 1, 0,
        0, 0, -1,
        0, 0, 1
    ])
    f = pt.CLaurent(3)
    f.set_Laurent_by_terms(coeffs, degs)
    return f


def get_beta_GBZ(k1, k2):
    Jx = 1 + 2j
    Jy = 1 - 2j
    delta = np.pi / 3
    gammax = 0.9
    gammay = 0.8
    Delta_xy = 0.2 * np.pi

    deltax = delta + Delta_xy / 2
    deltay = delta - Delta_xy / 2

    Jx1 = Jx * gammax * np.exp(1j * deltax)
    Jx2 = Jx.conjugate() / gammax * np.exp(1j * deltax)
    Jy1 = Jy * gammay * np.exp(1j * deltay)
    Jy2 = Jy.conjugate() / gammay * np.exp(1j * deltay)

    r = np.sqrt(np.abs(Jx1 * Jy1 / Jx2 / Jy2))
    beta_1 = r * np.exp(1j * k1)
    beta_2 = np.sqrt(np.abs((Jx2 * beta_1 + Jy1) / (Jx1 / beta_1 + Jy2))) * np.exp(1j * k2)
    E_ref = Jx1 * beta_2 / beta_1 + Jx2 * beta_1 / beta_2 + Jy1 / beta_2 + Jy2 * beta_2
    E_ref = 1e-2

    # 1 plot strip GBZ spectrum
    plt.figure()
    bound, loop = HN.get_11_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2)
    plt.plot(loop.real, loop.imag)
    bound, loop = HN.get_x_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2)
    plt.plot(loop.real, loop.imag)
    plt.plot(E_ref.real, E_ref.imag, 'x')
    plt.plot([-np.cos(delta), np.cos(delta)], [-np.sin(delta), np.sin(delta)], 'k')
    plt.plot([np.sin(delta), -np.sin(delta)], [-np.cos(delta), np.cos(delta)], 'k')
    plt.axis("equal")
    

    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )
    coeffs, degs = model.get_characteristic_polynomial_data()

    f = pt.CLaurent(3)
    f.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )
    # print(f.to_str(pt.CStrVec(['E', 'x', 'y'])))
    # input()
    solver = pGs.GBZMainSolver(f)
    solver.get_beta_GBZ(E_ref)

    plt.figure()
    for loop in solver.beta_GBZ_glued:
        all_points = np.zeros((len(loop.point_vec), 2), dtype=complex)
        for point_id, point in enumerate(loop.point_vec):
            Gm.CP1_to_chart(point, [0,0])
            all_points[point_id, :] = point.coords
        plt.plot(all_points[:,0].real, all_points[:,0].imag, 'b-')

    plt.axis([-2, 2, -2, 2])

    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(r * np.cos(theta), r * np.sin(theta), 'r--')

    plt.show()


def paper_get_non_Bloch_sols(E_ref: complex):
    ''' Plot non-Bloch solutions to compare with Amoeba formulation '''
    model = HN_model_tightbinding(*HN.Default_Model_Params)
    model = model.get_supercell(
        [(0,0)], 
        np.array(
            [[1,0],
             [1,1]],
            dtype=int
        )
    )
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(3)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    # solve non-Bloch solutions
    solver = pGs.GBZMainSolver(char_poly)
    solver.get_beta_GBZ(E_ref)
    non_Bloch_sol = solver.beta_GBZ_glued

    # To ndarray
    all_sols = []
    for curr_loop in non_Bloch_sol:
        curr_sol = np.zeros(
            (len(curr_loop.point_vec), 2),
            dtype=complex
        )
        for point_ind, curr_point in enumerate(curr_loop.point_vec):
            Gm.CP1_to_chart(curr_point, [0,0])
            curr_sol[point_ind,:] = curr_point.coords
        all_sols.append(curr_sol)
    
    # save
    with open("data/paper-non-Bloch-sol-E_%.2f_%.2f.pkl" % (E_ref.real, E_ref.imag), "wb") as fp:
        pickle.dump(all_sols, fp)


def random_polygon_OBC(radius_min: int = 25, radius_max: int = 50, n_sides: int = 4):
    r_list = np.random.rand(n_sides) * (radius_max - radius_min) + radius_min
    min_total_angle = 1.2 * pi
    theta_list = np.random.rand(n_sides - 1) * ((2 * pi - min_total_angle) / (n_sides - 1)) + min_total_angle/(n_sides - 1)
    theta_initial = np.random.rand() * 2 * pi
    arg_list = np.hstack([[0], np.cumsum(theta_list)]) + theta_initial
    return polygon_OBC(r_list, arg_list)

def polygon_OBC(r_list, arg_list):
    n_sides = len(r_list)
    unit_vec = np.exp(1j * arg_list)

    # get cusps
    cusp_list = []
    for j in range(n_sides):
        x0 = unit_vec[j] * r_list[j]
        v0 = unit_vec[j] * 1j
        x1 = unit_vec[(j+1)%n_sides] * r_list[(j+1)%n_sides]
        v1 = unit_vec[(j+1)%n_sides] * 1j

        M = np.array([
            [v0.real, v1.real],
            [v0.imag, v1.imag]
        ])
        b = np.array([x1.real - x0.real, x1.imag - x0.imag])
        t = la.solve(M, b)
        x = x0 + t[0] * v0
        cusp_list.append((x.real, x.imag))
    
    # get bound box
    cusp_list = np.asarray(cusp_list)
    xmin = np.min(cusp_list[:,0])
    xmax = np.max(cusp_list[:,0])
    ymin = np.min(cusp_list[:,1])
    ymax = np.max(cusp_list[:,1])
    xmin = int(np.floor(xmin))
    xmax = int(np.ceil(xmax))
    ymin = int(np.floor(ymin))
    ymax = int(np.ceil(ymax))

    # get all points
    all_points = []
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            is_in_polygon = True
            for side_id in range(n_sides):
                if((x * unit_vec[side_id].real + y * unit_vec[side_id].imag) > r_list[side_id]):
                    is_in_polygon = False
                    break
            if(is_in_polygon):
                all_points.append((x,y))

    return r_list, arg_list, all_points


def generate_diamond_or_square_OBC(radius: int, is_diamond: bool = True):
    r_list = [radius, radius/4, radius, radius/4]
    n_sides = 4
    if(is_diamond):
        arg_list = np.array([
            pi / 4, 
            3 * pi / 4, 
            5 * pi / 4, 
            7 * pi / 4])
    else:
        arg_list = np.array([
            0, 
            pi / 2,
            pi,
            3 * pi / 2
        ])
    unit_vec = np.exp(1j * arg_list)

    # get cusps
    cusp_list = []
    for j in range(n_sides):
        x0 = unit_vec[j] * r_list[j]
        v0 = unit_vec[j] * 1j
        x1 = unit_vec[(j+1)%n_sides] * r_list[(j+1)%n_sides]
        v1 = unit_vec[(j+1)%n_sides] * 1j

        M = np.array([
            [v0.real, v1.real],
            [v0.imag, v1.imag]
        ])
        b = np.array([x1.real - x0.real, x1.imag - x0.imag])
        t = la.solve(M, b)
        x = x0 + t[0] * v0
        cusp_list.append((x.real, x.imag))
    
    # get bound box
    cusp_list = np.asarray(cusp_list)
    xmin = np.min(cusp_list[:,0])
    xmax = np.max(cusp_list[:,0])
    ymin = np.min(cusp_list[:,1])
    ymax = np.max(cusp_list[:,1])
    xmin = int(np.floor(xmin))
    xmax = int(np.ceil(xmax))
    ymin = int(np.floor(ymin))
    ymax = int(np.ceil(ymax))
    
    # get all points
    all_points = []
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            is_in_polygon = True
            for side_id in range(n_sides):
                if((x * unit_vec[side_id].real + y * unit_vec[side_id].imag) > r_list[side_id]):
                    is_in_polygon = False
                    break
            if(is_in_polygon):
                all_points.append((x,y))
    
    return r_list, arg_list, all_points


def random_elliptic_OBC(radius: int = 50):
    aspect = np.random.rand() * 0.5 + 0.5
    angle = np.random.rand() * pi/2
    shape_param = (aspect, angle)
    a = radius
    b = radius * aspect

    # find the bounding box
    theta = np.linspace(0, 2 * pi, 100)
    p = (a * np.cos(theta) + 1j * b * np.sin(theta)) * np.exp(1j * angle)
    xmin = int(np.floor(np.min(p.real)))
    xmax = int(np.ceil(np.max(p.real))) 
    ymin = int(np.floor(np.min(p.imag))) 
    ymax = int(np.ceil(np.max(p.imag))) 

    # traverse all the points in the box
    point_list = []
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            curr_point = x + 1.0j * y
            curr_point /= exp(1j * angle)
            if((curr_point.real/a) ** 2 + (curr_point.imag/b) ** 2 < 1):
                point_list.append((x,y))
    
    return shape_param, point_list


def test_polygon():
    r_list, arg_list, all_points = random_polygon_OBC(20, 40)
    print(len(all_points))
    all_points = np.asarray(all_points)
    plt.plot(all_points[:,0], all_points[:,1], '.')
    plt.show()


def test_ellipse():
    N = 25
    shape_param, point_list = random_elliptic_OBC(N)
    print(shape_param[0], shape_param[1]/pi)
    fig = plt.figure()
    ax = fig.gca()
    for point in point_list:
        ax.plot(point[0], point[1], 'b.')
    
    theta = np.linspace(0, 2 * pi, 500) 
    p = N * np.cos(theta) + 1j * N * shape_param[0] * np.sin(theta)
    p *= exp(1j * shape_param[1])
    ax.plot(p.real, p.imag)
    plt.show()


def spectrum_ellipse():
    file_id = 0
    while(os.path.exists("data/HN-model-OBC-ellipse-%d-0.pkl"%(file_id))):
        file_id += 1
    fname_prefix = "data/HN-model-OBC-ellipse-%d"%(file_id)

    Jx1 = np.random.randn() + 1j * np.random.randn()
    Jx2 = np.random.randn() + 1j * np.random.randn()
    Jy1 = np.random.randn() + 1j * np.random.randn()
    Jy2 = np.random.randn() + 1j * np.random.randn()
    # Jy2_norm = np.random.randn()
    # Jy2_arg = log(Jx1).imag + log(Jx2).imag - log(Jy1).imag
    # Jy2 = Jy2_norm * exp(1j * Jy2_arg)
    model_param = (Jx1, Jx2, Jy1, Jy2)

    model = HN_model_tightbinding(*model_param)

    r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
    model = model.get_exponentially_scaled_system(r_vec)

    N = 25
    for j in range(10):
        shape_param, all_points = random_elliptic_OBC(N)
        H = model.generate_OBC_bulk(all_points)
        eigv, eigvec = la.eig(H.todense())

        with open("%s-%d.pkl"%(fname_prefix, j), "wb") as fp:
            pickle.dump((N, model_param, shape_param, all_points, eigv, eigvec, r_vec), fp)


def spectrum_polygon():
    file_id = 0
    while(os.path.exists("data/HN-model-OBC-polygon-%d-0.pkl"%(file_id))):
        file_id += 1
    fname_prefix = "data/HN-model-OBC-polygon-%d"%(file_id)

    # Jx1 = np.random.randn() + 1j * np.random.randn()
    # Jx2 = np.random.randn() + 1j * np.random.randn()
    # Jy1 = np.random.randn() + 1j * np.random.randn()
    # Jy2 = np.random.randn() + 1j * np.random.randn()
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params 
    # Jy2_norm = np.random.randn()
    # Jy2_arg = log(Jx1).imag + log(Jx2).imag - log(Jy1).imag
    # Jy2 = Jy2_norm * exp(1j * Jy2_arg)
    model_param = (Jx1, Jx2, Jy1, Jy2)

    model = HN_model_tightbinding(*model_param)

    r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
    model = model.get_exponentially_scaled_system(r_vec)

    radius_min = 30
    radius_max = 30
    n_sides = 4
    for j in range(10):
        r_list, arg_list, all_points = random_polygon_OBC(radius_min, radius_max, n_sides)
        H = model.generate_OBC_bulk(all_points)
        eigv, eigvec = la.eig(H.todense())

        with open("%s-%d.pkl"%(fname_prefix, j), "wb") as fp:
            pickle.dump((r_list, arg_list, model_param, all_points, eigv, eigvec, r_vec), fp)


def spectrum_diamond_square():
    file_id = 0
    while(os.path.exists("data/HN-model-OBC-diamond-square-%d-0.pkl"%(file_id))):
        file_id += 1
    fname_prefix = "data/HN-model-OBC-diamond-square-%d"%(file_id)

    Jx1 = np.random.randn() + 1j * np.random.randn()
    Jx2 = np.random.randn() + 1j * np.random.randn()
    Jy1 = np.random.randn() + 1j * np.random.randn()
    Jy2 = np.random.randn() + 1j * np.random.randn()
    # Jy2_norm = np.random.randn()
    # Jy2_arg = log(Jx1).imag + log(Jx2).imag - log(Jy1).imag
    # Jy2 = Jy2_norm * exp(1j * Jy2_arg)
    model_param = (Jx1, Jx2, Jy1, Jy2)

    model = HN_model_tightbinding(*model_param)

    r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
    model = model.get_exponentially_scaled_system(r_vec)

    radius = 60.5
    for j in range(2):
        r_list, arg_list, all_points = generate_diamond_or_square_OBC(radius, j)
        H = model.generate_OBC_bulk(all_points)
        eigv, eigvec = la.eig(H.todense())

        with open("%s-%d.pkl"%(fname_prefix, j), "wb") as fp:
            pickle.dump((r_list, arg_list, model_param, all_points, eigv, eigvec, r_vec), fp)


def paper_spectrum_triganles():
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params 
    # Jy2_norm = np.random.randn()
    # Jy2_arg = log(Jx1).imag + log(Jx2).imag - log(Jy1).imag
    # Jy2 = Jy2_norm * exp(1j * Jy2_arg)
    model_param = (Jx1, Jx2, Jy1, Jy2)

    model = HN_model_tightbinding(*model_param)

    r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
    model = model.get_exponentially_scaled_system(r_vec)

    param_set = [
        {
            "arg_list": np.array([0, 1, 2]) * 2 * np.pi / 3,
            "r_list": np.array([30.5, 30.5, 30.5]),
            "suffix": "equilateral"
        },
        {
            "arg_list": np.array([0, 3 * np.pi / 4, 5 * np.pi / 4]),
            "r_list": np.array([30.5, 30.5 / np.sqrt(2), 30.5 / np.sqrt(2)]),
            "suffix": "othogonal-1"
        },
        {
            "arg_list": np.array([0, 3 * np.pi / 4, 3 * np.pi / 2]),
            "r_list": np.array([30.5, 30.5 / np.sqrt(2), 30.5]),
            "suffix": "othogonal-2"
        }
    ]
    for curr_param in param_set:
        r_list, arg_list, all_points = polygon_OBC(
            curr_param["r_list"], curr_param["arg_list"]
        )
        H = model.generate_OBC_bulk(all_points)
        eigv, eigvec = la.eig(H.todense())

        with open("data/OBC-spectrum-triangles-%s.pkl"%(curr_param["suffix"]), "wb") as fp:
            pickle.dump((r_list, arg_list, model_param, all_points, eigv, eigvec, r_vec), fp)


def get_quasi_1D_GBZ(Jx1, Jx2, Jy1, Jy2):
    N = 9
    model = HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2)
    model_2d = model.get_supercell(
        [(0,0), (0,1)],
        np.array([
            [1, -1],
            [1, 1]
        ], dtype=int)
    )
    model_super = model_2d.get_supercell(
        [(0,j) for j in range(N)],
        np.array([
            [1, 0], 
            [0, N]
        ], dtype=int)
    )
    model_super = model_super.truncate(1)
    flag, E, beta = model_super.get_GBZ(N_process=6)
    # coeffs, degs = model_super.get_characteristic_polynomial_data()

    # GBZ_point_list, aGBZ_point_list = pGs.batch_brute_force_solve_GBZ(
    #     coeffs, degs,
    #     N_process = 6, GBZ_checker_tol = 1e-2)

    # plt.plot(E.real, E.imag, '.')
    # plt.show()
    # plt.plot(aGBZ_point_list[:,0].real, aGBZ_point_list[:,0].imag, '.')
    # plt.show()
    with open("data/current-quasi-1D-GBZ.pkl", "wb") as fp:
        # pickle.dump((GBZ_point_list, aGBZ_point_list, (Jx1, Jx2, Jy1, Jy2), N), fp)
        pickle.dump((E, beta, (Jx1, Jx2, Jy1, Jy2), N), fp)


def plot_ellipse_spectrum():
    file_id = 1
    fname_prefix = "data/HN-model-OBC-ellipse-%d"%(file_id)
    for j in range(10):
        with open("%s-%d.pkl"%(fname_prefix, j), "rb") as fp:
            N, model_param, shape_param, all_points, eigv, eigvec, r_vec = pickle.load(fp)
        print(model_param)
        print(np.sum(np.log(model_param[:2]).imag) - np.sum(np.log(model_param[2:]).imag))
        print(shape_param)
        print(len(all_points))
        all_points = np.asarray(all_points)
        plt.plot(eigv.real, eigv.imag, '.')
        plt.show()

        plt.scatter(all_points[:,0],all_points[:,1], c = np.abs(eigvec[:,10]))
        plt.show()


def plot_GBZ_new_program():
    with open("data/current-quasi-1D-GBZ.pkl", "rb") as fp:
        point_list, aGBZ_point_list, params, N = pickle.load(fp)
    Jx1, Jx2, Jy1, Jy2 = params
    k = np.linspace(0, 2 * pi, 100)
    bound = 2 * np.sqrt(2 * np.sqrt(Jx1 * Jx2 * Jy1 * Jy2) * np.cos(k) + Jx1 * Jx2 + Jy1 * Jy2)
    print(params)
    plt.plot(point_list[:,0].real, point_list[:,0].imag,'.')
    plt.plot(bound.real, bound.imag, 'r.')
    plt.plot(-bound.real, -bound.imag, 'r.')
    plt.axis("equal")
    plt.show()
    plt.plot(aGBZ_point_list[:,0].real, aGBZ_point_list[:,0].imag,'.')
    plt.plot(bound.real, bound.imag, 'r.')
    plt.plot(-bound.real, -bound.imag, 'r.')
    plt.axis("equal")
    plt.show()


def plot_polygon_spectrum():
    file_id = 1
    fname_prefix = "data/HN-model-OBC-polygon-%d"%(file_id)
    for j in range(10):
        with open("%s-%d.pkl"%(fname_prefix, j), "rb") as fp:
            r_list, arg_list, model_param, all_points, eigv, eigvec, r_vec = pickle.load(fp)
        print(model_param)
        print(np.sum(np.log(model_param[:2]).imag) - np.sum(np.log(model_param[2:]).imag))
        print(len(all_points))
        all_points = np.asarray(all_points)
        plt.figure(1)
        plt.plot(eigv.real, eigv.imag, '.')

        plt.figure(2)
        plt.scatter(all_points[:,0],all_points[:,1], c = np.abs(eigvec[:,10]))
        plt.show()


def plot_triangles():
    fname_suffix = [
        "equilateral",
        "othogonal-1",
        "othogonal-2"
    ]
    for suffix in fname_suffix:
        with open("data/OBC-spectrum-triangles-%s.pkl"%(suffix), "rb") as fp:
            r_list, arg_list, model_param, all_points, eigv, eigvec, r_vec = pickle.load(fp)
        print(model_param)
        print(np.sum(np.log(model_param[:2]).imag) - np.sum(np.log(model_param[2:]).imag))
        print(len(all_points))
        all_points = np.asarray(all_points)
        plt.figure(1)
        plt.plot(eigv.real, eigv.imag, '.')

        plt.figure(2)
        plt.scatter(all_points[:,0],all_points[:,1], c = np.abs(eigvec[:,10]))
        plt.show()


def plot_diamond_square_spectrum():
    file_id = 65
    fname_prefix = "data/HN-model-OBC-diamond-square-%d"%(file_id)
    for j in range(2):
        with open("%s-%d.pkl"%(fname_prefix, j), "rb") as fp:
            r_list, arg_list, model_param, all_points, eigv, eigvec, r_vec = pickle.load(fp)
        Jx1, Jx2, Jy1, Jy2 = model_param
        # if(j == 0):
        #     get_quasi_1D_GBZ(Jx1, Jx2, Jy1, Jy2)
        # with open("data/current-quasi-1D-GBZ.pkl", "rb") as fp:
        #     E, beta, _ = pickle.load(fp)
        # print(Jx1, Jx2, Jy1, Jy2)
        # k = np.linspace(0, 2 * pi, 100)
        # bound = 2 * np.sqrt(2 * np.sqrt(Jx1 * Jx2 * Jy1 * Jy2) * np.cos(k) + Jx1 * Jx2 + Jy1 * Jy2)
        # plt.plot(E.real, E.imag, '.')
        # plt.plot(bound.real, bound.imag, 'r.')
        # plt.plot(-bound.real, -bound.imag, 'r.')
        # plt.show()
        p_center = 2 * np.sqrt(Jy1 * Jy2)
        all_points = np.asarray(all_points)
        plt.figure(1)
        plt.plot(eigv.real, eigv.imag, '.')
        plt.plot(p_center.real, p_center.imag, 'rx')
        plt.plot(-p_center.real, -p_center.imag, 'rx')

        k = np.linspace(0, 2 * pi, 100)
        bound = 2 * np.sqrt(2 * np.sqrt(Jx1 * Jx2 * Jy1 * Jy2) * np.cos(k) + Jx1 * Jx2 + Jy1 * Jy2)
        plt.plot(bound.real, bound.imag, 'r.')
        plt.plot(-bound.real, -bound.imag, 'r.')
        p_center = 2 * np.sqrt(Jy1 * Jy2 + Jx1 * Jx2)
        plt.plot(p_center.real, p_center.imag, 'gx')

        plt.figure(2)
        plt.scatter(all_points[:,0],all_points[:,1], c = np.abs(eigvec[:,10]))
        plt.axis('equal')
        plt.show()


def plot_HN_BZ():
    kx = np.linspace(-pi, pi, 100)
    ky = np.linspace(-pi, pi, 100)
    Jx1 = 1
    Jx2 = 2
    Jy1 = 1
    Jy2 = 1.5
    Kx, Ky = np.meshgrid(kx, ky)
    E = Jx1 * np.exp(-1j * Kx) + Jx2 * np.exp(1j * Kx) + Jy1 * np.exp(-1j * Ky) + Jy2 * np.exp(1j * Ky)
    plt.plot(E.flatten().real, E.flatten().imag, '.')
    plt.show()


def plot_GBZ_candidates():
    file_id = 62
    j = 0
    fname_prefix = "data/HN-model-OBC-diamond-square-%d"%(file_id)
    with open("%s-%d.pkl"%(fname_prefix, j), "rb") as fp:
        r_list, arg_list, model_param, all_points, eigv, eigvec, r_vec = pickle.load(fp)
    Jx1, Jx2, Jy1, Jy2 = model_param
    Jy2_norm = np.abs(Jy2)
    Jy2_arg = log(Jx1).imag + log(Jx2).imag - log(Jy1).imag
    Jy2 = Jy2_norm * exp(1j * Jy2_arg)
    E = 0.5 * np.sqrt(Jx1 * Jx2)
    model = HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2)
    model_another = model.get_supercell(
        [(0,0)],
        np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )
    coeffs, degs = model_another.get_characteristic_polynomial_data()

    f = pt.CLaurent(3)
    f.set_Laurent_by_terms(pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten()))

    solver = pGs.GBZMainSolver(f)
    err_code = solver.get_beta_GBZ(E)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim([-6,6])
    ax.set_ylim([-5,5])
    for loop in solver.beta_GBZ_glued:
        curr_beta = np.zeros((len(loop.point_vec), 2), dtype=complex)
        for point_id, point in enumerate(loop.point_vec):
            Gm.CP1_to_chart(point,[0,0])
            curr_beta[point_id,:] = point.coords
        ax.plot(curr_beta[:,0].real, curr_beta[:,0].imag)
    ax.set_aspect(1)
    ax.set_xlabel("Re$\\beta_1$")
    ax.set_ylabel("Im$\\beta_1$")
    plt.show()


def paper_HN_sweep_aspect_and_size():
    # boundary: (1,1), (0,1)
    # aspect: 8, 4, 2, 1, 1/2, 1/4, 1/8
    # size: (10, 20, 30) * sqrt(8 / x)

    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params

    model = HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2)
    r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
    model = model.get_exponentially_scaled_system(r_vec)
    model = model.get_supercell(
        [(0,0)], np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )

    for aspect in [8, 4, 2, 1, 1/2, 1/4, 1/8]:
        for N in [20, 30, 40]:
            print(aspect, N)
            N1 = int(np.round(N * np.sqrt(8 / aspect)))
            N2 = int(np.round(N1 * aspect)) 

            model_1d = model.get_supercell(
                [(j,0) for j in range(N1)],
                np.array([
                    [N1, 0],
                    [0, 1]
                ], dtype=int)
            )
            model_2d = model_1d.get_supercell(
                [(0,j) for j in range(N2)],
                np.array([
                    [1, 0],
                    [0, N2]
                ], dtype=int)
            )

            H = model_2d.get_bulk_Hamiltonian_complex((None, None))
            eigv, eigvec = la.eig(H.todense())

            with open("data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl"%(aspect, N), "wb") as fp:
                pickle.dump((
                    eigv, eigvec, model_2d.lattice2cart(np.asarray(model_2d.SiteCoord).transpose()).transpose(),
                    (Jx1, Jx2, Jy1, Jy2),
                    (N1, N2),
                ), fp)


def paper_HN_calculate_OBC():
    # boundary: (1,1), (0,1)

    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params

    model = HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2)
    r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
    model = model.get_exponentially_scaled_system(r_vec)
    model = model.get_supercell(
        [(0,0)], np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )

    N1 = 320
    N2 = 20

    model_1d = model.get_supercell(
        [(j,0) for j in range(N1)],
        np.array([
            [N1, 0],
            [0, 1]
        ], dtype=int)
    )
    model_2d = model_1d.get_supercell(
        [(0,j) for j in range(N2)],
        np.array([
            [1, 0],
            [0, N2]
        ], dtype=int)
    )

    H = model_2d.get_bulk_Hamiltonian_complex((None, None))
    eigv, eigvec = la.eig(H.todense())

    with open("data/paper-HN-OBC-2-N1_%d-N2_%d.pkl"%(N1, N2), "wb") as fp:
        pickle.dump((
            eigv, eigvec, model_2d.lattice2cart(np.asarray(model_2d.SiteCoord).transpose()).transpose(),
            (Jx1, Jx2, Jy1, Jy2),
            (N1, N2)
        ), fp)


def paper_HN_sweep_aspect_and_size_xy():
    # boundary: (1,0), (0,1)
    # aspect: 8, 4, 2, 1, 1/2, 1/4, 1/8
    # size: (10, 20, 30) * sqrt(8 / x)

    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params

    model = HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2)
    r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
    model = model.get_exponentially_scaled_system(r_vec)

    for aspect in [8, 4, 2, 1, 1/2, 1/4, 1/8]:
        for N in [20, 30, 40]:
            print(aspect, N)
            N1 = int(np.round(N * np.sqrt(8 / aspect)))
            N2 = int(np.round(N1 * aspect))

            model_1d = model.get_supercell(
                [(j, 0) for j in range(N1)],
                np.array([
                    [N1, 0],
                    [0, 1]
                ], dtype=int)
            )
            model_2d = model_1d.get_supercell(
                [(0, j) for j in range(N2)],
                np.array([
                    [1, 0],
                    [0, N2]
                ], dtype=int)
            )

            H = model_2d.get_bulk_Hamiltonian_complex((None, None))
            eigv, eigvec = la.eig(H.todense())

            with open("data/paper-HN-OBC-2-xy-aspect_%.6f-N_%d.pkl"%(aspect, N), "wb") as fp:
                pickle.dump((
                    eigv, eigvec, model_2d.lattice2cart(np.asarray(model_2d.SiteCoord).transpose()).transpose(),
                    (Jx1, Jx2, Jy1, Jy2),
                    (N1, N2),
                ), fp)


def paper_add_shape_data_to_data_file():
    for aspect in [1/32, 1/16, 1/8, 1/4, 1/2]:
        for N in [20, 30, 40]:
            N1 = int(np.round(N * np.sqrt(8 / aspect)))
            N2 = int(np.round(N1 * aspect))
            print(N1, N2)

            with open("data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl"%(aspect, N), "rb") as fp:
                all_data = pickle.load(fp)

            if len(all_data) == 6:
                all_data = all_data[:-1]
            elif len(all_data) == 4:
                all_data += ((N1, N2),)

            with open("data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl"%(aspect, N), "wb") as fp:
                pickle.dump(all_data, fp)


def test_paperplot_OBC():

    aspect = 1/16
    N = 20
    with open("data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl"%(aspect, N), "rb") as fp:
        data = pickle.load(fp)
        if len(data) == 6:
            data = data[:-1]
        eigv, eigvec, point_vec, params, shape = data
    fig = plt.figure()
    ax = fig.gca()
    print(point_vec.shape[0])
    ax.plot(point_vec[:,0], point_vec[:,1], '.')
    ax.set_aspect(1)
    plt.show()

    Jx1, Jx2, Jy1, Jy2 = params
    print(params)
    boundary_list = np.array([
        4 * (2 * np.sqrt(Jx1 * Jx2 * Jy1 * Jy2) 
             +Jx1 * Jx2 + Jy1 * Jy2),
        4 * (-2 * np.sqrt(Jx1 * Jx2 * Jy1 * Jy2) 
             +Jx1 * Jx2 + Jy1 * Jy2),
        0
    ], dtype=complex)
    boundary_matrix = np.column_stack(
        [boundary_list.real, boundary_list.imag]
    )
    boundary_loop = np.vstack([boundary_matrix, boundary_matrix[0,:]])
    polygon_obj = shapely.Polygon(boundary_matrix)

    # get_quasi_1D_GBZ(*params)
    # return
    # with open("data/current-quasi-1D-GBZ.pkl", "rb") as fp:
    #     E, beta, _ = pickle.load(fp)
    # plt.plot(E.real, E.imag, '.')
    # plt.show()

    perc_points = []
    for aspect in [1/32, 1/16, 1/8, 1/4, 1/2]:
        for N in [40, 30, 20]:
            # fig = plt.figure()
            # ax = fig.gca()
            # ax.set_title("N = %d"%(N))
            with open("data/paper-HN-OBC-2-aspect_%.6f-N_%d.pkl"%(aspect, N), "rb") as fp:
                data = pickle.load(fp)
                # if len(data) == 6:
                #     data = data[:-1]
                # if len(data) == 4:
                #     eigv, eigvec, point_vec, params = data
                # else:
                eigv, eigvec, point_vec, params, shape = data

            # ax.plot(point_vec[:,0], point_vec[:,1],'.')
            k = np.linspace(0, 2 * pi, 100)
            bound = 2 * np.sqrt(2 * np.sqrt(Jx1 * Jx2 * Jy1 * Jy2) * np.cos(k) + Jx1 * Jx2 + Jy1 * Jy2)
            # ax.plot(eigv.real, eigv.imag, '.')
            # ax.plot(bound.real, bound.imag, 'r.')
            # ax.plot(-bound.real, -bound.imag, 'r.')
            # ax.plot((eigv**2).real, (eigv**2).imag, '.')
            # ax.plot(boundary_loop[:,0], boundary_loop[:,1])

            # count points
            N_interior = 0
            for j in range(len(eigv)):
                if(polygon_obj.contains(shapely.Point([(eigv[j]**2).real, (eigv[j]**2).imag]))):
                    N_interior += 1
            print(aspect, N_interior, len(eigv), N_interior/len(eigv))
            perc_points += [(aspect, N, N_interior/len(eigv))]

    with open("data/percentage-of-points.pkl", "wb") as fp:
        pickle.dump(perc_points, fp)
        # ax.set_aspect(1)
    # plt.show()


def plot_spectrum_percentage():
    with open("data/percentage-of-points.pkl", "rb") as fp:
        perc_points = pickle.load(fp)
    perc_points = np.asarray(perc_points)
    for N in [20, 30, 40]:
        curr_val = perc_points[np.where(np.abs(perc_points[:,1] - N) < 1e-2),:][0]
        print(curr_val)
        plt.plot(1/curr_val[:,0], curr_val[:,2], '.-')
    plt.show()



def check_OBC_strip():
    pass


def paper_quasi_1D_GBZ_x():
    Jx1 = 1 + 1j
    Jx2 = 1.5 + 1.2j
    Jy1 = -1 + 1j
    Jy2 = -1.2 - 0.5j

    N = 15
    model = HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2)

    model_super = model.get_supercell(
        [(0,j) for j in range(N)],
        np.array([
            [1, 0], 
            [0, N]
        ], dtype=int)
    )
    model_super = model_super.truncate(1)
    flag, E, beta = model_super.get_GBZ(N_process=6)

    with open("data/paper-quasi-1D-x-N_%d.pkl"%(N), "wb") as fp:
        pickle.dump((E, beta, (Jx1, Jx2, Jy1, Jy2), N), fp)


def paper_quasi_1D_GBZ_y():
    Jx1 = 1 + 1j
    Jx2 = 1.5 + 1.2j
    Jy1 = -1 + 1j
    Jy2 = -1.2 - 0.5j

    N = 15
    model = HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2)

    model_super = model.get_supercell(
        [(j, 0) for j in range(N)],
        np.array([
            [N, 0], 
            [0, 1]
        ], dtype=int)
    )
    model_super = model_super.truncate(0)
    # flag, E, beta = model_super.get_GBZ(N_process=6)
    coeffs, degs = model_super.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(2)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten())
    )
    GBZ_sols = bfG.solve_GBZ(char_poly, n_processes=12)
    E = GBZ_sols[:,0]
    beta = GBZ_sols[:,1]

    with open("data/paper-quasi-1D-y-N_%d.pkl"%(N), "wb") as fp:
        pickle.dump((E, beta, (Jx1, Jx2, Jy1, Jy2), N), fp)


def paper_quasi_1D_GBZ_11(N: int):
    Jx1 = 1 + 1j
    Jx2 = 1.5 + 1.2j
    Jy1 = -1 + 1j
    Jy2 = -1.2 - 0.5j

    model = HN_model_tightbinding(Jx1, Jx2, Jy1, Jy2)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )

    model_super = model.get_supercell(
        [(0,j) for j in range(N)],
        np.array([
            [1, 0], 
            [0, N]
        ], dtype=int)
    )
    model_super = model_super.truncate(1)
    # flag, E, beta = model_super.get_GBZ(N_process=6)
    coeffs, degs = model_super.get_characteristic_polynomial_data()
    char_poly = pt.CLaurent(2)
    char_poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs), pt.CLaurentIndexVec(degs.flatten())
    )
    GBZ_sols = bfG.solve_GBZ(char_poly, n_processes=12)
    E = GBZ_sols[:,0]
    beta = GBZ_sols[:,1]

    with open("data/paper-quasi-1D-11-N_%d.pkl"%(N), "wb") as fp:
        pickle.dump((E, beta, (Jx1, Jx2, Jy1, Jy2), N), fp)
    
    print("11", N)


def paper_get_quasi_1D_non_Bloch_modes_x():
    ''' Get non-Bloch modes of the quasi-1D model '''
    N = 15
    with open("data/paper-quasi-1D-x-N_%d.pkl"%(N), "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)
    model = HN_model_tightbinding(*params)

    r = np.sqrt(abs(params[2] / params[3]))
    print(r)

    # plt.plot(r_fit_list, '.')
    # plt.show()
    # return

    model_super = model.get_supercell(
        [(0,j) for j in range(N)],
        np.array([
            [1, 0],
            [0, N]
        ], dtype=int)
    )
    model_super = model_super.truncate(1)

    r_fit_list = np.zeros_like(E, dtype=float)
    for point_ind in range(len(E)):
        curr_E = E[point_ind]
        curr_beta = beta[point_ind]
        curr_H = model_super.get_bulk_Hamiltonian_complex((curr_beta,))
        curr_eigv, curr_eigvec = spla.eigs(
            curr_E * sparse.eye(curr_H.shape[0]) - curr_H,
            k=1, which='SM'
        )
        curr_eigvec = curr_eigvec.flatten()
        if abs(curr_eigv[0]) > 1e-4:
            print("Current eigenvalue:", curr_eigv[0])
        # plt.plot(curr_eigvec.real)
        # plt.plot(curr_eigvec.imag)

        fit_popt = []
        residuals = []
        for j in range(1, len(curr_eigvec) + 2):
            try:
                res = optimize.curve_fit(fit_fun, np.arange(len(curr_eigvec)), np.abs(curr_eigvec)**2,
                                         p0=(1, r, j * np.pi/(len(curr_eigvec) + 2), np.pi/2),
                                         full_output=True)
                fit_popt.append(res[0])
                residuals.append(
                    la.norm(res[2]['fvec'])
                )
            except RuntimeError:
                pass

        min_ind = np.argmin(residuals)
        C_fit, r_fit, k_fit, phi_fit = fit_popt[min_ind]
        r_fit_list[point_ind] = r_fit
        print(' ' * 100, end='\r')
        print(point_ind / len(E), r_fit, end='\r')

        if abs(r_fit - r) > 1e-4:
            print(fit_popt)
            print(residuals)
            plt.plot(np.abs(curr_eigvec))
            fit_val = fit_fun(np.arange(len(curr_eigvec)), C_fit, r_fit, k_fit, phi_fit)
            plt.plot(np.sqrt(fit_val), '--')
            plt.show()

    with open("data/paper-quasi-1D-x-N_%d.pkl"%(N), "wb") as fp:
        pickle.dump((E, beta, params, N, r_fit_list), fp)


def paper_get_quasi_1D_non_Bloch_modes_y():
    ''' Get non-Bloch modes of the quasi-1D model '''
    N = 15
    with open("data/paper-quasi-1D-y-N_%d.pkl"%(N), "rb") as fp:
        E, beta, params, N = pickle.load(fp)
    model = HN_model_tightbinding(*params)

    r = np.sqrt(abs(params[0] / params[1]))
    print(r)

    # plt.plot(r_fit_list, '.')
    # plt.show()
    # return

    model_super = model.get_supercell(
        [(j,0) for j in range(N)],
        np.array([
            [N, 0],
            [0, 1]
        ], dtype=int)
    )
    model_super = model_super.truncate(0)

    r_fit_list = np.zeros_like(E, dtype=float)
    for point_ind in range(len(E)):
        curr_E = E[point_ind]
        curr_beta = beta[point_ind]
        curr_H = model_super.get_bulk_Hamiltonian_complex((curr_beta,))
        curr_eigv, curr_eigvec = spla.eigs(
            curr_E * sparse.eye(curr_H.shape[0]) - curr_H,
            k=1, which='SM'
        )
        curr_eigvec = curr_eigvec.flatten()
        if abs(curr_eigv[0]) > 1e-4:
            print("Current eigenvalue:", curr_eigv[0])
        # plt.plot(curr_eigvec.real)
        # plt.plot(curr_eigvec.imag)

        fit_popt = []
        residuals = []
        for j in range(1, len(curr_eigvec) + 2):
            try:
                res = optimize.curve_fit(fit_fun, np.arange(len(curr_eigvec)), np.abs(curr_eigvec)**2,
                                         p0=(1, r, j * np.pi/(len(curr_eigvec) + 2), np.pi/2),
                                         full_output=True)
                fit_popt.append(res[0])
                residuals.append(
                    la.norm(res[2]['fvec'])
                )
            except RuntimeError:
                pass

        min_ind = np.argmin(residuals)
        C_fit, r_fit, k_fit, phi_fit = fit_popt[min_ind]
        r_fit_list[point_ind] = r_fit
        print(' ' * 100, end='\r')
        print(point_ind / len(E), r_fit, residuals[min_ind], end='\r')

        if abs(r_fit - r) > 1e-4:
            print(fit_popt)
            print(residuals)
            plt.plot(np.abs(curr_eigvec))
            fit_val = fit_fun(np.arange(len(curr_eigvec)), C_fit, r_fit, k_fit, phi_fit)
            plt.plot(np.sqrt(fit_val), '--')
            plt.show()

    plt.plot(r_fit_list, '.')
    plt.show()

    with open("data/paper-quasi-1D-y-N_%d.pkl"%(N), "wb") as fp:
        pickle.dump((E, beta, params, N, r_fit_list), fp)


def fit_fun(x, C, r, k, phi):
    val = C * (r ** (2*(x - 8))) * (np.cos(k * x + phi) ** 2)
    return val


def paper_get_quasi_1D_non_Bloch_modes_11():
    ''' Get non-Bloch modes of the quasi-1D model '''
    N = 15
    with open("data/paper-quasi-1D-11-N_%d.pkl"%(N), "rb") as fp:
        E, beta, params, N, r_fit_list = pickle.load(fp)

    Jx1, Jx2, Jy1, Jy2 = params
    plt.plot(np.log(beta).imag, r_fit_list, 'rx')
    plt.plot(np.log(beta).imag, np.sqrt(np.abs(Jx2 * beta + Jy1) / np.abs(Jx1 / beta + Jy2)), 'b.')
    plt.show()
    return

    model = HN_model_tightbinding(*params)
    model = model.get_supercell(
        [(0,0)],
        np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )

    model_super = model.get_supercell(
        [(0,j) for j in range(N)],
        np.array([
            [1, 0],
            [0, N]
        ], dtype=int)
    )
    model_super = model_super.truncate(1)

    r_fit_list = np.zeros_like(E, dtype=float)
    for point_ind in range(len(E)):
        curr_E = E[point_ind]
        curr_beta = beta[point_ind]
        curr_H = model_super.get_bulk_Hamiltonian_complex((curr_beta,))
        curr_eigv, curr_eigvec = spla.eigs(
            curr_E * sparse.eye(curr_H.shape[0]) - curr_H,
            k=1, which='SM'
        )
        curr_eigvec = curr_eigvec.flatten()
        if abs(curr_eigv[0]) > 1e-4:
            print("Current eigenvalue:", curr_eigv[0])
        # plt.plot(curr_eigvec.real)
        # plt.plot(curr_eigvec.imag)

        r = np.sqrt(np.abs(Jx2 * curr_beta + Jy1) / np.abs(Jx1 / curr_beta + Jy2))

        fit_popt = []
        residuals = []
        for j in range(1, len(curr_eigvec) + 2):
            try:
                res = optimize.curve_fit(fit_fun, np.arange(len(curr_eigvec)), np.abs(curr_eigvec)**2,
                                         p0=(1, r, j * np.pi/(len(curr_eigvec) + 2), np.pi/2),
                                         full_output=True)
                fit_popt.append(res[0])
                residuals.append(
                    la.norm(res[2]['fvec'])
                )
            except RuntimeError:
                pass

        if len(residuals) == 0:
            r_fit_list[point_ind] = np.nan
            continue

        min_ind = np.argmin(residuals)
        C_fit, r_fit, k_fit, phi_fit = fit_popt[min_ind]
        r_fit_list[point_ind] = r_fit
        print(' ' * 100, end='\r')
        print(point_ind / len(E), r_fit, residuals[min_ind], end='\r')

        if abs(residuals[min_ind]) > 1e-4:
            print(fit_popt)
            print(residuals)
            plt.plot(np.abs(curr_eigvec))
            fit_val = fit_fun(np.arange(len(curr_eigvec)), C_fit, r_fit, k_fit, phi_fit)
            plt.plot(np.sqrt(fit_val), '--')
            plt.show()

    with open("data/paper-quasi-1D-11-N_%d.pkl"%(N), "wb") as fp:
        pickle.dump((E, beta, params, N, r_fit_list), fp)


def plot_quasi_1D_spectrum():
    which = '11'
    N = 20
    with open("data/paper-quasi-1D-%s-N_%d.pkl" % (which, N), "rb") as fp:
        E, beta, params, N = pickle.load(fp)

    plt.plot(E.real, E.imag, '.')
    plt.show()


if __name__ == '__main__':
    get_beta_GBZ(0, 0)
    # test_ellipse()
    # spectrum_ellipse()
    # plot_ellipse_spectrum()
    # test_polygon()
    # spectrum_polygon()
    # plot_polygon_spectrum()
    # spectrum_diamond_square()
    # plot_diamond_square_spectrum()
    # plot_HN_BZ()
    # plot_GBZ_candidates()
    # paper_HN_sweep_aspect_and_size()
    # test_paperplot_OBC()
    # plot_spectrum_percentage()
    # plot_GBZ_new_program()
    # paper_quasi_1D_GBZ_x()
    # for N in [22, 24, 26, 28]:
    #     paper_quasi_1D_GBZ_11(N)
    #     print("N in [22, 24, 26, 28]")
    # paper_add_shape_data_to_data_file()
    # paper_HN_sweep_aspect_and_size_xy()
    # paper_get_quasi_1D_non_Bloch_modes_x()
    # paper_quasi_1D_GBZ_y()
    # paper_get_quasi_1D_non_Bloch_modes_y()
    # paper_get_quasi_1D_non_Bloch_modes_11()
    # plot_quasi_1D_spectrum()

    # for E_ref in [1+1j, 2+2j, 3+3j]:
    #     paper_get_non_Bloch_sols(E_ref)

    # paper_HN_calculate_OBC()
    # paper_spectrum_triganles()
    # plot_triangles()
