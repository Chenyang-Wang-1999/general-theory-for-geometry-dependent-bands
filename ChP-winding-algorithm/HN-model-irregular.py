'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-03-20
Copyright © Department of Physics, Tsinghua University. All rights reserved

Calculate OBC spectrum of HN model in irregular geometries
'''

import numpy as np
from scipy import linalg as la
import sys
from shapely.geometry import Point, Polygon
import pickle
import matplotlib.pyplot as plt

sys.path.append("../paper/HN-model")
import HN_model_common as HN


def get_sites_in_polygon(
    polygon: list[tuple[float, float]]  # coordinates of the cusps of the polygon
) -> list[tuple[int, int]]:
    ''' Generate the point set of the integer points inside a polygon 
        Input:
            polygon: list of coordinates of the cusps of the polygon
        Output:
            sites: list of coordinates of the integer points inside the polygon
    '''
    poly_obj = Polygon(polygon)
    xmin = int(np.floor(min([x for x, y in polygon])))
    xmax = int(np.ceil(max([x for x, y in polygon])))
    ymin = int(np.floor(min([y for x, y in polygon])))
    ymax = int(np.ceil(max([y for x, y in polygon])))
    sites = []
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            if poly_obj.contains(Point(x, y)):
                sites.append((x, y))
    return sites


def calculate_OBC_in_polygon(
    polygon: list[tuple[float, float]],
    HN_params: tuple[complex, complex, complex, complex],   # Jx1, Jx2, Jy1, Jy2
    data_fname: str,
    r_vec: list[float, float] = [1, 1]
):
    print(data_fname)
    model = HN.get_HN_model(*HN_params)
    model_scaled = model.get_exponentially_scaled_system(r_vec)
    sites = get_sites_in_polygon(polygon)
    H_mat = model_scaled.generate_OBC_bulk(sites).todense()
    eigv, eigvec = la.eig(H_mat)
    with open("%s.pkl" % data_fname, "wb") as f:
        pickle.dump((eigv, eigvec), f)


def main_diamond(L1: int, L2: int, scaled=False):
    ''' Calculate the OBC spectrum of the diamond lattice
        Input:
            L1: number of unit cells in $[11]$-direction
            L2: number of unit cells in $[1\bar{1}]$-direction
        Output:
            None
    '''
    polygon = [(0, 0), (-L1, L1), (-L1 + L2, L1 + L2), (L2, L2)]
    # sites = get_sites_in_polygon(polygon)
    # sites_arr = np.array(sites)
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    if scaled:
        r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
        name_suffix = "-scaled"
    else:
        r_vec = [1, 1]
        name_suffix = ""

    calculate_OBC_in_polygon(
        polygon,
        HN.Default_Model_Params,
        "data/diamond-OBC-%d-%d%s" % (L1, L2, name_suffix),
        r_vec
    )


def main_trapezoid(L1: int, Ly: int, scaled=False):
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params
    if scaled:
        r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
        name_suffix = "-scaled"
    else:
        r_vec = [1, 1]
        name_suffix = ""

    if L1 > Ly:
        xmid = L1 - Ly
    else:
        xmid = L1
    polygon = [
        (0, 0),
        (0, Ly),
        (xmid, max([Ly, L1])),
        (L1, L1)
    ]
    # sites = get_sites_in_polygon(polygon)
    # sites_arr = np.array(sites)
    # plt.plot(sites_arr[:,0], sites_arr[:,1], '.')
    # plt.axis("equal")
    # plt.show()
    calculate_OBC_in_polygon(
        polygon,
        HN.Default_Model_Params,
        "data/trapezoid-OBC-%d-%d%s" % (L1, Ly, name_suffix)
    )


def visualize_diamond(L1: int, L2: int, name_suffix: str = ""):
    with open("data/diamond-OBC-%d-%d%s.pkl" % (L1, L2, name_suffix), "rb") as f:
        eigv, eigvec = pickle.load(f)
    _, loop = HN.get_11_DGBZ_spectrum(*HN.Default_Model_Params)
    plt.figure()
    plt.title("%d - %d"%(L1, L2))
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(loop.real, loop.imag)

def visualize_trapezoid(L1: int, Ly: int, name_suffix: str = ""):
    with open("data/trapezoid-OBC-%d-%d%s.pkl" % (L1, Ly, name_suffix), "rb") as f:
        eigv, eigvec = pickle.load(f)
    _, loop = HN.get_11_DGBZ_spectrum(*HN.Default_Model_Params)
    plt.figure()
    plt.title("%d - %d"%(L1, Ly))
    plt.plot(eigv.real, eigv.imag, '.')
    plt.plot(loop.real, loop.imag)


if __name__ == "__main__":
    name_suffix = "-scaled"
    visualize_diamond(80, 80, name_suffix)
    visualize_diamond(80, 80)
    plt.show()
    visualize_diamond(40, 160, name_suffix)
    visualize_diamond(40, 160)
    plt.show()
    visualize_diamond(160, 40, name_suffix)
    visualize_diamond(160, 40)
    plt.show()
    visualize_diamond(320, 20, name_suffix)
    visualize_diamond(320, 20)
    plt.show()
    visualize_diamond(20, 320, name_suffix)
    visualize_diamond(20, 320)
    plt.show()
    # main_diamond(80, 80, True)
    # main_diamond(160, 40, True)
    # main_trapezoid(80, 80, True)
    # main_trapezoid(160, 40, True)
    # main_trapezoid(40, 160, True)
    # main_diamond(320, 20, True)
    # main_diamond(40, 160, True)
    # main_diamond(20, 320, True)
    # main_trapezoid(320, 20, True)
    # main_trapezoid(20, 320, True)
    visualize_trapezoid(20, 320, name_suffix)
    visualize_trapezoid(20, 320)
    plt.show()
    visualize_trapezoid(40, 160, name_suffix)
    visualize_trapezoid(40, 160)
    plt.show()
    visualize_trapezoid(80, 80, name_suffix)
    visualize_trapezoid(80, 80)
    plt.show()
    visualize_trapezoid(160, 40, name_suffix)
    visualize_trapezoid(160, 40)
    plt.show()
    visualize_trapezoid(320, 20, name_suffix)
    visualize_trapezoid(320, 20)
    plt.show()
