'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-02-09
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''
import numpy as np
import shapely.geometry as geom
from shapely.geometry import LineString
from scipy import linalg as la
from math import sin, cos, sqrt, pi
from cmath import exp, log
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree
import multiprocessing as mp


def shoelace_area(polygon: np.ndarray[complex, 1]):
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    Parameters:
    - polygon: np.ndarray[complex, 1], vertices of the polygon (first and last elements should be equal)
    
    Returns:
    - float, area of the polygon
    """
    n = len(polygon)
    if n < 3:
        return 0.0  # A polygon must have at least 3 vertices
    
    if abs(polygon[0] - polygon[-1]) > 1e-10:
        polygon = np.append(polygon, polygon[0])
    
    area = np.sum(polygon[:-1].real * polygon[1:].imag - polygon[1:].real * polygon[:-1].imag)
    return area / 2.0


def find_self_intersection_points(h_values):
    """
    Find self-intersection points of h(z) where z traces a closed loop in the complex plane.
    
    Parameters:
    - h_values: np.ndarray[complex, 1], a closed loop in the complex plane (first and last elements should be equal)
    
    Returns:
    - list of tuples (i, j), where (h(z[i]), h(z[i+1])) intersects (h(z[j]), h(z[j+1]))
    """
    
    # Create line segments and check for intersections
    intersect_segments = []
    
    for i in range(len(h_values) - 1):
        for j in range(i + 2, len(h_values) - 1):  # Skip adjacent segments

            # Create line segments from complex coordinates
            seg1 = LineString([
                (h_values[i].real, h_values[i].imag),
                (h_values[i+1].real, h_values[i+1].imag)
            ])

            seg2 = LineString([
                (h_values[j].real, h_values[j].imag),
                (h_values[j+1].real, h_values[j+1].imag)
            ])

            # Check for intersection
            if seg1.intersects(seg2):
                intersect_segments.append((i, j))

    return intersect_segments


def get_intersection_and_neighborhood(
    seg1: tuple[complex, complex],
    seg2: tuple[complex, complex]
):
    # 1. Find the intersection point:
    # p_sect = seg1[0] * (1-s) + seg1[1] * s = seg2[0] * (1 - t) + seg2[1] * t
    # solve s and t
    #  - s * seg1[0] + s * seg1[1] + t * seg2[0] - t * seg2[1] = seg2[0] - seg1[0]

    s_coeff = seg1[1] - seg1[0]
    t_coeff = seg2[0] - seg2[1]
    rhs = seg2[0] - seg1[0]
    s, t = la.solve(
        np.array([
            [s_coeff.real, t_coeff.real],
            [s_coeff.imag, t_coeff.imag]
        ]),
        np.array([rhs.real, rhs.imag])
    )

    # 2. Calculate the interior points
    vec1 = seg1[1] - seg1[0]
    vec2 = seg2[1] - seg2[0]
    vec1 /= abs(vec1)
    vec2 /= abs(vec2)
    dp = vec2 - vec1
    return s, t, dp


def get_winding_number(val_loop: np.ndarray[complex]):
    """
    Calculate the winding number of a loop in the complex plane.
    
    Parameters:
    - val_loop: np.ndarray[complex, 1], a loop in the complex plane (first and last elements should be equal)
    
    Returns:
    - int, the winding number of the loop
    """
    exp_dphi = val_loop[1:] / val_loop[:-1]
    return np.sum(np.log(exp_dphi).imag) / (2 * pi)


def sweep_GBZ_points_for_mu(
    h: callable, # complex -> complex
    mu: float,
    N_points: int = 501,
    winding_distance: float = 1e-3
):
    theta_list = np.linspace(-pi, np.pi, N_points)
    h_values = np.array([
        h(exp(mu + 1j * theta)) for theta in theta_list
    ])
    intersect_segments = find_self_intersection_points(h_values)
    GBZ_sols = []
    for ind1, ind2 in intersect_segments:
        s, t, dp = get_intersection_and_neighborhood(
            (h_values[ind1], h_values[ind1+1]),
            (h_values[ind2], h_values[ind2+1])
        )
        E_val = h_values[ind1] * (1 - s) + h_values[ind1+1] * s
        w1 = int(np.round(get_winding_number(h_values - (E_val + winding_distance * dp))))
        w2 = int(np.round(get_winding_number(h_values - (E_val - winding_distance * dp))))
        if max(w1, w2) > 0 and min(w1, w2) < 0:
            GBZ_sols.append((
                E_val,
                mu,
                theta_list[ind1] * (1 - s) + theta_list[ind1+1] * s,
                theta_list[ind2] * (1 - t) + theta_list[ind2+1] * t
            ))
    if len(GBZ_sols) > 0:
        return GBZ_sols, 0
    else:
        E0 = h_values[0]
        dE = h_values[1] - E0
        dE /= abs(dE) * 1j
        w1 = int(np.round(get_winding_number(h_values - (E0 + winding_distance * dE))))
        w2 = int(np.round(get_winding_number(h_values - (E0 - winding_distance * dE))))
        return GBZ_sols, (w1, w2)


def main_sweep_GBZ_points(
    h: callable, # complex -> complex
    mu_list: np.ndarray[float, 1],
    N_points: int = 501,
    winding_distance: float = 1e-3,
    n_processes: int = 1
):
    GBZ_sols = []
    with mp.Pool(n_processes) as pool:
        results = pool.starmap(
            sweep_GBZ_points_for_mu,
            [(h, mu, N_points, winding_distance) for mu in mu_list]
        )
    for sols, winding_num in results:
        GBZ_sols.extend(sols)
    return GBZ_sols


def complex_to_polygon(complex_array: np.ndarray) -> geom.Polygon:
    """复数数组转Shapely多边形"""
    if not np.isclose(complex_array[0], complex_array[-1]):
        complex_array = np.append(complex_array, complex_array[0])
    coords = [(z.real, z.imag) for z in complex_array]
    return geom.Polygon(coords)


def polygon_to_complex(polygon: geom.Polygon) -> np.ndarray:
    """Shapely多边形转复数数组"""
    if polygon.is_empty:
        return np.array([], dtype=complex)
    exterior_coords = list(polygon.exterior.coords)[:-1]
    return np.array([complex(x, y) for x, y in exterior_coords])


def find_balanced_regions_complex(
    polygons_cw: list[np.ndarray],
    polygons_ccw: list[np.ndarray]
) -> list[geom.Polygon]:
    """
    找到被相同数量顺时针和逆时针多边形包围的区域
    
    Args:
        polygons_cw: 顺时针多边形的复数顶点数组列表
        polygons_ccw: 逆时针多边形的复数顶点数组列表
        
    Returns:
        平衡区域的复数顶点数组列表
    """
    # 转换为Shapely多边形
    shapely_cw = [complex_to_polygon(arr) for arr in polygons_cw]
    shapely_ccw = [complex_to_polygon(arr) for arr in polygons_ccw]
    
    # 收集所有边界（外环和内环）
    all_boundaries = []
    for poly in shapely_cw + shapely_ccw:
        all_boundaries.append(poly.exterior)
        all_boundaries.extend(poly.interiors)
    
    if not all_boundaries:
        return []
    
    # 创建切割网络并生成最小单元格
    cutting_network = unary_union(all_boundaries)
    cells = list(polygonize([cutting_network]))
    
    # 构建空间索引
    tree_cw = STRtree(shapely_cw)
    tree_ccw = STRtree(shapely_ccw)
    
    # 统计每个单元格
    balanced_cells = []
    for cell in cells:
        point = cell.representative_point()  # 使用代表点（保证在内部）
        
        # 查询可能包含该点的多边形
        possible_cw = tree_cw.query(point)
        possible_ccw = tree_ccw.query(point)

        if len(possible_cw) == 0 or len(possible_ccw) == 0:
            continue
        
        # 精确计数
        cw_count = sum(1 for poly in tree_cw.geometries.take(possible_cw) if poly.contains(point))
        ccw_count = sum(1 for poly in tree_ccw.geometries.take(possible_ccw) if poly.contains(point))
        
        # 检查条件：相同数量且大于0
        if cw_count == ccw_count > 0:
            balanced_cells.append(cell)
    
    return balanced_cells