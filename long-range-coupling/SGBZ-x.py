'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-02-10
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

from model import LongRangeModel, DEFAULT_PARAMS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin, cos, sqrt, pi
from cmath import exp, log
from scipy import ndimage
import networkx as nx
import long_range_GBZ_1D as lrg
import multiprocessing as mp
import pickle
import time
import os
import cProfile
import pstats
import sys

PARAMS = DEFAULT_PARAMS.copy()
PARAMS["tx1"] = PARAMS["tx"] * exp(PARAMS["gammax"])
PARAMS["tx2"] = PARAMS["tx"] * exp(-PARAMS["gammax"])
PARAMS["ty1"] = PARAMS["ty"] * exp(PARAMS["gammay"])
PARAMS["ty2"] = PARAMS["ty"] * exp(-PARAMS["gammay"])
PARAMS.pop("tx")
PARAMS.pop("ty")
PARAMS.pop("gammax")
PARAMS.pop("gammay")


class ErrCode:
    MultipleIntersect: int = -1


def plot_y_winding(mux):
    model = LongRangeModel(**PARAMS)
    thetax = np.linspace(-pi, pi, 101)
    thetay = np.linspace(-pi, pi, 101)
    for curr_thetax in thetax:
        betax = exp(mux + 1j * curr_thetax)
        betay = np.exp(1j * thetay)
        h_values = [complex(model.char_poly(curr_betay, betax)) for curr_betay in betay]
        h_values = np.asarray(h_values)
        plt.plot(h_values.real, h_values.imag)
        plt.title(f"$\\theta_x = {curr_thetax / pi:.2f} \\pi$")
    plt.show()


def construct_edge_graph(curve: np.ndarray[complex, 1]) -> nx.MultiDiGraph:
    # 1. Find self-intersections
    intersects = lrg.find_self_intersection_points(curve)
    if len(intersects) == 0:
        return nx.MultiDiGraph(), [], []

    intersection_nodes = []
    cut_intersect_pair = []
    for node_ind, (ind1, ind2) in enumerate(intersects):
        cut_intersect_pair.append((ind1, node_ind))
        cut_intersect_pair.append((ind2, node_ind))
        s, _, _ = lrg.get_intersection_and_neighborhood(
            (curve[ind1], curve[ind1+1]),
            (curve[ind2], curve[ind2+1]),
        )
        intersection_nodes.append(
            curve[ind1] * (1 - s) + curve[ind1 + 1] * s,
        )

    # 2. Cut curve into edges
    cut_intersect_pair.sort(key=lambda x: x[0])
    all_edges = []
    curr_ind = 0
    multiple_intersection_list = []
    for next_ind, node_ind in cut_intersect_pair:
        if next_ind == curr_ind - 1:
            multiple_intersection_list.append(next_ind)
        else:
            all_edges.append(curve[curr_ind:next_ind+1])

        curr_ind = next_ind + 1

    if multiple_intersection_list:
        return ErrCode.MultipleIntersect, np.unique(multiple_intersection_list), None

    all_edges[0] = np.concatenate((curve[curr_ind:], all_edges[0]))

    G = nx.MultiDiGraph()
    N_edges = len(all_edges)
    for edge_ind in range(N_edges):
        prev_ind = (edge_ind - 1) % N_edges
        G.add_edge(
            cut_intersect_pair[prev_ind][1],
            cut_intersect_pair[edge_ind][1],
            edge_ind
        )
    return G, all_edges, intersection_nodes


def split_graph_to_cycles(
    G: nx.MultiDiGraph,
    all_edges: list[np.ndarray[complex, 1]],
    intersection_nodes: list[complex]
):
    all_cycles = []
    while G.edges:
        cycle = nx.find_cycle(G)
        curr_loop = []
        for node1, node2, edge_ind in cycle:
            curr_loop.extend(
                [[intersection_nodes[node1]], all_edges[edge_ind]]
            )
            G.remove_edge(node1, node2, edge_ind)
        curr_loop.append(
            [intersection_nodes[node2]]
        )
        all_cycles.append(np.concatenate(curr_loop))

    return all_cycles


def calculate_char_poly(
    params: tuple,
    betax: complex,
    betay: complex,
):
    model = LongRangeModel(**params)
    return complex(model.char_poly(betax, betay))


def get_cw_and_ccw(
    col_ind: int,
    h_mesh: np.ndarray[complex, 2],
    thetax: np.ndarray[complex, 1],
    thetay: np.ndarray[complex, 1],
    Nx: int,
    max_points: int,
    nx_insert: int,
    mux: float,
):
    cycle_cw = []
    cycle_ccw = []

    h_values = h_mesh[:, col_ind]

    successful_flag = False
    updated_thetax = thetax
    while not successful_flag:
        G, all_edges, intersection_nodes = construct_edge_graph(h_values)
        if isinstance(G, int):
            successful_flag = False
            if len(h_values) > max_points:
                raise ValueError(f"Failed to find a loop around x axis for thetay = {thetay[col_ind] / pi:.2f} pi with nx = {Nx}")

            multiple_intersection_list = sorted(all_edges)
            print(f"multiple_intersection_list = {multiple_intersection_list}")

            # Insert nx_insert points between each pair of multiple_intersection_list
            new_thetax_list = []
            for thetax_ind in multiple_intersection_list:
                new_thetax_list.append(
                    np.linspace(
                        updated_thetax[thetax_ind],
                        updated_thetax[thetax_ind + 1],
                        nx_insert + 2,
                    )[1:-1]
                )
            new_thetax_list = np.concatenate(new_thetax_list)

            curr_thetay = thetay[col_ind]
            data_pack = [(PARAMS, exp(mux + 1j * tx), exp(1j * curr_thetay)) for tx in new_thetax_list]
            h_list = [calculate_char_poly(*data) for data in data_pack]
            
            new_h_values = []
            curr_ind = 0
            updated_thetax_list = []
            for seg_ind, next_ind in enumerate(multiple_intersection_list):
                new_h_values.append(h_values[curr_ind:next_ind + 1])
                updated_thetax_list.append(updated_thetax[curr_ind:next_ind + 1])
                new_h_values.append(h_list[seg_ind * nx_insert: (seg_ind + 1) * nx_insert])
                updated_thetax_list.append(new_thetax_list[seg_ind * nx_insert: (seg_ind + 1) * nx_insert])
                curr_ind = next_ind + 1
            new_h_values.append(h_values[curr_ind:])
            updated_thetax_list.append(updated_thetax[curr_ind:])
            h_values = np.concatenate(new_h_values)
            updated_thetax = np.concatenate(updated_thetax_list)

        else:
            successful_flag = True

    all_cycles = split_graph_to_cycles(G, all_edges, intersection_nodes)
    for c in all_cycles:
        if lrg.shoelace_area(c) > 0:
            cycle_cw.append(c)
        else:
            cycle_ccw.append(c)
    return cycle_cw, cycle_ccw


def calculate_tot_winding(
    E_ref: complex,
    E_cycles: np.ndarray[complex, 2]
):
    E_diff = E_cycles - E_ref
    E_diff_ratio = E_diff[1:,:] / E_diff[:-1, :]
    E_diff_arg = np.arctan2(E_diff_ratio.imag, E_diff_ratio.real)
    winding_each = np.sum(E_diff_arg, axis=0)
    return int(np.round(np.sum(winding_each) / (2 * np.pi)))


if False:   # Deprecated
    def calculate_tot_winding_new(
        E_ref: complex, 
        E_cycles: np.ndarray[complex, 2]
    ):
        ''' Calculate the total winding number of E_cycles around E_ref 
            Input:
                E_ref: complex, the reference energy
                E_cycles: np.ndarray[complex, 2], each column represents one cycle
            Output:
                tot_winding: int, the total winding number of E_cycles around E_ref
        '''
        E_diff = E_cycles - E_ref
        E_diff_arg = np.arctan2(E_diff.imag, E_diff.real)
        arg_incre = E_diff_arg[1:, :] - E_diff_arg[:-1, :]
        n_jumps = -np.sum(arg_incre > pi, axis=0) + np.sum(arg_incre < -pi, axis=0)
        return sum(n_jumps)


def calculate_tot_winding_vectorized(E_refs, h_mesh):
    """
    向量化版本的缠绕数计算函数，可以同时处理多个E_ref值
    """
    # 将E_refs转换为numpy数组以便向量化操作
    E_refs = np.asarray(E_refs)
    
    # 计算E_diff: 这里假设h_mesh是一个2D复数数组
    # 我们需要为每个E_ref值计算E_diff
    # E_refs形状为(N,)，h_mesh形状为(M, K)
    # 扩展维度以便进行广播
    E_refs_expanded = E_refs[:, np.newaxis, np.newaxis]  # 形状: (N, 1, 1)
    E_diff = h_mesh[np.newaxis, :, :] - E_refs_expanded  # 形状: (N, M, K)
    
    # 计算每个E_ref对应的缠绕数
    # E_diff[1:, :] / E_diff[:-1, :] 相当于沿第1维计算相邻项的比值
    E_diff_upper = E_diff[:, 1:, :]  # shape: (N, M-1, K)
    E_diff_lower = E_diff[:, :-1, :]  # shape: (N, M-1, K)
    
    ratio = E_diff_upper / E_diff_lower  # shape: (N, M-1, K)
    log_imag = np.log(ratio).imag  # shape: (N, M-1, K)
    
    # 沿着第一个空间维度求和得到每列的缠绕数
    winding_each = np.sum(log_imag, axis=1)  # shape: (N, K)
    
    # 沿着第二维求和得到总的缠绕数
    total_winding_raw = np.sum(winding_each, axis=1) / (2 * np.pi)  # shape: (N,)
    
    # 四舍五入并转换为整数
    return np.round(total_winding_raw).astype(int).tolist()



def find_connected_component_vectorized(masks: np.ndarray, start_i: int, start_j: int) -> np.ndarray:
    """
    使用向量化操作找到与给定点连通的相同标签的最大集合
    
    Args:
        masks: 二维整数标签矩阵
        start_i: 起始点行坐标
        start_j: 起始点列坐标
    
    Returns:
        返回一个布尔数组，其中与起始点连通的点标记为True，其他点标记为False
    """
    rows, cols = masks.shape
    
    # 检查起点是否有效
    if start_i < 0 or start_i >= rows or start_j < 0 or start_j >= cols:
        return np.zeros((rows, cols), dtype=bool)
    
    target_label = masks[start_i][start_j]
    
    # 创建仅包含目标标签的二值掩码
    binary_mask = (masks == target_label)
    
    # 定义结构元素（4连通）
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])
    
    # 标记连通组件
    labeled_array, num_features = ndimage.label(binary_mask, structure=structure)
    
    # 获取起始点所在连通组件的标签
    start_component_label = labeled_array[start_i][start_j]
    
    # 如果起始点不在任何组件中（例如在背景中），返回全零数组
    if start_component_label == 0:
        return np.zeros((rows, cols), dtype=bool)
    
    # 返回与起始点属于同一连通组件的位置
    result = (labeled_array == start_component_label)
    
    return result



def sweep_SGBZ_energy(
    mux: float,
    Nx: int = 11,
    Ny: int = 11,
    n_processes: int = mp.cpu_count(),
    n_E_mesh: int = 501,
    n_trunks: int = 2000,
    E_box_margin: float = 0.01,
    debug: bool = False
):
    print(mux)
    # 1. Sweep thetax and thetay
    thetax = np.linspace(-pi, pi, Nx)
    thetay = np.linspace(-pi, pi, Ny)[:-1]
    thetax_mesh, thetay_mesh = np.meshgrid(thetax, thetay, indexing='ij')

    thetax_list = thetax_mesh.flatten()
    thetay_list = thetay_mesh.flatten()

    data_pack = [(PARAMS, exp(mux + 1j * tx), exp(1j * ty)) for tx, ty in zip(thetax_list, thetay_list)]
    # for row_ind in range(Nx):
    #     for col_ind in range(Ny - 1):
    #         betax = exp(mux + 1j * thetax_mesh[row_ind, col_ind])
    #         betay = exp(1j * thetay_mesh[row_ind, col_ind])
    #         h_mesh[row_ind, col_ind] = complex(model.char_poly(betax, betay))

    if debug:
        time1 = time.time()

    if n_processes > 1:
        with mp.Pool(n_processes) as pool:
            h_list = pool.starmap(calculate_char_poly, data_pack)
    else:
        h_list = [calculate_char_poly(*data) for data in data_pack]
    h_mesh = np.reshape(h_list, thetax_mesh.shape)

    if debug:
        with open("data/h_mesh.pkl", "wb") as f:
            pickle.dump(h_mesh, f)
        time2 = time.time()
        print(f"Time cost for char poly calculation: {time2 - time1}")

    # 2. Generate E_mesh
    h_real_range = np.max(h_mesh.real) - np.min(h_mesh.real)
    h_imag_range = np.max(h_mesh.imag) - np.min(h_mesh.imag)
    E_box = (
        np.min(h_mesh.real) - E_box_margin * h_real_range, 
        np.max(h_mesh.real) + E_box_margin * h_real_range, 
        np.min(h_mesh.imag) - E_box_margin * h_imag_range, 
        np.max(h_mesh.imag) + E_box_margin * h_imag_range
    )
    E_mesh_real = np.linspace(E_box[0], E_box[1], n_E_mesh)
    E_mesh_imag = np.linspace(E_box[2], E_box[3], n_E_mesh)
    E_re, E_im = np.meshgrid(E_mesh_real, E_mesh_imag, indexing='ij')
    E_mesh = E_re + 1j * E_im

    # 3. Calculate winding number
    if True:
        E_mesh_flat = E_mesh.flatten()
        data_pack = [(E_ref, h_mesh) for E_ref in E_mesh_flat]
        if n_processes > 1:
            with mp.Pool(n_processes) as pool:
                winding_list = pool.starmap(calculate_tot_winding, data_pack)
        else:
            winding_list = [calculate_tot_winding(E_ref, h_mesh) for E_ref in E_mesh_flat]
        winding_mesh = np.reshape(winding_list, E_mesh.shape)

    else:
        E_mesh_flat = E_mesh.flatten()
        E_mesh_trunk_list = [(E_mesh_flat[i:i+n_trunks], h_mesh) for i in range(0, len(E_mesh_flat), n_trunks)]
        if n_processes > 1:
            with mp.Pool(n_processes) as pool:
                winding_list = pool.starmap(calculate_tot_winding_vectorized, E_mesh_trunk_list)
        else:
            winding_list = [calculate_tot_winding_vectorized(E_trunk, h_trunk) for E_trunk, h_trunk in E_mesh_trunk_list]
        winding_mesh = np.reshape(np.concatenate(winding_list), E_mesh.shape)

    if debug:
        time3 = time.time()
        print(f"Time cost for winding number calculation: {time3 - time2}")

    bg_mask = find_connected_component_vectorized(winding_mesh, 0, 0)

    # 4. Find zeros of winding number
    zero_inds = np.where((winding_mesh == 0) & ~bg_mask)
    E_zeros = E_mesh[zero_inds]

    # 5. Find left-right sign changes
    sign_changes = np.where((winding_mesh[1:,:] * winding_mesh[:-1,:] < 0))
    E_sc1 = (E_mesh[1:,:][sign_changes] + E_mesh[:-1,:][sign_changes]) / 2

    # 6. Find up-down sign changes
    sign_changes = np.where((winding_mesh[:,1:] * winding_mesh[:,:-1] < 0))
    E_sc2 = (E_mesh[:,1:][sign_changes] + E_mesh[:,:-1][sign_changes]) / 2

    if debug:
        time4 = time.time()
        print(f"Time cost for sign change calculation: {time4 - time3}")

    res = np.concatenate([E_zeros, E_sc1, E_sc2])

    return res


def demo_SGBZ_sweep(

):
    Nx = 51
    Ny = 51
    mux = 0.2
    n_processes = mp.cpu_count()
    n_E_mesh = 501
    n_trunks = 2000
    E_box_margin = 0.01
    debug = False

    # 1. Sweep thetax and thetay
    thetax = np.linspace(-pi, pi, Nx)
    thetay = np.linspace(-pi, pi, Ny)[:-1]
    thetax_mesh, thetay_mesh = np.meshgrid(thetax, thetay, indexing='ij')

    thetax_list = thetax_mesh.flatten()
    thetay_list = thetay_mesh.flatten()

    data_pack = [(PARAMS, exp(mux + 1j * tx), exp(1j * ty)) for tx, ty in zip(thetax_list, thetay_list)]
    # for row_ind in range(Nx):
    #     for col_ind in range(Ny - 1):
    #         betax = exp(mux + 1j * thetax_mesh[row_ind, col_ind])
    #         betay = exp(1j * thetay_mesh[row_ind, col_ind])
    #         h_mesh[row_ind, col_ind] = complex(model.char_poly(betax, betay))

    if n_processes > 1:
        with mp.Pool(n_processes) as pool:
            h_list = pool.starmap(calculate_char_poly, data_pack)
    else:
        h_list = [calculate_char_poly(*data) for data in data_pack]
    h_mesh = np.reshape(h_list, thetax_mesh.shape)

    # 2. Generate E_mesh
    h_real_range = np.max(h_mesh.real) - np.min(h_mesh.real)
    h_imag_range = np.max(h_mesh.imag) - np.min(h_mesh.imag)
    E_box = (
        np.min(h_mesh.real) - E_box_margin * h_real_range, 
        np.max(h_mesh.real) + E_box_margin * h_real_range, 
        np.min(h_mesh.imag) - E_box_margin * h_imag_range, 
        np.max(h_mesh.imag) + E_box_margin * h_imag_range
    )
    E_mesh_real = np.linspace(E_box[0], E_box[1], n_E_mesh)
    E_mesh_imag = np.linspace(E_box[2], E_box[3], n_E_mesh)
    E_re, E_im = np.meshgrid(E_mesh_real, E_mesh_imag, indexing='ij')
    E_mesh = E_re + 1j * E_im

    # 3. Calculate winding number
    E_mesh_flat = E_mesh.flatten()
    data_pack = [(E_ref, h_mesh) for E_ref in E_mesh_flat]
    if n_processes > 1:
        with mp.Pool(n_processes) as pool:
            winding_list = pool.starmap(calculate_tot_winding, data_pack)
    else:
        winding_list = [calculate_tot_winding(E_ref, h_mesh) for E_ref in E_mesh_flat]
    winding_mesh = np.reshape(winding_list, E_mesh.shape)

    print("Plotting ...")
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制初始的winding number点
    ax.plot(E_mesh[winding_mesh == 0].real, E_mesh[winding_mesh == 0].imag, 'k.', markeredgecolor="none", label='winding=0')
    ax.plot(E_mesh[winding_mesh > 0].real, E_mesh[winding_mesh > 0].imag, 'r.', markeredgecolor="none", label='winding>0')
    ax.plot(E_mesh[winding_mesh < 0].real, E_mesh[winding_mesh < 0].imag, 'b.', markeredgecolor="none", label='winding<0')
    
    # 设置图形属性
    ax.set_xlabel('Re(E)')
    ax.set_ylabel('Im(E)')
    ax.set_title(f'SGBZ Sweep Demo (μx={mux})')
    ax.legend()
    
    import tempfile
    import os
    from PIL import Image
    
    # 创建临时目录保存帧图片
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    print("开始生成动画帧...")
    
    # 保存初始帧（只有winding number点）
    frame_path = os.path.join(temp_dir, f"frame_000.png")
    fig.savefig(frame_path, dpi=100, bbox_inches='tight')
    frame_files.append(frame_path)
    print(f"已保存帧 0/{Ny-1}")
    
    # 绘制h_mesh曲线并保存每一帧
    lines = []
    for col_ind in range(Ny - 1):
        # 绘制新曲线
        line, = ax.plot(h_mesh[:, col_ind].real, h_mesh[:, col_ind].imag, '-', color='green', alpha=0.7)
        lines.append(line)
        
        # 保存当前帧为图片
        frame_path = os.path.join(temp_dir, f"frame_{col_ind+1:03d}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        frame_files.append(frame_path)
        
        print(f"已保存帧 {col_ind+1}/{Ny-1}")
    
    print("所有帧已保存，正在合成GIF...")
    
    # 使用PIL合成GIF
    gif_filename = f"demo_SGBZ_sweep_mux_{mux}.gif"
    
    # 读取第一帧获取尺寸
    first_frame = Image.open(frame_files[0])
    
    # 创建GIF（优化设置）
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        # 转换为P模式（调色板模式）减少文件大小
        if img.mode != 'P':
            img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
        frames.append(img)
    
    # 保存GIF
    frames[0].save(
        gif_filename,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=300,  # 每帧300毫秒
        loop=0,        # 无限循环
        optimize=True  # 优化文件大小
    )
    
    # 清理临时文件
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)
    
    print(f"GIF保存完成: {gif_filename} (共{len(frame_files)}帧)")


def batch_sweep_SGBZ(
    mu_list: np.ndarray[float, 1],
    nx: int = 101,
    ny: int = 101
):
    result_list = []
    for mux in mu_list:
        try:
            SGBZ_points = sweep_SGBZ_energy(mux, Nx=nx, Ny=ny)
            success = True
            result_list.append(
                {
                    'mux': mux,
                    'success': success,
                    'SGBZ_points': SGBZ_points
                }
            )
        except Exception as e:
            success = False
            result_list.append(
                {
                    'mux': mux,
                    'success': success,
                    "info": str(e),
                }
            )
    
    SGBZ_mu_list = [item['mux'] for item in result_list if (item['success'] and len(item['SGBZ_points']) > 0)]
    print("Find SGBZ for mu:", SGBZ_mu_list)
    mu_failed = [item['mux'] for item in result_list if not item['success']]
    print("Failed for mu:", mu_failed)
    fname_prefix = "data/SGBZ-x"
    fid = 0
    while os.path.exists(fname_prefix + f"{fid:03d}.pkl"):
        fid += 1
    with open(fname_prefix + f"{fid:03d}.pkl", "wb") as f:
        pickle.dump(result_list, f)


def test_edge_graph():
    model = LongRangeModel(**PARAMS)
    mux = 0.1
    thetax = np.linspace(-pi, pi, 101)
    thetay = 0
    betax_list = np.exp(mux + 1j * thetax)
    betay = exp(1j * thetay)
    h_values = [complex(model.char_poly(betax, betay)) for betax in betax_list]
    h_values = np.asarray(h_values)

    # h_values = np.concatenate([
    #     np.linspace(-1 + 1j, -1 - 1j, 11, endpoint=False),
    #     np.linspace(-1 - 1j, 2 + 2j, 11, endpoint=False),
    #     np.linspace(2 + 2j, 1 + 2j, 11, endpoint=False),
    #     np.linspace(1 + 2j, 1 - 2j, 11, endpoint=False),
    #     np.linspace(1 - 2j, 2 - 2j, 11, endpoint=False),
    #     np.linspace(2 - 2j, -1 + 1j, 11, endpoint=False),
    # ])

    G, all_edges, intersection_nodes = construct_edge_graph(h_values)
    # for node1, node2, edge_ind in G.edges:
    #     plt.plot(h_values.real, h_values.imag)
    #     plt.plot(all_edges[edge_ind].real, all_edges[edge_ind].imag)
    #     plt.plot(all_edges[edge_ind][0].real, all_edges[edge_ind][0].imag, 'x')
    #     plt.plot(all_edges[edge_ind][-1].real, all_edges[edge_ind][-1].imag, '+')
    #     plt.plot(intersection_nodes[node1].real, intersection_nodes[node1].imag, 'x')
    #     plt.plot(intersection_nodes[node2].real, intersection_nodes[node2].imag, '+')
    #     plt.show()

    all_cycles = split_graph_to_cycles(G, all_edges, intersection_nodes)
    plt.plot(h_values.real, h_values.imag)
    for cycle in all_cycles:
        plt.plot(cycle.real, cycle.imag)
    plt.show()
 

def test_SGBZ_energy():
    mux = 0.1
    SGBZ_cell, cell_areas = sweep_SGBZ_energy(mux, Nx=101, Ny=101)
    print(SGBZ_cell)
    print(cell_areas)
    plt.plot([p.real for p in SGBZ_cell], [p.imag for p in SGBZ_cell], 'x')
    plt.show()


def main_sweep_SGBZ():
    mu_list = np.linspace(0.0, 0.26, 201)
    n_batch = 10
    for i in range(0, len(mu_list), n_batch):
        # batch_sweep_SGBZ(mu_list[i::n_batch], 301, 301)
        print(f"Batch {i//n_batch}: {mu_list[i:i+n_batch]}")
        batch_sweep_SGBZ(mu_list[i:i+n_batch], 201, 201)


def check_mux_err():
    # with open("data/mux-failed.pkl", "rb") as f:
    #     mux_failed = pickle.load(f)
    
    SGBZ_points = sweep_SGBZ_energy(0.1, Nx=201, Ny=201)



def plot_SGBZ_energy(fid_start: int, fid_end: int):
    results = []
    for fid in range(fid_start, fid_end + 1):
        with open(f"data/SGBZ-x{fid:03d}.pkl", "rb") as f:
            results.extend(pickle.load(f))

    err_list = []
    E_list = []
    for r in results:
        if r["success"]:
            E_list.extend(r["SGBZ_points"])
        else:
            err_list.append(r["mux"])

    E_list = np.array(E_list)
    print(err_list)
    plt.plot(E_list.real, E_list.imag, '.', markeredgecolor="none")


def plot_OBC_spectrum(nx: int, ny: int, type=""):
    with open(f"data/OBC-default{type}-nx{nx}-ny{ny}.pkl", "rb") as f:
        params, coord_list, eigv, eigvec = pickle.load(f)
    plt.plot(eigv.real, eigv.imag, '.')


def check_SGBZ_success(fid_start: int, fid_end: int):
    results = []
    for fid in range(fid_start, fid_end + 1):
        with open(f"data/SGBZ-x{fid:03d}.pkl", "rb") as f:
            results.extend(pickle.load(f))

    mux_list = [r["mux"] for r in results]
    success_list = [r["success"] for r in results]
    plt.plot(mux_list, success_list, '.')
    mux_failed = [mux_list[i] for i in range(len(results)) if not success_list[i]]
    print(mux_failed)
    with open("data/mux-failed.pkl", "wb") as f:
        pickle.dump(mux_failed, f)

def profile_winding_number(which="new"):
    with open("data/h_mesh.pkl", "rb") as f:
        h_mesh = pickle.load(f)
    print(h_mesh.shape)
    h_re_min = np.min(h_mesh.real)
    h_re_max = np.max(h_mesh.real)
    h_im_min = np.min(h_mesh.imag)
    h_im_max = np.max(h_mesh.imag)
    if which == "new":
        fun_call = calculate_tot_winding
    else:
        fun_call = calculate_tot_winding_old

    for _ in range(10000):
        E_ref = np.random.uniform(h_re_min, h_re_max) + 1j * np.random.uniform(h_im_min, h_im_max)
        w = fun_call(E_ref, h_mesh)


def check_SGBZ_x_data():
    file_ind = 0
    with open(f"data/SGBZ-x{file_ind:03d}.pkl", "rb") as f:
        results = pickle.load(f)
    for res in results:
        print(res)


if __name__ == "__main__":
    # plot_y_winding(0)
    # test_edge_graph()
    # test_SGBZ_energy()
    # main_sweep_SGBZ()
    plot_SGBZ_energy(0, 21)
    plot_OBC_spectrum(300, 20, type="")
    # plot_OBC_spectrum(50, 50, type="")
    # plt.show()

    # check_SGBZ_success(1, 1)

    # check_mux_err()
    # args = sys.argv[1:]
    # profile_winding_number(args[0])    # run with python3 -m cProfile -o winding_number.prof SGBZ-x.py new
    # test_tot_winding_new_algorithm()
    # check_SGBZ_x_data()
    plt.show()
    # demo_SGBZ_sweep()
