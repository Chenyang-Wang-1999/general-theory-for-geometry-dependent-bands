'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-03-13
Copyright © Department of Physics, Tsinghua University. All rights reserved

'''
import shapely
import pickle
import numpy as np
import csv
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt


def check_point(
    point: np.ndarray,
    polygon_obj: shapely.Polygon
):
    return polygon_obj.contains(shapely.Point([point.real, point.imag]))


def calculate_DOS(eigv: np.ndarray, real_grids: np.ndarray, imag_grids: np.ndarray):
    ''' Indexing: ij '''
    point_num = np.zeros((real_grids.shape[0] - 1, imag_grids.shape[0] - 1))
    for row_ind in range(real_grids.shape[0] - 1):
        for col_ind in range(imag_grids.shape[0] - 1):
            point_num[row_ind, col_ind] = np.sum(
                (eigv.real >= real_grids[row_ind]) & 
                (eigv.real < real_grids[row_ind + 1]) &
                (eigv.imag >= imag_grids[col_ind]) & 
                (eigv.imag < imag_grids[col_ind + 1]))
    return point_num


def run_DOS_calculation(n1, n2, E_real_grid, E_imag_grid):
    print(n1, n2)
    filename = "data/paper-HN-OBC-2-n1_%d-n2_%d.pkl"%(n1, n2)
    with open(filename, "rb") as f:
        eigv, _, _, _, _ = pickle.load(f)
    state_count = calculate_DOS(eigv, E_real_grid, E_imag_grid)
    result_fname = "data/paper-HN-OBC-2-DOS-n1-%d-n2-%d.pkl"%(n1, n2)
    with open(result_fname, "wb") as f:
        pickle.dump((state_count, E_real_grid, E_imag_grid), f)


def batch_calculate_DOS():
    ''' Sweeping data folder, extract n1 and n2 by the filename, and calculate the DOS '''
    data_folder = "data"
    n_processing = 12

    E_real_grid = np.linspace(-6, 6, 200)
    E_imag_grid = np.linspace(-3, 3, 200)
    task_list = []
    for filename in os.listdir(data_folder):
        if filename.startswith("paper-HN-OBC-2-n1") and filename.endswith(".pkl"):
            filename = filename[:-4]
            n1_str, n2_str = filename.split("-")[-2:]
            if n1_str[:3] != "n1_":
                continue
            if n2_str[:3] != "n2_":
                continue

            n1, n2 = map(int, [n1_str[3:], n2_str[3:]])
            result_fname = "data/paper-HN-OBC-2-DOS-n1-%d-n2-%d.pkl"%(n1, n2)
            if os.path.exists(result_fname):
                print("Found %s, skip"%result_fname)
                continue

            task_list.append((n1, n2, E_real_grid, E_imag_grid))
    
    for n1, n2, E_real_grid, E_imag_grid in task_list:
        run_DOS_calculation(n1, n2, E_real_grid, E_imag_grid)
    # with Pool(n_processing) as p:
    #     p.starmap(run_DOS_calculation, task_list)


def plot_DOS(n1, n2):
    result_fname = "data/paper-HN-OBC-2-DOS-n1-%d-n2-%d.pkl"%(n1, n2)
    with open(result_fname, "rb") as f:
        state_count, E_real_grid, E_imag_grid = pickle.load(f)
    E_area = (E_real_grid[1] - E_real_grid[0]) * (E_imag_grid[1] - E_imag_grid[0])
    pcl = plt.pcolormesh(
        E_real_grid, E_imag_grid, state_count.T / E_area,
        vmin=0, vmax=n1*n2/500/E_area)
    plt.colorbar(pcl)
    plt.show()


if __name__ == "__main__":
    batch_calculate_DOS()
    # plot_DOS(640, 40)
