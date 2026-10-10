'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-03-12
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''


import numpy as np
import sys
from scipy import linalg as la
import pickle
sys.path.append("../paper/HN-model")
import HN_model_common as HN


def calculate_OBC_eigenvalues(n1, n2: int):
    Jx1, Jx2, Jy1, Jy2 = HN.Default_Model_Params

    model = HN.get_HN_model(Jx1, Jx2, Jy1, Jy2)
    r_vec = [np.sqrt(np.abs(Jx1/Jx2)), np.sqrt(np.abs(Jy1/Jy2))]
    model = model.get_exponentially_scaled_system(r_vec)
    model = model.get_supercell(
        [(0,0)], np.array([
            [1,0],
            [1,1]
        ], dtype=int)
    )

    model_1d = model.get_supercell(
        [(j,0) for j in range(n1)],
        np.array([
            [n1, 0],
            [0, 1]
        ], dtype=int)
    )
    model_2d = model_1d.get_supercell(
        [(0,j) for j in range(n2)],
        np.array([
            [1, 0],
            [0, n2]
        ], dtype=int)
    )

    H = model_2d.get_bulk_Hamiltonian_complex((None, None))
    eigv, eigvec = la.eig(H.todense())

    with open("data/paper-HN-OBC-2-n1_%d-n2_%d.pkl"%(n1, n2), "wb") as fp:
        pickle.dump((
            eigv, eigvec, model_2d.lattice2cart(np.asarray(model_2d.SiteCoord).transpose()).transpose(),
            (Jx1, Jx2, Jy1, Jy2),
            (n1, n2)
        ), fp)


def sweep_OBC_aspect():
    for n1_over_n2 in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
        totol_num_list = np.linspace(3600, 25600, 8)
        n2_list = np.round(np.sqrt(totol_num_list / n1_over_n2)).astype(int)
        n2_list = np.unique(n2_list)
        print(n2_list)
        print(n2_list * n1_over_n2 * n2_list)
        for n2 in n2_list:
            calculate_OBC_eigenvalues(n1_over_n2 * n2, n2)


if __name__ == "__main__":
    sweep_OBC_aspect()