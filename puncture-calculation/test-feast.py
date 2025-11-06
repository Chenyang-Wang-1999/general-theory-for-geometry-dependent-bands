'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-04-12
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import poly_tools as pt
import partial_GBZ_solver as pGs
import GBZ_manifold as Gm
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse

import feast


def test_feast():
    dtype = complex
    
    A = np.random.randn(100,100) + 1j * np.random.randn(100,100)
    A = sparse.csr_matrix(A)
    solver = feast.GSolver(A, M0 = 100, Emid=0.0, r=10)
    eigv, eigvec, M, info = solver.eigs()
    print(info)
    print(eigv)
    print(M)

if __name__ == '__main__':
    test_feast()
