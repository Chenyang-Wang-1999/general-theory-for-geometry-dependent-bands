'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-04-16
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import unittest
import BerryPy.TightBinding as tb
import partial_GBZ_solver as pGs
import numpy as np

def ssh_model(t1, t2, gamma):
    dim = 1
    site_num = 2
    intra_cell = [
        [0, 1, t1 - gamma/2],
        [1, 0, t1 + gamma/2]
    ]
    inter_cell = [
        [1, 0, t2, (1,)],
        [0, 1, t2, (-1,)]
    ]
    coords = [[0,], [0.5,]]
    return tb.TightBindingModel(
        dim, site_num, [[1]], intra_cell, inter_cell, coords
    )

class TestExponentFitting(unittest.TestCase):
    def test_coordinate_generation(self):
        t1 = 1
        t2 = 0.5
        gamma = 0.2
        model = ssh_model(t1, t2, gamma)
        N = 1000
        model_1d = model.get_supercell(
            [(j,) for j in range(N)],
            [[N]]
        )
        all_coords = [model_1d.lattice2cart(p) for p in model_1d.SiteCoord]

        exponent_fitter = pGs.ExponentsFitting(all_coords, [N], 2)
        exponent_fitter.pick_boundary_coordinates(5)
        print(exponent_fitter.boundary_coord)
        print(exponent_fitter.boundary_coord_id_list)
        print([all_coords[j] for j in exponent_fitter.boundary_coord_id_list])
    
    def test_exponent_fit(self):
        r = np.random.rand()
        t1 = 1
        t2 = 0.5
        gamma = 0.2
        model = ssh_model(t1, t2, gamma)
        N = 1000
        model_1d = model.get_supercell(
            [(j,) for j in range(N)],
            [[N]]
        )
        all_coords = [model_1d.lattice2cart(p) for p in model_1d.SiteCoord]

        psi_mag = np.array([r ** (p[0]) for p in all_coords])

        exponent_fitter = pGs.ExponentsFitting(all_coords, [N], 2)
        exponent_fitter.pick_boundary_coordinates(5)
        res = exponent_fitter.fit_exponents(psi_mag)
        print(r, np.exp(res))