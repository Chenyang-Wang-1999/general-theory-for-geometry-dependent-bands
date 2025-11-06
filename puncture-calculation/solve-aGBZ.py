'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-11-17
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import partial_GBZ_solver as pGs
import GBZ_manifold as Gm
import BerryPy.TightBinding as tb
import poly_tools as pt
import matplotlib.pyplot as plt
import numpy as np

def test_model(t1, t2, gamma1, gamma2):
    ''' Model proposed by https://arxiv.org/pdf/2209.12534 '''
    inter_cell = [
        [0, 0, t1 - gamma1, (1,)],
        [0, 0, t2 - gamma2, (2,)],
        [0, 0, t1 + gamma1, (-1,)], 
        [0, 0, t2 + gamma2, (-2,)]
    ]
    return tb.TightBindingModel(1, 1, [[1]], [], inter_cell)


def test_aGBZ_solution():
    '''
    test the solution of aGBZ
    '''
    print("test the solution of aGBZ")
    t1 = 1
    gamma1 = 0.0
    t2 = 1
    gamma2 = 0.5
    model = test_model(t1, t2, gamma1, gamma2)
    coeffs, degs = model.get_characteristic_polynomial_data()
    poly = pt.CLaurent(2)
    poly.set_Laurent_by_terms(
        pt.CScalarVec(coeffs),
        pt.CLaurentIndexVec(degs.flatten())
    )

    _, params, aGBZ_data = pGs.solve_GBZ(
        'GBZ', poly, glue_tol=0.5, N_points=2000
    )

    # Classify the aGBZ data by the aGBZ index
    classified_data = {}
    for param_ind in range(len(aGBZ_data)):
        full_point_list = aGBZ_data[param_ind]
        for point in full_point_list:
            Gm.GBZ_to_chart(point, [0])
            sol_ind = pGs.get_solution_index(
                poly, point.coords[0], point.coords[1]
            )

            if sol_ind not in classified_data.keys():
                classified_data[sol_ind] = [Gm.GBZPointVec() for _ in range(len(aGBZ_data))]

            classified_data[sol_ind][param_ind].append(point)

    colors = {0: "b", -1: "r", 1: "g"}
    # for sol_ind in classified_data.keys():
    #     for param_ind in range(len(classified_data[sol_ind])):
    #         for point in classified_data[sol_ind][param_ind]:
    #             plt.plot(point.coords[1].real, point.coords[1].imag, colors[sol_ind])
    #             plt.plot((point.coords[1] * np.exp(1j * params[param_ind][0])).real, 
    #                      (point.coords[1] * np.exp(1j * params[param_ind][0])).imag, 
    #                      colors[sol_ind])
    
    # plt.show()
    # return
    
    # Generate GBZ loops for each aGBZ index
    aGBZ_loops = {}
    for sol_ind in classified_data.keys():
        curr_aGBZ, curr_params, all_aGBZ = pGs.expand_solution(
            classified_data[sol_ind], aGBZ_data, params)
        aGBZ_segments = pGs.get_GBZ_segments(
            curr_aGBZ, curr_params, 100,
            pGs.aGBZEquation1DGBZCP1(poly)
        )
        aGBZ_loops[sol_ind] = pGs.batch_glue(pGs.GBZ_segments_to_loops(aGBZ_segments)) 
    
    # plot
    for sol_ind in classified_data.keys():
        for seg in aGBZ_loops[sol_ind]:
            # for seg in loop:
            points = np.zeros((len(seg.point_vec), 2), dtype=complex)
            for point_ind, point in enumerate(seg.point_vec):
                Gm.GBZ_to_chart(point, [0])
                points[point_ind,:] = point.coords
            plt.plot(points[:,1].real, points[:,1].imag, colors[sol_ind], label="%d" % (sol_ind))
    
    # plt.legend()
    plt.show()

if __name__ == '__main__':
    test_aGBZ_solution()
