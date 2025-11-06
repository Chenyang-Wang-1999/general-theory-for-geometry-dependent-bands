'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-03-17
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' Improve the algorithm for aGBZ intersection '''

import poly_tools as pt
import numpy as np
import partial_GBZ_solver as pGs
import matplotlib.pyplot as plt

def test_char_poly():
    ''' characteristic polynomial for algorithm check '''
    coeffs = pt.CScalarVec([
        1, -1, -1, -1, -1
    ])
    degs = pt.CLaurentIndexVec([
        # E, beta1, beta2
        1, 0, 0,
        0, 1, 0,
        0, -1, 0,
        0, 1, -1,
        0, -2, 1
    ])
    f = pt.CLaurent(3)
    f.set_Laurent_by_terms(coeffs, degs)
    return f

def get_aGBZ():
    E = 3
    f = test_char_poly()
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([2,1])
    )
    beta_loop, phi_loop = pGs.solve_aGBZ(f_partial, N_points=500)

    print(len(beta_loop))

    beta_minor_list = np.zeros(
        (len(beta_loop), 3), dtype=complex
    )

    for j, loop in enumerate(beta_loop):
        if(len(loop) < 3):
            continue
        all_colors = ['r.', 'g.', 'b.']
        for k in range(3):
            if(loop[k].chart_labels[0] == 0):
                plt.plot(loop[k].coords[1].real, loop[k].coords[1].imag, all_colors[k])
                plt.plot((phi_loop[j,0] * loop[k].coords[1]).real, (phi_loop[j,0] * loop[k].coords[1]).imag, all_colors[k])
            else:
                plt.plot((1/loop[k].coords[1]).real, (1/loop[k].coords[1]).imag, all_colors[k])
                plt.plot((phi_loop[j,0]/loop[k].coords[1]).real, (phi_loop[j,0]/loop[k].coords[1]).imag, all_colors[k])
        plt.pause(0.5)
    # for j, loop in enumerate(beta_loop):
    #     if(len(loop) < 3):
    #         continue
    #     for k in range(3):
    #         if(loop[k].chart_labels[0] == 0):
    #             beta_minor_list[j,k] = loop[k].coords[1]
    #         else:
    #             beta_minor_list[j,k] = 1/(loop[k].coords[1])

    # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # ax = fig.gca()
    # ax.plot(beta_minor_list.real, beta_minor_list.imag, '.-')
    plt.show()

def test_aGBZ():
    E = 3
    f = test_char_poly()
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([2,1])
    )
    beta_loop, phi_loop = pGs.solve_aGBZ("GBZ", f_partial, N_points=200)


    GBZ_loops = pGs.pick_GBZ(f_partial, beta_loop, 1e-4)

    N_GBZ_points = np.zeros((len(phi_loop)), dtype=int)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--')
    ax2.plot(np.cos(theta), np.sin(theta), 'k--')
    # ax1.set_aspect(1)
    # ax2.set_aspect(1)

    point_colors = ['r.', 'g.', 'b.']
    for j, point_vec in enumerate(GBZ_loops):
        ax1.set_title("%s pi"%(np.log(phi_loop[j,0]).imag / np.pi))
        for k, point in enumerate(point_vec):
            if(point.chart_labels[0] == 0):
                ax1.plot(
                    point.coords[1].real, 
                    point.coords[1].imag, 
                    abs(point.coords[0]),
                    point_colors[k])
                ax1.plot(
                    (point.coords[1]*phi_loop[j,0]).real, 
                    (point.coords[1]*phi_loop[j,0]).imag, 
                    abs(point.coords[0]),
                    point_colors[k])
            else:
                ax2.plot(
                    point.coords[1].real, 
                    point.coords[1].imag, 
                    abs(point.coords[0]),
                    point_colors[k])
                ax2.plot(
                    (point.coords[1]/phi_loop[j,0]).real, 
                    (point.coords[1]/phi_loop[j,0]).imag, 
                    abs(point.coords[0]),
                    point_colors[k])
        plt.pause(0.1)
                
    plt.show()

def check_CP1_calculation():
    E = 3
    f = test_char_poly()
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([2,1])
    )

    beta_loop, phi_loop = pGs.solve_aGBZ("GBZ", f_partial, N_points=200)
    GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DGBZCP1(f_partial), beta_loop, 1e-4, 'GBZ')

    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--')
    ax2.plot(np.cos(theta), np.sin(theta), 'k--')
    # ax1.set_aspect(1)
    # ax2.set_aspect(1)

    point_colors = ['r.', 'g.', 'b.']
    for j, point_vec in enumerate(GBZ_loops):
        ax1.set_title("%s pi"%(np.log(phi_loop[j,0]).imag / np.pi))
        for k, point in enumerate(point_vec):
            if(point.chart_labels[0] == 0):
                ax1.plot(
                    point.coords[1].real, 
                    point.coords[1].imag, 
                    abs(point.coords[0]),
                    point_colors[k])
                ax1.plot(
                    (point.coords[1]*phi_loop[j,0]).real, 
                    (point.coords[1]*phi_loop[j,0]).imag, 
                    abs(point.coords[0]),
                    point_colors[k])
            else:
                ax2.plot(
                    point.coords[1].real, 
                    point.coords[1].imag, 
                    abs(point.coords[0]),
                    point_colors[k])
                ax2.plot(
                    (point.coords[1]/phi_loop[j,0]).real, 
                    (point.coords[1]/phi_loop[j,0]).imag, 
                    abs(point.coords[0]),
                    point_colors[k])
        # plt.pause(0.1)
                
    beta_loop, phi_loop = pGs.solve_aGBZ("CP1", f_partial, N_points=200)
    GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DCP1(f_partial), beta_loop, 1e-4, 'CP1')

    fig = plt.figure(2)
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--')
    ax2.plot(np.cos(theta), np.sin(theta), 'k--')
    # ax1.set_aspect(1)
    # ax2.set_aspect(1)

    point_colors = ['r.', 'g.', 'b.']
    for j, point_vec in enumerate(GBZ_loops):
        ax1.set_title("%s pi"%(np.log(phi_loop[j,0]).imag / np.pi))
        for k, point in enumerate(point_vec):
            if(point.chart_labels[0] == 0):
                beta_major = point.coords[0]
            else:
                # beta_major = 1/point.coords[0]
                beta_major = point.coords[0]
            if(point.chart_labels[1] == 0):
                ax1.plot(
                    point.coords[1].real, 
                    point.coords[1].imag, 
                    abs(beta_major),
                    point_colors[k])
                ax1.plot(
                    (point.coords[1]*phi_loop[j,0]).real, 
                    (point.coords[1]*phi_loop[j,0]).imag, 
                    abs(beta_major),
                    point_colors[k])
            else:
                ax2.plot(
                    point.coords[1].real, 
                    point.coords[1].imag, 
                    abs(beta_major),
                    point_colors[k])
                ax2.plot(
                    (point.coords[1]/phi_loop[j,0]).real, 
                    (point.coords[1]/phi_loop[j,0]).imag, 
                    abs(beta_major),
                    point_colors[k])
        # plt.pause(0.1)
 

    plt.show()

def test_CP1_GBZ():
    E = 3
    f = test_char_poly()
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([2,1])
    )

    beta_loop, phi_loop = pGs.solve_aGBZ("GBZ", f_partial, N_points=200)
    GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DGBZCP1(f_partial), beta_loop, 1e-4, 'GBZ')

    point_num = np.zeros(len(GBZ_loops), dtype=int)
    for j, curr_vec in enumerate(GBZ_loops):
        point_num[j] = len(curr_vec)
    
    plt.plot(point_num)
    plt.show()

def test_CP1_segments():
    E = 3
    f = test_char_poly()
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([2,1])
    )

    beta_loop, phi_loop = pGs.solve_aGBZ("CP1", f_partial, N_points=200)
    GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DCP1(f_partial), beta_loop, 1e-4, 'CP1')
    print(len(GBZ_loops))
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.5)
    print(len(all_segments))
    all_segments = pGs.expand_segments(all_segments)
    print(len(all_segments))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax = fig.gca()
    for seg in all_segments:
        all_points = np.zeros((len(seg.point_vec), 2), dtype=complex)
        # beta_norm = np.zeros(len(seg.point_vec), dtype=float)
        params = np.asarray(seg.param_list).flatten()
        exp_iphi = np.exp(1j * params)
        for j, point in enumerate(seg.point_vec):
            if(point.chart_labels[0] == 1):
                all_points[j,0] = 1/point.coords[0]
            else:
                all_points[j,0] = point.coords[0]
            if(point.chart_labels[1] == 1):
                all_points[j,1] = 1/point.coords[1]
            else:
                all_points[j,1] = point.coords[1]

        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])
        # ax.plot(all_points[:,1].real, all_points[:,1].imag, 
        #         2*np.arctan(np.abs(all_points[:,0])), '.-')
        ax.plot(all_points[:,1].real, all_points[:,1].imag, 
                params, '-')
        ax.plot(all_points[0,1].real, all_points[0,1].imag, 
                params[0], 'x')
        # ax.plot((all_points[:,1] * exp_iphi).real, (all_points[:,1] * exp_iphi).imag, 
        #         2*np.arctan(np.abs(all_points[:,0])), '.-')

        # ax.plot(all_points[:,0].real, all_points[:,0].imag, '.-')

        # ax.plot((all_points[:,1] * params).real, (all_points[:,1] * params).imag, '.-')
        # ax.plot(all_points[:,1].real, all_points[:,1].imag, '.-')
    
    plt.show()

def test_loops():
    E = 3
    f = test_char_poly()
    f_partial = f.partial_eval(
        pt.CScalarVec([E]),
        pt.CIndexVec([0]),
        pt.CIndexVec([2,1])
    )

    beta_loop, phi_loop = pGs.solve_aGBZ("CP1", f_partial, N_points=200, glue_tol=0.05)
    GBZ_loops = pGs.pick_GBZ(pGs.aGBZEquation1DCP1(f_partial), beta_loop, 1e-4, 'CP1')
    print(len(GBZ_loops))
    all_segments = pGs.get_GBZ_segments(GBZ_loops, phi_loop, 0.5)
    print(len(all_segments))
    all_segments = pGs.expand_segments(all_segments)
    print(len(all_segments))
    all_loops = pGs.GBZ_segments_to_loops(all_segments)
    all_loop_global = pGs.batch_glue(all_loops)
    print(len(all_loop_global))
    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.gca()
    for seg in all_loop_global:
        all_points = np.zeros((len(seg.point_vec), 2), dtype=complex)
        # beta_norm = np.zeros(len(seg.point_vec), dtype=float)
        params = np.asarray(seg.param_list).flatten()
        exp_iphi = np.exp(1j * params)
        for j, point in enumerate(seg.point_vec):
            if(point.chart_labels[0] == 1):
                all_points[j,0] = 1/point.coords[0]
            else:
                all_points[j,0] = point.coords[0]
            if(point.chart_labels[1] == 1):
                all_points[j,1] = 1/point.coords[1]
            else:
                all_points[j,1] = point.coords[1]

        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])
        # ax.plot(all_points[:,1].real, all_points[:,1].imag, 
        #         2*np.arctan(np.abs(all_points[:,0])), '-')
        # ax.plot(all_points[0,1].real, all_points[0,1].imag, 
        #         2*np.arctan(np.abs(all_points[0,0])), 'x')
        # ax.plot(all_points[:,1].real, all_points[:,1].imag, 
        #         params, '-')
        # ax.plot(all_points[0,1].real, all_points[0,1].imag, 
        #         params[0], 'x')
        # ax.plot((all_points[:,1] * exp_iphi).real, (all_points[:,1] * exp_iphi).imag, 
        #         2*np.arctan(np.abs(all_points[:,0])), '.-')

        ax.plot(all_points[:,0].real, all_points[:,0].imag, '.-')

        theta = np.linspace(0, 2*np.pi, 500)
        beta_norm = 2
        ax.plot(beta_norm * np.cos(theta), beta_norm * np.sin(theta))
        
        all_critical = pGs.find_critical_beta(beta_norm, seg)
        all_critical = np.asarray(all_critical)
        print(np.abs(all_critical))
        ax.plot(all_critical.real, all_critical.imag, 'x')

        # ax.plot((all_points[:,1] * params).real, (all_points[:,1] * params).imag, '.-')
        # ax.plot(all_points[:,1].real, all_points[:,1].imag, '.-')
    
    plt.show()



if __name__ == '__main__':
    # test_aGBZ()
    # check_CP1_calculation()
    # test_CP1_GBZ()
    # test_CP1_segments()
    test_loops()
