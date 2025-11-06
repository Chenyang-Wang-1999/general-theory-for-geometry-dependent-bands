'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-02-21
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

''' Check GBZ condition near the bandgap '''




import numpy as np
from BerryPy import TightBinding as tb
import matplotlib.pyplot as plt
import poly_tools as pt
import partial_GBZ_solver as pGs
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
import pickle
import os

coeffs = pt.CScalarVec([
    1, -1.15, -0.3, -0.3, -0.3, -0.3, 0.02, 0.02, 0.02, 0.02, 
    -0.5, -0.4, 0.04, 0.04
])
degs = pt.CLaurentIndexVec([
    2, 0, 0, 
    0, 0, 0,
    1, 1, 0,
    1, -1, 0,
    1, 0, -1,
    1, 0, 1,
    0, 2, 0,
    0, -2, 0,
    0, 0, 2,
    0, 0, -2,
    0, 1, 1,
    0, -1, -1,
    0, 1, -1,
    0, -1, 1 
])

f = pt.CLaurent(3)
f.set_Laurent_by_terms(coeffs, degs)

E_target = 0.520263020298578+0.0175837923526226j
beta_major_list = [
    -0.963117-0.372106j,
    0.564072-0.864686j
]
beta_major_outer = [
    -0.9767599 - 0.36437531j,
    0.53156588 - 0.89660395j
]
beta_major_inner = [
    -0.94818341 - 0.38091157j,
    0.59508064 - 0.83060485j
]

def main_calculate_GBZ_for_given_E():
    coeffs = pt.CScalarVec([
        1, -1.15, -0.3, -0.3, -0.3, -0.3, 0.02, 0.02, 0.02, 0.02, 
        -0.5, -0.4, 0.04, 0.04
    ])
    degs = pt.CLaurentIndexVec([
        2, 0, 0, 
        0, 0, 0,
        1, 1, 0,
        1, -1, 0,
        1, 0, -1,
        1, 0, 1,
        0, 2, 0,
        0, -2, 0,
        0, 0, 2,
        0, 0, -2,
        0, 1, 1,
        0, -1, -1,
        0, 1, -1,
        0, -1, 1 
    ])

    f = pt.CLaurent(3)
    f.set_Laurent_by_terms(coeffs, degs)

    ### given E ###
    g = f.partial_eval(
        pt.CVarScalarVec([E_target]), 
        pt.CIndexVec([0]), 
        pt.CIndexVec([1,2]))
    
    ### calculate GBZ ###
    GBZ_loops, phi_loops = pGs.solve_GBZ(g, rel_tol=1e-4)
    with open("data/punctures-E_%.6f_%.6f.pkl"%(E_target.real, E_target.imag), 'wb') as fp:
        pickle.dump((GBZ_loops, phi_loops), fp)
    
def plot_GBZ_for_given_E():
    with open("data/punctures-E_%.6f_%.6f.pkl"%(E_target.real, E_target.imag), 'rb') as fp:
        GBZ_loops, phi_loops = pickle.load(fp)
    
    print(len(GBZ_loops))
    for loop in GBZ_loops:
        plt.plot(loop[:,0].real, loop[:,0].imag, '-')
    
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot((beta_major_list[0]*np.exp(1j*theta)).real, 
             (beta_major_list[0]*np.exp(1j*theta)).imag, 'g--')
    plt.plot(1.01*(beta_major_list[0]*np.exp(1j*theta)).real, 
             1.01*(beta_major_list[0]*np.exp(1j*theta)).imag, 'r--')
    plt.plot(0.99*(beta_major_list[0]*np.exp(1j*theta)).real, 
             0.99*(beta_major_list[0]*np.exp(1j*theta)).imag, 'r--')
    for beta_target in beta_major_list:
        plt.plot(beta_target.real, beta_target.imag, 'rx')
    plt.show()

def get_critical_punctures():
    with open("data/punctures-E_%.6f_%.6f.pkl"%(E_target.real, E_target.imag), 'rb') as fp:
        GBZ_loops, phi_loops = pickle.load(fp)
    
    min_dist = 0.0002
    print(len(GBZ_loops))
    for j, loop in enumerate(GBZ_loops):
        near_beta_norm = np.abs(abs(beta_major_list[0]) - np.abs(loop[:,0])) < min_dist
        picked_id = np.arange(loop.shape[0])[near_beta_norm]
        if(len(picked_id)):
            print(j, picked_id)
            print(np.abs(loop[picked_id,0]) - abs(beta_major_list[0]))
    ''' picked pairs
        0: 518, 519
        0: 680, 681
        1: 727, 728
        1: 889, 890
    '''

    print(GBZ_loops[0][518,:])
    print(GBZ_loops[0][519,:])
    print(GBZ_loops[0][680,:])
    print(GBZ_loops[0][681,:])
    print(GBZ_loops[1][727,:])
    print(GBZ_loops[1][728,:])
    print(GBZ_loops[1][889,:])
    print(GBZ_loops[1][890,:])

def calculate_partial_GBZ():
    os.system("rm data/partial-GBZ-*.pkl")
    for j in range(2):
        g = f.partial_eval(
            pt.CVarScalarVec([beta_major_list[j]]), 
            pt.CIndexVec([1]), 
            pt.CIndexVec([0,2]))
        GBZ_loops, phi_loops = pGs.solve_GBZ(g, rel_tol=1e-4)
        print(len(GBZ_loops))
        with open("data/partial-GBZ-%d.pkl"%(j), "wb") as fp:
            pickle.dump((GBZ_loops, phi_loops), fp)

        g = f.partial_eval(
            pt.CVarScalarVec([beta_major_outer[j]]), 
            pt.CIndexVec([1]), 
            pt.CIndexVec([0,2]))
        GBZ_loops, phi_loops = pGs.solve_GBZ(g, rel_tol=1e-4)
        print(len(GBZ_loops))
        with open("data/partial-GBZ-outer-%d.pkl"%(j), "wb") as fp:
            pickle.dump((GBZ_loops, phi_loops), fp)

        g = f.partial_eval(
            pt.CVarScalarVec([beta_major_inner[j]]), 
            pt.CIndexVec([1]), 
            pt.CIndexVec([0,2]))
        GBZ_loops, phi_loops = pGs.solve_GBZ(g, rel_tol=1e-4)
        print(len(GBZ_loops))
        with open("data/partial-GBZ-inner-%d.pkl"%(j), "wb") as fp:
            pickle.dump((GBZ_loops, phi_loops), fp)

def get_special_point_list(j):
    with open("data/punctures-E_%.6f_%.6f.pkl"%(E_target.real, E_target.imag), 'rb') as fp:
        GBZ_loops, phi_loops = pickle.load(fp)
    if(j == 0):
        return [
            [E_target] + list(GBZ_loops[0][518,:]),
            [E_target] + list(GBZ_loops[0][681,:])
        ]
    elif(j == 1):
        return [
            [E_target] + list(GBZ_loops[0][519,:]),
            [E_target] + list(GBZ_loops[0][680,:])
        ]
    elif(j == 2):
        return [
            [E_target] + list(GBZ_loops[1][727,:]),
            [E_target] + list(GBZ_loops[1][890,:])
        ]
    else:
        return [
            [E_target] + list(GBZ_loops[1][728,:]),
            [E_target] + list(GBZ_loops[1][889,:])
        ]

def plot_partial_GBZ():
    for j in range(2):
        with open("data/partial-GBZ-%d.pkl"%(j), "rb") as fp:
            GBZ_loops, phi_loops = pickle.load(fp)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for loop in GBZ_loops:
            ax.plot(loop[:,1].real, loop[:,1].imag, loop[:,0].real)
        fig = plt.figure()
        ax = fig.gca()
        for loop in GBZ_loops:
            ax.plot(loop[:,0].real, loop[:,0].imag)
        ax.plot(E_target.real, E_target.imag, 'x')
        plt.show()

def find_critical_points_in_partial_GBZ(suffix: str = ''):
    char_poly_info = pGs.CharPolyInfo(f)
    for j in range(2):
        print("Major beta: ", beta_major_list[j])
        with open("data/partial-GBZ%s-%d.pkl"%(suffix, j), "rb") as fp:
            GBZ_loops, phi_loops = pickle.load(fp)
        for k, loop in enumerate(GBZ_loops):
            selected_points = np.abs(loop[:,0] - E_target) <= 6e-4
            selected_indices = np.arange(loop.shape[0])[selected_points]
            if(len(selected_indices)):
                print(k)
                print(selected_indices)
                detE, dbeta = char_poly_info.get_det_and_dbeta(
                    E_target, 
                    beta_major_list[j], 
                    loop[selected_indices[0], 1], 
                    loop[selected_indices[1], 1] 
                    )
                print(detE, dbeta)
                print(selected_indices[1]-selected_indices[0])
                print(phi_loops[k][selected_indices[0],0])
                print(phi_loops[k][selected_indices[1],0])
                print(phi_loops[k][selected_indices[1],0] - phi_loops[k][selected_indices[0],0])

def match_point_id():
    for j in range(4):
        with open("data/partial-GBZ-%d.pkl"%(j), "rb") as fp:
            GBZ_loops, phi_loops = pickle.load(fp)
        point_list = get_special_point_list(j)
        point_list = np.array(point_list)

def check_partial_GBZ_coordinate():
    for j in range(4):
        with open("data/partial-GBZ-%d.pkl"%(j), "rb") as fp:
            GBZ_loops, phi_loops = pickle.load(fp)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for k, loop in enumerate(GBZ_loops):
            ax.plot(loop[:,1].real, loop[:,1].imag, phi_loops[k][:,0])
        plt.show()

def find_points_in_supercell_GBZ():
    # load data of supercell GBZ
    with open("data/GBZ_in_x_9.pkl", "rb") as fp:
        E, beta = pickle.load(fp)
    
    selected_points = np.abs(E - E_target) <= 1.1 * np.min(np.abs(E-E_target))
    selected_indices = np.arange(len(E))[selected_points]
    for point_id in selected_indices:
        print("E:", E[point_id], "beta:", beta[point_id], "|beta|:", abs(beta[point_id]))

def plot_OBC_spectrum():
    with open("data/OBC-spectrum-N-50.pkl", "rb") as fp:
        eigv, eigvec = pickle.load(fp)
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(eigv.real, eigv.imag, '.')
    ax.plot(E_target.real, E_target.imag, 'x')
    plt.show()

if __name__ == "__main__":
    # find_points_in_supercell_GBZ()
    # plot_GBZ_for_given_E()
    calculate_partial_GBZ()
    plot_partial_GBZ()
    # find_critical_points_in_partial_GBZ('')
    # plot_OBC_spectrum()
