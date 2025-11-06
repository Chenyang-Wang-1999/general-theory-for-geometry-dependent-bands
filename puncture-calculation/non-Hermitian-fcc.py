'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-01-18
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from BerryPy import TightBinding as tb
import matplotlib.pyplot as plt
import poly_tools as pt
import partial_GBZ_solver as pGs
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
import pickle

def 面心二能级模型(coupling_terms:dict): 
    '''
        Definition of fcc 2-level system
        For each sublattice, define the coupling terms on each side
        and in the diagonal direction, the two sites form a SSH model
    '''

    dim = 2
    site_num = 2
    lattice_vec = np.eye(2)

    v右上 = coupling_terms['vUR']
    v左下 = coupling_terms['vDL']
    w右上 = coupling_terms['wUR']
    w左下 = coupling_terms['wDL']
    上0 = coupling_terms['U0']
    下0 = coupling_terms['D0']
    左0 = coupling_terms['L0']
    右0 = coupling_terms['R0']
    上1 = coupling_terms['U1']
    下1 = coupling_terms['D1']
    左1 = coupling_terms['L1']
    右1 = coupling_terms['R1']

    intra_cell = [
        [0, 1, v右上],
        [1, 0, v左下]
    ]

    inter_cell = [
        [1, 0, w右上, (1,1)],
        [0, 1, w左下, (-1,-1)],
        [1, 1, 右1, (1,0)],
        [1, 1, 左1, (-1,0)],
        [1, 1, 上1, (0,1)],
        [1, 1, 下1, (0,-1)],
        [0, 0, 右0, (1,0)],
        [0, 0, 左0, (-1,0)],
        [0, 0, 上0, (0,1)],
        [0, 0, 下0, (0,-1)]
    ]

    return tb.TightBindingModel(dim, site_num, lattice_vec, 
                                intra_cell, inter_cell)

####### calculate supercell GBZ #######

def 计算超胞GBZ(v, dv, w, dw, t0, t1): 
    '''
        v: SSH v
        w: SSH w
        dv: non-reciprocity of v
        dw: non-reciprocity of w
        t0: hopping of lattice 0
        t1: hopping of lattice 1
    '''

    coupling_terms = {
        "vUR": v + dv/2,
        "vDL": v - dv/2,
        "wUR": w + dw/2,
        "wDL": w - dw/2,
        "U0": t0,
        "D0": t0, 
        "L0": t0,
        "R0": t0,
        "U1": t1,
        "D1": t1, 
        "L1": t1,
        "R1": t1
    }

    model = 面心二能级模型(coupling_terms)

    # truncate in y direction
    for 超胞格点数 in [10,11,12]:
        model_1d = model.get_supercell( 
            [(0, j) for j in range(超胞格点数)],
            [[1, 0],
            [0, 超胞格点数]]
        )
        model_1d = model_1d.truncate(1)

        # # truncate in x direction
        # 超胞格点数 = 6
        # model_1d = model.get_supercell( 
        #     [(j, 0) for j in range(超胞格点数)],
        #     [[超胞格点数, 0],
        #      [0, 1]]
        # )
        # model_1d = model_1d.truncate(0)

        flag, E, beta = model_1d.get_GBZ(N_phi = 51, N_process = 6)

        with open("data/GBZ_in_x_%d.pkl"%(超胞格点数), "wb") as fp:
            pickle.dump((E,beta), fp)
    
    return 
    
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.plot(beta.real, beta.imag, E.real, '.')

    fig = plt.figure(2)
    ax = fig.gca()
    ax.plot(E.real, E.imag, '.')

    fig = plt.figure(3)
    ax = fig.gca()
    ax.plot(beta.real, beta.imag, '.')
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'r--')
    
    plt.show()

def 输出特征多项式(v, dv, w, dw, t0, t1): 
    coupling_terms = {
        "vUR": v + dv/2,
        "vDL": v - dv/2,
        "wUR": w + dw/2,
        "wDL": w - dw/2,
        "U0": t0,
        "D0": t0, 
        "L0": t0,
        "R0": t0,
        "U1": t1,
        "D1": t1, 
        "L1": t1,
        "R1": t1
    }

    model = 面心二能级模型(coupling_terms)
    return model.get_characteristic_polynomial()

def SSH方向趋肤效应(v, dv, w, dw, t0, t1):
    '''
        Skin effect along the direction (1,1)
    '''
    coupling_terms = {
        "vUR": v + dv/2,
        "vDL": v - dv/2,
        "wUR": w + dw/2,
        "wDL": w - dw/2,
        "U0": t0,
        "D0": t0, 
        "L0": t0,
        "R0": t0,
        "U1": t1,
        "D1": t1, 
        "L1": t1,
        "R1": t1
    }

    model = 面心二能级模型(coupling_terms)

    # change lattice vectors
    model_along_diagonal = model.get_supercell(
        [(0,0)], 
        [[1, 0],
         [1, 1]]
    )

    # get supercell
    超胞格点数 = 6
    model_1d = model_along_diagonal.get_supercell(
        [(0, j) for j in range(超胞格点数)],
        [[1,0],
         [0,超胞格点数]]
    ).truncate(1)

    # get GBZ
    flag, E, beta = model_1d.get_GBZ(N_process = 6)
    
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.plot(beta.real, beta.imag, E.real, '.')

    fig = plt.figure(2)
    ax = fig.gca()
    ax.plot(E.real, E.imag, '.')

    fig = plt.figure(3)
    ax = fig.gca()
    ax.plot(beta.real, beta.imag, '.')
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'r--')
    
    plt.show()

####### calculate partial GBZ #######
def 检查偏GBZ是否存在尖点():
    '''
        check partial GBZ
    '''

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

    r = 1.5
    beta1_list = r * np.exp(1j*np.linspace(0, 2*np.pi, 11))
    for beta1 in beta1_list:
        curr_poly = f.partial_eval(
            pt.CVarScalarVec([beta1]),
            pt.CIndexVec([1]),
            pt.CIndexVec([0,2])
        )
        GBZ_loops, phi_loops = pGs.solve_GBZ(curr_poly)
        fig1 = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        ax1 = fig1.gca()
        fig2 = plt.figure()
        ax2 = fig2.gca()
        for j, loop in enumerate(GBZ_loops) :
            # ax.plot(loop[:,1].real, loop[:,1].imag, loop[:,0].real)
            # ax.plot(loop[:,1].real, loop[:,1].imag, phi_loops[j][:,0])
            # ax.plot(phi_loops[j][:,0], np.log(loop[:,1]).imag, '.')
            ax1.plot(np.arange(phi_loops[j].shape[0]), phi_loops[j][:,0], '.')
            ax2.plot(np.arange(loop.shape[0]), np.log(loop[:,1]).imag, '.')
        
        # ax.set_title("$%f \pi$"%(np.log(beta1).imag/(2*np.pi)))
        ax1.set_title("$%f \pi$"%(np.log(beta1).imag/(2*np.pi)))
        ax2.set_title("$%f \pi$"%(np.log(beta1).imag/(2*np.pi)))
        
        plt.show()

def calculate_OBC_specturm(v, dv, w, dw, t0, t1):
    coupling_terms = {
        "vUR": v + dv/2,
        "vDL": v - dv/2,
        "wUR": w + dw/2,
        "wDL": w - dw/2,
        "U0": t0,
        "D0": t0, 
        "L0": t0,
        "R0": t0,
        "U1": t1,
        "D1": t1, 
        "L1": t1,
        "R1": t1
    }

    model = 面心二能级模型(coupling_terms)

    N_cells = 50
    model_1d = model.get_supercell(
        [(j, 0) for j in range(N_cells)], 
        [
            [N_cells, 0],
            [0, 1]
        ])
    model_2d = model_1d.get_supercell(
        [(0,j) for j in range(N_cells)],
        [
            [1, 0],
            [0, N_cells]
        ]
    )
    H = model_2d.get_bulk_Hamiltonian_complex((None, None))
    eigv, eigvec = la.eig(H.todense())
    with open("data/OBC-spectrum-N-%d.pkl"%(N_cells), "wb") as fp:
        pickle.dump((eigv, eigvec), fp)



def calculate_punctures(E:complex, poly:pt.CLaurent, major_var:int=1):
    ''' 
        Calculate the punctures for given E
    '''

    minor_var = int(not major_var)
    poly_beta = poly.partial_eval(
        pt.CVarScalarVec([E]), pt.CIndexVec([0]), 
        pt.CIndexVec([major_var + 1, minor_var + 1])
    )

    GBZ_loop, beta_loop = pGs.solve_GBZ(poly_beta)
    with open("data/punctures-E_%f_%f-major_var_%d.pkl"%(E.real, E.imag, major_var), "wb") as fp:
        pickle.dump({
            'GBZ_loop': GBZ_loop,
            'beta_loop': beta_loop
        }, fp)
    
    for loop in GBZ_loop:
        plt.plot(loop[:,0].real, loop[:,0].imag)
    plt.show()
    # plt.savefig("Figure/temp.pdf")
    

####### main functions #######
def 主_计算超胞GBZ(): 
    v = 1
    w = 0.5
    dv = 0.2
    dw = -0.2
    t0 = 0.2
    t1 = 0.1
    计算超胞GBZ(v, dv, w, dw, t0, t1)

def 主_输出特征多项式(): 
    v = 1
    w = 0.5
    dv = 0.2
    dw = -0.2
    t0 = 0.2
    t1 = 0.1
    poly_str = str(输出特征多项式(v, dv, w, dw, t0, t1))
    poly_str = poly_str.replace("beta", "\\beta").replace("**", "^")
    print(poly_str)

def 主_SSH方向趋肤效应():
    v = 1
    w = 0.5
    dv = 0.2
    dw = -0.2
    t0 = 0.2
    t1 = 0.1
    SSH方向趋肤效应(v, w, dv, dw, t0, t1)

def main_calculate_OBC_spectrum():
    v = 1
    w = 0.5
    dv = 0.2
    dw = -0.2
    t0 = 0.2
    t1 = 0.1
    calculate_OBC_specturm(v, dv, w, dw, t0, t1)

def main_calculate_puncture():
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
    calculate_punctures(0.4, f, 0)

def main_plot_phase():
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

    beta1 = 1.5 * np.exp(1j * np.random.rand() * 2 * np.pi)
    f_truncate = f.partial_eval(
        pt.CVarScalarVec([beta1]), pt.CIndexVec([1]), pt.CIndexVec([0,2]))

    GBZ_loops, phi_loops = pGs.solve_GBZ(f_truncate)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for j, loop in enumerate(GBZ_loops) :
        print(phi_loops[j].shape)
        ax.plot(loop[:,1].real, loop[:,1].imag, phi_loops[j][:,0])
    plt.show()
 
def main_plot_OBC_spectrum():
    N = 50
    with open("data/OBC-spectrum-N-%d.pkl"%(N), "rb") as fp:
        eigv, eigvec = pickle.load(fp)
    
    plt.plot(eigv.real, eigv.imag, '.')
    plt.show()

def main_plot_1D_GBZ_spectrum():
    for N in [9]:
        with open("data/GBZ_in_x_%d.pkl"%(N), "rb") as fp:
            E, beta = pickle.load(fp)
        with open("data/OBC-spectrum-N-50.pkl", "rb") as fp:
            eigv, eigvec = pickle.load(fp)
        plt.plot(eigv.real, eigv.imag, '.')
        plt.plot(E.real, E.imag, '.')
        plt.show()

def calculate_eigen_states(H: tb.TightBindingModel, 
                           GBZ_point_list: list[tuple[complex]]):
    ''' Calculate the eigen state of a Hamiltonian for given GBZ points
        H: BerryPy TightBindingModel object
        GBZ_point_list: list of GBZ points
    '''
    def GBZ_point_to_eigvec(E_beta: tuple[complex]) -> np.ndarray:
        E = E_beta[0]
        beta = E_beta[1:]
        H_mat = H.get_bulk_Hamiltonian_complex(beta)
        eig_eq = E * sparse.eye(H_mat.shape[0]) - H_mat

        #### solve eigen state ####

def check_supercell_GBZ_data_format():
    with open("data/GBZ_in_x.pkl", "rb") as fp:
        GBZ_data = pickle.load(fp)
    
    E, beta = GBZ_data
    plt.figure(1)
    plt.plot(E.real, E.imag, ".")
    plt.figure(2)
    plt.plot(beta.real, beta.imag, ".")
    plt.show()

def pick_GBZ_point(point_id: int, whether_to_plot=True):
    with open("data/GBZ_in_x_9.pkl", "rb") as fp:
        GBZ_data = pickle.load(fp)
    
    E, beta = GBZ_data
    if(whether_to_plot):
        plt.figure(1)
        plt.plot(E.real, E.imag, ".")
        plt.plot(E[point_id].real, E[point_id].imag, "x")
        plt.figure(2)
        plt.plot(beta.real, beta.imag, ".")
        plt.plot(beta[point_id].real, beta[point_id].imag, "x")
        plt.show()
    return E[point_id], beta[point_id]

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
    # E, beta = pick_GBZ_point(100, False)
    E = 0.520263020298578+0.0175837923526226j
    g = f.partial_eval(
        pt.CVarScalarVec([E]), 
        pt.CIndexVec([0]), 
        pt.CIndexVec([1,2]))
    
    ### calculate GBZ ###
    GBZ_loops, phi_loops = pGs.solve_GBZ(g, rel_tol=1e-4)
    with open("data/punctures-E_%.6f_%.6f.pkl"%(E.real, E.imag), 'wb') as fp:
        pickle.dump((GBZ_loops, phi_loops), fp)
    
def plot_GBZ_for_given_E():
    # E, beta = pick_GBZ_point(100, False)
    E = 0.520263020298578+0.0175837923526226j
    beta = -0.9574457124464715-0.3875133499764865j
    with open("data/punctures-E_%.6f_%.6f.pkl"%(E.real, E.imag), 'rb') as fp:
        GBZ_loops, phi_loops = pickle.load(fp)
    
    print(len(GBZ_loops))
    for loop in GBZ_loops:
        plt.plot(loop[:,0].real, loop[:,0].imag, '.')
    
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot((beta*np.exp(1j*theta)).real, (beta*np.exp(1j*theta)).imag, 'r--')
    plt.plot(beta.real, beta.imag, 'rx')
    plt.show()

def get_critical_punctures():
    E, beta = pick_GBZ_point(100, False)
    with open("data/punctures-E_%.6f_%.6f.pkl"%(E.real, E.imag), 'rb') as fp:
        GBZ_loops, phi_loops = pickle.load(fp)
    
    min_dist = 0.0002
    print(len(GBZ_loops))
    for j, loop in enumerate(GBZ_loops):
        near_beta_norm = np.abs(abs(beta) - np.abs(loop[:,0])) < min_dist
        picked_id = np.arange(loop.shape[0])[near_beta_norm]
        if(len(picked_id)):
            print(j, picked_id)
            print(np.abs(loop[picked_id,0]) - abs(beta))
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
    E, beta = pick_GBZ_point(100, False)
    with open("data/punctures-E_%.6f_%.6f.pkl"%(E.real, E.imag), 'rb') as fp:
        GBZ_loops, phi_loops = pickle.load(fp)
    
    GBZ_beta_list = [
        GBZ_loops[0][518,0], 
        GBZ_loops[0][519,0],
        GBZ_loops[1][727,0],
        GBZ_loops[1][728,0]
    ]
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

    for j in range(4):
        g = f.partial_eval(
            pt.CVarScalarVec([GBZ_beta_list[j]]), 
            pt.CIndexVec([1]), 
            pt.CIndexVec([0,2]))
        GBZ_loops, phi_loops = pGs.solve_GBZ(g, rel_tol=1e-4)
        print(len(GBZ_loops))
        with open("data/partial-GBZ-%d.pkl"%(j), "wb") as fp:
            pickle.dump((GBZ_loops, phi_loops), fp)

def get_special_point_list(j):
    E, beta = pick_GBZ_point(100, False)
    with open("data/punctures-E_%.6f_%.6f.pkl"%(E.real, E.imag), 'rb') as fp:
        GBZ_loops, phi_loops = pickle.load(fp)
    if(j == 0):
        return [
            [E] + list(GBZ_loops[0][518,:]),
            [E] + list(GBZ_loops[0][681,:])
        ]
    elif(j == 1):
        return [
            [E] + list(GBZ_loops[0][519,:]),
            [E] + list(GBZ_loops[0][680,:])
        ]
    elif(j == 2):
        return [
            [E] + list(GBZ_loops[1][727,:]),
            [E] + list(GBZ_loops[1][890,:])
        ]
    else:
        return [
            [E] + list(GBZ_loops[1][728,:]),
            [E] + list(GBZ_loops[1][889,:])
        ]

def plot_partial_GBZ():
    for j in range(4):
        with open("data/partial-GBZ-%d.pkl"%(j), "rb") as fp:
            GBZ_loops, phi_loops = pickle.load(fp)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for loop in GBZ_loops:
            ax.plot(loop[:,1].real, loop[:,1].imag, loop[:,0].real)
        plt.show()

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
    target_E = 0.52 + 0.018j

    # load data of supercell GBZ
    with open("data/GBZ_in_x_9.pkl", "rb") as fp:
        E, beta = pickle.load(fp)
    
    selected_points = np.abs(E - target_E) < 1.1 * np.min(np.abs(E-target_E))
    selected_indices = np.arange(len(E))[selected_points]
    for point_id in selected_indices:
        print("E:", E[point_id], "beta:", beta[point_id], "|beta|:", abs(beta[point_id]))


if __name__ == '__main__':
    # 主_计算超胞GBZ()
    # 主_输出特征多项式()
    # 主_SSH方向趋肤效应()
    # 检查偏GBZ是否存在尖点()
    # main_calculate_puncture()
    # main_plot_phase()
    # check_supercell_GBZ_data_format()
    # print(pick_GBZ_point(100))
    # main_calculate_OBC_spectrum()
    # main_plot_OBC_spectrum()
    main_plot_1D_GBZ_spectrum()
    # main_calculate_puncture()
    # main_calculate_GBZ_for_given_E()
    # plot_GBZ_for_given_E()
    # get_critical_punctures()
    # calculate_partial_GBZ()
    # plot_partial_GBZ()
    # check_partial_GBZ_coordinate()
    # find_points_in_supercell_GBZ()
