'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2023-10-12
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import pickle
from scipy import io as sio
import numpy as np
from scipy import linalg as la
import os

import pyvista as pv
import matplotlib.pyplot as plt

from non_Hermitian_Haldane import Haldane_non_Hermitian_phase

def plot_Haldane_band():
    tube_radius = 0.1
    kx = np.linspace(-0.5,0.5)
    ky = np.linspace(-0.5,0.5)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    E_list = np.zeros((2,) + kx_mesh.shape)

    model_info = sio.loadmat("data/GBZ_info_5.mat")
    t1 = model_info['t1'][0,0]
    t2 = model_info['t2'][0,0]
    M = model_info['M'][0,0]
    phi = model_info['phi'][0,0]
    gamma = model_info['gamma'][0,0]

    Haldane_model = Haldane_non_Hermitian_phase(t1, t2, phi, M, gamma)

    for i in range(kx_mesh.shape[0]):
        for j in range(kx_mesh.shape[1]):
            H = Haldane_model.get_bulk_Hamiltonian_dense((kx_mesh[i,j], ky_mesh[i,j]))
            E_list[:,i,j] = la.eigh(H)[0]
    
    all_meshes = []
    for j in range(2):
        all_meshes.append(pv.StructuredGrid(kx_mesh*2*np.pi, ky_mesh*2*np.pi, E_list[j,:,:]))

    pl = pv.Plotter()
    for j in range(2):
        # ax.plot_surface(kx_mesh*2*np.pi, ky_mesh*2*np.pi, E_list[j,:,:])
        pl.add_mesh(all_meshes[j])
    for j in range(50):
        print(j)
        curr_fname = "data/GBZ_data_6/x_major_%d.pkl"%(j)
        if(os.path.exists(curr_fname)):
            # ax = fig.add_subplot(projection='3d')
            with open(curr_fname,"rb") as fp:
                all_segs = pickle.load(fp)
            for seg in all_segs:
                ax.plot()
                # ax.plot(np.log(seg[:,1]).imag, np.log(seg[:,2]).imag, seg[:,0].real, '.')
                if(seg.shape[0] > 1):
                    # curr_curve = pv.MultipleLines(np.column_stack([np.log(seg[:,1]).imag, np.log(seg[:,2]).imag, seg[:,0].real])).tube(tube_radius)
                    # pl.add_mesh(curr_curve, color="#0000FF")
                    pl.add_points(np.column_stack([np.log(seg[:,1]).imag, np.log(seg[:,2]).imag, seg[:,0].real]), color="#0000ff")
    pl.show()

def plot_Haldane_spectrum():
    fig = plt.figure()
    ax = fig.gca()
    for j in range(50):
        print(j)
        curr_fname = "data/GBZ_data_6/x_major_%d.pkl"%(j)
        if(os.path.exists(curr_fname)):
            # ax = fig.add_subplot(projection='3d')
            with open(curr_fname,"rb") as fp:
                all_segs = pickle.load(fp)
            for seg in all_segs:
                if(seg.shape[0] > 1):
                    ax.plot(seg[:,0].real, seg[:,0].imag,'.')

    fig2 = plt.figure()
    N1 =50
    N2 = 50
    ax = fig2.gca()
    data = sio.loadmat("data/OBC_spectrum_6_N1%d_N2%d.mat"%(N1, N2))
    eigv = data['eigv']
    ax.plot(eigv.real, eigv.imag,'.')
    plt.show()



def plot_Haldane_OBC():
    with open("data/GBZ_model_7.pkl", "rb") as fp:
        model = pickle.load(fp)
    # print(model.InCell)
    # print(model.InterCell)
    # return
    
    # get supercell
    N1 = 50
    N2 = 50
    model_super1d = model.get_supercell(
        [(j,0) for j in range(N1)],
        [
            [N1, 0],
            [0, 1]
        ]
    )
    model_super2d = model_super1d.get_supercell(
        [(0,j) for j in range(N2)],
        [
            [1, 0],
            [0, N2]
        ]
    )
    H = model_super2d.get_bulk_Hamiltonian_dense((None, None))
    # print(H)
    eigv, eigvec = la.eig(H)
    data = {"eigv":eigv, "eigvec":eigvec}
    sio.savemat("data/OBC_spectrum_7_N1%d_N2%d.mat"%(N1, N2), data)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(eigv.real, eigv.imag, '.')
    plt.show()

if __name__ == "__main__":
    # plot_Haldane_spectrum()
    plot_Haldane_OBC()