'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-03-15
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import brute_force_GBZ as bfG
import BerryPy.TightBinding as tb
import pickle
import matplotlib.pyplot as plt
import numpy as np


def ssh_model(u, v, w, m, gamma) -> tb.TightBindingModel:
    dim = 1
    site_num = 2
    lattice_vec = [[1]]
    intra_cell = [
        [0, 0, m],
        [1, 1, -m],
        [0, 1, u],
        [1, 0, u]
    ]
    inter_cell = [
        [1, 0, v + gamma/2, (1,)],
        [0, 1, v - gamma/2, (-1,)],
        [0, 0, w, (1,)],
        [0, 0, w, (-1,)],
        [1, 1, w, (1,)],
        [1, 1, w, (-1,)]
    ]
    return tb.TightBindingModel(
        dim, site_num, lattice_vec, intra_cell, inter_cell
    )


def calculate_GBZ_for_ssh():
    ''' Generate a test example for the 1D ssh model, 
        save the parameters and the GBZ to a pickle file in 'data' folder.
    '''
    # 1. Generate the model
    u = 1
    v = 1
    w = 1
    m = 1
    gamma = 1
    model = ssh_model(u, v, w, m, gamma)

    # 2. Get the characteristic polynomial
    coeffs, degs = model.get_characteristic_polynomial_data()
    char_poly = bfG.pt.CLaurent(2)
    char_poly.set_Laurent_by_terms(
        bfG.pt.CScalarVec(coeffs), 
        bfG.pt.CLaurentIndexVec(degs.flatten())
    )

    # 3. Calculate the GBZ
    GBZ = bfG.solve_GBZ(char_poly, n_phi=50, n_processes=1)

    # 4. Save the parameters and the GBZ to a pickle file
    data = {
        'u': u,
        'v': v,
        'w': w,
        'm': m,
        'gamma': gamma,
        'GBZ': GBZ
    }
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/ssh_GBZ.pickle', 'wb') as f:
        pickle.dump(data, f)


def visualize_GBZ():
    """
    Load the GBZ data from the pickle file and visualize it.
    """
    try:
        with open('data/ssh_GBZ.pickle', 'rb') as f:
            data = pickle.load(f)
        GBZ = data['GBZ']
        # eigenenergies = np.array([np.real(x) for x in GBZ])
        # betas = np.array(GBZ)

        plt.figure(figsize=(12, 6))

        # Plot eigenenergies
        plt.subplot(1, 2, 1)
        plt.plot(GBZ[:,0].real, GBZ[:,0].imag, '.')
        plt.title('Eigenenergies')

        # Plot betas on the complex plane
        plt.subplot(1, 2, 2)
        plt.plot(GBZ[:,1].real, GBZ[:,1].imag, '.')
        plt.title('Betas on Complex Plane')
        plt.xlabel('Real part')
        plt.ylabel('Imaginary part')

        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print("The pickle file was not found. Please run calculate_GBZ_for_ssh first.")


if __name__ == "__main__":
    # calculate_GBZ_for_ssh()
    visualize_GBZ()
