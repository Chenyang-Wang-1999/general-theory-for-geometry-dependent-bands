'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-04-17
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import struct
import h5py
from scipy import sparse
import numpy as np

def mat_to_file(M: sparse.coo_matrix, fname: str):
    row_indices = M.row
    col_indices = M.col
    data = M.data
    n_terms = len(M.row)
    mat_size = M.shape[0]
    with open(fname, "wb") as fp:
        fp.write(
            struct.pack('>i', mat_size)
        )
        fp.write(
            struct.pack('>i', n_terms)
        )
        for row_id in row_indices:
            fp.write(
                struct.pack('>i', row_id)
            )
        for col_id in col_indices:
            fp.write(
                struct.pack('>i', col_id)
            )
        for curr_val in data:
            fp.write(
                struct.pack('>d', curr_val.real)
            )
            fp.write(
                struct.pack('>d', curr_val.imag)
            )

def get_eigensystem_h5(fname, mat_size):
    hf = h5py.File(fname)
    eigv_data = hf.get("eigv")
    if(len(eigv_data.shape) == 3):
        eigv = eigv_data[0,:,0] + 1j * eigv_data[0,:,1]
    else:
        eigv = eigv_data[:,0] + 1j * eigv_data[:,1]

    eigvec = np.zeros((mat_size, len(eigv)), dtype=complex)
    for j in range(len(eigv)):
        eigvec_data = hf.get("eigvec-%d"%(j))
        eigvec[:,j] = eigvec_data[:,0] + 1j * eigvec_data[:,1]
    return eigv, eigvec
 

           