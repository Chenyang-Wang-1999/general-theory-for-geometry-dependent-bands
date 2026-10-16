'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-02-11
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt

# The color palette is shared by all the plotting scripts of this repository,
# and lives in colors.py at its root.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from colors import COLORS

plt.style.use("../settings-and-materials/paper_plot.mplstyle")

CM = 1/2.54
DATA_DIR = "../../ChP-winding-algorithm/data"

def plot_general(fname, r_list, w_list, w_strip_list, ylim):
    fig = plt.figure(figsize=(3.5 * CM, 4 * CM))
    ax = fig.gca()
    ax.set_position([0.25, 0.2, 0.75, 0.8])
    ax.plot(np.log(r_list), w_list, '-', label="$W_{\\mathrm{q1D}}/L_2$")
    ax.plot(np.log(r_list), w_strip_list, '--', label="$W$")
    # ax.set_xlim([0,3.2])
    ax.set_xlim([-1.1, 1.1])
    # ax.set_xticks([0,1,2,3])
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("Winding number")
    ax.set_ylim(ylim)
    ax.legend()
    fig.savefig("Figures/%s.pdf" % fname)

def main_plot(f_ind: int):
    fname = "nH-Haldane-test"
    ylim = [-2, 2]

    # fname = "2D-HN-test"
    # ylim = [-0.5, 0.5]
    with open("%s/%s-%d.pkl" % (DATA_DIR, fname, f_ind), "rb") as f:
        data = pickle.load(f)
    plot_general("%s-%d" % (fname, f_ind), data["r_list"], data["w_list_q1D"] / data["n_minor"], data["w_list"] / (2 * np.pi),
                 ylim)

def print_info(f_ind: int):
    fname = "nH-Haldane-test"
    # fname = "2D-HN-test"
    with open("%s/%s-%d.pkl" % (DATA_DIR, fname, f_ind), "rb") as f:
        data = pickle.load(f)
    print(f_ind)
    print(data["info"])
    print(data["E_ref"])

if __name__ == '__main__':
    for f_ind in range(6):
        main_plot(f_ind)
        print_info(f_ind)