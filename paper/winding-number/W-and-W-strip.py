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
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from colors import COLORS
import figure_data as fd

plt.style.use("../settings-and-materials/paper_plot.mplstyle")

CM = 1/2.54
DATA_DIR = "../../ChP-winding-algorithm/data"

# The two model families of Fig. S3, with the winding-number range of their panel.
MODELS = {
    "nH-Haldane-test": ("Fig. S3(d)", [-2, 2]),
    "2D-HN-test": ("Fig. S3(b)", [-0.5, 0.5]),
}

def plot_general(fname, model, r_list, w_list, w_strip_list, ylim):
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

    panel, _ = MODELS[model]
    fd.save_csv(
        "Figures/%s" % fname,
        "%s | W_q1D/L2 and W/(2 pi) versus mu_1 = ln(r_1) of %s"
        % (panel, model),
        mu_1=np.log(r_list), w_q1d_over_L2=w_list, w=w_strip_list,
    )

def main_plot(f_ind: int, model: str = "nH-Haldane-test"):
    _, ylim = MODELS[model]
    with open("%s/%s-%d.pkl" % (DATA_DIR, model, f_ind), "rb") as f:
        data = pickle.load(f)
    plot_general("%s-%d" % (model, f_ind), model, data["r_list"], data["w_list_q1D"] / data["n_minor"], data["w_list"] / (2 * np.pi),
                 ylim)

def print_info(f_ind: int, model: str = "nH-Haldane-test"):
    with open("%s/%s-%d.pkl" % (DATA_DIR, model, f_ind), "rb") as f:
        data = pickle.load(f)
    print(f_ind)
    print(data["info"])
    print(data["E_ref"])

if __name__ == '__main__':
    for model in MODELS:
        for f_ind in range(6):
            main_plot(f_ind, model)
            print_info(f_ind, model)
