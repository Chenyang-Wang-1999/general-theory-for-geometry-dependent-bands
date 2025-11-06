'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-01-02
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import matplotlib.pyplot as plt
import pickle
from math import pi


def plot_minimal_nontrivial():
    # fname = "data/minimal-nontrivial-test"
    # fname = "data/2D-SSH-test"
    # fname = "data/nH-Haldane-test"
    fname = "data/2D-HN-test-alpha2"
    for file_ind in range(1):
        with open("%s-%d.pkl" % (fname, file_ind), "rb") as f:
            data = pickle.load(f)
        print(data["info"])
        print(data["E_ref"])
        fig = plt.figure(1)
        plt.plot(data["r_list"], data["w_list"] * data["n_minor"] / (2 * pi), '.-', label="total winding")
        # plt.title("Total winding number")
        # fig = plt.figure(2)
        plt.plot(data["r_list"], data["w_list_q1D"], '.-', label="quasi-1D winding")
        plt.legend()
        # plt.title("Quasi-1D winding number")
        plt.show()


if __name__ == "__main__":
    plot_minimal_nontrivial()