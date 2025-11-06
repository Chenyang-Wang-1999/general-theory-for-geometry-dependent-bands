'''
author:        Wang Chenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-10-21
Copyright © Department of Physics, Tsinghua University. All rights reserved

Plot schematic diagram of eigenstate composition
'''

import matplotlib.pyplot as plt
import numpy as np


def plot_schematic():
    # Plot schematic diagram by sin function
    kappa = 1
    k = 5 * 2 * np.pi
    phi = 0.3 * 2 * np.pi
    x = np.linspace(0, 1, 1000)
    f = np.exp(kappa * x) * np.sin(k * x + phi)

    kappa1 = -20
    kappa2 = 20

    f1 = f[0] * np.exp(kappa1 * x)
    f2 = f[-1] * np.exp(kappa2 * (x - 1))

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, np.abs(f))
    ax.plot([x[0], x[-1]], [0, 0])
    fig.savefig("Figures/schematic-non-Bloch.pdf")

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, np.abs(f1))
    ax.plot([x[0], x[-1]], [0, 0])
    fig.savefig("Figures/schematic-f1.pdf")

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, np.abs(f2))
    ax.plot([x[0], x[-1]], [0, 0])
    fig.savefig("Figures/schematic-f2.pdf")

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, np.abs(f - f1 - f2))
    ax.plot([x[0], x[-1]], [0, 0])
    fig.savefig("Figures/schematic-OBC.pdf")
    plt.show()


if __name__ == '__main__':
    plot_schematic()
