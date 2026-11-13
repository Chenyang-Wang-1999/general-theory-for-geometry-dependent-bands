'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-11-13
Copyright © Department of Physics, Tsinghua University. All rights reserved

The plotting functions of the paper, copied from the scripts of the paper
repository.  Only the data source changed: the arrays come from the CSV files of
this package instead of the pickles of the numerical data set.  Figure sizes,
axis ranges, ticks, labels, markers, the analytic curves and the sequence of
`save` calls (the transparent png with the data, the pdf with the axes) are the
ones of the original scripts, so every file written here is the same file that
the paper repository writes into its ``Figures`` folder.

The original ``fig.savefig("Figures/name.png", ...)`` calls became
``save(fig, "name.png", ...)``, where ``save`` writes into the folder of the
panel; the original functions that draw several panels take an extra argument
that selects the ones wanted.

Sources:

    paper/HN-model/OBC-spectrum.py                    Fig. 3(e), 3(f)
    paper/winding-number/W-and-W-strip.py             Fig. S3(b), S3(d)
    paper/HN-model/quasi-1D-spectrum.py               Fig. S8(a-f)
    paper/HN-model/plot-OBC-dos.py                    Fig. S10
    paper/HN-model/eigenstate-comparison.py           Fig. S11(a-e)
    paper/Haldane-gainloss/plot-spectrum.py           Fig. S12(b, c, e, f)
    paper/long-range-coupling/spectrum.py             Fig. S13(a-f)
    paper/example-for-geom-dep-band/plot-spectrum.py  Fig. S18(c), S18(d)
    paper/HN-model/Amoeba-related-plot.py             Fig. S19(a), S19(b)
    paper/HN-model/BZ_spectrum.py                     Fig. S23(c)

The analytic helpers are copied from ``paper/HN-model/HN_model_common.py`` and
the solvable-model script; they need NumPy only.
'''

from cmath import sqrt

import numpy as np
import matplotlib.pyplot as plt

from colors import COLORS

CM = 1 / 2.54

#: coupling terms of the 2D HN model (HN_model_common.Default_Model_Params)
DEFAULT_MODEL_PARAMS = (1 + 1j, 1.5 + 1.2j, -1 + 1j, -1.2 - 0.5j)

#: colour of a filled SGBZ region (quasi-1D-spectrum.py)
FILL_COLOR = "#ff7f0eff"

#: panel limits of the gain-loss Haldane model (plot-spectrum.py)
HALDANE_XLIM = (-3.4, 4.7)
HALDANE_YLIM = (-0.53, 0.53)
HALDANE_XTICKS = (-2, 0, 2, 4)
HALDANE_YTICKS = (-0.4, -0.2, 0, 0.2, 0.4)

#: parameters of the solvable model (example-for-geom-dep-band/plot-spectrum.py)
SOLVABLE_PARAMETERS = {"a0": 0.8, "c0": 0.4 * np.exp(1j * np.pi / 12)}
SOLVABLE_ZOOM = ((1, 3), (-0.15, 0.15))

#: labels of the momenta of the three strips (quasi-1D-spectrum.py)
MOMENTUM_LABELS = {
    "x": (r"$\theta_x$ ($\pi$)", r"$\mu_x$", r"$\mu_y$"),
    "y": (r"$\theta_y$ ($\pi$)", r"$\mu_y$", r"$\mu_x$"),
    "11": (r"$\theta_{[11]}$ ($\pi$)", r"$\mu_{[11]}$", r"$\mu_y$"),
}


# --------------------------------------------------------------------------
# analytic helpers (paper/HN-model/HN_model_common.py)
# --------------------------------------------------------------------------


def continuous_sqrt(x: np.ndarray) -> np.ndarray:
    '''Continuously calculate the square root of an array of complex numbers.

    The first point takes the principal square root.  For every later point the
    sign of the previous root is kept when that root is closer to the
    difference of the two candidate square roots, is flipped when it is closer
    to their sum, and is reset to the negated root when the two distances are
    equal (the tie rule of the original implementation).
    '''
    root = np.sqrt(np.asarray(x, dtype=complex))
    if root.ndim != 1:
        raise ValueError("continuous_sqrt expects a 1-D array of complex numbers")
    if root.size < 2:
        return root

    dist_diff = root[:-1] - root[1:]
    dist_sum = root[:-1] + root[1:]
    n_diff = dist_diff.real * dist_diff.real + dist_diff.imag * dist_diff.imag
    n_sum = dist_sum.real * dist_sum.real + dist_sum.imag * dist_sum.imag
    keep = n_diff < n_sum
    flip = n_sum < n_diff
    tie = ~(keep | flip)

    last_tie = np.maximum.accumulate(np.where(tie, np.arange(flip.size), -1))
    flips = np.concatenate(([0], np.cumsum(flip)))
    odd = (flips[1:] - flips[last_tie + 1]) % 2 == 1
    negative = np.where(last_tie < 0, odd, ~odd)

    out = root.copy()
    out[1:] = np.where(negative, -root[1:], root[1:])
    return out


def get_11_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2):
    '''Boundary of the [11]-SGBZ.'''
    bound = 2 * continuous_sqrt(
            2 * sqrt(Jx1 * Jx2 * Jy1 * Jy2) * np.linspace(-1, 1, 500)
            + (Jx1 * Jx2)
            + (Jy1 * Jy2)
    )
    loop = np.hstack([bound, -bound[::-1], bound[0]])
    return bound, loop


def get_x_DGBZ_spectrum(Jx1, Jx2, Jy1, Jy2):
    '''Boundary of the [10]([01])-SGBZ.'''
    all_points = [
        2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
        2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
        -2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
        -2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
    ]
    return all_points, np.asarray(all_points + [all_points[0]])


def get_theoretical_spectrum(Lx: int, Ly: int):
    '''Analytical OBC spectrum of the solvable model.'''
    m = np.arange(1, Lx + 1)
    a0 = SOLVABLE_PARAMETERS["a0"]
    b0 = SOLVABLE_PARAMETERS["c0"]
    A = 2 * (1 + a0 * np.cos(m * np.pi / (Lx + 1)))
    B = 2 * (1 + b0 * np.cos(m * np.pi / (Lx + 1)))
    E_mesh = np.zeros((Lx, Ly), dtype=complex)
    sqrt_AB = np.sqrt(A * B)
    for n_ind in range(Ly):
        E_mesh[:, n_ind] = 2 * sqrt_AB * np.cos((n_ind + 1) * np.pi / (Ly + 1))
    return E_mesh.flatten()


# --------------------------------------------------------------------------
# Fig. 3(e), 3(f): OBC spectra  (paper/HN-model/OBC-spectrum.py)
# --------------------------------------------------------------------------


def obc_spectrum(eigv, stem, overlay, save):
    '''``plot_OBC_with_given_aspect`` / ``plot_11_OBC_spectrum``.

    The png holds the spectrum without axes (it is overlaid on the pdf, which
    carries the axes and the analytic SGBZ curve), exactly as in the paper.
    '''
    ax_size = 2.45
    fig_size = ax_size / 0.8
    fig = plt.figure(figsize=(fig_size * CM, fig_size * CM))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlabel("ReE")
    ax.set_ylabel("ImE")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-4, 4])

    dots = ax.plot(
        eigv.real, eigv.imag, '.',
        markersize=2, markeredgecolor='none'
    )

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    save(fig, stem + ".png", dpi=600)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    if overlay == "x":
        bound, loop = get_x_DGBZ_spectrum(*DEFAULT_MODEL_PARAMS)
        stripe = ax.plot(loop.real, loop.imag, 'r--')
        ax.legend(dots + stripe, ["OBC", "Stripe GBZ"])
    elif overlay == "11":
        bound, loop = get_11_DGBZ_spectrum(*DEFAULT_MODEL_PARAMS)
        ax.plot(loop.real, loop.imag, 'r--')

    dots[0].remove()
    save(fig, stem + ".pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. S3(b), S3(d): winding numbers  (paper/winding-number/W-and-W-strip.py)
# --------------------------------------------------------------------------


def winding(mu_1, w_q1d_over_L2, w, ylim, stem, save):
    '''``plot_general``.'''
    fig = plt.figure(figsize=(3.5 * CM, 4 * CM))
    ax = fig.gca()
    ax.set_position([0.25, 0.2, 0.75, 0.8])
    ax.plot(mu_1, w_q1d_over_L2, '-', label="$W_{\\mathrm{q1D}}/L_2$")
    ax.plot(mu_1, w, '--', label="$W$")
    ax.set_xlim([-1.1, 1.1])
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("Winding number")
    ax.set_ylim(ylim)
    ax.legend()
    save(fig, stem + ".pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. S8(a-c), S8(d-f): QMGBZ spectra and momenta
# (paper/HN-model/quasi-1D-spectrum.py)
# --------------------------------------------------------------------------


def qmgbz_spectrum(E, strip, stem, save):
    '''The spectrum panel of ``plot_quasi_1D_x`` / ``_y`` / ``_11``.'''
    Jx1, Jx2, Jy1, Jy2 = DEFAULT_MODEL_PARAMS

    fig = plt.figure(figsize=(4 * CM, 4 * CM))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    plt_E = ax.plot(E.real, E.imag, '.', markersize=2, markeredgecolor='none',
                    label="QMGBZ")
    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-4, 4])
    ax.minorticks_off()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    save(fig, stem + ".png", dpi=1200, transparent=True)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    if strip in ("x", "y"):
        all_points = np.asarray([
            2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
            2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
            -2 * sqrt(Jx1 * Jx2) - 2 * sqrt(Jy1 * Jy2),
            -2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
            2 * sqrt(Jx1 * Jx2) + 2 * sqrt(Jy1 * Jy2),
        ])
        ax.fill(all_points.real, all_points.imag, color=FILL_COLOR,
                linewidth=0, label="SGBZ")
    else:
        bound = 2 * continuous_sqrt(
            2 * sqrt(Jx1 * Jx2 * Jy1 * Jy2) * np.linspace(-1, 1, 500)
            + (Jx1 * Jx2)
            + (Jy1 * Jy2)
        )
        bound = np.hstack([bound, [0]])
        ax.fill(bound.real, bound.imag, color=FILL_COLOR, linewidth=0,
                label="SGBZ")
        ax.fill(-bound.real, -bound.imag, color=FILL_COLOR, linewidth=0)

    ax.legend(handlelength=1)
    plt_E[0].remove()
    save(fig, stem + ".pdf")
    plt.close(fig)


def major_momentum(theta_pi, mu, strip, stem, save):
    '''The beta panel of ``plot_quasi_1D_x`` / ``_y`` / ``_11``.'''
    theta_label, major_label, _ = MOMENTUM_LABELS[strip]
    Jx1, Jx2, Jy1, Jy2 = DEFAULT_MODEL_PARAMS

    fig = plt.figure(figsize=(4 * CM, 2 * CM))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.3, 1.3])
    ax.set_xlabel(theta_label)
    ax.set_ylabel(major_label)
    ax.set_xticks([-1, 0, 1])

    if strip == "x":
        r = sqrt(abs(Jx1 / Jx2))
    elif strip == "y":
        r = sqrt(abs(Jy1 / Jy2))
    else:
        r = sqrt(abs(Jx1 * Jy1 / Jx2 / Jy2))

    plt_beta = ax.plot(theta_pi, mu, '.', markersize=2, label="QMGBZ")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    save(fig, stem + ".png", dpi=1200, transparent=True)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.plot([-1, 1], [np.log(r).real, np.log(r).real], "--", label="SGBZ")
    ax.legend()
    plt_beta[0].remove()
    save(fig, stem + ".pdf")
    plt.close(fig)


def minor_momentum(theta_pi, mu, strip, stem, save):
    '''``plot_minor_beta_11`` / ``_x`` / ``_y``.'''
    theta_label, _, minor_label = MOMENTUM_LABELS[strip]
    Jx1, Jx2, Jy1, Jy2 = DEFAULT_MODEL_PARAMS
    theta = np.linspace(-np.pi, np.pi, 100)

    if strip == "11":
        beta_th = (np.sqrt(np.abs(Jx1 * Jy1 / Jx2 / Jy2))
                   * np.exp(1j * theta))
        r_th = np.sqrt(np.abs((Jx2 * beta_th + Jy1) / (Jx1 / beta_th + Jy2)))
    elif strip == "x":
        r_th = np.sqrt(np.abs(Jy1 / Jy2)) * np.ones_like(theta)
    else:
        r_th = np.sqrt(np.abs(Jx1 / Jx2)) * np.ones_like(theta)

    fig = plt.figure(figsize=(4 * CM, 2 * CM))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.minorticks_off()
    beta_plt = ax.plot(theta_pi, mu, '.', markersize=3,
                       markeredgecolor='none', label="QMGBZ")
    ax.set_xlabel(theta_label)
    ax.set_ylabel(minor_label)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.3, 1.3])

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    save(fig, stem + ".png", dpi=1200, transparent=True)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.plot(theta / np.pi, np.log(r_th), '--', label="SGBZ")
    ax.legend()
    beta_plt[0].remove()
    save(fig, stem + ".pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. S10: density of states  (paper/HN-model/plot-OBC-dos.py)
# --------------------------------------------------------------------------


def obc_dos(x_edges, y_edges, values, n1, n2, stem, save, with_axes=False):
    '''``plot_OBC_dos``; the csv stores the same cell edges as pcolormesh.'''
    fig = plt.figure(
        figsize=(2 * CM / 0.8, 1.5 * CM / 0.8)
    )
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_xticks(range(-4, 5, 4))
    ax.set_yticks([-2, 0, 2])
    ax.set_ylim(y_edges[0], y_edges[-1])

    pcl = ax.pcolormesh(
        x_edges, y_edges, values,
        vmin=0, vmax=n1 * n2 / 6)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    save(fig, stem + ".png", dpi=1200)

    if with_axes:
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        pcl.remove()
        ax.set_xlabel("Re($E$)")
        ax.set_ylabel("Im($E$)")
        save(fig, stem + "-axes.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. S11(a-c): eigenstate distributions
# (paper/HN-model/eigenstate-comparison.py)
# --------------------------------------------------------------------------


def eigenstate_distribution(x_mesh, y_mesh, curr_eigvec, approx_eigvec, shape,
                            save):
    '''``plot_eigenstate_distribution``; writes the seven original files.'''
    vmax = max(np.abs(curr_eigvec))

    full_view_xlim = [-10, shape[0] + 10]
    full_view_ylim = [-10, shape[1] + 10]
    full_view_x_ticks = np.linspace(0, shape[0], 5)
    full_view_y_ticks = [0, shape[1]]

    # plot selected eigenstate
    fig = plt.figure(figsize=(7.5 * CM, 2 * CM))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_xlim(full_view_xlim)
    ax.set_ylim(full_view_ylim)
    ax.set_xticks(full_view_x_ticks)
    ax.set_yticks(full_view_y_ticks)
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap="coolwarm",
        vmax=vmax,
        vmin=-vmax
    )
    save(fig, "eigenstate-distribution-%d-%d.pdf" % (shape[0], shape[1]))
    plt.colorbar(pcl)
    save(fig, "eigenstate-distribution-colorbar-%d-%d.pdf"
         % (shape[0], shape[1]))
    plt.close(fig)

    fig = plt.figure(figsize=(7.5 * CM, 2 * CM))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_xlim(full_view_xlim)
    ax.set_ylim(full_view_ylim)
    ax.set_xticks(full_view_x_ticks)
    ax.set_yticks(full_view_y_ticks)
    ax.pcolor(
        x_mesh,
        y_mesh,
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap="coolwarm",
        vmax=vmax,
        vmin=-vmax
    )
    save(fig, "approx-eigenstate-distribution-%d-%d.pdf"
         % (shape[0], shape[1]))
    plt.close(fig)

    fig = plt.figure(figsize=(7.5 * CM, 2 * CM))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_xlim(full_view_xlim)
    ax.set_ylim(full_view_ylim)
    ax.set_xticks(full_view_x_ticks)
    ax.set_yticks(full_view_y_ticks)
    diff_vector = curr_eigvec - approx_eigvec
    pcl = ax.pcolor(
        x_mesh,
        y_mesh,
        np.abs(diff_vector).reshape(shape, order='F'),
        cmap='Reds',
        vmax=max(np.abs(diff_vector)),
        vmin=0
    )
    save(fig, "diff-eigenstate-distribution-%d-%d.pdf" % (shape[0], shape[1]))
    plt.colorbar(pcl)
    save(fig, "diff-eigenstate-distribution-colorbar-%d-%d.pdf"
         % (shape[0], shape[1]))
    plt.close(fig)

    fig = plt.figure(figsize=(4.5 * CM, 4 * CM))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_ylim([-2, shape[1] + 2])
    ax.set_xlim([200, 240])
    ax.pcolor(
        x_mesh,
        y_mesh,
        (curr_eigvec.real).reshape(shape, order='F'),
        cmap="coolwarm",
        vmax=vmax,
        vmin=-vmax
    )
    save(fig, "eigenstate-distribution-zoomed-in-%d-%d.pdf"
         % (shape[0], shape[1]))
    plt.close(fig)

    fig = plt.figure(figsize=(4.5 * CM, 4 * CM))
    ax = fig.gca()
    ax.set_aspect(1)
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_ylim([-2, shape[1] + 2])
    ax.set_xlim([200, 240])
    ax.pcolor(
        x_mesh,
        y_mesh,
        (approx_eigvec.real).reshape(shape, order='F'),
        cmap="coolwarm",
        vmax=vmax,
        vmin=-vmax
    )
    save(fig, "approx-eigenstate-distribution-zoomed-in-%d-%d.pdf"
         % (shape[0], shape[1]))
    plt.close(fig)


def eigenstate_slices(x_mesh, y_mesh, curr_eigvec, approx_eigvec, shape, save,
                      only=None):
    '''``plot_eigenstate_zoomed_in``; ``only`` selects the upper or lower edge.'''
    vmax = max(np.abs(curr_eigvec))

    full_view_x_ticks = np.linspace(0, shape[0], 5)
    full_view_xlim = [full_view_x_ticks[1], full_view_x_ticks[-2]]
    full_view_x_ticks = full_view_x_ticks[1:-1]
    upper_slice_ylim = [shape[1] - 6, shape[1]]
    upper_slice_yticks = np.linspace(shape[1] - 6, shape[1], 2)
    lower_slice_ylim = [-1, 6]
    lower_slice_yticks = np.linspace(0, 6, 2)

    edges = [("lower", lower_slice_ylim, lower_slice_yticks)]
    if only != "lower":
        edges = [("upper", upper_slice_ylim, upper_slice_yticks)]
        if only is None:
            edges.append(("lower", lower_slice_ylim, lower_slice_yticks))

    for name, ylim, yticks in edges:
        for values, prefix in ((curr_eigvec, "eigenstate"),
                               (approx_eigvec, "approx-eigenstate")):
            fig = plt.figure(figsize=(7.5 * CM, 1 * CM))
            ax = fig.gca()
            ax.set_xlabel("n1")
            ax.set_ylabel("n2")
            ax.set_ylim(ylim)
            ax.set_yticks(yticks)
            ax.set_xlim(full_view_xlim)
            ax.set_xticks(full_view_x_ticks)
            ax.pcolor(
                x_mesh,
                y_mesh,
                (values.real).reshape(shape, order='F'),
                cmap="coolwarm",
                vmax=vmax,
                vmin=-vmax
            )
            save(fig, "%s-distribution-sliced-%s-%d-%d.pdf"
                 % (prefix, name, shape[0], shape[1]))
            plt.close(fig)


def eigenstate_difference(n1, columns, save, which=None):
    '''``plot_eigenstate_difference``; ``which`` selects the upper/lower panel.'''
    panels = [("lower", "eigenstate-difference-lower.pdf"),
              ("upper", "eigenstate-difference-upper.pdf")]
    if which is not None:
        panels = [panel for panel in panels if panel[0] == which]

    for _, name in panels:
        fig = plt.figure(figsize=(4 * CM, 4 * CM))
        ax = fig.gca()
        ax.set_xlabel("n1")
        ax.set_xlim([0, 320])
        ax.set_xticks(np.linspace(0, 320, 5))
        ax.set_ylim([0, 2])
        for values, style, label in zip(
                columns, ("--", "-", ":"),
                ("$L_2 = 30$", "$L_2 = 40$", "$L_2 = 50$")):
            ax.plot(n1, values, style, label=label)
        ax.legend()
        save(fig, name)
        plt.close(fig)


# --------------------------------------------------------------------------
# Fig. S12(b, c, e, f): gain-loss Haldane model
# (paper/Haldane-gainloss/plot-spectrum.py)
# --------------------------------------------------------------------------


def haldane_spectrum(eigv, stem, save):
    '''``plot_SGBZ_spectrum`` and ``plot_OBC_spectrum`` (same body).'''
    wx = 3.929   # cm
    wy = 3.1712  # cm

    fig = plt.figure(figsize=(
        wx * CM / 0.8,
        wy * CM / 0.8,
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])

    ax.set_xlim(HALDANE_XLIM)
    ax.set_ylim(HALDANE_YLIM)
    ax.set_xticks(HALDANE_XTICKS)
    ax.set_yticks(HALDANE_YTICKS)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    data_plt = ax.plot(eigv.real, eigv.imag, '.', markeredgecolor="none")
    save(fig, stem + ".png", dpi=1200, transparent=True)

    data_plt[0].remove()

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.set_xlabel("Re($E$)")
    ax.set_ylabel("Im($E$)")

    save(fig, stem + ".pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. S13(a-f): long-range hopping model
# (paper/long-range-coupling/spectrum.py)
# --------------------------------------------------------------------------


def long_range_spectrum(eigv, stem, save):
    '''``plot_spectrum``.'''
    fig = plt.figure(figsize=(
        4 / 0.8 * CM, 4 / 0.8 * CM
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlabel(r"Re($E$)")
    ax.set_ylabel(r"Im($E$)")
    ax.set_ylim([-0.42, 0.42])
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xlim([-2.6, 6.5])
    ax.set_xticks([-2, 0, 2, 4, 6])
    pl = ax.plot(eigv.real, eigv.imag, '.', markeredgecolor="none")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    save(fig, stem + ".png", dpi=1200)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    pl[0].remove()

    save(fig, stem + ".pdf")
    plt.close(fig)


def long_range_mu(E_real, mu, stem, save):
    '''``plot_mux``.'''
    fig = plt.figure(figsize=(
        4 / 0.8 * CM, 2 / 0.8 * CM
    ))
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])
    ax.set_xlabel(r"Re($E$)")
    ax.set_ylabel(r"$\mu_x$")
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_xlim([-2.6, 6.5])
    ax.set_xticks([-2, 0, 2, 4, 6])
    pl = ax.plot(E_real, mu, '.', markeredgecolor="none")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    save(fig, stem + ".png", dpi=1200)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    pl[0].remove()

    ax.plot([-2.6, 6.5], [1, 1], "k--")

    save(fig, stem + ".pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. S18(c), S18(d): solvable model
# (paper/example-for-geom-dep-band/plot-spectrum.py)
# --------------------------------------------------------------------------


def _spectrum_region_figure(E_num, boundary, E_ana=None, size=(4 * CM / 0.8,
                                                              3 * CM / 0.8),
                            zoom=False, num_size=1):
    x_range, y_range = SOLVABLE_ZOOM
    if zoom:
        width = 4 * CM / 0.8 * (x_range[1] - x_range[0]) / (6.6 * 2) * 5
        height = 3 * CM / 0.8 * (y_range[1] - y_range[0]) / (0.9 * 2) * 5
        fig = plt.figure(figsize=(width, height))
    else:
        fig = plt.figure(figsize=size)
    ax = fig.gca()
    ax.set_position([0.2, 0.2, 0.8, 0.8])

    if zoom:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xticks([])
        ax.set_yticks([])
        ana_size = 2.4
        num_size = 3 if num_size == 1 else num_size
        ana_width = 0.3
    else:
        ax.set_xlim([-6.6, 6.6])
        ax.set_ylim([-0.9, 0.9])
        ax.set_xticks(range(-8, 9, 4))
        ax.set_yticks(np.arange(-0.8, 0.9, 0.4))
        ana_size = 0.8
        ana_width = 0.1

    plotted = []
    if E_ana is not None:
        plotted += ax.plot(E_ana.real, E_ana.imag, 'o', label="Analytical",
                           color=COLORS["gl"], markeredgewidth=ana_width,
                           markersize=ana_size, markerfacecolor="none")
    plotted += ax.plot(E_num.real, E_num.imag, '.', label="Numerical",
                       color=COLORS["ol"], markersize=num_size,
                       markeredgecolor="none")
    return fig, ax, plotted


def square_region(E_num, boundary, stem, save):
    '''``plot_square_region``.'''
    E_ana = get_theoretical_spectrum(57, 57)
    fig, ax, plotted = _spectrum_region_figure(E_num, boundary, E_ana)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    save(fig, stem + ".png", dpi=1200, transparent=True)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.fill(boundary.real, boundary.imag, color=COLORS["bf"], label="Amoebic")
    ax.legend()

    for line in plotted:
        line.remove()

    x_range, y_range = SOLVABLE_ZOOM
    ax.plot(
        [x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]],
        [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]],
        "--",
        color=COLORS["rl"]
    )

    save(fig, stem + ".pdf")
    plt.close(fig)


def square_region_zoomed_in(E_num, boundary, stem, save):
    '''``plot_square_region_zoomed_in``.'''
    E_ana = get_theoretical_spectrum(57, 57)
    fig, ax, plotted = _spectrum_region_figure(E_num, boundary, E_ana,
                                               zoom=True)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    save(fig, stem + ".png", dpi=1200, transparent=True)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.fill(boundary.real, boundary.imag, color=COLORS["bf"], label="Amoebic")
    ax.legend()

    for line in plotted:
        line.remove()

    save(fig, stem + ".pdf")
    plt.close(fig)


def diamond_region(E_num, boundary, stem, save):
    '''``plot_diamond_region`` (the original uses a slightly larger marker).'''
    fig, ax, plotted = _spectrum_region_figure(E_num, boundary, num_size=1.5)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    save(fig, stem + ".png", dpi=1200, transparent=True)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    ax.fill(boundary.real, boundary.imag, color=COLORS["bf"], label="Amoebic")
    ax.legend()

    for line in plotted:
        line.remove()

    save(fig, stem + ".pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. S19(a), S19(b): amoebas
# (paper/HN-model/Amoeba-related-plot.py::plot_Amoeba)
# --------------------------------------------------------------------------


def amoeba(amoeba_layers, stem, save):
    '''``plot_Amoeba``.

    The original pcolor paints one degenerate mesh per entry and branch; the CSV
    files store the two envelopes of each mesh together with the axis that the
    pcolor call puts horizontally, and the fills below reproduce the same
    region.  The original script saves the pdf before painting the bands, which
    leaves it empty; here both files are written after the bands are painted.
    '''
    fig = plt.figure(figsize=(
        3 * CM, 3 * CM
    ))
    ax = fig.gca()
    ax.set_position([
        0.2, 0.2, 0.75, 0.75
    ])
    ax.set_xlabel("mux")
    ax.set_ylabel("muy")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])

    for axis, mu, mu_min, mu_max in amoeba_layers:
        if axis == "x":
            ax.fill_between(mu, mu_min, mu_max, color="grey", linewidth=0)
        else:
            ax.fill_betweenx(mu, mu_min, mu_max, color="grey", linewidth=0)

    save(fig, stem + ".pdf")
    save(fig, stem + ".png", dpi=4800)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. S23(c): OBC spectrum of the uniform-band model
# (paper/HN-model/BZ_spectrum.py::plot_OBC_spectrum)
# --------------------------------------------------------------------------


def bz_obc_spectrum(eigv, stem, save):
    '''``plot_OBC_spectrum``.'''
    plt.plot(eigv.real, eigv.imag, '.', label="OBC")
    bound, loop = get_x_DGBZ_spectrum(1, 2, 0.5, 1)
    plt.plot(loop.real, loop.imag, label="SGBZ")
    plt.legend()
    save(plt.gcf(), stem + ".png")
    plt.close(plt.gcf())
