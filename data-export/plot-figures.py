#!/usr/bin/env python3
'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-11-13
Copyright © Department of Physics, Tsinghua University. All rights reserved

Redraw the figures of "General theory for geometry-dependent non-Hermitian bands"
from the CSV files of this package.

    python plot-figures.py                    # every panel folder
    python plot-figures.py Fig.3e Fig.S8d     # only these folders
    python plot-figures.py --list             # what the package contains

The drawing functions are the ones of the paper repository (see
``code/paper_plots.py``); only the data source changed (CSV instead of pickle).
The images are written **into the folder of the panel**, with the same names the
paper repository gives them, so ``Fig.3e/`` ends up holding the csv files and
the images ``xy-0.125000-320-40-12800.png`` / ``.pdf`` that ``paper/Figures``
holds.  Some panels give several images, which is why the folders are allowed to
hold more than one figure; a few scripts save a transparent png with the data
and a pdf with the axes, and the manuscript overlays the two, so both are
written here exactly as the original script does.

Only NumPy and Matplotlib are needed.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))

import paper_plots as P                               # noqa: E402  (needs the path)

STYLE = os.path.join(HERE, "code", "paper_plot.mplstyle")

#: the panel folders, in the order they are drawn
PANELS = [
    "Fig.3e", "Fig.3f",
    "Fig.S3b", "Fig.S3d",
    "Fig.S8a", "Fig.S8b", "Fig.S8c", "Fig.S8d", "Fig.S8e", "Fig.S8f",
    "Fig.S10",
    "Fig.S11a", "Fig.S11b", "Fig.S11c", "Fig.S11d", "Fig.S11e",
    "Fig.S12b", "Fig.S12c", "Fig.S12e", "Fig.S12f",
    "Fig.S13a", "Fig.S13b", "Fig.S13c", "Fig.S13d", "Fig.S13e", "Fig.S13f",
    "Fig.S18c", "Fig.S18d",
    "Fig.S19a", "Fig.S19b",
    "Fig.S23c",
]

DESCRIPTIONS = {
    "Fig.3e": "OBC spectra, parallel region with compatible SGBZs",
    "Fig.3f": "OBC spectra, parallel region with incompatible SGBZs",
    "Fig.S3b": "W and W_q1D/L2 in the 2D Hatano-Nelson model",
    "Fig.S3d": "W and W_q1D/L2 in the non-Hermitian Haldane model",
    "Fig.S8a": "QMGBZ spectrum of the [10]-strip",
    "Fig.S8b": "QMGBZ spectrum of the [01]-strip",
    "Fig.S8c": "QMGBZ spectrum of the [11]-strip",
    "Fig.S8d": "Major and minor momenta of the [10]-strip",
    "Fig.S8e": "Major and minor momenta of the [01]-strip",
    "Fig.S8f": "Major and minor momenta of the [11]-strip",
    "Fig.S10": "Density of states of the 2D HN model, one image per size",
    "Fig.S11a": "Eigenstate and SGBZ mode",
    "Fig.S11b": "Upper edge of the eigenstate and the SGBZ mode",
    "Fig.S11c": "Lower edge of the eigenstate and the SGBZ mode",
    "Fig.S11d": "Residual R_u for different strip widths",
    "Fig.S11e": "Residual R_l for different strip widths",
    "Fig.S12b": "SGBZ spectra along the a1, a2 and x axes",
    "Fig.S12c": "SGBZ spectrum along the y axis",
    "Fig.S12e": "OBC spectrum in the rhombic geometry",
    "Fig.S12f": "OBC spectrum in the rectangular geometry",
    "Fig.S13a": "OBC spectrum with Lx = Ly = 77",
    "Fig.S13b": "OBC spectrum with Lx = 300, Ly = 20",
    "Fig.S13c": "OBC spectrum with Lx = 20, Ly = 300",
    "Fig.S13d": "mu_x on the x-SGBZ and on the y-SGBZ",
    "Fig.S13e": "x-SGBZ spectrum",
    "Fig.S13f": "y-SGBZ spectrum",
    "Fig.S18c": "Numerical spectrum, amoebic band and zoom, square geometry",
    "Fig.S18d": "Numerical spectrum and amoebic band, diamond geometry",
    "Fig.S19a": "Amoeba for E = 1 + i",
    "Fig.S19b": "Amoeba for E = 3 + 3i",
    "Fig.S23c": "Numerical OBC spectrum of the uniform-band model",
}


# --------------------------------------------------------------------------
# reading the csv files
# --------------------------------------------------------------------------


def read_table(path):
    '''Read a table CSV file and return ``{column name: 1-D array}``.'''
    with open(path, encoding="ascii") as fp:
        lines = [line.rstrip("\n") for line in fp if line.strip()]
    if not lines[0].startswith("#"):
        raise ValueError("%s: line 1 should be a comment" % path)
    names = [name.strip() for name in lines[1].split(",")]
    columns = {name: [] for name in names}
    for line in lines[2:]:
        for name, value in zip(names, line.split(",")):
            columns[name].append(value)
    data = {}
    for name, values in columns.items():
        try:
            data[name] = np.array([float(value) for value in values])
        except ValueError:
            data[name] = np.array(values)
    return data


def read_grid(path):
    '''Read a matrix CSV file (Fig. S10) as ``(x_edges, y_edges, values)``.

    Line 2 holds the x cell edges and line 3 the y cell edges (each after the
    name of its axis); every later line holds the cells of one row.
    '''
    with open(path, encoding="ascii") as fp:
        lines = [line.rstrip("\n") for line in fp if line.strip()]
    x_edges = np.array([float(value) for value in lines[1].split(",")[1:]])
    y_edges = np.array([float(value) for value in lines[2].split(",")[1:]])
    values = np.array([[float(value) for value in line.split(",")]
                       for line in lines[3:]])
    return x_edges, y_edges, values


def read_complex(path):
    '''The complex quantity stored in the ``Re_E``/``Im_E`` columns.'''
    data = read_table(path)
    return data["Re_E"] + 1j * data["Im_E"]


def folder_files(folder):
    return [os.path.join(folder, name)
            for name in sorted(os.listdir(folder)) if name.endswith(".csv")]


def stem_of(path):
    return os.path.basename(path)[:-len(".csv")]


def saver(folder):
    '''The ``save`` callback of the plotting functions: into the panel folder.'''
    def save(fig, name, **kwargs):
        fig.savefig(os.path.join(folder, name), **kwargs)
    return save


def aspect_of(path):
    parts = stem_of(path).split("-")
    return int(parts[2]), int(parts[3])


def aspect_order(path):
    n1, n2 = aspect_of(path)
    return -n1 / n2


def region_data(path):
    '''The numerical spectrum and the amoebic boundary of one region panel.'''
    data = read_table(path)
    numeric = data["series"] == "numerical"
    amoebic = data["series"] == "amoebic"
    return (data["Re_E"][numeric] + 1j * data["Im_E"][numeric],
            data["Re_E"][amoebic] + 1j * data["Im_E"][amoebic])


def eigenstate_data(folder):
    '''The mesh and the two eigenstate columns of Fig. S11.'''
    data = read_table(folder_files(folder)[0])
    shape = (int(data["n1"].max()) + 1, int(data["n2"].max()) + 1)
    x_mesh = data["x"].reshape(shape, order="F")
    y_mesh = data["y"].reshape(shape, order="F")
    return (x_mesh, y_mesh, data["Re_psi"], data["Re_psi_approx"], shape)


# --------------------------------------------------------------------------
# the panels
# --------------------------------------------------------------------------


def run_fig3e(folder):
    save = saver(folder)
    for path in sorted(folder_files(folder), key=aspect_order):
        P.obc_spectrum(read_complex(path), stem_of(path), "x", save)


def run_fig3f(folder):
    save = saver(folder)
    overlays = {"0.125000": "11", "1.000000": None, "8.000000": "x"}
    for path in sorted(folder_files(folder), key=aspect_order):
        aspect = stem_of(path).split("-")[1]
        P.obc_spectrum(read_complex(path), stem_of(path), overlays[aspect],
                       save)


def run_figS3(folder, ylim):
    save = saver(folder)
    for path in folder_files(folder):
        data = read_table(path)
        P.winding(data["mu_1"], data["w_q1d_over_L2"], data["w"], ylim,
                  stem_of(path), save)


def run_figS8_spectrum(folder, strip):
    save = saver(folder)
    path = folder_files(folder)[0]
    P.qmgbz_spectrum(read_complex(path), strip, stem_of(path), save)


def run_figS8_momenta(folder, strip):
    save = saver(folder)
    beta, minor = folder_files(folder)
    for path, function in ((beta, P.major_momentum),
                           (minor, P.minor_momentum)):
        data = read_table(path)
        function(data["theta_pi"], data[list(data)[1]], strip,
                 stem_of(path), save)


def run_figS10(folder):
    save = saver(folder)
    entries = []
    for path in folder_files(folder):
        parts = stem_of(path).split("-")
        entries.append((int(parts[6]), int(parts[8]), path))
    entries.sort(key=lambda entry: (entry[0] // entry[1], entry[0]))
    for n1, n2, path in entries:
        x, y, values = read_grid(path)
        P.obc_dos(x, y, values, n1, n2, stem_of(path), save)


def run_figS11a(folder):
    save = saver(folder)
    P.eigenstate_distribution(*eigenstate_data(folder), save)


def run_figS11b(folder):
    save = saver(folder)
    P.eigenstate_slices(*eigenstate_data(folder), save, only="upper")


def run_figS11c(folder):
    save = saver(folder)
    P.eigenstate_slices(*eigenstate_data(folder), save, only="lower")


def run_figS11d(folder):
    save = saver(folder)
    data = read_table(folder_files(folder)[0])
    P.eigenstate_difference(
        data["n1"], [data["L2_30"], data["L2_40"], data["L2_50"]], save,
        which="upper")


def run_figS11e(folder):
    save = saver(folder)
    data = read_table(folder_files(folder)[0])
    P.eigenstate_difference(
        data["n1"], [data["L2_30"], data["L2_40"], data["L2_50"]], save,
        which="lower")


def run_figS12(folder):
    save = saver(folder)
    for path in folder_files(folder):
        P.haldane_spectrum(read_complex(path), stem_of(path), save)


def run_figS13_spectrum(folder):
    save = saver(folder)
    for path in folder_files(folder):
        P.long_range_spectrum(read_complex(path), stem_of(path), save)


def run_figS13_mu(folder):
    save = saver(folder)
    for path in folder_files(folder):
        data = read_table(path)
        P.long_range_mu(data["Re_E"], data["mu"], stem_of(path), save)


def run_figS18c(folder):
    save = saver(folder)
    for path in folder_files(folder):
        E_num, boundary = region_data(path)
        if "zoomed-in" in path:
            P.square_region_zoomed_in(E_num, boundary, stem_of(path), save)
        else:
            P.square_region(E_num, boundary, stem_of(path), save)


def run_figS18d(folder):
    save = saver(folder)
    path = folder_files(folder)[0]
    E_num, boundary = region_data(path)
    P.diamond_region(E_num, boundary, stem_of(path), save)


def run_figS19(folder):
    save = saver(folder)
    data = read_table(folder_files(folder)[0])
    layers = []
    for entry, branch in sorted({(int(entry), int(branch)) for entry, branch in
                                 zip(data["entry"], data["branch"])}):
        selected = (data["entry"] == entry) & (data["branch"] == branch)
        order = np.argsort(data["mu"][selected])
        layers.append((data["axis"][selected][0],
                       data["mu"][selected][order],
                       data["mu_min"][selected][order],
                       data["mu_max"][selected][order]))
    P.amoeba(layers, stem_of(folder_files(folder)[0]), save)


def run_figS23c(folder):
    # BZ_spectrum.py applies the paper style only in plot_spectra, so this
    # panel is drawn with the Matplotlib defaults, as in the original run
    save = saver(folder)
    path = folder_files(folder)[0]
    with plt.style.context("default"):
        P.bz_obc_spectrum(read_complex(path), stem_of(path), save)


BUILDERS = {
    "Fig.3e": run_fig3e,
    "Fig.3f": run_fig3f,
    "Fig.S3b": lambda folder: run_figS3(folder, (-0.5, 0.5)),
    "Fig.S3d": lambda folder: run_figS3(folder, (-2, 2)),
    "Fig.S8a": lambda folder: run_figS8_spectrum(folder, "x"),
    "Fig.S8b": lambda folder: run_figS8_spectrum(folder, "y"),
    "Fig.S8c": lambda folder: run_figS8_spectrum(folder, "11"),
    "Fig.S8d": lambda folder: run_figS8_momenta(folder, "x"),
    "Fig.S8e": lambda folder: run_figS8_momenta(folder, "y"),
    "Fig.S8f": lambda folder: run_figS8_momenta(folder, "11"),
    "Fig.S10": run_figS10,
    "Fig.S11a": run_figS11a,
    "Fig.S11b": run_figS11b,
    "Fig.S11c": run_figS11c,
    "Fig.S11d": run_figS11d,
    "Fig.S11e": run_figS11e,
    "Fig.S12b": run_figS12,
    "Fig.S12c": run_figS12,
    "Fig.S12e": run_figS12,
    "Fig.S12f": run_figS12,
    "Fig.S13a": run_figS13_spectrum,
    "Fig.S13b": run_figS13_spectrum,
    "Fig.S13c": run_figS13_spectrum,
    "Fig.S13d": run_figS13_mu,
    "Fig.S13e": run_figS13_spectrum,
    "Fig.S13f": run_figS13_spectrum,
    "Fig.S18c": run_figS18c,
    "Fig.S18d": run_figS18d,
    "Fig.S19a": run_figS19,
    "Fig.S19b": run_figS19,
    "Fig.S23c": run_figS23c,
}


# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Redraw the figures from the CSV data of this package.")
    parser.add_argument("panels", nargs="*", metavar="PANEL",
                        help="panel folders to draw (default: all of them), for "
                             "example Fig.3e or Fig.S8d")
    parser.add_argument("--list", action="store_true",
                        help="list the panel folders and exit")
    args = parser.parse_args(argv)

    if args.list:
        width = max(len(name) for name in PANELS)
        for name in PANELS:
            print("%-*s  %-54s %d csv file(s)"
                  % (width, name, DESCRIPTIONS[name],
                     len(folder_files(os.path.join(HERE, name)))))
        return 0

    names = args.panels or PANELS
    unknown = [name for name in names if name not in BUILDERS]
    if unknown:
        parser.error("unknown panel(s): %s" % ", ".join(unknown))

    plt.style.use(STYLE)
    # the paper fonts may be missing; fall back to the Matplotlib defaults
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

    for name in names:
        folder = os.path.join(HERE, name)
        BUILDERS[name](folder)
        print("wrote the figures of %s" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
