'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-11-13
Copyright © Department of Physics, Tsinghua University. All rights reserved

Write the numerical data behind a figure panel to a CSV file next to it.

The plotting scripts in this repository only draw panels whose data are loaded
from the published pickles (the "data -> script" arrows of the README flow
charts); panels that are computed analytically inside a plotting script are
reproduced from the formulas in that script and get no CSV.  Every exported
panel calls one of the two writers below once, where the panel is saved, so
that the figure and its text data always come from the same arrays:

    import figure_data as fd
    fd.save_csv("Figures/quasi-1D-x-N_15-E", "Fig. S8(a) | ...",
                Re_E=E.real, Im_E=E.imag)
    fd.save_grid("Figures/paper-HN-OBC-2-DOS-n1-320-n2-40", x_edges, y_edges,
                 dos, "Fig. S10 | ...")

``save_csv`` writes a comma separated table: one ``#`` line describing the
panel, one header row with the column names, then one row per point.
``save_grid`` writes a regular grid as a matrix: the same ``#`` line, then a
header row holding the column coordinates (prefixed by the name of the row
axis), then one row per line coordinate and cell values.  Numbers are written
with ``%.17g``, so a float64 survives a round trip exactly; ``nan`` and ``inf``
are written as such.  The files are ASCII with LF line endings, no timestamps,
no locale dependent formatting, so re-running a script reproduces them byte for
byte.  The CSV files are not tracked by git (see .gitignore): they are produced
by the plotting scripts together with the figures.

Nothing in this module draws anything, and it never modifies the arrays it is
given, so adding these calls does not change any figure.
'''

import os

import numpy as np

#: significant digits used for every number written to a file
DIGITS = 17

#: figure extensions that ``save_csv``/``save_grid`` strip from their argument.
#: Only these are stripped: a file stem such as ``xy-0.125000-320-40-12800``
#: contains dots that are part of the name, not an extension.
FIGURE_SUFFIXES = (".pdf", ".png", ".svg", ".eps", ".jpg", ".jpeg", ".tif", ".tiff")


def _figure_stem(stem):
    '''Return the path of a figure without its image extension.'''
    stem = str(stem)
    for suffix in FIGURE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _target(stem):
    '''Return the CSV path that belongs to a figure, creating its folder.'''
    path = _figure_stem(stem) + ".csv"
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    return path


def _format(values):
    '''Format a one dimensional array of numbers as a list of strings.'''
    fmt = "%%.%dg" % DIGITS
    return [fmt % value for value in np.asarray(values).tolist()]


def _comment_line(parts):
    '''Build the single ``#`` line that opens every CSV file.'''
    text = " | ".join(str(part).strip() for part in parts if str(part).strip())
    text = " ".join(text.split())
    return "# " + text


def _write_columns(stem, columns, comment):
    '''Write already formatted columns (list of (name, list of strings)).'''
    target = _target(stem)
    with open(target, "w", encoding="ascii", newline="\n") as fp:
        fp.write(_comment_line(["data of " + _figure_stem(stem) + ".pdf",
                                comment]) + "\n")
        fp.write(",".join(name for name, _ in columns) + "\n")
        rows = zip(*[values for _, values in columns])
        fp.write("".join(",".join(row) + "\n" for row in rows))
    return target


def save_csv(stem, comment="", **columns):
    '''Write the plotted data of one figure as a comma separated table.

    Parameters
    ----------
    stem : str
        Path of the figure without extension, for example
        ``"Figures/quasi-1D-x-N_15-E"``; the file
        ``Figures/quasi-1D-x-N_15-E.csv`` is written next to the figure.  A
        path with an extension (``...-E.pdf``) is accepted as well.
    comment : str, optional
        Free-form description of the panel and of the quantities; it becomes
        the single ``#`` line above the header row (newlines are collapsed).
    **columns
        The columns of the table, in the order they are written.  A column is
        a one dimensional array, or a scalar that is repeated in every row.
        Complex arrays are split into ``<name>_re`` and ``<name>_im``;
        string arrays are written as they are (for a grouping column such as
        ``series``).  All columns must have the same length.

    Returns
    -------
    str
        Path of the written file.
    '''
    if not columns:
        raise ValueError("save_csv() needs at least one column")

    lengths = {np.asarray(value).size for value in columns.values()
               if np.asarray(value).ndim > 0}
    if len(lengths) > 1:
        raise ValueError("columns of %r have different lengths: %r"
                         % (stem, sorted(lengths)))
    n_rows = lengths.pop() if lengths else 1

    formatted = []
    for name, value in columns.items():
        array = np.asarray(value)
        if array.ndim == 0:
            array = np.full(n_rows, array)
        if array.ndim != 1:
            raise ValueError("column %r of %r is not one dimensional"
                             % (name, stem))
        if array.size != n_rows:
            raise ValueError("column %r of %r has %d rows, expected %d"
                             % (name, stem, array.size, n_rows))
        if array.dtype.kind in "US":
            formatted.append((name, list(array.astype(str))))
        elif np.iscomplexobj(array):
            formatted.append((name + "_re", _format(array.real)))
            formatted.append((name + "_im", _format(array.imag)))
        else:
            formatted.append((name, _format(array)))

    return _write_columns(stem, formatted, comment)


def save_grid(stem, x_edges, y_edges, values, comment="",
              x_name="x", y_name="y", value_name="value"):
    '''Write a regular grid as a matrix: header row plus one row per line.

    Parameters
    ----------
    stem : str
        Path of the figure without extension (see ``save_csv``).
    x_edges, y_edges : array_like
        The cell edges, exactly as ``pcolormesh`` takes them: one value more
        than the number of columns/rows of ``values``.
    values : ndarray
        Two dimensional array with shape ``(len(y_edges) - 1, len(x_edges) - 1)``.
    comment : str, optional
        Free-form description; it becomes the single ``#`` line above the
        header row (newlines are collapsed).
    x_name, y_name, value_name : str, optional
        Names used in the header row, the first column and the comment line.

    Returns
    -------
    str
        Path of the written file.
    '''
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values of %r must be two dimensional" % (stem,))
    n_rows, n_columns = values.shape

    x_edges = np.asarray(x_edges, dtype=float).ravel()
    y_edges = np.asarray(y_edges, dtype=float).ravel()
    if x_edges.size != n_columns + 1 or y_edges.size != n_rows + 1:
        raise ValueError(
            "%r: grid is %d x %d but the edges have %d and %d values"
            % (stem, n_rows, n_columns, y_edges.size, x_edges.size))

    fmt = "%%.%dg" % DIGITS
    target = _target(stem)
    with open(target, "w", encoding="ascii", newline="\n") as fp:
        fp.write(_comment_line([
            "data of " + _figure_stem(stem) + ".pdf",
            "line 2: %s edges, line 3: %s edges, later lines: %s of the cells"
            % (x_name, y_name, value_name),
            comment,
        ]) + "\n")
        fp.write(x_name + "," +
                 ",".join(fmt % value for value in x_edges) + "\n")
        fp.write(y_name + "," +
                 ",".join(fmt % value for value in y_edges) + "\n")
        for index in range(n_rows):
            fp.write(",".join(fmt % value for value in values[index]) + "\n")
    return target
