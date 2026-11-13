# Numerical data of the figures

This folder contains the numerical data behind the figure panels of
*General theory for geometry-dependent non-Hermitian bands* that come from a
numerical calculation, together with the code that draws the panels from that
data.  It is meant to let a reader look up, replot or re-analyse the numbers of
a panel without unpacking the (far larger) pickle files of the numerical runs.

```
data-export/
├── Fig.3e/ … Fig.S23c/    one folder per caption panel: its CSV data and the
│                          figures that the paper scripts draw from it
├── code/                  the plotting functions of the paper, plus the style
├── plot-figures.py        redraws the figures from the CSV files
└── README.md
```

The folder names are the panel labels of the captions in the main text
(`sn-main.tex`) and in the Supplementary Information (`sn-supp.tex`); a panel
that is made of several pieces, such as the three aspect ratios of Fig. 3(e),
has one CSV file and one image per piece.

The folders are filled by `export-figure-csv.py` in the repository: it runs the
plotting scripts of `paper/` that write the CSV files (only when they are not
there yet) and copies each file into the folder of its panel.

Every folder also holds the figures of that panel with the file names that the
paper repository gives them (`paper/Figures/…`): the same script saves a
transparent png with the data and a pdf with the axes, and the manuscript
overlays the two, so both are reproduced.  Some panels give several images
(`Fig.3e/xy-0.125000-320-40-12800.png`, `Fig.S10/paper-HN-OBC-2-DOS-n1-116-n2-58.png`,
…), which is why a folder can hold more than one figure.


## The CSV files

Every file has the same three-part layout:

```
# data of Figures/xy-0.125000-320-40-12800.pdf | Fig. 3(e) | OBC spectrum of the 2D HN model in the xy-parallelogram with (L1, L2) = (320, 40); Re(E), Im(E)
Re_E,Im_E
-4.2979301576658345,-2.6713290892113529
...
```

* line 1 is a comment: it names the file of the original figure, the caption
  panel and the quantities of the columns;
* line 2 is the header with the column names (ASCII, comma separated);
* every later line is one data point.

Numbers are written with `%.17g`, so a float64 survives a round trip exactly;
`nan` and `inf` appear as such.  The files are ASCII with LF line endings and
carry no timestamps, so replotting always gives the same image.
Complex quantities are stored as a pair of columns (`Re_E,Im_E`); a `series`,
`entry` or `branch` column groups several polylines of different length in one
file.

The 56 density-of-states files of Fig. S10 are the exception: they are a matrix
instead of a table, because the grid is regular.

```
# data of ... | line 2: Re_E edges, line 3: Im_E edges, later lines: dos of the cells | ...
Re_E,-6,-5.939698492462312,-5.879396984924624,...        (200 edges)
Im_E,-3,-2.969849246231156,-2.939698492462312,...        (200 edges)
0,0,1.234567890123457,...,0                              (199 values, one cell row)
```

* line 1 is the same kind of comment;
* line 2 holds the `Re(E)` cell edges and line 3 the `Im(E)` cell edges, each
  after the name of its axis — the same edges that `pcolormesh` takes;
* every later line holds the values of one cell row (199 numbers), so the file
  is the 199 x 199 matrix of the panel.

`np.loadtxt(..., delimiter=",", skiprows=3)` gives that matrix directly.

Reading a file with NumPy or pandas:

```python
import numpy as np
import pandas as pd

# a table: column 0 is the first header name, ...
table = np.loadtxt("Fig.3e/xy-0.125000-320-40-12800.csv",
                   delimiter=",", skiprows=2)
frame = pd.read_csv("Fig.3e/xy-0.125000-320-40-12800.csv", comment="#")

# the matrix of Fig. S10: the cells only, plus the two axes
grid = np.loadtxt("Fig.S10/paper-HN-OBC-2-DOS-n1-320-n2-40.csv",
                  delimiter=",", skiprows=3)
lines = open("Fig.S10/paper-HN-OBC-2-DOS-n1-320-n2-40.csv").readlines()
re_edges = np.array([float(v) for v in lines[1].split(",")[1:]])
im_edges = np.array([float(v) for v in lines[2].split(",")[1:]])
```


## The panels and their data

`Re_E, Im_E` means the real and the imaginary part of the complex quantity
that the panel shows.

| folder | CSV files | content | columns |
|---|---|---|---|
| `Fig.3e` | `xy-<aspect>-<L1>-<L2>-<total>.csv` (3) | OBC spectra of the parallel region with compatible SGBZs; the three aspect ratios 8:1, 1:1, 1:8 | `Re_E,Im_E` |
| `Fig.3f` | `11-<aspect>-<L1>-<L2>-<total>.csv` (3) | the same for the region with incompatible SGBZs | `Re_E,Im_E` |
| `Fig.S3b` | `2D-HN-test-0.csv` … `-3.csv` (4) | `W` and `W_q1D/L2` versus `mu_1`, reference energies (i)–(iv), 2D HN model | `mu_1,w_q1d_over_L2,w` |
| `Fig.S3d` | `nH-Haldane-test-0.csv` … `-3.csv` (4) | the same for the non-Hermitian Haldane model | `mu_1,w_q1d_over_L2,w` |
| `Fig.S8a` | `quasi-1D-x-N_15-E.csv` | QMGBZ spectrum of the [10]-strip | `Re_E,Im_E` |
| `Fig.S8b` | `quasi-1D-y-N_15-E.csv` | QMGBZ spectrum of the [01]-strip | `Re_E,Im_E` |
| `Fig.S8c` | `quasi-1D-11-N_15-E.csv` | QMGBZ spectrum of the [11]-strip | `Re_E,Im_E` |
| `Fig.S8d` | `quasi-1D-x-N_15-beta.csv`, `-minor.csv` | major (i) and minor (ii) momenta of the [10]-strip | `theta_pi,mu_x`; `theta_pi,mu_y` |
| `Fig.S8e` | `quasi-1D-y-N_15-beta.csv`, `-minor.csv` | the same for the [01]-strip | `theta_pi,mu_y`; `theta_pi,mu_x` |
| `Fig.S8f` | `quasi-1D-11-N_15-beta.csv`, `-minor.csv` | the same for the [11]-strip | `theta_pi,mu_11`; `theta_pi,mu_y` |
| `Fig.S10` | `paper-HN-OBC-2-DOS-n1-<L1>-n2-<L2>.csv` (56) | density of states, one file per size; the order of the published grid is given below | matrix, cells = `dos` |
| `Fig.S11a` | `eigenstate-distribution-320-40.csv` | eigenstate (i) and SGBZ mode (ii) on the lattice, L1 = 320, L2 = 40 | `n1,n2,x,y,Re_psi,Re_psi_approx,abs_diff` |
| `Fig.S11b` | `eigenstate-distribution-sliced-320-40.csv` | the same on the upper edge (i, ii) | `n1,n2,x,y,Re_psi,Re_psi_approx` |
| `Fig.S11c` | `eigenstate-distribution-sliced-320-40.csv` | the same on the lower edge (i, ii); the data are the full mesh, the panel crops it | `n1,n2,x,y,Re_psi,Re_psi_approx` |
| `Fig.S11d` | `eigenstate-difference-upper.csv` | residual `R_u` for L2 = 30, 40, 50 | `n1,L2_30,L2_40,L2_50` |
| `Fig.S11e` | `eigenstate-difference-lower.csv` | residual `R_l` | `n1,L2_30,L2_40,L2_50` |
| `Fig.S12b` | `Haldane-SGBZ-spectrum-a1.csv`, `-a2.csv`, `-x.csv` | SGBZ spectra along the a1, a2 and x axes | `Re_E,Im_E` |
| `Fig.S12c` | `Haldane-SGBZ-spectrum-y.csv` | SGBZ spectrum along the y axis | `Re_E,Im_E` |
| `Fig.S12e` | `Haldane-OBC-spectrum-80-80.csv` | OBC spectrum, rhombic geometry, 80 x 80 | `Re_E,Im_E` |
| `Fig.S12f` | `Haldane-OBC-spectrum-80-40-square.csv` | OBC spectrum, rectangular geometry, 80 x 40 | `Re_E,Im_E` |
| `Fig.S13a` | `OBC-spectrum-nx77-ny77.csv` | OBC spectrum, Lx = Ly = 77 | `Re_E,Im_E` |
| `Fig.S13b` | `OBC-spectrum-nx300-ny20.csv` | OBC spectrum, Lx = 300, Ly = 20 | `Re_E,Im_E` |
| `Fig.S13c` | `OBC-spectrum-nx20-ny300.csv` | OBC spectrum, Lx = 20, Ly = 300 | `Re_E,Im_E` |
| `Fig.S13d` | `x-SGBZ-mux.csv`, `y-SGBZ-mux.csv` | `mu_x` on the x-SGBZ (i) and on the y-SGBZ (ii) | `Re_E,mu` |
| `Fig.S13e` | `x-SGBZ-spectrum.csv` | x-SGBZ spectrum | `Re_E,Im_E` |
| `Fig.S13f` | `y-SGBZ-spectrum.csv` | y-SGBZ spectrum | `Re_E,Im_E` |
| `Fig.S18c` | `square-region.csv`, `square-region-zoomed-in.csv` | numerical spectrum and amoebic boundary of the square geometry; the second file is the inset | `series,Re_E,Im_E` |
| `Fig.S18d` | `diamond-region.csv` | the same for the diamond geometry | `series,Re_E,Im_E` |
| `Fig.S19a` | `paper-Amoeba-E_1.00_1.00.csv` | amoeba of the characteristic polynomial at E = 1 + i | `entry,branch,axis,mu,mu_min,mu_max` |
| `Fig.S19b` | `paper-Amoeba-E_3.00_3.00.csv` | the same at E = 3 + 3i (the case with a central hole) | `entry,branch,axis,mu,mu_min,mu_max` |
| `Fig.S23c` | `OBC_eigenenergy_no_phase.csv` | numerical OBC spectrum of the uniform-band model | `Re_E,Im_E` |

A few files need a word of explanation:

* **Fig. S10.** Each of the 56 files is one panel of the published grid; the
  images follow the same order as the grid: rows are fixed `L1/L2`
  (2:1, 4:1, … 16:1) and the columns grow with `L1`.
* **Fig. S11b,c.** Both edges come from one mesh, so the two folders share the
  same file; the panel crops the mesh to `n2` in [34, 40] (upper edge) or
  [-1, 6] (lower edge).  The `abs_diff` column of Fig. S11a is the difference
  between the two eigenstate columns; it belongs to the same mesh but is not
  one of the caption panels.
* **Fig. S19.** `entry` and `branch` number the twelve degenerate meshes that
  the plotting script paints; each one is a band that covers, at a fixed `mu`
  and in the direction given by `axis`, the interval `[mu_min, mu_max]`
  (`axis = x`: the band runs along `mu_y` at a fixed `mu_x`, and the other way
  round for `axis = y`).  Storing the two envelopes reproduces the drawn region
  exactly and keeps the file at about 0.1 MB instead of 45 MB.
* **Fig. S3b, S3d.** The images `_1` … `_4` are the reference energies (i)–(iv)
  of the caption; the CSV files are numbered in the same order.
* **Fig. S8d,e,f.** `theta_pi` is the argument of the complex momentum in units
  of π, `mu_*` its logarithm.  A few rows of the `-minor.csv` files are `nan`:
  the fit of the minor momentum did not converge there, and the panel shows
  the same gaps.
* **Fig. S23c.** The OBC spectrum of the uniform-band model is real; the
  imaginary parts in the file are the 1e-9 noise of the eigensolver, which is
  why the image (like the original one) shows an axis in units of 1e-9.


## Drawing the panels

`plot-figures.py` redraws the figures from the CSV files and writes them back
into the folders:

```bash
python plot-figures.py                  # every folder
python plot-figures.py Fig.3e Fig.S8d   # only these folders
python plot-figures.py --list           # names, contents and file counts
```

Only NumPy and Matplotlib are needed (`pip install numpy matplotlib`).

The drawing functions are **the functions of the paper repository**, copied into
`code/paper_plots.py`; only the data source changed (a CSV read instead of a
pickle read).  Figure sizes, axis ranges, ticks, labels, markers, the analytic
curves (the SGBZ boundaries and the analytical spectrum of Fig. S18) and the
sequence of png/pdf saves are the ones of the original scripts, so each file in
a folder is the file that the paper repository writes into its `Figures` folder
for the same panel.  The only two deviations are documented in the code: the
amoeba band is painted from the stored envelopes instead of the pcolor of the
degenerate mesh, and the amoeba pdf is written after the bands are painted (the
original saves it before, which leaves it empty).

The layout of the paper figures (the assembly into a multi-panel figure, the
panel letters and the aspect-ratio labels of Fig. 3) is not part of this
package: the figures were assembled when the manuscript was written.


## What is not in this package

* Panels **computed analytically** by the plotting scripts have no data file:
  Fig. 3(c,d) and S7(d,f) (the [10]/[01]- and [11]-SGBZ bands), Fig. S17, the
  BZ region of Fig. S23(c), and the schematic panels (Figs. 1, 2, 4, S1, S2,
  S4–S6, S9, S11(f,g), S14–S16, S18(a,b), S21, S22).  These curves follow from
  the formulas printed in the scripts of the paper repository.
* The **pickles** of the numerical runs.  The complete data set, together with
  the scripts that produced it, is published at
  [cstr.cn/31253.11.sciencedb.015c0](https://cstr.cn/31253.11.sciencedb.015c0).
* Panels that the paper does not use (the density-of-states panels of the
  gain-loss Haldane model and the GBZ-candidate figures) and 2 + 2i of the
  amoeba family: they are working data of the repository, not figure data.


## How the files were produced

The CSV files are written by the plotting scripts of the paper repository
(`paper/figure_data.py`) at the same time as the figures, so the data and the
figure of a panel always come from one run.  The folder names were taken from
the LaTeX captions of the main text and the Supplementary Information, and
`export-figure-csv.py` holds the table of which file belongs to which panel:
running it from the repository reproduces this folder, and
`python export-figure-csv.py --check` reports whether it is up to date.

The model parameters are the ones of the paper: the 2D HN model has
`(Jx1, Jx2, Jy1, Jy2) = (1 + i, 1.5 + 1.2i, -1 + i, -1.2 - 0.5i)`, the
gain-loss Haldane model has `t1 = 1, t2 = 0.5, phi = pi/3, gamma = 0.5`, the
long-range model has `tx = 1, ty = 0.8, v = 0.5, cx = 0.1, cy = 0.2,
kappa_x = kappa_y = 1, gamma_x = 0.2`, and the analytically solvable model has
`t = 2, a = 0.8, b = 0.4 exp(i pi/12)`.
