# Codes and plotting scripts for "General theory for geometry-dependent non-Hermitian bands"

This repo contains codes and plotting scripts for replicating all the figures in the main text and Supplementary Information (SI) of the article "General theory for geometry-dependent non-Hermitian bands".
Detailed information of the codes and data are listed in following "main text" and "Supplementary Information" sections.

The LaTeX source of the main text and SI is archived in `paper/manu-and-supp`. To replicate the figures, you can:
1. Run the full numerical calculation scripts listed below. Note that some scripts depend on packages implemented in this repo. 
   You can find them in the folder where the script is located, or in the folder `submodules`.
   There are also C++ source files in some folders. Pybind11 is required to build them.
2. Use numerical data files published on ScienceDB [cstr.cn/31253.11.sciencedb.015c0](https://cstr.cn/31253.11.sciencedb.015c0) and run plotting scripts.

All the scripts should be executed at its directory.
For example, to run "paper/schematic-GDSE/plot-schematic-GDSE.py":
```bash
cd paper/schematic-GDSE
python plot-schematic-GDSE.py
```

For git version control, data are not included in this repo.
The dataset [cstr.cn/31253.11.sciencedb.015c0](https://cstr.cn/31253.11.sciencedb.015c0) contains all data files (in "pickle" format) to generate the figures used by both the main text and the SI.
The folder tree in the dataset is the same as the current repo.
For example, the folder `<dataset>/puncture-calculation/data` contains the data that should be placed in `puncture-calculation/data` folder in this repo.


To use the data, download and unzip `general-theory-geometry-dependent-bands-data.zip`, copy all contents of the unzipped folder except for README, and paste them at the root of this repo.


## Data format and runtime environment

All numerical data are Python pickle files (`*.pkl`), 260 files and 8.4 GB in total.
They store only Python built-in types (`dict`, `tuple`, `list`, `int`, `float`,
`complex`, `bool`) and `numpy.ndarray` objects, so no class defined by this
repository is needed to read them back.

### What is needed to unpickle the data

Every file uses **pickle protocol 4**, which is readable by Python >= 3.4.
The globals referenced by the pickles were extracted from all 260 files without
loading them, with `python tools/check-pickle-deps.py` (which also saves the
report to `tools/pickle-dep-report.json`):

| data folder | files | size | library needed to unpickle |
|---|---:|---:|---|
| `Amoeba-calculation/data` | 3 | 30 MB | NumPy |
| `ChP-winding-algorithm/data` | 21 | 0.1 MB | NumPy |
| `long-range-coupling/data` | 30 | 8.4 MB | NumPy |
| `paper/HN-model/data` | 1 | 100 MB | NumPy |
| `paper/schematic-GDSE/data` | 2 | 2.2 MB | NumPy |
| `phcpy-free-algorithm/data` | 10 | 6.3 MB | NumPy; **SymPy** for `Haldane-gain-loss-amoeba-0.pkl` |
| `puncture-calculation/data` | 193 | 8.25 GB | NumPy |
| **total** | **260** | **8.4 GB** | |

The complete set of globals recorded in the data is
`numpy.core.multiarray._reconstruct`, `numpy.core.multiarray.scalar`,
`numpy.ndarray`, `numpy.dtype` (NumPy), `builtins.complex` (a Python built-in) and
`sympy.core.numbers.Float`. In particular **no SciPy, pandas, PyTorch or
Matplotlib object is stored**, and only one file needs SymPy: the working file
`phcpy-free-algorithm/data/Haldane-gain-loss-amoeba-0.pkl`, which is written and
read only by `phcpy-free-algorithm/Haldane-model-gainloss.py` and is not read by
any script in `paper/`. **SymPy is therefore not required to reproduce the
figures.**

### NumPy version

All 260 files carry the **NumPy 1.x** pickle layout: they reference
`numpy.core.multiarray.*`, and none of them references `numpy._core.multiarray`
(the layout written by NumPy >= 2.0). A pickle does not record the NumPy release
that produced it, but the environment this code and data were archived with is

| package | version |
|---|---|
| Python | 3.9.16 |
| NumPy | **1.23.5** |

and a pickle written by NumPy 1.23.5 has exactly the layout found in the data
files. Please use NumPy 1.23.5 (or another 1.x release) when reading or rewriting
the data. Note that the rename of the private module in NumPy 2.0
(`numpy.core` -> `numpy._core`) makes the layouts one-directional: `numpy.core`
is still importable in NumPy 2.x as a deprecated compatibility alias, so NumPy
2.x can read these files, but data re-saved with NumPy 2.x can no longer be read
by NumPy 1.x. Do not re-save the published data with NumPy 2.x if the files have
to remain readable by NumPy 1.x.

### matplotlib version and backend

The figures were drawn with **matplotlib 3.8.0**. The committed output files
state the release that produced them:

* 105 of the 116 PNG files under `paper/` carry
  `Software: Matplotlib version3.8.0` in their PNG metadata and the remaining 11
  carry `Matplotlib version3.6.1`.
* Of the 112 PDF files under `paper/` (excluding the assembled manuscripts in
  `paper/manu-and-supp`), 37 report
  `Producer (Matplotlib pdf backend v3.8.0)`, 4 report `v3.7.2` and 71 report
  `v3.6.1`. The `v3.6.1` and `v3.7.2` files all live in
  `paper/HN-model/Figures` and `paper/winding-number/Figures` and were carried
  over from earlier runs; they differ from a 3.8.0 re-run only in sub-pixel
  details, so exact byte-for-byte reproduction of those panels would need the
  older release.

Backend: every figure is written by `Figure.savefig`, i.e. the `.pdf` output uses
the Matplotlib PDF backend and the `.png` output uses the PNG (Agg) writer, so
the interactive backend does not influence the saved files. All scripts can
therefore run headless; if no display is available, set `MPLBACKEND=Agg`, in
which case the `plt.show()` calls in the `__main__` blocks become no-ops. Two
scripts select a backend explicitly:
`paper/settings-and-materials/plot-latex-formula.py` and
`paper/HN-model/Amoeba-related-plot.py` call `matplotlib.use("pdf")`.

The style file shared by the plotting scripts,
`paper/settings-and-materials/paper_plot.mplstyle`, requests **Arial**
(`font.sans-serif`) and **Times New Roman** (`font.serif`). These are the
Microsoft core fonts; if they are missing, Matplotlib silently falls back to
DejaVu Sans and every label in the figures changes. Install them (for example
`ttf-mscorefonts-installer` on Debian/Ubuntu) before regenerating the figures.

### Python packages needed to reproduce the figures

Only the scripts in `paper/` are needed to rebuild the figures from the published
data. They are all covered by `requirements.txt` in the root of this repository:

```bash
python -m pip install -r requirements.txt

# BerryPy is not a pip package: it ships its own setup.py, which has to be run
# from inside a clone of the repository (it installs into the user site-packages).
git clone https://github.com/Chenyang-Wang-1999/BerryPy.git
cd BerryPy
python setup.py          # add --without-binary to skip the optional binary
cd ..
```

| package | version in the archived environment | needed by |
|---|---|---|
| NumPy | 1.23.5 | every script |
| SciPy | 1.13.0 | `HN_model_common.py`, `quasi-1D-spectrum.py`, `BZ_spectrum.py` |
| Matplotlib | 3.8.0 | every plotting script |
| SymPy | 1.10.1 (optional) | only BerryPy's symbolic routines (`get_GBZ`, `get_characteristic_polynomial_data`), which no figure script calls |
| PHCpy | 1.1.4 (optional) | only BerryPy's `get_GBZ`, which no figure script calls |
| BerryPy | 3.12 develop (as reported by the installed package) | `HN_model_common.py` (imported by most `paper/HN-model` scripts) and `paper/schematic-GDSE/plot-schematic-GDSE.py` |
| shapely | 2.0.4 | only `OBC-spectrum.py::plot_11_OBC_spectrum_marked_SGBZ`, which is not called by default |

BerryPy is not distributed on PyPI and **cannot be installed with pip**: its
`setup.py` is a plain script that never calls `setuptools.setup()`. Run from
inside a clone of <https://github.com/Chenyang-Wang-1999/BerryPy.git>, it probes
for the optional packages, writes `BerryPy/flags.py` from the result, and copies
the package into the user site-packages. BerryPy treats SymPy and
PHCpy as **optional**: the routines that need them raise
`ModuleNotFoundError` when they are missing (`get_GBZ` needs both, while
`get_characteristic_polynomial_data` needs SymPy). The `paper/` figure scripts
only use `TightBindingModel`, `get_supercell` and `truncate`, so SymPy and PHCpy
are **not** needed to reproduce the figures. Both are installed in the archived
environment, and they become necessary only for the auxiliary symbolic scripts
such as `paper/HN-model/quasi-1D-beta.py`.

**No compiled extension is needed to reproduce the figures.** The C++ modules
(`_poly_tools_cc`, `lib_GBZ_manifold`) and the PHCpy solver are only used by the
numerical-calculation scripts and by the auxiliary routines
`get_GBZ_candidates()`, `plot_GBZ_candidates()` and `plot_total_winding_11()` of
`paper/HN-model/GBZ-candidates.py`. That file imports them lazily, so the figure
it produces, `Figures/DGBZ-spectrum-11-direction.pdf`, is drawn from the analytic
`[11]`-SGBZ and runs with the packages listed above alone. To run those
solver-based routines as well, build `submodules/poly_tools/pybind` and
`submodules/GBZ-manifold` with `make` (they need `g++`, the Eigen3 headers in
`/usr/include/eigen3`, Pybind11 and the Python development headers) and place the
resulting `_poly_tools_cc*.so` / `lib_GBZ_manifold*.so` next to the vendored
`poly_tools` / `GBZ_manifold` packages under `puncture-calculation/`, which is
where those scripts import them from.

Following sections describe the figures in the main text and the supplementary information, respectively.

## Main text

### Figure 1: Illustration of geometry-dependent non-Hermitian bands

- Source: `paper/manu-and-supp/GDSE-20251014.pdf`

The figure is pure schematic. The plots of eigenstates are generated with synthetic data (superposition of non-Bloch waves).

Related files and scripts:

```mermaid
flowchart TB
    A[paper/schematic-GDSE/plot-schematic-GDSE.py]
    B1[paper/schematic-GDSE/Figures/eigenstate-square.pdf]
    B2[paper/schematic-GDSE/Figures/eigenstate-slanted.pdf]
    C[Full pdf]
    A --draw_eigenstate_distribution_square --> B1
    A --draw_eigenstate_distribution_slanted --> B2
    B1 --assembled in--> C
    B2 --assembled in--> C
```

### Figure 2: Strip geometry and SGBZ constraint 

- Source: `paper/manu-and-supp/SGBZ-formulation-20260325.pdf`

The figure is pure schematic. 

### Figure 3: Relation between SGBZs and geometry-dependent bands

- Source: `paper/manu-and-supp/OBC-comparison-20250821.pdf`

The panels **a**, **b** are schematic, **c**, **d** are created using analytical solutions of the HN model, and **e**, **f** are numerical.

Related files, data and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    B1["paper/HN-model/Figures/11-[aspect]-[n1]-[n2]-[total].[pdf/png]"]
    B2["paper/HN-model/Figures/xy-[aspect]-[n1]-[n2]-[total].[pdf/png]"]
    C[paper/HN-model/OBC-spectrum.py]
    D1[puncture-calculation/data/paper-HN-OBC-2-aspect*.pkl]
    D2[puncture-calculation/data/paper-HN-OBC-2-xy-aspect*.pkl]
    E[puncture-calculation/HN-model.py]
    H[paper/HN-model/GBZ-candidates.py]
    G[paper/HN-model/Figures/DGBZ-spectrum-11-direction.pdf]
    C --plot_OBC_with_given_aspect--> B1
    C --plot_11_OBC_spectrum--> B2
    B1 --assembled in--> A
    B2 --assembled in--> A
    E --paper_HN_sweep_aspect_and_size--> D1
    E --paper_HN_sweep_aspect_and_size_xy--> D2
    D1 --> C
    D2 --> C
    H --plot_11_DGBZ_spectrum--> G
    G --assembled in--> A
```

\* "DGBZ" is a deprecated name for "SGBZ"

`paper/HN-model/GBZ-candidates.py` draws `DGBZ-spectrum-11-direction.pdf`
directly when it is run; the `[11]`-SGBZ it plots is analytic, so no data file
and no compiled extension is involved. The remaining routines in that file
(`get_GBZ_candidates`, `plot_GBZ_candidates`, `plot_total_winding_11`) do use the
punctured-GBZ solver and are commented out in its `__main__`.

### Figure 4: Criterion for uniform or geometry-dependent bands

- Source: paper/manu-and-supp/illustration-uniform-band-cond-20251014.pdf

The figure is pure schematic.

## Supplementary Information

### Figure S1: Schematics of the SGBZ

- Source: paper/manu-and-supp/main-idea-20250429.pdf
  
The figure is pure schematic.

### Figure S2: Distribution of the zeros of $F_{L_{2}}$ and properties of the major-axis winding number

- Source: paper/manu-and-supp/supp-SGBZ-formulation-20250901.pdf
  
The figure is pure schematic

### Figure S3: Comparison of $W(E,\mu_{1})$ and $W_{\text{q1D}}(E,\mu_{1},L_{2})/L_{2}$ in the 2D Hatano-Nelson model and the non-Hermitian Haldane model

- Source: paper/manu-and-supp/winding-number-numerical-20250906.pdf
  
The panels **a** and **c** are schematic, while **b** and **d** are numerical.

Related files:
```mermaid
flowchart TB
    A[Full pdf]
    B["paper/winding-number/Figures/*"]
    C[paper/winding-number/W-and-W-strip.py]
    D1["ChP-winding-algorithm/data/2D-HN-test-[n].pkl"]
    D2["ChP-winding-algorithm/data/nH-Haldane-test-[n].pkl"]
    E[ChP-winding-algorithm/model-test-winding.py]
    E --test_2D_HN--> D1
    E --test_nH_Haldane--> D2
    D1 --> C
    D2 --> C
    B --assembled in--> A
    C --main_plot--> B
```

### Figure S4: SGBZ in higher dimensions

- Source: paper/manu-and-supp/higher-dim-SGBZ-2.pdf

The figure is pure schematic.

### Figure S5: Coordinate transformations of minor axes

- Source: paper/manu-and-supp/transformation-minor-20250919.pdf

The figure is pure schematic.

### Figure S6: Geometrical illustrations of equivalent formulations for $W(E,\mu_1)$

- Source: paper/manu-and-supp/winding-general-loop-20250527.pdf

The figure is pure schematic.

### Figure S7: Schematic diagrams of the 2D HN model, three special strips and the SGBZ bands

- Source: paper/manu-and-supp/HN-example-20250524.pdf

The panels **a**, **b**, **c**, **e** are schematic. The panels **d** and
**f** are the same as Figs. 3c and 3d in the main text (see the README
entry of Figure 3 for their codes and data).

### Figure S8: Comparison between SGBZs and QMGBZs

- Source: paper/manu-and-supp/quasi-1D-comparison-20250220.pdf

All panels are numerical. Panels **a**, **b**, **c** are the spectra of
the $[10]$-, $[01]$- and $[11]$-strip; panels **d**(i, ii), **e**(i, ii),
**f**(i, ii) are the major and minor complex momenta of the three strips.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    B["puncture-calculation/data/paper-quasi-1D-{x,y,11}-N_15.pkl"]
    C[paper/HN-model/quasi-1D-spectrum.py]
    P1["paper/HN-model/Figures/quasi-1D-{x,y,11}-N_15-{E,beta}.{pdf,png}"]
    P3["paper/HN-model/Figures/quasi-1D-{x,y,11}-N_15-minor.{pdf,png}"]
    E[puncture-calculation/HN-model.py]

    P1 --assembled in--> A
    P3 --assembled in--> A
    C --"plot_quasi_1D_{x,y,11}" --> P1
    C --"plot_minor_beta_{x,y,11}" --> P3
    B --> C
    E --"paper_quasi_1D_GBZ_{x,y,11}"--> B
    E --"paper_get_quasi_1D_non_Bloch_modes_{x,y,11}"--> B

```

### Figure S9: Illustration of major-axis winding numbers in the 3D HN model

- Source: paper/manu-and-supp/higher-dim-HN.pdf

The figure is pure schematic.

### Figure S10: OBC spectra of the 2D HN model for different aspect ratios and sizes

- Source: paper/manu-and-supp/larger-size.pdf

All panels are numerical. Each panel is the density of states of the 2D
HN model in a parallelogram region spanned by the $[11]$-axis and the
$[01]$-axis, with the aspect ratio fixed along each row and the total
number of sites ($L_{1}L_{2}$) fixed along each column.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    P["paper/HN-model/Figures/paper-HN-OBC-2-DOS-n1-[n1]-n2-[n2].png"]
    C[paper/HN-model/plot-OBC-dos.py]
    B["puncture-calculation/data/paper-HN-OBC-2-DOS-n1-[n1]-n2-[n2].pkl"]
    E[puncture-calculation/HN-model-DOS-calculation.py]
    F["puncture-calculation/data/paper-HN-OBC-2-n1_[n1]-n2_[n2].pkl"]
    G[puncture-calculation/HN-model-OBC-sweep.py]

    P --assembled in--> A
    C --plot_OBC_dos--> P
    B --> C
    E --batch_calculate_DOS--> B
    F --> E
    G --sweep_OBC_aspect--> F

```


### Figure S11: Eigenstate distributions and the competition between incompatible SGBZs

- Source: paper/manu-and-supp/competition-SGBZ-20250919.pdf

The panels **a**--**e** are numerical, the panels **f** and **g** are
schematic. The coupling coefficients are the same as in Fig. S8, and the
eigenenergy of the eigenstate is $1.19+1.23 \mathrm{i}$.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    P1["paper/HN-model/Figures/eigenstate-distribution-*.pdf"]
    P5["paper/HN-model/Figures/eigenstate-difference-{upper,lower}.pdf"]
    C[paper/HN-model/eigenstate-comparison.py]
    B1[puncture-calculation/data/paper-eigenstate-compare-direct-*.pkl]
    B4["puncture-calculation/data/paper-HN-OBC-2-N1_[N1]-N2_[N2].pkl"]
    B6["puncture-calculation/data/paper-HN-OBC-2-aspect_[aspect]-N_[N].pkl"]
    E[puncture-calculation/HN-eigenstate-comparison.py]
    F[puncture-calculation/HN-model.py]

    P1 --assembled in--> A
    P5 --assembled in--> A
    C --plot_eigenstate_distribution--> P1
    C --plot_eigenstate_difference--> P5
    B1 --> C
    B4 --> C
    B6 --> C
    E --decompose_under_quasi_1D_direct--> B1
    F --paper_HN_sweep_aspect_and_size--> B4
    F --paper_HN_sweep_aspect_and_size--> B6

```

\* `paper-eigenstate-compare-direct-[n1]-[n2].pkl` stores the `(aspect,
n_feature)` pair of the finite-size OBC spectrum it was built from, and
`eigenstate-comparison.py` turns that pair back into a file name. The mapping
used by the published figure is

| comparison file | OBC spectrum |
|---|---|
| `paper-eigenstate-compare-direct-320-30.pkl` | `paper-HN-OBC-2-N1_320-N2_30.pkl` |
| `paper-eigenstate-compare-direct-320-40.pkl` | `paper-HN-OBC-2-aspect_0.125000-N_40.pkl` |
| `paper-eigenstate-compare-direct-320-50.pkl` | `paper-HN-OBC-2-N1_320-N2_50.pkl` |

Panels **a**--**c** are drawn from the row with shape $(L_{1},L_{2})=(320,40)$,
i.e. `paper-HN-OBC-2-aspect_0.125000-N_40.pkl`; panels **d**, **e** use all
three rows. That identification is not an assumption: the published panels
carry the axis label $n_{2}=40$ with $y$-ticks $0$ and $40$, and their tick set
and mesh geometry are reproduced exactly (identical bounding box, 64078 marker
operations) when this file is plotted. Note that the shape is $(320,40)$, so
the `aspect` value stored in the comparison file is the *aspect ratio* `0.125`
and not the `320` that one of the panels suggests - that is why the panels are
named `320-40` even though the file they come from is an `aspect_*` file.


### Figure S12: Geometry-dependent bulk spectra of the non-Hermitian Haldane model

- Source: paper/manu-and-supp/gain-loss-Haldane.pdf

The panels **a** and **d** are schematic. The panels **b**, **c** are
the $[10]$($[01]$)- and $[11]$-SGBZ spectra, and the panels **e**, **f**
are the OBC spectra in the square and the rhombus geometries.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    P1["paper/Haldane-gainloss/Figures/Haldane-SGBZ-spectrum-{a1,a2,x,y}.{pdf,png}"]
    P3["paper/Haldane-gainloss/Figures/Haldane-OBC-spectrum-*.{pdf,png}"]
    C[paper/Haldane-gainloss/plot-spectrum.py]
    B1["phcpy-free-algorithm/data/Haldane-gain-loss-{a1,a2,x,y}-SGBZ-0.pkl"]
    B3[phcpy-free-algorithm/data/paper-Haldane-gain-loss-OBC-*.pkl]
    E[phcpy-free-algorithm/Haldane-model-gainloss.py]

    P1 --assembled in--> A
    P3 --assembled in--> A
    C --plot_SGBZ_spectrum--> P1
    C --plot_OBC_spectrum--> P3
    B1 --> C
    B3 --> C
    E --sweep_SGBZ_{a1,a2,x,y}--> B1
    E --calculate_Haldane_{OBC,OBC_square}--> B3

```

### Figure S13: OBC spectra and SGBZ for a model with exponentially decaying long-range hoppings

- Source: paper/manu-and-supp/long-range.pdf

All panels are numerical. The panels **a**--**c** are the OBC spectra in
three parallelogram geometries, the panel **d** shows $\mu_{x}$ (from the
$x$-SGBZ) and $\mu_{y}$ (from the $y$-SGBZ) as functions of $\mathrm{Re}(E)$,
and the panels **e**, **f** are the $x$-SGBZ and $y$-SGBZ spectra.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    B1["paper/long-range-coupling/Figures/OBC-spectrum-nx[nx]-ny[ny].{pdf,png}"]
    B2["paper/long-range-coupling/Figures/{x,y}-SGBZ-spectrum.{pdf,png}"]
    B3["paper/long-range-coupling/Figures/{x,y}-SGBZ-mux.{pdf,png}"]
    C[paper/long-range-coupling/spectrum.py]
    D1["long-range-coupling/data/OBC-default-nx[nx]-ny[ny].pkl"]
    D2["long-range-coupling/data/SGBZ-x0[nn].pkl"]
    D3["long-range-coupling/data/y-SGBZ-results-[n].pkl"]
    E1["long-range-coupling/OBC-calculation.py"]
    E2["long-range-coupling/SGBZ-calculation.py"]
    E3["long-range-coupling/SGBZ-x.py"]
    B1 --assembled in--> A
    B2 --assembled in--> A
    B3 --assembled in--> A
    C --plot_OBC_spectrum --> B1
    C --plot_{x,y}_SGBZ --> B2
    C --plot_{x,y}_SGBZ_mux --> B3
    D1 --> C
    D2 --> C
    D3 --> C
    E1 --get_OBC --> D1
    E2 --batch_sweep_y_SGBZ--> D2
    E3 --batch_sweep_SGBZ --> D3
```

### Figure S14: Derivation of necessary conditions for uniform bands

- Source: paper/manu-and-supp/sfig-uniform-bands-cond-20251010.pdf

The figure is pure schematic.

### Figure S15: Properties of the zeros of $f$ and sufficient conditions for uniform bands

- Source: paper/manu-and-supp/ChP-zeros-20251004.pdf

The figure is pure schematic.

### Figure S16: General criterion in $n$-dimensional lattices

- Source: paper/manu-and-supp/nD-general-criterion-20251018.pdf

The figure is pure schematic.

### Figure S17: Characterization of the geometric dependence of the 2D HN model via the $[11]$-SGBZ

- Source: paper/manu-and-supp/example-uniform-bands.pdf

The panel **a** is schematic, while the panels **b**--**d** are numerical
(the spectrum, $\mu_{[11]}$ and $\mu_{y}$ of the $[11]$-SGBZ). They are
computed from the analytical expressions of the $[11]$-SGBZ, so no data
file is needed.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    P[paper/non-Bloch-DDS/Figures/*]
    C[paper/non-Bloch-DDS/SGBZ-no-area.py]

    P --assembled in--> A
    C --plot_mu_spectrum--> P

```

### Figure S18: Schematic diagram and spectra of the analytically solvable non-Hermitian model

- Source: paper/manu-and-supp/analytical-model.pdf

The panels **a** and **b** are schematic. The panel **c** shows the
analytical and numerical OBC spectra with the square geometry
($L_{x}=L_{y}=57$) together with the amoebic band, and the panel **d**
shows the numerical OBC spectrum with the diamond geometry.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    B1["paper/example-for-geom-dep-band/Figures/square-region.[pdf/png]"]
    B2["paper/example-for-geom-dep-band/Figures/diamond-region.[pdf/png]"]
    B3["paper/example-for-geom-dep-band/Figures/square-region-zoomed-in.[pdf/png]"]
    C[paper/example-for-geom-dep-band/plot-spectrum.py]
    D1[phcpy-free-algorithm/data/special-test-OBC-spectrum-xy-57.pkl]
    D2[phcpy-free-algorithm/data/special-test-OBC-spectrum-diamond-40.pkl]
    D3[phcpy-free-algorithm/data/special-test-amoeba-boundary-rearranged.pkl]
    E[phcpy-free-algorithm/special-model.py]
    B1 --assembled in--> A
    B2 --assembled in--> A
    B3 --assembled in--> A
    C --plot_square_region--> B1
    C --plot_diamond_region--> B2
    C --plot_square_region_zoomed_in--> B3
    D1 --> C
    D2 --> C
    D3 --> C
    E --get_OBC_spectrum_xy--> D1
    E --get_OBC_spectrum_diamond--> D2
    E --sweep_amoeba/refine_amoeba_corner--> D3
    E --rearrange_amoeba_boundary--> D3
```

### Figure S19: Illustrations of the amoeba formulation

- Source: paper/manu-and-supp/Amoeba-schematics-20260208.pdf

All panels are numerical: the amoebas of the characteristic polynomial
for the reference energies $E=1+\mathrm{i}$ and $3+3\mathrm{i}$,
respectively.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    P["paper/HN-model/Figures/paper-Amoeba-E_[E].[pdf/png]"]
    C[paper/HN-model/Amoeba-related-plot.py]
    B["Amoeba-calculation/data/paper-Amoeba-E_[E].pkl"]
    E[Amoeba-calculation/Amoeba-calculation.py]

    P --assembled in--> A
    C --plot_Amoeba--> P
    B --> C
    E --paper_calculate_HN_Amoeba--> B

```

### Figure S20: Relation between amoeba and SGBZ formulations

- Source: paper/manu-and-supp/Amoeba-20260207.pdf

The panels **a**--**c** are schematic. The panel **d** shows the amoeba
spectrum together with the $[11]$-SGBZ spectra of the
2D HN model, and the panel **e** shows the $\beta_{y}$-zeros of the
characteristic polynomial at $\mu_{[11]}=\gamma_{x}+\gamma_{y}$ for the
three reference energies.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    P1[paper/HN-model/Figures/Amoeba-spectra.pdf]
    P2["paper/HN-model/Figures/paper-abs-minor-beta-E_[E].pdf"]
    C[paper/HN-model/Amoeba-related-plot.py]
    B["puncture-calculation/data/paper-non-Bloch-sol-E_[E].pkl"]
    E[puncture-calculation/HN-model.py]

    P1 --assembled in--> A
    P2 --assembled in--> A
    C --plot_Amoeba_and_GBZ_spectra--> P1
    C --plot_minor_beta_abs--> P2
    B --> C
    E --paper_get_non_Bloch_sols--> B

```


### Figure S21: Illustration of the compatibility of 1D GBZ constraints


- Source: paper/manu-and-supp/GBZ-compatibility.pdf

The figure is pure schematic.

### Figure S22: Schematic diagram of the GFS framework

- Source: paper/manu-and-supp/GFS-algebraic-260318.pdf

The figure is pure schematic.

### Figure S23: Spectral area of BZ bands

- Source: paper/manu-and-supp/spectral-area.pdf

The panels **a** and **b** are schematic, while the panel **c** is
numerical: the BZ spectra of the 2D HN model and of the uniform-band
model, respectively, compared with the SGBZ/OBC spectrum.

Related files and scripts:
```mermaid
flowchart TB
    A[Full pdf]
    P["paper/HN-model/Figures/BZ-spectrum.[pdf/png]"]
    C[paper/HN-model/BZ_spectrum.py]
    B[paper/HN-model/data/OBC_eigenenergy_no_phase.pkl]
    D[paper/HN-model/Figures/OBC_eigenenergy_no_phase.png]

    P --assembled in--> A
    C --plot_spectra--> P
    C -- calculate_OBC_eigenenergy --> B
    B --> C
    C -- plot_OBC_spectrum --> D

```

\* The BZ spectrum is computed analytically inside `BZ_spectrum.py`; the
data file `OBC_eigenenergy_no_phase.pkl` is to check the equivalence between the OBC spectrum and SGBZ spectrum.
