# Codes and plotting scripts for "General theory for geometry-dependent non-Hermitian bands"

This repo contains codes and plotting scripts for replicating all the figures in the main text and Supplementary Information (SI) of the article "General theory for geometry-dependent non-Hermitian bands".
Detailed information of the codes and data are listed in following "main text" and "Supplementary Information" sections.

The LaTeX source of the main text and SI is archived in `paper/manu-and-supp`. To replicate the figures, you can:
1. Run the full numerical calculation scripts listed below. Note that some scripts depend on packages implemented in this repo. 
   You can find them in the folder where the script is located, or in the folder `submodules`.
   There are also C++ source files in some folders. Pybind11 is required to build them.
2. Use numerical data files published on ScienceDB [cstr.cn/31253.11.sciencedb.015c0](https://cstr.cn/31253.11.sciencedb.015c0) and run plotting scripts.

For git version control, data are not included in this repo.
The dataset [cstr.cn/31253.11.sciencedb.015c0](https://cstr.cn/31253.11.sciencedb.015c0) contains all data files (in "pickle" format) to generate the figures used by both the main text and the SI.
The folder tree in the dataset is the same as the current repo.
For example, the folder `<dataset>/puncture-calculation/data` contains the data that should be placed in `puncture-calculation/data` folder in this repo.


To use the data, download and unzip `general-theory-geometry-dependent-bands-data.zip`, copy all contents of the unzipped folder except for README, and paste them at the root of this repo.


## Main text

### Figure 1: Illustration of geometry-dependent non-Hermitian bands

- Source: `paper/manu-and-supp/GDSE-20251014.pdf`

The figure is pure schematic. The plots of eigenstates are generated with synthetic data (superposition of non-Bloch waves).

Related files and scripts:

```mermaid
flowchart TB
    A[paper/schematic-GDSE/plot-schematic-GDSE.py]
    B1[paper/schematic-GDSE/Figures/正矩形态分布.pdf]
    B2[paper/schematic-GDSE/Figures/斜矩形态分布.pdf]
    C[Full pdf]
    A --画正矩形本征态分布--> B1
    A --画斜矩形本征态分布--> B2
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
