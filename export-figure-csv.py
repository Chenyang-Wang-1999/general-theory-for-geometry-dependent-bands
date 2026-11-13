#!/usr/bin/env python3
'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-11-13
Copyright © Department of Physics, Tsinghua University. All rights reserved

Collect the CSV text data of the figure panels in ``data-export/``.

    python export-figure-csv.py                  # fill data-export/
    python export-figure-csv.py Fig.3e Fig.S8d   # only these panels
    python export-figure-csv.py --check          # report, write nothing
    python export-figure-csv.py --list           # the panels and their files
    python export-figure-csv.py --force          # regenerate the CSV files first

The numerical data behind a panel are written next to the figure by the
plotting scripts of ``paper/`` (``paper/<area>/Figures/*.csv``, see
``paper/figure_data.py``); those files are not tracked by git.  The table
``JOBS`` below says which panel of the captions of the main text and of the
Supplementary Information is drawn from which of those files, and this script
copies them into ``data-export/Fig.<number><letter>/``, one folder per caption
panel.  A file that backs several panels (the sliced eigenstate of
Fig. S11(b,c), for example) is copied into every one of them; the panels that a
plotting script computes analytically have no CSV and get no folder.

A plotting script is run - in its own folder, with the Agg backend - when one of
the CSV files it writes is missing, so that a clone without the (ignored) CSV
files is filled by a single call; ``--force`` runs the scripts even when the
files are already there.  Such a run rewrites the figures of
``paper/<area>/Figures`` as well, exactly as the original scripts do.  Only the
standard library is used here; the plotting scripts themselves need NumPy,
Matplotlib and the data pickles.
'''

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "paper")
EXPORT = os.path.join(HERE, "data-export")

#: the caption panels and the CSV files they are drawn from.
#:
#: Every entry is ``(folder of the plotting script, the script, panels)``; the
#: keys of ``panels`` are the folders of ``data-export/`` and the values are the
#: names of the CSV files of ``paper/<folder>/Figures/``.  The order is the
#: order of the paper.  The comment after each panel is the label the CSV files
#: carry on their first line.
JOBS = [
    ("HN-model", "OBC-spectrum.py", {
        # Fig. 3(e), Fig. 3(f): OBC spectra in the two parallelogram geometries
        "Fig.3e": [
            "xy-0.125000-320-40-12800.csv",
            "xy-1.000000-113-113-12769.csv",
            "xy-8.000000-40-320-12800.csv",
        ],
        "Fig.3f": [
            "11-0.125000-320-40-12800.csv",
            "11-1.000000-113-113-12769.csv",
            "11-8.000000-40-320-12800.csv",
        ],
    }),
    ("HN-model", "quasi-1D-spectrum.py", {
        # Fig. S8: QMGBZ spectra and momenta of the [10], [01] and [11] strips
        "Fig.S8a": [
            "quasi-1D-x-N_15-E.csv",
        ],
        "Fig.S8b": [
            "quasi-1D-y-N_15-E.csv",
        ],
        "Fig.S8c": [
            "quasi-1D-11-N_15-E.csv",
        ],
        "Fig.S8d": [
            "quasi-1D-x-N_15-beta.csv",
            "quasi-1D-x-N_15-minor.csv",
        ],
        "Fig.S8e": [
            "quasi-1D-y-N_15-beta.csv",
            "quasi-1D-y-N_15-minor.csv",
        ],
        "Fig.S8f": [
            "quasi-1D-11-N_15-beta.csv",
            "quasi-1D-11-N_15-minor.csv",
        ],
    }),
    ("HN-model", "eigenstate-comparison.py", {
        # Fig. S11: eigenstate, its non-Bloch decomposition and the residuals.
        # The sliced mesh backs both edges, hence the file in two folders.
        "Fig.S11a": [
            "eigenstate-distribution-320-40.csv",
        ],
        "Fig.S11b": [
            "eigenstate-distribution-sliced-320-40.csv",
        ],
        "Fig.S11c": [
            "eigenstate-distribution-sliced-320-40.csv",
        ],
        "Fig.S11d": [
            "eigenstate-difference-upper.csv",
        ],
        "Fig.S11e": [
            "eigenstate-difference-lower.csv",
        ],
    }),
    ("HN-model", "plot-OBC-dos.py", {
        # Fig. S10: 56 of the 60 density-of-states panels that batch_plot()
        # draws; the smallest size of every aspect-ratio row is not in the grid.
        "Fig.S10": [
            "paper-HN-OBC-2-DOS-n1-116-n2-58.csv",
            "paper-HN-OBC-2-DOS-n1-140-n2-70.csv",
            "paper-HN-OBC-2-DOS-n1-162-n2-81.csv",
            "paper-HN-OBC-2-DOS-n1-164-n2-41.csv",
            "paper-HN-OBC-2-DOS-n1-180-n2-90.csv",
            "paper-HN-OBC-2-DOS-n1-196-n2-98.csv",
            "paper-HN-OBC-2-DOS-n1-200-n2-50.csv",
            "paper-HN-OBC-2-DOS-n1-204-n2-34.csv",
            "paper-HN-OBC-2-DOS-n1-212-n2-106.csv",
            "paper-HN-OBC-2-DOS-n1-226-n2-113.csv",
            "paper-HN-OBC-2-DOS-n1-228-n2-57.csv",
            "paper-HN-OBC-2-DOS-n1-232-n2-29.csv",
            "paper-HN-OBC-2-DOS-n1-246-n2-41.csv",
            "paper-HN-OBC-2-DOS-n1-256-n2-64.csv",
            "paper-HN-OBC-2-DOS-n1-260-n2-26.csv",
            "paper-HN-OBC-2-DOS-n1-276-n2-69.csv",
            "paper-HN-OBC-2-DOS-n1-280-n2-35.csv",
            "paper-HN-OBC-2-DOS-n1-282-n2-47.csv",
            "paper-HN-OBC-2-DOS-n1-288-n2-24.csv",
            "paper-HN-OBC-2-DOS-n1-300-n2-75.csv",
            "paper-HN-OBC-2-DOS-n1-308-n2-22.csv",
            "paper-HN-OBC-2-DOS-n1-310-n2-31.csv",
            "paper-HN-OBC-2-DOS-n1-312-n2-52.csv",
            "paper-HN-OBC-2-DOS-n1-320-n2-40.csv",
            "paper-HN-OBC-2-DOS-n1-320-n2-80.csv",
            "paper-HN-OBC-2-DOS-n1-336-n2-21.csv",
            "paper-HN-OBC-2-DOS-n1-342-n2-57.csv",
            "paper-HN-OBC-2-DOS-n1-348-n2-29.csv",
            "paper-HN-OBC-2-DOS-n1-360-n2-36.csv",
            "paper-HN-OBC-2-DOS-n1-360-n2-45.csv",
            "paper-HN-OBC-2-DOS-n1-366-n2-61.csv",
            "paper-HN-OBC-2-DOS-n1-378-n2-27.csv",
            "paper-HN-OBC-2-DOS-n1-390-n2-65.csv",
            "paper-HN-OBC-2-DOS-n1-392-n2-49.csv",
            "paper-HN-OBC-2-DOS-n1-396-n2-33.csv",
            "paper-HN-OBC-2-DOS-n1-400-n2-25.csv",
            "paper-HN-OBC-2-DOS-n1-400-n2-40.csv",
            "paper-HN-OBC-2-DOS-n1-424-n2-53.csv",
            "paper-HN-OBC-2-DOS-n1-434-n2-31.csv",
            "paper-HN-OBC-2-DOS-n1-440-n2-44.csv",
            "paper-HN-OBC-2-DOS-n1-444-n2-37.csv",
            "paper-HN-OBC-2-DOS-n1-456-n2-57.csv",
            "paper-HN-OBC-2-DOS-n1-464-n2-29.csv",
            "paper-HN-OBC-2-DOS-n1-470-n2-47.csv",
            "paper-HN-OBC-2-DOS-n1-476-n2-34.csv",
            "paper-HN-OBC-2-DOS-n1-480-n2-40.csv",
            "paper-HN-OBC-2-DOS-n1-510-n2-51.csv",
            "paper-HN-OBC-2-DOS-n1-512-n2-32.csv",
            "paper-HN-OBC-2-DOS-n1-516-n2-43.csv",
            "paper-HN-OBC-2-DOS-n1-518-n2-37.csv",
            "paper-HN-OBC-2-DOS-n1-552-n2-46.csv",
            "paper-HN-OBC-2-DOS-n1-560-n2-35.csv",
            "paper-HN-OBC-2-DOS-n1-560-n2-40.csv",
            "paper-HN-OBC-2-DOS-n1-592-n2-37.csv",
            "paper-HN-OBC-2-DOS-n1-602-n2-43.csv",
            "paper-HN-OBC-2-DOS-n1-640-n2-40.csv",
        ],
    }),
    ("HN-model", "BZ_spectrum.py", {
        # Fig. S23(c): the OBC spectrum of the uniform-band model
        "Fig.S23c": [
            "OBC_eigenenergy_no_phase.csv",
        ],
    }),
    ("HN-model", "Amoeba-related-plot.py", {
        # Fig. S19: the amoebas of the two energies that the paper shows
        "Fig.S19a": [
            "paper-Amoeba-E_1.00_1.00.csv",
        ],
        "Fig.S19b": [
            "paper-Amoeba-E_3.00_3.00.csv",
        ],
    }),
    ("Haldane-gainloss", "plot-spectrum.py", {
        # Fig. S12: SGBZ spectra and OBC spectra of the gain-loss Haldane model
        "Fig.S12b": [
            "Haldane-SGBZ-spectrum-a1.csv",
            "Haldane-SGBZ-spectrum-a2.csv",
            "Haldane-SGBZ-spectrum-x.csv",
        ],
        "Fig.S12c": [
            "Haldane-SGBZ-spectrum-y.csv",
        ],
        "Fig.S12e": [
            "Haldane-OBC-spectrum-80-80.csv",
        ],
        "Fig.S12f": [
            "Haldane-OBC-spectrum-80-40-square.csv",
        ],
    }),
    ("long-range-coupling", "spectrum.py", {
        # Fig. S13: OBC and SGBZ spectra of the long-range model
        "Fig.S13a": [
            "OBC-spectrum-nx77-ny77.csv",
        ],
        "Fig.S13b": [
            "OBC-spectrum-nx300-ny20.csv",
        ],
        "Fig.S13c": [
            "OBC-spectrum-nx20-ny300.csv",
        ],
        "Fig.S13d": [
            "x-SGBZ-mux.csv",
            "y-SGBZ-mux.csv",
        ],
        "Fig.S13e": [
            "x-SGBZ-spectrum.csv",
        ],
        "Fig.S13f": [
            "y-SGBZ-spectrum.csv",
        ],
    }),
    ("winding-number", "W-and-W-strip.py", {
        # Fig. S3: the four reference energies (i)-(iv) of the two models; the
        # script also writes a fifth and a sixth, which the paper does not use
        "Fig.S3b": [
            "2D-HN-test-0.csv",
            "2D-HN-test-1.csv",
            "2D-HN-test-2.csv",
            "2D-HN-test-3.csv",
        ],
        "Fig.S3d": [
            "nH-Haldane-test-0.csv",
            "nH-Haldane-test-1.csv",
            "nH-Haldane-test-2.csv",
            "nH-Haldane-test-3.csv",
        ],
    }),
    ("example-for-geom-dep-band", "plot-spectrum.py", {
        # Fig. S18(c,d): numerical spectrum and amoebic boundary
        "Fig.S18c": [
            "square-region.csv",
            "square-region-zoomed-in.csv",
        ],
        "Fig.S18d": [
            "diamond-region.csv",
        ],
    }),
]


def figures_folder(area):
    '''Return the ``Figures`` folder of the plotting script of one area.'''
    return os.path.join(PAPER, area, "Figures")


def panel_table():
    '''Return ``[(panel, area, script, [file names])]`` in the order of JOBS.'''
    table = []
    for area, script, panels in JOBS:
        for panel, files in panels.items():
            table.append((panel, area, script, list(files)))
    names = [panel for panel, _, _, _ in table]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise SystemExit("export-figure-csv.py: panel in JOBS twice: %s"
                         % ", ".join(duplicated))
    return table


def select(table, names):
    '''Return the rows of ``table`` whose panel is in ``names`` (all if empty).'''
    if not names:
        return table
    known = [panel for panel, _, _, _ in table]
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(
            "export-figure-csv.py: no such panel: %s\nknown panels: %s"
            % (", ".join(unknown), " ".join(known)))
    wanted = set(names)
    return [row for row in table if row[0] in wanted]


def missing_files(area, files):
    '''Return the file names that the plotting script has not written yet.'''
    folder = figures_folder(area)
    return [name for name in files if not os.path.isfile(os.path.join(folder, name))]


def scripts_to_run(table, force):
    '''Return the ``(area, script)`` pairs that have to run before the copy.'''
    groups = {}
    for panel, area, script, files in table:
        groups.setdefault((area, script), []).extend(files)
    pending = []
    for (area, script), files in groups.items():
        if force or missing_files(area, files):
            pending.append((area, script))
    return pending


def run_script(area, script):
    '''Run the ``__main__`` block of a plotting script and report the outcome.'''
    folder = os.path.join(PAPER, area)
    print("  run    paper/%s/%s" % (area, script), flush=True)
    start = time.time()
    environment = dict(os.environ, MPLBACKEND="Agg")
    try:
        result = subprocess.run(
            [sys.executable, script], cwd=folder, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as error:
        print("  !!     could not start it: %s" % error)
        return False
    output = result.stdout.decode("utf-8", "replace").strip()
    elapsed = time.time() - start
    if result.returncode != 0:
        print("  !!     failed after %.1f s with exit code %d"
              % (elapsed, result.returncode))
        for line in output.splitlines()[-20:]:
            print("         " + line)
        return False
    print("  done   %.1f s" % elapsed, flush=True)
    return True


def copy_panel(panel, area, files, check):
    '''Copy the CSV files of one panel into its folder of data-export/.

    Returns ``(copied, unchanged, missing)``, where ``copied`` counts the files
    that are written (or, with ``check``, that would be written).
    '''
    source = figures_folder(area)
    target = os.path.join(EXPORT, panel)
    copied = unchanged = 0
    missing = []
    for name in files:
        origin = os.path.join(source, name)
        destination = os.path.join(target, name)
        if not os.path.isfile(origin):
            missing.append(name)
            continue
        if os.path.isfile(destination) and filecmp.cmp(origin, destination,
                                                       shallow=False):
            unchanged += 1
            continue
        if not check:
            os.makedirs(target, exist_ok=True)
            shutil.copyfile(origin, destination)
        copied += 1
    return copied, unchanged, missing


def stray_files(table):
    '''Return the CSV files of data-export/ that no panel claims.'''
    claimed = {}
    for panel, area, script, files in table:
        claimed.setdefault(panel, set()).update(files)
    stray = []
    if not os.path.isdir(EXPORT):
        return stray
    for panel in sorted(os.listdir(EXPORT)):
        folder = os.path.join(EXPORT, panel)
        if not os.path.isdir(folder) or panel == "code":
            continue
        for name in sorted(os.listdir(folder)):
            if name.endswith(".csv") and name not in claimed.get(panel, ()):
                stray.append(os.path.join(panel, name))
    return stray


def main():
    parser = argparse.ArgumentParser(
        description="Copy the CSV text data of the figure panels into "
                    "data-export/.")
    parser.add_argument("panels", nargs="*", metavar="PANEL",
                        help="only these folders, for example Fig.3e")
    parser.add_argument("--check", action="store_true",
                        help="report what would be written, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="run the plotting scripts even if their CSV "
                             "files are already there")
    parser.add_argument("--list", action="store_true",
                        help="list the panels and the CSV files they need")
    arguments = parser.parse_args()

    try:                        # keep the log readable when it goes to a file
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    table = panel_table()
    rows = select(table, arguments.panels)

    if arguments.list:
        for panel, area, script, files in rows:
            present = len(files) - len(missing_files(area, files))
            print("%-9s %2d/%-2d file(s)  paper/%s/%s"
                  % (panel, present, len(files), area, script))
            for name in files:
                print("          %s" % name)
        return 0

    failed = []
    pending = scripts_to_run(rows, arguments.force)
    if pending:
        if arguments.check:
            print("the plotting scripts would run (their CSV files are not "
                  "all there):")
            for area, script in pending:
                print("  paper/%s/%s" % (area, script))
        else:
            print("the plotting scripts write the CSV files first:")
            for area, script in pending:
                if not run_script(area, script):
                    failed.append((area, script))
            print()

    total_copied = total_unchanged = 0
    missing = []
    for panel, area, script, files in rows:
        copied, unchanged, absent = copy_panel(panel, area, files,
                                               arguments.check)
        total_copied += copied
        total_unchanged += unchanged
        notes = []
        if copied:
            notes.append("%d %s" % (copied,
                                    "to write" if arguments.check else "written"))
        if unchanged:
            notes.append("%d unchanged" % unchanged)
        if absent:
            notes.append("%d missing" % len(absent))
        print("%-9s %2d file(s)  %s" % (panel, len(files),
                                        ", ".join(notes) or "nothing to do"))
        for name in absent:
            print("  !!     %s is not in paper/%s/Figures" % (name, area))
            missing.append((panel, name))

    stray = stray_files(panel_table())
    if stray:
        print("\nCSV files of data-export/ that no panel of the paper uses:")
        for path in stray:
            print("  %s" % path)

    print("\n%d file(s) of %d panel(s): %d %s, %d unchanged, %d missing"
          % (total_copied + total_unchanged + len(missing), len(rows),
             total_copied, "to write" if arguments.check else "written",
             total_unchanged, len(missing)))
    for area, script in failed:
        print("the plotting script paper/%s/%s failed, so its panels are "
              "incomplete" % (area, script))
    if arguments.check:
        print("--check: nothing was written")
    return 1 if (missing or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
