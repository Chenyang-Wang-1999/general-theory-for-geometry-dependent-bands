#!/usr/bin/env python3
'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
Copyright © Department of Physics, Tsinghua University. All rights reserved

Build a pruned copy of the data that is actually needed to reproduce the
figures of the paper.

The figures are generated from a much larger working data set (about 38 GB,
dominated by dense eigenvector matrices).  This script copies *only* the files
that the plotting scripts in ``paper/`` read, and trims the payload of the
large pickles: eigenvectors and other arrays that the plotting scripts load but
never use are replaced by ``None``, so that the tuple layout - and therefore
every ``pickle.load`` site - keeps working unchanged.

Usage:
    python3 tools/prune-figure-data.py --src /path/to/repo-root --dst /path/to/pruned

``--src`` is the root of a checkout that contains the data directories
(``puncture-calculation/data``, ``phcpy-free-algorithm/data``, ...); it defaults
to the directory that contains this script.  ``--dst`` is where the pruned tree
is written, mirroring the source layout, so it can be copied straight back over
the repository and every relative data path still resolves.

The script prints a report and writes ``prune-report.md`` next to ``--dst``.
It never modifies ``--src``.
'''

import argparse
import glob
import os
import pickle
import shutil
import sys

# --------------------------------------------------------------------------
# trim rules
#
# Each rule receives the object loaded from the source pickle and returns the
# object to store.  Returning ``None`` means "keep the file as it is".  A rule
# that raises is treated the same way, so an unexpected file layout can never
# produce a corrupted - or silently lossy - data file.
# --------------------------------------------------------------------------


def trim_obc_spectrum(obj):
    '''``(eigv, eigvec, coords, ...)`` -> keep the eigenvalues, drop the rest.

    The consumers only plot ``eigv``; ``eigvec`` holds one dense square
    eigenvector matrix per file, which is by far the largest payload.
    '''
    if not isinstance(obj, tuple) or len(obj) < 3:
        return None
    out = list(obj)
    if out[1] is not None:
        out[1] = None
    return tuple(out)


def trim_long_range_obc(obj):
    '''``(params, coord_list, eigv, eigvec)`` -> drop the eigenvector matrix.

    Every consumer - ``long-range-coupling/spectrum.py``,
    ``SGBZ-calculation.py`` and ``SGBZ-x.py`` - has the shape
    ``params, coord_list, eigv, eigvec = pickle.load(...)`` and then plots
    ``eigv`` only, so the last element is dead weight.
    '''
    if not isinstance(obj, tuple) or len(obj) != 4:
        return None
    params, coord_list, eigv, eigvec = obj
    return (params, coord_list, eigv, None)


def trim_basis_beta(obj, n_keep):
    '''Keep only the non-Bloch basis vectors and coefficients actually used.'''
    if not isinstance(obj, tuple) or len(obj) < 4:
        return None
    beta_list, eigv_ind, basis, coeffs = obj[0], obj[1], obj[2], obj[3]
    if hasattr(basis, 'ndim') and basis.ndim == 2:
        basis = basis[:, :n_keep]
    if hasattr(coeffs, 'ndim') and coeffs.ndim == 1:
        coeffs = coeffs[:n_keep]
    return (beta_list, eigv_ind, basis, coeffs) + tuple(obj[4:])


# --------------------------------------------------------------------------
# manifest: every data file that a figure needs
# --------------------------------------------------------------------------


def build_manifest(src):
    '''Return ``[(relative path, trim rule or None, reason), ...]``.

    ``reason`` is the figure that needs the file.
    '''

    def g(directory, pattern):
        hits = sorted(glob.glob(os.path.join(src, directory, pattern)))
        return [os.path.relpath(h, src) for h in hits]

    items = []

    # ---- S8: QMGBZ of the three strips --------------------------------
    for d in ['x', 'y', '11']:
        items.append(('puncture-calculation/data/paper-quasi-1D-%s-N_15.pkl' % d,
                      None, 'S8 a-f'))

    # ---- S10: density of states sweeps --------------------------------
    for p in g('puncture-calculation/data', 'paper-HN-OBC-2-DOS-n1-*.pkl'):
        items.append((p, None, 'S10'))
    # raw spectra upstream of the DOS data: only the eigenvalues feed the DOS
    # calculation, so the eigenvectors can be dropped
    for p in g('puncture-calculation/data', 'paper-HN-OBC-2-n1_*.pkl'):
        items.append((p, trim_obc_spectrum, 'S10 (raw spectra)'))

    # ---- Fig. 3 e,f: OBC spectra in xy regions ------------------------
    for p in g('puncture-calculation/data', 'paper-HN-OBC-2-xy-aspect_*.pkl'):
        items.append((p, trim_obc_spectrum, 'Fig. 3 e,f'))

    # ---- Fig. 3 / S7d,S7f / S11: which finite-size OBC file is which --
    #
    # ``paper-eigenstate-compare-direct-<n1>-<n2>.pkl`` stores the (aspect,
    # n_feature) pair that identifies the OBC spectrum the comparison was built
    # from, and ``eigenstate-comparison.py`` turns that pair back into a file
    # name.  The resulting mapping is
    #
    #   compare-direct-320-30 -> aspect 320   n_feature 30 -> N1_320-N2_30
    #   compare-direct-320-40 -> aspect 0.125 n_feature 40 -> aspect_0.125000-N_40
    #   compare-direct-320-50 -> aspect 320   n_feature 50 -> N1_320-N2_50
    #
    # and those three pickles are what Figs. S11 a-e (and main-text Fig. 3) are
    # actually drawn from.  Their full eigenvector matrices have to be kept:
    # the panels read ``eigvec[:, eigv_ind]`` with the ``eigv_ind`` stored in a
    # *different* member of the family, so reducing a matrix to one column would
    # break the indexing.  These three files - about 7 GB in total - are the only
    # large, irreducible part of the pruned data set.  Move them to an external
    # data deposit if the repository has to stay small.
    for n2 in [30, 50]:
        items.append(('puncture-calculation/data/paper-HN-OBC-2-N1_320-N2_%d.pkl' % n2,
                      None, 'S11 a-e / main-text Fig. 3 (full matrix required)'))
    items.append(('puncture-calculation/data/paper-HN-OBC-2-aspect_0.125000-N_40.pkl',
                  None, 'S11 a-e / main-text Fig. 3 (full matrix required)'))

    # ---- S11: non-Bloch decomposition of one eigenstate ---------------
    for n2 in [30, 40, 50]:
        items.append(('puncture-calculation/data/paper-eigenstate-compare-direct-320-%d.pkl' % n2,
                      lambda o: trim_basis_beta(o, 4), 'S11 a-e'))

    # ---- S12: gain-loss Haldane model ---------------------------------
    for which in ['a1', 'a2']:
        items.append(('phcpy-free-algorithm/data/Haldane-gain-loss-%s-SGBZ-0.pkl' % which,
                      None, 'S12 b,c'))
    for name in ['paper-Haldane-gain-loss-OBC-80-80.pkl',
                 'paper-Haldane-gain-loss-OBC-square-80-40.pkl']:
        items.append(('phcpy-free-algorithm/data/' + name,
                      trim_obc_spectrum, 'S12 e,f'))

    # ---- S13: long-range hopping model --------------------------------
    for name in ['OBC-default-nx20-ny300.pkl', 'OBC-default-nx300-ny20.pkl',
                 'OBC-default-nx77-ny77.pkl']:
        items.append(('long-range-coupling/data/' + name,
                      trim_long_range_obc, 'S13 a-c'))
    for p in g('long-range-coupling/data', 'SGBZ-x0*.pkl'):
        items.append((p, None, 'S13 d,e'))
    for p in g('long-range-coupling/data', 'y-SGBZ-results-*.pkl'):
        items.append((p, None, 'S13 d,f'))

    # ---- S18: analytically solvable model -----------------------------
    items.append(('phcpy-free-algorithm/data/special-test-OBC-spectrum-xy-57.pkl',
                  trim_obc_spectrum, 'S18 c'))
    items.append(('phcpy-free-algorithm/data/special-test-OBC-spectrum-diamond-40.pkl',
                  trim_obc_spectrum, 'S18 d'))
    items.append(('phcpy-free-algorithm/data/special-test-amoeba-boundary-rearranged.pkl',
                  None, 'S18 c,d'))

    # ---- S19 / S20: amoeba formulation --------------------------------
    for p in g('Amoeba-calculation/data', 'paper-Amoeba-E_*.pkl'):
        items.append((p, None, 'S19'))
    for p in g('puncture-calculation/data', 'paper-non-Bloch-sol-E_*.pkl'):
        items.append((p, None, 'S20 (plot_non_Bloch_sols)'))

    return items


# --------------------------------------------------------------------------


def human(n):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n < 1024 or unit == 'GB':
            return '%.1f %s' % (n, unit)
        n /= 1024.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_src = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=default_src,
                    help='repository root holding the full data (default: this repo)')
    ap.add_argument('--dst', required=True,
                    help='directory to write the pruned data tree into')
    ap.add_argument('--dry-run', action='store_true',
                    help='only report; copy nothing')
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    manifest = build_manifest(src)

    missing = [p for p, _, _ in manifest if not os.path.isfile(os.path.join(src, p))]
    rows, size_before, size_after = [], 0, 0

    for rel, rule, reason in manifest:
        s = os.path.join(src, rel)
        if not os.path.isfile(s):
            continue
        before = os.path.getsize(s)
        d = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)

        trimmed, note = False, 'kept as is'
        if rule is not None:
            try:
                with open(s, 'rb') as fp:
                    obj = pickle.load(fp)
                trimmed_obj = rule(obj)
                if trimmed_obj is not None:
                    if args.dry_run:
                        after = len(pickle.dumps(trimmed_obj, protocol=4))
                    else:
                        with open(d, 'wb') as fp:
                            pickle.dump(trimmed_obj, fp, protocol=4)
                        after = os.path.getsize(d)
                    trimmed, note = True, 'trimmed'
            except Exception:                             # keep the original
                trimmed, note = False, 'trim rule not applicable, kept as is'
        if not trimmed:
            after = before
            if not args.dry_run:
                shutil.copy2(s, d)
        rows.append((rel, reason, before, after, note))
        size_before += before
        size_after += after

    print('%-62s %10s %10s  %s' % ('file', 'before', 'after', 'note'))
    for rel, reason, before, after, note in rows:
        print('%-62s %10s %10s  %s' % (rel, human(before), human(after), note))
    print('-' * 100)
    print('%d files | %s -> %s (%.1f%% of the original)'
          % (len(rows), human(size_before), human(size_after),
             100.0 * size_after / size_before if size_before else 0.0))

    if missing:
        print('\nMISSING source files (%d):' % len(missing))
        for p in missing:
            print('   ', p)

    if not args.dry_run:
        report = os.path.join(os.path.dirname(dst), 'prune-report.md')
        with open(report, 'w', encoding='utf-8') as fp:
            fp.write('# Pruned data set for the figures\n\n')
            fp.write('generated by `tools/prune-figure-data.py`\n\n')
            fp.write('%d files, %s -> %s\n\n' % (len(rows), human(size_before), human(size_after)))
            fp.write('| file | used by | size (pruned) | note |\n|---|---|---|---|\n')
            for rel, reason, before, after, note in rows:
                fp.write('| `%s` | %s | %s | %s |\n' % (rel, reason, human(after), note))
        print('\nreport written to', report)

    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
