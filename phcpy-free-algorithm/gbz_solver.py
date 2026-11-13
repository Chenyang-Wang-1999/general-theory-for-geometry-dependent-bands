'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2026-11-13
Copyright © Department of Physics, Tsinghua University. All rights reserved

The amoeba and SGBZ solvers used by the scripts of this folder, on top of the
``pygbz2d`` package (brute-force-non-hermitian).

The functions keep the names, the calling convention and the record format of
the former ``brute_force_solver`` / ``brute_force_amoeba`` packages, so that
the sweep and the plotting code around them stays the same:

    check_SGBZ(coeffs, degs, E_ref, perc=0.0, debug_mode=False)   -> dict
    check_amoeba(coeffs, degs, E_ref, perc=0.0, debug_mode=False) -> dict

Both wrap ``pygbz2d.sgbz.collect_GBZ_subsets`` / ``pygbz2d.amoeba.collect_GBZ_subsets``
and turn the returned ``GBZResult`` into a dictionary of plain Python types, so
the data files stay readable without pygbz2d installed (the repository publishes
its data as built-in types plus NumPy arrays only).  Extra keyword arguments are
forwarded to pygbz2d.

Record of an SGBZ point:

    {"success": bool, "is_PMGBZ": bool, "mu1": float,
     "PMGBZ_points": [{"theta1": float, "beta1": complex,
                       "beta2": complex, "is_continuum": False}, ...]}

``is_PMGBZ`` is ``GBZResult.is_gbz``, i.e. the reference energy lies in the
SGBZ.  pygbz2d returns the PMGBZ points themselves instead of one critical
``mu1``, so ``mu1`` is the mean of ``ln|beta1|`` over those points (for a
continuum segment it is the radius stored in the subset).

Record of an amoeba point:

    {"success": bool, "is_amoeba": bool, "mu1": float, "mu2": float,
     "is_continuum": bool}

``is_amoeba`` is ``GBZResult.is_gbz`` of the amoeba solver, and ``mu1``/``mu2``
are mean log-moduli of the returned GBZ points.
'''

from cmath import log

import numpy as np

import pygbz2d
from pygbz2d.core import LineSubset, PointSubset


def _summarize(subsets):
    '''Mean log-moduli of a list of subsets plus the continuum flag.'''
    mu1 = []
    mu2 = []
    has_continuum = False
    for subset in subsets:
        if isinstance(subset, PointSubset):
            mu1.append(float(subset.mu1))
            mu2.append(float(log(abs(subset.beta2)).real))
        else:
            has_continuum = True
            mu1.append(float(subset.mu1))
            mu2.append(float(np.mean(np.log(np.abs(subset.beta2_arr)))))
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return mean(mu1), mean(mu2), has_continuum


def sgbz_record(result):
    '''Turn a ``pygbz2d`` SGBZ result into the record used by this folder.'''
    if not result.success:
        return {"success": False, "error": result.error}

    points = []
    for subset in result.subsets:
        if isinstance(subset, PointSubset):
            points.append({
                "theta1": float(subset.theta1),
                "beta1": complex(subset.beta1),
                "beta2": complex(subset.beta2),
                "is_continuum": False,
            })
        else:
            points.append({
                "theta1_start": float(subset.theta1_start),
                "theta1_end": float(subset.theta1_end),
                "mu1": float(subset.mu1),
                "theta1": [float(value) for value in subset.theta1_arr],
                "beta2": [complex(value) for value in subset.beta2_arr],
                "is_continuum": True,
            })

    mu1, _, _ = _summarize(result.subsets)
    return {"success": True, "is_PMGBZ": bool(result.is_gbz),
            "mu1": mu1, "PMGBZ_points": points}


def amoeba_record(result):
    '''Turn a ``pygbz2d`` amoeba result into the record used by this folder.'''
    if not result.success:
        return {"success": False, "error": result.error}

    mu1, mu2, has_continuum = _summarize(result.subsets)
    return {"success": True, "is_amoeba": bool(result.is_gbz),
            "mu1": mu1, "mu2": mu2, "is_continuum": has_continuum}


def check_SGBZ(coeffs, degs, E_ref, perc=0.0, debug_mode=False, **options):
    '''SGBZ condition for one reference energy: the former ``check_SGBZ``.'''
    if perc is not None:
        print("%.2f" % (perc * 100) + r"%")
    result = pygbz2d.sgbz.collect_GBZ_subsets(
        np.asarray(coeffs), np.asarray(degs), E_ref,
        debug_mode=debug_mode, **options
    )
    return sgbz_record(result)


def check_amoeba(coeffs, degs, E_ref, perc=0.0, debug_mode=False, **options):
    '''Amoeba condition for one reference energy: the former ``check_amoeba``.'''
    if perc is not None:
        print("%.2f" % (perc * 100) + r"%")
    result = pygbz2d.amoeba.collect_GBZ_subsets(
        np.asarray(coeffs), np.asarray(degs), E_ref,
        debug_mode=debug_mode, **options
    )
    return amoeba_record(result)


def results_to_momenta(results, E_list):
    '''(E, k1, k2) for every PMGBZ point of a sweep.

    Replacement of the former ``brute_force_solver.convert_results_to_triplet``;
    it reads both the record written here and the older one, which stored the
    +/- solutions as ``beta2_pairs`` / ``beta2_sols``.
    '''
    triplets = []
    for E_ref, record in zip(E_list, results):
        if not (record.get("success") and record.get("is_PMGBZ")):
            continue
        mu1 = record.get("mu1", 0.0)
        for point in record["PMGBZ_points"]:
            if point.get("is_continuum"):
                radius = point.get("mu1", mu1)
                thetas = point.get("theta1", [])
                betas = point.get("beta2", [])
                for theta1, beta2 in zip(np.atleast_1d(thetas),
                                         np.atleast_1d(betas)):
                    triplets.append((E_ref, theta1 - 1j * radius,
                                     log(complex(beta2)) / 1j))
                continue

            theta1 = point["theta1"]
            theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
            if "beta2" in point:                       # record written here
                betas = [point["beta2"]]
            elif "beta2_pairs" in point:               # older record
                betas = [beta for pair in point["beta2_pairs"] for beta in pair]
            else:                                      # older record
                betas = [beta for part in point["beta2_sols"] for beta in part]
            for beta2 in betas:
                triplets.append((E_ref, theta1 - 1j * mu1,
                                 log(complex(beta2)) / 1j))
    return triplets
