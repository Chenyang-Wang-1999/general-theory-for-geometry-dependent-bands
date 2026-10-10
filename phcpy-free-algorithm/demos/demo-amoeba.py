'''
Demo: brute_force_amoeba — amoebic spectrum calculation.

Demonstrates the full pipeline:
  1. get_hungarian_sorted_roots  – continuous beta2 root tracks
  2. get_a2_average_winding      – average winding in a2 direction
  3. get_a1_average_winding      – average winding in a1 direction
  4. bisect_a2_winding           – find mu2 where a2 winding crosses target
  5. bisect_amoeba_ronkin_min    – find (mu1, mu2) where both windings vanish
'''

import sys
sys.path.insert(0, "../")               # phcpy-free-algorithm
sys.path.insert(0, "../brute_force_solver")

import numpy as np
import poly_tools as pt

from brute_force_amoeba.amoeba import (
    get_hungarian_sorted_roots,
    get_a2_average_winding,
    get_a1_average_winding,
    _find_mu2_for_a2_zero,
    bisect_amoeba_ronkin_min,
    _compute_crossings_and_winding,
)

# ---------------------------------------------------------------------------
# Build test models
# ---------------------------------------------------------------------------
# CLaurent triplet format: [E_exponent, beta1_exponent, beta2_exponent]


def build_square_lattice() -> pt.CLaurent:
    r"""f = E - beta1 - beta1^{-1} - beta2 - beta2^{-1}.

    Nearest-neighbour hopping on a square lattice.
    """
    coeffs, degs = [], []
    coeffs.append(1.0);   degs.extend([1, 0, 0])    # +E
    coeffs.append(-1.0);  degs.extend([0, 1, 0])    # -beta1
    coeffs.append(-1.0);  degs.extend([0, -1, 0])   # -beta1^{-1}
    coeffs.append(-1.0);  degs.extend([0, 0, 1])    # -beta2
    coeffs.append(-1.0);  degs.extend([0, 0, -1])   # -beta2^{-1}
    return _make_char_poly(coeffs, degs)


def build_square_with_nnn(g: float = 0.3) -> pt.CLaurent:
    r"""f = E - beta1 - beta1^{-1} - beta2 - beta2^{-1}
            - g*(beta1*beta2 + beta1^{-1}*beta2^{-1}).

    Square lattice with next-nearest-neighbour coupling g.
    """
    coeffs, degs = [], []
    coeffs.append(1.0);   degs.extend([1, 0, 0])
    coeffs.append(-1.0);  degs.extend([0, 1, 0])
    coeffs.append(-1.0);  degs.extend([0, -1, 0])
    coeffs.append(-1.0);  degs.extend([0, 0, 1])
    coeffs.append(-1.0);  degs.extend([0, 0, -1])
    coeffs.append(-g);    degs.extend([0, 1, 1])
    coeffs.append(-g);    degs.extend([0, -1, -1])
    return _make_char_poly(coeffs, degs)


def _make_char_poly(coeffs, degs):
    c = pt.CScalarVec(np.array(coeffs, dtype=complex))
    d = pt.CLaurentIndexVec(np.array(degs, dtype=np.int32))
    cp = pt.CLaurent(3)
    cp.set_Laurent_by_terms(c, d)
    return cp


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_winding_sweep():
    """Sweep mu2 at fixed mu1 and compare a1 and a2 windings."""
    print("=" * 60)
    print("Demo 1 — winding sweep (square lattice, E=1, mu1=0.3)")
    print("=" * 60)
    cp = build_square_lattice()
    E = 1.0 + 0j
    mu1 = 0.3

    for mu2 in [-0.5, -0.2, 0.0, 0.2, 0.5]:
        _, zeros, cont = _compute_crossings_and_winding(
            cp, E, mu1, mu2, N_points=201)
        w1 = get_a1_average_winding(cp, E, mu1, mu2, N_points=201)
        w2 = get_a2_average_winding(cp, E, mu1, mu2, N_points=201)
        print(f"  mu2={mu2:+.1f}  a1={w1:+.6f}  a2={w2:+.6f}"
              f"  zeros={len(zeros)}  continuum={cont}")


def demo_hungarian_roots():
    """Show Hungarian-sorted root tracks."""
    print("\n" + "=" * 60)
    print("Demo 2 — Hungarian-sorted beta2 roots (square, E=1, mu1=0)")
    print("=" * 60)
    cp = build_square_lattice()
    theta1, tracked, M, N = get_hungarian_sorted_roots(
        cp, 1.0 + 0j, 0.0, N_points=51)

    print(f"  M={M}, N={N}  (total {M + N} beta2 roots)")
    print(f"  theta1 mesh: {len(theta1)} points, [0, 2π)")
    print(f"  Sample |beta2| values:")
    for k in range(0, len(theta1), 10):
        t1 = theta1[k]
        abs_vals = np.abs(tracked[k])
        print(f"    theta1={t1:.4f}  |beta2|={abs_vals}")


def demo_bisect_a2():
    """Bisect mu2 to find where a2 winding crosses 0."""
    print("\n" + "=" * 60)
    print("Demo 3 — bisect mu2 for a2=0 (square, E=1, mu1=0.3)")
    print("=" * 60)
    cp = build_square_lattice()

    result = _find_mu2_for_a2_zero(
        cp, 1.0 + 0j, 0.3, -2.0, 2.0, target_winding=0.0,
        N_points=201, max_range_expansions=5)

    print(f"  mu2       = {result['mu2']:.8f}")
    print(f"  continuum = {result['is_continuum']}")
    print(f"  winding   = {result['winding']}")
    print(f"  success   = {result['success']}")
    print(f"  n_zeros   = {len(result['zeros'])}")
    if result['zeros']:
        for t1, t2 in result['zeros'][:4]:
            print(f"    (theta1={t1:.6f}, theta2={t2:.6f})")


def demo_ronkin_min():
    """Find the Ronkin function minimum (a1 = a2 = 0)."""
    print("\n" + "=" * 60)
    print("Demo 4 — Ronkin minimum (square, E=1)")
    print("=" * 60)
    cp = build_square_lattice()

    result = bisect_amoeba_ronkin_min(
        cp, 1.0 + 0j, -0.5, 0.5, -2.0, 2.0, N_points=201, max_iter=40)

    print(f"  mu1        = {result['mu1']:.8f}")
    print(f"  mu2        = {result['mu2']:.8f}")
    print(f"  is_continuum = {result['is_continuum']}")
    print(f"  success    = {result['success']}")
    print(f"  n_zeros    = {len(result['zeros'])}")

    if result['success']:
        w1 = get_a1_average_winding(
            cp, 1.0 + 0j, result['mu1'], result['mu2'], N_points=201)
        w2 = get_a2_average_winding(
            cp, 1.0 + 0j, result['mu1'], result['mu2'], N_points=201)
        print(f"  verify a1  = {w1:.8f}")
        print(f"  verify a2  = {w2:.8f}")


def demo_nnn_model():
    """Demo with next-nearest-neighbour coupling."""
    print("\n" + "=" * 60)
    print("Demo 5 — NNN model Ronkin minimum (g=0.3, E=1)")
    print("=" * 60)
    cp = build_square_with_nnn(0.3)

    result = bisect_amoeba_ronkin_min(
        cp, 1.0 + 0j, -0.5, 0.5, -2.0, 2.0, N_points=201, max_iter=40)

    print(f"  mu1        = {result['mu1']:.8f}")
    print(f"  mu2        = {result['mu2']:.8f}")
    print(f"  is_continuum = {result['is_continuum']}")
    print(f"  success    = {result['success']}")
    print(f"  n_zeros    = {len(result['zeros'])}")


if __name__ == "__main__":
    demo_winding_sweep()
    demo_hungarian_roots()
    demo_bisect_a2()
    demo_ronkin_min()
    demo_nnn_model()
    print("\nDone.")
