"""
Standalone density-of-states computation via the preimage area method.

Given a 2D meshgrid in the complex energy plane and a function theta_fun(E)
that returns the preimage set in k-space, compute the k-space area enclosed
by the preimage of each grid cell boundary.

No dependencies on other project files.  Only requires numpy, scipy.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from typing import Callable, List, Tuple

# ==============================================================================
# 1.  Torus geometry helpers
# ==============================================================================

def torus_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> float:
    dx = abs(p1[0] - p2[0]); dx = min(dx, Lx - dx)
    dy = abs(p1[1] - p2[1]); dy = min(dy, Ly - dy)
    return np.sqrt(dx * dx + dy * dy)


# ==============================================================================
# 2.  Point-set matching (Hungarian + birth/death pairing)
# ==============================================================================

def _hungarian_match(cost: np.ndarray) -> Tuple[List[int], List[int]]:
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(row_ind), list(col_ind)


def _pair_points(
    points: List[Tuple[float, float]],
    idxs: List[int],
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> List[Tuple[int, int]]:
    """Minimum-weight perfect matching for a small set of indices."""
    k = len(idxs)
    if k == 0:   return []
    if k == 2:   return [(idxs[0], idxs[1])]

    best_pairs = None
    best_cost  = float("inf")

    def _generate(remaining, current):
        nonlocal best_pairs, best_cost
        if not remaining:
            cost = sum(torus_distance(points[a], points[b], Lx, Ly)
                       for a, b in current)
            if cost < best_cost:
                best_cost = cost; best_pairs = current[:]
            return
        a = remaining[0]
        for j in range(1, len(remaining)):
            b = remaining[j]
            new_rem = [remaining[r] for r in range(1, len(remaining)) if r != j]
            _generate(new_rem, current + [(a, b)])

    _generate(idxs, [])
    return best_pairs


def match_sets(
    set1: List[Tuple[float, float]],
    set2: List[Tuple[float, float]],
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> Tuple[dict, dict, list, list]:
    """Match two consecutive preimage sets on a torus.

    Returns (forward_edges, backward_edges, birth_pairs, death_pairs).
    Handles arbitrary even count differences.
    """
    m, n = len(set1), len(set2)
    if m == 0 and n == 0:
        return {}, {}, [], []

    if m == n:
        cost = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                cost[i, j] = torus_distance(set1[i], set2[j], Lx, Ly)
        row_ind, col_ind = _hungarian_match(cost)
        fwd = {int(row_ind[i]): int(col_ind[i]) for i in range(m)}
        bwd = {int(col_ind[i]): int(row_ind[i]) for i in range(m)}
        return fwd, bwd, [], []

    size = max(m, n)
    cost = np.full((size, size), 1e10)
    for i in range(m):
        for j in range(n):
            cost[i, j] = torus_distance(set1[i], set2[j], Lx, Ly)
    row_ind, col_ind = _hungarian_match(cost)

    fwd, bwd = {}, {}
    matched_set1, matched_set2 = set(), set()
    for i in range(m):
        j = int(col_ind[i])
        if j < n:
            fwd[i] = j; bwd[j] = i
            matched_set1.add(i); matched_set2.add(j)

    unmatched1 = [i for i in range(m) if i not in matched_set1]
    unmatched2 = [j for j in range(n) if j not in matched_set2]

    diff = n - m
    if diff > 0:
        assert diff % 2 == 0, f"Odd birth count {diff}"
        return fwd, bwd, _pair_points(set2, unmatched2, Lx, Ly), []
    elif diff < 0:
        assert (-diff) % 2 == 0, f"Odd death count {-diff}"
        return fwd, bwd, [], _pair_points(set1, unmatched1, Lx, Ly)
    return fwd, bwd, [], []

# ==============================================================================
# 4.  EdgeBundle — preimage matching data along one grid edge
# ==============================================================================

@dataclass
class EdgeBundle:
    """Raw preimage matching along a directed edge in E-space.

    samples[k]     : preimage point set at sample k
    fwd_edges[k]   : dict {idx_in_k → idx_in_{k+1}} from Hungarian matching
    birth_pairs[k] : pairs of indices in sample_{k+1} that newly appear
    death_pairs[k] : pairs of indices in sample_k that disappear
    """
    samples: List[List[Tuple[float, float]]] = field(default_factory=list)
    fwd_edges: List[dict] = field(default_factory=list)
    birth_pairs: List[List[Tuple[int, int]]] = field(default_factory=list)
    death_pairs: List[List[Tuple[int, int]]] = field(default_factory=list)

    @property
    def n_samples(self) -> int: return len(self.samples)

    def reverse(self) -> "EdgeBundle":
        """Return the same edge traversed in the opposite direction."""
        M = self.n_samples
        rev_samples = list(reversed(self.samples))
        rev_fwd, rev_birth, rev_death = [], [], []
        for t in range(M - 1):
            orig_t = M - 2 - t
            orig_fwd = self.fwd_edges[orig_t]
            rev_fwd.append({v: k for k, v in orig_fwd.items()})
            rev_birth.append(list(self.death_pairs[orig_t]))
            rev_death.append(list(self.birth_pairs[orig_t]))
        return EdgeBundle(rev_samples, rev_fwd, rev_birth, rev_death)


def compute_edge_bundle(
    E_start: complex,
    E_end: complex,
    theta_fun: Callable[[complex], List[Tuple[float, float]]],
    n_samples: int,
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> EdgeBundle:
    """Compute preimage matching data along an E-space line segment."""
    samples = []
    for k in range(n_samples):
        t = k / (n_samples - 1) if n_samples > 1 else 0.0
        samples.append(theta_fun(E_start + t * (E_end - E_start)))

    fwd_edges, birth_pairs, death_pairs = [], [], []
    for k in range(n_samples - 1):
        fwd, _bwd, births, deaths = match_sets(
            samples[k], samples[k + 1], Lx, Ly)
        fwd_edges.append(fwd); birth_pairs.append(births); death_pairs.append(deaths)

    return EdgeBundle(samples, fwd_edges, birth_pairs, death_pairs)


# ==============================================================================
# 5.  Cell loop assembly (4 edges → closed loops)
# ==============================================================================

def _assign_node_ids(bundle: EdgeBundle, offset: int
                     ) -> Tuple[dict, dict, int]:
    node_to_id, id_to_info = {}, {}
    nid = offset
    for s, pts in enumerate(bundle.samples):
        for p, (kx, ky) in enumerate(pts):
            node_to_id[(s, p)] = nid
            id_to_info[nid] = (s, p, kx, ky)
            nid += 1
    return node_to_id, id_to_info, nid


def _assemble_cell_loops(
    bottom: EdgeBundle,
    right: EdgeBundle,
    top_rev: EdgeBundle,
    left_rev: EdgeBundle,
) -> List[List[Tuple[float, float]]]:
    """Combine 4 edge bundles into closed loops for one grid cell.

    Polygon (counterclockwise): bottom(left→right), right(bottom→top),
    top(right→left, reversed), left(top→bottom, reversed).
    Corners are matched by preimage index (deterministic theta_fun order).
    """
    edges = [bottom, right, top_rev, left_rev]

    all_node_to_id, all_id_to_info = [], {}
    next_id = 0
    for b in edges:
        n2i, i2n, nid = _assign_node_ids(b, next_id)
        all_node_to_id.append(n2i); all_id_to_info.update(i2n); next_id = nid

    adj = {nid: set() for nid in range(next_id)}

    # horizontal + vertical edges within each bundle
    for e_idx, bundle in enumerate(edges):
        n2i = all_node_to_id[e_idx]; M = bundle.n_samples
        for t in range(M - 1):
            for src_p, tgt_p in bundle.fwd_edges[t].items():
                sid = n2i.get((t, src_p)); tid = n2i.get((t + 1, tgt_p))
                if sid is not None and tid is not None:
                    adj[sid].add(tid); adj[tid].add(sid)
            for a, b in bundle.birth_pairs[t]:
                aid = n2i.get((t + 1, a)); bid = n2i.get((t + 1, b))
                if aid is not None and bid is not None:
                    adj[aid].add(bid); adj[bid].add(aid)
            for a, b in bundle.death_pairs[t]:
                aid = n2i.get((t, a)); bid = n2i.get((t, b))
                if aid is not None and bid is not None:
                    adj[aid].add(bid); adj[bid].add(aid)

    # corner connections (by index equality)
    corners = [
        (0, 0,                        3, edges[3].n_samples - 1),  # BL
        (0, edges[0].n_samples - 1,   1, 0),                        # BR
        (1, edges[1].n_samples - 1,   2, 0),                        # TR
        (2, edges[2].n_samples - 1,   3, 0),                        # TL
    ]
    for e_a, s_a, e_b, s_b in corners:
        n2i_a = all_node_to_id[e_a]; n2i_b = all_node_to_id[e_b]
        n_common = min(len(edges[e_a].samples[s_a]),
                       len(edges[e_b].samples[s_b]))
        for p in range(n_common):
            id_a = n2i_a.get((s_a, p)); id_b = n2i_b.get((s_b, p))
            if id_a is not None and id_b is not None:
                adj[id_a].add(id_b); adj[id_b].add(id_a)

    # trace cycles
    visited, loops = set(), []
    for start_id in range(next_id):
        if start_id in visited or start_id not in adj:
            continue
        loop, cur, prev = [], start_id, None
        while cur is not None and cur not in visited:
            if cur not in all_id_to_info: break
            visited.add(cur)
            _, _, kx, ky = all_id_to_info[cur]
            loop.append((kx, ky))
            nbrs = [n for n in adj.get(cur, set()) if n != prev]
            if not nbrs: break
            prev, cur = cur, nbrs[0]
        if len(loop) >= 2:
            loops.append(loop)
    return loops


# ==============================================================================
# 6.  Torus-aware area of a closed loop
# ==============================================================================

def _unwrap_loop(
    loop: List[Tuple[float, float]],
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> Tuple[List[Tuple[float, float]], int, int]:
    """Lift loop from torus to R². Returns (unwrapped_pts, wind_x, wind_y)."""
    if not loop: return [], 0, 0
    uw = [(float(loop[0][0]), float(loop[0][1]))]
    for k in range(1, len(loop)):
        dx = loop[k][0] - uw[-1][0]; dy = loop[k][1] - uw[-1][1]
        uw.append((loop[k][0] - round(dx / Lx) * Lx,
                   loop[k][1] - round(dy / Ly) * Ly))
    wx = round((uw[-1][0] - uw[0][0]) / Lx)
    wy = round((uw[-1][1] - uw[0][1]) / Ly)
    return uw, wx, wy


def _shoelace_area(points: List[Tuple[float, float]]) -> float:
    n = len(points)
    if n < 3: return 0.0
    a = 0.0
    for k in range(n):
        x1, y1 = points[k]; x2, y2 = points[(k + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _point_in_polygon(pt: Tuple[float, float],
                      polygon: List[Tuple[float, float]]) -> bool:
    x, y = pt; inside = False; n = len(polygon); j = n - 1
    for i in range(n):
        xi, yi = polygon[i]; xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _compute_loop_area(
    loop: List[Tuple[float, float]],
    test_pts: List[Tuple[float, float]],
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> float:
    """Area enclosed by one loop on a torus.

    Test points (preimage of cell centre) decide which side is interior.
    """
    if len(loop) < 3: return 0.0
    uw, wx, wy = _unwrap_loop(loop, Lx, Ly)
    signed = _shoelace_area(uw)
    abs_a = abs(signed); T = Lx * Ly
    if abs_a < 1e-15: return 0.0

    # winding or no test points → take the smaller region
    if wx != 0 or wy != 0 or len(test_pts) == 0:
        return min(abs_a, T - abs_a)

    cx = sum(p[0] for p in uw) / len(uw)
    cy = sum(p[1] for p in uw) / len(uw)
    for tx, ty in test_pts:
        sx = round((cx - tx) / Lx); sy = round((cy - ty) / Ly)
        if _point_in_polygon((tx + sx * Lx, ty + sy * Ly), uw):
            return abs_a
    return T - abs_a


# ==============================================================================
# 7.  Main entry point — row-by-row DoS grid computation
# ==============================================================================

def compute_dos_grid(
    E_r: np.ndarray,
    E_im: np.ndarray,
    theta_fun: Callable[[complex], List[Tuple[float, float]]],
    n_edge: int = 30,
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
    verbose: bool = True,
) -> np.ndarray:
    """Compute k-space preimage area for every cell of an E-space meshgrid.

    Parameters
    ----------
    E_r  : real-axis grid points, shape (Nx,)
    E_im : imag-axis grid points, shape (Ny,)
    theta_fun : complex E → list of (kx, ky) preimage points
    n_edge : samples per grid edge
    Lx, Ly : k-space torus periods (default 2π)

    Returns
    -------
    dos : shape (Ny-1, Nx-1), k-space area per E-cell
    """
    Nx, Ny = len(E_r), len(E_im)
    dos = np.full((Ny - 1, Nx - 1), np.nan)

    # --- corner preimages (reused by all incident edges) ---
    if verbose:
        print(f"Precomputing corner preimages ({Nx}×{Ny} grid) …")
    corner_pts = {}
    for j in range(Ny):
        for i in range(Nx):
            corner_pts[(i, j)] = theta_fun(complex(E_r[i], E_im[j]))

    # --- first row bottom edges ---
    if verbose:
        print("Computing bottom-row horizontal edges …")
    bottom_row = []
    for i in range(Nx - 1):
        bottom_row.append(compute_edge_bundle(
            complex(E_r[i], E_im[0]), complex(E_r[i + 1], E_im[0]),
            theta_fun, n_edge, Lx, Ly))

    # --- row-by-row sweep ---
    for j in range(Ny - 1):
        if verbose:
            print(f"Row {j+1}/{Ny-1}: Im(E) = {E_im[j]:.3f} → {E_im[j+1]:.3f}")

        # top horizontal edges
        top_row = []
        for i in range(Nx - 1):
            top_row.append(compute_edge_bundle(
                complex(E_r[i], E_im[j + 1]), complex(E_r[i + 1], E_im[j + 1]),
                theta_fun, n_edge, Lx, Ly))

        # vertical edges
        vert_cols = []
        for i in range(Nx):
            vert_cols.append(compute_edge_bundle(
                complex(E_r[i], E_im[j]), complex(E_r[i], E_im[j + 1]),
                theta_fun, n_edge, Lx, Ly))

        # assemble each cell
        for i in range(Nx - 1):
            bottom  = bottom_row[i]
            right   = vert_cols[i + 1]
            top_rev = top_row[i].reverse()
            left_rev = vert_cols[i].reverse()

            total_pts = (sum(len(s) for s in bottom.samples) +
                         sum(len(s) for s in right.samples) +
                         sum(len(s) for s in top_rev.samples) +
                         sum(len(s) for s in left_rev.samples))
            if total_pts == 0:
                dos[j, i] = 0.0
                continue

            loops = _assemble_cell_loops(bottom, right, top_rev, left_rev)

            Ec = complex(E_r[i] + 0.5 * (E_r[i + 1] - E_r[i]),
                         E_im[j] + 0.5 * (E_im[j + 1] - E_im[j]))
            test_pts = theta_fun(Ec)

            area = 0.0
            for loop in loops:
                area += _compute_loop_area(loop, test_pts, Lx, Ly)
            dos[j, i] = area

        bottom_row = top_row   # shift cache

    return dos
