"""
Density of states via preimage area method on a 2D E-space meshgrid.

Computes the k-space area enclosed by the preimage of each grid cell boundary,
reusing shared edges between adjacent cells via row-by-row sweep.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional
import sys
sys.path.append("../paper/HN-model")
import HN_model_common as HN

from preimage_loops import match_sets, torus_distance, make_test_theta_fun


# ==============================================================================
# EdgeBundle — preimage matching data along one grid edge
# ==============================================================================

@dataclass
class EdgeBundle:
    """Raw preimage matching data along a directed edge in E-space.

    samples[k] : preimage point set at sample k along the edge
    fwd_edges[k] : dict {idx_in_k: idx_in_{k+1}} from Hungarian matching
    birth_pairs[k] : pairs of indices in sample_{k+1} that newly appear
    death_pairs[k] : pairs of indices in sample_k that disappear
    """
    samples: List[List[Tuple[float, float]]] = field(default_factory=list)
    fwd_edges: List[dict] = field(default_factory=list)
    birth_pairs: List[List[Tuple[int, int]]] = field(default_factory=list)
    death_pairs: List[List[Tuple[int, int]]] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_transitions(self) -> int:
        return len(self.fwd_edges)

    def reverse(self) -> "EdgeBundle":
        """Return a new EdgeBundle representing the same edge traversed backwards."""
        M = self.n_samples
        rev_samples = list(reversed(self.samples))

        rev_fwd = []
        rev_birth = []
        rev_death = []

        for t in range(M - 1):
            # Original transition t: sample[t] -> sample[t+1]
            # Reversed transition corresponds to original transition M-2-t
            # from sample[M-1-t] to sample[M-2-t]
            # Backward mapping is the inverse of fwd_edges[M-2-t]
            orig_t = M - 2 - t
            orig_fwd = self.fwd_edges[orig_t]
            inv_fwd = {v: k for k, v in orig_fwd.items()}
            rev_fwd.append(inv_fwd)

            # Birth at transition t (rev) = death at transition orig_t (orig)
            rev_birth.append(list(self.death_pairs[orig_t]))

            # Death at transition t (rev) = birth at transition orig_t (orig)
            rev_death.append(list(self.birth_pairs[orig_t]))

        return EdgeBundle(
            samples=rev_samples,
            fwd_edges=rev_fwd,
            birth_pairs=rev_birth,
            death_pairs=rev_death,
        )

    def __repr__(self) -> str:
        n_pts = sum(len(s) for s in self.samples)
        return (f"EdgeBundle(samples={self.n_samples}, "
                f"total_pts={n_pts}, "
                f"births={sum(len(b) for b in self.birth_pairs)}, "
                f"deaths={sum(len(d) for d in self.death_pairs)})")


# ==============================================================================
# Edge preimage computation
# ==============================================================================

def compute_edge_bundle(
    E_start: complex,
    E_end: complex,
    theta_fun: Callable[[complex], List[Tuple[float, float]]],
    n_samples: int,
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> EdgeBundle:
    """Compute preimage matching data along a line segment in E-space.

    Samples n_samples points (including both endpoints), evaluates theta_fun
    at each, and runs match_sets between consecutive samples.
    """
    # Sample points along the edge
    samples = []
    for k in range(n_samples):
        t = k / (n_samples - 1) if n_samples > 1 else 0.0
        E = E_start + t * (E_end - E_start)
        pts = theta_fun(E)
        samples.append(pts)

    # Match consecutive sample sets
    fwd_edges = []
    birth_pairs = []
    death_pairs = []

    for k in range(n_samples - 1):
        fwd, _bwd, births, deaths = match_sets(
            samples[k], samples[k + 1], Lx, Ly
        )
        fwd_edges.append(fwd)
        birth_pairs.append(births)
        death_pairs.append(deaths)

    return EdgeBundle(
        samples=samples,
        fwd_edges=fwd_edges,
        birth_pairs=birth_pairs,
        death_pairs=death_pairs,
    )


# ==============================================================================
# Cell loop assembly
# ==============================================================================

def _assign_node_ids(bundle: EdgeBundle, offset: int, reversed_dir: bool = False
                     ) -> Tuple[dict, dict, int]:
    """Assign unique integer IDs to every preimage point in an EdgeBundle.

    Returns (node_to_id, id_to_info, next_id) where
      id_to_info[nid] = (sample_idx, point_idx, kx, ky)
    """
    node_to_id = {}
    id_to_info = {}
    nid = offset
    M = bundle.n_samples

    for s in range(M):
        pts = bundle.samples[s]
        for p in range(len(pts)):
            node_to_id[(s, p)] = nid
            id_to_info[nid] = (s, p, pts[p][0], pts[p][1])
            nid += 1

    return node_to_id, id_to_info, nid


def _assemble_cell_loops(
    bottom: EdgeBundle,
    right: EdgeBundle,
    top_rev: EdgeBundle,
    left_rev: EdgeBundle,
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> List[List[Tuple[float, float]]]:
    """Combine 4 edge bundles into closed loops for one grid cell.

    The cell polygon goes counterclockwise:
      bottom:  left  → right  (forward)
      right:   bottom → top    (forward)
      top:     right  → left   (reversed)
      left:    top    → bottom (reversed)

    Corner matching: at each corner, the endpoint preimage sets of the two
    incident edges are both theta_fun(E_corner), so indices match directly.
    """
    edges = [bottom, right, top_rev, left_rev]

    # 1. Assign global node IDs
    all_node_to_id = []
    all_id_to_info = {}
    next_id = 0

    for bundle in edges:
        n2i, i2n, nid = _assign_node_ids(bundle, next_id)
        all_node_to_id.append(n2i)
        all_id_to_info.update(i2n)
        next_id = nid

    # 2. Build adjacency
    adj = {nid: set() for nid in range(next_id)}

    for e_idx, bundle in enumerate(edges):
        n2i = all_node_to_id[e_idx]
        M = bundle.n_samples

        # Horizontal edges (forward matching within this edge)
        for t in range(M - 1):
            fwd = bundle.fwd_edges[t]
            for src_p, tgt_p in fwd.items():
                src_id = n2i.get((t, src_p))
                tgt_id = n2i.get((t + 1, tgt_p))
                if src_id is not None and tgt_id is not None:
                    adj[src_id].add(tgt_id)
                    adj[tgt_id].add(src_id)

        # Birth vertical edges (within this edge)
        for t in range(M - 1):
            for a, b in bundle.birth_pairs[t]:
                id_a = n2i.get((t + 1, a))
                id_b = n2i.get((t + 1, b))
                if id_a is not None and id_b is not None:
                    adj[id_a].add(id_b)
                    adj[id_b].add(id_a)

        # Death vertical edges (within this edge)
        for t in range(M - 1):
            for a, b in bundle.death_pairs[t]:
                id_a = n2i.get((t, a))
                id_b = n2i.get((t, b))
                if id_a is not None and id_b is not None:
                    adj[id_a].add(id_b)
                    adj[id_b].add(id_a)

    # 3. Corner connections
    # Corner order: BL(bottom.start, left.end), BR(bottom.end, right.start),
    #               TR(right.end, top.start), TL(top.end, left.start)
    corners = [
        # (edge_a, sample_a, edge_b, sample_b)
        (0, 0,        3, edges[3].n_samples - 1),   # BL: bottom[0] ↔ left_rev[-1]
        (0, edges[0].n_samples - 1, 1, 0),           # BR: bottom[-1] ↔ right[0]
        (1, edges[1].n_samples - 1, 2, 0),           # TR: right[-1] ↔ top[0]
        (2, edges[2].n_samples - 1, 3, 0),           # TL: top[-1] ↔ left_rev[0]
    ]

    for e_a, s_a, e_b, s_b in corners:
        n2i_a = all_node_to_id[e_a]
        n2i_b = all_node_to_id[e_b]
        pts_a = edges[e_a].samples[s_a]
        pts_b = edges[e_b].samples[s_b]

        # Match by index: both are theta_fun at the same corner E
        n_common = min(len(pts_a), len(pts_b))
        for p in range(n_common):
            id_a = n2i_a.get((s_a, p))
            id_b = n2i_b.get((s_b, p))
            if id_a is not None and id_b is not None:
                adj[id_a].add(id_b)
                adj[id_b].add(id_a)

    # 4. Trace cycles (graph should be 2-regular)
    visited = set()
    loops = []

    for start_id in range(next_id):
        if start_id in visited or start_id not in adj:
            continue

        loop = []
        cur = start_id
        prev = None

        while cur is not None and cur not in visited:
            if cur not in all_id_to_info:
                break
            visited.add(cur)
            _, _, kx, ky = all_id_to_info[cur]
            loop.append((kx, ky))

            neighbors = [n for n in adj.get(cur, set()) if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            prev = cur
            cur = nxt

        if len(loop) >= 2:
            loops.append(loop)

    return loops


# ==============================================================================
# Torus-aware area computation
# ==============================================================================

def _unwrap_loop(
    loop: List[Tuple[float, float]],
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> Tuple[List[Tuple[float, float]], int, int]:
    """Lift a loop from torus to R² using shortest-path continuation.

    Returns (unwrapped_points, wind_x, wind_y) where wind_x, wind_y are
    the net winding numbers (integer multiples of Lx, Ly).
    """
    n = len(loop)
    if n == 0:
        return [], 0, 0

    uw = [(float(loop[0][0]), float(loop[0][1]))]
    for k in range(1, n):
        dx = loop[k][0] - uw[-1][0]
        dy = loop[k][1] - uw[-1][1]
        sx = round(dx / Lx) * Lx
        sy = round(dy / Ly) * Ly
        uw.append((loop[k][0] - sx, loop[k][1] - sy))

    # Net winding
    wx = round((uw[-1][0] - uw[0][0]) / Lx)
    wy = round((uw[-1][1] - uw[0][1]) / Ly)

    return uw, wx, wy


def _shoelace_area(points: List[Tuple[float, float]]) -> float:
    """Signed area via shoelace formula (polygon in R²)."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for k in range(n):
        x1, y1 = points[k]
        x2, y2 = points[(k + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]],
) -> bool:
    """Ray-casting point-in-polygon test (unbounded R² polygon)."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _compute_loop_area(
    loop: List[Tuple[float, float]],
    test_pts: List[Tuple[float, float]],
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> float:
    """Compute the area enclosed by one loop on a torus.

    Uses unwrap + shoelace. Test points determine which side of the loop
    is the "interior" (the side containing the preimage of the cell center).
    """
    if len(loop) < 3:
        return 0.0

    uw, wx, wy = _unwrap_loop(loop, Lx, Ly)
    signed_area = _shoelace_area(uw)
    abs_area = abs(signed_area)
    torus_area = Lx * Ly

    if abs_area < 1e-15:
        return 0.0

    # If the loop winds around the torus, or we have no test points
    # (cell center outside image), fall back to the smaller area.
    if wx != 0 or wy != 0 or len(test_pts) == 0:
        return min(abs_area, torus_area - abs_area)

    # Compute loop centroid (in R²)
    cx = sum(p[0] for p in uw) / len(uw)
    cy = sum(p[1] for p in uw) / len(uw)

    # Check if any test point (shifted to nearest sheet) is inside
    for tx, ty in test_pts:
        # Shift test point to nearest image of the loop
        sx = round((cx - tx) / Lx)
        sy = round((cy - ty) / Ly)
        shifted = (tx + sx * Lx, ty + sy * Ly)

        if _point_in_polygon(shifted, uw):
            return abs_area

    # Test points outside → the "interior" is the complement
    return torus_area - abs_area


# ==============================================================================
# DoS grid computation (row-by-row sweep)
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
    """Compute DoS on a 2D E-space meshgrid via the preimage area method.

    Parameters
    ----------
    E_r : real-axis grid points, shape (Nx,)
    E_im : imag-axis grid points, shape (Ny,)
    theta_fun : E -> list of (kx, ky) preimage points
    n_edge : samples per grid edge
    Lx, Ly : k-space torus periods

    Returns
    -------
    dos : shape (Ny-1, Nx-1), area in k-space per E-cell
    """
    Nx = len(E_r)
    Ny = len(E_im)
    dos = np.full((Ny - 1, Nx - 1), np.nan)

    # --- Precompute corner preimages ---
    if verbose:
        print(f"Precomputing corner preimages ({Nx}x{Ny} grid)...")
    corner_pts = {}
    for j in range(Ny):
        for i in range(Nx):
            E = complex(E_r[i], E_im[j])
            corner_pts[(i, j)] = theta_fun(E)

    # --- First row: compute bottom horizontal edges ---
    if verbose:
        print("Computing bottom row horizontal edges...")
    bottom_row = []
    for i in range(Nx - 1):
        E1 = complex(E_r[i], E_im[0])
        E2 = complex(E_r[i + 1], E_im[0])
        bundle = compute_edge_bundle(E1, E2, theta_fun, n_edge, Lx, Ly)
        bottom_row.append(bundle)

    # --- Row-by-row sweep ---
    for j in range(Ny - 1):
        if verbose:
            print(f"Row {j+1}/{Ny-1}: y = {E_im[j]:.3f} -> {E_im[j+1]:.3f}")

        # Compute top horizontal edges for this row
        top_row = []
        for i in range(Nx - 1):
            E1 = complex(E_r[i], E_im[j + 1])
            E2 = complex(E_r[i + 1], E_im[j + 1])
            bundle = compute_edge_bundle(E1, E2, theta_fun, n_edge, Lx, Ly)
            top_row.append(bundle)

        # Compute vertical edges for this row (Nx of them)
        vert_cols = []
        for i in range(Nx):
            E1 = complex(E_r[i], E_im[j])
            E2 = complex(E_r[i], E_im[j + 1])
            bundle = compute_edge_bundle(E1, E2, theta_fun, n_edge, Lx, Ly)
            vert_cols.append(bundle)

        # Process each cell in this row
        for i in range(Nx - 1):
            bottom = bottom_row[i]
            right = vert_cols[i + 1]
            top_rev = top_row[i].reverse()
            left_rev = vert_cols[i].reverse()

            # Skip if any edge has zero total preimage points
            total_pts = (sum(len(s) for s in bottom.samples) +
                         sum(len(s) for s in right.samples) +
                         sum(len(s) for s in top_rev.samples) +
                         sum(len(s) for s in left_rev.samples))
            if total_pts == 0:
                dos[j, i] = 0.0
                continue

            loops = _assemble_cell_loops(bottom, right, top_rev, left_rev, Lx, Ly)

            # Cell center test points
            Ec = complex(
                E_r[i] + 0.5 * (E_r[i + 1] - E_r[i]),
                E_im[j] + 0.5 * (E_im[j + 1] - E_im[j]),
            )
            test_pts = theta_fun(Ec)

            area = 0.0
            for loop in loops:
                area += _compute_loop_area(loop, test_pts, Lx, Ly)
            dos[j, i] = area

        # Shift: top becomes bottom for next row
        bottom_row = top_row

    return dos


# ==============================================================================
# Convenience: full pipeline with the test model
# ==============================================================================

def run_test_dos(
    E_r: np.ndarray = None,
    E_im: np.ndarray = None,
    n_edge: int = 30,
    Jx: complex = 1.0 + 0.0j,
    Jy: complex = 1.0 + 0.0j,
    delta_x: float = 0.0,
    delta_y: float = np.pi / 2,
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
    verbose: bool = True,
) -> np.ndarray:
    """Run DoS computation with the analytic test model."""
    if E_r is None:
        E_r = np.linspace(-2.5, 2.5, 20)
    if E_im is None:
        E_im = np.linspace(-2.5, 2.5, 20)

    theta_fun = make_test_theta_fun(Jx, Jy, delta_x, delta_y)
    dos = compute_dos_grid(E_r, E_im, theta_fun, n_edge=n_edge, Lx=Lx, Ly=Ly,
                           verbose=verbose)
    return dos


# ==============================================================================
# Standalone test
# ==============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("DoS Preimage Area Method — Test")
    print("=" * 60)

    # Test on a coarse grid
    E_r = np.linspace(-6, 6, 200)
    E_im = np.linspace(-3, 3, 200)
    n_edge = 20

    print(f"\nGrid: {len(E_r)}x{len(E_im)}, n_edge={n_edge}")

    Jx, Jy, gammax, gammay, deltax, deltay = HN.get_new_param_from_coupling_terms(*HN.Default_Model_Params)
    dos = run_test_dos(E_r, E_im, n_edge=n_edge, Jx=Jx, Jy=Jy, delta_x=deltax, delta_y=deltay)

    # Verify
    total_area = np.nansum(dos)
    expected = 4.0 * np.pi * np.pi  # Lx*Ly = 4π²
    n_cells = (len(E_r) - 1) * (len(E_im) - 1)
    avg_per_cell = total_area / n_cells

    print(f"\nResults:")
    print(f"  Total area:     {total_area:.6f}")
    print(f"  Expected (4π²):  {expected:.6f}")
    print(f"  Ratio:          {total_area / expected:.6f}")
    print(f"  Avg per cell:   {avg_per_cell:.6f}")
    print(f"  NaN cells:      {np.sum(np.isnan(dos))} / {n_cells}")

    # Identify image region (where |Re(E)| <= 2 and |Im(E)| <= 2 for test model)
    inside_mask = np.zeros((len(E_im) - 1, len(E_r) - 1), dtype=bool)
    for j in range(len(E_im) - 1):
        for i in range(len(E_r) - 1):
            Ec = complex(E_r[i] + 0.5 * (E_r[i+1] - E_r[i]),
                         E_im[j] + 0.5 * (E_im[j+1] - E_im[j]))
            if abs(Ec.real) <= 2.0 and abs(Ec.imag) <= 2.0:
                inside_mask[j, i] = True

    interior_dos = dos[inside_mask]
    if len(interior_dos) > 0:
        print(f"\nInterior cells (inside image region):")
        print(f"  Count:          {len(interior_dos)}")
        print(f"  Mean area:      {np.mean(interior_dos):.6f}")
        print(f"  Expected total: {expected:.6f}")
        print(f"  Actual total:   {np.sum(interior_dos):.6f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    dos_plot = dos.copy()
    # Convert area to state count: N = area * Lx*Ly / (2π)² = area
    im = ax.pcolormesh(E_r, E_im, dos_plot, shading="flat", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Area in k-space")
    ax.set_xlabel("Re(E)")
    ax.set_ylabel("Im(E)")
    ax.set_title(f"DoS via preimage area (grid {len(E_r)}x{len(E_im)}, n_edge={n_edge})")
    # Draw image region
    ax.plot([-2, 2, 2, -2, -2], [-2, -2, 2, 2, -2], "r--", linewidth=1, alpha=0.7)
    ax.set_aspect("equal")
    fig.savefig("data/dos_area_test.png", dpi=150)
    print("\n  -> Saved dos_area_test.png")
    print("\nDone.")
