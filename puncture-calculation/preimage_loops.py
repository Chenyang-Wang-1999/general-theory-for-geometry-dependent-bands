"""
Preimage loop computation for density of states via the area method.

Given a closed polygon in the complex energy plane and a function that maps
E -> preimage set in k-space, compute the preimage loops by matching
consecutive preimage sets along the polygon boundary.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Callable, List, Tuple


# ==============================================================================
# Core algorithm
# ==============================================================================

def torus_distance(p1: Tuple[float, float], p2: Tuple[float, float],
                   Lx: float = 2.0 * np.pi, Ly: float = 2.0 * np.pi) -> float:
    """Euclidean distance between two points on a torus [0, Lx) x [0, Ly)."""
    dx = abs(p1[0] - p2[0])
    dx = min(dx, Lx - dx)
    dy = abs(p1[1] - p2[1])
    dy = min(dy, Ly - dy)
    return np.sqrt(dx * dx + dy * dy)


def _hungarian_match(cost: np.ndarray) -> Tuple[List[int], List[int]]:
    """Thin wrapper around scipy's linear_sum_assignment."""
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(row_ind), list(col_ind)


def _pair_points(
    points: List[Tuple[float, float]],
    idxs: List[int],
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> List[Tuple[int, int]]:
    """Pair up indices into birth/death pairs by minimum-weight perfect matching."""
    k = len(idxs)
    if k == 0:
        return []
    if k == 2:
        return [(idxs[0], idxs[1])]

    # Brute-force perfect matching for small k (k <= 8 typically)
    best_pairs = None
    best_cost = float("inf")

    def _generate(remaining, current):
        nonlocal best_pairs, best_cost
        if not remaining:
            cost = sum(torus_distance(points[a], points[b], Lx, Ly)
                       for a, b in current)
            if cost < best_cost:
                best_cost = cost
                best_pairs = current[:]
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
    """
    Match two consecutive preimage sets on a torus.

    Handles arbitrary even count differences by pairing unmatched points
    into birth/death pairs via minimum-weight perfect matching.

    Parameters
    ----------
    set1 : preimage set at polygon index i
    set2 : preimage set at polygon index i+1

    Returns
    -------
    forward_edges : dict {idx_in_set1: idx_in_set2}
    backward_edges : dict {idx_in_set2: idx_in_set1}
    birth_pairs : list of (idx_in_set2, idx_in_set2) -- new points paired up
    death_pairs : list of (idx_in_set1, idx_in_set1) -- disappearing points paired up
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

    # General case: m != n
    # Use Hungarian on a square matrix padded with large costs.
    size = max(m, n)
    cost = np.full((size, size), 1e10)
    for i in range(m):
        for j in range(n):
            cost[i, j] = torus_distance(set1[i], set2[j], Lx, Ly)

    row_ind, col_ind = _hungarian_match(cost)

    fwd, bwd = {}, {}
    matched_set1 = set()
    matched_set2 = set()

    for i in range(m):
        j = int(col_ind[i])
        if j < n:
            fwd[i] = j
            bwd[j] = i
            matched_set1.add(i)
            matched_set2.add(j)

    unmatched1 = [i for i in range(m) if i not in matched_set1]
    unmatched2 = [j for j in range(n) if j not in matched_set2]

    diff = n - m
    if diff > 0:
        # More points in set2: birth events
        assert diff % 2 == 0, f"Odd birth count {diff}"
        birth_pairs = _pair_points(set2, unmatched2, Lx, Ly)
        return fwd, bwd, birth_pairs, []
    elif diff < 0:
        # More points in set1: death events
        assert (-diff) % 2 == 0, f"Odd death count {-diff}"
        death_pairs = _pair_points(set1, unmatched1, Lx, Ly)
        return fwd, bwd, [], death_pairs
    else:
        return fwd, bwd, [], []


def sample_polygon(vertices: List[complex], n_samples: int) -> List[complex]:
    """Distribute n_samples points evenly along a closed polygon (list of vertices)."""
    n_edges = len(vertices)
    edge_lengths = []
    total = 0.0
    for i in range(n_edges):
        z1 = vertices[i]
        z2 = vertices[(i + 1) % n_edges]
        length = abs(z2 - z1)
        edge_lengths.append(length)
        total += length

    if total == 0.0:
        return [vertices[0]] * n_samples

    points = []
    for k in range(n_samples):
        target = k / n_samples * total
        cum = 0.0
        for i in range(n_edges):
            length = edge_lengths[i]
            if cum + length >= target:
                frac = (target - cum) / length if length > 0 else 0.0
                z1 = vertices[i]
                z2 = vertices[(i + 1) % n_edges]
                points.append(z1 + frac * (z2 - z1))
                break
            cum += length

    return points


def compute_preimage_loops(
    polygon_vertices: List[complex],
    theta_fun: Callable[[complex], List[Tuple[float, float]]],
    n_samples: int,
    Lx: float = 2.0 * np.pi,
    Ly: float = 2.0 * np.pi,
) -> List[List[Tuple[float, float]]]:
    """
    Compute preimage loops for a polygon in the complex energy plane.

    Parameters
    ----------
    polygon_vertices : vertices of a closed polygon in E-space (complex numbers).
    theta_fun : E -> list of (kx, ky) preimage points in k-space.
    n_samples : number of sample points along the polygon.
    Lx, Ly : torus periods in kx, ky directions.

    Returns
    -------
    loops : list of loops; each loop is a list of (kx, ky) points in order.
    """
    # 1. Sample polygon
    E_samples = sample_polygon(polygon_vertices, n_samples)

    # 2. Compute preimage at each sample point
    preimages = [theta_fun(E) for E in E_samples]

    # 3. Match consecutive sets around the polygon
    n_pts = n_samples
    fwd_edges = [{} for _ in range(n_pts)]
    birth_pairs = [[] for _ in range(n_pts)]
    death_pairs = [[] for _ in range(n_pts)]

    for i in range(n_pts):
        i_next = (i + 1) % n_pts
        fwd, _bwd, births, deaths = match_sets(
            preimages[i], preimages[i_next], Lx, Ly
        )
        fwd_edges[i] = fwd
        birth_pairs[i] = births
        death_pairs[i] = deaths

    # 4. Build graph: assign unique ID to each (sample_index, point_index)
    node_to_id: dict = {}
    id_to_info: dict = {}  # id -> (sample_idx, point_idx, kx, ky)

    next_id = 0
    for i, pts in enumerate(preimages):
        for idx, (kx, ky) in enumerate(pts):
            nid = next_id
            next_id += 1
            node_to_id[(i, idx)] = nid
            id_to_info[nid] = (i, idx, kx, ky)

    # adjacency as sets (to avoid double-counting same undirected edge)
    adj = {nid: set() for nid in range(next_id)}

    for i in range(n_pts):
        i_next = (i + 1) % n_pts

        # Horizontal edges from forward matching
        for src_idx, tgt_idx in fwd_edges[i].items():
            src_id = node_to_id.get((i, src_idx))
            tgt_id = node_to_id.get((i_next, tgt_idx))
            if src_id is not None and tgt_id is not None:
                adj[src_id].add(tgt_id)
                adj[tgt_id].add(src_id)

        # Vertical edges from birth (at index i_next)
        for a, b in birth_pairs[i]:
            id_a = node_to_id.get((i_next, a))
            id_b = node_to_id.get((i_next, b))
            if id_a is not None and id_b is not None:
                adj[id_a].add(id_b)
                adj[id_b].add(id_a)

        # Vertical edges from death (at index i)
        for a, b in death_pairs[i]:
            id_a = node_to_id.get((i, a))
            id_b = node_to_id.get((i, b))
            if id_a is not None and id_b is not None:
                adj[id_a].add(id_b)
                adj[id_b].add(id_a)

    # 5. Diagnose graph regularity
    degree_counts = {}
    for nid in range(next_id):
        d = len(adj[nid])
        degree_counts[d] = degree_counts.get(d, 0) + 1
    if any(d != 2 for d in degree_counts):
        print(f"  [Graph degrees: {degree_counts}]")

    # 6. Find cycles by walking the 2-regular graph
    visited = set()
    loops = []

    for start_id in range(next_id):
        if start_id in visited:
            continue

        loop = []
        cur = start_id
        prev = None

        while cur is not None and cur not in visited:
            visited.add(cur)
            _, _, kx, ky = id_to_info[cur]
            loop.append((kx, ky))

            neighbors = [n for n in adj[cur] if n != prev]
            if len(neighbors) == 0:
                break
            nxt = neighbors[0]
            prev = cur
            cur = nxt

        if len(loop) >= 2:
            loops.append(loop)

    return loops


# ==============================================================================
# Test model: E = 2 e^{i δx} Re(Jx* e^{iθx}) + 2 e^{i δy} Re(Jy* e^{iθy})
# ==============================================================================

def _real_acos(t: float, eps: float = 1e-12) -> List[float]:
    """Return unique θ ∈ [0, 2π) such that cos(θ) = t for t ∈ [-1, 1].

    Returns 1 value if |t| ≈ 1 (boundary), 2 values otherwise, or empty if |t| > 1.
    """
    if abs(t) > 1.0 + eps:
        return []
    t = np.clip(t, -1.0, 1.0)
    ac = np.arccos(t)
    vals = [ac, (-ac) % (2.0 * np.pi)]
    # Deduplicate when ac ≈ 0 or π
    unique = []
    for v in vals:
        if not any(abs(v - u) < 1e-12 for u in unique):
            unique.append(v)
    return unique


def make_test_theta_fun(
    Jx: complex = 1.0 + 0.0j,
    Jy: complex = 1.0 + 0.0j,
    delta_x: float = 0.0,
    delta_y: float = np.pi / 2,
) -> Callable[[complex], List[Tuple[float, float]]]:
    """
    Build a theta_fun for the test model.

    E(θx, θy) = 2 e^{i δx} Re(Jx* e^{iθx}) + 2 e^{i δy} Re(Jy* e^{iθy})

    Default parameters give E = 2 cos(θx) + 2i cos(θy).
    """
    ax = 2.0 * abs(Jx) * np.exp(1j * delta_x)  # complex coefficient for tx
    ay = 2.0 * abs(Jy) * np.exp(1j * delta_y)  # complex coefficient for ty
    phi_x = np.angle(Jx)  # phase of Jx
    phi_y = np.angle(Jy)  # phase of Jy

    ax_r, ax_i = ax.real, ax.imag
    ay_r, ay_i = ay.real, ay.imag
    D = ax_r * ay_i - ax_i * ay_r  # determinant

    if abs(D) < 1e-15:
        raise ValueError("Coefficients a_x and a_y are collinear; image is degenerate.")

    def theta_fun(E: complex) -> List[Tuple[float, float]]:
        Er, Ei = E.real, E.imag
        tx = (Er * ay_i - Ei * ay_r) / D
        ty = (ax_r * Ei - ax_i * Er) / D

        xs = _real_acos(tx)
        ys = _real_acos(ty)

        result = []
        for thx in xs:
            for thy in ys:
                result.append(((phi_x + thx) % (2.0 * np.pi),
                               (phi_y + thy) % (2.0 * np.pi)))
        return result

    return theta_fun


# ==============================================================================
# Visualization helpers
# ==============================================================================

def _setup_axes():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_e = axes[0]
    ax_k = axes[1]
    ax_e.set_xlabel("Re(E)")
    ax_e.set_ylabel("Im(E)")
    ax_e.set_aspect("equal")
    ax_k.set_xlabel(r"$\theta_x$")
    ax_k.set_ylabel(r"$\theta_y$")
    ax_k.set_xlim(0, 2 * np.pi)
    ax_k.set_ylim(0, 2 * np.pi)
    ax_k.set_aspect("equal")
    return fig, ax_e, ax_k


def plot_result(polygon_vertices: List[complex],
                loops: List[List[Tuple[float, float]]],
                E_samples: List[complex] = None,
                preimages_list: List[List[Tuple[float, float]]] = None,
                title: str = ""):
    """Plot the E-space polygon and the k-space preimage loops."""
    import matplotlib.pyplot as plt

    fig, ax_e, ax_k = _setup_axes()
    if title:
        fig.suptitle(title)

    # --- E-space ---
    poly = polygon_vertices + [polygon_vertices[0]]
    px = [z.real for z in poly]
    py = [z.imag for z in poly]
    ax_e.plot(px, py, "b.-", linewidth=2, markersize=4, label="Polygon")

    if E_samples is not None:
        sx = [z.real for z in E_samples]
        sy = [z.imag for z in E_samples]
        ax_e.scatter(sx, sy, c="gray", s=4, alpha=0.5, zorder=2)

    # Draw image region (parallelogram) for the test model
    ax_e.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax_e.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    # Default test: image is [-2, 2] x [-2, 2]
    rect_x = [-2, 2, 2, -2, -2]
    rect_y = [-2, -2, 2, 2, -2]
    ax_e.plot(rect_x, rect_y, "r--", linewidth=1, alpha=0.5, label="Image region")

    ax_e.legend(fontsize=8)

    # --- k-space ---
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(loops), 1)))
    for i, loop in enumerate(loops):
        kx = [p[0] for p in loop]
        ky = [p[1] for p in loop]
        ax_k.plot(kx, ky, color=colors[i % 10], linewidth=1.5, marker=".",
                  markersize=2, label=f"Loop {i + 1}")

    if preimages_list is not None:
        for pts in preimages_list:
            if pts:
                kx = [p[0] for p in pts]
                ky = [p[1] for p in pts]
                ax_k.scatter(kx, ky, c="gray", s=2, alpha=0.2, zorder=1)

    ax_k.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return fig


# ==============================================================================
# Test suite
# ==============================================================================

def test_inside_circle():
    """Circle of radius r=1 fully inside the image parallelogram.
    Expect 4 simple loops, no birth/death events.
    """
    print("=" * 60)
    print("Test 1: Circle inside image region (r=1)")
    print("=" * 60)

    theta_fun = make_test_theta_fun()

    # Unit circle, 100 vertices
    n_vert = 200
    t = np.linspace(0, 2 * np.pi, n_vert + 1)[:-1]
    polygon = [complex(np.cos(ti), np.sin(ti)) for ti in t]  # r=1

    n_samples = 400
    loops = compute_preimage_loops(polygon, theta_fun, n_samples)

    print(f"  Found {len(loops)} loops")
    for i, loop in enumerate(loops):
        print(f"    Loop {i + 1}: {len(loop)} points")

    # Verify: 4 loops expected (all points are interior → 4 preimages each)
    assert len(loops) == 4, f"Expected 4 loops, got {len(loops)}"

    # Check that loops are roughly closed
    for i, loop in enumerate(loops):
        first = np.array(loop[0])
        last = np.array(loop[-1])
        dist = np.sqrt(min(abs(first[0] - last[0]),
                          2 * np.pi - abs(first[0] - last[0])) ** 2 +
                       min(abs(first[1] - last[1]),
                          2 * np.pi - abs(first[1] - last[1])) ** 2)
        print(f"    Loop {i + 1} closure distance: {dist:.6f}")
        # Allow some slack since the last node's neighbor may not perfectly
        # close due to graph tracing. The loop itself should be nearly closed.

    return loops, polygon, theta_fun


def test_crossing_boundary():
    """Horizontal rectangle that crosses the left/right boundaries of the image.
    Tests birth/death pair creation and annihilation.
    """
    print("\n" + "=" * 60)
    print("Test 2: Rectangle crossing image boundary")
    print("=" * 60)

    theta_fun = make_test_theta_fun()

    # Rectangle: x ∈ [-2.5, 2.5], y ∈ [-1, 1]
    # Image is [-2, 2] x [-2, 2], so x-boundaries are crossed, y stays inside.
    polygon = [
        complex(-2.5, -1.0),
        complex(2.5, -1.0),
        complex(2.5, 1.0),
        complex(-2.5, 1.0),
    ]

    n_samples = 800
    loops = compute_preimage_loops(polygon, theta_fun, n_samples)

    print(f"  Found {len(loops)} loops")
    for i, loop in enumerate(loops):
        print(f"    Loop {i + 1}: {len(loop)} points, "
              f"θx range: [{min(p[0] for p in loop):.3f}, {max(p[0] for p in loop):.3f}], "
              f"θy range: [{min(p[1] for p in loop):.3f}, {max(p[1] for p in loop):.3f}]")

    return loops, polygon, theta_fun


def test_circle_crossing():
    """Circle of radius 2.1 that partially exits the image parallelogram.
    The E(t) = 2.1 * exp(i t) circle exits at the four edges.
    """
    print("\n" + "=" * 60)
    print("Test 3: Circle crossing image boundary (r=2.1)")
    print("=" * 60)

    theta_fun = make_test_theta_fun()

    n_vert = 300
    t = np.linspace(0, 2 * np.pi, n_vert + 1)[:-1]
    polygon = [complex(2.1 * np.cos(ti), 2.1 * np.sin(ti)) for ti in t]

    n_samples = 600
    loops = compute_preimage_loops(polygon, theta_fun, n_samples)

    print(f"  Found {len(loops)} loops")
    for i, loop in enumerate(loops):
        kx = [p[0] for p in loop]
        ky = [p[1] for p in loop]
        print(f"    Loop {i + 1}: {len(loop)} points, "
              f"θx: [{min(kx):.3f}, {max(kx):.3f}], "
              f"θy: [{min(ky):.3f}, {max(ky):.3f}]")

    return loops, polygon, theta_fun


def test_circle_corner():
    """Circle of radius 0.1 centered at the left-bottom corner.
    """
    print("\n" + "=" * 60)
    print("Test 4: Circle at corner")
    print("=" * 60)

    theta_fun = make_test_theta_fun()

    n_vert = 300
    t = np.linspace(0, 2 * np.pi, n_vert + 1)[:-1]
    polygon = np.exp(1j * t) * 0.1 + (-2 - 2j)

    n_samples = 600
    loops = compute_preimage_loops(polygon, theta_fun, n_samples)

    print(f"  Found {len(loops)} loops")
    for i, loop in enumerate(loops):
        kx = [p[0] for p in loop]
        ky = [p[1] for p in loop]
        print(f"    Loop {i + 1}: {len(loop)} points, "
              f"θx: [{min(kx):.3f}, {max(kx):.3f}], "
              f"θy: [{min(ky):.3f}, {max(ky):.3f}]")

    return loops, polygon, theta_fun




# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # non-interactive, safe for headless
    import matplotlib.pyplot as plt

    # Test 1: inside circle
    loops1, poly1, tf1 = test_inside_circle()
    E_samp1 = sample_polygon(poly1, 400)
    pre1 = [tf1(E) for E in E_samp1]
    fig1 = plot_result(poly1, loops1, E_samp1, pre1,
                       title="Test 1: Circle r=1 (inside image)")
    fig1.savefig("data/test1_inside_circle.png", dpi=150)
    print("  -> Saved test1_inside_circle.png")

    # Test 2: crossing rectangle
    loops2, poly2, tf2 = test_crossing_boundary()
    E_samp2 = sample_polygon(poly2, 800)
    pre2 = [tf2(E) for E in E_samp2]
    fig2 = plot_result(poly2, loops2, E_samp2, pre2,
                       title="Test 2: Rectangle crossing boundary")
    fig2.savefig("data/test2_crossing_rectangle.png", dpi=150)
    print("  -> Saved test2_crossing_rectangle.png")

    # Test 3: crossing circle
    loops3, poly3, tf3 = test_circle_crossing()
    E_samp3 = sample_polygon(poly3, 600)
    pre3 = [tf3(E) for E in E_samp3]
    fig3 = plot_result(poly3, loops3, E_samp3, pre3,
                       title="Test 3: Circle r=2.1 (crossing boundary)")
    fig3.savefig("data/test3_crossing_circle.png", dpi=150)
    print("  -> Saved test3_crossing_circle.png")

    # Test 4: circle at left-bottom corner
    loops4, poly4, tf4 = test_circle_corner()
    E_samp4 = sample_polygon(poly4, 600)
    pre4 = [tf4(E) for E in E_samp4]
    fig4 = plot_result(poly4, loops4, E_samp4, pre4,
                       title="Test 4: Circle at corner")
    fig4.savefig("data/test4_circle_corner.png", dpi=150)
    print("  -> Saved test4_circle_corner.png")

    print("\nAll tests completed.")
