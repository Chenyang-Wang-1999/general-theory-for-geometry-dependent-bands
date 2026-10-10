# Profile Report: `bisect_amoeba_ronkin_min`

Date: 2026-05-20
Profiling tool: cProfile (stdlib)

---

## 1. Executive Summary

| Metric | Square (N=201) | Square (N=51) | NNN g=0.3 (N=201) |
|--------|---------------|--------------|-------------------|
| Total wall time | 6.52s | 1.94s | 23.98s |
| `_compute_crossings_and_winding` calls | 26 | 26 | **102** |
| `get_hungarian_sorted_roots` calls | 26 | 26 | **102** |
| `_find_mu2_for_a2_zero` calls | 5 | 5 | 5 |
| `calculate_point_roots` calls | 5,262 | 1,362 | 20,696 |
| `_hungarian_match_indices` calls | 5,200 | 1,300 | 20,400 |

The NNN model triggers **3.9× more** `_compute_crossings_and_winding` calls than the square model (102 vs 26), despite the outer bisection structure being identical (5 inner bisections). The difference comes entirely from more bisection iterations within `_find_mu2_for_a2_zero` (86 direct calls vs 10).

---

## 2. Call Hierarchy

```
bisect_amoeba_ronkin_min  [1 call, ~100% of total time]
│
├── _find_mu2_for_a2_zero  [5 calls, ~85–96% of total]
│   │
│   ├── get_a2_average_winding  [12 calls total]
│   │   └── _compute_crossings_and_winding
│   │       ├── get_hungarian_sorted_roots  [~99% of per-call time]
│   │       │   ├── calculate_point_roots  [N_points calls]
│   │       │   │   ├── poly_to_np_coefficients
│   │       │   │   ├── numpy.linalg.eigvals
│   │       │   │   └── numpy.polynomial.roots
│   │       │   └── _hungarian_match_indices  [N_points-1 calls]
│   │       │       ├── _chordal_cost_matrix
│   │       │       │   └── to_sphere_r3  [2× per match]
│   │       │       └── scipy.optimize.linear_sum_assignment
│   │       ├── _find_cyclic_true_intervals
│   │       ├── crossing detection loop (per root track)
│   │       ├── _find_exact_crossing  [fsolve + analytic Jacobian]
│   │       └── _get_average_winding_from_zeros  [direction=2]
│   │
│   └── _compute_crossings_and_winding  [direct calls in bisection loop]
│
├── get_a1_average_winding  [4 calls]
│   ├── _compute_crossings_and_winding  [same full pipeline again]
│   └── _get_average_winding_from_zeros  [direction=1]
│
└── (degenerate a1 path: extra _find_mu2_for_a2_zero calls at μ₁ ± ε)
```

---

## 3. Per-Function Time Breakdown

### Square lattice (N_points=201, total 6.52s)

| Function | ncalls | tottime | cumtime | % of total |
|----------|--------|---------|---------|------------|
| `bisect_amoeba_ronkin_min` | 1 | — | 6.52s | 100% |
| `_compute_crossings_and_winding` | 26 | 0.03s | 6.51s | 99.9% |
| `get_hungarian_sorted_roots` | 26 | 0.19s | 6.42s | 98.5% |
| `_hungarian_match_indices` | 5,200 | 0.26s | 3.15s | 48.3% |
| `calculate_point_roots` | 5,262 | 0.27s | 3.03s | 46.4% |
| `_chordal_cost_matrix` | 5,200 | 0.23s | 2.04s | 31.4% |
| `get_a2_average_winding` | 12 | — | 2.95s | 45.3% |
| `get_a1_average_winding` | 4 | — | 1.01s | 15.4% |
| `linear_sum_assignment` | 5,200 | 0.04s | 0.04s | 0.7% |
| `_find_exact_crossing` | 20 | — | 0.02s | 0.3% |
| `_get_average_winding_from_zeros` | 30 | — | 0.03s | 0.5% |

**Key observation**: `_compute_crossings_and_winding` dominates at 99.9%. Within it, Hungarian matching + point root solving split roughly 50/50. The `to_sphere_r3` sphere projection inside `_chordal_cost_matrix` is the single most expensive operation (pure Python array manipulation).

### NNN lattice (N_points=201, total 23.98s)

| Function | ncalls | tottime | cumtime | % of total |
|----------|--------|---------|---------|------------|
| `bisect_amoeba_ronkin_min` | 1 | — | 23.98s | 100% |
| `_compute_crossings_and_winding` | 102 | 0.09s | 23.98s | 100% |
| `get_hungarian_sorted_roots` | 102 | 0.66s | 23.38s | 97.5% |
| `_hungarian_match_indices` | 20,400 | 1.04s | 11.66s | 48.6% |
| `calculate_point_roots` | 20,696 | 1.02s | 10.88s | 45.4% |
| `_chordal_cost_matrix` | 20,400 | 0.85s | 7.47s | 31.1% |
| `_find_mu2_for_a2_zero` | 5 | — | 23.10s | 96.3% |
| `_find_exact_crossing` | 200 | — | 0.28s | 1.2% |

The time distribution is nearly identical between square and NNN — the NNN model is slower purely because it executes more iterations, not because individual calls are more expensive.

---

## 4. Redundancy Analysis

### 4.1 Double-call in `_find_mu2_for_a2_zero` bisection (lines 464 + 498)

Each bisection iteration calls `_compute_crossings_and_winding` **twice at the same (μ₁, μ₂)**:

- **Line 464**: checks `has_continuum` only — the returned `winding` and `zeros` are discarded.
- **Line 498**: recomputes everything to get `w_mid` and `zeros`.

```python
# Line 464 — computes winding + zeros + continuum, uses ONLY continuum
_, _, has_continuum = _compute_crossings_and_winding(...)  

# ... (branches on has_continuum, may continue) ...

# Line 498 — recomputes everything again at the same (mu1, mu2_mid)
w_mid, zeros, _ = _compute_crossings_and_winding(...)
```

**Impact estimate**:

| Model | Direct ccw calls | Redundant (50%) | Wasted time |
|-------|-----------------|-----------------|-------------|
| Square (N=201) | 10 | 5 | ~1.25s (19%) |
| NNN (N=201) | 86 | 43 | ~10.1s (42%) |

For the NNN model, nearly **half** the direct calls are redundant — the continuum check at line 464 already computed all the data line 498 needs.

### 4.2 Redundant `_compute_crossings_and_winding` between a1 and a2

`get_a1_average_winding` (line 408) calls `_compute_crossings_and_winding` to get the zero points, but the same function was already called during the inner μ₂ bisection at the same (μ₁, μ₂). The zero points are identical.

| Model | a1 ccw calls | Wasted time |
|-------|-------------|-------------|
| Square (N=201) | 4 | ~1.0s (15%) |
| NNN (N=201) | 4 | ~0.9s (4%) |

### 4.3 Redundant Hungarian matching for non-continuum μ₂

When `has_continuum` is false (the common case), the Hungarian matching at line 464 serves no purpose — it's only needed for continuum detection. But the call still runs the full Hungarian pipeline.

### 4.4 Redundancy summary

| Source | Square waste | NNN waste |
|--------|-------------|----------|
| Double-call per bisection iter | ~1.25s (19%) | ~10.1s (42%) |
| Redundant a1/a2 ccw | ~1.0s (15%) | ~0.9s (4%) |
| Hungarian for continuum-only checks | embedded in above | embedded in above |
| **Estimated total waste** | **~2.3s (35%)** | **~11.0s (46%)** |

---

## 5. Model Comparison: Square vs NNN

### Why 102 vs 26 calls?

Both models run 5 inner μ₂ bisections (`_find_mu2_for_a2_zero`). The difference is in **bisection iterations per inner call**:

| | Square (N=201) | NNN (N=201) |
|---|---|---|
| Direct ccw calls in inner bisection | 10 | 86 |
| Per `_find_mu2_for_a2_zero` | 2.0 | 17.2 |
| Estimated bisection iterations | ~1 | ~8.6 |

The NNN model converges slower because the a2 winding near μ₂=0 has a flatter gradient (continuum-like behavior where multiple roots stay near |β₂| = e^(μ₂)). This requires more bisection steps to pin down the exact zero-crossing.

Additionally, the NNN model may hit more continuum branches, triggering the `get_a2_average_winding` calls at μ₂ ± continuum_perturb (though the data shows only 2 extra a2 calls for both models, so this is minor).

---

## 6. Scaling Analysis

### 6.1 Effect of N_points

Comparing Square N=51 vs N=201:

| | N=51 | N=201 | Ratio |
|---|---|---|---|
| Total time | 1.94s | 6.52s | 3.4× |
| Hungarian matching time | 0.91s | 3.15s | 3.5× |
| Point root time | 0.88s | 3.03s | 3.4× |

The scaling is roughly **linear in N_points** (201/51 = 3.94×, observed ~3.4×), with some fixed overhead reducing the ratio slightly.

### 6.2 Per-call cost of `_compute_crossings_and_winding`

| Model | N_points | Time per call | Hungarian | Root solving |
|-------|----------|--------------|-----------|-------------|
| Square | 51 | 0.075s | 0.035s | 0.034s |
| Square | 201 | 0.250s | 0.121s | 0.117s |
| NNN | 201 | 0.235s | 0.114s | 0.107s |

Per-call costs are nearly identical for square and NNN at the same N_points, confirming the model difference is purely about iteration count.

### 6.3 Projected costs at higher N_points

Assuming linear scaling and current call counts:

| N_points | Square (26 calls) | NNN (102 calls) |
|----------|-------------------|-----------------|
| 201 | 6.5s | 24.0s |
| 401 | ~13s | ~48s |
| 801 | ~26s | ~96s |
| 1601 | ~52s | ~192s |

For the NNN model at N=801, the runtime would exceed 1.5 minutes per `bisect_amoeba_ronkin_min` call. In a typical energy scan (hundreds of E values), this becomes prohibitive.

---

## 7. Per-Function Deep Dive

### 7.1 `get_hungarian_sorted_roots` (98.5% of total time)

```
For each theta1 point (N_points = 201):
  calculate_point_roots(char_poly, ...)        [201 calls, ~0.015s each]
    → poly_to_np_coefficients                  [build companion matrix]
    → numpy.linalg.eigvals                     [eigenvalue solve]
    → numpy.polynomial.roots                   [companion matrix eigenvalues]

  _hungarian_match_indices(tracked[i], roots_next)  [200 calls, ~0.016s each]
    → _chordal_cost_matrix                     [build 2×2 cost matrix]
      → to_sphere_r3(roots_from)              [array allocation + arithmetic]
      → to_sphere_r3(roots_to)                [array allocation + arithmetic]
      → Euclidean distance in R³
    → linear_sum_assignment(cost)             [solve 2×2 assignment]
    → list comprehension                      [format match pairs]
```

**Observations**:
- For 2×2 cost matrices, `linear_sum_assignment` is essentially free (0.04s out of 3.15s). The bottleneck is `to_sphere_r3` which allocates (n_roots, 3) arrays 2× per match step.
- `calculate_point_roots` involves solving a polynomial eigenvalue problem per θ₁ point. This calls `numpy.roots` which builds a companion matrix then calls `eigvals`.

### 7.2 `_find_exact_crossing` (negligible)

Only 20–200 calls across the entire run. Each uses `scipy.optimize.fsolve` with analytic Jacobian. This is not a bottleneck.

### 7.3 `_get_average_winding_from_zeros` (negligible)

30–106 calls, each evaluating `calculate_point_roots` at segment midpoints (~2 points per call). This is O(number of zero crossings), which is very small (2–4 crossings typical).

---

## 8. Call Count Breakdown

### Square, N=201, 26 `_compute_crossings_and_winding` calls

| Source | Count | Description |
|--------|-------|-------------|
| `get_a2_average_winding` (endpoints) | 10 | 5 inner bisections × 2 endpoints each |
| `get_a2_average_winding` (continuum perturb) | 2 | 2 extra calls within bisection for continuum checks |
| Direct in `_find_mu2_for_a2_zero` | 10 | 5 inner bisections × ~1 iter × 2 calls/iter |
| `get_a1_average_winding` | 4 | 4 outer bisection iterations |
| **Total** | **26** | |

### NNN, N=201, 102 `_compute_crossings_and_winding` calls

| Source | Count | Description |
|--------|-------|-------------|
| `get_a2_average_winding` (endpoints) | 10 | 5 inner bisections × 2 endpoints each |
| `get_a2_average_winding` (continuum perturb) | 2 | 2 extra calls within bisection |
| Direct in `_find_mu2_for_a2_zero` | 86 | 5 inner bisections × ~8.6 iter × 2 calls/iter |
| `get_a1_average_winding` | 4 | 4 outer bisection iterations |
| **Total** | **102** | |

### Non-redundant equivalent

If we eliminated the double-call and a1/a2 redundancies:

| Model | Current calls | Minimal calls | Reduction |
|-------|-------------|--------------|-----------|
| Square (N=201) | 26 | ~13 | 50% |
| NNN (N=201) | 102 | ~51 | 50% |

Note: "Minimal" still includes one Hungarian matching per bisection iteration + one per a1 evaluation. Further reduction could merge a1/a2 into a single call.

---

## 9. Raw cProfile Output (Square, N=201)

```
799,221 function calls (751,625 primitive calls) in 6.496 seconds

ncalls  tottime  cumtime  filename:lineno(function)
    26    0.021    6.492  amoeba.py:263(_compute_crossings_and_winding)
    26    0.178    6.394  amoeba.py:107(get_hungarian_sorted_roots)
     5    0.000    5.384  amoeba.py:416(_find_mu2_for_a2_zero)
  5,200    0.243    3.239  amoeba.py:66(_hungarian_match_indices)
  5,262    0.258    2.916  root_solver.py:73(calculate_point_roots)
    12    0.000    2.876  amoeba.py:361(get_a2_average_winding)
  5,200    0.237    2.162  amoeba.py:44(_chordal_cost_matrix)
  5,262    0.013    1.746  numpy:177(roots)
 10,400    1.655    1.699  amoeba.py:45(to_sphere_r3)
  5,262    0.353    1.639  numpy/lib/polynomial.py:171(roots)
     4    0.000    1.112  amoeba.py:390(get_a1_average_winding)
  5,262    0.013    0.585  numpy:177(eigvals)
  5,262    0.277    0.489  root_solver.py:55(poly_to_np_coefficients)
  5,200    0.053    0.053  {scipy.optimize._lsap.linear_sum_assignment}
```

---

## 10. Recommendations

Based on the profiling data, the following optimizations would yield the largest impact (ordered by expected gain):

1. **Eliminate the double-call in `_find_mu2_for_a2_zero`** (lines 464 + 498): Save the result from line 464 and reuse it at line 498 when `has_continuum` is False. **Expected saving: 19-42%** depending on model.

2. **Cache `_compute_crossings_and_winding` results**: When `bisect_amoeba_ronkin_min` calls `get_a1_average_winding` at (μ₁, μ₂) that was just computed by the inner μ₂ bisection, reuse the zeros. **Expected saving: 4-15%**.

3. **Lazy continuum check**: The Hungarian matching at line 464 is wasted when the result is not continuum. A cheaper pre-check (e.g., checking only a few roots for near-boundary behavior) could defer the full matching. **Expected saving: up to 50% of direct calls** (the ones at line 464 that don't lead to continuum branches).

4. **Vectorize `to_sphere_r3`**: This pure-Python/numpy function is called 10,400 times and accounts for 1.66s (25% of total). Moving to Cython or using pre-allocated arrays could significantly reduce overhead.

5. **Profile `poly_to_np_coefficients` + `partial_eval`**: These are called per θ₁ point and account for ~0.7s. Any overhead in the poly_tools C++/Python bridge could be worth investigating.

---

*Generated by cProfile on 2026-05-20. Profiling scripts: `profile_ronkin.py`, `profile_detail.py`.*
