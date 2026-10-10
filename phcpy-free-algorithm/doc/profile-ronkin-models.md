# Profile Report: `bisect_amoeba_ronkin_min` — Real Model Benchmarks

Date: 2026-05-20
Profiling tool: cProfile (stdlib)
Models: Haldane non-Hermitian phase, C4-symmetry, Special (1-site asymmetric)
Reference: [`profile-ronkin.md`](profile-ronkin.md) (Square & NNN toy models)

---

## 1. Model Overview

| Model | Sites | Poly terms | M₁ | N₁ | M₂ | N₂ | β₂ roots | Source file |
|-------|-------|-----------|----|----|----|----|----------|-------------|
| Square | 1 | 5 | 1 | 1 | 1 | 1 | 2 | demo-amoeba.py |
| NNN (g=0.3) | 1 | 7 | 1 | 1 | 1 | 1 | 2 | demo-amoeba.py |
| Haldane | 2 | 26 | 2 | 6 | 2 | 6 | **8** | Haldane-model-phase.py |
| C4 | 5 | 42 | 1 | 4 | 1 | 3 | 4 | C4-symmetry.py |
| Special | 1 | 7 | — | — | — | — | 2 | special-model.py |

**Key**: β₂ roots = M₂ + N₂, the number of complex roots solved at each θ₁ point. Each root is tracked across the full θ₁ mesh.

---

## 2. Runtime Summary

### 2.1 Haldane Model (N_points=201, M₂+N₂=8, max_iter=60)

| E_ref | Wall time | ccw calls | `_find_mu2` calls | mu1 | mu2 | Result |
|-------|-----------|-----------|-------------------|-----|-----|--------|
| 0 | 188.8s | 723 | 29 | −0.518 | +0.259 | non-continuum, 2 zeros |
| 1 | 4.7s | 19 | 3 | 0.000 | 0.000 | non-continuum, 6 zeros |
| 2 | 308.9s | 1,217 | 49 | −0.023 | +0.012 | non-continuum, 2 zeros |

**Spread**: 4.7s → 309s (66× variation) depending on E. E=1 is the "easy" case; E=0 and E=2 trigger many more bisection iterations.

### 2.2 C4 Model (N_points=51, M₂+N₂=4, max_iter=60)

| E_ref | Wall time | ccw calls | `_find_mu2` calls | mu1 | mu2 |
|-------|-----------|-----------|-------------------|-----|-----|
| 0 | 64.2s | 806 | 31 | −0.087 | +0.032 |
| 1 | 209.8s | 2,655 | 37 | −0.221 | +0.056 |
| 2 | 156.8s | 2,168 | 30 | −0.016 | +0.078 |

**Note**: N=51 only. At N=201, estimated 3.9× slower (250s / 820s / 610s). The C4 model at E=1 would be ~14 minutes at N=201 — totally impractical for energy scans with hundreds of E points.

### 2.3 Special Model

| E_ref | N_points | Wall time | ccw calls | `_find_mu2` calls | mu1 | mu2 |
|-------|----------|-----------|-----------|-------------------|-----|-----|
| 0 | 51 | 19.6s | 296 | 5 | 0.000 | 0.000 (continuum) |
| 1 | 51 | 6.5s | 97 | 3 | 0.000 | 0.000 |
| 2 | 51 | 11.0s | 167 | 3 | 0.000 | −0.047 |
| 1 | 201 | 39.0s | 151 | 3 | 0.000 | 0.000 |

The Special model is the cheapest of the three real models: ~39s at N=201 for E=1, but still shows E-dependent variation (6.5s → 19.6s at N=51 for E=0 vs E=1, a ~3× spread).

---

## 3. Call Count Deep Dive

### 3.1 Haldane — Why the 66× E-dependence?

| E | ccw calls | Outer iters | Inner calls | Calls/inner | Bisect iters/inner |
|---|-----------|-------------|-------------|-------------|-------------------|
| 0 | 723 | ~23 | 29 | 723/29 ≈ 24.9 | ~(24.9−2×(2+1))/2 ≈ 9.5 |
| 1 | 19 | ~3 | 3 | 19/3 ≈ 6.3 | ~(6.3−4)/2 ≈ 1.2 |
| 2 | 1,217 | ~43 | 49 | 1217/49 ≈ 24.8 | ~(24.8−4)/2 ≈ 10.4 |

**Explanation**: At E=1, the a2 winding crosses zero cleanly near μ₂=0, so the inner μ₂ bisection converges in ~1 iteration. At E=0 and E=2, the winding function has very flat regions (near-continuum behavior), requiring ~10 bisection iterations per inner call. Combined with more outer μ₁ bisection iterations (23–43 vs 3), this produces the 66× spread.

### 3.2 C4 — E=1 is the worst case

The C4 model at E=1 triggers **2,655** ccw calls even at N=51. The `find_mu2` count (37) multiplied by the per-inner ccw calls (2655/37 ≈ 71.8) is extremely high, suggesting ~36 bisection iterations per inner call.

Additionally, `_find_exact_crossing` is called 14,616 times at E=1 (vs 3,252 at E=0), indicating many more root crossings are detected and need to be refined with fsolve.

### 3.3 Redundancy Impact (from [profile-ronkin.md](profile-ronkin.md) findings)

The two main redundancy sources apply to all models:

| Redundancy | Mechanism | Typical waste |
|-----------|-----------|---------------|
| Double-call per inner bisection iter | Lines 464 + 498 call `_compute_crossings_and_winding` at the same (μ₁, μ₂) | ~50% of direct ccw calls |
| Redundant a1/a2 ccw | Outer loop calls `_compute_crossings_and_winding` again even though zeros are already known | ~`n_a1_calls` × per-call cost |

For Haldane E=2 (1,217 ccw calls): estimated ~600 of those are redundant (\(\approx\) 150s wasted).

For C4 E=1 (2,655 ccw calls): estimated ~1,300 are redundant (\(\approx\) 100s wasted at N=51).

---

## 4. Per-Root Cost Scaling

The Hungarian matching cost matrix is O(n²) in the number of β₂ roots, and `to_sphere_r3` (the dominant cost inside matching) produces (n, 3) arrays that feed into an (n, n, 3) broadcast subtraction.

| Model | n_roots | to_sphere_r3 calls | to_sphere_r3 time | Time/call |
|-------|---------|-------------------|-------------------|-----------|
| Square (N=201) | 2 | 10,400 | 1.66s | 0.16 ms |
| Haldane (N=201, E=2) | 8 | 486,800 | 71.5s | 0.15 ms |
| C4 (N=51, E=1) | 4 | 265,500 | 41.5s | 0.16 ms |

The per-call cost of `to_sphere_r3` is **constant** (~0.15 ms regardless of n_roots), since it operates on a single roots vector. The quadratic scaling comes from the number of entries in the broadcast subtraction `p_from[:, None, :] - p_to[None, :, :]`, which grows as n². For Haldane (n=8, 64 entries/matrix) vs Square (n=2, 4 entries/matrix), this is a 16× factor per match.

Similarly, `numpy.roots` / `eigvals` solves a companion matrix of size equal to the polynomial degree. For Haldane (8 roots → degree-8 polynomial on β₂), this is significantly more expensive than the degree-2 polynomial for Square.

---

## 5. Call Count Comparison (N=201, except C4 at N=51)

| Metric | Square E=1 | NNN E=1 | Haldane E=0 | Haldane E=1 | Haldane E=2 | C4 E=1 (N=51) |
|--------|-----------|---------|-------------|-------------|-------------|---------------|
| Total time | 6.5s | 24.0s | 188.8s | 4.7s | 308.9s | 209.8s |
| ccw calls | 26 | 102 | 723 | 19 | 1,217 | 2,655 |
| Hungarian calls | 5,200 | 20,400 | 144,600 | 3,800 | 243,400 | 132,750 |
| Point roots | 5,262 | 20,696 | 147,651 | 3,889 | 248,416 | 150,272 |
| Exact crossing | 20 | 200 | 3,420 | 112 | 4,520 | 14,616 |
| Avg winding zeros | 30 | 106 | 746 | 22 | 1,260 | 2,692 |

**Key takeaways**:

1. Hungarian matching dominates all models (40–50% of total time).
2. `_find_exact_crossing` (fsolve) is negligible for Square/NNN but becomes **significant for C4** (21s out of 210s = 10%) and **moderate for Haldane** (7s out of 309s = 2%).
3. `_get_average_winding_from_zeros` grows with ccw calls and can become non-negligible: 12s for C4 E=1.
4. The total number of `calculate_point_roots` calls is roughly:
   - N_points × ccw_calls (from Hungarian matching)
   - + 2 × number of zeros × (direction calls) (from segment midpoint evaluations)
   - For Haldane E=2: 201 × 1217 = 244,617 from Hungarian + ~4,000 from midpoint evals ≈ 248,416 ✓

---

## 6. N=51 vs N=201 Scaling

Comparing Special model E=1:

| | N=51 | N=201 | Ratio |
|---|---|---|---|
| Wall time | 6.5s | 39.0s | 6.0× |
| ccw calls | 97 | 151 | 1.56× |
| Point roots | 5,139 | 30,655 | 5.96× |
| Hungarian matches | 4,850 | 30,200 | 6.23× |

The wall time scales **faster than linear** in N_points because increasing N not only increases per-call cost (~N) but also increases the number of ccw calls (1.56× more), likely because the finer mesh resolves more crossing events that trigger additional bisection refinements.

---

## 7. Practical Scan Feasibility

### Projected scan cost for 100×100 energy grid at N=201

| Model | Per-E time (best) | Per-E time (worst) | 10K-E scan (best) | 10K-E scan (worst) |
|-------|-------------------|-------------------|--------------------|--------------------|
| Square | 6.5s | ~7s | **18 hours** | ~19 hours |
| NNN | 24s | ~25s | **67 hours** | ~70 hours |
| Haldane | 4.7s | 309s | **13 hours** | **858 hours (36 days)** |
| C4 (est. N=201) | ~250s | ~820s | **29 days** | **95 days** |
| Special | ~10s | ~40s | **28 hours** | ~111 hours |

**Reality check**: The Haldane and C4 models are **not feasible** for dense energy scans without optimization. Even with parallelization across 64 cores, a full Haldane scan would take 0.5–13 hours at best, and the worst-case E values would dominate.

---

## 8. Model-Specific Observations

### Haldane (2-site honeycomb with non-Hermitian NNN hopping)

- **E=1 is anomalously fast** (4.7s) while E=0 and E=2 are 40–65× slower. This suggests E=1 sits in a "clean" region where the a2 winding has a sharp zero-crossing, while E=0 and E=2 are near band edges where the winding function flattens.
- The Ronkin minimum is far from the origin: (μ₁, μ₂) = (−0.518, +0.259) at E=0 and (−0.023, +0.012) at E=2, indicating strong non-reciprocity.
- 8 β₂ roots means the companion matrix eigenvalue solve is substantially more expensive than for 2-root models.

### C4 (5-site unit cell, C₄ rotation symmetry)

- **E=1 is the worst case** (unlike Haldane where E=1 is fastest). The a2 winding at E=1 appears to have complex structure requiring many bisection iterations.
- High `_find_exact_crossing` count (14,616 at N=51) suggests numerous root crossings are detected. Some of these may be spurious near-crossings that survive the continuum filter.
- With 4 β₂ roots, the per-point cost is intermediate between Square (2 roots) and Haldane (8 roots).

### Special (1-site, asymmetric x/y hoppings)

- Generally well-behaved. The Ronkin minimum is at the origin (μ₁=μ₂=0) for all E tested.
- Continuum detected at E=0 only.
- The most practical model for scans among the three.

---

## 9. Summary of Top Bottlenecks (all models)

| Bottleneck | % of total | Comment |
|-----------|-----------|---------|
| Hungarian matching (`_hungarian_match_indices`) | 40–50% | `to_sphere_r3` dominates; O(n²) in root count |
| Point root solving (`calculate_point_roots`) | 40–50% | `numpy.roots` → companion matrix `eigvals` |
| Double-call redundancy (lines 464+498) | 19–42% | Wasted; first call's result is discarded |
| a1/a2 redundant ccw | 4–15% | Same zeros computed twice |
| `_find_exact_crossing` (fsolve) | 0.3–10% | Negligible for simple models; significant for C4 |
| `_get_average_winding_from_zeros` | 0.5–6% | Grows with number of zero crossings |

---

*Generated by cProfile on 2026-05-20. Profiling scripts: `profile_models.py`, `profile_quick.py`.*
