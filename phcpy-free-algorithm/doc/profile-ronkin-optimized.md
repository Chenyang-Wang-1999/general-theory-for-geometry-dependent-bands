# Optimization Report: `bisect_amoeba_ronkin_min`

Date: 2026-05-20
Base reports: [`profile-ronkin.md`](profile-ronkin.md), [`profile-ronkin-models.md`](profile-ronkin-models.md)

---

## 1. Overview: Three-Round Optimization

| Round | Change | Key mechanism |
|-------|--------|--------------|
| **v1 → v2** | Cache Hungarian matching per (E, μ₁) | Split `_compute_crossings_and_winding` into `_compute_root_tracks` + `_compute_winding_from_tracks`; cache tracks in `_find_mu2_for_a2_zero` |
| **v2 → v3** | Lazy crossing refinement + analytical dW/dμ₂ | Skip fsolve during bisection; refine only at final μ₂ + Newton correction |
| **v3 → v4** | Zero-plateau early exit | Detect a1=0 plateau in outer bisection's degenerate path; return immediately |

### Final result: Haldane E=2 from **309s → 1.07s (289× speedup)**

---

## 2. Before/After Comparison (N_points=201 except C4 at N=51)

| Model | E | v1 (original) | v4 (final) | Speedup | root_tracks calls (v1→v4) |
|-------|---|---------------|------------|---------|---------------------------|
| Square | 0 | 6.5s | **1.4s** | **4.6×** | 26 → 5 |
| Square | 1 | 6.5s | **0.83s** | **7.8×** | 26 → 3 |
| NNN | 1 | 24.0s | **1.6s** | **15.0×** | 102 → 5 |
| Haldane | 0 | 188.8s | **0.99s** | **191×** | 723 → 3 |
| Haldane | 1 | 4.7s | **1.0s** | **4.7×** | 19 → 3 |
| Haldane | 2 | 308.9s | **1.1s** | **289×** | 1,217 → 3 |
| C4 (N=51) | 0 | 64.2s | **0.83s** | **77×** | 806 → 3 |
| C4 (N=51) | 1 | 209.8s | **11.2s** | **18.7×** | 2,655 → 37 |
| Special | 0 | 19.6s | **1.2s** | **16.3×** | 296 → 3 |
| Special | 1 | 39.0s | **1.2s** | **32.5×** | 151 → 3 |

### Why Haldane E=0, E=2 improved so dramatically with v4

These cases hit the a1 zero-plateau: a1 is always ≤ 0, approaching 0 from below as μ₁ increases, then staying at 0. The outer bisection's degenerate path now detects `a1_mid = 0` and returns immediately instead of iterating until bracket exhaustion. This cut `root_tracks` from 27–49 (v3) to 3 (v4).

### C4 E=1 remains the most expensive case

C4 E=1 at N=51 still takes 11.2s (37 `root_tracks` calls). The a1 winding here has genuine crossings (non-plateau, 6 zeros), so the outer bisection must iterate normally. At N=201, projected ~45s per E point.

---

## 3. Final Call Counts

| Model | E | root_tracks | hungarian | winding_from_tracks | find_exact_crossing | pt_roots |
|-------|---|------------|-----------|---------------------|--------------------|----------|
| Square | 0 | 5 | 5 | 28 | — | 1,038 |
| Square | 1 | 3 | 3 | 14 | — | 626 |
| NNN | 1 | 5 | 5 | 28 | — | 1,296 |
| Haldane | 0 | 3 | 3 | 14 | ~4 | 682 |
| Haldane | 1 | 3 | 3 | 14 | ~56 | 667 |
| Haldane | 2 | 3 | 3 | 14 | ~0 | 650 |
| C4 (N=51) | 0 | 3 | 3 | 73 | ~0 | 562 |
| C4 (N=51) | 1 | 37 | 37 | 1,429 | 436 | 9,932 |
| Special | 0 | 3 | 3 | 114 | ~100 | 824 |
| Special | 1 | 3 | 3 | 79 | ~180 | 873 |

- `ccw`, `a1_avg`, `a2_avg` are all 0 — old code paths completely bypassed.
- `root_tracks` = Hungarian matching count (once per `_find_mu2_for_a2_zero` call).
- `find_exact_crossing` = fsolve calls (only during the final refinement step, not during bisection).

---

## 4. Remaining Bottlenecks (C4 E=1, N=51, 11.2s — worst case)

| Bottleneck | Time | % |
|-----------|------|-----|
| `_get_average_winding_from_zeros` (1,466 calls) | 6.4s | **57%** |
| `calculate_point_roots` → `numpy.roots` (9,932 calls) | 6.3s | **56%** |
| `_compute_root_tracks` → Hungarian matching (37 calls) | 2.5s | **22%** |
| `_compute_winding_from_tracks` (non-refine part, 1,429 calls) | 1.3s | **12%** |
| `_find_exact_crossing` (436 calls) | 0.8s | **7%** |

**Note**: The C4 model has 4 β₂ roots. `_get_average_winding_from_zeros` dominates at 57% because the bisection requires many iterations (1,429 `_compute_winding_from_tracks` calls), each triggering winding-from-zeros evaluations at segment midpoints. Each `_get_average_winding_from_zeros` call itself calls `calculate_point_roots` at segment midpoints.

---

## 5. Projected Scan Feasibility

### Per-E time at N=201, v4 final

| Model | Typical E | Worst E | 10K-E scan |
|-------|----------|---------|-----------|
| Square | 0.8–1.4s | ~2s | **~3 hours** |
| NNN | ~1.6s | ~2s | **~4.5 hours** |
| Haldane | **0.99–1.1s** | ~1.1s | **~3 hours** |
| C4 (N=51) | 0.8s | 11.2s | **~10 hours** |
| C4 (N=201, est.) | ~3s | ~45s | **~34 hours** |
| Special | ~1.2s | ~1.2s | **~3.3 hours** |

### Before → After

| Model | v1 worst-E | v4 worst-E | Improvement |
|-------|-----------|-----------|-------------|
| Haldane | 36 days | **3 hours** | 289× |
| C4 | 95 days | ~34 hours (est.) | 67× |
| Square | 19 hours | 3 hours | 6× |

---

## 6. Code Changes Summary

All changes in `brute_force_amoeba/amoeba.py`:

| Change | Description |
|--------|-------------|
| `_compute_root_tracks` | New: Hungarian matching + track data, μ₂-independent |
| `_compute_winding_from_tracks` | Modified: accept `refine_crossings` param + pre-computed `tracks` dict |
| `_compute_zero_dtheta1_dmu2` | New: analytical dθ₁/dμ₂ via 2×2 linear solve |
| `_refine_and_correct` | New: refine zeros at μ₂_0 + Newton correction with dW/dμ₂ |
| `_find_mu2_for_a2_zero` | Modified: cache tracks, lazy refinement during bisection, Newton correction at end |
| `bisect_amoeba_ronkin_min` | Modified: zero-plateau early exit in degenerate a1 path; reuse zeros for a1 |

All public APIs (`get_a2_average_winding`, `get_a1_average_winding`, `bisect_a2_winding`, `bisect_amoeba_ronkin_min`, `check_amoeba`) unchanged.

---

*Profiling scripts: `profile_verify.py`, `profile_optimized.py` (kept in repo).*
