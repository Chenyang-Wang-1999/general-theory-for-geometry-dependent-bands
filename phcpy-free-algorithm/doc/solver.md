# brute_force_solver — SGBZ Spectrum Calculation

基于 SGBZ（Strip Generalized Brillouin Zone）formulation 的非厄米能谱计算模块。

## 1. 理论背景

### 1.1 SGBZ 与 strip winding number

SGBZ 沿 $\theta_1$ 方向积分，基流形为 $\mu_2 = \rho_{2,0}(\theta_1; E, \mu_1)$（由 $|\beta_M| = |\beta_{M+1}|$ 决定的 PMGBZ 曲线）。strip winding number $W(E, \mu_1)$ 是绕 $\theta_2$ 方向的卷绕数沿 $\theta_1$ 的积分。

### 1.2 PMGBZ 点分类

在 $\theta_1$ 轴上，$\beta_2$ 根按模长排序。当 $|\beta_M| \approx |\beta_{M+1}|$ 时出现简并，称为 PMGBZ 点：

- **Continuum**：$|\beta_M| = |\beta_{M+1}|$ 在连续 $\theta_1$ 区间上成立
- **Accidental**：只在孤立 $\theta_1$ 位置 $|\beta_M| = |\beta_{M+1}|$

每个 accidental PMGBZ 点输出三个元组：
```python
beta2_sols = [pos_tuple, neg_tuple, zero_tuple]
```
按 $d\mu_2/d\theta_1$ 从大到小排序，前 $M - i_{\min}$ 个记为正荷。

### 1.3 谱包含关系

$$\sigma_{\text{Amoeba}} \supset \cup_j \sigma_{\text{SGBZ},j}$$

一般情况下 SGBZ 谱是 amoeba spectrum 的子集。对于 uniform bands（几何依赖能谱退化），两者相等。

## 2. 架构总览

```
┌──────────────────────────────────────────────────────────┐
│                     Public API                            │
│  SGBZSolver  SGBZChecker  check_SGBZ  convert_results... │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│              strip_winding_number.py                      │
│  get_roots_and_PMGBZ()    ← 根求解 + PMGBZ 检测           │
│  get_strip_winding()      ← strip winding 计算            │
│  get_loop_winding()       ← 单环卷绕数                    │
│  process_interval()       ← 统一递归区间处理器             │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                  winding.py                                │
│  PolyDiffContext   ← 多项式 + 偏导缓存                     │
│  WindingFun        ← 卷绕数积分参数化                       │
│  MatWindingFun     ← 矩阵形式卷绕数                         │
│  get_winding_number ← 数值积分                             │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                 root_solver.py                             │
│  calculate_point_roots  ← 单点多项式求根                    │
│  complex_root           ← 复方程求根（含 Jacobian）         │
│  ComplexEqConverter     ← 复→实方程转换                    │
│  poly_to_np_coefficients ← Laurent → numpy 系数            │
└──────────────────────────────────────────────────────────┘
```

## 3. 关键组件

### 3.1 `PolyDiffContext`（`winding.py`）

多项式及其偏导数的缓存上下文，职责分离自 `WindingFun`：

```python
class PolyDiffContext:
    char_poly: pt.CLaurent          # 特征多项式
    dchar_poly: list[pt.CLaurent]   # 对各变量的偏导（预计算缓存）

    def eval_val(var) -> complex             # 多项式求值
    def eval_partials(var) -> list[complex]  # 偏导求值
    def eval_dmu2(var) -> tuple[float, float]  # (dμ₂/dμ₁, dμ₂/dθ₁)
```

### 3.2 `WindingFun` / `MatWindingFun`（`winding.py`）

卷绕数计算的壳层：
- `WindingFun`：多项式绕数，参数化回路 + 解析导数
- `MatWindingFun`：矩阵绕数（sparse solver），用于大矩阵模型

### 3.3 `process_interval()`（`strip_winding_number.py`）

统一递归区间处理器（本轮重构核心）：

```
process_interval(θ_left, roots_left, θ_right, roots_right):
  1. 分析边界簇匹配可信度 (_analyze_boundary_matching)
     - 在 M-1/M 边界等模簇的局部代价矩阵上评估
     - 计算"当前匹配"与"边界交换替代匹配"的代价差
     - confidence = min_exchange_margin / cost_scale
  2. 若不可信 → 细分区间（递归）
  3. 若可信且无跨边界交换 → 当前区间结束
  4. 若可信且有跨边界交换 → 执行 PMGBZ 搜索
     - refine_accidental_point() 二分定位
     - _validate_pmgbz_point() 验证并分类正/负/零荷
  5. 左右子区间递归
```

### 3.4 复球面弦距匹配（`strip_winding_number.py`）

Hungarian 匹配的代价函数改用复球面弦距（避免复平面距离下的 $0/\infty$ 伪交换）：

```python
def _chordal_cost_matrix(roots_from, roots_to):
    # 将复根投影到 Riemann 球面 (R³)
    # 计算 R³ 中的欧氏距离作为匹配代价
```

### 3.5 Continuum 处理

当 $|\beta_M| = |\beta_{M+1}|$ 在连续 $\theta_1$ 区间上成立时：
- Strip winding 返回 $(W_{\text{left}}, W_{\text{right}})$ 元组
- 通过对 $\mu_1$ 做 $\pm\varepsilon$ 微扰来解析左右极限

## 4. API 参考

### 4.1 `calculate_point_roots`

```python
def calculate_point_roots(
    char_poly: pt.CLaurent,
    param_ind_ctype: pt.CIndexVec,   # 参数变量索引 (0,1) = (E,β₁)
    param_val: np.ndarray,           # 参数值
    var_ind_ctype: pt.CIndexVec,     # 求解变量索引 [2] = β₂
    M_max: int,                      # 分母阶数
    N_max: int,                      # 分子最大阶 - M
) -> list[complex]:
```

固定 $(E, \beta_1)$，求解 $\beta_2$ 的一元 Laurent 多项式根。

### 4.2 `get_roots_and_PMGBZ`

```python
def get_roots_and_PMGBZ(
    poly_diff: PolyDiffContext,
    E_ref: complex,
    mu1: float,
    N_points: int = 301,
    zero_tol: float = 1e-10,
    GBZ_check_tol: float = 1e-6,
) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
```

**返回**：`(PMGBZ_points, theta1_arr, sols_arr, info)`
- `PMGBZ_points`：PMGBZ 点列表，每个点含 `theta1`、`beta2_sols`、`is_continuum`
- `theta1_arr`：扩展后的 θ₁ 网格（含周期闭包）
- `sols_arr`：对应的排序根数组
- `info`：`{"M", "N", "continuum_flag"}`

### 4.3 `get_strip_winding`

```python
def get_strip_winding(
    poly_diff: PolyDiffContext,
    E_ref: complex,
    mu1: float,
    N_points: int = 301,
    with_gap_info: bool = False,
    continuum_perturb: float = 1e-2,
    zero_tol: float = 1e-10,
    GBZ_check_tol: float = 1e-6,
) -> tuple:
```

**返回**：`(W_strip, PMGBZ_points)` 或 `(W_strip, PMGBZ_points, gap_info)`

Continuum 存在时 $W_{\text{strip}}$ 为 $(W_{\text{left}}, W_{\text{right}})$ 元组。

### 4.4 `get_loop_winding`

```python
def get_loop_winding(
    poly_diff: PolyDiffContext,
    E_ref: complex,
    mu1: float,
    mu2_fun: callable,    # μ₂(θ₁) 插值函数
    theta2: float,        # 固定 θ₂
    N_seg: int = 5,
) -> float:
```

单环（固定 θ₂、沿 PMGBZ 曲线）的卷绕数。

### 4.5 `SGBZSolver`

```python
class SGBZSolver:
    def __init__(self, char_poly: pt.CLaurent)
    def solve_for_E(
        self, E_ref: complex,
        mu1_guess: tuple[float, float] = (-1, 1),
        zero_tol: float = 1e-10,
        N_points: int = 101,
    ) -> tuple[float, list[dict]]:
```

二分 μ₁ 使 strip winding = 0，返回 (μ₁, PMGBZ_points)。

### 4.6 `check_SGBZ`

```python
def check_SGBZ(
    coeffs: np.ndarray, degs: np.ndarray,
    E_ref: complex, perc: float,
    debug_mode: bool = False,
) -> dict:
```

批量检查入口——给定多项式系数、能量、进度百分比，返回 SGBZ 判定结果。

### 4.7 其他导出

| 函数 | 功能 |
|------|------|
| `complex_root` | 复方程求根（含解析 Jacobian） |
| `ComplexEqConverter` | 复→实方程 Jacobian 转换 |
| `poly_to_np_coefficients` | Laurent 多项式 → numpy 系数数组 |
| `get_winding_number` | 数值积分求卷绕数 |
| `get_minor_degrees` | 读取 (M, N) |

## 5. 关键数值参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `zero_tol` | 1e-10 | PMGBZ gap 判零阈值 |
| `GBZ_check_tol` | 1e-6 | 等模簇边界检测容差 |
| `match_confidence_tol` | 1e-3 | 匈牙利匹配可信度相对裕量 |
| `continuum_perturb` | 1e-2 | Continuum 时 μ₁ 微扰量 |
| `double_root_tol` | `GBZ_check_tol` | 重根检测容差 |

`match_confidence_tol` 是经验阈值，可能需要根据实际模型扫描结果微调。

## 6. 与 `brute_force_amoeba` 的关系

| 方面 | brute_force_solver (SGBZ) | brute_force_amoeba |
|------|---------------------------|-------------------|
| 基流形 | $\mu_2 = \rho_{2,0}(\theta_1)$ 变曲线 | $\mu_2 = \text{const}$ 水平面 |
| 根排序 | 按 $\vert\beta_2\vert$ | Hungarian 匹配连续追踪 |
| 积分方向 | $\theta_1$ 单方向 | $\theta_1$ 和 $\theta_2$ 双方向 |
| continuum 判据 | $\vert\beta_M\vert \approx \vert\beta_{M+1}\vert$ | $\vert\ln\vert\beta_2\vert - \mu_2\vert \approx 0$ |
| 匹配可信度 | 交换代价差分析 | 不需要（连续追踪） |
| 谱包含 | $\sigma_{\text{SGBZ}} \subseteq \sigma_{\text{Amoeba}}$ | — |
