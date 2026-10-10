# brute_force_amoeba — Amoebic Spectrum Calculation

基于 amoeba formulation 的非厄米能谱计算模块。与 SGBZ formulation（`brute_force_solver/`）互补。

## 1. 理论背景

### 1.1 Amoeba 定义

对 $n$ 元 Laurent 多项式 $p(\boldsymbol{\beta})$，其 **amoeba** 定义为：

$$\mathcal{A}_p \equiv \left\{ \ln|\boldsymbol{\beta}| \;\middle|\; p(\boldsymbol{\beta}) = 0 \right\}$$

2D 非厄米晶格固定参考能量 $E$ 后，特征多项式 $f(E, \boldsymbol{\beta}) = \det[E - h(\boldsymbol{\beta})]$ 是二元 Laurent 多项式，对应 amoeba：

$$\mathcal{A}(E) \equiv \mathcal{A}_{f(E,\cdot)}$$

直观上 $\mathcal{A}(E)$ 是给定 $E$ 下所有允许的非 Bloch 缩放指数 $\ln|\beta|$ 的集合。

### 1.2 Ronkin 函数与卷绕数

**Ronkin 函数**：

$$R_p(\boldsymbol{\mu}) = \int_{\mathbb{T}^n} \left(\frac{d\boldsymbol{\theta}}{2\pi}\right)^n \ln|p(e^{\boldsymbol{\mu}+i\boldsymbol{\theta}})|$$

其梯度与 **平均卷绕数** 的关系：

$$\frac{\partial R_p}{\partial \mu_j} = \int_{\mathbb{T}^{n-1}} \frac{d\theta_1 \cdots \widehat{d\theta_j} \cdots d\theta_n}{(2\pi)^{n-1}} \; u_j(\theta_1, \dots, \widehat{\theta_j}, \dots, \theta_n; \boldsymbol{\mu})$$

其中 $u_j$ 是固定其他变量时关于 $\theta_j$ 的卷绕数。

### 1.3 Amoeba Spectrum 判定

在 amoeba 的 **洞**（hole）中——即 $\boldsymbol{\mu} \notin \mathcal{A}(E)$ 中被 $\mathcal{A}(E)$ 包围的连通开集——卷绕数 $u_j$ 是拓扑不变量，$\nabla R_p$ 保持常数。

- **中心洞**（central hole）：$\nabla R_E = 0$ 的洞
- $E \in \sigma_{\text{Amoeba}}$：$\mathcal{A}(E)$ **没有**中心洞
- $E \notin \sigma_{\text{Amoeba}}$：$\mathcal{A}(E)$ **存在**中心洞

### 1.4 与 SGBZ 的关系

| 方面 | SGBZ | Amoeba |
|------|------|--------|
| 基流形 | $X(E,\mu_1)$，$\mu_2 = \rho_{2,0}(\theta_1;E,\mu_1)$ 随 $\theta_1$ 变化 | 水平面 $\mu_2 = \text{const}$ |
| 根排序 | 按 $\vert\beta_2\vert$ 排序 | Hungarian 匹配连续追踪 |
| continuum 检测 | $\vert\beta_M\vert \approx \vert\beta_{M+1}\vert$ | $\vert\ln\vert\beta_2\vert - \mu_2\vert < \text{tol}$ |
| 谱包含 | $\sigma_{\text{Amoeba}} \supset \cup_j \sigma_{\text{SGBZ},j}$ | — |

## 2. 算法架构

```
┌──────────────────────────────────────────────────────────┐
│                      Public API                           │
│  get_a2_average_winding  get_a1_average_winding           │
│  bisect_a2_winding       bisect_amoeba_ronkin_min         │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│              _compute_root_tracks                         │
│  Hungarian 匹配 → 根轨迹追踪 → tracks dict                │
│  依赖：(E, μ₁)  only  —  与 μ₂ 无关，可缓存               │
└────────────────────┬─────────────────────────────────────┘
                     │  tracks dict
┌────────────────────▼─────────────────────────────────────┐
│           _compute_winding_from_tracks                    │
│  continuum 检测 → crossing 检测（含绕数跳变标签）          │
│  → fsolve 精化（可选，refine_crossings=True/False）       │
│  → 输出：(winding, zeros, has_continuum, dW/dμ₂)         │
└────────────────────┬─────────────────────────────────────┘
                     │  zeros
┌────────────────────▼─────────────────────────────────────┐
│          _get_average_winding_from_zeros                  │
│  零点分区 → 分段中点求根 → 计数 → 加权平均                │
│  direction=2: θ₁ 分区, 解 β₂                              │
│  direction=1: θ₂ 分区, 解 β₁                              │
└──────────────────────────────────────────────────────────┘
```

### 2.1 Hungarian 匹配根追踪

与 SGBZ 按 $\vert\beta_2\vert$ 排序不同，amoeba 使用 Hungarian 匹配追踪每根 track：

1. 在 $\theta_1$ 网格上逐点求解 $\beta_2$ 根
2. 从 $\theta_1 = 0$ 开始，用 Hungarian 匹配逐点传播匹配关系
3. 代价函数：复球面弦距（避免 $0/\infty$ 伪交换）
4. 输出：连续 root tracks `tracked[θ₁_idx, root_j]`

### 2.2 Crossing 检测与懒精化

对每根 track，检测 $\ln\vert\beta_2\vert$ 穿越 $\mu_2$ 的位置：

- **离散 crossing**：$d_0 \cdot d_1 < 0$（$d_i = \ln\vert\beta_2\vert - \mu_2$）。每个 crossing 标注绕数跳变方向：穿过 μ₂ 上升 → jump=+1，下降 → jump=−1。
- **Continuum 噪声过滤**：若 crossing 两侧点都在 continuum band 内，则为数值噪声，跳过。
- **懒精化**：二分迭代期间跳过 fsolve（`refine_crossings=False`），直接用线性插值的 $(\theta_1^{\text{approx}}, \theta_2^{\text{approx}})$ 作为零点。仅在二分收敛到最终 $\mu_2$ 后才做一次 fsolve 精化。

### 2.3 平均绕数对 μ₂ 的解析导数

精化后，对每个零点 $(\theta_1, \theta_2)$ 求解 $2\times2$ 线性系统得 $\dot{\theta}_1 = d\theta_1/d\mu_2$：

$$\frac{dW}{d\mu_2} = \frac{1}{2\pi} \sum_i \text{jump}_i \cdot \dot{\theta}_{1,i}$$

用于精化后对 $\mu_2$ 做一步 Newton 校正，避免重做二分。

### 2.4 统一绕数计算

`_get_average_winding_from_zeros` 统一处理 a1 和 a2 方向：

```
给定零点列表 [(θ₁,θ₂), ...] 和方向 d ∈ {1,2}:
  1. 提取方向 d 的角度坐标 → 排序 → N 个分段
  2. 对每个分段中点：
     - 固定 |β_param| = exp(μ_param)，解另一变量的根
     - 统计 |β_solve| < exp(μ_count) 的根数
     - u_d = count_below - M_d
  3. average = Σ(u_d × segment_width) / (2π)
```

a2 方向有独立的 Hungarian matching 管线；a1 复用 a2 的零点结果（不同方向用不同角度坐标分区）。

### 2.5 二分求解

**内层**（$\mu_2$ 二分，`_find_mu2_for_a2_zero`）：

```
给定 (E, μ₁)，二分 μ₂ 使 a2 绕数 = target:
  1. 预计算 _compute_root_tracks（一次 Hungarian）→ tracks dict 缓存
  2. 自适应区间扩展：用未精化绕数（refine_crossings=False）快速评估
  3. 二分迭代：每步一次 _compute_winding_from_tracks（不做 fsolve）
  4. 收敛后精化：refine_crossings=True → 得精化零点 + dW/dμ₂
  5. Newton 校正：μ₂ ← μ₂₀ − W_ref / (dW/dμ₂)
  6. continuum 处理：μ₂ ± ε 微扰，异号→停止，同号→继续
```

**外层**（$\mu_1$ 二分，`bisect_amoeba_ronkin_min`）：

```
给定 E，二分 μ₁ 使 a1 绕数 = 0（同时保证 a2 = 0）:
  1. 在 μ₁_mid 处，内层二分求 μ₂ 使 a2 = 0 → 得 (μ₂, zeros)
  2. 直接用 zeros 计算 a1（复用零点，不需重跑 Hungarian）
  3. a1 degenerate（零点无法分区 θ₂）：
     - 先检查 a1_mid 是否已为零（零平台提前退出）
     - 否则 μ₁ ± ε 微扰，重做内层二分 → 比较两侧 a1 符号
     - 异号 → 到达边界，停止
     - 同号 → 用符号指导外层二分方向
```

## 3. API 参考

### 3.1 `get_hungarian_sorted_roots`

```python
def get_hungarian_sorted_roots(
    char_poly: pt.CLaurent,
    E_ref: complex,
    mu1: float,
    N_points: int = 301,
) -> tuple[np.ndarray, np.ndarray, int, int]:
```

**返回**：`(theta1_arr, tracked_roots, M, N)`
- `theta1_arr`: (N_points,) θ₁ 网格
- `tracked_roots`: (N_points, M+N) 连续 root tracks
- `M`: β₂ 分母阶数
- `N`: β₂ 分子最高阶 - M

### 3.2 `get_a2_average_winding`

```python
def get_a2_average_winding(
    char_poly: pt.CLaurent,
    E_ref: complex,
    mu1: float,
    mu2: float,
    N_points: int = 301,
    continuum_tol: float = 1e-8,
    min_continuum_pts: int = 3,
) -> float:
```

**返回**：$\partial R / \partial \mu_2$，a2 方向平均绕数。

### 3.3 `get_a1_average_winding`

```python
def get_a1_average_winding(
    char_poly: pt.CLaurent,
    E_ref: complex,
    mu1: float,
    mu2: float,
    N_points: int = 301,
    continuum_tol: float = 1e-8,
    min_continuum_pts: int = 3,
) -> float:
```

**返回**：$\partial R / \partial \mu_1$。复用 a2 零点（不需独立的 Hungarian 匹配）。

### 3.4 `bisect_a2_winding`

```python
def bisect_a2_winding(
    char_poly: pt.CLaurent,
    E_ref: complex,
    mu1: float,
    mu2_low: float,
    mu2_high: float,
    target_winding: float = 0.0,
    N_points: int = 301,
    continuum_tol: float = 1e-8,
    min_continuum_pts: int = 3,
    continuum_perturb: float = 1e-4,
    max_iter: int = 60,
    xtol: float = 1e-10,
) -> dict:
```

**返回**：`{"mu2", "zeros", "is_continuum", "winding", "success"}`

二分 μ₂ 使 a2 绕数穿过 `target_winding`。无自适应区间扩展（区间由调用者保证异号）。

### 3.5 `bisect_amoeba_ronkin_min`

```python
def bisect_amoeba_ronkin_min(
    char_poly: pt.CLaurent,
    E_ref: complex,
    mu1_low: float,
    mu1_high: float,
    mu2_low: float,
    mu2_high: float,
    N_points: int = 301,
    continuum_tol: float = 1e-8,
    min_continuum_pts: int = 3,
    continuum_perturb: float = 1e-4,
    max_iter: int = 60,
    xtol: float = 1e-10,
    max_range_expansions: int = 10,
    range_expand_factor: float = 2.0,
) -> dict:
```

**返回**：`{"mu1", "mu2", "zeros", "is_continuum", "success"}`

求 Ronkin 函数极小值——同时满足 a1 = a2 = 0 的 $(\mu_1, \mu_2)$。外层二分 μ₁，内层二分 μ₂，含自适应区间扩展和 continuum 处理。

## 4. CLaurent 多项式格式

特征多项式 $f(E, \beta_1, \beta_2)$ 用三项组编码：

```
[E_exponent, beta1_exponent, beta2_exponent]
```

**示例**：方格子最近邻 $f = E - \beta_1 - \beta_1^{-1} - \beta_2 - \beta_2^{-1}$

```python
coeffs, degs = [], []
coeffs.append(1.0);   degs.extend([ 1,  0,  0])   # +E
coeffs.append(-1.0);  degs.extend([ 0,  1,  0])   # -beta1
coeffs.append(-1.0);  degs.extend([ 0, -1,  0])   # -beta1^{-1}
coeffs.append(-1.0);  degs.extend([ 0,  0,  1])   # -beta2
coeffs.append(-1.0);  degs.extend([ 0,  0, -1])   # -beta2^{-1}
```

## 5. 内部函数

| 函数 | 功能 |
|------|------|
| `_get_minor_degrees_for_direction` | 读取指定方向的 (M, N) |
| `_find_cyclic_true_intervals` | 循环布尔数组中找连续 True 区间 |
| `_find_exact_crossing` | `fsolve` + 解析 Jacobian 精化单个零点 |
| `_compute_zero_dtheta1_dmu2` | 精化零点处求解 dθ₁/dμ₂（2×2 线性系统） |
| `_compute_root_tracks` | Hungarian 匹配 + 根轨迹追踪 → tracks dict（与 μ₂ 无关） |
| `_compute_winding_from_tracks` | 从 tracks 检测 crossing + continuum + 绕数 + dW/dμ₂ |
| `_compute_crossings_and_winding` | 薄封装（调 `_compute_root_tracks` + `_compute_winding_from_tracks`） |
| `_get_average_winding_from_zeros` | 统一绕数函数（支持 a1/a2 双向） |
| `_refine_and_correct` | 精化零点 + 解析导数 Newton 校正 μ₂ |
| `_find_mu2_for_a2_zero` | 内层 μ₂ 二分（含自适应区间扩展 + 懒精化 + Newton 校正） |
| `_a1_is_degenerate` | 启发式判断 a1 是否退化 |

## 6. 演示

运行 `demos/demo-amoeba.py`：

```bash
cd phcpy-free-algorithm/demos
PYTHONPATH="../:../brute_force_solver" python demo-amoeba.py
```

演示内容：
1. winding sweep — 固定 μ₁ 扫描 μ₂，对比 a1/a2 绕数
2. Hungarian 根追踪 — 展示连续 root tracks
3. μ₂ 二分 — 找 a2=0 的位置
4. Ronkin 极小值 — 外层二分求 (μ₁, μ₂) 使 a1=a2=0
5. 次近邻模型 — NNN coupling 下的 Ronkin 极小值
