# SciPy

> Fundamental algorithms for scientific computing in Python.

SciPy builds on NumPy to provide optimization, integration, interpolation, linear algebra, statistics, signal and image processing, sparse matrices, spatial data structures, special functions, and more. Most routines wrap battle-tested compiled libraries (LAPACK, ARPACK, HiGHS, QUADPACK, FITPACK, Qhull, PocketFFT) behind a consistent Python API.

Covers SciPy 1.17 (examples verified against SciPy 1.17.1 with NumPy 2.4). Most examples also run on SciPy 1.11+; version-specific features are marked.

## Overview

### What SciPy is

SciPy is a collection of subpackages, each focused on a domain of numerical computing. Unlike NumPy, you import subpackages explicitly:

```python
from scipy import optimize, stats, linalg, sparse, integrate, interpolate, signal
```

| Subpackage | Purpose |
|------------|---------|
| `scipy.optimize` | Minimization, root finding, curve fitting, linear and integer programming, assignment |
| `scipy.stats` | 100+ probability distributions, hypothesis tests, descriptive stats, resampling, KDE, QMC |
| `scipy.linalg` | Dense linear algebra (superset of `numpy.linalg`), matrix functions, decompositions |
| `scipy.sparse` | Sparse arrays (CSR, CSC, COO, ...) and `scipy.sparse.linalg` solvers and eigensolvers |
| `scipy.integrate` | Numerical integration (quadrature) and ODE solvers |
| `scipy.interpolate` | 1-D and N-D interpolation, splines, RBFs |
| `scipy.signal` | Filtering, spectral analysis, peak finding, convolution |
| `scipy.fft` | Fast Fourier transforms (faster, more featureful than `numpy.fft`) |
| `scipy.spatial` | KD-trees, distance metrics, convex hulls, Delaunay, Voronoi, rotations |
| `scipy.special` | Special functions (gamma, erf, Bessel, `expit`, `logsumexp`, ...) |
| `scipy.cluster` | Hierarchical clustering and vector quantization (k-means) |
| `scipy.ndimage` | N-dimensional image processing (filters, morphology, labeling) |
| `scipy.io` | MATLAB `.mat`, WAV, Matrix Market, ARFF, NetCDF readers and writers |
| `scipy.differentiate` | Numerical differentiation (new in 1.15) |
| `scipy.constants` | Physical constants and unit conversions |
| `scipy.datasets` | Small example datasets (downloaded on demand via `pooch`) |

### History and maintainers

- 2001: Travis Oliphant, Eric Jones, and Pearu Peterson merge their numerical code into SciPy, built on Numeric.
- 2006: Moves onto NumPy.
- 2017: SciPy 1.0 released after 16 years of development.
- 2020: SciPy 1.0 paper published in *Nature Methods*.
- 2022+: Build system moved to Meson; ongoing Array API support (CuPy, PyTorch, JAX inputs for selected functions); sparse **arrays** (`csr_array`) recommended over sparse matrices.
- SciPy is a NumFOCUS fiscally sponsored project, BSD-3-Clause licensed, maintained by a large community of volunteers and funded maintainers.

### When to use SciPy

- Fitting model parameters (`curve_fit`, `least_squares`, `minimize`).
- Statistical testing and probability distributions for A/B tests, experiment analysis, and model evaluation.
- Sparse matrices for text features (bag-of-words, TF-IDF), graphs, and recommender systems.
- Signal processing on sensors, audio, and time-series before feature extraction.
- Fast nearest-neighbor queries (`KDTree`) and pairwise distances (`cdist`).
- Numerically stable special functions (`logsumexp`, `expit`, `gammaln`) inside ML code.

### When not to use SciPy

| Situation | Better tool |
|-----------|-------------|
| Training ML models end to end | scikit-learn, PyTorch, XGBoost |
| Gradient-based optimization of neural networks | PyTorch / JAX optimizers (autodiff, GPU) |
| Regression tables, GLMs, time-series models with summaries | statsmodels |
| Bayesian modeling | PyMC, NumPyro, Stan |
| Large-scale graph analytics | NetworkX, igraph, graph-tool (SciPy has `scipy.sparse.csgraph` for basics) |
| Image processing pipelines with many algorithms | scikit-image, OpenCV |
| GPU array math | CuPy (`cupyx.scipy` mirrors many SciPy modules) |

### Where it fits in the ML stack

```text
   scikit-learn / statsmodels / scikit-image / PyMC
                 |  (use SciPy internally)
   SciPy: optimize, stats, sparse, linalg, signal, spatial, special
                 |
   NumPy ndarray
                 |
   Compiled libraries: LAPACK/BLAS, HiGHS, ARPACK, QUADPACK, Qhull, PocketFFT
```

scikit-learn depends on SciPy directly: sparse inputs are `scipy.sparse` objects, many estimators call `scipy.optimize` and `scipy.linalg`, and its `TfidfVectorizer` returns a SciPy sparse matrix.

## Installation

### pip

```bash
pip install scipy
pip install "scipy>=1.15"        # minimum version for scipy.differentiate
pip install scipy pooch          # pooch is needed for scipy.datasets
```

Wheels bundle OpenBLAS, so no Fortran compiler is required.

### conda

```bash
conda install -c conda-forge scipy
```

### GPU variant

SciPy is CPU-only, but selected functions accept CuPy, PyTorch, or JAX arrays via the Array API (set `SCIPY_ARRAY_API=1`). For broad GPU coverage use CuPy's `cupyx.scipy`:

```bash
pip install cupy-cuda12x
```

```python
import cupy as cp
from cupyx.scipy import ndimage, sparse, signal   # GPU versions of SciPy modules
```

### Verifying the install

```python
import scipy

print(scipy.__version__)     # e.g. 1.17.1
scipy.show_config()          # BLAS/LAPACK and build information
```

```bash
python -c "import scipy; print(scipy.__version__)"
```

Requirements for SciPy 1.17: Python 3.11+, NumPy 1.26.4 or newer (up to the 2.x series it was tested against).

## Core Concepts

### Subpackages are imported explicitly

```python
import scipy
from scipy import stats          # recommended style
import scipy.optimize as opt     # also common

stats.norm.cdf(1.96)             # 0.9750021048517795
```

### Result objects

Most SciPy functions return result objects with named attributes, rather than bare tuples. They usually still unpack like tuples for backward compatibility.

```python
import numpy as np
from scipy import optimize, stats

res = optimize.minimize(lambda x: (x[0] - 3) ** 2, x0=[0.0])
print(res.x, res.fun, res.success, res.message)

t = stats.ttest_ind([1, 2, 3], [4, 5, 6])
print(t.statistic, t.pvalue)   # -3.674..., 0.0213...
stat, p = t                    # tuple unpacking still works
print(t.df)                    # 4.0, extra attributes are available only by name
ci = t.confidence_interval(confidence_level=0.95)
```

### Distributions are objects

Every distribution in `scipy.stats` exposes the same methods: `pdf`/`pmf`, `cdf`, `sf` (1 - cdf), `ppf` (inverse cdf), `isf`, `rvs` (random samples), `mean`, `var`, `std`, `interval`, `entropy`, `fit`. You can call them with parameters each time or **freeze** a distribution:

```python
from scipy import stats

stats.norm.pdf(0, loc=0, scale=1)        # 0.3989...
d = stats.norm(loc=100, scale=15)        # frozen distribution
d.cdf(130)                               # P(X <= 130) = 0.9772...
d.ppf(0.975)                             # 97.5th percentile = 129.39...
d.interval(0.95)                         # (70.6..., 129.4...)
d.rvs(size=5, random_state=0)
stats.binom(n=10, p=0.3).pmf(3)          # 0.2668...
```

All continuous distributions take `loc` (shift) and `scale` (stretch) in addition to their shape parameters. For example, `stats.expon(scale=1/lam)` is an exponential with rate `lam`, and `stats.gamma(a, scale=theta)`.

SciPy 1.15 added a new object-oriented random-variable infrastructure (`stats.Normal`, `stats.Uniform`, `stats.make_distribution`) with keyword-only parameters, which is the long-term direction; the classic `stats.norm` API remains fully supported.

### Randomness: the rng keyword

Since SciPy 1.15, functions that use randomness accept `rng=` (a `numpy.random.Generator` or seed), standardizing the older `random_state=` and `seed=` keywords (SPEC 7). Older keywords keep working for now.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
data = rng.normal(size=50)
res = stats.bootstrap((data,), np.mean, rng=rng)
```

### Dense vs sparse

Sparse arrays store only non-zero entries. They are essential for high-dimensional, mostly-zero data (one-hot encodings, text, graphs). Use the **array** classes (`csr_array`, `coo_array`, ...); the older `*_matrix` classes behave like `np.matrix` (`*` means matrix multiplication) and are kept for compatibility.

```python
import numpy as np
from scipy import sparse

A = sparse.csr_array(np.array([[0, 0, 1], [2, 0, 0]]))
print(A.nnz)            # 2 stored values
print(A @ np.ones(3))   # [1. 2.]
print(A.toarray())
```

### Callable-based APIs

Optimizers, integrators, and root finders take a Python function. Make it vectorized and cheap, because it is called many times, and pass extra parameters with `args=`:

```python
from scipy import optimize

def f(x, a, b):
    return (x[0] - a) ** 2 + (x[1] - b) ** 2

res = optimize.minimize(f, x0=[0, 0], args=(1.0, -2.0))
print(res.x.round(4))   # [ 1. -2.]
```

### Notable deprecations and removals

| Removed / deprecated | Use instead |
|----------------------|-------------|
| `integrate.simps`, `trapz`, `cumtrapz` (removed 1.14) | `integrate.simpson`, `trapezoid`, `cumulative_trapezoid` |
| `interpolate.interp2d` (removed 1.14) | `RegularGridInterpolator`, `RectBivariateSpline`, `bisplrep` |
| `interpolate.interp1d` (legacy, not recommended) | `np.interp`, `make_interp_spline`, `CubicSpline`, `PchipInterpolator` |
| `stats.binom_test` (removed 1.12) | `stats.binomtest` |
| `special.sph_harm` (removed 1.17) | `special.sph_harm_y` |
| `signal.cwt`, `signal.ricker`, `signal.morlet` (removed 1.15) | PyWavelets |
| `spatial.distance.kulsinski` (removed 1.11) | `kulczynski1` or other metrics |
| `scipy.misc` (deprecated) | `scipy.datasets` for sample images; `scipy.differentiate` for derivatives |
| `scipy.odr` (deprecated in 1.17, removal planned for 1.19) | the separate `odrpack` package |
| `*_matrix` sparse classes (still supported) | `*_array` classes, `sparse.diags_array`, `sparse.eye_array`, `sparse.random_array` |
| `random_state=` / `seed=` keywords | `rng=` (1.15+) |
| `integrate.odeint` (legacy) | `integrate.solve_ivp` |
| `optimize.fmin`, `fmin_bfgs`, ... (legacy) | `optimize.minimize(method=...)` |
| Non-public modules like `scipy.optimize.minpack` | Import from the public subpackage namespace |

## API Reference

All examples assume `import numpy as np` plus the subpackage import shown.

### scipy.optimize

#### minimize

```python
scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fun` | callable | required | Objective `fun(x, *args) -> float`; `x` is a 1-D array. |
| `x0` | array | required | Initial guess. |
| `args` | tuple | `()` | Extra arguments passed to `fun`, `jac`, `hess`. |
| `method` | str | `None` | `'BFGS'` (unconstrained default), `'L-BFGS-B'` (default with bounds), `'SLSQP'` (default with constraints), `'Nelder-Mead'`, `'Powell'`, `'CG'`, `'Newton-CG'`, `'trust-constr'`, `'TNC'`, `'COBYLA'`, `'COBYQA'`, ... |
| `jac` | callable, bool, or str | `None` | Gradient. `True` means `fun` returns `(value, grad)`. Finite differences if omitted. |
| `bounds` | sequence or `Bounds` | `None` | `[(lo, hi), ...]`; use `None` for unbounded sides. |
| `constraints` | dict, list, `LinearConstraint`, `NonlinearConstraint` | `()` | Constraints (SLSQP, trust-constr, COBYLA, COBYQA). |
| `tol` | float | `None` | Termination tolerance. |
| `options` | dict | `None` | Solver-specific, e.g. `{"maxiter": 500, "disp": True}`. |

Returns: `OptimizeResult` with `x`, `fun`, `success`, `status`, `message`, `nit`, `nfev`, and (method dependent) `jac`, `hess_inv`.

```python
import numpy as np
from scipy import optimize

def rosen(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

res = optimize.minimize(rosen, x0=np.zeros(5), method="L-BFGS-B", jac=optimize.rosen_der)
print(res.success, res.x.round(4))     # True [1. 1. 1. 1. 1.]

# bounds and a constraint x0 + x1 <= 1
cons = [{"type": "ineq", "fun": lambda x: 1 - x[0] - x[1]}]
res = optimize.minimize(lambda x: -(x[0] * x[1]), x0=[0.1, 0.1],
                        bounds=[(0, None), (0, None)], constraints=cons, method="SLSQP")
print(res.x.round(3))                  # [0.5 0.5]
```

#### minimize_scalar

```python
scipy.optimize.minimize_scalar(fun, bracket=None, bounds=None, args=(), method=None, tol=None, options=None)
```

| Parameter | Description |
|-----------|-------------|
| `method` | `'brent'` (default), `'bounded'` (requires `bounds`), `'golden'` |
| `bounds` | `(lo, hi)` for `'bounded'` |

```python
res = optimize.minimize_scalar(lambda x: (x - 2) ** 2 + 1, bounds=(0, 5), method="bounded")
print(round(res.x, 4), round(res.fun, 4))   # 2.0 1.0
```

#### curve_fit

```python
scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-inf, inf), method=None, jac=None, *, full_output=False, nan_policy=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `f` | callable | required | Model `f(x, *params)`. |
| `xdata`, `ydata` | arrays | required | Observations. |
| `p0` | sequence | `None` (all ones) | Initial parameter guess. Good guesses matter for nonlinear models. |
| `sigma` | array | `None` | Uncertainty of `ydata` (weights). |
| `bounds` | 2-tuple | `(-inf, inf)` | Lower and upper bounds on parameters. |
| `method` | {'lm','trf','dogbox'} | `'lm'` unconstrained, `'trf'` with bounds | Algorithm. |

Returns: `(popt, pcov)`; standard errors are `np.sqrt(np.diag(pcov))`.

```python
def model(x, a, k, c):
    return a * np.exp(-k * x) + c

x = np.linspace(0, 4, 50)
rng = np.random.default_rng(0)
y = model(x, 2.5, 1.3, 0.5) + rng.normal(0, 0.05, x.size)
popt, pcov = optimize.curve_fit(model, x, y, p0=[1, 1, 0])
perr = np.sqrt(np.diag(pcov))
print(popt.round(2))      # [2.5  1.38 0.53]  (true values 2.5, 1.3, 0.5)
```

#### least_squares

```python
scipy.optimize.least_squares(fun, x0, jac='2-point', bounds=(-inf, inf), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=None, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options=None, jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs=None, callback=None, workers=None)
```

| Parameter | Description |
|-----------|-------------|
| `fun` | Returns the residual vector `r(x)`; minimizes `0.5 * sum(loss(r**2))`. |
| `loss` | `'linear'`, `'soft_l1'`, `'huber'`, `'cauchy'`, `'arctan'`; robust losses reduce outlier influence. |
| `f_scale` | Residual scale at which robust loss kicks in. |
| `method` | `'trf'` (default, supports bounds), `'dogbox'`, `'lm'`. |

```python
xs = np.linspace(0, 10, 30)
ys = 3 * xs + 1
ys[5] = 80                           # outlier
res = optimize.least_squares(lambda p: p[0] * xs + p[1] - ys, x0=[1, 0], loss="soft_l1", f_scale=1.0)
print(res.x.round(2))                # close to [3. 1.] despite the outlier
```

#### root_scalar, brentq, root

```python
scipy.optimize.root_scalar(f, args=(), method=None, bracket=None, fprime=None, fprime2=None, x0=None, x1=None, xtol=None, rtol=None, maxiter=None, options=None)
scipy.optimize.brentq(f, a, b, args=(), xtol=2e-12, rtol=8.88e-16, maxiter=100, full_output=False, disp=True)
scipy.optimize.root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None)
```

```python
r = optimize.root_scalar(lambda x: x**3 - 2 * x - 5, bracket=[2, 3], method="brentq")
print(round(r.root, 6))              # 2.094551
optimize.brentq(np.cos, 0, 3)        # 1.5707963... (pi / 2)

# system of equations: x + y = 3, x * y = 2
sol = optimize.root(lambda v: [v[0] + v[1] - 3, v[0] * v[1] - 2], x0=[0, 1])
print(sol.x.round(4))                # [2. 1.]
```

`brentq` requires `f(a)` and `f(b)` to have opposite signs, otherwise it raises `ValueError: f(a) and f(b) must have different signs`.

#### linprog and milp

```python
scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, None), method='highs', callback=None, options=None, x0=None, integrality=None)
scipy.optimize.milp(c, *, integrality=None, bounds=None, constraints=None, options=None)
```

Both **minimize** `c @ x`. Negate `c` to maximize.

| Parameter | Description |
|-----------|-------------|
| `A_ub`, `b_ub` | Inequality constraints `A_ub @ x <= b_ub` |
| `A_eq`, `b_eq` | Equality constraints |
| `bounds` | Per-variable `(lo, hi)`; default `(0, None)` means non-negative |
| `integrality` | 1 for integer variables, 0 for continuous |

```python
# maximize 3x + 2y  s.t.  x + y <= 4,  x + 3y <= 6,  x, y >= 0
res = optimize.linprog(c=[-3, -2], A_ub=[[1, 1], [1, 3]], b_ub=[4, 6])
print(res.x, -res.fun)               # [4. 0.] 12.0

from scipy.optimize import LinearConstraint, milp
res = milp(c=[-1, -2], constraints=LinearConstraint([[2, 3]], ub=[12.5]),
           integrality=[1, 1], bounds=optimize.Bounds(0, 10))
print(res.x)                         # integer solution x=0, y=4 (prints [-0.  4.])
```

#### Global optimizers

```python
scipy.optimize.differential_evolution(func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, rng=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1, constraints=(), x0=None, *, integrality=None, vectorized=False)
scipy.optimize.dual_annealing(func, bounds, args=(), maxiter=1000, ...)
scipy.optimize.basinhopping(func, x0, niter=100, ...)
scipy.optimize.shgo(func, bounds, ...)
```

```python
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

res = optimize.differential_evolution(rastrigin, bounds=[(-5.12, 5.12)] * 2, rng=1)
print(res.x.round(4), round(res.fun, 6))   # [-0.  0.] 0.0  (global minimum)
```

Use `workers=-1` to parallelize function evaluations across all cores.

#### linear_sum_assignment

```python
scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
```

Solves the assignment problem (Hungarian-style). Used for matching predicted to true objects, or aligning cluster labels.

```python
cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
rows, cols = optimize.linear_sum_assignment(cost)
print(cols, cost[rows, cols].sum())   # [1 0 2] 5
```

### scipy.stats

#### Distributions

| Kind | Common distributions |
|------|---------------------|
| Continuous | `norm`, `uniform`, `expon`, `gamma`, `beta`, `lognorm`, `t`, `chi2`, `f`, `cauchy`, `laplace`, `weibull_min`, `pareto`, `truncnorm`, `vonmises` |
| Discrete | `binom`, `poisson`, `geom`, `nbinom`, `hypergeom`, `bernoulli`, `randint`, `zipf`, `betabinom` |
| Multivariate | `multivariate_normal`, `multivariate_t`, `dirichlet`, `multinomial`, `wishart` |

| Method | Meaning |
|--------|---------|
| `pdf(x)` / `pmf(k)` | Density / mass |
| `logpdf(x)` / `logpmf(k)` | Log density (use for likelihoods) |
| `cdf(x)`, `sf(x)` | P(X <= x), P(X > x) |
| `ppf(q)`, `isf(q)` | Inverse of `cdf`, inverse of `sf` |
| `rvs(size, random_state)` | Random samples |
| `mean()`, `var()`, `std()`, `median()`, `moment(n)`, `stats(moments='mvsk')` | Moments |
| `interval(confidence)` | Central interval |
| `fit(data)` | Maximum-likelihood estimates (class method, e.g. `stats.norm.fit(data)`) |

```python
from scipy import stats

data = stats.gamma(a=2.0, scale=1.5).rvs(size=2000, random_state=1)
a, loc, scale = stats.gamma.fit(data, floc=0)    # fix loc at 0
print(round(a, 2), round(scale, 2))               # 1.92 1.55  (true values 2.0, 1.5)

stats.poisson(mu=3).pmf([0, 1, 2]).round(4)       # [0.0498 0.1494 0.224 ]
stats.t(df=10).ppf(0.975)                         # 2.228...
mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
mvn.logpdf([0.2, -0.1])
```

#### Descriptive statistics

```python
scipy.stats.describe(a, axis=0, ddof=1, bias=True, nan_policy='propagate')
scipy.stats.zscore(a, axis=0, ddof=0, nan_policy='propagate')
scipy.stats.sem(a, axis=0, ddof=1, nan_policy='propagate', *, keepdims=False)
scipy.stats.iqr(x, axis=None, rng=(25, 75), scale=1.0, nan_policy='propagate', interpolation='linear', keepdims=False)
scipy.stats.median_abs_deviation(x, axis=0, center=None, scale=1.0, nan_policy='propagate', *, keepdims=False)
```

`nan_policy` accepts `'propagate'` (return NaN), `'omit'` (ignore NaNs), or `'raise'`.

```python
x = np.array([2.0, 4.0, 4.0, 5.0, 7.0, 9.0, np.nan])
stats.describe(x, nan_policy="omit")
stats.zscore(x, nan_policy="omit")
stats.iqr(x, nan_policy="omit")                   # 2.5
stats.skew(x, nan_policy="omit"), stats.kurtosis(x, nan_policy="omit")
stats.mode([1, 2, 2, 3])                          # ModeResult(mode=2, count=2)
stats.rankdata([10, 30, 20, 20])                  # [1.  4.  2.5 2.5]
stats.gmean([1, 4, 16])                           # 4.0
```

#### Hypothesis tests

| Question | Test | Function |
|----------|------|----------|
| Two independent means | Student / Welch t-test | `ttest_ind(a, b, equal_var=False)` |
| Paired means | Paired t-test | `ttest_rel(a, b)` |
| One mean vs value | One-sample t-test | `ttest_1samp(a, popmean)` |
| Two independent distributions (non-parametric) | Mann-Whitney U | `mannwhitneyu(x, y)` |
| Paired, non-parametric | Wilcoxon signed-rank | `wilcoxon(x, y)` |
| 3+ group means | One-way ANOVA | `f_oneway(*groups)` |
| 3+ groups, non-parametric | Kruskal-Wallis | `kruskal(*groups)` |
| Post-hoc pairwise | Tukey HSD | `tukey_hsd(*groups)` |
| Independence of categoricals | Chi-square | `chi2_contingency(table)` |
| 2x2 small counts | Fisher exact | `fisher_exact(table)` |
| Observed vs expected counts | Chi-square goodness of fit | `chisquare(f_obs, f_exp)` |
| Proportion | Binomial test | `binomtest(k, n, p)` |
| Normality | Shapiro-Wilk, D'Agostino | `shapiro(x)`, `normaltest(x)` |
| Same distribution | Kolmogorov-Smirnov | `ks_2samp(x, y)`, `kstest(x, "norm")` |
| Equal variances | Levene, Bartlett | `levene(*g)`, `bartlett(*g)` |
| Linear correlation | Pearson | `pearsonr(x, y)` |
| Rank correlation | Spearman, Kendall | `spearmanr(x, y)`, `kendalltau(x, y)` |
| Multiple testing | Benjamini-Hochberg | `false_discovery_control(pvalues)` |

Signature of the most-used test:

```python
scipy.stats.ttest_ind(a, b, *, axis=0, equal_var=True, nan_policy='propagate', alternative='two-sided', trim=0, method=None, keepdims=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `equal_var` | bool | `True` | `False` gives Welch's t-test (safer default in practice). |
| `alternative` | {'two-sided','less','greater'} | `'two-sided'` | Alternative hypothesis. |
| `nan_policy` | str | `'propagate'` | NaN handling. |
| `trim` | float | `0` | Trimmed (Yuen's) t-test proportion. |
| `method` | `PermutationMethod`, `MonteCarloMethod` | `None` | Compute p-value by resampling instead of the t distribution. |

Returns: `TtestResult(statistic, pvalue, df)` with a `confidence_interval()` method.

```python
rng = np.random.default_rng(0)
a = rng.normal(10.0, 2.0, 40)
b = rng.normal(11.0, 2.0, 40)

r = stats.ttest_ind(a, b, equal_var=False)
print(f"t={r.statistic:.2f}, p={r.pvalue:.4f}")

stats.mannwhitneyu(a, b, alternative="two-sided")

table = np.array([[30, 10], [20, 40]])          # rows: group, cols: outcome
chi2 = stats.chi2_contingency(table)
print(chi2.statistic, chi2.pvalue, chi2.dof)    # attributes: statistic, pvalue, dof, expected_freq

stats.pearsonr([1, 2, 3, 4], [2, 4, 5, 9])      # statistic=0.9648, pvalue=0.0352
stats.spearmanr([1, 2, 3, 4], [1, 3, 2, 4]).statistic   # 0.8
stats.shapiro(a).pvalue
stats.false_discovery_control([0.01, 0.02, 0.03, 0.5])   # adjusted p-values
```

#### Resampling: bootstrap and permutation_test

```python
scipy.stats.bootstrap(data, statistic, *, n_resamples=9999, batch=None, vectorized=None, paired=False, axis=0, confidence_level=0.95, alternative='two-sided', method='BCa', bootstrap_result=None, rng=None)
scipy.stats.permutation_test(data, statistic, *, permutation_type='independent', vectorized=None, n_resamples=9999, batch=None, alternative='two-sided', axis=0, rng=None)
```

| Parameter | Description |
|-----------|-------------|
| `data` | Tuple of samples, e.g. `(x,)` or `(x, y)` |
| `statistic` | Function of the samples; accept an `axis` argument for speed (vectorized) |
| `method` | `'percentile'`, `'basic'`, `'BCa'` (bias-corrected, default) |
| `permutation_type` | `'independent'`, `'samples'` (paired), `'pairings'` (correlation) |

```python
rng = np.random.default_rng(42)
x = rng.exponential(2.0, 100)

boot = stats.bootstrap((x,), np.median, confidence_level=0.95, rng=rng)
print(boot.confidence_interval)                  # ConfidenceInterval(low=..., high=...)

def diff_means(u, v, axis):
    return np.mean(u, axis=axis) - np.mean(v, axis=axis)

y = rng.exponential(2.5, 100)
perm = stats.permutation_test((x, y), diff_means, n_resamples=5000, rng=rng)
print(perm.statistic, perm.pvalue)
```

#### linregress, gaussian_kde, entropy

```python
scipy.stats.linregress(x, y, alternative='two-sided', *, axis=0, nan_policy='propagate', keepdims=False)
scipy.stats.gaussian_kde(dataset, bw_method=None, weights=None)
scipy.stats.entropy(pk, qk=None, base=None, axis=0, *, nan_policy='propagate', keepdims=False)
```

```python
x = np.arange(10)
y = 2 * x + 1 + np.random.default_rng(0).normal(0, 0.5, 10)
lr = stats.linregress(x, y)
print(round(lr.slope, 2), round(lr.intercept, 2), round(lr.rvalue**2, 3))

kde = stats.gaussian_kde(np.random.default_rng(1).normal(size=500))
grid = np.linspace(-3, 3, 7)
kde(grid)                                        # estimated density at grid points

p = [0.5, 0.5]; q = [0.9, 0.1]
stats.entropy(p, base=2)                         # 1.0 bit
stats.entropy(p, q)                              # KL divergence D(p || q) in nats: 0.5108...
```

#### Quasi-Monte Carlo (stats.qmc)

```python
from scipy.stats import qmc

sampler = qmc.Sobol(d=2, scramble=True, rng=0)
pts = sampler.random_base2(m=6)                  # 64 low-discrepancy points in [0, 1)^2
scaled = qmc.scale(pts, l_bounds=[1e-4, 10], u_bounds=[1e-1, 500])   # e.g. hyperparameter search space
lhs = qmc.LatinHypercube(d=3, rng=0).random(n=10)
```

### scipy.linalg

`scipy.linalg` contains everything in `numpy.linalg` plus more decompositions and matrix functions, and is always compiled against LAPACK.

```python
scipy.linalg.solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=None, transposed=False)
scipy.linalg.lu_factor(a, overwrite_a=False, check_finite=True)
scipy.linalg.cho_factor(a, lower=False, overwrite_a=False, check_finite=True)
scipy.linalg.eigh(a, b=None, *, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, type=1, check_finite=True, subset_by_index=None, subset_by_value=None, driver=None)
scipy.linalg.svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')
scipy.linalg.expm(A)
```

| Parameter | Description |
|-----------|-------------|
| `assume_a` | Matrix structure for `solve`: `'general'`, `'symmetric'`, `'hermitian'`, `'positive definite'`, `'banded'`, `'diagonal'`, `'tridiagonal'`, `'upper triangular'`, `'lower triangular'`. `None` (the current default) detects simple structure automatically. |
| `check_finite` | Set `False` to skip the NaN/inf check for speed (unsafe with bad input). |
| `subset_by_index` | `[lo, hi]` to compute only some eigenvalues with `eigh`. |
| `overwrite_a` | Allow reuse of input memory. |

| Function | Purpose |
|----------|---------|
| `solve`, `solve_triangular`, `solve_banded`, `cho_solve`, `lu_solve` | Linear systems |
| `lu`, `lu_factor`, `cholesky`, `cho_factor`, `qr`, `svd`, `schur`, `polar` | Decompositions |
| `eig`, `eigh`, `eigvalsh` | Eigenproblems (generalized with `b=`) |
| `inv`, `pinv`, `det`, `norm`, `null_space`, `orth` | Matrix utilities |
| `expm`, `logm`, `sqrtm`, `funm` | Matrix functions |
| `toeplitz`, `circulant`, `block_diag`, `hadamard` | Special matrices |
| `solve_sylvester`, `solve_continuous_lyapunov` | Matrix equations (control theory) |

```python
from scipy import linalg

A = np.array([[4.0, 2.0], [2.0, 3.0]])
b = np.array([1.0, 2.0])
linalg.solve(A, b, assume_a="positive definite")   # uses Cholesky

# factor once, solve many times
c, low = linalg.cho_factor(A)
for rhs in (b, 2 * b):
    linalg.cho_solve((c, low), rhs)

lu, piv = linalg.lu_factor(A)
linalg.lu_solve((lu, piv), b)

w = linalg.eigh(A, eigvals_only=True)                # [1.438..., 5.561...]
w_top = linalg.eigh(A, subset_by_index=[1, 1])[0]    # only the largest eigenvalue
linalg.expm(np.zeros((2, 2)))                        # identity
linalg.null_space(np.array([[1.0, 1.0]]))            # basis of the null space, shape (2, 1)
linalg.block_diag(np.eye(2), 3 * np.eye(1))
```

### scipy.sparse and scipy.sparse.linalg

#### Formats

| Format | Class | Best for |
|--------|-------|----------|
| COO | `coo_array` | Building from (row, col, value) triplets; conversion |
| CSR | `csr_array` | Fast row slicing, matrix-vector products; default for ML features |
| CSC | `csc_array` | Fast column slicing, some direct solvers |
| LIL | `lil_array` | Incremental construction by indexing |
| DOK | `dok_array` | Random single-element access/updates |
| DIA | `dia_array` | Banded matrices |
| BSR | `bsr_array` | Block-sparse matrices |

```python
scipy.sparse.csr_array(arg1, shape=None, dtype=None, copy=False, *, maxprint=None)
scipy.sparse.diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=None)
scipy.sparse.eye_array(m, n=None, *, k=0, dtype=float, format=None)
scipy.sparse.random_array(shape, *, density=0.01, format='coo', dtype=None, rng=None, data_sampler=None)
```

`arg1` may be a dense array, another sparse array, a `(data, (row, col))` tuple, or a `(data, indices, indptr)` tuple.

```python
from scipy import sparse

rows = np.array([0, 0, 1, 2])
cols = np.array([0, 2, 1, 2])
vals = np.array([1.0, 2.0, 3.0, 4.0])
A = sparse.coo_array((vals, (rows, cols)), shape=(3, 3)).tocsr()

print(A.nnz, A.shape)                  # 4 (3, 3)
A @ np.ones(3)                         # [3. 3. 4.]
A.T @ A                                # sparse result
A.sum(axis=0)                          # dense 1-D array of column sums
A[[0, 2]]                              # row selection (CSR)
A.toarray()                            # dense ndarray

D = sparse.diags_array([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(5, 5))   # 1-D Laplacian
I = sparse.eye_array(5, format="csr")
R = sparse.random_array((1000, 1000), density=0.001, format="csr", rng=0)
sparse.hstack([A, A], format="csr")
sparse.save_npz("A.npz", A)
B = sparse.load_npz("A.npz")
sparse.issparse(B)                     # True
```

#### Sparse linear algebra

```python
scipy.sparse.linalg.spsolve(A, b, permc_spec=None, use_umfpack=True)
scipy.sparse.linalg.cg(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
scipy.sparse.linalg.eigsh(A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, ...)
scipy.sparse.linalg.svds(A, k=6, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True, solver='arpack', rng=None, options=None)
```

| Function | Purpose |
|----------|---------|
| `spsolve` | Direct solve of sparse `A x = b` |
| `splu`, `factorized` | Reusable sparse LU factorization |
| `cg`, `gmres`, `bicgstab`, `minres`, `lsqr`, `lsmr` | Iterative solvers (`cg` needs symmetric positive definite) |
| `eigsh`, `eigs` | A few eigenvalues of large symmetric / general matrices |
| `svds` | Truncated SVD (top-k singular vectors), e.g. LSA on TF-IDF |
| `LinearOperator` | Matrix-free operator defined by a matvec function |
| `norm`, `expm_multiply` | Norms; action of the matrix exponential |

Note: `cg` and friends use `rtol=` since 1.12 (the old `tol=` keyword was removed in 1.14).

```python
from scipy.sparse import linalg as sla

n = 100
L = sparse.diags_array([-1.0, 2.0, -1.0], offsets=[-1, 0, 1], shape=(n, n), format="csr")
b = np.ones(n)
x = sla.spsolve(L, b)
x_cg, info = sla.cg(L, b, rtol=1e-10)
print(info, np.allclose(x, x_cg, atol=1e-6))   # 0 True

vals = sla.eigsh(L, k=3, which="SA", return_eigenvectors=False)   # 3 smallest eigenvalues
X = sparse.random_array((500, 200), density=0.05, format="csr", rng=0)
U, s, Vt = sla.svds(X, k=10)                    # singular values ascending
```

### scipy.integrate

```python
scipy.integrate.quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False)
scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
scipy.integrate.trapezoid(y, x=None, dx=1.0, axis=-1)
scipy.integrate.simpson(y, x=None, *, dx=1.0, axis=-1)
scipy.integrate.cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None)
```

| Parameter (`solve_ivp`) | Default | Description |
|-------------------------|---------|-------------|
| `fun` | required | `fun(t, y, *args) -> dy/dt` |
| `t_span` | required | `(t0, tf)` |
| `y0` | required | Initial state (1-D array) |
| `method` | `'RK45'` | `'RK23'`, `'DOP853'`, `'Radau'`, `'BDF'` (stiff), `'LSODA'` (auto-switching) |
| `t_eval` | `None` | Times at which to store the solution |
| `events` | `None` | Functions whose zero crossings are detected (can terminate) |
| `rtol`, `atol` | `1e-3`, `1e-6` | Tolerances (passed via `**options`) |

`quad` returns `(value, abs_error_estimate)`. `solve_ivp` returns an object with `t`, `y` (shape `(n_states, n_times)`), `success`, `message`, and `sol` (when `dense_output=True`).

```python
from scipy import integrate

val, err = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
print(round(val, 6), round(np.sqrt(np.pi), 6))   # 1.772454 1.772454

integrate.dblquad(lambda y, x: x * y, 0, 1, 0, 1)  # (0.25, error); note func(y, x) argument order

x = np.linspace(0, np.pi, 101)
integrate.trapezoid(np.sin(x), x)                 # 1.99983...
integrate.simpson(np.sin(x), x=x)                 # 2.00000...

# logistic growth dy/dt = r y (1 - y / K)
def logistic(t, y, r, K):
    return r * y * (1 - y / K)

sol = integrate.solve_ivp(logistic, (0, 10), [0.1], args=(1.0, 5.0), t_eval=np.linspace(0, 10, 6), rtol=1e-8)
print(sol.y[0].round(3))
```

### scipy.interpolate

| Need | Recommended |
|------|-------------|
| 1-D linear | `np.interp` |
| 1-D smooth spline | `make_interp_spline(x, y, k=3)` or `CubicSpline` |
| 1-D monotone, no overshoot | `PchipInterpolator`, `Akima1DInterpolator` |
| Noisy 1-D data (smoothing) | `make_smoothing_spline`, `UnivariateSpline` |
| Regular N-D grid | `RegularGridInterpolator`, `interpn` |
| Scattered N-D points | `RBFInterpolator`, `griddata`, `LinearNDInterpolator` |

```python
scipy.interpolate.make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0, check_finite=True)
scipy.interpolate.CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=None)
scipy.interpolate.PchipInterpolator(x, y, axis=0, extrapolate=None)
scipy.interpolate.RegularGridInterpolator(points, values, method='linear', bounds_error=True, fill_value=nan, *, solver=None, solver_args=None)
scipy.interpolate.RBFInterpolator(y, d, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
scipy.interpolate.griddata(points, values, xi, method='linear', fill_value=nan, rescale=False)
```

```python
from scipy import interpolate

x = np.linspace(0, 10, 11)
y = np.sin(x)
spl = interpolate.make_interp_spline(x, y, k=3)
spl(2.5)                                      # close to sin(2.5) = 0.598
spl.derivative()(2.5)                         # derivative of the spline

cs = interpolate.CubicSpline(x, y, bc_type="natural")
pchip = interpolate.PchipInterpolator(x, y)

# regular 2-D grid
gx, gy = np.linspace(0, 1, 5), np.linspace(0, 2, 6)
values = np.add.outer(gx, gy)                 # f(x, y) = x + y, shape (5, 6)
rgi = interpolate.RegularGridInterpolator((gx, gy), values)
rgi([[0.5, 1.0]])                             # [1.5]

# scattered data
rng = np.random.default_rng(0)
pts = rng.random((50, 2))
vals = pts[:, 0] ** 2 + pts[:, 1]
rbf = interpolate.RBFInterpolator(pts, vals, kernel="thin_plate_spline")
rbf(np.array([[0.5, 0.5]]))                   # approximately 0.75
interpolate.griddata(pts, vals, (0.5, 0.5), method="cubic")
```

### scipy.signal

```python
scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
scipy.signal.sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None)
scipy.signal.find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
scipy.signal.welch(x, fs=1.0, window=..., nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
scipy.signal.fftconvolve(in1, in2, mode='full', axes=None)
```

| Parameter | Description |
|-----------|-------------|
| `N` (butter) | Filter order |
| `Wn` | Cutoff frequency (Hz if `fs` given, else normalized to Nyquist) or `[low, high]` |
| `btype` | `'low'`, `'high'`, `'bandpass'`, `'bandstop'` |
| `output` | `'ba'`, `'zpk'`, or `'sos'` (second-order sections: numerically safest) |
| `height`, `distance`, `prominence`, `width` | Peak selection criteria |
| `nperseg` | Segment length for Welch / spectrogram |
| `window_length`, `polyorder` | Savitzky-Golay smoothing window (odd) and polynomial order |

| Function | Purpose |
|----------|---------|
| `butter`, `cheby1`, `ellip`, `bessel`, `iirnotch`, `firwin` | Filter design |
| `sosfilt`, `sosfiltfilt`, `lfilter`, `filtfilt` | Apply filters (`*filtfilt` = zero phase) |
| `freqz`, `freqz_sos` | Frequency response |
| `welch`, `periodogram`, `spectrogram`, `ShortTimeFFT`, `stft`, `csd`, `coherence` | Spectral analysis |
| `find_peaks`, `peak_widths`, `argrelextrema` | Peak detection |
| `convolve`, `fftconvolve`, `oaconvolve`, `correlate`, `correlation_lags` | Convolution and correlation |
| `savgol_filter`, `medfilt`, `detrend` | Smoothing and trend removal |
| `resample`, `resample_poly`, `decimate` | Change sampling rate |
| `hilbert` | Analytic signal / envelope |
| `get_window`, `signal.windows.*` | Window functions |

```python
from scipy import signal

fs = 500
t = np.arange(0, 2, 1 / fs)
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)

sos = signal.butter(4, 20, btype="low", fs=fs, output="sos")
clean = signal.sosfiltfilt(sos, x)                    # 60 Hz component removed, no phase shift

f, Pxx = signal.welch(x, fs=fs, nperseg=1000)        # 0.5 Hz frequency resolution
pk, _ = signal.find_peaks(Pxx, height=0.01 * Pxx.max())
print(f[pk])                                         # [ 5. 60.]

peaks, props = signal.find_peaks(clean, height=0.5, distance=fs // 10)
print(len(peaks))                                    # 10 peaks: 5 Hz for 2 s

smooth = signal.savgol_filter(x, window_length=31, polyorder=3)
lag_corr = signal.correlate(x, x, mode="full")
```

### scipy.fft

```python
scipy.fft.fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None)
scipy.fft.rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None)
scipy.fft.dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None)
scipy.fft.next_fast_len(target, real=False)
```

```python
from scipy import fft

x = np.random.default_rng(0).random(1000)
X = fft.rfft(x, workers=-1)                 # multithreaded
freqs = fft.rfftfreq(1000, d=0.01)
n = fft.next_fast_len(1000)                 # efficient padded length
c = fft.dct(x, norm="ortho")                # used for compression / MFCCs
x_back = fft.idct(c, norm="ortho")
np.allclose(x, x_back)                      # True
```

### scipy.spatial

```python
scipy.spatial.KDTree(data, leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
KDTree.query(x, k=1, eps=0.0, p=2.0, distance_upper_bound=inf, workers=1)
KDTree.query_ball_point(x, r, p=2.0, eps=0.0, workers=1, return_sorted=None, return_length=False)
scipy.spatial.distance.cdist(XA, XB, metric='euclidean', *, out=None, **kwargs)
scipy.spatial.distance.pdist(X, metric='euclidean', *, out=None, **kwargs)
scipy.spatial.distance.squareform(X, force='no', checks=True)
```

| Metric names for `cdist`/`pdist` |
|----------------------------------|
| `'euclidean'`, `'sqeuclidean'`, `'cityblock'`, `'chebyshev'`, `'minkowski'` (`p=`), `'cosine'`, `'correlation'`, `'mahalanobis'` (`VI=`), `'seuclidean'`, `'hamming'`, `'jaccard'`, `'braycurtis'`, `'canberra'`, or a callable |

```python
from scipy.spatial import KDTree, ConvexHull, distance

rng = np.random.default_rng(0)
points = rng.random((1000, 3))
tree = KDTree(points)
dist, idx = tree.query([[0.5, 0.5, 0.5]], k=5)        # 5 nearest neighbors
nearby = tree.query_ball_point([0.5, 0.5, 0.5], r=0.1)
pairs = tree.query_pairs(r=0.02)                       # all close pairs

A = rng.random((4, 3)); B = rng.random((6, 3))
D = distance.cdist(A, B, metric="cosine")              # (4, 6)
condensed = distance.pdist(A)                          # (6,) upper-triangle distances
square = distance.squareform(condensed)                # (4, 4)
distance.cosine([1, 0], [0, 1])                        # 1.0

hull = ConvexHull(rng.random((30, 2)))
print(hull.volume)                                     # area in 2-D

from scipy.spatial.transform import Rotation
r = Rotation.from_euler("z", 90, degrees=True)
r.apply([1, 0, 0]).round(6)                            # [0. 1. 0.]
```

### scipy.special

| Function | Description |
|----------|-------------|
| `expit(x)`, `logit(p)`, `log_expit(x)` | Sigmoid, its inverse, and numerically stable log-sigmoid |
| `softmax(x, axis=None)`, `log_softmax(x, axis=None)` | Stable softmax |
| `logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False)` | Stable `log(sum(exp(a)))` |
| `xlogy(x, y)`, `xlog1py(x, y)` | `x * log(y)` with `0 * log(0) = 0` (cross-entropy) |
| `rel_entr(p, q)`, `kl_div(p, q)`, `entr(p)` | Elementwise KL / entropy terms |
| `gamma`, `gammaln`, `digamma`, `beta`, `betaln` | Gamma-family functions |
| `erf`, `erfc`, `erfinv`, `ndtr`, `ndtri` | Error function; normal CDF and its inverse |
| `comb(N, k, exact=False)`, `perm`, `factorial`, `binom` | Combinatorics |
| `jv`, `yv`, `iv`, `kv` | Bessel functions |
| `zeta`, `lambertw`, `sph_harm_y` | Other special functions |
| `boxcox`, `inv_boxcox` | Box-Cox transform |

```python
from scipy import special

special.expit([-1000, 0, 1000])                 # [0.  0.5 1. ] no overflow
special.logsumexp([1000, 1000])                 # 1000.6931... (naive form overflows)
special.softmax(np.array([[1.0, 2.0, 3.0]]), axis=1).round(3)   # [[0.09  0.245 0.665]]
special.comb(10, 3, exact=True)                 # 120
special.gammaln(100)                            # log(99!) = 359.134...
y_true = np.array([1, 0]); p = np.array([0.9, 0.2])
-np.mean(special.xlogy(y_true, p) + special.xlogy(1 - y_true, 1 - p))   # binary cross-entropy 0.164...
special.ndtri(0.975)                            # 1.959963...
```

### scipy.cluster

```python
scipy.cluster.hierarchy.linkage(y, method='single', metric='euclidean', optimal_ordering=False)
scipy.cluster.hierarchy.fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)
scipy.cluster.hierarchy.dendrogram(Z, p=30, truncate_mode=None, ...)
scipy.cluster.vq.kmeans2(data, k, iter=10, thresh=1e-05, minit='random', missing='warn', check_finite=True, *, rng=None)
```

| Parameter | Description |
|-----------|-------------|
| `method` (linkage) | `'single'`, `'complete'`, `'average'`, `'weighted'`, `'centroid'`, `'median'`, `'ward'` |
| `criterion` (fcluster) | `'maxclust'` (t = number of clusters), `'distance'` (t = cut height), `'inconsistent'` |
| `minit` (kmeans2) | `'random'`, `'points'`, `'++'` (k-means++), `'matrix'` |

```python
from scipy.cluster import hierarchy, vq

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(3, 0.3, (20, 2))])
Z = hierarchy.linkage(X, method="ward")
labels = hierarchy.fcluster(Z, t=2, criterion="maxclust")
print(np.bincount(labels))                      # [ 0 20 20]  (labels start at 1)

centroids, lab = vq.kmeans2(X, 2, minit="++", rng=0)
# hierarchy.dendrogram(Z) draws with matplotlib
```

### scipy.ndimage

```python
from scipy import ndimage

img = np.zeros((20, 20)); img[5:9, 5:9] = 1; img[12:16, 10:18] = 1
blurred = ndimage.gaussian_filter(img, sigma=1.0)
labels, n = ndimage.label(img)                  # connected components
print(n)                                        # 2
ndimage.center_of_mass(img, labels, [1, 2])     # centroids of each object
ndimage.sum_labels(img, labels, [1, 2])         # areas: [16. 32.]
ndimage.zoom(img, 0.5, order=1).shape           # (10, 10)
ndimage.rotate(img, 45, reshape=False)
ndimage.median_filter(img, size=3)
ndimage.binary_dilation(img > 0, iterations=1)
ndimage.distance_transform_edt(img == 0)
```

### scipy.io

```python
from scipy import io
from scipy.io import wavfile

io.savemat("data.mat", {"X": np.eye(3), "label": "demo"})
m = io.loadmat("data.mat")
print(m["X"].shape)                             # (3, 3); also keys __header__, __version__, __globals__

rate = 16000
tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
wavfile.write("tone.wav", rate, tone)
sr, audio = wavfile.read("tone.wav")
print(sr, audio.dtype, audio.shape)             # 16000 float32 (16000,)

io.mmwrite("A.mtx", np.eye(3))                  # Matrix Market format
```

`loadmat` does not read MATLAB v7.3 files (which are HDF5); use `h5py` for those.

### scipy.differentiate (1.15+)

```python
from scipy import differentiate

res = differentiate.derivative(np.exp, 1.0)
print(res.df, res.error)                        # 2.718281828... with error estimate

def f(xy):
    x, y = xy
    return x**2 * y

J = differentiate.jacobian(f, np.array([1.0, 2.0])).df     # gradient [4. 1.]
```

### scipy.constants

```python
from scipy import constants

constants.c                     # speed of light, 299792458.0 m/s
constants.k                     # Boltzmann constant
constants.physical_constants["electron mass"]   # (value, unit, uncertainty)
constants.convert_temperature(100, "Celsius", "Kelvin")   # 373.15
```

## Tutorials

### Tutorial 1: Analyzing an A/B test

An online store tests a new checkout page. We compare conversion rates (a proportion) and order values (a continuous, skewed metric).

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(2024)
n_a, n_b = 5000, 5000
conv_a = rng.random(n_a) < 0.100           # control converts at 10.0 %
conv_b = rng.random(n_b) < 0.115           # variant converts at 11.5 %
value_a = rng.lognormal(mean=3.5, sigma=0.6, size=conv_a.sum())
value_b = rng.lognormal(mean=3.55, sigma=0.6, size=conv_b.sum())

# 1. Conversion: chi-square test on the 2x2 contingency table
table = np.array([[conv_a.sum(), n_a - conv_a.sum()],
                  [conv_b.sum(), n_b - conv_b.sum()]])
chi = stats.chi2_contingency(table)
print(f"conversion A={conv_a.mean():.3%} B={conv_b.mean():.3%} p={chi.pvalue:.4f}")   # A=9.760% B=11.680% p=0.0021

# 2. Order value: skewed data, so use Welch's t-test on log values and Mann-Whitney U
t = stats.ttest_ind(np.log(value_b), np.log(value_a), equal_var=False)
u = stats.mannwhitneyu(value_b, value_a, alternative="two-sided")
print(f"log-value Welch t p={t.pvalue:.4f}, Mann-Whitney p={u.pvalue:.4f}")

# 3. Bootstrap confidence interval for the difference in median order value
def median_diff(b, a, axis):
    return np.median(b, axis=axis) - np.median(a, axis=axis)

boot = stats.bootstrap((value_b, value_a), median_diff, n_resamples=5000, method="percentile", rng=rng)
ci = boot.confidence_interval
print(f"median uplift 95% CI: [{ci.low:.2f}, {ci.high:.2f}]")

# 4. Power check: z-test style sample size for detecting 10% -> 11.5%
p1, p2, alpha, power = 0.10, 0.115, 0.05, 0.8
z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
p_bar = (p1 + p2) / 2
n_needed = ((z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
print(f"required n per group: {int(np.ceil(n_needed))}")   # about 6,700
```

Interpretation: a p-value below your pre-registered alpha (for example 0.05) suggests a real difference; the bootstrap interval tells you how large the effect plausibly is. The power calculation shows that 5,000 users per arm is somewhat underpowered for a 1.5-point lift.

### Tutorial 2: Fitting a nonlinear model with uncertainty

Fit a Michaelis-Menten (saturation) curve, a common shape for dose-response and learning curves, and report parameter confidence intervals.

```python
import numpy as np
from scipy import optimize, stats

def mm(s, vmax, km):
    return vmax * s / (km + s)

rng = np.random.default_rng(7)
s = np.array([0.5, 1, 2, 4, 8, 16, 32, 64], dtype=float)
v = mm(s, vmax=10.0, km=4.0) * (1 + rng.normal(0, 0.05, s.size))

# 1. fit with sensible starting values and positivity bounds
popt, pcov = optimize.curve_fit(mm, s, v, p0=[v.max(), np.median(s)], bounds=(0, np.inf))
perr = np.sqrt(np.diag(pcov))

# 2. 95% confidence intervals using the t distribution
dof = len(s) - len(popt)
tval = stats.t.ppf(0.975, dof)
for name, val, se in zip(["vmax", "km"], popt, perr):
    print(f"{name} = {val:.2f}  95% CI [{val - tval * se:.2f}, {val + tval * se:.2f}]")
# vmax = 10.35  95% CI [9.75, 10.95]
# km = 4.64  95% CI [3.69, 5.60]

# 3. goodness of fit
resid = v - mm(s, *popt)
r2 = 1 - np.sum(resid**2) / np.sum((v - v.mean()) ** 2)
print(f"R^2 = {r2:.4f}")

# 4. use the model: substrate level giving 90% of vmax
s90 = optimize.brentq(lambda x: mm(x, *popt) - 0.9 * popt[0], 1e-6, 1e6)
print(f"S at 90% vmax: {s90:.1f}")      # 41.8, analytically 9 * km
```

### Tutorial 3: Cleaning and analyzing a noisy sensor signal

```python
import numpy as np
from scipy import signal

fs = 1000                                   # Hz
t = np.arange(0, 5, 1 / fs)
rng = np.random.default_rng(0)
heartbeat = np.sin(2 * np.pi * 1.2 * t) ** 63          # sharp periodic pulses (72 bpm)
drift = 0.5 * t / t.max()                                # slow baseline drift
mains = 0.3 * np.sin(2 * np.pi * 50 * t)                 # 50 Hz power-line noise
x = heartbeat + drift + mains + 0.05 * rng.normal(size=t.size)

# 1. remove drift with a high-pass filter and mains noise with a notch filter
sos_hp = signal.butter(2, 0.5, btype="highpass", fs=fs, output="sos")
b_notch, a_notch = signal.iirnotch(w0=50, Q=30, fs=fs)
y = signal.sosfiltfilt(sos_hp, x)
y = signal.filtfilt(b_notch, a_notch, y)

# 2. smooth while preserving peak shape
y = signal.savgol_filter(y, window_length=21, polyorder=3)

# 3. detect beats: at least 0.3 s apart, clearly prominent
peaks, props = signal.find_peaks(y, distance=int(0.3 * fs), prominence=0.5)
rr = np.diff(peaks) / fs                                  # seconds between beats
print(f"beats: {len(peaks)}, heart rate: {60 / rr.mean():.1f} bpm")   # about 72 bpm

# 4. confirm noise removal in the spectrum
f, p_raw = signal.welch(x, fs=fs, nperseg=2048)
f, p_clean = signal.welch(y, fs=fs, nperseg=2048)
i50 = np.argmin(np.abs(f - 50))
print(f"50 Hz power reduced by a factor of {p_raw[i50] / p_clean[i50]:.0f}")
```

`sosfiltfilt` / `filtfilt` run the filter forward and backward, so peaks are not shifted in time, which matters when you measure intervals.

### Tutorial 4: Sparse TF-IDF document similarity

Text features are sparse: most words do not appear in most documents. This tutorial builds a TF-IDF matrix by hand with `scipy.sparse`, then finds similar documents and reduces dimensionality with truncated SVD.

```python
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets",
    "stock markets fell sharply today",
    "markets rally as stocks rise",
]
vocab = sorted({w for d in docs for w in d.split()})
index = {w: i for i, w in enumerate(vocab)}

# 1. term counts as a COO array, converted to CSR
rows, cols, vals = [], [], []
for r, d in enumerate(docs):
    words, counts = np.unique(d.split(), return_counts=True)
    rows += [r] * len(words)
    cols += [index[w] for w in words]
    vals += counts.tolist()
tf = sparse.coo_array((vals, (rows, cols)), shape=(len(docs), len(vocab)), dtype=float).tocsr()

# 2. inverse document frequency and L2 row normalization
df = np.asarray((tf > 0).sum(axis=0)).ravel()
idf = np.log((1 + len(docs)) / (1 + df)) + 1
X = tf @ sparse.diags_array(idf)
norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
X = sparse.diags_array(1 / norms) @ X
print(X.shape, f"density={X.nnz / np.prod(X.shape):.2f}")

# 3. cosine similarity = dot product of normalized rows (stays sparse)
S = (X @ X.T).toarray()
np.fill_diagonal(S, 0)
for i, d in enumerate(docs):
    print(f"{d!r:40} -> most similar: {docs[S[i].argmax()]!r}")

# 4. latent semantic analysis: 2 components via truncated SVD
U, s, Vt = sla.svds(X, k=2)
emb = U * s                                   # 2-D document embeddings
print(emb.round(2))
```

Expected output: the "cat"/"dog" sentences match each other and the two finance sentences match each other. "cats and dogs are pets" shares no exact tokens with any other document (no stemming: "cats" is not "cat"), so all its similarities are 0 and its "most similar" result is arbitrary, a classic bag-of-words limitation. The SVD embedding places the two topic clusters on separate axes. With scikit-learn, `TfidfVectorizer` produces the same kind of `scipy.sparse` CSR matrix, and `TruncatedSVD` wraps the same idea.

### Tutorial 5: Solving and fitting an ODE model (SIR epidemic)

```python
import numpy as np
from scipy import integrate, optimize

def sir(t, y, beta, gamma):
    s, i, r = y
    return [-beta * s * i, beta * s * i - gamma * i, gamma * i]

t_obs = np.arange(0, 60, 3)
true = integrate.solve_ivp(sir, (0, 60), [0.99, 0.01, 0.0], args=(0.4, 0.1), t_eval=t_obs)
rng = np.random.default_rng(1)
infected_obs = true.y[1] * (1 + rng.normal(0, 0.05, t_obs.size))

def residuals(params):
    beta, gamma = params
    sol = integrate.solve_ivp(sir, (0, 60), [0.99, 0.01, 0.0], args=(beta, gamma), t_eval=t_obs, rtol=1e-6)
    return sol.y[1] - infected_obs

fit = optimize.least_squares(residuals, x0=[0.2, 0.05], bounds=([0, 0], [2, 1]))
beta, gamma = fit.x
print(f"beta={beta:.3f} gamma={gamma:.3f} R0={beta / gamma:.2f}")   # close to 0.4, 0.1, R0 = 4

peak = integrate.solve_ivp(sir, (0, 60), [0.99, 0.01, 0.0], args=(beta, gamma), dense_output=True)
tt = np.linspace(0, 60, 601)
print(f"infection peak on day {tt[peak.sol(tt)[1].argmax()]:.1f}")
```

## Performance & Best Practices

1. **Use the modern APIs**: `minimize` over `fmin_*`, `solve_ivp` over `odeint`, `make_interp_spline`/`CubicSpline` over `interp1d`, sparse `*_array` over `*_matrix`, `rng=` over `random_state=`.
2. **Supply gradients** (`jac=`) to `minimize` and `least_squares` when you can; finite differences cost `n` extra function evaluations per step and lose accuracy.
3. **Scale your problem.** Optimizers and fitters behave best when parameters and residuals are of order 1. Rescale inputs or use `x_scale="jac"` in `least_squares`.
4. **Good starting points matter** for `curve_fit`, `minimize`, and `root`. For multimodal problems use `differential_evolution`, `dual_annealing`, `shgo`, or multiple restarts.
5. **Vectorize callbacks.** `quad`, `minimize`, and friends call your function many times; keep it NumPy-vectorized and avoid Python loops inside. For `stats.bootstrap`/`permutation_test`, write statistics that accept `axis=` so they run vectorized.
6. **Factor once, solve many.** Use `linalg.lu_factor`/`cho_factor` or `sparse.linalg.splu`/`factorized` when solving with the same matrix repeatedly.
7. **Tell solvers about structure**: `assume_a="positive definite"` in `linalg.solve`, `eigh` with `subset_by_index` for a few eigenvalues, `solve_banded` for banded systems.
8. **Keep sparse data sparse.** Avoid `.toarray()` on large matrices; use CSR for row operations and arithmetic, CSC for column slicing, COO/LIL for construction. Building a CSR matrix element by element is slow; collect triplets and convert once.
9. **Use stiff solvers** (`method="BDF"`, `"Radau"`, or `"LSODA"`) when `RK45` takes tiny steps; supply a Jacobian for large stiff systems.
10. **Parallelize where offered**: `workers=` in `scipy.fft`, `KDTree.query`, `differential_evolution`, `least_squares`; control BLAS threads with `threadpoolctl`.
11. **Use `check_finite=False`** in `scipy.linalg` only when you are sure the input has no NaN/inf; it saves a full pass over the data.
12. **Use stable special functions**: `logsumexp`, `expit`, `log_expit`, `xlogy`, `gammaln` instead of naive formulas.

```python
import numpy as np
from scipy import optimize

def f(x):
    return np.sum((x - np.arange(x.size)) ** 2)

def grad(x):
    return 2 * (x - np.arange(x.size))

x0 = np.zeros(200)
r_fd = optimize.minimize(f, x0, method="BFGS")              # finite-difference gradient
r_an = optimize.minimize(f, x0, method="BFGS", jac=grad)    # analytic gradient
print(r_fd.nfev, r_an.nfev)    # analytic gradient needs far fewer function evaluations
```

## Common Errors & Troubleshooting

| Error / warning | Cause | Fix |
|-----------------|-------|-----|
| `RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800.` (`curve_fit`) | Bad starting values or poorly scaled model. | Provide `p0`, add `bounds`, rescale data, or increase `maxfev=`. |
| `OptimizeWarning: Covariance of the parameters could not be estimated` | Parameters not identifiable (degenerate Jacobian), or too few points. | Simplify the model, fix redundant parameters, check data. |
| `res.success == False`, message `ABNORMAL_TERMINATION_IN_LNSRCH` / `Desired error not necessarily achieved due to precision loss.` | Inaccurate gradient or badly scaled problem. | Check `jac` with `optimize.check_grad`, rescale, try another `method`. |
| `ValueError: f(a) and f(b) must have different signs` | Root not bracketed in `brentq`/`bisect`. | Choose an interval where the function changes sign, or use `root_scalar` with `x0`. |
| `IntegrationWarning: The maximum number of subdivisions (50) has been achieved.` | Oscillatory or singular integrand in `quad`. | Increase `limit=`, split the interval, pass `points=` for singularities, or use `weight=`. |
| `solve_ivp` returns `success=False`, message `Required step size is less than spacing between numbers.` | Stiff system or blow-up. | Use `method="BDF"`/`"Radau"`/`"LSODA"`, check the model for singularities. |
| `numpy.linalg.LinAlgError: Singular matrix` / `LinAlgError: Matrix is singular.` | Non-invertible system. | Use `lstsq`, `pinv`, or regularize. |
| `LinAlgWarning: Ill-conditioned matrix (rcond=1.2e-17): result may not be accurate.` | Nearly singular matrix. | Regularize, rescale columns, or use `lstsq`. |
| `MatrixRankWarning: Matrix is exactly singular` (`spsolve`) | Singular sparse matrix. | Check for empty rows/columns; add regularization. |
| `SparseEfficiencyWarning: Changing the sparsity structure of a csr_array is expensive. lil and dok are more efficient.` | Assigning new nonzeros into CSR/CSC. | Build with `lil_array`/`coo_array`, then `.tocsr()`. |
| `TypeError: cg() got an unexpected keyword argument 'tol'` | `tol` renamed to `rtol` (removed in 1.14). | Use `rtol=`. |
| `ImportError: cannot import name 'simps' from 'scipy.integrate'` | Removed in 1.14. | `from scipy.integrate import simpson` (same for `trapz` to `trapezoid`, `cumtrapz` to `cumulative_trapezoid`). |
| `NotImplementedError: interp2d has been removed in SciPy 1.14.0.` | Legacy API. | `RegularGridInterpolator` or `RectBivariateSpline`. |
| `ImportError: cannot import name 'binom_test' from 'scipy.stats'` | Removed. | `stats.binomtest(k, n, p).pvalue`. |
| `ValueError: array must not contain infs or NaNs` | `check_finite=True` found bad values. | Clean data (`np.isfinite`), or use `nan_policy="omit"` in `scipy.stats`. |
| `ValueError: x must be strictly increasing sequence.` (`CubicSpline`, `make_interp_spline`) | Unsorted or duplicated x. | `order = np.argsort(x)`; deduplicate (`np.unique`). |
| `ValueError: The requested sample points xi have dimension 1 but this RegularGridInterpolator has dimension 2` | Query points shaped wrongly. | Pass an array of shape `(n_points, n_dims)`. |
| `ValueError: Digital filter critical frequencies must be 0 < Wn < 1` | Cutoff given in Hz without `fs=`. | Pass `fs=sample_rate` to `butter` etc. |
| `ValueError: The length of the input vector x must be greater than padlen` (`filtfilt`) | Signal too short for the filter. | Use a lower order, `padlen=`, or a longer signal. |
| `ModuleNotFoundError: No module named 'pooch'` (`scipy.datasets`) | Optional dependency. | `pip install pooch`. |

## Interoperability

| Library | How it connects to SciPy |
|---------|--------------------------|
| NumPy | All inputs and outputs are ndarrays; SciPy functions accept array-likes and Python lists. |
| scikit-learn | Accepts `scipy.sparse` inputs everywhere sparse makes sense; `TfidfVectorizer`/`OneHotEncoder` return SciPy sparse matrices; uses `scipy.optimize`, `scipy.linalg`, `scipy.sparse.linalg`, and `scipy.stats` (e.g. `loguniform` for `RandomizedSearchCV`). |
| pandas | Pass columns (`df["x"]`) directly; `pd.DataFrame.sparse.from_spmatrix(m)` wraps a sparse matrix; `df.sparse.to_coo()` goes back. |
| statsmodels | Builds on `scipy.stats` and `scipy.optimize`. Use statsmodels for regression summaries, SciPy for individual tests. |
| PyTorch / JAX | Convert with `.numpy()` / `np.asarray`; `torch.sparse_coo_tensor` from COO `row`, `col`, `data`. JAX has `jax.scipy` mirroring parts of the API. Selected SciPy functions accept PyTorch/JAX/CuPy arrays with `SCIPY_ARRAY_API=1`. |
| CuPy | `cupyx.scipy` provides GPU versions of `sparse`, `ndimage`, `signal`, `special`, `fft`, `linalg`, `stats` subsets. |
| Matplotlib | `hierarchy.dendrogram`, plotting `signal.welch` spectra, distributions via `dist.pdf(grid)`. |
| scikit-image / OpenCV | Use `scipy.ndimage` for N-D filters; scikit-image builds on it. |
| NetworkX | `nx.to_scipy_sparse_array(G)` and `nx.from_scipy_sparse_array(A)`; `scipy.sparse.csgraph` for shortest paths, connected components, minimum spanning trees. |

```python
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression

# scikit-learn consumes SciPy sparse matrices directly
X = sparse.random_array((200, 1000), density=0.01, format="csr", rng=0)
y = np.random.default_rng(0).integers(0, 2, 200)
LogisticRegression(max_iter=1000).fit(X, y)

# SciPy distributions as hyperparameter search spaces in scikit-learn
from sklearn.model_selection import RandomizedSearchCV
search_space = {"C": stats.loguniform(1e-3, 1e2)}

# pandas columns go straight into scipy.stats
df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "v": [1, 2, 3, 4, 5, 3, 4, 5, 6, 7]})
stats.ttest_ind(df.loc[df.g == "a", "v"], df.loc[df.g == "b", "v"])

# graph algorithms on a sparse adjacency matrix
from scipy.sparse.csgraph import connected_components, shortest_path
adj = sparse.csr_array(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
n_comp, labels = connected_components(adj, directed=False)   # 2 components
dist = shortest_path(adj, directed=False)
```

## Cheat Sheet

### Optimization

| Task | Code |
|------|------|
| Minimize a function | `optimize.minimize(f, x0)` |
| Minimize with bounds | `optimize.minimize(f, x0, bounds=[(0, 1)] * n, method="L-BFGS-B")` |
| Constrained | `optimize.minimize(f, x0, constraints=[{"type": "ineq", "fun": g}], method="SLSQP")` |
| 1-D minimum | `optimize.minimize_scalar(f, bounds=(a, b), method="bounded")` |
| Fit a curve | `popt, pcov = optimize.curve_fit(model, x, y, p0=...)` |
| Robust least squares | `optimize.least_squares(resid, x0, loss="huber")` |
| Scalar root | `optimize.brentq(f, a, b)` |
| System of equations | `optimize.root(F, x0)` |
| Linear program | `optimize.linprog(c, A_ub=A, b_ub=b)` |
| Integer program | `optimize.milp(c, constraints=..., integrality=...)` |
| Global optimum | `optimize.differential_evolution(f, bounds)` |
| Assignment / matching | `optimize.linear_sum_assignment(cost)` |

### Statistics

| Task | Code |
|------|------|
| Normal CDF / quantile | `stats.norm.cdf(x)`, `stats.norm.ppf(0.975)` |
| Sample from a distribution | `stats.gamma(a=2).rvs(100, random_state=0)` |
| Fit a distribution | `stats.lognorm.fit(data)` |
| Two-sample t-test (Welch) | `stats.ttest_ind(a, b, equal_var=False)` |
| Non-parametric two-sample | `stats.mannwhitneyu(a, b)` |
| Paired test | `stats.ttest_rel(a, b)`, `stats.wilcoxon(a, b)` |
| ANOVA | `stats.f_oneway(g1, g2, g3)` |
| Chi-square independence | `stats.chi2_contingency(table)` |
| Correlation with p-value | `stats.pearsonr(x, y)`, `stats.spearmanr(x, y)` |
| Normality | `stats.shapiro(x)` |
| Bootstrap CI | `stats.bootstrap((x,), np.mean)` |
| FDR correction | `stats.false_discovery_control(pvals)` |
| Z-scores | `stats.zscore(x)` |
| Simple regression | `stats.linregress(x, y)` |
| Density estimate | `stats.gaussian_kde(x)(grid)` |

### Linear algebra and sparse

| Task | Code |
|------|------|
| Solve SPD system | `linalg.solve(A, b, assume_a="positive definite")` |
| Reuse factorization | `c = linalg.cho_factor(A); linalg.cho_solve(c, b)` |
| Few eigenvalues | `linalg.eigh(A, subset_by_index=[n - 3, n - 1])` |
| Matrix exponential | `linalg.expm(A)` |
| Sparse from triplets | `sparse.coo_array((v, (i, j)), shape=s).tocsr()` |
| Sparse identity / diagonal | `sparse.eye_array(n)`, `sparse.diags_array(d)` |
| Sparse solve | `sparse.linalg.spsolve(A, b)` |
| Iterative solve | `sparse.linalg.cg(A, b, rtol=1e-8)` |
| Truncated SVD | `sparse.linalg.svds(X, k=50)` |
| Save / load sparse | `sparse.save_npz("m.npz", A)`, `sparse.load_npz("m.npz")` |

### Integration, interpolation, signal, spatial

| Task | Code |
|------|------|
| Definite integral | `integrate.quad(f, 0, np.inf)` |
| Integrate samples | `integrate.simpson(y, x=x)` |
| Solve ODE | `integrate.solve_ivp(f, (t0, t1), y0, t_eval=ts)` |
| Smooth 1-D interpolant | `interpolate.make_interp_spline(x, y)` |
| Monotone interpolant | `interpolate.PchipInterpolator(x, y)` |
| Scattered N-D | `interpolate.RBFInterpolator(pts, vals)` |
| Low-pass filter | `signal.sosfiltfilt(signal.butter(4, fc, fs=fs, output="sos"), x)` |
| Power spectrum | `f, P = signal.welch(x, fs=fs)` |
| Peaks | `signal.find_peaks(x, prominence=1)` |
| Smooth | `signal.savgol_filter(x, 11, 3)` |
| Nearest neighbors | `KDTree(X).query(q, k=5)` |
| Pairwise distances | `distance.cdist(A, B, "cosine")` |
| Hierarchical clusters | `hierarchy.fcluster(hierarchy.linkage(X, "ward"), 3, "maxclust")` |
| Stable log-sum-exp | `special.logsumexp(a)` |
| Sigmoid | `special.expit(z)` |

## Further Resources

- Official documentation: https://docs.scipy.org/doc/scipy/
- User guide: https://docs.scipy.org/doc/scipy/tutorial/index.html
- API reference: https://docs.scipy.org/doc/scipy/reference/index.html
- Release notes: https://docs.scipy.org/doc/scipy/release.html
- `scipy.stats` tutorial: https://docs.scipy.org/doc/scipy/tutorial/stats.html
- Optimization tutorial: https://docs.scipy.org/doc/scipy/tutorial/optimize.html
- Sparse arrays migration guide: https://docs.scipy.org/doc/scipy/reference/sparse.migration_to_sparray.html
- GitHub: https://github.com/scipy/scipy
- Project site: https://scipy.org/
- Paper: Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python", Nature Methods 17, 2020: https://www.nature.com/articles/s41592-019-0686-2
- Scientific Python Lectures (free): https://lectures.scientific-python.org/
- Book: Jake VanderPlas, *Python Data Science Handbook* (free online): https://jakevdp.github.io/PythonDataScienceHandbook/
- Book: Allen B. Downey, *Think Stats* (free online): https://allendowney.github.io/ThinkStats/
- CuPy SciPy-compatible API (GPU): https://docs.cupy.dev/en/stable/reference/scipy.html
