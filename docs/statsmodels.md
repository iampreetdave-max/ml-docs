# statsmodels

> Statistical modeling, hypothesis testing, and econometrics in Python, with R-style formulas and full inferential output.

statsmodels provides classes and functions for estimating many different statistical models, conducting statistical tests, and exploring data. Where scikit-learn focuses on prediction, statsmodels focuses on **inference**: coefficient standard errors, p-values, confidence intervals, likelihood-ratio tests, diagnostic tests, and model summaries that look like the output of R or Stata. It also contains the most complete classical time-series toolkit in Python (ARIMA/SARIMAX, exponential smoothing, VAR, state space models, unit-root tests).

This page covers **statsmodels 0.14.x** (the current stable series at the time of writing). Removed and renamed APIs from earlier releases (notably the old `arima_model` module) are flagged where relevant.

## Overview

### What it is

statsmodels is organized around **model classes** that take data at construction time and a `fit()` method that returns a **results object**. Major areas:

| Area | Module / entry point | Examples |
|---|---|---|
| Linear regression | `statsmodels.regression` | `OLS`, `WLS`, `GLS`, `QuantReg`, `RecursiveLS`, `RollingOLS` |
| Generalized linear models | `statsmodels.genmod` | `GLM` with Binomial, Poisson, Gamma, Tweedie, NegativeBinomial families; `GEE` |
| Discrete choice / count | `statsmodels.discrete` | `Logit`, `Probit`, `MNLogit`, `Poisson`, `NegativeBinomial`, zero-inflated models |
| Robust regression | `statsmodels.robust` | `RLM` with Huber, Tukey biweight norms |
| Mixed effects | `statsmodels.regression.mixed_linear_model` | `MixedLM` |
| Time series | `statsmodels.tsa` | `ARIMA`, `SARIMAX`, `AutoReg`, `ExponentialSmoothing`, `ETSModel`, `STL`, `VAR`, `UnobservedComponents` |
| Statistical tests | `statsmodels.stats` | t-tests, ANOVA, multiple comparisons, power analysis, proportions, diagnostics |
| Nonparametric | `statsmodels.nonparametric` | `lowess`, `KDEUnivariate`, `KDEMultivariate` |
| Multivariate | `statsmodels.multivariate` | `PCA`, `Factor`, `MANOVA`, `CanCorr` |
| Graphics | `statsmodels.graphics` | Q-Q plots, ACF/PACF plots, regression diagnostics, influence plots |

Two API entry points:

- `import statsmodels.api as sm` - array-based interface (`sm.OLS(y, X)`).
- `import statsmodels.formula.api as smf` - R-style formula interface using patsy (`smf.ols("y ~ x1 + C(group)", data=df)`).

### History and maintainers

statsmodels began as the `models` module in SciPy written by Jonathan Taylor. It was split out and developed as a Google Summer of Code project in 2009 by Skipper Seabold, with Josef Perktold as the long-time lead developer and Kevin Sheppard as a major contributor (time series, state space). It is a NumFOCUS-affiliated project developed at `github.com/statsmodels/statsmodels` and is BSD licensed.

### When to use it

- You need **inference**: which features matter, how much, and with what uncertainty (p-values, confidence intervals).
- You are doing **econometrics** or applied statistics (robust/HAC standard errors, instrumental variables, panel-like clustered errors).
- **Classical forecasting**: ARIMA, SARIMAX with exogenous regressors, exponential smoothing, VAR.
- **Hypothesis testing and experiment analysis**: A/B tests, ANOVA, multiple-comparison correction, power and sample size.
- **Model diagnostics**: heteroskedasticity, autocorrelation, normality, multicollinearity, influential points.

### When not to use it

- **Pure predictive performance** on large tabular data: scikit-learn, XGBoost, LightGBM.
- **Very large data** (tens of millions of rows) or **high-dimensional** sparse features: statsmodels builds dense design matrices and computes full covariance matrices.
- **Deep learning or non-linear representation learning**: PyTorch, TensorFlow.
- **Bayesian modeling**: PyMC, Bambi, or Stan.
- **Automated large-scale forecasting** across thousands of series: statsforecast, Prophet, sktime, Darts (several of these wrap or reimplement statsmodels models).
- **Panel data econometrics and IV** beyond the basics: the `linearmodels` package complements statsmodels.

### Where it fits in the ML stack

```text
pandas (data)  ->  statsmodels (inference, diagnostics, classical forecasting)
               ->  scikit-learn (prediction pipelines, CV, model selection)
Visualization: Matplotlib / seaborn (statsmodels graphics return Matplotlib figures)
```

In ML projects statsmodels is typically used to: understand feature effects before or alongside a predictive model, provide a strong interpretable baseline, analyze A/B tests of deployed models, check residual assumptions, and build baseline forecasts.

## Installation

### pip

```bash
python -m pip install -U statsmodels
```

statsmodels 0.14 requires NumPy, SciPy, pandas, patsy, and packaging (installed automatically); recent 0.14.x releases require Python 3.9 or newer. Binary wheels exist for Windows, macOS, and Linux.

### conda

```bash
conda install -c conda-forge statsmodels
```

### Optional dependencies

| Package | Used for |
|---|---|
| `matplotlib` | All plotting in `statsmodels.graphics` and `plot_*` methods |
| `cvxopt` | Some constrained/regularized fitting (e.g. L1 with `method="l1_cvxopt_cp"`) |
| `joblib` | Parallel computations in some functions |
| `pmdarima` (separate project) | Automated ARIMA order selection (`auto_arima`) built on statsmodels |

### Verifying the install

```python
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

print(statsmodels.__version__)          # e.g. '0.14.4'

data = sm.datasets.longley.load_pandas()
res = sm.OLS(data.endog, sm.add_constant(data.exog)).fit()
print(res.rsquared)                     # about 0.995
```

Running the test suite (slow, requires pytest):

```python
import statsmodels
statsmodels.test()
```

## Core Concepts

### Model -> fit -> results

Every estimator follows the same three-step pattern:

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2))
y = 1.5 + 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.8, size=200)

model = sm.OLS(y, sm.add_constant(X))   # 1. specify: data are bound at construction
results = model.fit()                   # 2. estimate: returns a Results object
print(results.summary())                # 3. inspect: params, bse, pvalues, tests
print(results.params)                   # array([ 1.5..,  2.0.., -0.5..])
```

Contrast with scikit-learn, where data are passed to `fit`. In statsmodels, the model instance *is* the data plus specification; you can call `fit` repeatedly with different options (`cov_type`, optimizer, regularization) on the same model.

### endog and exog

statsmodels uses econometric naming:

- **endog** - the endogenous (dependent, response, target) variable, `y`.
- **exog** - the exogenous (independent, explanatory) variables, the design matrix `X`.

### The intercept is NOT added automatically (array API)

With the array API you must add a constant column yourself; otherwise the model is fit through the origin and R-squared is the uncentered version.

```python
X_const = sm.add_constant(X)           # prepends a column of ones named 'const' for DataFrames
```

The formula API adds an intercept automatically (remove it with `- 1` or `+ 0`).

### Formulas (patsy)

The formula API turns a DataFrame and an R-style formula into `endog`/`exog`, handling categorical encoding and transformations:

| Formula syntax | Meaning |
|---|---|
| `y ~ x1 + x2` | Main effects with intercept |
| `y ~ x1 + x2 - 1` | No intercept |
| `y ~ C(group)` | Treat `group` as categorical (dummy-coded, first level is reference) |
| `y ~ C(group, Treatment(reference="b"))` | Choose reference level |
| `y ~ C(group, Sum)` | Sum (effect) coding |
| `y ~ x1 * x2` | `x1 + x2 + x1:x2` (main effects + interaction) |
| `y ~ x1:x2` | Interaction only |
| `y ~ np.log(x1)` | Any NumPy function |
| `y ~ I(x1 ** 2)` | Arithmetic inside `I()` |
| `y ~ Q("weird name")` | Column names with spaces or special characters |
| `y ~ center(x1) + standardize(x2)` | patsy built-in stateful transforms (reapplied correctly at predict time) |
| `y ~ bs(x, df=4)` / `cr(x, df=4)` | B-spline / natural cubic regression spline basis |

```python
import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame({
    "y": [3.1, 4.2, 5.9, 7.1, 8.8, 9.7, 12.1, 13.0],
    "x": [1, 2, 3, 4, 5, 6, 7, 8],
    "g": ["a", "b", "a", "b", "a", "b", "a", "b"],
})
res = smf.ols("y ~ x + C(g)", data=df).fit()
print(res.params)
# Intercept    ...
# C(g)[T.b]    ...
# x            ...
print(res.model.exog_names)
```

When you call `results.predict(new_df)` on a formula model, the same transformations (including category coding and stateful transforms) are applied to the new data automatically.

### Results objects

Results objects carry estimates, inference, and diagnostics. Commonly used attributes and methods (exact availability varies by model):

| Attribute / method | Description |
|---|---|
| `params` | Estimated coefficients (Series when inputs are pandas) |
| `bse` | Standard errors |
| `tvalues` / `pvalues` | Test statistics and p-values for H0: coefficient = 0 |
| `conf_int(alpha=0.05)` | Confidence intervals |
| `summary()` / `summary2()` | Formatted tables (`summary2` returns more DataFrame-friendly tables) |
| `fittedvalues`, `resid` | In-sample predictions and residuals |
| `predict(exog=None)` | Point predictions |
| `get_prediction(exog=None)` | Predictions with standard errors and intervals |
| `rsquared`, `rsquared_adj` | (OLS family) Goodness of fit |
| `llf`, `aic`, `bic` | Log-likelihood and information criteria |
| `nobs`, `df_model`, `df_resid` | Sample size and degrees of freedom |
| `cov_params()` | Parameter covariance matrix |
| `f_test`, `t_test`, `wald_test` | Linear hypothesis tests |
| `save(fname)` / `sm.load(fname)` | Pickle persistence |

### Covariance types (robust standard errors)

The covariance estimator is chosen at fit time, not model construction:

| `cov_type` | Use when | Extra `cov_kwds` |
|---|---|---|
| `"nonrobust"` | Classical homoskedastic errors (default) | |
| `"HC0"`..`"HC3"` | Heteroskedasticity (HC3 recommended for small samples) | |
| `"HAC"` | Heteroskedasticity and autocorrelation (Newey-West) | `{"maxlags": 4}` |
| `"cluster"` | Correlated errors within groups | `{"groups": group_array}` |
| `"hac-panel"`, `"hac-groupsum"` | Panel-type data | see docs |

```python
res_robust = model.fit(cov_type="HC3")
res_cluster = model.fit(cov_type="cluster", cov_kwds={"groups": rng.integers(0, 20, 200)})
print(res_robust.bse, res_cluster.bse)
```

The coefficients are identical; only the standard errors, p-values, and confidence intervals change.

### pandas in, pandas out

If you pass pandas objects, results use your column names and index (and time-series models use the date index for forecasting). If you pass NumPy arrays, names default to `const`, `x1`, `x2`, ...

### Missing data

Models raise `MissingDataError` on NaN by default (`missing="none"`). Pass `missing="drop"` to drop incomplete rows, or clean the data first. The formula API drops rows with missing values in the used columns by default (patsy's `NA_action="drop"`).

## API Reference

Examples assume:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

### Data helpers and datasets

#### add_constant

```python
sm.add_constant(data, prepend=True, has_constant="skip")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | array or DataFrame | required | Design matrix |
| `prepend` | bool | True | Put the constant first (otherwise last) |
| `has_constant` | `"skip"`, `"add"`, `"raise"` | `"skip"` | Behavior if a constant column already exists |

Returns the same type with a `const` column added.

```python
X = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [0.5, 0.1, 0.9]})
print(sm.add_constant(X))
```

#### Built-in datasets

```python
sm.datasets.<name>.load_pandas()                     # -> Dataset with .data, .endog, .exog
sm.datasets.get_rdataset(dataname, package="datasets", cache=False)   # R datasets (internet)
```

Bundled datasets include `longley`, `macrodata`, `co2`, `sunspots`, `spector`, `star98`, `fair`, `scotland`, `anes96`, `randhie`, `grunfeld`, `nile`, `elnino`, `stackloss`, `engel`, `copper`, `committee`, `cpunish`, `heart`, `statecrime`, `strikes`, `china_smoking`, `interest_inflation`, `modechoice`, `danish_data`, `fertility`.

```python
macro = sm.datasets.macrodata.load_pandas().data       # quarterly US macro data
spector = sm.datasets.spector.load_pandas()            # .endog = GRADE, .exog = GPA, TUCE, PSI
duncan = sm.datasets.get_rdataset("Duncan", "carData").data   # downloads from Rdatasets
```

### Linear regression

#### OLS

```python
sm.OLS(endog, exog=None, missing="none", hasconst=None, **kwargs)
OLS.fit(method="pinv", cov_type="nonrobust", cov_kwds=None, use_t=None, **kwargs)
OLS.fit_regularized(method="elastic_net", alpha=0.0, L1_wt=1.0,
                    start_params=None, profile_scale=False, refit=False, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `endog` | array-like (n,) | required | Response |
| `exog` | array-like (n, k) | None | Regressors (add a constant yourself) |
| `missing` | `"none"`, `"drop"`, `"raise"` | `"none"` | NaN handling |
| `hasconst` | bool | None | Declare whether `exog` has a user-supplied constant |
| `method` (fit) | `"pinv"` or `"qr"` | `"pinv"` | Linear algebra method |
| `cov_type` (fit) | str | `"nonrobust"` | Covariance estimator (see Core Concepts) |
| `cov_kwds` (fit) | dict | None | Extra arguments for `cov_type` |
| `alpha`, `L1_wt` (fit_regularized) | float | 0.0, 1.0 | Penalty weight; `L1_wt=1` lasso, `0` ridge |

`fit` returns `RegressionResultsWrapper` (`OLSResults`). `fit_regularized` returns a results object with `params` only (no standard errors).

```python
data = sm.datasets.longley.load_pandas()
X = sm.add_constant(data.exog)
res = sm.OLS(data.endog, X).fit()
print(res.summary())
print(res.rsquared_adj, res.fvalue, res.f_pvalue)
print(res.conf_int(alpha=0.05))
```

Key OLS-specific results: `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`, `ess`, `ssr`, `mse_resid`, `condition_number`, `get_influence()`, `outlier_test()`, `compare_f_test(restricted)`, `compare_lr_test(restricted)`.

#### smf.ols (formula)

```python
smf.ols(formula, data, subset=None, drop_cols=None, *args, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `formula` | str | required | R-style formula, e.g. `"y ~ x1 + C(g)"` |
| `data` | DataFrame or dict | required | Data source |
| `subset` | array-like | None | Row subset (boolean or index) |

Every model class has a formula constructor in `smf`: `ols`, `wls`, `gls`, `glm`, `logit`, `probit`, `mnlogit`, `poisson`, `negativebinomial`, `quantreg`, `rlm`, `mixedlm`, `gee`, `phreg`, and more.

```python
tips = pd.DataFrame({
    "tip": [1.01, 1.66, 3.50, 3.31, 3.61, 4.71, 2.00, 3.12, 1.96, 3.23],
    "total_bill": [16.99, 10.34, 21.01, 23.68, 24.59, 25.29, 8.77, 26.88, 15.04, 14.78],
    "smoker": ["No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes"],
})
res = smf.ols("tip ~ total_bill + C(smoker)", data=tips).fit(cov_type="HC3")
print(res.params.round(3))
```

#### WLS and GLS

```python
sm.WLS(endog, exog, weights=1.0, missing="none", hasconst=None, **kwargs)
sm.GLS(endog, exog, sigma=None, missing="none", hasconst=None, **kwargs)
sm.GLSAR(endog, exog=None, rho=1, missing="none", hasconst=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `weights` | array (n,) | 1.0 | Inverse-variance weights (proportional to 1 / Var(error_i)) |
| `sigma` | scalar, (n,), or (n, n) | None | Error covariance structure |
| `rho` (GLSAR) | int or array | 1 | AR order of errors; use `iterative_fit()` |

```python
rng = np.random.default_rng(1)
x = np.linspace(1, 10, 100)
y = 2 + 3 * x + rng.normal(scale=x)            # variance grows with x
res_wls = sm.WLS(y, sm.add_constant(x), weights=1.0 / x ** 2).fit()
print(res_wls.params, res_wls.bse)
```

#### QuantReg

```python
sm.QuantReg(endog, exog, **kwargs)
QuantReg.fit(q=0.5, vcov="robust", kernel="epa", bandwidth="hsheather",
             max_iter=1000, p_tol=1e-06, **kwargs)
smf.quantreg(formula, data)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | float in (0, 1) | 0.5 | Quantile to model (0.5 = median regression) |
| `vcov` | str | `"robust"` | `"robust"` or `"iid"` |

```python
engel = sm.datasets.engel.load_pandas().data
for q in (0.1, 0.5, 0.9):
    r = smf.quantreg("foodexp ~ income", engel).fit(q=q)
    print(q, r.params["income"].round(4))
```

#### RLM (robust linear model)

```python
sm.RLM(endog, exog, M=None, missing="none", **kwargs)
RLM.fit(maxiter=50, tol=1e-8, scale_est="mad", init=None, cov="H1",
        update_scale=True, conv="dev", start_params=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `M` | `RobustNorm` | `HuberT()` | `sm.robust.norms.HuberT()`, `TukeyBiweight()`, `Hampel()`, `AndrewWave()`, `LeastSquares()` |
| `scale_est` | str or object | `"mad"` | Scale estimator |

```python
stack = sm.datasets.stackloss.load_pandas()
res = sm.RLM(stack.endog, sm.add_constant(stack.exog),
             M=sm.robust.norms.TukeyBiweight()).fit()
print(res.params)
print(res.weights.round(2))       # downweighted observations have weight < 1
```

#### RollingOLS and RecursiveLS

```python
from statsmodels.regression.rolling import RollingOLS
RollingOLS(endog, exog, window=None, *, min_nobs=None, missing="drop", expanding=False)
sm.RecursiveLS(endog, exog, constraints=None, **kwargs)
```

```python
macro = sm.datasets.macrodata.load_pandas().data
X = sm.add_constant(macro[["realgdp"]])
roll = RollingOLS(macro["realcons"], X, window=40).fit()
print(roll.params.tail())         # one row of coefficients per window end
```

### Generalized linear models

#### GLM

```python
sm.GLM(endog, exog, family=None, offset=None, exposure=None,
       freq_weights=None, var_weights=None, missing="none", **kwargs)
GLM.fit(start_params=None, maxiter=100, method="IRLS", tol=1e-8,
        scale=None, cov_type="nonrobust", cov_kwds=None, use_t=None,
        full_output=True, disp=False, max_start_irls=3, **kwargs)
smf.glm(formula, data, family=None, offset=None, exposure=None, ...)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `family` | `sm.families.Family` | `Gaussian()` | Distribution and link |
| `offset` | array | None | Added to the linear predictor with coefficient 1 |
| `exposure` | array | None | `log(exposure)` is used as offset (log link only) |
| `freq_weights` | array | None | Frequency (replication) weights |
| `var_weights` | array | None | Analytic/variance weights |
| `method` (fit) | str | `"IRLS"` | Or a scipy optimizer name (`"bfgs"`, `"newton"`) |

Families and their default links:

| Family | Default link | Typical use |
|---|---|---|
| `sm.families.Gaussian()` | Identity | Continuous outcome (same as OLS) |
| `sm.families.Binomial()` | Logit | Binary outcome or proportions |
| `sm.families.Poisson()` | Log | Counts |
| `sm.families.NegativeBinomial(alpha=1.0)` | Log | Overdispersed counts (alpha fixed) |
| `sm.families.Gamma()` | InversePower | Positive skewed continuous (costs, durations) |
| `sm.families.InverseGaussian()` | InverseSquared | Positive skewed continuous |
| `sm.families.Tweedie(var_power=1.5)` | Log | Insurance claims: zeros plus positive amounts |

Links live in `sm.families.links`: `Logit()`, `Probit()`, `CLogLog()`, `Log()`, `Identity()`, `InversePower()`, `Sqrt()`. Version note: lowercase link aliases such as `links.logit` and `links.log` are deprecated in 0.14; instantiate the CamelCase classes (`links.Log()`).

GLM results add `deviance`, `pearson_chi2`, `null_deviance`, `pseudo_rsquared(kind="cs")`, `resid_deviance`, `resid_pearson`, `resid_response`.

```python
star98 = sm.datasets.star98.load_pandas()
endog = star98.endog          # two columns: successes, failures
exog = sm.add_constant(star98.exog[["LOWINC", "PERASIAN", "PERBLACK"]])
res = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
print(res.summary())

# Gamma regression with log link
rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.uniform(0, 2, 300)})
df["cost"] = rng.gamma(shape=2.0, scale=np.exp(0.5 + 0.8 * df["x"]) / 2.0)
gam = smf.glm("cost ~ x", data=df,
              family=sm.families.Gamma(link=sm.families.links.Log())).fit()
print(gam.params)             # close to [0.5, 0.8]
```

#### GEE

```python
sm.GEE(endog, exog, groups, time=None, family=None, cov_struct=None,
       missing="none", offset=None, exposure=None, dep_data=None,
       constraint=None, update_dep=True, weights=None, **kwargs)
smf.gee(formula, groups, data, cov_struct=None, family=None, ...)
```

Population-averaged models for clustered/longitudinal data. Working correlation structures: `sm.cov_struct.Independence()`, `Exchangeable()`, `Autoregressive()`, `Nested()`.

```python
rng = np.random.default_rng(2)
n_groups, per = 50, 6
g = np.repeat(np.arange(n_groups), per)
u = rng.normal(size=n_groups)[g]
x = rng.normal(size=g.size)
p = 1 / (1 + np.exp(-(-0.3 + 0.8 * x + u)))
df = pd.DataFrame({"y": rng.binomial(1, p), "x": x, "g": g})
res = smf.gee("y ~ x", groups="g", data=df, family=sm.families.Binomial(),
              cov_struct=sm.cov_struct.Exchangeable()).fit()
print(res.summary())
```

### Discrete choice and count models

#### Logit and Probit

```python
sm.Logit(endog, exog, offset=None, check_rank=True, **kwargs)
sm.Probit(endog, exog, offset=None, check_rank=True, **kwargs)
Logit.fit(start_params=None, method="newton", maxiter=35, full_output=1,
          disp=1, callback=None, **kwargs)
Logit.fit_regularized(start_params=None, method="l1", maxiter="defined_by_method",
                      full_output=1, disp=1, callback=None, alpha=0,
                      trim_mode="auto", auto_trim_tol=0.01, size_trim_tol=0.0001,
                      qc_tol=0.03, **kwargs)
smf.logit(formula, data)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `endog` | array of 0/1 | required | Binary outcome |
| `method` (fit) | str | `"newton"` | `"newton"`, `"bfgs"`, `"lbfgs"`, `"nm"`, `"powell"`, ... |
| `maxiter` | int | 35 | Maximum iterations |
| `disp` | bool/int | 1 | Print convergence messages (use `disp=0` to silence) |
| `alpha` (fit_regularized) | float or array | 0 | L1 penalty weight |

Results add `prsquared` (McFadden pseudo R-squared), `llr`, `llr_pvalue`, `get_margeff(at="overall", method="dydx")`, `pred_table(threshold=0.5)`, and `predict()` returns probabilities.

```python
spector = sm.datasets.spector.load_pandas()
X = sm.add_constant(spector.exog)
logit_res = sm.Logit(spector.endog, X).fit(disp=0)
print(logit_res.summary())
print(np.exp(logit_res.params))                     # odds ratios
print(logit_res.get_margeff().summary())            # average marginal effects
print(logit_res.pred_table())                       # confusion matrix at 0.5
```

#### MNLogit

```python
sm.MNLogit(endog, exog, check_rank=True, **kwargs)
```

Multinomial logit for unordered categories (K-1 sets of coefficients; the first category is the reference). `predict()` returns an (n, K) probability matrix.

```python
anes = sm.datasets.anes96.load_pandas().data
X = sm.add_constant(anes[["TVnews", "selfLR", "age", "educ", "income"]])
mn = sm.MNLogit(anes["PID"], X).fit(disp=0)   # PID: 7 party-identification levels
print(mn.params.shape)        # (6 regressors, 6 non-reference outcome categories)
print(mn.predict(X)[:2].round(3))
```

#### OrderedModel

```python
from statsmodels.miscmodels.ordinal_model import OrderedModel
OrderedModel(endog, exog, offset=None, distr="probit", **kwargs)
OrderedModel.from_formula(formula, data, ...)
```

Ordinal regression (`distr="logit"` for proportional odds). Do not include a constant: the thresholds play that role.

```python
rng = np.random.default_rng(3)
x = rng.normal(size=500)
latent = 1.2 * x + rng.logistic(size=500)
y = pd.Series(pd.cut(latent, [-np.inf, -1, 1, np.inf], labels=["low", "mid", "high"]))
om = OrderedModel(y, x[:, None], distr="logit").fit(method="bfgs", disp=False)
print(om.params)              # slope then two threshold parameters
```

#### Poisson, NegativeBinomial, and zero-inflated models

```python
sm.Poisson(endog, exog, offset=None, exposure=None, missing="none", check_rank=True, **kwargs)
sm.NegativeBinomial(endog, exog, loglike_method="nb2", offset=None,
                    exposure=None, missing="none", check_rank=True, **kwargs)
sm.ZeroInflatedPoisson(endog, exog, exog_infl=None, offset=None,
                       exposure=None, inflation="logit", missing="none", **kwargs)
sm.ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=None, offset=None,
                                 exposure=None, inflation="logit", p=2, missing="none", **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `exposure` | array | None | Time/size at risk; `log(exposure)` enters as offset |
| `loglike_method` | str | `"nb2"` | `"nb2"`, `"nb1"`, or `"geometric"` |
| `exog_infl` | array | None | Regressors for the zero-inflation part (default: constant) |

`NegativeBinomial` estimates the dispersion `alpha` (reported as a parameter), unlike `GLM(family=NegativeBinomial(alpha=...))`, which holds it fixed.

```python
rng = np.random.default_rng(4)
n = 1000
x = rng.normal(size=n)
mu = np.exp(0.5 + 0.7 * x)
y = rng.negative_binomial(n=2, p=2 / (2 + mu))       # overdispersed counts
X = sm.add_constant(x)
pois = sm.Poisson(y, X).fit(disp=0)
nb = sm.NegativeBinomial(y, X).fit(disp=0)
print(pois.aic, nb.aic)       # NB fits much better
print(nb.params)              # const, x1, alpha
```

### Mixed effects models

#### MixedLM

```python
sm.MixedLM(endog, exog, groups, exog_re=None, exog_vc=None, use_sqrt=True,
           missing="none", **kwargs)
smf.mixedlm(formula, data, re_formula=None, vc_formula=None, subset=None,
            use_sparse=False, missing="none", *args, **kwargs)
MixedLM.fit(start_params=None, reml=True, niter_sa=0, do_cg=True,
            fe_pen=None, cov_pen=None, free=None, full_output=False,
            method=None, **fit_kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups` | str or array | required | Grouping variable for random effects |
| `re_formula` | str | None (random intercept) | Random-effects design, e.g. `"~ time"` for random slopes |
| `vc_formula` | dict | None | Variance components, e.g. `{"classroom": "0 + C(classroom)"}` |
| `reml` (fit) | bool | True | REML (True) or ML (False; needed for LR tests of fixed effects) |
| `method` (fit) | str or list | None | Optimizer(s), e.g. `["lbfgs"]` |

Results add `random_effects` (dict of per-group BLUPs), `cov_re`, and `fe_params`.

```python
rng = np.random.default_rng(5)
subjects = np.repeat(np.arange(30), 8)
time = np.tile(np.arange(8), 30)
b0 = rng.normal(0, 2, 30)[subjects]
b1 = rng.normal(0, 0.3, 30)[subjects]
y = 10 + b0 + (0.8 + b1) * time + rng.normal(0, 1, subjects.size)
df = pd.DataFrame({"y": y, "time": time, "subject": subjects})

mlm = smf.mixedlm("y ~ time", df, groups=df["subject"], re_formula="~time").fit()
print(mlm.summary())
print(mlm.fe_params)
print(list(mlm.random_effects.items())[0])
```

### Statistical tests (statsmodels.stats)

#### t-tests and DescrStatsW

```python
from statsmodels.stats.weightstats import ttest_ind, ztest, DescrStatsW, CompareMeans

ttest_ind(x1, x2, alternative="two-sided", usevar="pooled", weights=(None, None), value=0)
ztest(x1, x2=None, value=0, alternative="two-sided", usevar="pooled", ddof=1.0)
DescrStatsW(data, weights=None, ddof=0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alternative` | str | `"two-sided"` | `"larger"` or `"smaller"` for one-sided tests |
| `usevar` | str | `"pooled"` | `"unequal"` gives Welch's t-test |
| `value` | float | 0 | Hypothesized difference |

`ttest_ind` returns `(tstat, pvalue, df)`.

```python
rng = np.random.default_rng(6)
a, b = rng.normal(10, 2, 80), rng.normal(10.8, 2.5, 90)
t, p, dof = ttest_ind(a, b, usevar="unequal")
print(round(t, 3), round(p, 4))
d = DescrStatsW(a)
print(d.mean, d.std, d.tconfint_mean())
cm = CompareMeans(DescrStatsW(a), DescrStatsW(b))
print(cm.tconfint_diff(usevar="unequal"))
```

#### Proportions

```python
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportions_chisquare

proportions_ztest(count, nobs, value=None, alternative="two-sided", prop_var=False)
proportion_confint(count, nobs, alpha=0.05, method="normal")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `count` | int or array | required | Successes (array for two-sample test) |
| `nobs` | int or array | required | Trials |
| `method` (confint) | str | `"normal"` | `"wilson"`, `"agresti_coull"`, `"beta"` (Clopper-Pearson), `"jeffreys"`, `"binom_test"` |

```python
# A/B test: conversions in control vs treatment
stat, p = proportions_ztest(count=np.array([420, 480]), nobs=np.array([5000, 5000]))
print(stat, p)
print(proportion_confint(480, 5000, method="wilson"))
```

#### ANOVA

```python
from statsmodels.stats.anova import anova_lm, AnovaRM

anova_lm(*args, typ=1, robust=None, test="F", scale=None)
AnovaRM(data, depvar, subject, within=None, between=None, aggregate_func=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `*args` | fitted OLS results | required | One model: ANOVA table. Several nested models: model comparison |
| `typ` | 1, 2, 3 | 1 | Sum of squares type (use 2 or 3 for unbalanced designs) |
| `robust` | str | None | Heteroskedasticity-robust covariance for typ 2/3 (`"hc3"`) |

Returns a DataFrame.

```python
rng = np.random.default_rng(7)
df = pd.DataFrame({
    "dose": np.repeat(["low", "mid", "high"], 40),
    "sex": np.tile(["F", "M"], 60),
})
effect = df["dose"].map({"low": 0, "mid": 1, "high": 2.5})
df["y"] = 5 + effect + rng.normal(size=120)
model = smf.ols("y ~ C(dose) * C(sex)", data=df).fit()
print(anova_lm(model, typ=2))

reduced = smf.ols("y ~ C(dose)", data=df).fit()
print(anova_lm(reduced, model))       # F-test comparing nested models
```

#### Multiple comparisons

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.stats.multitest import multipletests, fdrcorrection

pairwise_tukeyhsd(endog, groups, alpha=0.05)
multipletests(pvals, alpha=0.05, method="hs", maxiter=1, is_sorted=False, returnsorted=False)
```

| `method` | Procedure |
|---|---|
| `"bonferroni"` | Bonferroni |
| `"holm"` | Holm step-down |
| `"hs"` | Holm-Sidak (default) |
| `"fdr_bh"` | Benjamini-Hochberg FDR |
| `"fdr_by"` | Benjamini-Yekutieli FDR |

`multipletests` returns `(reject, pvals_corrected, alphacSidak, alphacBonf)`.

```python
tukey = pairwise_tukeyhsd(df["y"], df["dose"], alpha=0.05)
print(tukey.summary())

pvals = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
print(reject, p_adj.round(3))
```

#### Power and sample size

```python
from statsmodels.stats.power import TTestIndPower, TTestPower, NormalIndPower, GofChisquarePower
from statsmodels.stats.proportion import proportion_effectsize

TTestIndPower().solve_power(effect_size=None, nobs1=None, alpha=None,
                            power=None, ratio=1.0, alternative="two-sided")
NormalIndPower().solve_power(effect_size=None, nobs1=None, alpha=None,
                             power=None, ratio=1.0, alternative="two-sided")
proportion_effectsize(prop1, prop2, method="normal")
```

Leave exactly one of `effect_size`, `nobs1`, `alpha`, `power` as None; it is solved for.

```python
n = TTestIndPower().solve_power(effect_size=0.3, alpha=0.05, power=0.8)
print(round(n))                       # about 175 per group

es = proportion_effectsize(0.12, 0.10)          # treatment 12% vs baseline 10% conversion
n_ab = NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.8)
print(round(n_ab))                    # per-group sample size
```

#### Regression diagnostics

```python
from statsmodels.stats.diagnostic import (het_breuschpagan, het_white, acorr_ljungbox,
                                          acorr_breusch_godfrey, linear_reset, lilliefors)
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence

het_breuschpagan(resid, exog_het, robust=True)    # -> (lm, lm_pvalue, fvalue, f_pvalue)
het_white(resid, exog)                            # -> (lm, lm_pvalue, fvalue, f_pvalue)
durbin_watson(resids, axis=0)                     # ~2 means no first-order autocorrelation
jarque_bera(resids, axis=0)                       # -> (jb, jb_pvalue, skew, kurtosis)
acorr_ljungbox(x, lags=None, boxpierce=False, model_df=0, period=None,
               return_df=True, auto_lag=False)    # -> DataFrame with lb_stat, lb_pvalue
acorr_breusch_godfrey(res, nlags=None, store=False)
linear_reset(res, power=3, test_type="fitted", use_f=False, cov_type="nonrobust", cov_kwargs=None)
variance_inflation_factor(exog, exog_idx)         # exog as ndarray including the constant
```

| Test | Null hypothesis | Small p-value means |
|---|---|---|
| Breusch-Pagan / White | Homoskedastic errors | Heteroskedasticity: use robust SEs |
| Durbin-Watson | (statistic, not p-value) | Values far from 2 suggest autocorrelation |
| Ljung-Box | No autocorrelation up to lag k | Residual autocorrelation remains |
| Jarque-Bera / Omnibus | Normal residuals | Non-normal residuals |
| RESET | Correct functional form | Missing non-linearity |
| VIF | (statistic) | VIF > 5-10 signals multicollinearity |

```python
data = sm.datasets.longley.load_pandas()
X = sm.add_constant(data.exog)
res = sm.OLS(data.endog, X).fit()

lm, lm_p, f, f_p = het_breuschpagan(res.resid, res.model.exog)
print("BP p-value:", round(lm_p, 4))
print("DW:", round(durbin_watson(res.resid), 3))
print(acorr_ljungbox(res.resid, lags=[4]))
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns)
print(vif.round(1))

infl = res.get_influence()
summary = infl.summary_frame()          # cooks_d, hat_diag, student_resid, dffits, ...
print(summary["cooks_d"].sort_values(ascending=False).head(3))
```

### Time series analysis (statsmodels.tsa)

#### Stationarity tests: adfuller and kpss

```python
from statsmodels.tsa.stattools import adfuller, kpss

adfuller(x, maxlag=None, regression="c", autolag="AIC", store=False, regresults=False)
kpss(x, regression="c", nlags="auto", store=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `regression` | str | `"c"` | ADF: `"c"`, `"ct"`, `"ctt"`, `"n"`; KPSS: `"c"` (level) or `"ct"` (trend) |
| `autolag` | str or None | `"AIC"` | Lag selection: `"AIC"`, `"BIC"`, `"t-stat"`, None |
| `nlags` | str or int | `"auto"` | KPSS lags (`"auto"` or `"legacy"`) |

`adfuller` returns `(adf_stat, pvalue, usedlag, nobs, critical_values, icbest)`; null hypothesis = unit root (non-stationary). `kpss` returns `(stat, pvalue, lags, critical_values)`; null hypothesis = stationary. Using both gives a more reliable conclusion.

```python
co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().ffill()
adf = adfuller(co2)
print(f"ADF stat={adf[0]:.2f}, p={adf[1]:.3f}")          # non-stationary
adf_d = adfuller(co2.diff(12).diff().dropna())
print(f"after differencing p={adf_d[1]:.4f}")
```

#### acf, pacf, and their plots

```python
from statsmodels.tsa.stattools import acf, pacf, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

acf(x, adjusted=False, nlags=None, qstat=False, fft=True, alpha=None,
    bartlett_confint=True, missing="none")
pacf(x, nlags=None, method="ywadjusted", alpha=None)
plot_acf(x, ax=None, lags=None, *, alpha=0.05, use_vlines=True, adjusted=False,
         fft=False, missing="none", title="Autocorrelation", zero=True,
         auto_ylims=False, bartlett_confint=True, vlines_kwargs=None, **kwargs)
plot_pacf(x, ax=None, lags=None, alpha=0.05, method="ywm", use_vlines=True,
          title="Partial Autocorrelation", zero=True, vlines_kwargs=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nlags` / `lags` | int | None (about `10*log10(n)`) | Number of lags |
| `alpha` | float | None / 0.05 | If given, return/plot confidence intervals |
| `method` (pacf) | str | `"ywadjusted"` | `"ywm"`, `"ols"`, `"ldb"`, `"burg"`, ... |
| `zero` | bool | True | Include lag 0 in plots |

Version note: the `unbiased=` argument was renamed `adjusted=` (and method names like `"ywunbiased"` became `"ywadjusted"`).

```python
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
plot_acf(co2.diff().dropna(), lags=36, ax=axs[0])
plot_pacf(co2.diff().dropna(), lags=36, ax=axs[1], method="ywm")
fig.savefig("acf_pacf.png")
```

#### seasonal_decompose and STL

```python
from statsmodels.tsa.seasonal import seasonal_decompose, STL

seasonal_decompose(x, model="additive", filt=None, period=None,
                   two_sided=True, extrapolate_trend=0)
STL(endog, period=None, seasonal=7, trend=None, low_pass=None,
    seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False,
    seasonal_jump=1, trend_jump=1, low_pass_jump=1)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `"additive"` or `"multiplicative"` | `"additive"` | Decomposition type (seasonal_decompose) |
| `period` | int | None (inferred from index freq) | Seasonal period (`freq=` was renamed `period=`) |
| `seasonal` | odd int | 7 | STL seasonal smoother length |
| `robust` | bool | False | STL robust to outliers |

Both return a result with `.trend`, `.seasonal`, `.resid`, `.observed`, and `.plot()`.

```python
stl = STL(co2, period=12, robust=True).fit()
fig = stl.plot()
fig.savefig("stl.png")
deseasonalized = co2 - stl.seasonal
```

#### ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA

ARIMA(endog, exog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0),
      trend=None, enforce_stationarity=True, enforce_invertibility=True,
      concentrate_scale=False, trend_offset=1, dates=None, freq=None,
      missing="none", validate_specification=True)
ARIMA.fit(start_params=None, transformed=True, includes_fixed=False,
          method=None, method_kwargs=None, gls=None, gls_kwargs=None,
          cov_type=None, cov_kwds=None, return_params=False, low_memory=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `endog` | Series/array | required | Time series (pandas with a frequency is best) |
| `exog` | array/DataFrame | None | Exogenous regressors (ARIMAX / regression with ARIMA errors) |
| `order` | (p, d, q) | (0, 0, 0) | AR order, differencing, MA order |
| `seasonal_order` | (P, D, Q, s) | (0, 0, 0, 0) | Seasonal part with period `s` |
| `trend` | str or list | None | `"n"`, `"c"`, `"t"`, `"ct"`; defaults to `"c"` if d=0 and D=0, else `"n"` |
| `method` (fit) | str | None (`"statespace"`) | Also `"innovations_mle"`, `"hannan_rissanen"`, `"burg"`, `"yule_walker"` |

Results methods: `summary()`, `forecast(steps=1, exog=None)`, `get_forecast(steps, exog=None)` (`.predicted_mean`, `.conf_int(alpha)`, `.summary_frame()`), `predict(start=None, end=None, dynamic=False)`, `get_prediction(...)`, `plot_diagnostics(figsize=(15, 10))`, `append(new_endog, refit=False)`, `apply(new_endog)`, `simulate(nsimulations)`, `aic`, `bic`, `aicc`, `resid`.

Removal note: the legacy `statsmodels.tsa.arima_model.ARMA` and `statsmodels.tsa.arima_model.ARIMA` classes (and `sm.tsa.ARMA`) were deprecated in 0.12 and removed in 0.13. Use `statsmodels.tsa.arima.model.ARIMA` (also available as `sm.tsa.arima.ARIMA`). The new class takes `order=` as a keyword and uses `trend=` instead of the old `fit(trend=...)`.

```python
from statsmodels.tsa.arima.model import ARIMA

macro = sm.datasets.macrodata.load_pandas().data
idx = pd.period_range("1959Q1", periods=len(macro), freq="Q").to_timestamp()
infl = pd.Series(macro["infl"].values, index=idx).asfreq("QS")

res = ARIMA(infl, order=(2, 0, 1)).fit()
print(res.summary().tables[1])
fc = res.get_forecast(steps=8)
print(fc.summary_frame(alpha=0.05).head())    # mean, mean_se, mean_ci_lower, mean_ci_upper
```

#### SARIMAX

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

SARIMAX(endog, exog=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
        trend=None, measurement_error=False, time_varying_regression=False,
        mle_regression=True, simple_differencing=False,
        enforce_stationarity=True, enforce_invertibility=True,
        hamilton_representation=False, concentrate_scale=False,
        trend_offset=1, use_exact_diffuse=False, dates=None, freq=None,
        missing="none", validate_specification=True, **kwargs)
SARIMAX.fit(start_params=None, transformed=True, includes_fixed=False,
            cov_type=None, cov_kwds=None, method="lbfgs", maxiter=50,
            full_output=1, disp=5, callback=None, return_params=False,
            optim_score=None, optim_complex_step=None, optim_hessian=None,
            flags=None, low_memory=False, **kwargs)
```

The general state space SARIMAX model; `ARIMA` is a restricted, simpler interface on the same machinery. Use `SARIMAX` when you need its extra options (time-varying regression, measurement error, optimizer control). Pass `disp=False` to silence optimizer output.

```python
co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().ffill()
train, test = co2[:-24], co2[-24:]
mod = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
res = mod.fit(disp=False)
pred = res.get_forecast(steps=24)
mae = (pred.predicted_mean - test).abs().mean()
print(f"MAE = {mae:.3f}")
```

#### AutoReg

```python
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

AutoReg(endog, lags, trend="c", seasonal=False, exog=None, hold_back=None,
        period=None, missing="none", *, deterministic=None, old_names=False)
ar_select_order(endog, maxlag, ic="bic", glob=False, trend="c",
                seasonal=False, exog=None, hold_back=None, period=None,
                missing="none", old_names=False)
```

Fast OLS-estimated autoregression. Replaces the removed `statsmodels.tsa.ar_model.AR` class.

```python
sun = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"]
sel = ar_select_order(sun, maxlag=15, ic="aic")
print(sel.ar_lags)
res = AutoReg(sun, lags=sel.ar_lags).fit()
print(res.predict(start=len(sun), end=len(sun) + 9))
```

#### Exponential smoothing

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

ExponentialSmoothing(endog, trend=None, damped_trend=False, seasonal=None, *,
                     seasonal_periods=None, initialization_method="estimated",
                     initial_level=None, initial_trend=None, initial_seasonal=None,
                     use_boxcox=False, bounds=None, dates=None, freq=None, missing="none")
ExponentialSmoothing.fit(smoothing_level=None, smoothing_trend=None,
                         smoothing_seasonal=None, damping_trend=None, *,
                         optimized=True, remove_bias=False, start_params=None,
                         method=None, minimize_kwargs=None, use_brute=True,
                         use_boxcox=None, use_basinhopping=None, initial_level=None,
                         initial_trend=None)
ETSModel(endog, error="add", trend=None, damped_trend=False, seasonal=None,
         seasonal_periods=None, initialization_method="estimated",
         initial_level=None, initial_trend=None, initial_seasonal=None,
         bounds=None, dates=None, freq=None, missing="none")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `trend` | `"add"`, `"mul"`, None | None | Trend component |
| `damped_trend` | bool | False | Damped trend (formerly `damped=`) |
| `seasonal` | `"add"`, `"mul"`, None | None | Seasonal component |
| `seasonal_periods` | int | None | Season length (e.g. 12 for monthly data) |
| `error` (ETSModel) | `"add"` or `"mul"` | `"add"` | Error type; ETSModel provides likelihood-based prediction intervals |

`ExponentialSmoothing` results: `forecast(steps)`, `fittedvalues`, `params`, `sse`, `aic`. `ETSModel` results additionally offer `get_prediction(start, end).summary_frame()` with intervals.

```python
hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
print(hw.forecast(12).round(2).head())

ets = ETSModel(train, error="add", trend="add", seasonal="add",
               damped_trend=True, seasonal_periods=12).fit(disp=False)
print(ets.get_prediction(start=len(train), end=len(train) + 11).summary_frame().head())
```

#### VAR and Granger causality

```python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, coint

VAR(endog, exog=None, dates=None, freq=None, missing="none")
VAR.select_order(maxlags=None, trend="c")
VAR.fit(maxlags=None, method="ols", ic=None, trend="c", verbose=False)
grangercausalitytests(x, maxlag, addconst=True, verbose=None)
coint(y0, y1, trend="c", method="aeg", maxlag=None, autolag="aic", return_results=None)
```

VAR results: `summary()`, `k_ar` (selected lag order), `forecast(y, steps)` (y = last `k_ar` observations as an array), `forecast_interval(y, steps, alpha=0.05)`, `irf(periods=10)` (impulse responses, `.plot()`), `fevd(periods)`, `test_causality(caused, causing, kind="f")`, `test_whiteness(nlags)`, `plot_forecast(steps)`.

`grangercausalitytests` tests whether the **second** column Granger-causes the **first**. Version note: the `verbose` argument is deprecated in 0.14; the function returns a dict keyed by lag with test results.

```python
macro = sm.datasets.macrodata.load_pandas().data
data = np.log(macro[["realgdp", "realcons", "realinv"]]).diff().dropna()
model = VAR(data)
print(model.select_order(maxlags=8).summary())
res = model.fit(maxlags=8, ic="aic")
print(res.k_ar)
fc = res.forecast(data.values[-res.k_ar:], steps=4)
print(fc)

gc = grangercausalitytests(data[["realcons", "realgdp"]], maxlag=2)
print(gc[2][0]["ssr_ftest"])         # (F, p-value, df_denom, df_num)
```

#### UnobservedComponents

```python
sm.tsa.UnobservedComponents(endog, level=False, trend=False, seasonal=None,
                            freq_seasonal=None, cycle=False, autoregressive=None,
                            exog=None, irregular=False, stochastic_level=False,
                            stochastic_trend=False, stochastic_seasonal=True, ...)
```

Structural time series models (local level, local linear trend, seasonal, cycles). `level` also accepts string specs such as `"local level"`, `"local linear trend"`, `"smooth trend"`.

```python
uc = sm.tsa.UnobservedComponents(train, level="local linear trend", seasonal=12)
uc_res = uc.fit(disp=False)
print(uc_res.forecast(6))
```

### Nonparametric methods

```python
sm.nonparametric.lowess(endog, exog, frac=2/3, it=3, delta=0.0, xvals=None,
                        is_sorted=False, missing="drop", return_sorted=True)
sm.nonparametric.KDEUnivariate(endog)
KDEUnivariate.fit(kernel="gau", bw="normal_reference", fft=True, weights=None,
                  gridsize=None, adjust=1, cut=3, clip=(-np.inf, np.inf))
```

Note the argument order of `lowess`: **endog (y) first, then exog (x)**. With `return_sorted=True` it returns an (n, 2) array of sorted x and fitted y.

```python
rng = np.random.default_rng(8)
x = np.sort(rng.uniform(0, 10, 300))
y = np.sin(x) + rng.normal(scale=0.3, size=300)
smoothed = sm.nonparametric.lowess(y, x, frac=0.2)
kde = sm.nonparametric.KDEUnivariate(y)
kde.fit()
print(smoothed[:3], kde.support[:3], kde.density[:3])
```

### Graphics (statsmodels.graphics)

| Function | Purpose |
|---|---|
| `sm.qqplot(data, dist=stats.norm, fit=False, line=None, ax=None)` | Q-Q plot; `line="45"`, `"s"`, `"r"`, `"q"` adds a reference line |
| `sm.graphics.plot_regress_exog(results, exog_idx, fig=None)` | 4-panel diagnostics for one regressor |
| `sm.graphics.plot_partregress_grid(results, exog_idx=None, fig=None)` | Partial regression (added-variable) plots |
| `sm.graphics.plot_ccpr_grid(results, exog_idx=None, fig=None)` | Component-plus-residual plots |
| `sm.graphics.influence_plot(results, external=True, alpha=0.05, criterion="cooks", size=48, ax=None)` | Leverage vs studentized residuals sized by Cook's distance |
| `sm.graphics.plot_leverage_resid2(results, ax=None)` | Leverage vs squared residual |
| `sm.graphics.plot_fit(results, exog_idx, ax=None)` | Fitted vs observed against one regressor |
| `sm.graphics.interaction_plot(x, trace, response, ax=None)` | Interaction plot for factorial designs |
| `sm.graphics.tsa.plot_acf` / `plot_pacf` | Correlograms |
| `sm.graphics.tsa.month_plot(x)` / `quarter_plot(x)` | Seasonal subseries plots |
| `results.plot_diagnostics()` | State space model residual diagnostics (SARIMAX, ARIMA, UC) |

All return Matplotlib `Figure` objects (and most accept `ax=`).

```python
import matplotlib.pyplot as plt

longley = sm.datasets.longley.load_pandas()
res = sm.OLS(longley.endog, sm.add_constant(longley.exog)).fit()
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
sm.qqplot(res.resid, line="s", ax=axs[0])
sm.graphics.influence_plot(res, ax=axs[1], criterion="cooks")
fig.savefig("ols_diagnostics.png")
```

### Linear hypothesis tests on results

```python
results.t_test(r_matrix, cov_p=None, use_t=None)
results.f_test(r_matrix, cov_p=None, invcov=None)
results.wald_test(r_matrix, cov_p=None, invcov=None, use_f=None, df_constraints=None, scalar=None)
results.compare_f_test(restricted)          # OLS: F-test vs a nested restricted model
results.compare_lr_test(restricted, large_sample=False)
```

Hypotheses can be strings using parameter names:

```python
df = pd.DataFrame(np.random.default_rng(9).normal(size=(200, 3)), columns=["x1", "x2", "x3"])
df["y"] = 1 + 2 * df["x1"] + 2 * df["x2"] + np.random.default_rng(10).normal(size=200)
res = smf.ols("y ~ x1 + x2 + x3", data=df).fit()
print(res.t_test("x1 = x2"))                # equal effects?
print(res.f_test("x2 = 0, x3 = 0"))         # joint significance
print(res.wald_test("x3 = 0", scalar=True))
```

### Prediction with intervals

```python
results.predict(exog=None, transform=True, *args, **kwargs)
results.get_prediction(exog=None, transform=True, weights=None, row_labels=None, **kwargs)
PredictionResults.summary_frame(alpha=0.05)
```

For OLS, `summary_frame` returns `mean`, `mean_se`, `mean_ci_lower`, `mean_ci_upper` (confidence interval for the mean response) and `obs_ci_lower`, `obs_ci_upper` (prediction interval for a new observation). With formula models, `exog` can be a DataFrame with the original column names (`transform=True`).

```python
new = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.5, -0.5], "x3": [0.0, 0.0]})
print(res.get_prediction(new).summary_frame(alpha=0.05).round(3))
```

### Persistence

```python
results.save(fname, remove_data=False)
sm.load(fname)                       # alias of statsmodels.iolib.smpickle.load_pickle
results.remove_data()                # shrink the object before saving
```

```python
res.save("ols_model.pickle", remove_data=True)
loaded = sm.load("ols_model.pickle")
print(loaded.params)
```

## Tutorials

### Tutorial 1: Linear regression with inference and diagnostics

Goal: estimate how house characteristics relate to price, with valid standard errors and checked assumptions.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Simulate a realistic dataset (replace with your own DataFrame)
rng = np.random.default_rng(42)
n = 600
df = pd.DataFrame({
    "sqft": rng.normal(1800, 500, n).clip(500, None),
    "age": rng.integers(0, 80, n),
    "bedrooms": rng.integers(1, 6, n),
    "hood": rng.choice(["north", "south", "east"], n, p=[0.4, 0.4, 0.2]),
})
hood_effect = df["hood"].map({"north": 0.15, "south": 0.0, "east": -0.10})
log_price = (11 + 0.00045 * df["sqft"] - 0.003 * df["age"] + 0.02 * df["bedrooms"]
             + hood_effect + rng.normal(0, 0.15 + 0.00005 * df["sqft"], n))
df["price"] = np.exp(log_price)

# 2. Fit a log-linear model with a chosen reference category
formula = "np.log(price) ~ sqft + age + bedrooms + C(hood, Treatment(reference='south'))"
ols = smf.ols(formula, data=df).fit()
print(ols.summary())

# 3. Check heteroskedasticity; refit with robust standard errors if needed
bp_lm, bp_p, _, _ = het_breuschpagan(ols.resid, ols.model.exog)
print(f"Breusch-Pagan p = {bp_p:.4f}")
if bp_p < 0.05:
    ols = smf.ols(formula, data=df).fit(cov_type="HC3")
    print("Refit with HC3 robust standard errors")

# 4. Functional form and multicollinearity
print(linear_reset(ols, power=2, use_f=True))
X = ols.model.exog
vif = pd.Series([variance_inflation_factor(X, i) for i in range(1, X.shape[1])],
                index=ols.model.exog_names[1:])
print(vif.round(2))

# 5. Interpret: coefficients on log scale -> percent changes
pct = (np.exp(ols.params) - 1) * 100
ci = (np.exp(ols.conf_int()) - 1) * 100
table = pd.DataFrame({"pct_change": pct, "ci_low": ci[0], "ci_high": ci[1],
                      "p": ols.pvalues}).round(3)
print(table)
print(f"+100 sqft -> {100 * (np.exp(100 * ols.params['sqft']) - 1):.1f}% price change")

# 6. Residual diagnostics plots
infl = ols.get_influence()
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
axs[0].scatter(ols.fittedvalues, ols.resid, s=8, alpha=0.5)
axs[0].axhline(0, color="k")
axs[0].set(xlabel="fitted", ylabel="residual", title="Residuals vs fitted")
sm.qqplot(infl.resid_studentized_internal, line="45", ax=axs[1])
axs[1].set_title("Q-Q of studentized residuals")
sm.graphics.influence_plot(ols, ax=axs[2], size=24)
fig.savefig("ols_tutorial_diagnostics.png", dpi=120)

# 7. Predictions with confidence and prediction intervals
new = pd.DataFrame({"sqft": [1500, 2500], "age": [10, 40], "bedrooms": [3, 4],
                    "hood": ["north", "east"]})
pred = ols.get_prediction(new).summary_frame(alpha=0.05)
print(np.exp(pred[["mean", "obs_ci_lower", "obs_ci_upper"]]).round(0))
```

Notes: the formula handles the log transform and categorical coding, and `get_prediction` re-applies both to `new`. Exponentiating a log-scale prediction gives the conditional *median*, not the mean; apply a smearing correction if you need the mean price.

### Tutorial 2: Logistic regression for interpretation (vs scikit-learn)

Goal: model the probability of an extramarital affair with the built-in Fair (1978) dataset and interpret effects.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Load data and create a binary target
fair = sm.datasets.fair.load_pandas().data
fair["any_affair"] = (fair["affairs"] > 0).astype(int)
print(fair["any_affair"].mean().round(3))     # base rate

# 2. Train/test split to also check predictive performance
train, test = train_test_split(fair, test_size=0.25, random_state=0,
                               stratify=fair["any_affair"])

# 3. Fit with a formula: categorical occupation, continuous others
formula = ("any_affair ~ rate_marriage + age + yrs_married + children"
           " + religious + educ + C(occupation)")
logit = smf.logit(formula, data=train).fit(disp=0)
print(logit.summary())

# 4. Odds ratios with confidence intervals
or_table = np.exp(pd.concat([logit.params, logit.conf_int()], axis=1))
or_table.columns = ["odds_ratio", "ci_low", "ci_high"]
print(or_table.round(3))

# 5. Average marginal effects: change in probability per unit change
print(logit.get_margeff(at="overall", method="dydx").summary())

# 6. Likelihood-ratio test: does occupation matter jointly?
reduced = smf.logit("any_affair ~ rate_marriage + age + yrs_married + children"
                    " + religious + educ", data=train).fit(disp=0)
lr_stat = 2 * (logit.llf - reduced.llf)
df_diff = logit.df_model - reduced.df_model
from scipy import stats
print(f"LR = {lr_stat:.2f}, df = {df_diff:.0f}, p = {stats.chi2.sf(lr_stat, df_diff):.4f}")

# 7. Out-of-sample performance
p_test = logit.predict(test)
print("statsmodels AUC:", round(roc_auc_score(test["any_affair"], p_test), 3))

# 8. The same model in scikit-learn: note penalty=None to match the MLE
X_cols = ["rate_marriage", "age", "yrs_married", "children", "religious", "educ"]
X_tr = pd.get_dummies(train[X_cols + ["occupation"]], columns=["occupation"],
                      drop_first=True, dtype=float)
X_te = pd.get_dummies(test[X_cols + ["occupation"]], columns=["occupation"],
                      drop_first=True, dtype=float)
sk = LogisticRegression(penalty=None, max_iter=5000).fit(X_tr, train["any_affair"])
print("sklearn AUC:", round(roc_auc_score(test["any_affair"], sk.predict_proba(X_te)[:, 1]), 3))
sk_coef = pd.Series(sk.coef_[0], index=X_tr.columns)
print(pd.DataFrame({"statsmodels": logit.params[X_cols],
                    "sklearn": sk_coef[X_cols]}).round(3))
```

The coefficient vectors agree (up to optimizer tolerance and column order); the value statsmodels adds is the inference layer: standard errors, likelihood-ratio tests, odds-ratio intervals, and marginal effects. Note that scikit-learn's default `LogisticRegression` applies L2 regularization (`C=1.0`), so its coefficients differ unless you set `penalty=None` (scikit-learn 1.2+; older versions use `penalty="none"`).

### Tutorial 3: Seasonal forecasting with SARIMAX and ETS

Goal: forecast monthly atmospheric CO2 two years ahead, validate on a holdout, and compare two model families.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

# 1. Load weekly data, aggregate to monthly start frequency, fill gaps
co2 = sm.datasets.co2.load_pandas().data["co2"]
y = co2.resample("MS").mean().interpolate()
y = y["1965":]
train, test = y[:-24], y[-24:]
print(train.index.freq)            # <MonthBegin>, needed for clean forecasting

# 2. Decompose to understand trend and seasonality
stl = STL(train, period=12, robust=True).fit()
stl.plot().savefig("co2_stl.png", dpi=110)

# 3. Stationarity: original vs seasonally and first differenced
for name, series in [("raw", train), ("diff(1)+diff(12)", train.diff().diff(12).dropna())]:
    adf_p = adfuller(series)[1]
    kpss_p = kpss(series, regression="c", nlags="auto")[1]
    print(f"{name:18s} ADF p={adf_p:.3f}  KPSS p={kpss_p:.3f}")

# 4. Identify orders from ACF/PACF of the differenced series
d = train.diff().diff(12).dropna()
fig, axs = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
plot_acf(d, lags=36, ax=axs[0])
plot_pacf(d, lags=36, ax=axs[1], method="ywm")
fig.savefig("co2_acf_pacf.png", dpi=110)

# 5. Small grid search on AIC (keep it small: each fit takes a moment)
best = None
for p in (0, 1, 2):
    for q in (0, 1, 2):
        res = SARIMAX(train, order=(p, 1, q), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
        if best is None or res.aic < best[0]:
            best = (res.aic, (p, 1, q), res)
print("best order:", best[1], "AIC:", round(best[0], 1))
sarima = best[2]

# 6. Residual checks: no remaining autocorrelation
print(acorr_ljungbox(sarima.resid[13:], lags=[12, 24]))
sarima.plot_diagnostics(figsize=(10, 7)).savefig("co2_sarima_diag.png", dpi=110)

# 7. Forecast with intervals and evaluate on the holdout
fc = sarima.get_forecast(steps=len(test)).summary_frame(alpha=0.05)
mae_sarima = (fc["mean"] - test).abs().mean()

ets = ETSModel(train, error="add", trend="add", damped_trend=True,
               seasonal="add", seasonal_periods=12).fit(disp=False)
ets_fc = ets.get_prediction(start=test.index[0], end=test.index[-1]).summary_frame(alpha=0.05)
mae_ets = (ets_fc["mean"] - test).abs().mean()
print(f"MAE SARIMA = {mae_sarima:.3f} ppm, MAE ETS = {mae_ets:.3f} ppm")

# 8. Plot
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
ax.plot(train["2010":], label="train")
ax.plot(test, label="actual", color="k")
ax.plot(fc["mean"], label="SARIMA")
ax.fill_between(fc.index, fc["mean_ci_lower"], fc["mean_ci_upper"], alpha=0.25)
ax.plot(ets_fc["mean"], label="ETS", ls="--")
ax.set(ylabel="CO2 (ppm)", title="Holdout forecast")
ax.legend()
fig.savefig("co2_forecast.png", dpi=110)

# 9. Refit on all data and forecast the real future
final = SARIMAX(y, order=best[1], seasonal_order=(0, 1, 1, 12)).fit(disp=False)
print(final.forecast(steps=12).round(2))
```

Key practices: give pandas a regular frequency (`resample("MS")`) so forecasts carry dates; use both ADF and KPSS; choose orders by AIC within a small, sensible grid; confirm white-noise residuals with Ljung-Box; and always evaluate on a holdout before trusting AIC alone.

### Tutorial 4: Analyzing an A/B test of a new recommendation model

Goal: decide whether a new model improves conversion, with a power analysis before the test and multiple-metric correction after.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (confint_proportions_2indep,
                                          proportion_effectsize, proportions_ztest)
from statsmodels.stats.weightstats import ttest_ind

# 1. Before the test: how many users per arm to detect 5.0% -> 5.5% conversion?
es = proportion_effectsize(0.055, 0.050)
n_per_arm = NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.8,
                                         alternative="two-sided")
print(f"required users per arm: {int(np.ceil(n_per_arm))}")

# 2. Simulated experiment results (replace with your logged data)
rng = np.random.default_rng(0)
n = 32000
df = pd.DataFrame({"arm": rng.choice(["control", "treatment"], size=n)})
df["platform"] = rng.choice(["web", "ios", "android"], size=n, p=[0.5, 0.3, 0.2])
base = np.where(df["arm"] == "treatment", 0.056, 0.050)
df["converted"] = rng.binomial(1, base)
df["revenue"] = np.where(df["converted"] == 1, rng.gamma(2.0, 20.0, n), 0.0)
df["session_min"] = rng.gamma(3.0, 4.0, n) + 0.2 * (df["arm"] == "treatment")

# 3. Primary metric: conversion, two-proportion z-test and CI for the difference
g = df.groupby("arm")["converted"].agg(["sum", "count"])
stat, p_conv = proportions_ztest(g["sum"].values, g["count"].values)
low, upp = confint_proportions_2indep(g.loc["treatment", "sum"], g.loc["treatment", "count"],
                                      g.loc["control", "sum"], g.loc["control", "count"],
                                      method="newcomb", compare="diff")
print(g.assign(rate=g["sum"] / g["count"]))
print(f"z = {stat:.2f}, p = {p_conv:.4f}, diff 95% CI = [{low:.4f}, {upp:.4f}]")

# 4. Secondary metrics: Welch t-tests on revenue and session length
t_rev, p_rev, _ = ttest_ind(df.loc[df.arm == "treatment", "revenue"],
                            df.loc[df.arm == "control", "revenue"], usevar="unequal")
t_ses, p_ses, _ = ttest_ind(df.loc[df.arm == "treatment", "session_min"],
                            df.loc[df.arm == "control", "session_min"], usevar="unequal")

# 5. Correct for testing several metrics
names = ["conversion", "revenue", "session_min"]
reject, p_adj, _, _ = multipletests([p_conv, p_rev, p_ses], alpha=0.05, method="holm")
print(pd.DataFrame({"metric": names, "p_raw": [p_conv, p_rev, p_ses],
                    "p_holm": p_adj, "significant": reject}).round(4))

# 6. Regression adjustment and heterogeneity by platform
lpm = smf.ols("converted ~ C(arm) * C(platform)", data=df).fit(cov_type="HC1")
print(lpm.summary().tables[1])
logit = smf.logit("converted ~ C(arm) + C(platform)", data=df).fit(disp=0)
print(logit.get_margeff().summary())
```

Interpretation checklist: the primary metric decision uses the pre-registered test (step 3); secondary metrics are judged after Holm correction (step 5); the interaction terms in step 6 are exploratory and should not override the primary result.

### Tutorial 5: Count data with exposure (Poisson vs Negative Binomial GLM)

Goal: model insurance claim counts where policies have different durations.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

rng = np.random.default_rng(1)
n = 4000
df = pd.DataFrame({
    "driver_age": rng.integers(18, 80, n),
    "region": rng.choice(["urban", "rural"], n, p=[0.6, 0.4]),
    "years": rng.uniform(0.2, 3.0, n),                 # exposure: policy-years
})
rate = np.exp(-2.0 + 0.6 * (df["driver_age"] < 25) + 0.3 * (df["region"] == "urban"))
mu = rate * df["years"]
df["claims"] = rng.negative_binomial(1.5, 1.5 / (1.5 + mu))

df["young"] = (df["driver_age"] < 25).astype(int)
formula = "claims ~ young + C(region)"

# 1. Poisson GLM with exposure (log(years) enters as an offset)
pois = smf.glm(formula, data=df, family=sm.families.Poisson(),
               exposure=df["years"]).fit()
print(pois.summary().tables[1])

# 2. Check overdispersion: Pearson chi2 / df_resid should be near 1 for Poisson
print("dispersion:", round(pois.pearson_chi2 / pois.df_resid, 2))

# 3. Negative binomial with estimated alpha (discrete model) handles overdispersion
nb = smf.negativebinomial(formula, data=df, exposure=df["years"]).fit(disp=0)
print(nb.summary().tables[1])
print("AIC Poisson:", round(pois.aic, 1), " AIC NB:", round(nb.aic, 1))

# 4. Rate ratios
print(np.exp(nb.params.drop("alpha")).round(3))

# 5. Expected claims for a new policy, 1.5 years, young urban driver
new = pd.DataFrame({"young": [1], "region": ["urban"]})
print(nb.predict(new, exposure=np.array([1.5])))
```

Poisson standard errors are too small when data are overdispersed; the negative binomial model (or `cov_type="HC0"` / a quasi-Poisson `scale="X2"` fit) gives honest uncertainty.

## Performance & Best Practices

### Modeling practice

- **Always add a constant** with the array API (`sm.add_constant`), or use formulas.
- **Pick standard errors deliberately**: `HC3` for cross-sectional heteroskedasticity, `HAC` for time series regressions, `cluster` for grouped data. Report which one you used.
- **Check assumptions** before trusting p-values: residual plots, Breusch-Pagan, Ljung-Box, Q-Q plots, influence measures.
- **Don't p-hack**: correct for multiple comparisons (`multipletests`) and pre-specify primary metrics in experiments.
- **Use likelihood-ratio or F tests** for nested models (`compare_lr_test`, `anova_lm(reduced, full)`) instead of eyeballing individual p-values.
- **Scale predictors** when coefficients span orders of magnitude; a large "condition number" warning in `summary()` often just reflects units.
- **Distinguish confidence and prediction intervals**: `mean_ci_*` vs `obs_ci_*` in `summary_frame`.

### Speed and memory

| Situation | Technique |
|---|---|
| Large n OLS | Use `fit(method="qr")` or `"pinv"` (default) - both fine; avoid repeated `summary()` calls in loops |
| Many repeated fits (bootstraps, rolling) | Use `RollingOLS`, or the array API with pre-built NumPy matrices (formula parsing has overhead) |
| Huge formula models with many dummies | Build the design matrix once with `patsy.dmatrices` and reuse it |
| Pickled results too large | `results.remove_data()` or `save(..., remove_data=True)` |
| Slow SARIMAX | `simple_differencing=True`, fewer seasonal terms, `low_memory=True` in `fit`, or `ARIMA(..., ).fit(method="innovations_mle")` for non-seasonal models |
| Many series to forecast | Parallelize with joblib, or switch to statsforecast for bulk AutoARIMA/ETS |
| Long-running discrete model fits | Try `method="bfgs"` or `"lbfgs"`, provide `start_params`, increase `maxiter` |
| Grid search over ARIMA orders | Keep grids small; compare by AIC and validate on a holdout |

### Time series hygiene

- Give the index a frequency (`series.asfreq("MS")` or `resample`) to avoid frequency warnings and to get dated forecasts.
- Difference via the model (`d`, `D` in the order) rather than manually, so forecasts come back on the original scale.
- Use `results.append(new_obs)` (or `apply`) to update a fitted state space model with new data without re-estimating parameters.
- Use expanding-window backtests rather than a single split for robust evaluation.

## Common Errors & Troubleshooting

| Error / warning | Cause | Fix |
|---|---|---|
| `MissingDataError: exog contains inf or nans` | NaN/inf in data with `missing="none"` | Clean data, or pass `missing="drop"` |
| `ValueError: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).` | Non-numeric (string/object or nullable) columns in array API | Convert to numeric, one-hot encode (`pd.get_dummies(..., dtype=float)`), or use formulas with `C()` |
| `LinAlgError: Singular matrix` | Perfect multicollinearity (e.g. all dummy levels plus constant) | Drop a level (`drop_first=True`), remove duplicate columns |
| Summary note: `The condition number is large, ... This might indicate that there are strong multicollinearity or other numerical problems.` | Collinearity or very different scales | Check VIF, standardize predictors |
| Summary note: `The smallest eigenvalue is ... This might indicate that there are strong multicollinearity problems or that the design matrix is singular.` | Near-singular design | Remove redundant variables |
| R-squared suspiciously high, no `const` in params | Forgot `sm.add_constant` | Add the constant (uncentered R-squared is reported otherwise) |
| `PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified` (0.14; `PerfectSeparationError` in older versions) | A predictor perfectly predicts the binary outcome | Remove/merge the feature, use `fit_regularized`, or Firth-type penalization |
| `ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals` | Optimizer hit `maxiter` | Increase `maxiter`, change `method`, scale data, give `start_params` |
| `HessianInversionWarning: Inverting hessian failed, no bse or cov_params available` | Non-identified or degenerate model | Simplify the model, check for constant columns |
| `ValueWarning: No frequency information was provided, so inferred frequency MS will be used.` | Date index without `freq` | `y = y.asfreq("MS")` |
| `ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.` | Irregular date index | Resample to a regular frequency |
| `UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.` | Data not stationary for the chosen order | Usually harmless; consider differencing (`d=1`) |
| `ValueError: In models with integration (d > 0) or seasonal integration (D > 0), trend terms of lower order than d + D cannot be (as they would be eliminated due to the differencing operation)...` | `trend="c"` with `d >= 1` in `ARIMA` | Use `trend="t"` for drift with `d=1`, or `trend="n"` |
| `NotImplementedError: statsmodels.tsa.arima_model.ARMA and statsmodels.tsa.arima_model.ARIMA have been removed in favor of statsmodels.tsa.arima.model.ARIMA ...` | Old tutorial code | `from statsmodels.tsa.arima.model import ARIMA`; `ARIMA(y, order=(p, d, q)).fit()` |
| `TypeError: seasonal_decompose() got an unexpected keyword argument 'freq'` | Renamed | `period=` |
| `TypeError: __init__() got an unexpected keyword argument 'damped'` (ExponentialSmoothing) | Renamed | `damped_trend=True` |
| `TypeError: acf() got an unexpected keyword argument 'unbiased'` | Renamed | `adjusted=True` |
| `PatsyError: Error evaluating factor: NameError: name 'x' is not defined` | Column name typo, or name with spaces/dots | Fix the name or use `Q("my col")` |
| `PatsyError` when calling `predict` on a formula model with a NumPy array | patsy needs the named columns used in the formula | Pass a DataFrame with the original column names |
| `PatsyError: categorical data cannot be >1-dimensional` or new category at predict time | Unseen level in `C(var)` | Ensure new data levels exist in training data |
| `ValueError: shapes (n,k) and (m,) not aligned` in `predict` | Array API predict without the constant column | `results.predict(sm.add_constant(X_new, has_constant="add"))` |
| `ValueError: endog must be in the unit interval.` (Logit) | Target not 0/1 | Encode target as 0/1 integers |
| `FutureWarning` about `verbose` in `grangercausalitytests` | Deprecated in 0.14 | Drop the argument and read the returned dict |
| `FutureWarning` about link class names (`links.log`) | Lowercase link aliases deprecated in 0.14 | `sm.families.links.Log()` |
| Very slow `MixedLM` or convergence warnings | Complex random-effects structure | Simplify `re_formula`, try `method=["lbfgs"]`, scale covariates |

## Interoperability

### pandas

statsmodels is pandas-native: DataFrames in, labeled Series/DataFrames out (`params`, `bse`, `conf_int()`, `summary_frame()`). Date-indexed Series drive time-series models, so forecasts come back with dates. `results.summary2().tables[1]` returns the coefficient table as a DataFrame, convenient for reports.

```python
import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame({"y": [1.0, 2.1, 2.9, 4.2, 4.8], "x": [1, 2, 3, 4, 5]})
coef_table = smf.ols("y ~ x", df).fit().summary2().tables[1]
print(coef_table)          # DataFrame: Coef., Std.Err., t, P>|t|, [0.025, 0.975]
```

### NumPy and patsy

The array API accepts any NumPy array. `patsy.dmatrices("y ~ x + C(g)", df, return_type="dataframe")` builds the same design matrices the formula API uses, which you can pass to statsmodels or scikit-learn.

### scikit-learn

statsmodels estimators do not implement the scikit-learn estimator API, so they cannot go directly into `Pipeline` or `GridSearchCV`. Common integration patterns:

- Use scikit-learn for preprocessing and splitting, then fit statsmodels on the transformed NumPy arrays (add a constant).
- Write a thin wrapper class with `fit`/`predict` if you need cross-validation of a statsmodels model:

```python
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score

class SMWrapper(BaseEstimator, RegressorMixin):
    """Minimal scikit-learn wrapper around a statsmodels model class."""
    def __init__(self, model_class=sm.OLS, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")
        self.results_ = self.model_class(y, X).fit()
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")
        return self.results_.predict(X)

X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=0)
print(cross_val_score(SMWrapper(), X, y, cv=5, scoring="r2").round(3))
```

- Equivalences: `LinearRegression` == `sm.OLS` (point estimates); `LogisticRegression(penalty=None)` == `sm.Logit`; `PoissonRegressor(alpha=0)` == `GLM(family=Poisson())`; `QuantileRegressor(alpha=0)` == `QuantReg`.

### Matplotlib and seaborn

All `statsmodels.graphics` functions return Matplotlib figures and most accept `ax=` so they can be placed into your own layouts. seaborn's `regplot(lowess=True, robust=True, logistic=True)` and `residplot` call statsmodels internally.

### Forecasting ecosystem

- **pmdarima** wraps statsmodels SARIMAX with `auto_arima` order search.
- **sktime** and **Darts** provide adapters for statsmodels ARIMA, ETS, and Theta models inside a unified forecasting API.
- **statsforecast** reimplements ARIMA/ETS for speed on many series; results are comparable to statsmodels.
- **linearmodels** extends statsmodels-style APIs to panel data and IV estimation.

### Exporting results

- `results.summary().as_latex()`, `.as_html()`, `.as_text()` for reports.
- `from statsmodels.iolib.summary2 import summary_col` builds side-by-side regression tables for several models.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

rng = np.random.default_rng(0)
df = pd.DataFrame({"x1": rng.normal(size=100), "x2": rng.normal(size=100)})
df["y"] = 1 + df["x1"] - 0.5 * df["x2"] + rng.normal(size=100)
m1 = smf.ols("y ~ x1", df).fit()
m2 = smf.ols("y ~ x1 + x2", df).fit()
print(summary_col([m1, m2], stars=True, model_names=["(1)", "(2)"],
                  info_dict={"N": lambda r: f"{int(r.nobs)}"}))
```

## Cheat Sheet

### Setup and regression

| Task | Code |
|---|---|
| Imports | `import statsmodels.api as sm`; `import statsmodels.formula.api as smf` |
| Add intercept | `X = sm.add_constant(X)` |
| OLS (arrays) | `sm.OLS(y, X).fit()` |
| OLS (formula) | `smf.ols("y ~ x1 + C(g)", data=df).fit()` |
| Robust SEs | `.fit(cov_type="HC3")` |
| Newey-West SEs | `.fit(cov_type="HAC", cov_kwds={"maxlags": 4})` |
| Clustered SEs | `.fit(cov_type="cluster", cov_kwds={"groups": df["g"]})` |
| Weighted LS | `sm.WLS(y, X, weights=w).fit()` |
| Quantile regression | `smf.quantreg("y ~ x", df).fit(q=0.9)` |
| Robust regression | `sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()` |
| Lasso / ridge | `sm.OLS(y, X).fit_regularized(alpha=0.1, L1_wt=1.0)` |
| Summary | `res.summary()` / `res.summary2()` |
| Coefficients, SEs, p | `res.params`, `res.bse`, `res.pvalues` |
| Confidence intervals | `res.conf_int(alpha=0.05)` |
| Prediction intervals | `res.get_prediction(new).summary_frame(alpha=0.05)` |
| Linear hypothesis | `res.f_test("x1 = x2")` |
| Compare nested | `anova_lm(reduced, full)` / `full.compare_lr_test(reduced)` |

### GLM and discrete

| Task | Code |
|---|---|
| Logistic | `smf.logit("y ~ x", df).fit(disp=0)` |
| Odds ratios | `np.exp(res.params)` |
| Marginal effects | `res.get_margeff().summary()` |
| Probit | `smf.probit("y ~ x", df).fit()` |
| Multinomial | `sm.MNLogit(y, X).fit()` |
| Ordinal | `OrderedModel(y, X, distr="logit").fit(method="bfgs")` |
| Poisson GLM with exposure | `smf.glm("n ~ x", df, family=sm.families.Poisson(), exposure=df["t"]).fit()` |
| Negative binomial | `smf.negativebinomial("n ~ x", df).fit()` |
| Gamma log-link | `sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log())).fit()` |
| Mixed model | `smf.mixedlm("y ~ x", df, groups=df["g"], re_formula="~x").fit()` |
| GEE | `smf.gee("y ~ x", "g", df, cov_struct=sm.cov_struct.Exchangeable()).fit()` |

### Tests

| Task | Code |
|---|---|
| Welch t-test | `ttest_ind(a, b, usevar="unequal")` |
| Two proportions | `proportions_ztest([s1, s2], [n1, n2])` |
| Proportion CI | `proportion_confint(s, n, method="wilson")` |
| ANOVA table | `anova_lm(smf.ols("y ~ C(g)", df).fit(), typ=2)` |
| Tukey HSD | `pairwise_tukeyhsd(df["y"], df["g"])` |
| FDR correction | `multipletests(pvals, method="fdr_bh")` |
| Sample size | `TTestIndPower().solve_power(effect_size=0.3, alpha=0.05, power=0.8)` |
| Heteroskedasticity | `het_breuschpagan(res.resid, res.model.exog)` |
| Autocorrelation | `durbin_watson(res.resid)`; `acorr_ljungbox(res.resid, lags=[10])` |
| Normality | `jarque_bera(res.resid)`; `sm.qqplot(res.resid, line="s")` |
| VIF | `variance_inflation_factor(X.values, i)` |
| Influence | `res.get_influence().summary_frame()` |

### Time series

| Task | Code |
|---|---|
| Unit root (ADF) | `adfuller(y)[1]` (p-value) |
| Stationarity (KPSS) | `kpss(y, nlags="auto")[1]` |
| ACF / PACF plots | `plot_acf(y, lags=40)`; `plot_pacf(y, lags=40, method="ywm")` |
| Decompose | `STL(y, period=12).fit().plot()` |
| ARIMA | `ARIMA(y, order=(1, 1, 1)).fit()` |
| Seasonal ARIMA | `SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)` |
| Forecast with CI | `res.get_forecast(12).summary_frame()` |
| Point forecast | `res.forecast(steps=12)` |
| Holt-Winters | `ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit()` |
| ETS with intervals | `ETSModel(y, error="add", trend="add").fit().get_prediction(...)` |
| Autoregression | `AutoReg(y, lags=3).fit()` |
| VAR | `VAR(df).fit(maxlags=8, ic="aic")` |
| Granger causality | `grangercausalitytests(df[["y", "x"]], maxlag=4)` |
| Cointegration | `coint(y1, y2)` |
| Model diagnostics | `res.plot_diagnostics(figsize=(10, 8))` |
| Update with new data | `res.append(new_y)` |

## Further Resources

- Official documentation: https://www.statsmodels.org/stable/
- Getting started: https://www.statsmodels.org/stable/gettingstarted.html
- User guide: https://www.statsmodels.org/stable/user-guide.html
- API reference: https://www.statsmodels.org/stable/api.html
- Examples gallery: https://www.statsmodels.org/stable/examples/index.html
- Release notes: https://www.statsmodels.org/stable/release/index.html
- Time series analysis guide: https://www.statsmodels.org/stable/tsa.html
- Formula guide (patsy): https://www.statsmodels.org/stable/example_formulas.html
- patsy documentation: https://patsy.readthedocs.io/
- GitHub repository: https://github.com/statsmodels/statsmodels
- Mailing list / discussion group: https://groups.google.com/g/pystatsmodels
- Paper: Seabold, S. and Perktold, J. (2010). "statsmodels: Econometric and statistical modeling with python." Proceedings of the 9th Python in Science Conference. https://doi.org/10.25080/Majora-92bf1922-011
- Book: Rob J. Hyndman and George Athanasopoulos, "Forecasting: Principles and Practice" (free online; concepts map directly to statsmodels ETS/ARIMA): https://otexts.com/fpp3/
- Book: Wes McKinney, "Python for Data Analysis", 3rd ed., chapter "Introduction to Modeling Libraries in Python" covers statsmodels (free online): https://wesmckinney.com/book/
- Course material: Kevin Sheppard's "Python for Econometrics" notes: https://www.kevinsheppard.com/teaching/python/notes/
