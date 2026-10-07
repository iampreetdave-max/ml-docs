# scikit-learn

> The standard toolkit for classical machine learning in Python: consistent estimators, pipelines, model selection and metrics.

scikit-learn is the library most Python practitioners reach for first when they need a classifier, a regressor, a clustering algorithm, a preprocessing step or a cross-validation loop. Every algorithm follows the same `fit` / `predict` / `transform` contract, which means you can swap a logistic regression for a gradient-boosted model by changing one line, chain preprocessing and modelling into a single `Pipeline`, and tune the whole thing with a grid search.

Covers scikit-learn 1.5+ (examples verified against the 1.5 to 1.7 API). Where behaviour changed recently, the version is called out inline.

## Overview

### What it is

scikit-learn (imported as `sklearn`) is an open-source machine learning library built on NumPy, SciPy and joblib. It provides:

- **Supervised learning**: linear and logistic regression, support vector machines, nearest neighbours, decision trees, random forests, gradient boosting, naive Bayes, Gaussian processes, neural networks (multi-layer perceptron).
- **Unsupervised learning**: k-means, DBSCAN, HDBSCAN, hierarchical clustering, Gaussian mixtures, PCA, NMF, ICA, t-SNE, outlier detection.
- **Data preparation**: scaling, encoding, imputation, polynomial and spline features, discretisation, text vectorisation, feature selection.
- **Model selection and evaluation**: train/test splitting, cross-validation splitters, hyperparameter search, learning and validation curves, dozens of metrics, threshold tuning.
- **Composition**: `Pipeline`, `ColumnTransformer`, `FeatureUnion`, `TransformedTargetRegressor`, stacking and voting ensembles.
- **Inspection**: permutation importance, partial dependence and ICE plots, decision boundary displays.

### History and maintainers

The project began in 2007 as a Google Summer of Code project by David Cournapeau. In 2010 researchers at INRIA (Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort, Vincent Michel) made the first public release. The 2011 JMLR paper "Scikit-learn: Machine Learning in Python" remains the canonical citation. Today it is maintained by a distributed core team, with funding coordinated through the scikit-learn Consortium at Inria Foundation and the company :probabl. The license is BSD-3-Clause.

### When to use it

- Tabular data of small to medium size (up to a few million rows, fits in RAM).
- Baselines: a `LogisticRegression` or `HistGradientBoostingClassifier` baseline takes minutes and is often hard to beat.
- Anywhere you need rigorous, leakage-free evaluation: pipelines plus cross-validation are scikit-learn's strongest feature.
- Classical unsupervised tasks: clustering, dimensionality reduction, anomaly detection.
- As the "glue" layer: XGBoost, LightGBM, CatBoost, skorch, imbalanced-learn and many others expose scikit-learn-compatible estimators so they plug into its pipelines and search tools.

### When not to use it

- **Deep learning** on images, audio, text or sequences: use PyTorch, TensorFlow/Keras or JAX. `MLPClassifier` exists but has no GPU support and no convolution or attention layers.
- **Very large data** that does not fit in memory: consider Dask-ML, Spark MLlib, or out-of-core learners (`partial_fit` on `SGDClassifier`, `MiniBatchKMeans`, `IncrementalPCA`) as a partial answer.
- **GPU acceleration**: scikit-learn is CPU-based. NVIDIA cuML offers a largely compatible GPU API, and the experimental Array API support covers a limited set of estimators.
- **Maximum accuracy on large tabular data**: dedicated boosting libraries (XGBoost, LightGBM, CatBoost) are usually faster and slightly more accurate than scikit-learn's `HistGradientBoosting*`, though the gap is small.
- **Statistical inference** (p-values, confidence intervals on coefficients): use statsmodels.

### Where it fits in the ML stack

```text
Data loading        pandas / Polars / NumPy / PyArrow
        |
Preprocessing       sklearn.preprocessing, impute, compose  (or pandas)
        |
Modelling           sklearn estimators | XGBoost | LightGBM | CatBoost
        |
Selection & eval    sklearn.model_selection, sklearn.metrics
        |
Explainability      sklearn.inspection, SHAP
        |
Deployment          joblib / skops / ONNX (skl2onnx) / MLflow
```

## Installation

### pip

```bash
python -m pip install -U scikit-learn
```

scikit-learn 1.5 requires Python 3.9+; 1.6 and 1.7 require Python 3.9/3.10+. Wheels are provided for Windows, macOS (Intel and Apple Silicon) and Linux, so no compiler is needed.

Optional extras pulled in by some features:

```bash
python -m pip install matplotlib pandas   # plotting helpers and DataFrame output
```

### conda

```bash
conda install -c conda-forge scikit-learn
```

### Virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install -U pip scikit-learn pandas matplotlib
```

### GPU variants

scikit-learn itself has no GPU build. Options:

- **cuML** (RAPIDS) provides GPU estimators with a scikit-learn-like API, and `cuml.accel` can transparently accelerate unmodified scikit-learn code on supported estimators.
- **Intel Extension for Scikit-learn** (`scikit-learn-intelex`, `sklearnex`) patches many estimators for faster CPU (and some Intel GPU) execution:

```python
from sklearnex import patch_sklearn
patch_sklearn()  # call before importing sklearn estimators
from sklearn.cluster import KMeans
```

- **Array API dispatch** (experimental) lets a subset of estimators such as `LinearDiscriminantAnalysis`, `PCA` (with `svd_solver="full"` or randomized) and several metrics run on PyTorch or CuPy arrays:

```python
import sklearn
sklearn.set_config(array_api_dispatch=True)  # requires array-api-compat installed
```

### Verifying the install

```python
import sklearn
print(sklearn.__version__)
sklearn.show_versions()   # prints Python, NumPy, SciPy, BLAS and threadpool info
```

```bash
python -c "import sklearn; sklearn.show_versions()"
```

A quick smoke test:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=1000).fit(X, y)
print(clf.score(X, y))  # about 0.97
```

## Core Concepts

### The estimator API

Every object that learns from data is an **estimator**. The contract is small and uniform:

| Method | Who has it | What it does |
|---|---|---|
| `__init__(**params)` | all | Stores hyperparameters only. No validation, no data. |
| `fit(X, y=None)` | all | Learns from data, sets attributes ending in `_`, returns `self`. |
| `predict(X)` | classifiers, regressors, clusterers | Returns predictions, shape `(n_samples,)` or `(n_samples, n_outputs)`. |
| `predict_proba(X)` | most classifiers | Class probabilities, shape `(n_samples, n_classes)`. |
| `decision_function(X)` | linear models, SVMs, boosting | Unnormalised scores / margins. |
| `transform(X)` | transformers | Returns transformed features. |
| `fit_transform(X, y=None)` | transformers | `fit` then `transform`, often more efficient. |
| `inverse_transform(X)` | some transformers | Maps back to the original space. |
| `score(X, y)` | classifiers, regressors | Accuracy for classifiers, R^2 for regressors. |
| `get_params()` / `set_params()` | all | Read or change hyperparameters (supports `step__param` nesting). |
| `partial_fit(X, y)` | incremental learners | Update the model on a mini-batch. |

Conventions to remember:

- `X` is 2-D: `(n_samples, n_features)`. A single feature must be reshaped with `X.reshape(-1, 1)`.
- `y` is 1-D for single-output tasks.
- Learned attributes end with a trailing underscore: `coef_`, `classes_`, `feature_importances_`, `n_features_in_`, `feature_names_in_`.
- Hyperparameters are set in the constructor; data-dependent state is set in `fit`.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3))
y = X @ np.array([1.5, -2.0, 0.5]) + 3 + rng.normal(scale=0.1, size=100)

model = LinearRegression()          # 1. construct with hyperparameters
model.fit(X, y)                      # 2. learn
print(model.coef_, model.intercept_) # 3. inspect learned attributes
print(model.predict(X[:5]))          # 4. predict
print(model.get_params())            # {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}
```

### Transformers and the fit/transform split

Transformers learn statistics on training data and apply them to any data. Fitting on the test set (or on the whole dataset before splitting) is **data leakage**:

```python
# continuing with X, y from the previous example
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
scaler = StandardScaler().fit(X_train)   # mean/std from train only
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)      # reuse train statistics
```

### Pipelines: preventing leakage by construction

A `Pipeline` chains transformers and a final estimator into one estimator. When it is cross-validated, each fold re-fits the preprocessing on that fold's training portion only.

```python
# continuing with X, y from the previous example
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
print(scores.mean())
```

### Heterogeneous data with ColumnTransformer

Real tables mix numeric and categorical columns. `ColumnTransformer` applies a different transformer to each column subset and concatenates the results.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.DataFrame({
    "age": [25, 32, 47, 51],
    "income": [40_000, 52_000, 88_000, 61_000],
    "city": ["Paris", "Lyon", "Paris", "Nice"],
})
pre = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
print(pre.fit_transform(df))
print(pre.get_feature_names_out())
```

### Hyperparameters vs. parameters, and the bias-variance trade-off

- **Parameters** are learned (`coef_`, tree splits).
- **Hyperparameters** are chosen by you (`alpha`, `max_depth`, `n_neighbors`) and tuned with cross-validation.

Most hyperparameters control model complexity. Too simple underfits (high bias, both train and validation error high). Too complex overfits (high variance, train error low, validation error high). Use `validation_curve` and `learning_curve` to diagnose which regime you are in.

### Cross-validation

Instead of a single train/validation split, k-fold CV trains k models on k different splits and averages the scores, giving both a mean and a variance estimate.

```python
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = cross_validate(RandomForestClassifier(random_state=0), X, y, cv=cv,
                     scoring=["accuracy", "roc_auc"], return_train_score=True)
print(res["test_accuracy"].mean(), res["test_roc_auc"].mean())
```

Choose the splitter to match the data:

| Situation | Splitter |
|---|---|
| Regression, i.i.d. rows | `KFold(shuffle=True)` |
| Classification | `StratifiedKFold` (default when `cv=int` and estimator is a classifier) |
| Several rows per patient/user/session | `GroupKFold`, `StratifiedGroupKFold` |
| Time series | `TimeSeriesSplit` |
| Repeated estimates | `RepeatedStratifiedKFold` |

### Randomness and reproducibility

Any estimator with randomness has a `random_state` parameter. Pass an integer for reproducible results. Passing a `np.random.RandomState` instance makes successive `fit` calls differ, which is sometimes desirable inside CV.

### Global configuration and pandas output

```python
import sklearn
from sklearn.preprocessing import StandardScaler

sklearn.set_config(transform_output="pandas")   # all transformers return DataFrames
# or per estimator:
scaler = StandardScaler().set_output(transform="pandas")

with sklearn.config_context(assume_finite=True):
    pass  # skip finiteness checks inside this block
```

`set_output(transform="polars")` is supported since 1.4.

### Estimator HTML display and repr

In Jupyter, an estimator renders as an interactive diagram showing nested pipelines. In plain Python, `print(pipe)` shows only non-default parameters.

### Metadata routing (advanced)

Since 1.3/1.4 scikit-learn can route extra arguments such as `sample_weight` or `groups` through meta-estimators. It is opt-in:

```python
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GroupKFold

sklearn.set_config(enable_metadata_routing=True)
est = LogisticRegression().set_fit_request(sample_weight=True)
# cross_validate(est, X, y, params={"sample_weight": w, "groups": g}, cv=GroupKFold())
sklearn.set_config(enable_metadata_routing=False)
```

Note: `cross_validate(..., fit_params=...)` was deprecated in 1.4 in favour of `params=...`.

## API Reference

The reference below is organised by module. Signatures show the most important parameters with their defaults; every estimator accepts additional, less common parameters documented in the official API reference.

### sklearn.linear_model

#### LinearRegression

```python
from sklearn.linear_model import LinearRegression
LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
```

Ordinary least squares. No regularisation, no hyperparameters to tune.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fit_intercept` | bool | `True` | Whether to estimate an intercept. |
| `positive` | bool | `False` | Constrain coefficients to be non-negative. |
| `n_jobs` | int or None | `None` | Parallelism for multi-target problems. |

Attributes: `coef_`, `intercept_`, `rank_`, `singular_`. Note: the `normalize` parameter was removed in 1.2; use a `StandardScaler` in a pipeline.

#### Ridge, Lasso, ElasticNet

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
Ridge(alpha=1.0, *, fit_intercept=True, solver="auto", max_iter=None, tol=1e-4, positive=False, random_state=None)
Lasso(alpha=1.0, *, fit_intercept=True, max_iter=1000, tol=1e-4, warm_start=False, positive=False, selection="cyclic", random_state=None)
ElasticNet(alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4, selection="cyclic", random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | float | `1.0` | Regularisation strength. Larger means simpler model. |
| `l1_ratio` | float | `0.5` | ElasticNet mix: 0 is pure L2, 1 is pure L1. |
| `solver` | str | `"auto"` | Ridge: `"auto"`, `"svd"`, `"cholesky"`, `"lsqr"`, `"sparse_cg"`, `"sag"`, `"saga"`, `"lbfgs"`. |
| `max_iter` | int | `1000` | Coordinate descent iterations (Lasso/ElasticNet). |
| `selection` | str | `"cyclic"` | `"random"` can converge faster with many features. |

- **Ridge** (L2) shrinks all coefficients; good with correlated features.
- **Lasso** (L1) drives some coefficients exactly to zero: built-in feature selection.
- **ElasticNet** combines both.

Cross-validated variants pick `alpha` automatically: `RidgeCV(alphas=(0.1, 1.0, 10.0))`, `LassoCV(cv=5)`, `ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5)`.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=300, n_features=50, n_informative=8, noise=5, random_state=0)
model = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)).fit(X, y)
lasso = model[-1]
print("alpha:", lasso.alpha_, "non-zero coefs:", np.sum(lasso.coef_ != 0))
```

#### LogisticRegression

```python
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty="l2", *, dual=False, tol=1e-4, C=1.0, fit_intercept=True,
                   class_weight=None, random_state=None, solver="lbfgs", max_iter=100,
                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `penalty` | `"l1"`, `"l2"`, `"elasticnet"`, None | `"l2"` | Regularisation type. Not every solver supports every penalty. |
| `C` | float | `1.0` | Inverse regularisation strength. Smaller means stronger regularisation. |
| `solver` | str | `"lbfgs"` | `"lbfgs"`, `"liblinear"`, `"newton-cg"`, `"newton-cholesky"`, `"sag"`, `"saga"`. |
| `max_iter` | int | `100` | Raise this if you see `ConvergenceWarning`. |
| `class_weight` | dict, `"balanced"`, None | `None` | Re-weight classes for imbalanced data. |
| `l1_ratio` | float | `None` | Only with `penalty="elasticnet"` and `solver="saga"`. |

Solver compatibility:

| Solver | L1 | L2 | ElasticNet | None | Multiclass multinomial | Notes |
|---|---|---|---|---|---|---|
| `lbfgs` | no | yes | no | yes | yes | Good default. |
| `liblinear` | yes | yes | no | no | no (one-vs-rest) | Small datasets. |
| `newton-cg` | no | yes | no | yes | yes | |
| `newton-cholesky` | no | yes | no | yes | yes (1.6+) | Fast when n_samples much larger than n_features. |
| `sag` | no | yes | no | yes | yes | Large data, needs scaled features. |
| `saga` | yes | yes | yes | yes | yes | Large data, sparse, all penalties. |

Deprecation: the `multi_class` parameter was deprecated in 1.5 (multinomial is always used for multiclass with capable solvers). For one-vs-rest, wrap the model in `sklearn.multiclass.OneVsRestClassifier`. Recent releases are also reworking how `penalty` is expressed (in favour of `l1_ratio`), so check the docs for your exact version if you see a `FutureWarning`.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
clf.fit(X, y)
print(clf.predict_proba(X[:3]))
```

`LogisticRegressionCV(Cs=10, cv=5, scoring=None, solver="lbfgs", max_iter=100)` tunes `C` by cross-validation.

#### SGDClassifier and SGDRegressor

```python
from sklearn.linear_model import SGDClassifier, SGDRegressor
SGDClassifier(loss="hinge", *, penalty="l2", alpha=1e-4, l1_ratio=0.15, max_iter=1000, tol=1e-3,
              learning_rate="optimal", eta0=0.0, early_stopping=False, validation_fraction=0.1,
              n_iter_no_change=5, class_weight=None, random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss` | str | `"hinge"` | `"hinge"` (linear SVM), `"log_loss"` (logistic), `"modified_huber"`, `"squared_hinge"`, `"perceptron"`. Regressor: `"squared_error"`, `"huber"`, `"epsilon_insensitive"`. |
| `alpha` | float | `1e-4` | Regularisation constant. |
| `learning_rate` | str | `"optimal"` | `"constant"`, `"optimal"`, `"invscaling"`, `"adaptive"`. |
| `early_stopping` | bool | `False` | Hold out `validation_fraction` and stop when it stops improving. |

Supports `partial_fit` for streaming / out-of-core learning. Always scale features first. Note `loss="log"` was renamed `"log_loss"` (removed in 1.3).

#### Other linear models

| Class | Use case |
|---|---|
| `HuberRegressor(epsilon=1.35, alpha=1e-4)` | Regression robust to outliers. |
| `QuantileRegressor(quantile=0.5, alpha=1.0, solver="highs")` | Conditional quantiles (linear). |
| `PoissonRegressor(alpha=1.0)`, `GammaRegressor`, `TweedieRegressor(power=..., link=...)` | Generalised linear models for counts / positive targets. |
| `RANSACRegressor(estimator=None)` | Fits on inliers only. |
| `BayesianRidge()`, `ARDRegression()` | Bayesian linear regression with uncertainty. |
| `Perceptron()`, `PassiveAggressiveClassifier()` | Online linear classifiers. |
| `RidgeClassifier(alpha=1.0)` | Fast classifier using ridge regression on {-1, 1} targets. |
| `Lars`, `LassoLars`, `OrthogonalMatchingPursuit` | Sparse regression via least-angle methods. |

### sklearn.tree

#### DecisionTreeClassifier / DecisionTreeRegressor

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
DecisionTreeClassifier(*, criterion="gini", splitter="best", max_depth=None, min_samples_split=2,
                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                       random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                       class_weight=None, ccp_alpha=0.0, monotonic_cst=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `criterion` | str | `"gini"` | Classifier: `"gini"`, `"entropy"`, `"log_loss"`. Regressor: `"squared_error"`, `"friedman_mse"`, `"absolute_error"`, `"poisson"`. |
| `max_depth` | int or None | `None` | Maximum depth; `None` grows until leaves are pure. |
| `min_samples_split` | int or float | `2` | Minimum samples to split an internal node. |
| `min_samples_leaf` | int or float | `1` | Minimum samples per leaf; strong smoothing control. |
| `max_features` | int, float, `"sqrt"`, `"log2"`, None | `None` | Features considered per split. (`"auto"` was removed in 1.3.) |
| `max_leaf_nodes` | int or None | `None` | Best-first growth with a leaf cap. |
| `ccp_alpha` | float | `0.0` | Minimal cost-complexity pruning strength. |
| `monotonic_cst` | array-like | `None` | Per-feature monotonic constraint: 1, -1 or 0 (1.4+). |

Since 1.3 trees support missing values (`NaN`) natively with `splitter="best"`.

Attributes: `feature_importances_` (impurity-based), `tree_`, `get_depth()`, `get_n_leaves()`, `cost_complexity_pruning_path(X, y)`.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(iris.data, iris.target)
print(export_text(tree, feature_names=list(iris.feature_names)))

plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

Helper functions: `export_text(tree, feature_names=None, max_depth=10)`, `plot_tree(tree, ...)`, `export_graphviz(tree, out_file=None, ...)`.

`ExtraTreeClassifier` / `ExtraTreeRegressor` are single extremely randomised trees, mostly used inside ensembles.

### sklearn.ensemble

#### RandomForestClassifier / RandomForestRegressor

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
RandomForestClassifier(n_estimators=100, *, criterion="gini", max_depth=None, min_samples_split=2,
                       min_samples_leaf=1, max_features="sqrt", max_leaf_nodes=None, bootstrap=True,
                       oob_score=False, n_jobs=None, random_state=None, warm_start=False,
                       class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_estimators` | int | `100` | Number of trees. More is never worse for accuracy, only slower. |
| `max_features` | int, float, str | `"sqrt"` (clf), `1.0` (reg) | Features sampled per split: main decorrelation knob. |
| `max_depth` | int or None | `None` | Depth cap. |
| `min_samples_leaf` | int or float | `1` | Raise (e.g. 3 to 10) to reduce variance and model size. |
| `bootstrap` | bool | `True` | Sample rows with replacement. |
| `oob_score` | bool or callable | `False` | Estimate generalisation on out-of-bag rows. |
| `max_samples` | int, float, None | `None` | Rows drawn per tree when bootstrapping. |
| `class_weight` | dict, `"balanced"`, `"balanced_subsample"` | `None` | Imbalanced classes. |
| `n_jobs` | int | `None` | `-1` uses all cores. |

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1, random_state=0).fit(X, y)
print("OOB accuracy:", rf.oob_score_)
print("top importances:", sorted(rf.feature_importances_, reverse=True)[:5])
```

`ExtraTreesClassifier` / `ExtraTreesRegressor` share the API but choose split thresholds at random and default to `bootstrap=False`; they are often faster and similarly accurate.

#### HistGradientBoostingClassifier / HistGradientBoostingRegressor

The recommended scikit-learn gradient boosting implementation for anything over roughly 10,000 rows. LightGBM-inspired: bins features into at most 255 bins, supports missing values and categorical features natively, and has built-in early stopping.

```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
HistGradientBoostingClassifier(loss="log_loss", *, learning_rate=0.1, max_iter=100,
                               max_leaf_nodes=31, max_depth=None, min_samples_leaf=20,
                               l2_regularization=0.0, max_features=1.0, max_bins=255,
                               categorical_features="from_dtype", monotonic_cst=None,
                               interaction_cst=None, warm_start=False, early_stopping="auto",
                               scoring="loss", validation_fraction=0.1, n_iter_no_change=10,
                               tol=1e-7, verbose=0, random_state=None, class_weight=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss` | str | `"log_loss"` / `"squared_error"` | Regressor also supports `"absolute_error"`, `"gamma"`, `"poisson"`, `"quantile"` (with `quantile=`). |
| `learning_rate` | float | `0.1` | Shrinkage per iteration. |
| `max_iter` | int | `100` | Number of boosting iterations (trees per class). |
| `max_leaf_nodes` | int | `31` | Leaves per tree: main complexity knob. |
| `max_depth` | int or None | `None` | Optional depth cap. |
| `min_samples_leaf` | int | `20` | Minimum samples per leaf. |
| `l2_regularization` | float | `0.0` | L2 penalty on leaf values. |
| `max_features` | float | `1.0` | Fraction of features sampled per split (1.4+). |
| `categorical_features` | array, str | `"from_dtype"` | Columns to treat as categorical. `"from_dtype"` (default since 1.6) uses pandas `category` dtype. |
| `early_stopping` | `"auto"`, bool | `"auto"` | Enabled automatically when n_samples > 10,000. |
| `class_weight` | dict, `"balanced"` | `None` | Imbalanced classification (1.2+). |

```python
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
# object columns must be category dtype for "from_dtype" to pick them up
for col in X.select_dtypes("object").columns:
    X[col] = X[col].astype("category")
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
print(cross_val_score(hgb, X, y, cv=3, scoring="roc_auc").mean())
```

#### GradientBoostingClassifier / GradientBoostingRegressor

The original, exact-split implementation. Slower than the histogram version on large data, but supports `subsample`, `init` and `max_features` and is fine for small datasets.

```python
from sklearn.ensemble import GradientBoostingClassifier
GradientBoostingClassifier(*, loss="log_loss", learning_rate=0.1, n_estimators=100, subsample=1.0,
                           criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1,
                           max_depth=3, max_features=None, random_state=None,
                           validation_fraction=0.1, n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_estimators` | int | `100` | Boosting stages. |
| `learning_rate` | float | `0.1` | Shrinkage; trade off against `n_estimators`. |
| `max_depth` | int | `3` | Depth of each tree. |
| `subsample` | float | `1.0` | Values below 1.0 give stochastic gradient boosting. |
| `n_iter_no_change` | int or None | `None` | Enables early stopping on `validation_fraction`. |
| `loss` | str | `"log_loss"` | Regressor: `"squared_error"`, `"absolute_error"`, `"huber"`, `"quantile"`. |

`staged_predict(X)` and `staged_predict_proba(X)` yield predictions after each stage, useful for plotting test error vs. iterations.

#### AdaBoost and Bagging

```python
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=200, learning_rate=0.5)
bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, max_samples=0.8,
                        max_features=0.8, bootstrap=True, oob_score=True, n_jobs=-1, random_state=0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `estimator` | estimator | `None` | Base learner. Renamed from `base_estimator` (removed in 1.4). |
| `n_estimators` | int | `50` (Ada) / `10` (Bagging) | Number of base learners. |
| `learning_rate` | float | `1.0` | AdaBoost shrinkage. |
| `max_samples`, `max_features` | int or float | `1.0` | Bagging row / column sampling. |

AdaBoost's `algorithm="SAMME.R"` was deprecated in 1.4 and removed in 1.6; only SAMME remains.

#### Voting and Stacking

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

estimators = [
    ("rf", RandomForestClassifier(n_estimators=200, random_state=0)),
    ("svc", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
]
vote = VotingClassifier(estimators=estimators, voting="soft")
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5,
                           stack_method="auto", passthrough=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `estimators` | list of (name, est) | required | Base models. |
| `voting` | `"hard"`, `"soft"` | `"hard"` | Soft averages probabilities (needs `predict_proba`). |
| `weights` | list | `None` | Per-model weights. |
| `final_estimator` | estimator | `LogisticRegression` / `RidgeCV` | Meta-learner for stacking. |
| `cv` | int, splitter | `None` (5-fold) | How out-of-fold predictions are produced. |
| `passthrough` | bool | `False` | Also feed original features to the meta-learner. |

`VotingRegressor` and `StackingRegressor` are the regression equivalents.

#### IsolationForest

```python
from sklearn.ensemble import IsolationForest
IsolationForest(*, n_estimators=100, max_samples="auto", contamination="auto", max_features=1.0,
                bootstrap=False, n_jobs=None, random_state=None)
```

`fit_predict(X)` returns `1` for inliers and `-1` for outliers; `score_samples(X)` returns anomaly scores (lower is more abnormal).

### sklearn.svm

```python
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, OneClassSVM
SVC(*, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0, shrinking=True, probability=False,
    tol=1e-3, cache_size=200, class_weight=None, max_iter=-1, decision_function_shape="ovr",
    break_ties=False, random_state=None)
LinearSVC(penalty="l2", loss="squared_hinge", *, dual="auto", tol=1e-4, C=1.0, multi_class="ovr",
          fit_intercept=True, class_weight=None, random_state=None, max_iter=1000)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `C` | float | `1.0` | Inverse regularisation; higher fits training data harder. |
| `kernel` | str or callable | `"rbf"` | `"linear"`, `"poly"`, `"rbf"`, `"sigmoid"`, `"precomputed"`. |
| `gamma` | `"scale"`, `"auto"`, float | `"scale"` | RBF width; higher means more local, wigglier boundary. |
| `degree` | int | `3` | Polynomial kernel degree. |
| `probability` | bool | `False` | Enable `predict_proba` via internal 5-fold Platt scaling (slow). |
| `class_weight` | dict, `"balanced"` | `None` | Imbalanced classes. |
| `dual` | bool, `"auto"` | `"auto"` | LinearSVC: default changed to `"auto"` in 1.5. |
| `epsilon` | float | `0.1` | SVR: width of the no-penalty tube. |

Guidance:

- Kernel SVMs scale roughly O(n^2) to O(n^3) in samples; fine up to around 10,000 to 50,000 rows.
- For large or high-dimensional sparse data (text), use `LinearSVC` or `SGDClassifier`.
- Always standardise features.
- Tune `C` and `gamma` on a log grid together.

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

X, y = make_moons(n_samples=500, noise=0.25, random_state=0)
grid = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10]}, cv=5)
grid.fit(X, y)
print(grid.best_params_, grid.best_score_)
```

`OneClassSVM(kernel="rbf", nu=0.5, gamma="scale")` is used for novelty detection.

### sklearn.neighbors

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, LocalOutlierFactor
KNeighborsClassifier(n_neighbors=5, *, weights="uniform", algorithm="auto", leaf_size=30, p=2,
                     metric="minkowski", metric_params=None, n_jobs=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_neighbors` | int | `5` | k. Small k means low bias / high variance. |
| `weights` | `"uniform"`, `"distance"`, callable | `"uniform"` | Weight votes by inverse distance. |
| `algorithm` | str | `"auto"` | `"ball_tree"`, `"kd_tree"`, `"brute"`. |
| `metric` | str or callable | `"minkowski"` | With `p=2` this is Euclidean, `p=1` Manhattan. Also `"cosine"` (brute only). |

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

X = np.random.default_rng(0).normal(size=(1000, 16))
nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(X)
distances, indices = nn.kneighbors(X[:2])
print(indices)
```

Related: `RadiusNeighborsClassifier`, `KernelDensity(bandwidth=1.0, kernel="gaussian")`, `NearestCentroid`, `LocalOutlierFactor(n_neighbors=20, novelty=False)` (use `novelty=True` to call `predict` on new data), and `KNeighborsTransformer` for precomputed graphs.

### sklearn.cluster

#### KMeans and MiniBatchKMeans

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
KMeans(n_clusters=8, *, init="k-means++", n_init="auto", max_iter=300, tol=1e-4,
       verbose=0, random_state=None, copy_x=True, algorithm="lloyd")
MiniBatchKMeans(n_clusters=8, *, init="k-means++", max_iter=100, batch_size=1024,
                n_init="auto", random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | int | `8` | Number of clusters k. |
| `init` | str or array | `"k-means++"` | `"k-means++"`, `"random"`, or explicit centres. |
| `n_init` | int or `"auto"` | `"auto"` | Restarts; `"auto"` means 1 for k-means++ and 10 for random. Default changed from 10 in 1.4. |
| `algorithm` | str | `"lloyd"` | `"lloyd"` or `"elkan"` (`"auto"` and `"full"` were removed). |
| `batch_size` | int | `1024` | MiniBatchKMeans batch size. |

Attributes: `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`. `MiniBatchKMeans` supports `partial_fit`.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=1.0, random_state=0)
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
    print(k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3))
```

#### DBSCAN and HDBSCAN

```python
from sklearn.cluster import DBSCAN, HDBSCAN
DBSCAN(eps=0.5, *, min_samples=5, metric="euclidean", algorithm="auto", leaf_size=30, n_jobs=None)
HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, metric="euclidean",
        alpha=1.0, algorithm="auto", cluster_selection_method="eom", store_centers=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eps` | float | `0.5` | DBSCAN neighbourhood radius. Scale-sensitive: standardise first. |
| `min_samples` | int | `5` | Points needed to form a dense core. |
| `min_cluster_size` | int | `5` | HDBSCAN smallest cluster allowed. |
| `cluster_selection_method` | str | `"eom"` | `"eom"` or `"leaf"` (smaller, more homogeneous clusters). |

Both label noise points `-1`. `HDBSCAN` (added in 1.3) needs no `eps` and handles clusters of varying density. Neither has a `predict` method for new points.

```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN
import numpy as np

X, _ = make_moons(n_samples=500, noise=0.06, random_state=0)
X = StandardScaler().fit_transform(X)
db = DBSCAN(eps=0.3, min_samples=5).fit(X)
hdb = HDBSCAN(min_cluster_size=15).fit(X)
print("DBSCAN clusters:", len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))
print("HDBSCAN noise points:", np.sum(hdb.labels_ == -1))
```

#### AgglomerativeClustering and others

```python
from sklearn.cluster import AgglomerativeClustering
AgglomerativeClustering(n_clusters=2, *, metric="euclidean", linkage="ward", distance_threshold=None,
                        compute_distances=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | int or None | `2` | Must be `None` if `distance_threshold` is set. |
| `linkage` | str | `"ward"` | `"ward"`, `"complete"`, `"average"`, `"single"`. Ward requires Euclidean. |
| `metric` | str | `"euclidean"` | Renamed from `affinity` (removed in 1.4). |
| `distance_threshold` | float or None | `None` | Cut the dendrogram at this distance instead of fixing k. |

| Class | Strength |
|---|---|
| `SpectralClustering(n_clusters=8, affinity="rbf")` | Non-convex clusters, small data. |
| `MeanShift(bandwidth=None)` | Finds k automatically; slow. |
| `OPTICS(min_samples=5)` | Like DBSCAN across a range of eps. |
| `Birch(n_clusters=3, threshold=0.5)` | Large data, incremental. |
| `AffinityPropagation()` | Exemplar-based, finds k automatically. |
| `BisectingKMeans(n_clusters=8)` | Hierarchical divisive k-means (1.1+). |

#### sklearn.mixture.GaussianMixture

```python
from sklearn.mixture import GaussianMixture
GaussianMixture(n_components=1, *, covariance_type="full", tol=1e-3, reg_covar=1e-6, max_iter=100,
                n_init=1, init_params="kmeans", random_state=None)
```

Soft clustering with `predict_proba`, density estimation with `score_samples`, and model selection with `bic(X)` / `aic(X)`. `covariance_type` is `"full"`, `"tied"`, `"diag"` or `"spherical"`. `BayesianGaussianMixture` infers the effective number of components.

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=600, centers=3, random_state=1)
bics = [GaussianMixture(n_components=k, random_state=0).fit(X).bic(X) for k in range(1, 7)]
print("best k:", int(np.argmin(bics)) + 1)
```

### sklearn.decomposition and sklearn.manifold

#### PCA

```python
from sklearn.decomposition import PCA
PCA(n_components=None, *, copy=True, whiten=False, svd_solver="auto", tol=0.0,
    iterated_power="auto", n_oversamples=10, random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_components` | int, float, `"mle"`, None | `None` | int keeps that many; float in (0, 1) keeps enough to explain that variance fraction. |
| `whiten` | bool | `False` | Scale components to unit variance. |
| `svd_solver` | str | `"auto"` | `"full"`, `"arpack"`, `"randomized"`, `"covariance_eigh"` (1.5+). |

Attributes: `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`, `n_components_`.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
pca = PCA(n_components=0.95).fit(X)
print(pca.n_components_, "components keep 95% variance")
X_reduced = pca.transform(X)
X_back = pca.inverse_transform(X_reduced)
print("reconstruction MSE:", np.mean((X - X_back) ** 2))
```

PCA centres data but does not scale it; put a `StandardScaler` first when features have different units. For sparse text matrices use `TruncatedSVD`, which does not centre and therefore keeps the matrix sparse.

#### Other decompositions

| Class | Signature (key params) | Use |
|---|---|---|
| `TruncatedSVD` | `(n_components=2, algorithm="randomized", n_iter=5, random_state=None)` | Sparse matrices, latent semantic analysis. |
| `IncrementalPCA` | `(n_components=None, batch_size=None)` | Out-of-core PCA via `partial_fit`. |
| `KernelPCA` | `(n_components=None, kernel="linear", gamma=None, fit_inverse_transform=False)` | Non-linear PCA. |
| `NMF` | `(n_components="auto", init=None, solver="cd", beta_loss="frobenius", max_iter=200)` | Non-negative parts-based decomposition (topics, spectra). |
| `FastICA` | `(n_components=None, algorithm="parallel", whiten="unit-variance")` | Blind source separation. |
| `LatentDirichletAllocation` | `(n_components=10, learning_method="batch")` | Topic modelling on count matrices. |
| `SparsePCA`, `DictionaryLearning`, `FactorAnalysis` | | Specialised factorisations. |

#### Manifold learning

```python
from sklearn.manifold import TSNE
TSNE(n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate="auto",
     max_iter=None, init="pca", metric="euclidean", random_state=None)
```

t-SNE is for visualisation only: it has `fit_transform` but no `transform` for new points. `n_iter` was renamed `max_iter` in 1.5 (effective default 1000). Other manifold learners: `Isomap`, `MDS`, `LocallyLinearEmbedding`, `SpectralEmbedding`. For a transform-capable non-linear embedding, use the separate `umap-learn` package.

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
plt.scatter(emb[:, 0], emb[:, 1], c=y, s=5, cmap="tab10")
plt.show()
```

### sklearn.preprocessing

#### Scalers

| Class | Signature | Effect | When |
|---|---|---|---|
| `StandardScaler` | `(*, copy=True, with_mean=True, with_std=True)` | Zero mean, unit variance. | Default for linear models, SVMs, kNN, PCA. |
| `MinMaxScaler` | `(feature_range=(0, 1), *, copy=True, clip=False)` | Scales to a range. | Bounded inputs, neural nets. |
| `MaxAbsScaler` | `(*, copy=True)` | Divides by max absolute value. | Sparse data (keeps sparsity). |
| `RobustScaler` | `(*, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))` | Median / IQR. | Data with outliers. |
| `Normalizer` | `(norm="l2")` | Scales each **row** to unit norm. | Text vectors, cosine similarity. |
| `PowerTransformer` | `(method="yeo-johnson", *, standardize=True)` | Makes features more Gaussian. | Skewed features. `"box-cox"` needs strictly positive data. |
| `QuantileTransformer` | `(*, n_quantiles=1000, output_distribution="uniform")` | Maps to uniform or normal via ranks. | Heavy outliers, non-linear monotone fix. |

Tree-based models (random forests, gradient boosting) do not need scaling.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

X = np.array([[1.0, 200], [2.0, 220], [3.0, 10_000]])
print(StandardScaler().fit_transform(X))
print(RobustScaler().fit_transform(X))
```

#### Encoders

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder
OneHotEncoder(*, categories="auto", drop=None, sparse_output=True, dtype=np.float64,
              handle_unknown="error", min_frequency=None, max_categories=None, feature_name_combiner="concat")
OrdinalEncoder(*, categories="auto", dtype=np.float64, handle_unknown="error", unknown_value=None,
               encoded_missing_value=np.nan, min_frequency=None, max_categories=None)
TargetEncoder(categories="auto", target_type="auto", smooth="auto", cv=5, shuffle=True, random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `handle_unknown` | str | `"error"` | OneHot: `"ignore"` or `"infrequent_if_exist"`. Ordinal: `"use_encoded_value"` with `unknown_value=-1`. |
| `sparse_output` | bool | `True` | Renamed from `sparse` in 1.2 (old name removed in 1.4). |
| `drop` | `"first"`, `"if_binary"`, array, None | `None` | Drop a column per feature to avoid collinearity in unregularised linear models. |
| `min_frequency` | int or float | `None` | Group rare categories into an "infrequent" bucket. |
| `max_categories` | int | `None` | Cap the number of output columns per feature. |
| `smooth` | `"auto"` or float | `"auto"` | TargetEncoder shrinkage toward the global mean. |
| `cv` | int | `5` | TargetEncoder uses cross-fitting in `fit_transform` to avoid leakage. |

`LabelEncoder` is for encoding **targets** (`y`), not features. `TargetEncoder` (added in 1.3) is the right tool for high-cardinality categoricals; `fit_transform` intentionally differs from `fit(...).transform(...)` because of cross-fitting.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df = pd.DataFrame({"size": ["S", "M", "L", "M"], "color": ["red", "blue", "red", "green"]})
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[["color"]])
print(ohe.get_feature_names_out(), ohe.transform(pd.DataFrame({"color": ["purple"]})))

ordenc = OrdinalEncoder(categories=[["S", "M", "L"]])
print(ordenc.fit_transform(df[["size"]]).ravel())   # [0. 1. 2. 1.]
```

#### Feature generation and other transformers

| Class | Key params | Purpose |
|---|---|---|
| `PolynomialFeatures` | `degree=2, interaction_only=False, include_bias=True` | Polynomial and interaction terms. |
| `SplineTransformer` | `n_knots=5, degree=3, extrapolation="constant"` | Smooth non-linear basis for linear models; `extrapolation="periodic"` for cyclic features. |
| `KBinsDiscretizer` | `n_bins=5, encode="onehot", strategy="quantile"` | Bin continuous features. |
| `Binarizer` | `threshold=0.0` | Threshold to 0/1. |
| `FunctionTransformer` | `func=None, inverse_func=None, validate=False, feature_names_out=None` | Wrap any function, e.g. `np.log1p`. |
| `LabelBinarizer`, `MultiLabelBinarizer` | | Encode targets for multilabel problems. |

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

log = FunctionTransformer(np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")
print(log.fit_transform(np.array([[0.0], [9.0]])))
print(PolynomialFeatures(degree=2, include_bias=False).fit_transform([[2, 3]]))  # [[2. 3. 4. 6. 9.]]
```

### sklearn.impute

```python
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
SimpleImputer(*, missing_values=np.nan, strategy="mean", fill_value=None, copy=True,
              add_indicator=False, keep_empty_features=False)
KNNImputer(*, missing_values=np.nan, n_neighbors=5, weights="uniform", metric="nan_euclidean",
           add_indicator=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | str or callable | `"mean"` | `"mean"`, `"median"`, `"most_frequent"`, `"constant"` (a callable is accepted since 1.5). |
| `fill_value` | scalar | `None` | Used with `"constant"`; e.g. `"missing"` for strings. |
| `add_indicator` | bool | `False` | Append binary "was missing" columns: often a useful signal. |
| `n_neighbors` | int | `5` | KNNImputer neighbours. |

`IterativeImputer` (MICE-style) is still experimental and must be enabled explicitly:

```python
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

X = np.array([[1, 2, np.nan], [3, np.nan, 6], [7, 8, 9], [np.nan, 5, 4]], dtype=float)
imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=0),
                       max_iter=10, random_state=0)
print(imp.fit_transform(X))
```

### sklearn.compose

#### ColumnTransformer

```python
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
ColumnTransformer(transformers, *, remainder="drop", sparse_threshold=0.3, n_jobs=None,
                  transformer_weights=None, verbose=False, verbose_feature_names_out=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `transformers` | list of (name, transformer, columns) | required | Columns may be names, indices, boolean masks, or a `make_column_selector`. Transformer may be `"drop"` or `"passthrough"`. |
| `remainder` | `"drop"`, `"passthrough"`, estimator | `"drop"` | What to do with unlisted columns. |
| `sparse_threshold` | float | `0.3` | Output sparse if overall density is below this. |
| `verbose_feature_names_out` | bool | `True` | Prefix output names with the transformer name (`num__age`). |

```python
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

pre = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
     make_column_selector(dtype_include="number")),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
     make_column_selector(dtype_include=["object", "category"])),
)
```

#### TransformedTargetRegressor

```python
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge

reg = TransformedTargetRegressor(regressor=Ridge(), func=np.log1p, inverse_func=np.expm1)
# fit(X, y) trains Ridge on log1p(y); predict returns values on the original scale
```

### sklearn.pipeline

```python
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
Pipeline(steps, *, memory=None, verbose=False)
make_pipeline(*steps, memory=None, verbose=False)
FeatureUnion(transformer_list, *, n_jobs=None, transformer_weights=None, verbose=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `steps` | list of (name, estimator) | required | All but the last must be transformers. A step can be `"passthrough"` or `None`. |
| `memory` | str or joblib.Memory | `None` | Cache fitted transformers (speeds up grid searches over the final step). |
| `transformer_list` | list of (name, transformer) | required | FeatureUnion runs transformers in parallel and concatenates outputs. |

Key behaviours:

- Access steps with `pipe.named_steps["clf"]`, `pipe["clf"]`, `pipe[-1]`, or slice `pipe[:-1]` to get the preprocessing part.
- Set nested parameters with double underscores: `pipe.set_params(clf__C=10)`, which is also how grid search keys are written.
- `make_pipeline` names steps automatically from the lowercase class name (`"standardscaler"`, `"logisticregression"`).
- `pipe.get_feature_names_out()` works when every transformer supports it.

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
features = FeatureUnion([("pca", PCA(n_components=2)), ("kbest", SelectKBest(k=1))])
pipe = Pipeline([("features", features), ("svc", SVC(kernel="linear"))])
pipe.fit(X, y)
print(pipe.score(X, y))
```

### sklearn.feature_selection

| Class / function | Signature (key params) | Idea |
|---|---|---|
| `VarianceThreshold` | `(threshold=0.0)` | Drop near-constant features. |
| `SelectKBest` | `(score_func=f_classif, *, k=10)` | Keep top-k by univariate score. |
| `SelectPercentile` | `(score_func=f_classif, *, percentile=10)` | Keep top percent. |
| `SelectFromModel` | `(estimator, *, threshold=None, prefit=False, max_features=None)` | Keep features whose `coef_` / `feature_importances_` exceed threshold. |
| `RFE` | `(estimator, *, n_features_to_select=None, step=1)` | Recursively drop weakest features. |
| `RFECV` | `(estimator, *, step=1, min_features_to_select=1, cv=None, scoring=None)` | RFE with CV to pick the count. |
| `SequentialFeatureSelector` | `(estimator, *, n_features_to_select="auto", tol=None, direction="forward", scoring=None, cv=5)` | Greedy forward/backward selection by CV score. |

Score functions: `f_classif`, `chi2` (non-negative features), `mutual_info_classif`, `f_regression`, `r_regression`, `mutual_info_regression`.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

kbest = SelectKBest(mutual_info_classif, k=10).fit(X, y)
print(kbest.get_feature_names_out())

sfm = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=0), threshold="median").fit(X, y)
print(sfm.get_support().sum(), "features kept")

rfecv = RFECV(LogisticRegression(max_iter=5000), step=1, cv=StratifiedKFold(5), scoring="roc_auc")
rfecv.fit(StandardScaler().fit_transform(X), y)
print("optimal number of features:", rfecv.n_features_)
```

Always put feature selection inside the pipeline so it is re-run on every CV fold; selecting on the full dataset first is a classic leakage bug.

### sklearn.model_selection

#### train_test_split

```python
from sklearn.model_selection import train_test_split
train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `*arrays` | arrays / DataFrames | required | Any number of same-length arrays (X, y, weights...). |
| `test_size` | float or int | `None` (0.25) | Fraction or absolute number of test rows. |
| `random_state` | int | `None` | Seed for reproducible splits. |
| `shuffle` | bool | `True` | Set `False` for time-ordered data. |
| `stratify` | array | `None` | Pass `y` to preserve class proportions. |

Returns a list `[X_train, X_test, y_train, y_test, ...]`.

#### Cross-validation splitters

| Class | Signature | Notes |
|---|---|---|
| `KFold` | `(n_splits=5, *, shuffle=False, random_state=None)` | Basic k-fold. |
| `StratifiedKFold` | `(n_splits=5, *, shuffle=False, random_state=None)` | Keeps class ratios. |
| `GroupKFold` | `(n_splits=5)` | Groups never split across folds (shuffle option added in 1.6). |
| `StratifiedGroupKFold` | `(n_splits=5, shuffle=False, random_state=None)` | Both constraints. |
| `TimeSeriesSplit` | `(n_splits=5, *, max_train_size=None, test_size=None, gap=0)` | Expanding window, never trains on the future. |
| `RepeatedStratifiedKFold` | `(*, n_splits=5, n_repeats=10, random_state=None)` | Lower-variance estimates. |
| `ShuffleSplit`, `StratifiedShuffleSplit` | `(n_splits=10, *, test_size=None, random_state=None)` | Random repeated splits. |
| `LeaveOneOut`, `LeaveOneGroupOut`, `PredefinedSplit` | | Special cases. |

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

X = np.arange(20).reshape(-1, 1)
for train_idx, test_idx in TimeSeriesSplit(n_splits=4, gap=1).split(X):
    print(train_idx.min(), train_idx.max(), "->", test_idx.min(), test_idx.max())
```

#### cross_val_score, cross_validate, cross_val_predict

```python
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None,
                verbose=0, params=None, error_score=np.nan)
cross_validate(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None,
               params=None, return_train_score=False, return_estimator=False, return_indices=False,
               error_score=np.nan)
cross_val_predict(estimator, X, y=None, *, groups=None, cv=None, n_jobs=None, params=None,
                  method="predict")
```

| Function | Returns |
|---|---|
| `cross_val_score` | 1-D array of test scores, one per fold. Single metric only. |
| `cross_validate` | dict with `fit_time`, `score_time`, `test_<metric>` (and `train_<metric>`, `estimator`, `indices` if requested). Multiple metrics. |
| `cross_val_predict` | Out-of-fold predictions for every sample (use for diagnostics or stacking, not as a score). |

`scoring` accepts a string (`"roc_auc"`, `"neg_root_mean_squared_error"`), a callable from `make_scorer`, a list or a dict. Error-type scorers are negated so that higher is always better. Use `sklearn.metrics.get_scorer_names()` for the full list.

#### GridSearchCV and RandomizedSearchCV

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None,
             verbose=0, pre_dispatch="2*n_jobs", error_score=np.nan, return_train_score=False)
RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None,
                   refit=True, cv=None, verbose=0, random_state=None, error_score=np.nan,
                   return_train_score=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `param_grid` / `param_distributions` | dict or list of dicts | required | Keys use `step__param` for pipelines. Distributions may be scipy.stats objects or lists. |
| `n_iter` | int | `10` | Random search candidates. |
| `scoring` | str, callable, list, dict | `None` | Estimator's `score` if None. |
| `refit` | bool or str | `True` | Refit best model on all data. With multi-metric scoring pass the metric name. |
| `cv` | int or splitter | `None` (5) | Folds. |
| `n_jobs` | int | `None` | `-1` for all cores. |

Attributes after fit: `best_estimator_`, `best_params_`, `best_score_`, `best_index_`, `cv_results_` (convert with `pd.DataFrame(search.cv_results_)`), `refit_time_`. The search object itself exposes `predict`, `predict_proba`, `score`, delegating to `best_estimator_`.

```python
from scipy.stats import loguniform, randint
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=0, n_jobs=-1),
    {"n_estimators": randint(100, 600), "max_depth": [None, 4, 8, 16],
     "min_samples_leaf": randint(1, 10), "max_features": ["sqrt", "log2", 0.5]},
    n_iter=25, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=0),
    random_state=0, n_jobs=-1,
)
search.fit(X, y)
print(search.best_params_, round(search.best_score_, 4))
```

Successive halving (faster for large grids) remains experimental:

```python
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
# HalvingRandomSearchCV(estimator, param_distributions, factor=3, resource="n_samples", ...)
```

#### Learning and validation curves

```python
from sklearn.model_selection import learning_curve, validation_curve, LearningCurveDisplay, ValidationCurveDisplay
learning_curve(estimator, X, y, *, train_sizes=np.linspace(0.1, 1.0, 5), cv=None, scoring=None, n_jobs=None, shuffle=False, random_state=None)
validation_curve(estimator, X, y, *, param_name, param_range, cv=None, scoring=None, n_jobs=None)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import ValidationCurveDisplay, LearningCurveDisplay

X, y = load_digits(return_X_y=True)
ValidationCurveDisplay.from_estimator(SVC(), X, y, param_name="gamma",
                                      param_range=np.logspace(-6, -1, 6), cv=5)
LearningCurveDisplay.from_estimator(SVC(gamma=0.001), X, y, cv=5)
plt.show()
```

#### Threshold tuning (1.5+)

```python
from sklearn.model_selection import TunedThresholdClassifierCV, FixedThresholdClassifier
TunedThresholdClassifierCV(estimator, *, scoring="balanced_accuracy", response_method="auto",
                           thresholds=100, cv=None, refit=True, random_state=None)
FixedThresholdClassifier(estimator, *, threshold="auto", pos_label=None, response_method="auto")
```

Binary classifiers predict class 1 when probability is at least 0.5 by default. `TunedThresholdClassifierCV` picks the decision threshold that maximises a metric (for example `"f1"` or a cost-based scorer) by cross-validation.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV

X, y = make_classification(n_samples=5000, weights=[0.95, 0.05], random_state=0)
tuned = TunedThresholdClassifierCV(LogisticRegression(max_iter=1000), scoring="f1", cv=5).fit(X, y)
print("best threshold:", round(tuned.best_threshold_, 3), "f1:", round(tuned.best_score_, 3))
```

### sklearn.metrics

#### Classification metrics

| Function | Signature (key params) | Notes |
|---|---|---|
| `accuracy_score` | `(y_true, y_pred, *, normalize=True, sample_weight=None)` | Misleading on imbalanced data. |
| `balanced_accuracy_score` | `(y_true, y_pred, *, adjusted=False)` | Mean recall per class. |
| `precision_score`, `recall_score`, `f1_score` | `(y_true, y_pred, *, labels=None, pos_label=1, average="binary", zero_division="warn")` | `average`: `"binary"`, `"micro"`, `"macro"`, `"weighted"`, `"samples"`, `None`. |
| `fbeta_score` | `(y_true, y_pred, *, beta, average="binary")` | beta > 1 favours recall. |
| `roc_auc_score` | `(y_true, y_score, *, average="macro", multi_class="raise", labels=None)` | Needs scores/probabilities. Multiclass: `multi_class="ovr"` or `"ovo"`. |
| `average_precision_score` | `(y_true, y_score, *, average="macro", pos_label=1)` | Area under PR curve; better than ROC AUC for rare positives. |
| `log_loss` | `(y_true, y_pred, *, normalize=True, sample_weight=None, labels=None)` | Probabilistic quality. |
| `brier_score_loss` | `(y_true, y_proba, *, pos_label=None)` | Calibration-sensitive. |
| `matthews_corrcoef` | `(y_true, y_pred)` | Robust single number for imbalanced binary. |
| `cohen_kappa_score` | `(y1, y2, *, weights=None)` | Agreement beyond chance. |
| `confusion_matrix` | `(y_true, y_pred, *, labels=None, normalize=None)` | Rows are true classes, columns predicted. |
| `classification_report` | `(y_true, y_pred, *, target_names=None, digits=2, output_dict=False, zero_division="warn")` | Text or dict summary. |
| `roc_curve`, `precision_recall_curve` | `(y_true, y_score)` | Curves for plotting or threshold picking. |
| `hamming_loss`, `jaccard_score` | | Multilabel. |

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay)
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
y_proba = clf.predict_proba(X_te)[:, 1]

print(classification_report(y_te, y_pred, digits=3))
print(confusion_matrix(y_te, y_pred))
print("ROC AUC:", roc_auc_score(y_te, y_proba), "PR AUC:", average_precision_score(y_te, y_proba))

ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te)
RocCurveDisplay.from_estimator(clf, X_te, y_te)
plt.show()
```

The old `plot_confusion_matrix`, `plot_roc_curve` and `plot_precision_recall_curve` functions were removed in 1.2. Use the `*Display.from_estimator` / `from_predictions` class methods: `ConfusionMatrixDisplay`, `RocCurveDisplay`, `PrecisionRecallDisplay`, `DetCurveDisplay`, `PredictionErrorDisplay` (regression), and `CalibrationDisplay` (in `sklearn.calibration`).

#### Regression metrics

| Function | Notes |
|---|---|
| `mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average")` | MSE. |
| `root_mean_squared_error(y_true, y_pred)` | Added in 1.4. Replaces `mean_squared_error(..., squared=False)`, which was deprecated in 1.4 and removed in 1.6. |
| `mean_absolute_error` | MAE, robust to outliers. |
| `median_absolute_error` | Very robust. |
| `mean_absolute_percentage_error` | MAPE, returned as a fraction (0.1 means 10%), not a percent. |
| `r2_score(y_true, y_pred)` | Coefficient of determination; can be negative. |
| `explained_variance_score` | Like R^2 but ignores bias. |
| `mean_squared_log_error`, `root_mean_squared_log_error` | Relative errors on positive targets. |
| `mean_pinball_loss(y_true, y_pred, *, alpha=0.5)` | Quantile regression loss. |
| `mean_poisson_deviance`, `mean_gamma_deviance`, `mean_tweedie_deviance` | GLM deviances. |
| `d2_absolute_error_score`, `d2_tweedie_score` | R^2-like skill scores for other losses. |
| `max_error` | Worst-case error. |

#### Clustering metrics

| With ground truth | Without ground truth |
|---|---|
| `adjusted_rand_score(labels_true, labels_pred)` | `silhouette_score(X, labels, *, metric="euclidean", sample_size=None)` |
| `adjusted_mutual_info_score`, `normalized_mutual_info_score` | `calinski_harabasz_score(X, labels)` (higher better) |
| `homogeneity_score`, `completeness_score`, `v_measure_score` | `davies_bouldin_score(X, labels)` (lower better) |
| `fowlkes_mallows_score` | `silhouette_samples(X, labels)` per-point |

#### Custom scorers

```python
from sklearn.metrics import make_scorer, fbeta_score, get_scorer_names
f2 = make_scorer(fbeta_score, beta=2)
# Probability-based custom metric (1.4+ uses response_method instead of needs_proba)
from sklearn.metrics import log_loss
neg_ll = make_scorer(log_loss, greater_is_better=False, response_method="predict_proba")
print(len(get_scorer_names()), "built-in scorers")
```

`needs_proba` / `needs_threshold` were deprecated in 1.4 in favour of `response_method`.

#### Pairwise utilities

`sklearn.metrics.pairwise` provides `cosine_similarity`, `euclidean_distances`, `pairwise_distances(X, Y=None, metric="euclidean", n_jobs=None)`, `rbf_kernel`, `linear_kernel` and friends.

### sklearn.inspection

#### permutation_importance

```python
from sklearn.inspection import permutation_importance
permutation_importance(estimator, X, y, *, scoring=None, n_repeats=5, n_jobs=None,
                       random_state=None, sample_weight=None, max_samples=1.0)
```

Returns a `Bunch` with `importances_mean`, `importances_std`, `importances` (shape `(n_features, n_repeats)`). Compute it on held-out data. Unlike impurity-based `feature_importances_`, it is not biased toward high-cardinality features, but correlated features share (and can hide) importance.

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
model = HistGradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
r = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=0, n_jobs=-1)
print(pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False))
```

#### Partial dependence and ICE

```python
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(estimator, X, features, *, kind="average",
                                        grid_resolution=100, categorical_features=None,
                                        centered=False, subsample=1000, random_state=None)
```

`kind="individual"` draws ICE curves, `kind="both"` overlays both. A tuple in `features` such as `("MedInc", "AveOccup")` draws a 2-D interaction plot.

```python
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
# continuing from the permutation_importance example
PartialDependenceDisplay.from_estimator(model, X_te, ["MedInc", "AveOccup", ("Latitude", "Longitude")],
                                        kind="average", grid_resolution=30)
plt.show()
```

#### DecisionBoundaryDisplay

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

X, y = make_moons(noise=0.2, random_state=0)
clf = SVC(gamma=2).fit(X, y)
disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", alpha=0.4)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
plt.show()
```

### Other frequently used modules

#### sklearn.naive_bayes

| Class | Data |
|---|---|
| `GaussianNB(var_smoothing=1e-9)` | Continuous features. |
| `MultinomialNB(alpha=1.0)` | Word counts / TF-IDF. |
| `ComplementNB(alpha=1.0)` | Imbalanced text classification. |
| `BernoulliNB(alpha=1.0, binarize=0.0)` | Binary features. |
| `CategoricalNB(alpha=1.0)` | Ordinal-encoded categoricals. |

#### sklearn.neural_network

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor
MLPClassifier(hidden_layer_sizes=(100,), activation="relu", *, solver="adam", alpha=1e-4,
              batch_size="auto", learning_rate_init=1e-3, max_iter=200, early_stopping=False,
              random_state=None)
```

Scale inputs first. Useful for small non-linear problems; for anything serious use a deep learning framework.

#### sklearn.calibration

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
CalibratedClassifierCV(estimator=None, *, method="sigmoid", cv=None, ensemble="auto")
```

`method="isotonic"` needs more data (over about 1,000 samples) but is more flexible. Calibrate SVMs, naive Bayes and boosted trees when you need trustworthy probabilities. Passing `cv="prefit"` was deprecated in 1.6; wrap a fitted model in `sklearn.frozen.FrozenEstimator` instead.

#### sklearn.feature_extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
TfidfVectorizer(*, lowercase=True, stop_words=None, ngram_range=(1, 1), max_df=1.0, min_df=1,
                max_features=None, sublinear_tf=False)
```

Text vectorisers take a 1-D iterable of strings, not a 2-D array. In a `ColumnTransformer`, pass the column name as a string (`"text"`), not a list (`["text"]`), so the vectoriser receives 1-D input.

#### sklearn.multiclass and sklearn.multioutput

`OneVsRestClassifier`, `OneVsOneClassifier`, `MultiOutputClassifier`, `MultiOutputRegressor`, `ClassifierChain`, `RegressorChain` wrap single-output estimators for multiclass, multilabel and multi-target tasks.

#### sklearn.datasets

| Function | Description |
|---|---|
| `load_iris`, `load_wine`, `load_breast_cancer`, `load_digits`, `load_diabetes` | Small bundled toy datasets. Pass `return_X_y=True` and/or `as_frame=True`. |
| `fetch_california_housing`, `fetch_covtype`, `fetch_20newsgroups`, `fetch_openml(name=..., version=...)` | Downloaded and cached in `~/scikit_learn_data`. |
| `make_classification`, `make_regression`, `make_blobs`, `make_moons`, `make_circles` | Synthetic generators. |

`load_boston` was removed in 1.2 for ethical reasons; use `fetch_california_housing` instead.

#### Persistence

```python
import joblib
# pipe is any fitted estimator or pipeline
joblib.dump(pipe, "model.joblib")
pipe = joblib.load("model.joblib")
```

Pickles are only guaranteed to load with the same scikit-learn version, and loading a pickle can execute arbitrary code: never load untrusted files. The `skops` library (`skops.io.dump` / `skops.io.load`) provides a safer format, and `skl2onnx` exports to ONNX for language-neutral serving.

## Tutorials

### Tutorial 1: End-to-end tabular classification with a ColumnTransformer pipeline

Goal: predict survival on the Titanic dataset with mixed numeric and categorical columns, missing values, and proper cross-validated tuning.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load data as a DataFrame
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
num_cols = ["age", "fare", "sibsp", "parch"]
cat_cols = ["pclass", "sex", "embarked"]
X = X[num_cols + cat_cols].copy()
X["pclass"] = X["pclass"].astype(str)
X[cat_cols] = X[cat_cols].astype(object)   # plain strings + NaN for the imputer

# 2. Hold out a final test set; stratify to keep the survival rate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Preprocessing per column type
numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])

# 4. Full pipeline: preprocessing + model
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

# 5. Baseline cross-validated score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("baseline ROC AUC:", cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean())

# 6. Search over models AND their hyperparameters in one grid
param_grid = [
    {"clf": [LogisticRegression(max_iter=1000)], "clf__C": [0.01, 0.1, 1, 10]},
    {"clf": [HistGradientBoostingClassifier(random_state=0)],
     "clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31],
     "clf__min_samples_leaf": [10, 30]},
]
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
search.fit(X_train, y_train)
print("best:", search.best_params_, round(search.best_score_, 4))

# 7. Final, untouched evaluation on the test set
proba = search.predict_proba(X_test)[:, 1]
print("test ROC AUC:", round(roc_auc_score(y_test, proba), 4))
print(classification_report(y_test, search.predict(X_test)))

# 8. Inspect the output feature names of the fitted preprocessor
print(search.best_estimator_["pre"].get_feature_names_out())
```

What each step buys you:

- Steps 3 and 4 put **every** learned transformation inside the pipeline, so step 5's cross-validation never leaks test-fold statistics (medians, categories) into training.
- `add_indicator=True` turns "age is missing" into a feature, which is informative on Titanic.
- `handle_unknown="ignore"` prevents crashes when an unseen category appears at predict time.
- Step 6 shows that the final pipeline step itself can be a searchable parameter.
- The test set is used exactly once, at the end.

### Tutorial 2: Regression with feature engineering and a log-transformed target

Goal: predict California median house values, compare a linear model with engineered features against gradient boosting, and handle a skewed target.

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Linear model: splines on location and income give it non-linear capacity
spline_cols = ["MedInc", "Latitude", "Longitude"]
other_cols = [c for c in X.columns if c not in spline_cols]
linear_pre = ColumnTransformer([
    ("splines", SplineTransformer(n_knots=8, degree=3), spline_cols),
    ("scale", StandardScaler(), other_cols),
])
linear = TransformedTargetRegressor(
    regressor=make_pipeline(linear_pre, RidgeCV(alphas=np.logspace(-3, 3, 13))),
    func=np.log1p, inverse_func=np.expm1,
)

# Boosted trees need no scaling and no splines
gbrt = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=True,
                                     random_state=0)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
for name, model in [("splines+ridge", linear), ("hist-gbrt", gbrt)]:
    res = cross_validate(model, X_train, y_train, cv=cv,
                         scoring=["neg_root_mean_squared_error", "r2"], n_jobs=-1)
    print(f"{name:15s} RMSE={-res['test_neg_root_mean_squared_error'].mean():.3f} "
          f"R2={res['test_r2'].mean():.3f}")

# Fit the winner on all training data and evaluate once on test
gbrt.fit(X_train, y_train)
pred = gbrt.predict(X_test)
print("test RMSE:", round(root_mean_squared_error(y_test, pred), 3),
      "MAE:", round(mean_absolute_error(y_test, pred), 3),
      "R2:", round(r2_score(y_test, pred), 3))
print("iterations used after early stopping:", gbrt.n_iter_)
```

Explanation:

- `TransformedTargetRegressor` fits on `log1p(y)` and automatically inverts predictions, so metrics are reported on the original scale.
- `SplineTransformer` lets a linear model capture non-linear effects of income and geography while staying interpretable.
- `cross_validate` with two scorers gives RMSE and R^2 in one pass; the RMSE scorer is negated, hence the minus sign.
- `early_stopping=True` holds out 10% of the training data internally; `n_iter_` reports how many trees were actually used.

### Tutorial 3: Text classification with TF-IDF and a linear model

Goal: classify newsgroup posts, tune n-grams and regularisation, and inspect the most predictive words.

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

categories = ["alt.atheism", "comp.graphics", "sci.space", "talk.religion.misc"]
remove = ("headers", "footers", "quotes")   # avoid leaking metadata
train = fetch_20newsgroups(subset="train", categories=categories, remove=remove)
test = fetch_20newsgroups(subset="test", categories=categories, remove=remove)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)),
    ("clf", LinearSVC()),
])

grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [0.5, 1.0],
    "clf__C": [0.1, 0.5, 1.0],
}
search = GridSearchCV(pipe, grid, cv=5, scoring="f1_macro", n_jobs=-1)
search.fit(train.data, train.target)
print(search.best_params_, round(search.best_score_, 3))

pred = search.predict(test.data)
print(classification_report(test.target, pred, target_names=test.target_names))
ConfusionMatrixDisplay.from_predictions(test.target, pred, display_labels=test.target_names,
                                        xticks_rotation=45)
plt.tight_layout()
plt.show()

# Top words per class from the linear model's coefficients
best = search.best_estimator_
vocab = best["tfidf"].get_feature_names_out()
coefs = best["clf"].coef_
for i, label in enumerate(test.target_names):
    top = np.argsort(coefs[i])[-8:][::-1]
    print(label, "->", ", ".join(vocab[top]))
```

Why it works: TF-IDF produces a large sparse matrix where linear models excel. `sublinear_tf=True` dampens repeated words. Removing headers, footers and quotes prevents the model from "cheating" on email signatures. Swapping `LinearSVC` for `SGDClassifier(loss="log_loss")` gives probabilities, and `ComplementNB` is a fast alternative baseline.

### Tutorial 4: Customer segmentation with scaling, PCA and clustering

Goal: segment customers using unsupervised learning, choose k, and profile the clusters.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Synthetic "customers": recency, frequency, monetary value, tenure, discounts used
rng = np.random.default_rng(0)
centers = np.array([[10, 25, 900, 48, 2], [60, 5, 150, 12, 6], [30, 12, 400, 30, 1], [90, 2, 60, 6, 9]])
spread = np.array([5, 2, 80, 6, 1.5])
X = np.abs(np.vstack([rng.normal(loc=c, scale=spread, size=(500, 5)) for c in centers]))
df = pd.DataFrame(X, columns=["recency", "frequency", "monetary", "tenure", "discounts"])

# 1. Scale (k-means uses Euclidean distance) then reduce noise with PCA
prep = make_pipeline(StandardScaler(), PCA(n_components=0.9, random_state=0))
Z = prep.fit_transform(df)
print("PCA components kept:", prep[-1].n_components_)

# 2. Choose k by silhouette (higher better) and Davies-Bouldin (lower better)
scores = []
for k in range(2, 9):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    scores.append((k, silhouette_score(Z, labels), davies_bouldin_score(Z, labels)))
print(pd.DataFrame(scores, columns=["k", "silhouette", "davies_bouldin"]).round(3))
best_k = max(scores, key=lambda s: s[1])[0]

# 3. Final model and cluster profiles in original units
km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(Z)
df["segment"] = km.labels_
print(df.groupby("segment").mean().round(1))
print(df["segment"].value_counts().sort_index())

# 4. Assign new customers with the same fitted preprocessing
new = pd.DataFrame([[15, 20, 800, 40, 2]], columns=df.columns[:-1])
print("new customer segment:", km.predict(prep.transform(new))[0])
```

Explanation: scaling is mandatory because `monetary` is in hundreds while `discounts` is single digits; without it, k-means would cluster on money alone. PCA with a variance fraction drops noisy directions. Two internal metrics guard against picking k from one noisy curve. Profiling in original units (step 3) is what makes clusters actionable for a business audience.

### Tutorial 5: Imbalanced classification with threshold tuning and calibration

Goal: a fraud-style problem with 2% positives, where the default 0.5 threshold is wrong.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             make_scorer, confusion_matrix)

X, y = make_classification(n_samples=40_000, n_features=20, n_informative=6,
                           weights=[0.98, 0.02], flip_y=0.01, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)

base = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)

# Calibrate probabilities (class_weight distorts them), then tune the threshold for F1
calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
tuned = TunedThresholdClassifierCV(calibrated, scoring=make_scorer(f1_score),
                                   cv=StratifiedKFold(5, shuffle=True, random_state=0))
tuned.fit(X_tr, y_tr)

proba = tuned.predict_proba(X_te)[:, 1]
pred = tuned.predict(X_te)
print("threshold:", round(tuned.best_threshold_, 3))
print("PR AUC:", round(average_precision_score(y_te, proba), 3))
print("precision:", round(precision_score(y_te, pred), 3), "recall:", round(recall_score(y_te, pred), 3),
      "F1:", round(f1_score(y_te, pred), 3))
print(confusion_matrix(y_te, pred))
```

Key points: use PR AUC (`average_precision_score`) rather than accuracy; `class_weight="balanced"` helps the model learn the minority class but distorts probabilities, which calibration fixes; the decision threshold is a separate business choice that `TunedThresholdClassifierCV` makes data-driven.

## Performance & Best Practices

### Correctness first

1. **Split before anything else.** Keep a test set that is touched once.
2. **Everything learned from data goes in the pipeline**: imputation, scaling, encoding, feature selection, resampling (via imbalanced-learn's pipeline). This makes CV honest.
3. **Match the CV splitter to the data**: groups (`GroupKFold`), time (`TimeSeriesSplit`), imbalance (`StratifiedKFold`).
4. **Pick the metric that reflects the business cost** before tuning. Accuracy is rarely right for imbalanced problems.
5. **Start with a dumb baseline**: `DummyClassifier(strategy="prior")` / `DummyRegressor()` show what "no skill" scores.
6. **Set `random_state`** everywhere for reproducibility.

### Choosing a model

| Data | Strong first choice | Alternative |
|---|---|---|
| Tabular, under 10k rows | `LogisticRegression` / `RidgeCV` baseline, then `RandomForest` | `GradientBoosting*`, SVM |
| Tabular, over 10k rows | `HistGradientBoosting*` | XGBoost / LightGBM / CatBoost |
| Sparse text | `LinearSVC`, `LogisticRegression(solver="saga")`, `ComplementNB` | `SGDClassifier` |
| Streaming / out-of-core | `SGDClassifier.partial_fit`, `MiniBatchKMeans` | `IncrementalPCA` |
| Need interpretability | Linear models, shallow trees | `HistGradientBoosting` with `monotonic_cst` + partial dependence |
| Clustering, unknown k, noise | `HDBSCAN` | `DBSCAN`, `GaussianMixture` + BIC |

### Hyperparameter tuning strategy

1. **Fix the evaluation protocol** (splitter, metric) first.
2. **Coarse random search** over wide, log-scaled ranges (`scipy.stats.loguniform` for `C`, `alpha`, `learning_rate`). Random search finds good regions far faster than a grid when only a few parameters matter.
3. **Narrow grid** around the best region if needed.
4. **Successive halving** (`HalvingRandomSearchCV`) for expensive models: many candidates on small budgets, survivors get more data.
5. **Cache transformers** with `Pipeline(memory="cache_dir")` when only the final step's parameters vary.
6. **Nested CV** for an unbiased estimate of the whole tuning procedure:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
inner = StratifiedKFold(3, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)
search = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                      {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.001]}, cv=inner)
print(cross_val_score(search, X, y, cv=outer).mean())
```

Typical search spaces:

| Estimator | Parameters and ranges |
|---|---|
| `LogisticRegression` | `C`: loguniform(1e-3, 1e2) |
| `Ridge` / `Lasso` | `alpha`: loguniform(1e-4, 1e2) |
| `SVC` (rbf) | `C`: loguniform(1e-2, 1e3), `gamma`: loguniform(1e-4, 1e0) |
| `RandomForest*` | `max_features`: ["sqrt", 0.3, 0.5, 1.0], `min_samples_leaf`: 1-20, `max_depth`: [None, 8, 16] |
| `HistGradientBoosting*` | `learning_rate`: loguniform(0.01, 0.3), `max_leaf_nodes`: 15-127, `min_samples_leaf`: 5-100, `l2_regularization`: loguniform(1e-6, 10), `max_features`: 0.5-1.0 |
| `KNeighbors*` | `n_neighbors`: 1-50, `weights`: ["uniform", "distance"] |

For Bayesian optimisation use Optuna (`optuna.integration.OptunaSearchCV` lives in `optuna-integration`) or scikit-optimize's `BayesSearchCV`.

### Speed

- `n_jobs=-1` on forests, searches, `cross_validate`, `permutation_importance`. Avoid nesting `n_jobs=-1` at two levels (oversubscription); parallelise the outer loop.
- Control BLAS/OpenMP threads with `threadpoolctl.threadpool_limits(limits=1)` when running many small parallel jobs.
- Use `float32` where precision allows; many estimators preserve it and halve memory.
- Keep sparse data sparse: `OneHotEncoder` outputs sparse by default; avoid `.toarray()`; use `MaxAbsScaler` or `StandardScaler(with_mean=False)` on sparse input.
- Prefer `HistGradientBoosting*` over `GradientBoosting*` beyond about 10k rows; it is often 10x to 100x faster.
- For kernel SVMs on large data, approximate kernels with `Nystroem` or `RBFSampler` plus a linear model.
- `warm_start=True` on forests lets you add trees incrementally (`n_estimators += 100`) without refitting.
- `sklearnex.patch_sklearn()` can speed up KMeans, SVC, RandomForest, kNN and others on Intel/AMD CPUs.

### Production hygiene

- Persist the **whole pipeline**, not just the model, and record the scikit-learn version alongside it.
- Use `set_output(transform="pandas")` while developing to keep column names visible.
- Check `n_features_in_` and `feature_names_in_` to catch schema drift at inference.
- Pin dependency versions; retrain rather than unpickle across major upgrades.

## Common Errors & Troubleshooting

| Error / warning message | Cause | Fix |
|---|---|---|
| `ValueError: Expected 2D array, got 1D array instead` | Passed a single feature as a 1-D array. | `X.reshape(-1, 1)` or `df[["col"]]` (double brackets). |
| `ValueError: could not convert string to float: 'male'` | Raw strings reached a numeric estimator. | Encode with `OneHotEncoder` / `OrdinalEncoder` inside a `ColumnTransformer`. |
| `ValueError: Input X contains NaN.` (older: `Input contains NaN, infinity or a value too large for dtype('float64')`) | Estimator does not support missing values. | Add `SimpleImputer`, or use `HistGradientBoosting*` / trees (1.3+) which handle NaN. Check for `np.inf` too. |
| `ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.` | Unscaled features or too few iterations. | Add `StandardScaler`; raise `max_iter` (e.g. 1000); try another solver. |
| `NotFittedError: This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.` | Called `transform`/`predict` before `fit`, or on an unfitted clone. | Fit first; after a search use `search.best_estimator_`. |
| `ValueError: Found unknown categories ['x'] in column 0 during transform` | New category at predict time. | `OneHotEncoder(handle_unknown="ignore")` or `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`. |
| `ValueError: X has 10 features, but LogisticRegression is expecting 12 features as input.` | Train and predict inputs differ in shape. | Apply the same fitted pipeline to both; do not one-hot encode train and test separately with `pd.get_dummies`. |
| `UserWarning: X does not have valid feature names, but ... was fitted with feature names` | Fitted on a DataFrame, predicted on a NumPy array. | Pass a DataFrame with the same columns at predict time. |
| `ValueError: The feature names should match those that were passed during fit.` | Column names/order differ. | Reorder: `X_new = X_new[model.feature_names_in_]`. |
| `ValueError: Invalid parameter 'C' for estimator Pipeline(...)` | Grid key missing the step prefix. | Use `"logisticregression__C"` / `"clf__C"`; list with `pipe.get_params().keys()`. |
| `TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'` | Parameter renamed. | Use `sparse_output=False` (1.2+). |
| `TypeError: mean_squared_error() got an unexpected keyword argument 'squared'` | Removed in 1.6. | `root_mean_squared_error(y_true, y_pred)`. |
| `ImportError: cannot import name 'plot_confusion_matrix' from 'sklearn.metrics'` | Removed in 1.2. | `ConfusionMatrixDisplay.from_estimator(...)`. |
| `ImportError: cannot import name 'load_boston'` | Removed in 1.2. | `fetch_california_housing()` or `fetch_openml(...)`. |
| `AttributeError: 'OneHotEncoder' object has no attribute 'get_feature_names'` | Renamed. | `get_feature_names_out()`. |
| `TypeError: AdaBoostClassifier.__init__() got an unexpected keyword argument 'base_estimator'` | Renamed in 1.2, removed in 1.4. | `estimator=`. |
| `InvalidParameterError: The 'max_features' parameter of RandomForestClassifier must be ... Got 'auto' instead.` | `"auto"` removed in 1.3. | Use `"sqrt"` (classifier) or `1.0` (regressor). |
| `ValueError: Solver lbfgs supports only 'l2' or None penalties, got l1 penalty.` | Solver/penalty mismatch. | `solver="liblinear"` or `"saga"` for L1. |
| `ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class` | A fold or subset has one class. | Use stratified splitting; check labels. |
| `UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.` | Very rare class. | Fewer folds, merge classes, or `RepeatedStratifiedKFold` with care. |
| `UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.` | Model never predicts a class. | Set `zero_division=0` explicitly; fix imbalance / threshold. |
| `ValueError: y should be a 1d array, got an array of shape (n, 1) instead.` (`DataConversionWarning`) | `y` is a column vector. | `y.ravel()` or `df["target"]`. |
| `ValueError: Unknown label type: continuous` (older: `'continuous'`) | Float targets passed to a classifier. | Use a regressor, or bin/cast labels to int. |
| `MemoryError` / very slow `SVC` | Kernel SVM on large n. | `LinearSVC`, `SGDClassifier`, or `Nystroem` + linear model. |
| Joblib `UserWarning: A worker stopped while some jobs were given to the executor` | Worker crashed (memory). | Lower `n_jobs`, reduce `pre_dispatch`, or use less memory per job. |
| `InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.3.0 when using version 1.6.1` | Pickle from another version. | Retrain under the current version, or pin the old version. |
| CV score far above test score | Leakage (preprocessing or feature selection fitted outside CV) or wrong splitter. | Move all steps into the pipeline; use `GroupKFold` / `TimeSeriesSplit`. |

Debugging tips:

- `pipe[:-1].fit_transform(X_train)` shows exactly what the final estimator sees.
- `sklearn.set_config(transform_output="pandas")` makes intermediate outputs readable.
- `error_score="raise"` in searches surfaces the real exception instead of a silent NaN score.

## Interoperability

### pandas and Polars

All estimators accept DataFrames; column names are stored in `feature_names_in_`. `set_output(transform="pandas")` (1.2+) or `"polars"` (1.4+) returns DataFrames from transformers. `ColumnTransformer` selects by column name and `make_column_selector(dtype_include=...)` selects by dtype.

### NumPy, SciPy sparse and the Array API

NumPy arrays are the native format. Many estimators accept `scipy.sparse` CSR/CSC matrices (linear models, SVMs, naive Bayes, `TruncatedSVD`, text vectorisers). With `array_api_dispatch=True`, supported estimators run on PyTorch or CuPy arrays without conversion.

### Gradient boosting libraries

XGBoost (`XGBClassifier`), LightGBM (`LGBMClassifier`) and CatBoost (`CatBoostClassifier`) implement the scikit-learn estimator API, so they work in `Pipeline`, `GridSearchCV`, `cross_validate`, `StackingClassifier` and `permutation_importance`:

```python
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

X, y = load_breast_cancer(return_X_y=True)
for model in (XGBClassifier(n_estimators=300, learning_rate=0.05),
              LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1)):
    print(type(model).__name__, cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean())
```

### Deep learning

- **skorch** wraps PyTorch modules as scikit-learn estimators (`NeuralNetClassifier`).
- **scikeras** wraps Keras models (`KerasClassifier`).
- Embeddings from PyTorch/TensorFlow/sentence-transformers can be fed into scikit-learn classifiers, clusterers or `NearestNeighbors`.

### Imbalanced data

`imbalanced-learn` provides `SMOTE`, `RandomUnderSampler` and an `imblearn.pipeline.Pipeline` that applies resampling only during `fit`, which scikit-learn's own `Pipeline` cannot do.

### Explainability

SHAP's `TreeExplainer` supports scikit-learn tree ensembles; `shap.Explainer(model.predict, X)` works for any estimator. `sklearn.inspection` covers permutation importance and partial dependence natively.

### Tuning frameworks

Optuna, Ray Tune (`TuneSearchCV` from `tune-sklearn`), scikit-optimize (`BayesSearchCV`) and Hyperopt all drive scikit-learn estimators.

### Deployment and tracking

- **MLflow**: `mlflow.sklearn.log_model(pipe, "model")` and `mlflow.sklearn.autolog()`.
- **ONNX**: `skl2onnx.to_onnx(pipe, X[:1])` then serve with `onnxruntime`.
- **skops**: secure serialisation and Hugging Face Hub model cards.
- **Dask-ML / joblib-spark**: distribute searches across a cluster with `joblib.parallel_backend("dask")`.

## Cheat Sheet

### Data preparation

| Task | Code |
|---|---|
| Train/test split, stratified | `X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)` |
| Standardise | `StandardScaler().fit_transform(X)` |
| Scale with outliers | `RobustScaler()` |
| One-hot encode | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` |
| Ordinal encode with order | `OrdinalEncoder(categories=[["low", "mid", "high"]])` |
| High-cardinality categorical | `TargetEncoder()` |
| Impute median + indicator | `SimpleImputer(strategy="median", add_indicator=True)` |
| Impute with kNN | `KNNImputer(n_neighbors=5)` |
| Log transform feature | `FunctionTransformer(np.log1p, feature_names_out="one-to-one")` |
| Log transform target | `TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1)` |
| Polynomial features | `PolynomialFeatures(degree=2, include_bias=False)` |
| Per-column preprocessing | `ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])` |
| Select columns by dtype | `make_column_selector(dtype_include="number")` |
| DataFrame output | `sklearn.set_config(transform_output="pandas")` |

### Models

| Task | Code |
|---|---|
| Linear regression | `LinearRegression()` |
| Regularised regression, auto alpha | `RidgeCV(alphas=np.logspace(-3, 3, 13))` / `LassoCV(cv=5)` |
| Logistic regression | `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))` |
| Random forest | `RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)` |
| Fast gradient boosting | `HistGradientBoostingClassifier(max_iter=500, early_stopping=True)` |
| SVM | `make_pipeline(StandardScaler(), SVC(C=1, gamma="scale", probability=True))` |
| kNN | `make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance"))` |
| Naive Bayes for text | `make_pipeline(TfidfVectorizer(), ComplementNB())` |
| Stacking | `StackingClassifier(estimators=[...], final_estimator=LogisticRegression())` |
| Baseline | `DummyClassifier(strategy="most_frequent")` |
| Anomaly detection | `IsolationForest(contamination=0.01, random_state=0)` |
| K-means | `KMeans(n_clusters=5, n_init="auto", random_state=0)` |
| Density clustering | `HDBSCAN(min_cluster_size=20)` |
| PCA 95% variance | `PCA(n_components=0.95)` |
| 2-D visualisation | `TSNE(n_components=2, init="pca", random_state=0).fit_transform(X)` |

### Evaluation and tuning

| Task | Code |
|---|---|
| CV score | `cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()` |
| Multiple metrics | `cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1_macro"])` |
| Out-of-fold predictions | `cross_val_predict(model, X, y, cv=5, method="predict_proba")` |
| Grid search | `GridSearchCV(pipe, {"clf__C": [0.1, 1, 10]}, cv=5, n_jobs=-1).fit(X, y)` |
| Random search | `RandomizedSearchCV(pipe, dists, n_iter=50, cv=5, random_state=0)` |
| Results table | `pd.DataFrame(search.cv_results_).sort_values("rank_test_score")` |
| List scorers | `sklearn.metrics.get_scorer_names()` |
| Classification report | `print(classification_report(y_te, y_pred))` |
| Confusion matrix plot | `ConfusionMatrixDisplay.from_estimator(model, X_te, y_te)` |
| ROC curve | `RocCurveDisplay.from_estimator(model, X_te, y_te)` |
| RMSE | `root_mean_squared_error(y_te, y_pred)` |
| Tune decision threshold | `TunedThresholdClassifierCV(model, scoring="f1").fit(X, y)` |
| Calibrate probabilities | `CalibratedClassifierCV(model, method="isotonic", cv=5)` |
| Permutation importance | `permutation_importance(model, X_te, y_te, n_repeats=10)` |
| Partial dependence | `PartialDependenceDisplay.from_estimator(model, X, ["feat"])` |
| Learning curve | `LearningCurveDisplay.from_estimator(model, X, y, cv=5)` |

### Utilities

| Task | Code |
|---|---|
| Version and environment | `sklearn.show_versions()` |
| Pipeline params | `pipe.get_params().keys()` |
| Set nested param | `pipe.set_params(clf__C=10)` |
| Output feature names | `pipe[:-1].get_feature_names_out()` |
| Clone unfitted copy | `sklearn.base.clone(model)` |
| Save / load | `joblib.dump(pipe, "m.joblib")` / `joblib.load("m.joblib")` |
| Class weights | `sklearn.utils.class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)` |

## Further Resources

- Official documentation: https://scikit-learn.org/stable/
- User guide: https://scikit-learn.org/stable/user_guide.html
- API reference: https://scikit-learn.org/stable/api/index.html
- Release highlights and changelog: https://scikit-learn.org/stable/whats_new.html
- Example gallery: https://scikit-learn.org/stable/auto_examples/index.html
- Choosing the right estimator (flowchart): https://scikit-learn.org/stable/machine_learning_map.html
- Common pitfalls and recommended practices: https://scikit-learn.org/stable/common_pitfalls.html
- GitHub repository: https://github.com/scikit-learn/scikit-learn
- Discussions: https://github.com/scikit-learn/scikit-learn/discussions
- Free MOOC by the core developers: https://inria.github.io/scikit-learn-mooc/
- Paper: Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, 2011: https://jmlr.org/papers/v12/pedregosa11a.html
- Paper: Buitinck et al., "API design for machine learning software: experiences from the scikit-learn project", 2013: https://arxiv.org/abs/1309.0238
- imbalanced-learn: https://imbalanced-learn.org/
- skops: https://skops.readthedocs.io/
