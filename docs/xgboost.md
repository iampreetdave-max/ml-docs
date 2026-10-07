# XGBoost

> Scalable, regularised gradient boosting for tabular data, on CPU, GPU and clusters.

XGBoost ("eXtreme Gradient Boosting") is a gradient-boosted decision tree library that has been a fixture of winning Kaggle solutions and production tabular ML systems for a decade. It offers a low-level native API (`DMatrix` + `xgb.train`), a scikit-learn compatible API (`XGBClassifier`, `XGBRegressor`, `XGBRanker`), first-class GPU training, native categorical and missing-value support, and distributed training on Dask, Spark and Ray.

Covers XGBoost 2.x and 3.x (examples verified against 3.x). Breaking changes from 1.x are flagged inline: `device="cuda"` replaces `tree_method="gpu_hist"` and `gpu_id`, `tree_method="hist"` is the default, and `early_stopping_rounds` / `eval_metric` / `callbacks` belong in the scikit-learn constructor rather than in `fit()`.

## Overview

### What it is

XGBoost implements gradient boosting: an additive ensemble of regression trees, each one fitted to the gradient (and Hessian) of a loss function with respect to the current ensemble's predictions. Its distinguishing features are:

- **Second-order optimisation**: uses both gradients and Hessians, giving a closed-form optimal leaf weight and split gain.
- **Explicit regularisation** in the objective: L1 (`alpha`) and L2 (`lambda`) penalties on leaf weights plus a per-leaf cost (`gamma`).
- **Sparsity-aware split finding**: missing values learn a default direction at each split.
- **Histogram algorithm** (`tree_method="hist"`): features are bucketed into at most `max_bin` bins; fast on CPU and GPU.
- **Native categorical support** with partition-based splits.
- **Many objectives**: regression (squared, absolute, quantile, Huber, Poisson, Gamma, Tweedie), classification, ranking (LambdaMART), survival (Cox, AFT).
- **Constraints**: monotone and interaction constraints.
- **Distributed and out-of-core** training: Dask, PySpark, Ray (via `xgboost_ray`), external-memory `DMatrix`.
- **Bindings** for Python, R, JVM (Scala/Java), C/C++, Julia and more.

### History and maintainers

XGBoost was created by Tianqi Chen in 2014 as a research project in the Distributed (Deep) Machine Learning Community (DMLC) at the University of Washington. It rose to fame in the Higgs Boson Kaggle challenge, and the 2016 KDD paper "XGBoost: A Scalable Tree Boosting System" by Chen and Guestrin is the canonical reference. Today it is developed by the DMLC/XGBoost community with major contributions from NVIDIA and others. The license is Apache-2.0.

Version milestones:

| Version | Notable changes |
|---|---|
| 1.3 to 1.6 | Experimental categorical support, `QuantileDMatrix` precursor, `early_stopping_rounds` and `eval_metric` moved to the sklearn constructor (1.6). |
| 2.0 (2023) | `device` parameter (`"cpu"`, `"cuda"`, `"cuda:1"`), `hist` becomes default tree method, `gpu_hist` and `gpu_id` deprecated, multi-target trees (`multi_strategy`), quantile regression (`reg:quantileerror`), new LambdaMART implementation, `base_score` auto-estimated. |
| 2.1 (2024) | Python wheels for Linux ship with CUDA support (needs NCCL from pip), better external memory, column-split federated learning. |
| 3.0 (2025) | Removal of long-deprecated parameters and `fit()` arguments, `ExtMemQuantileDMatrix` for GPU external memory, minimum Python raised. |
| 3.x | Categorical support on by default (`enable_categorical=True` default in recent releases), continued GPU and external memory improvements. |

### When to use it

- Structured/tabular data with heterogeneous features: the default strong model for classification, regression and ranking.
- When you need **GPU training** with a mature, well-tested implementation.
- Learning-to-rank with LambdaMART (`XGBRanker`).
- When you need monotone constraints, custom objectives, survival objectives or quantile regression.
- Distributed training on Spark or Dask clusters.

### When not to use it

- Images, audio, raw text, sequences: use deep learning.
- Very small datasets (a few hundred rows): a regularised linear model or random forest may generalise better and is easier to explain.
- Strict extrapolation requirements: trees predict constants outside the training range.
- If you have many high-cardinality categoricals and little tuning time, CatBoost's ordered target statistics may work better out of the box; LightGBM is often faster on very wide CPU workloads.

### Where it fits

```text
pandas / Polars / NumPy / cuDF / Arrow   ->  feature engineering
                 |
     scikit-learn preprocessing (optional; trees need no scaling)
                 |
        XGBoost  (XGBClassifier / xgb.train)   <-- tuned with Optuna / sklearn search
                 |
   SHAP (pred_contribs) / sklearn.inspection    -> explanation
                 |
   save_model("model.json" / ".ubj"), MLflow, Triton FIL, ONNX  -> serving
```

## Installation

### pip

```bash
python -m pip install -U xgboost
```

- Linux x86_64 and aarch64 wheels include CUDA GPU support (2.1+); they pull in `nvidia-nccl-cu12` for multi-GPU. A CUDA 12 capable driver is required for GPU use; CPU use works anywhere.
- Windows wheels include CUDA support for single GPU.
- macOS wheels are CPU only. Apple Silicon is supported natively.
- A smaller CPU-only package is published as `xgboost-cpu` (Linux and Windows):

```bash
python -m pip install xgboost-cpu
```

### conda

```bash
conda install -c conda-forge py-xgboost          # CPU or GPU, picked by the solver
conda install -c conda-forge py-xgboost=*=cuda*  # force the CUDA build
conda install -c conda-forge py-xgboost=*=cpu*   # force CPU only
```

### Optional extras

```bash
python -m pip install "xgboost[scikit-learn,pandas,plotting]"   # sklearn API, pandas, matplotlib/graphviz
python -m pip install "xgboost[dask]"                           # distributed via Dask
python -m pip install "xgboost[pyspark]"                        # PySpark estimators
```

`plot_tree` / `to_graphviz` also need the Graphviz system binary (`brew install graphviz`, `apt install graphviz`, or the Windows installer).

### macOS OpenMP

If import fails with `Library not loaded: @rpath/libomp.dylib`, install OpenMP:

```bash
brew install libomp
```

### Verifying the install

```python
import xgboost as xgb
print(xgb.__version__)
print(xgb.build_info()["USE_CUDA"])   # True if this build supports CUDA
```

Smoke test, including a GPU check:

```python
import numpy as np
import xgboost as xgb

X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
clf = xgb.XGBClassifier(n_estimators=50).fit(X, y)
print("CPU ok, accuracy:", clf.score(X, y))

try:
    xgb.XGBClassifier(n_estimators=10, device="cuda").fit(X, y)
    print("GPU ok")
except xgb.core.XGBoostError as e:
    print("GPU not available:", str(e).splitlines()[0])
```

## Core Concepts

### Additive tree ensembles

The model predicts with a sum of K trees on top of a base score:

```text
y_hat_i = base_score + sum_{k=1..K} eta * f_k(x_i)
```

where each `f_k` is a regression tree mapping a row to a leaf weight, and `eta` (`learning_rate`) shrinks each tree's contribution. For classification the sum is a **margin** (log-odds), converted to a probability by the link function (sigmoid or softmax).

### The regularised objective

At iteration t XGBoost adds the tree `f_t` that minimises:

```text
Obj = sum_i loss(y_i, y_hat_i^(t-1) + f_t(x_i)) + Omega(f_t)
Omega(f) = gamma * T + 0.5 * lambda * sum_j w_j^2 + alpha * sum_j |w_j|
```

`T` is the number of leaves and `w_j` the leaf weights. A second-order Taylor expansion of the loss around the current predictions gives, with `g_i` the gradient and `h_i` the Hessian of the loss for row i:

```text
Obj ~ sum_j [ G_j * w_j + 0.5 * (H_j + lambda) * w_j^2 ] + gamma * T      (alpha = 0 for clarity)
G_j = sum of g_i in leaf j,   H_j = sum of h_i in leaf j
```

This is a simple quadratic in each `w_j`, so the **optimal leaf weight** and the resulting objective are closed-form:

```text
w_j* = - G_j / (H_j + lambda)
Obj* = -0.5 * sum_j G_j^2 / (H_j + lambda) + gamma * T
```

The **gain** of splitting a node into left (L) and right (R) children is:

```text
Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma
```

A split is made only if `Gain > 0`, which is why `gamma` (alias `min_split_loss`) acts as pruning. `min_child_weight` requires `H_L` and `H_R` to each exceed a threshold: for squared error the Hessian is 1 per row, so it is a minimum row count; for logistic loss it is `p(1-p)` summed, so confident regions need more rows.

The same mechanics in a few lines of NumPy, to make it concrete:

```python
import numpy as np

def leaf_weight(g, h, lam=1.0):
    return -g.sum() / (h.sum() + lam)

def split_gain(g, h, mask, lam=1.0, gamma=0.0):
    def score(gs, hs):
        return gs.sum() ** 2 / (hs.sum() + lam)
    return 0.5 * (score(g[mask], h[mask]) + score(g[~mask], h[~mask]) - score(g, h)) - gamma

# Squared error: g = pred - y, h = 1
y = np.array([1.0, 1.2, 3.1, 2.9])
pred = np.full_like(y, y.mean())
g, h = pred - y, np.ones_like(y)
mask = np.array([True, True, False, False])
print("gain:", split_gain(g, h, mask), "left weight:", leaf_weight(g[mask], h[mask]))
```

### Tree growth

- `grow_policy="depthwise"` (default): split all nodes at one depth before going deeper, limited by `max_depth` (default 6).
- `grow_policy="lossguide"`: always split the leaf with the highest gain (LightGBM-style), limited by `max_leaves`. Set `max_depth=0` for unlimited depth with lossguide.

Split finding methods (`tree_method`):

| Value | Description |
|---|---|
| `"hist"` | Histogram-based, default since 2.0, CPU and GPU. Features are pre-binned into `max_bin` (256) bins. |
| `"approx"` | Re-sketches quantiles every iteration; rarely needed now. |
| `"exact"` | Enumerates all split points; slow, small data only. CPU only. |
| `"auto"` | Same as `"hist"` in 2.x+. |

### Missing values

XGBoost treats `np.nan` (or a custom `missing` value) as missing. At every split it tries sending missing values left and right and keeps the better default direction. You usually should **not** impute before XGBoost.

### Categorical features

With pandas `category` dtype (or `feature_types=["c", ...]`) and `enable_categorical=True`, XGBoost splits categoricals directly. For a feature with few categories (at most `max_cat_to_onehot`) it uses one-vs-rest splits; otherwise it sorts categories by gradient statistics and searches for the best partition, limited by `max_cat_threshold`. Categorical support requires `tree_method="hist"` or `"approx"`.

### Learning rate and number of trees

`learning_rate` and `n_estimators` trade off: halving the learning rate roughly doubles the trees needed but usually generalises a little better. The standard approach is to fix a small learning rate (0.01 to 0.1), set a large `n_estimators`, and let **early stopping** on a validation set choose the actual number.

### Randomisation

Row subsampling (`subsample`) and column subsampling (`colsample_bytree`, `colsample_bylevel`, `colsample_bynode`) decorrelate trees and reduce overfitting, as in random forests.

### Two APIs

```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Native API: DMatrix + params dict + xgb.train
dtrain = xgb.DMatrix(X_tr, label=y_tr)
dvalid = xgb.DMatrix(X_va, label=y_va)
params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.05, "max_depth": 4}
booster = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dvalid, "valid")],
                    early_stopping_rounds=50, verbose_eval=False)
print("native best iteration:", booster.best_iteration)

# scikit-learn API: same engine, estimator interface
clf = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4,
                        eval_metric="logloss", early_stopping_rounds=50)
clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print("sklearn best iteration:", clf.best_iteration)
```

Naming differences between the two APIs:

| Native (`params` dict) | scikit-learn constructor |
|---|---|
| `eta` | `learning_rate` |
| `num_boost_round` (argument to `train`) | `n_estimators` |
| `lambda` | `reg_lambda` |
| `alpha` | `reg_alpha` |
| `min_split_loss` / `gamma` | `gamma` |
| `seed` | `random_state` |
| `nthread` | `n_jobs` |

### Margins, probabilities and base_score

`base_score` is the initial prediction. Since 2.0 it is estimated from the training labels automatically (for example the mean for regression, or the log-odds of the positive rate for logistic). `output_margin=True` in `Booster.predict` returns raw margins before the link function.

## API Reference

### Data containers

#### xgboost.DMatrix

```python
xgb.DMatrix(data, label=None, *, weight=None, base_margin=None, missing=None, silent=False,
            feature_names=None, feature_types=None, nthread=None, group=None, qid=None,
            label_lower_bound=None, label_upper_bound=None, feature_weights=None,
            enable_categorical=False)   # recent 3.x releases default this to True
```

The internal data structure for the native API. Accepts NumPy arrays, pandas / Polars DataFrames, SciPy CSR/CSC matrices, cuDF and CuPy (for GPU), PyArrow tables, and file paths in LIBSVM / CSV text format (`"train.libsvm?format=libsvm"`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | array-like | required | Feature matrix. |
| `label` | array-like | `None` | Target. For binary classification use 0/1; multiclass 0..K-1. |
| `weight` | array-like | `None` | Per-row weights (per-group for ranking). |
| `base_margin` | array-like | `None` | Initial margin per row (offset), e.g. `log(exposure)` for Poisson. |
| `missing` | float | `np.nan` | Value treated as missing. |
| `feature_names` | list of str | `None` | Inferred from DataFrame columns. |
| `feature_types` | list of str | `None` | `"q"` (quantitative), `"c"` (categorical), `"int"`, `"float"`, `"i"` (indicator). |
| `group` / `qid` | array-like | `None` | Ranking: group sizes or query id per row. |
| `label_lower_bound`, `label_upper_bound` | array-like | `None` | Interval-censored labels for `survival:aft`. |
| `enable_categorical` | bool | version dependent | Use pandas `category` dtype columns as categorical. Set it explicitly for portable code. |

Useful methods: `get_label()`, `set_label()`, `set_weight()`, `num_row()`, `num_col()`, `feature_names`, `slice(rindex)`, `save_binary(path)`.

```python
import numpy as np
import pandas as pd
import xgboost as xgb

df = pd.DataFrame({"x1": [1.0, 2.0, np.nan, 4.0], "color": pd.Categorical(["r", "g", "r", "b"])})
dm = xgb.DMatrix(df, label=[0, 1, 0, 1], enable_categorical=True)
print(dm.num_row(), dm.num_col(), dm.feature_names, dm.feature_types)
```

#### xgboost.QuantileDMatrix

```python
xgb.QuantileDMatrix(data, label=None, *, weight=None, base_margin=None, missing=None,
                    feature_names=None, feature_types=None, nthread=None, max_bin=None,
                    ref=None, group=None, qid=None, enable_categorical=False)
```

Builds the quantised histogram representation directly, without keeping a full-precision copy. Uses much less memory than `DMatrix` with `tree_method="hist"` (the only method it supports). For validation data pass `ref=dtrain` so it reuses the training bin boundaries. The sklearn estimators use it internally when `tree_method="hist"`.

```python
dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, max_bin=256)
dvalid = xgb.QuantileDMatrix(X_va, label=y_va, ref=dtrain)
```

#### External memory: DataIter and ExtMemQuantileDMatrix

For datasets larger than RAM (or GPU memory), subclass `xgb.DataIter`, implement `next(input_data)` and `reset()`, then build a `DMatrix(iterator)` or, in 3.0+, `xgb.ExtMemQuantileDMatrix(iterator, ...)`. Batches are fetched from your iterator and cached on disk or in host memory.

```python
import numpy as np
import xgboost as xgb

class BatchIter(xgb.DataIter):
    def __init__(self, n_batches=4):
        self._i, self._n = 0, n_batches
        super().__init__(cache_prefix="./xgb_cache")

    def next(self, input_data):
        if self._i == self._n:
            return False
        rng = np.random.default_rng(self._i)
        X = rng.random((10_000, 20))
        y = (X[:, 0] > 0.5).astype(int)
        input_data(data=X, label=y)
        self._i += 1
        return True

    def reset(self):
        self._i = 0

dtrain_ext = xgb.ExtMemQuantileDMatrix(BatchIter(), max_bin=256)
booster_ext = xgb.train({"objective": "binary:logistic", "tree_method": "hist"}, dtrain_ext,
                        num_boost_round=20)
```

### Training functions

#### xgboost.train

```python
xgb.train(params, dtrain, num_boost_round=10, *, evals=None, obj=None, maximize=None,
          early_stopping_rounds=None, evals_result=None, verbose_eval=True, xgb_model=None,
          callbacks=None, custom_metric=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | dict | required | Booster parameters (see the hyperparameter reference). |
| `dtrain` | DMatrix | required | Training data. |
| `num_boost_round` | int | `10` | Maximum boosting rounds. Note the low default. |
| `evals` | list of (DMatrix, str) | `None` | Datasets evaluated each round. Early stopping uses the **last** one. |
| `obj` | callable | `None` | Custom objective `obj(preds, dtrain) -> (grad, hess)`. |
| `custom_metric` | callable | `None` | Custom metric `f(preds, dtrain) -> (name, value)`. Replaces the older `feval`. |
| `maximize` | bool | `None` | Whether the custom metric should be maximised. |
| `early_stopping_rounds` | int | `None` | Stop if the last eval metric on the last eval set has not improved for this many rounds. |
| `evals_result` | dict | `None` | Filled in-place with the metric history. |
| `verbose_eval` | bool or int | `True` | Print every round, or every N rounds. |
| `xgb_model` | Booster or path | `None` | Continue training from an existing model. |
| `callbacks` | list | `None` | `TrainingCallback` instances. |

Returns a `Booster`. When early stopping triggers, `booster.best_iteration` and `booster.best_score` are set; the returned model still contains the extra rounds, but `predict` on the sklearn wrapper (and the native `predict` with `iteration_range`) can use only the best ones.

```python
history = {}
booster = xgb.train(
    {"objective": "reg:squarederror", "eval_metric": ["rmse", "mae"], "eta": 0.05, "max_depth": 6},
    dtrain, num_boost_round=2000,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=100, evals_result=history, verbose_eval=200,
)
print(booster.best_iteration, history["valid"]["rmse"][booster.best_iteration])
```

With several metrics in `eval_metric`, early stopping uses the **last** metric in the list.

#### xgboost.cv

```python
xgb.cv(params, dtrain, num_boost_round=10, *, nfold=3, stratified=False, folds=None, metrics=(),
       obj=None, maximize=None, early_stopping_rounds=None, fpreproc=None, as_pandas=True,
       verbose_eval=None, show_stdv=True, seed=0, callbacks=None, shuffle=True, custom_metric=None)
```

Runs k-fold CV and returns a DataFrame with columns like `train-logloss-mean`, `train-logloss-std`, `test-logloss-mean`, `test-logloss-std`, one row per round (truncated at the best round when early stopping is used).

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
d = xgb.DMatrix(X, label=y)
cv = xgb.cv({"objective": "binary:logistic", "eval_metric": "auc", "eta": 0.05, "max_depth": 4},
            d, num_boost_round=1000, nfold=5, stratified=True, early_stopping_rounds=50, seed=0)
print(len(cv), "rounds; best test AUC:", cv["test-auc-mean"].iloc[-1])
```

### xgboost.Booster

The trained model object returned by `train`, and available from sklearn estimators via `get_booster()`.

| Method | Signature | Description |
|---|---|---|
| `predict` | `(data, *, output_margin=False, pred_leaf=False, pred_contribs=False, approx_contribs=False, pred_interactions=False, validate_features=True, training=False, iteration_range=(0, 0), strict_shape=False)` | Predict on a `DMatrix`. `pred_contribs=True` returns SHAP values (last column is the bias). `pred_interactions=True` returns SHAP interaction values. `pred_leaf=True` returns leaf indices. |
| `inplace_predict` | `(data, *, iteration_range=(0, 0), predict_type="value", missing=nan, validate_features=True, base_margin=None, strict_shape=False)` | Predict directly on NumPy / pandas / CuPy / cuDF without building a `DMatrix`; thread-safe and faster. `predict_type="margin"` for raw scores. |
| `save_model` | `(fname)` | Save in JSON (`.json`) or UBJSON (`.ubj`) format; the extension picks the format. Stable across versions. |
| `load_model` | `(fname)` | Load a saved model (also accepts a bytearray). |
| `save_raw` | `(raw_format="ubj")` | Serialise to bytes. |
| `get_score` | `(fmap="", importance_type="weight")` | Importance dict. Types: `"weight"`, `"gain"`, `"cover"`, `"total_gain"`, `"total_cover"`. |
| `get_fscore` | `(fmap="")` | Same as `get_score(importance_type="weight")`. |
| `trees_to_dataframe` | `(fmap="")` | All tree nodes as a pandas DataFrame. |
| `get_dump` | `(fmap="", with_stats=False, dump_format="text")` | Text / JSON dump of each tree. |
| `set_param` | `(params, value=None)` | Change parameters, e.g. after loading. |
| `num_boosted_rounds` | `()` | Number of rounds in the model. |
| `num_features` | `()` | Number of input features. |
| `__getitem__` | `booster[0:100]` | Slice a model to a range of trees. |
| `copy`, `attributes`, `set_attr` | | Utilities. |
| `best_iteration`, `best_score` | attributes | Set by early stopping. |
| `feature_names`, `feature_types` | attributes | Schema stored in the model. |

```python
# continuing with booster and X from the training examples above
import numpy as np
dtest = xgb.DMatrix(X[:5])
it = (0, booster.best_iteration + 1)                      # only the best trees
proba = booster.predict(dtest, iteration_range=it)
margin = booster.predict(dtest, output_margin=True, iteration_range=it)
shap_vals = booster.predict(dtest, pred_contribs=True, iteration_range=it)   # (5, n_features + 1)
assert np.allclose(shap_vals.sum(axis=1), margin, atol=1e-4)
booster.save_model("model.json")
loaded = xgb.Booster()
loaded.load_model("model.json")
```

For any binary/regression model, the SHAP values plus bias returned by `pred_contribs=True` sum to the margin (`output_margin=True`) over the same `iteration_range`.

### scikit-learn API

#### XGBClassifier / XGBRegressor

```python
xgb.XGBRegressor(*, objective="reg:squarederror", **kwargs)
xgb.XGBClassifier(*, objective="binary:logistic", **kwargs)
# Shared constructor parameters (all default to None = library default unless shown):
#   n_estimators, max_depth, max_leaves, max_bin, grow_policy, learning_rate, verbosity,
#   booster, tree_method, n_jobs, gamma, min_child_weight, max_delta_step, subsample,
#   sampling_method, colsample_bytree, colsample_bylevel, colsample_bynode, reg_alpha,
#   reg_lambda, scale_pos_weight, base_score, random_state, missing=np.nan,
#   num_parallel_tree, monotone_constraints, interaction_constraints, importance_type,
#   device, validate_parameters, enable_categorical, feature_types, feature_weights,
#   max_cat_to_onehot, max_cat_threshold, multi_strategy, eval_metric,
#   early_stopping_rounds, callbacks, **kwargs (any native parameter)
```

`fit` signature:

```python
XGBClassifier.fit(X, y, *, sample_weight=None, base_margin=None, eval_set=None, verbose=True,
                  xgb_model=None, sample_weight_eval_set=None, base_margin_eval_set=None,
                  feature_weights=None)
```

| `fit` parameter | Description |
|---|---|
| `eval_set` | List of `(X, y)` pairs evaluated each round. Early stopping uses the last one. |
| `verbose` | `True`, `False`, or an int N to print every N rounds. |
| `sample_weight` | Per-row weights. |
| `xgb_model` | Continue training from a Booster / model file / fitted estimator. |

Deprecation: `early_stopping_rounds`, `eval_metric` and `callbacks` were accepted by `fit()` in 1.x. They moved to the constructor in 1.6 and the `fit()` arguments are removed in current releases. Passing them to `fit()` now raises `TypeError: fit() got an unexpected keyword argument 'early_stopping_rounds'`.

Methods and attributes:

| Member | Description |
|---|---|
| `predict(X, *, output_margin=False, validate_features=True, base_margin=None, iteration_range=None)` | Class labels / values. Uses `best_iteration` automatically when early stopping was used. |
| `predict_proba(X, ...)` | Class probabilities (classifier). |
| `apply(X)` | Leaf indices per tree. |
| `evals_result()` | Metric history dict: `{"validation_0": {"logloss": [...]}}`. |
| `best_iteration`, `best_score` | Set when early stopping was used. |
| `feature_importances_` | Normalised importances using `importance_type` (default `"gain"` for tree boosters). |
| `get_booster()` | Underlying `Booster`. |
| `save_model(path)` / `load_model(path)` | JSON / UBJ persistence including sklearn metadata. |
| `classes_`, `n_classes_`, `n_features_in_`, `feature_names_in_` | sklearn-style attributes. |
| `get_params()` / `set_params()` | Hyperparameter access, so it works in `GridSearchCV`. |

Label requirement: `XGBClassifier` expects labels encoded as `0 .. n_classes-1`. Use `sklearn.preprocessing.LabelEncoder` for string or non-contiguous labels.

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=0)

clf = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=1, reg_lambda=1.0, eval_metric="auc", early_stopping_rounds=100,
    n_jobs=-1, random_state=0,
)
clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
print("best iteration:", clf.best_iteration, "valid AUC:", round(clf.best_score, 4))
print("test AUC:", round(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]), 4))
```

#### XGBRanker

```python
xgb.XGBRanker(*, objective="rank:ndcg", **kwargs)
XGBRanker.fit(X, y, *, group=None, qid=None, sample_weight=None, base_margin=None, eval_set=None,
              eval_group=None, eval_qid=None, verbose=True, xgb_model=None,
              sample_weight_eval_set=None, base_margin_eval_set=None, feature_weights=None)
```

Learning to rank with LambdaMART. Pass `qid` (query id per row, rows sorted by qid) or `group` (sizes of consecutive groups). Relevant extra parameters: `lambdarank_pair_method` (`"topk"` or `"mean"`), `lambdarank_num_pair_per_sample`, `lambdarank_unbiased`, `ndcg_exp_gain`. `predict` returns relevance scores; sort them within each query. See Tutorial 3.

#### XGBRFClassifier / XGBRFRegressor

Random forests trained with XGBoost's engine: one boosting round of `num_parallel_tree=n_estimators` trees, with `learning_rate=1`, `subsample=0.8`, `colsample_bynode=0.8` defaults. Useful as a fast GPU random forest.

```python
rf = xgb.XGBRFClassifier(n_estimators=300, max_depth=8, subsample=0.8, colsample_bynode=0.8)
```

### Callbacks (xgboost.callback)

| Callback | Signature | Purpose |
|---|---|---|
| `EarlyStopping` | `(*, rounds, metric_name=None, data_name=None, maximize=None, save_best=False, min_delta=0.0)` | Early stopping with control over which metric/dataset and tolerance. `save_best=True` truncates the model to the best iteration. |
| `LearningRateScheduler` | `(learning_rates)` | List or callable `f(epoch) -> lr`. |
| `EvaluationMonitor` | `(rank=0, period=1, show_stdv=False)` | Print metrics every `period` rounds. |
| `TrainingCheckPoint` | `(directory, name="model", as_pickle=False, interval=100)` | Periodically save the model. |
| `TrainingCallback` | base class | Subclass and override `before_training`, `after_iteration` (return True to stop), `after_training`. |

```python
import xgboost as xgb

es = xgb.callback.EarlyStopping(rounds=50, metric_name="logloss", data_name="valid",
                                save_best=True, min_delta=1e-4)
lr = xgb.callback.LearningRateScheduler(lambda epoch: 0.1 * (0.99 ** epoch))
booster = xgb.train({"objective": "binary:logistic", "eval_metric": "logloss"}, dtrain,
                    num_boost_round=2000, evals=[(dvalid, "valid")], callbacks=[es, lr],
                    verbose_eval=False)

# sklearn API: callbacks go in the constructor
clf_cb = xgb.XGBClassifier(n_estimators=2000,
                           callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)])
```

A custom callback:

```python
class PrintEvery(xgb.callback.TrainingCallback):
    def __init__(self, period):
        self.period = period

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period == 0:
            last = {d: {m: v[-1] for m, v in ms.items()} for d, ms in evals_log.items()}
            print(epoch, last)
        return False  # returning True stops training
```

### Plotting and interpretation helpers

```python
xgb.plot_importance(booster, *, ax=None, height=0.2, importance_type="weight",
                    max_num_features=None, show_values=True, ...)
xgb.plot_tree(booster, *, num_trees=0, rankdir=None, ax=None, ...)   # requires graphviz
xgb.to_graphviz(booster, *, num_trees=0, rankdir=None, ...)
```

```python
import matplotlib.pyplot as plt
xgb.plot_importance(clf, importance_type="gain", max_num_features=15)
plt.tight_layout()
plt.show()
```

`importance_type` meanings:

| Type | Meaning |
|---|---|
| `"weight"` | Number of times a feature is used to split. |
| `"gain"` | Average gain of splits using the feature (sklearn default for `feature_importances_`). |
| `"total_gain"` | Total gain across all splits. |
| `"cover"` | Average number of samples (Hessian) affected by splits. |
| `"total_cover"` | Total coverage. |

Gain-based importances are biased toward high-cardinality features; prefer SHAP (`pred_contribs=True` or the `shap` package) or permutation importance for decisions.

### Configuration

```python
xgb.set_config(verbosity=2)          # global
print(xgb.get_config())
with xgb.config_context(verbosity=0):
    pass                              # silence inside this block
```

`verbosity`: 0 silent, 1 warning (default), 2 info, 3 debug.

### Distributed training

#### Dask

```python
from dask.distributed import Client, LocalCluster
import dask.array as da
import xgboost as xgb

with LocalCluster(n_workers=4) as cluster, Client(cluster) as client:
    X = da.random.random((1_000_000, 20), chunks=(100_000, 20))
    y = (X[:, 0] > 0.5).astype(int)
    clf = xgb.dask.DaskXGBClassifier(n_estimators=200, tree_method="hist")
    clf.client = client
    clf.fit(X, y)
    preds = clf.predict(X).compute()
```

The functional form is `xgb.dask.train(client, params, xgb.dask.DaskDMatrix(client, X, y), num_boost_round=...)`, and `DaskQuantileDMatrix` saves memory.

#### PySpark

```python
from xgboost.spark import SparkXGBClassifier

clf = SparkXGBClassifier(features_col="features", label_col="label", num_workers=4,
                         device="cuda")   # or omit device for CPU
# model = clf.fit(train_df); preds = model.transform(test_df)
```

`SparkXGBRegressor` and `SparkXGBRanker` follow the same pattern. Ray users can use the separate `xgboost_ray` package or Ray Train's `XGBoostTrainer`.

### Full hyperparameter reference

All parameters can be passed in the native `params` dict. In the sklearn API use the sklearn name where one exists, or pass the native name as a keyword argument (it is forwarded via `**kwargs`).

#### General parameters

| Parameter (sklearn name) | Type | Default | Description |
|---|---|---|---|
| `booster` | str | `"gbtree"` | `"gbtree"`, `"gblinear"` (linear model boosting), `"dart"` (trees with dropout). |
| `device` | str | `"cpu"` | `"cpu"`, `"cuda"`, `"cuda:<ordinal>"`, `"gpu"`. Added in 2.0; replaces `gpu_id` and `tree_method="gpu_hist"`. |
| `verbosity` | int | `1` | 0 to 3. |
| `nthread` (`n_jobs`) | int | all cores | CPU threads. |
| `seed` (`random_state`) | int | `0` | Random seed. |
| `validate_parameters` | bool | `True` for Python | Warn about unused/misspelled parameters. |
| `disable_default_eval_metric` | bool | `False` | Do not add the default metric. |

#### Tree booster parameters

| Parameter (sklearn name) | Type | Default | Range | Description |
|---|---|---|---|---|
| `eta` (`learning_rate`) | float | `0.3` | (0, 1] | Shrinkage per tree. Typical 0.01 to 0.3. |
| `num_boost_round` (`n_estimators`) | int | `10` native / `100` sklearn | >= 1 | Number of boosting rounds. |
| `gamma` / `min_split_loss` | float | `0` | [0, inf) | Minimum loss reduction to make a split. Higher is more conservative. |
| `max_depth` | int | `6` | [0, inf) | Maximum tree depth. 0 means no limit (lossguide only). |
| `min_child_weight` | float | `1` | [0, inf) | Minimum sum of Hessian in a child. Higher is more conservative. |
| `max_delta_step` | float | `0` | [0, inf) | Caps leaf weight updates; 1 to 10 can help extremely imbalanced logistic regression. |
| `subsample` | float | `1` | (0, 1] | Row sampling ratio per tree. |
| `sampling_method` | str | `"uniform"` | | `"gradient_based"` (GPU only) samples rows by gradient magnitude, allowing very low `subsample`. |
| `colsample_bytree` | float | `1` | (0, 1] | Column sampling per tree. |
| `colsample_bylevel` | float | `1` | (0, 1] | Column sampling per depth level (multiplies bytree). |
| `colsample_bynode` | float | `1` | (0, 1] | Column sampling per split (multiplies the above). |
| `lambda` (`reg_lambda`) | float | `1` | [0, inf) | L2 regularisation on leaf weights. |
| `alpha` (`reg_alpha`) | float | `0` | [0, inf) | L1 regularisation on leaf weights. |
| `tree_method` | str | `"auto"` (= `"hist"`) | | `"hist"`, `"approx"`, `"exact"`. `"gpu_hist"` is deprecated: use `tree_method="hist", device="cuda"`. |
| `scale_pos_weight` | float | `1` | > 0 | Weight of positive class; common heuristic `sum(negatives) / sum(positives)`. |
| `grow_policy` | str | `"depthwise"` | | `"depthwise"` or `"lossguide"`. |
| `max_leaves` | int | `0` | [0, inf) | Maximum leaves (0 = no limit). Relevant for lossguide. |
| `max_bin` | int | `256` | >= 2 | Histogram bins per feature. Higher is more precise but slower. |
| `num_parallel_tree` | int | `1` | >= 1 | Trees per round; >1 gives boosted random forests. |
| `monotone_constraints` | str, tuple, dict | `None` | -1, 0, 1 | E.g. `"(1,0,-1)"` or `{"price": -1}` (dict by feature name). |
| `interaction_constraints` | str or list | `None` | | Nested lists of feature indices/names allowed to interact, e.g. `[[0, 1], [2, 3, 4]]`. |
| `multi_strategy` | str | `"one_output_per_tree"` | | `"multi_output_tree"` builds vector-leaf trees for multi-target / multiclass (hist only, experimental). |
| `max_cat_to_onehot` | int | `4` | | Categorical features with fewer categories use one-hot style splits. |
| `max_cat_threshold` | int | `64` | | Max categories considered per partition split. |
| `refresh_leaf`, `process_type`, `updater` | | | | Advanced: refresh or prune an existing model. |

#### DART booster parameters (`booster="dart"`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rate_drop` | float | `0.0` | Fraction of previous trees dropped each round. |
| `skip_drop` | float | `0.0` | Probability of skipping dropout in a round. |
| `one_drop` | bool | `False` | Always drop at least one tree. |
| `sample_type` | str | `"uniform"` | `"uniform"` or `"weighted"` tree selection. |
| `normalize_type` | str | `"tree"` | `"tree"` or `"forest"` weight normalisation. |

DART is slower and cannot use early stopping reliably; predictions use all trees.

#### Linear booster parameters (`booster="gblinear"`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lambda`, `alpha` | float | `0`, `0` | L2 and L1 regularisation on weights. |
| `updater` | str | `"shotgun"` | `"shotgun"` (parallel coordinate descent) or `"coord_descent"`. |
| `feature_selector` | str | `"cyclic"` | `"cyclic"`, `"shuffle"`, `"random"`, `"greedy"`, `"thrifty"`. |
| `top_k` | int | `0` | Features considered by greedy/thrifty selectors. |

#### Learning task parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `objective` | str or callable | `"reg:squarederror"` | Loss function (table below). |
| `base_score` | float | estimated | Initial prediction. Auto-estimated since 2.0. |
| `eval_metric` | str or list | depends on objective | Metric(s) for evaluation sets (table below). |
| `num_class` | int | | Required for `multi:*` objectives in the native API (sklearn infers it). |
| `quantile_alpha` | float or list | | Quantile(s) for `reg:quantileerror`. |
| `huber_slope` | float | `1.0` | Delta for `reg:pseudohubererror`. |
| `tweedie_variance_power` | float | `1.5` | In (1, 2) for `reg:tweedie`. |
| `aft_loss_distribution`, `aft_loss_distribution_scale` | str, float | `"normal"`, `1.0` | For `survival:aft`. |
| `lambdarank_pair_method` | str | `"topk"` | Ranking pair construction. |
| `lambdarank_num_pair_per_sample` | int | | Pairs per document (or top-k cut-off). |
| `lambdarank_unbiased` | bool | `False` | Position-debiasing for click data. |

Objectives:

| Objective | Task | Output |
|---|---|---|
| `reg:squarederror` | Regression, squared loss | value |
| `reg:squaredlogerror` | Regression on log scale (labels > -1) | value |
| `reg:absoluteerror` | Regression, L1 loss (2.0+) | value (median) |
| `reg:pseudohubererror` | Robust regression | value |
| `reg:quantileerror` | Quantile regression (2.0+), use `quantile_alpha` | quantile(s) |
| `reg:logistic` | Regression to [0, 1] | probability-like |
| `reg:gamma`, `reg:tweedie`, `count:poisson` | Positive / count targets (log link) | mean |
| `binary:logistic` | Binary classification | probability |
| `binary:logitraw` | Binary classification | margin |
| `binary:hinge` | Binary classification, hinge loss | 0/1 |
| `multi:softprob` | Multiclass | probability matrix |
| `multi:softmax` | Multiclass | class index |
| `rank:ndcg`, `rank:map`, `rank:pairwise` | Learning to rank | score |
| `survival:cox` | Cox proportional hazards (negative label = censored) | hazard ratio |
| `survival:aft` | Accelerated failure time (interval-censored) | time |

Evaluation metrics:

| Metric | Use |
|---|---|
| `rmse`, `rmsle`, `mae`, `mape`, `mphe` | Regression |
| `logloss`, `error`, `error@t` | Binary classification (error at threshold t) |
| `auc`, `aucpr` | Ranking quality for binary (also multiclass AUC) |
| `mlogloss`, `merror` | Multiclass |
| `ndcg`, `map`, `ndcg@n`, `map@n`, `pre@n` | Ranking |
| `poisson-nloglik`, `gamma-nloglik`, `gamma-deviance`, `tweedie-nloglik` | GLM-style |
| `cox-nloglik`, `aft-nloglik`, `interval-regression-accuracy` | Survival |
| `quantile` | Quantile regression (pinball loss) |

## Tutorials

### Tutorial 1: Binary classification with categorical features, early stopping and SHAP

Goal: predict income above 50K on the Adult census dataset using native categorical support, a proper validation split, early stopping and SHAP explanations.

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# 1. Load data; convert string columns to pandas category dtype
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
y = (y == ">50K").astype(int)
cat_cols = X.select_dtypes(include=["object", "category"]).columns
X[cat_cols] = X[cat_cols].astype("category")

# 2. Train / validation / test split (validation drives early stopping only)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=0.2,
                                                      stratify=y_tmp, random_state=0)

# 3. Model: small learning rate, many trees, early stopping on validation AUC
clf = xgb.XGBClassifier(
    tree_method="hist",
    enable_categorical=True,      # required in 2.x; default in recent 3.x
    max_cat_to_onehot=1,          # always use partition splits for categoricals
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric=["logloss", "auc"],   # early stopping uses the LAST metric: auc
    early_stopping_rounds=200,
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=500)
print("best iteration:", clf.best_iteration, "best valid AUC:", round(clf.best_score, 4))

# 4. Evaluate once on the untouched test set (predict uses best_iteration automatically)
proba = clf.predict_proba(X_test)[:, 1]
print("test ROC AUC:", round(roc_auc_score(y_test, proba), 4),
      "PR AUC:", round(average_precision_score(y_test, proba), 4))
print(classification_report(y_test, (proba >= 0.5).astype(int), digits=3))

# 5. Learning curves from the eval history
hist = clf.evals_result()
print("final train/valid logloss:",
      round(hist["validation_0"]["logloss"][clf.best_iteration], 4),
      round(hist["validation_1"]["logloss"][clf.best_iteration], 4))

# 6. SHAP values from XGBoost itself (no extra package needed)
dtest = xgb.DMatrix(X_test, enable_categorical=True)
contribs = clf.get_booster().predict(dtest, pred_contribs=True,
                                     iteration_range=(0, clf.best_iteration + 1))
shap_df = pd.DataFrame(contribs[:, :-1], columns=X_test.columns)
print(shap_df.abs().mean().sort_values(ascending=False).head(10))

# 7. Persist in the stable JSON format
clf.save_model("adult_xgb.json")
restored = xgb.XGBClassifier()
restored.load_model("adult_xgb.json")
assert np.allclose(restored.predict_proba(X_test), clf.predict_proba(X_test))
```

What each step does:

- **Step 1**: XGBoost reads pandas `category` dtype directly; no one-hot encoding is needed, and missing values (`NaN`) stay as they are.
- **Step 2**: a three-way split keeps the test set honest. Early stopping "peeks" at the validation set, so its score is slightly optimistic.
- **Step 3**: `eval_metric` and `early_stopping_rounds` are constructor arguments in current XGBoost. With two eval sets, early stopping monitors the last one (`validation_1`).
- **Step 6**: `pred_contribs=True` returns exact TreeSHAP values; the mean absolute SHAP value is a far more reliable importance measure than `feature_importances_`.
- **Step 7**: JSON/UBJ model files load across XGBoost versions; pickles do not.

### Tutorial 2: Regression with the native API, cross-validation and monotone constraints

Goal: predict California house prices, use `xgb.cv` to pick the number of rounds, enforce that price increases with median income, and get prediction intervals with quantile regression.

```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref=dtrain)

# 1. Monotone constraint: +1 on MedInc, unconstrained elsewhere (dict by feature name)
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 2.0,
    "monotone_constraints": {"MedInc": 1},
    "seed": 0,
}

# 2. 5-fold CV with early stopping chooses the number of rounds
cv = xgb.cv(params, xgb.DMatrix(X_train, label=y_train), num_boost_round=5000, nfold=5,
            early_stopping_rounds=100, seed=0, verbose_eval=False)
n_rounds = len(cv)
print("CV rounds:", n_rounds, "CV RMSE:", round(cv["test-rmse-mean"].iloc[-1], 4),
      "+/-", round(cv["test-rmse-std"].iloc[-1], 4))

# 3. Refit on all training data with that many rounds
model = xgb.train(params, dtrain, num_boost_round=n_rounds)
pred = model.predict(dtest)
print("test RMSE:", round(root_mean_squared_error(y_test, pred), 4),
      "MAE:", round(mean_absolute_error(y_test, pred), 4))

# 4. Check the monotone constraint empirically
probe = X_test.iloc[:200].copy()
low = model.inplace_predict(probe.assign(MedInc=2.0))
high = model.inplace_predict(probe.assign(MedInc=8.0))
print("constraint holds for all rows:", bool(np.all(high >= low)))

# 5. Prediction intervals with multi-quantile regression (one model, three quantiles)
q_params = {"objective": "reg:quantileerror", "quantile_alpha": np.array([0.1, 0.5, 0.9]),
            "tree_method": "hist", "eta": 0.05, "max_depth": 6, "seed": 0}
q_model = xgb.train(q_params, dtrain, num_boost_round=800)
q_pred = q_model.predict(dtest)                   # shape (n_test, 3)
coverage = np.mean((y_test.values >= q_pred[:, 0]) & (y_test.values <= q_pred[:, 2]))
print("80% interval empirical coverage:", round(coverage, 3))
```

Notes:

- `QuantileDMatrix` with `ref=dtrain` reuses the training histogram bins for the test set and saves memory.
- `xgb.cv` returns one row per round up to the best one, so `len(cv)` is the chosen number of rounds.
- Monotone constraints trade a little accuracy for guaranteed sensible behaviour, which regulators and stakeholders often require.
- `reg:quantileerror` with a vector `quantile_alpha` trains all quantiles in one model (2.0+).

### Tutorial 3: Learning to rank with XGBRanker

Goal: rank documents per query by relevance with LambdaMART and evaluate NDCG@10.

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score

rng = np.random.default_rng(0)
n_queries, docs_per_query, n_features = 300, 30, 12
qid = np.repeat(np.arange(n_queries), docs_per_query)
X = rng.normal(size=(len(qid), n_features))
true_score = 1.5 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + rng.normal(scale=0.5, size=len(qid))
y = np.clip(np.digitize(true_score, [-1, 0.5, 1.5, 2.5]), 0, 4)   # graded relevance 0..4

# Split by query, never by row
train_q = qid < 240
X_tr, y_tr, q_tr = X[train_q], y[train_q], qid[train_q]
X_te, y_te, q_te = X[~train_q], y[~train_q], qid[~train_q]

ranker = xgb.XGBRanker(
    objective="rank:ndcg",
    eval_metric="ndcg@10",
    lambdarank_pair_method="topk",
    lambdarank_num_pair_per_sample=10,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    tree_method="hist",
    early_stopping_rounds=50,
)
# rows must be grouped (sorted) by qid; they already are here
ranker.fit(X_tr, y_tr, qid=q_tr, eval_set=[(X_te, y_te)], eval_qid=[q_te], verbose=100)

scores = ranker.predict(X_te)
df = pd.DataFrame({"qid": q_te, "y": y_te, "score": scores})
ndcgs = [ndcg_score([g["y"].to_numpy()], [g["score"].to_numpy()], k=10) for _, g in df.groupby("qid")]
print("mean NDCG@10:", round(float(np.mean(ndcgs)), 4))

# Rank documents for one query
one = df[df.qid == 250].sort_values("score", ascending=False).head(5)
print(one)
```

Key points: labels are graded relevances; `qid` must be sorted so that rows of the same query are contiguous; evaluation must be grouped by query; and the split is by query to avoid leakage between train and test.

### Tutorial 4: Imbalanced classification on GPU with Optuna tuning

Goal: tune a fraud-like model with 1% positives using Optuna, running on GPU when available, optimising PR AUC.

```python
import numpy as np
import optuna
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

X, y = make_classification(n_samples=200_000, n_features=40, n_informative=12,
                           weights=[0.99, 0.01], flip_y=0.002, random_state=0)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=0)

device = "cuda" if xgb.build_info().get("USE_CUDA") else "cpu"   # falls back to CPU (with a warning) if no GPU
dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr)
dvalid = xgb.QuantileDMatrix(X_va, label=y_va, ref=dtrain)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": device,
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, spw, log=True),
        "seed": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dvalid, "valid")],
                        early_stopping_rounds=100, verbose_eval=False)
    trial.set_user_attr("best_iteration", booster.best_iteration)
    return booster.best_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=40)
print("best valid PR AUC:", round(study.best_value, 4))
print(study.best_params)

# Refit with the best params and the tuned number of rounds, evaluate on test
best = {**study.best_params, "objective": "binary:logistic", "tree_method": "hist",
        "device": device, "seed": 0}
n_rounds = study.best_trial.user_attrs["best_iteration"] + 1
final = xgb.train(best, dtrain, num_boost_round=n_rounds)
p_test = final.inplace_predict(X_test)
print("test PR AUC:", round(average_precision_score(y_test, p_test), 4))
```

Explanation: `aucpr` is the right metric for rare positives; `scale_pos_weight` is tuned rather than fixed at the full ratio (which often over-corrects and distorts probabilities); each trial uses early stopping so `n_estimators` is not a search dimension; and `inplace_predict` avoids building a `DMatrix` at inference time. On a GPU the same code runs several times faster simply because `device="cuda"`.

### Tutorial 5: XGBoost inside a scikit-learn pipeline with GridSearchCV

Goal: combine sklearn preprocessing (text column + numeric columns) with XGBoost and tune with standard sklearn tools.

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

rng = np.random.default_rng(0)
n = 2000
df = pd.DataFrame({
    "text": rng.choice(["great product", "terrible service", "ok value", "fast delivery",
                        "broken on arrival"], size=n),
    "price": rng.gamma(2.0, 20.0, size=n),
    "n_reviews": rng.poisson(30, size=n),
})
y = ((df["text"].isin(["great product", "fast delivery"])) ^ (df["price"] > 60)).astype(int)

pre = ColumnTransformer([
    ("tfidf", TfidfVectorizer(), "text"),            # string, not list: vectoriser needs 1-D
    ("num", "passthrough", ["price", "n_reviews"]),
])
pipe = Pipeline([("pre", pre),
                 ("xgb", xgb.XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=-1))])

grid = {"xgb__max_depth": [3, 5], "xgb__learning_rate": [0.05, 0.1], "xgb__subsample": [0.8, 1.0]}
search = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                      scoring="roc_auc", n_jobs=1)   # XGBoost already uses all cores
search.fit(df, y)
print(search.best_params_, round(search.best_score_, 4))
```

XGBoost accepts the sparse matrix produced by the `ColumnTransformer` directly. Set `n_jobs=1` on the search (or on XGBoost) to avoid thread oversubscription.

## Performance & Best Practices

### Hyperparameter tuning strategy

A reliable order of operations:

1. **Fix the protocol**: validation split or CV folds, metric, and early stopping. Never tune `n_estimators` by grid; use a large value plus early stopping.
2. **Pick a learning rate for tuning** (0.05 to 0.1) so experiments are fast.
3. **Tree complexity**: `max_depth` (3 to 10) and `min_child_weight` (1 to 100, log scale). These matter most.
4. **Randomness**: `subsample` (0.5 to 1.0), `colsample_bytree` (0.4 to 1.0), optionally `colsample_bynode`.
5. **Regularisation**: `reg_lambda` (1e-3 to 10, log), `reg_alpha` (1e-3 to 10, log), `gamma` (0 to 5).
6. **Lower the learning rate** (e.g. 0.01 to 0.03) for the final model and let early stopping find the new number of trees; this usually gives a small extra gain.
7. **Use Bayesian optimisation** (Optuna TPE) rather than grid search for more than 3 parameters; 30 to 100 trials is typical.

| Symptom | Adjust |
|---|---|
| Overfitting (train much better than valid) | Lower `max_depth`, raise `min_child_weight`, raise `gamma` / `reg_lambda`, lower `subsample` / `colsample_*`, lower `learning_rate`. |
| Underfitting | Raise `max_depth` or use `grow_policy="lossguide"` with more `max_leaves`, lower regularisation, more rounds. |
| Imbalanced binary | `scale_pos_weight`, `max_delta_step=1`, metric `aucpr`; consider leaving weights at 1 and tuning the threshold if calibrated probabilities matter. |
| Noisy labels / outliers | `reg:pseudohubererror` or `reg:absoluteerror`, higher `min_child_weight`. |
| Many irrelevant features | Lower `colsample_bytree` / `colsample_bynode`, raise `reg_alpha`. |

### Speed

- Use `tree_method="hist"` (the default) and, if available, `device="cuda"`: GPU training is often 5x to 20x faster on large data.
- Build `QuantileDMatrix` instead of `DMatrix` for hist training to cut memory.
- Lower `max_bin` (e.g. 64 or 128) for faster training with minimal accuracy loss; raise it for precise continuous features.
- Use `inplace_predict` for low-latency inference, and keep data on the same device as the model to avoid the "mismatched devices" warning and host-device copies.
- `nthread` / `n_jobs` defaults to all cores; when running several models in parallel (CV, search), divide threads among them.
- `sampling_method="gradient_based"` with `subsample` as low as 0.1 is effective on GPU for huge datasets.
- For data that does not fit in memory, use `ExtMemQuantileDMatrix` (3.0+) or Dask / Spark.

### Correctness and reproducibility

- Set `seed` / `random_state`. GPU histogram training is deterministic for a fixed configuration in recent versions, but results can differ slightly between CPU and GPU.
- Save with `save_model("model.json")` or `.ubj`; record the XGBoost version. Avoid pickle across versions.
- Keep the column order and dtypes identical between training and inference; XGBoost validates feature names when given DataFrames.
- Remember early stopping biases the validation score; report on a separate test set.
- When you need calibrated probabilities after using `scale_pos_weight`, recalibrate (`CalibratedClassifierCV`) or avoid the weighting.

## Common Errors & Troubleshooting

| Error / warning | Cause | Fix |
|---|---|---|
| `TypeError: fit() got an unexpected keyword argument 'early_stopping_rounds'` (also `eval_metric`, `callbacks`) | Arguments moved to the constructor (deprecated in 1.6, removed in recent releases). | `XGBClassifier(early_stopping_rounds=50, eval_metric="auc")` then `fit(X, y, eval_set=[...])`. |
| `ValueError: Must have at least 1 validation dataset for early stopping.` | `early_stopping_rounds` set but no `eval_set` / `evals`. | Pass `eval_set=[(X_val, y_val)]`, or unset early stopping (e.g. when used inside `cross_val_score`). |
| `XGBoostError: ... Invalid Input: 'gpu_hist', valid values are: {'approx', 'auto', 'exact', 'hist'}` or a deprecation warning about `gpu_hist` / `gpu_id` | Old GPU parameters. | `tree_method="hist", device="cuda"` (or `"cuda:1"`). |
| `WARNING: ... Falling back to prediction using DMatrix due to mismatched devices. This might lead to higher memory usage and slower performance.` | Model on GPU, input data on CPU (or vice versa). | Pass CuPy/cuDF data for GPU models, or `model.set_params(device="cpu")` before CPU inference. |
| `XGBoostError: ... No visible GPU is found for XGBoost` / `XGBoost is not compiled with CUDA support` | No GPU, driver issue, or CPU-only build (macOS, `xgboost-cpu`). | Check `nvidia-smi` and `xgb.build_info()["USE_CUDA"]`; install the standard Linux/Windows wheel or `py-xgboost=*=cuda*`. |
| `ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, the experimental DMatrix parameter enable_categorical must be set to True. Invalid columns: city: object` | String (`object`) columns or categoricals without the flag. | `df[c] = df[c].astype("category")` and `enable_categorical=True`, or encode first. |
| `ValueError: Invalid classes inferred from unique values of y. Expected: [0 1 2], got [1 2 3]` | Classifier labels not 0..K-1. | `LabelEncoder().fit_transform(y)`; decode predictions with `inverse_transform`. |
| `ValueError: feature_names mismatch` / `training data did not have the following fields` | Column names or order differ between train and predict. | Reorder: `X_new[model.get_booster().feature_names]`; or pass `validate_features=False` only if you are sure. |
| `XGBoostError: ... label must be in [0,1] for logistic regression` | Labels such as -1/1 or 1/2 with `binary:logistic`. | Map labels to 0/1. |
| `XGBoostError: ... Check failed: preds.size() == info.labels.Size()` / `label.size() ... num_class` | `num_class` missing or wrong for `multi:*` in the native API. | Set `"num_class": K` matching the labels. |
| `XGBoostError: ... Label contains NaN, infinity or a value too large.` | Bad target values. | Drop or fix rows with non-finite `y`. |
| `WARNING: ... Parameters: { "n_estimators" } are not used.` | sklearn-only name passed to `xgb.train`, or a typo. | Use `num_boost_round` argument; check spelling. |
| `XGBoostError: [...] Unknown objective function: 'reg:linear'` | Long-removed alias. | `reg:squarederror`. |
| `OSError: ... Library not loaded: @rpath/libomp.dylib` (macOS) | OpenMP runtime missing. | `brew install libomp`. |
| `ExecutableNotFound: failed to execute 'dot'` from `plot_tree` | Graphviz binary missing. | Install Graphviz and ensure `dot` is on PATH. |
| Model predictions identical / constant | Too strong regularisation, `min_child_weight` too large, or target leakage in validation. | Inspect `evals_result()`, lower constraints, check data. |
| Predictions differ after loading a pickle in a new version | Pickle is not a stable format. | Use `save_model`/`load_model` with JSON or UBJ. |
| `best_iteration` attribute missing | Early stopping was not used. | Use `num_boosted_rounds()` instead, or enable early stopping. |

## Interoperability

### scikit-learn

`XGBClassifier`, `XGBRegressor`, `XGBRanker` and `XGBRF*` are full sklearn estimators: use them in `Pipeline`, `ColumnTransformer` outputs (dense or sparse), `GridSearchCV`, `RandomizedSearchCV`, `cross_validate`, `StackingClassifier`, `CalibratedClassifierCV`, `permutation_importance` and `PartialDependenceDisplay`. When combining early stopping with sklearn CV helpers, be aware that the validation set is fixed in the constructor-free `fit` call; for CV with early stopping use `xgb.cv` or a custom loop.

### pandas, Polars, Arrow

DataFrames are accepted directly; column names become `feature_names`, and `category` dtype becomes categorical features. Polars and PyArrow tables are supported in recent releases.

### GPU data: CuPy and cuDF (RAPIDS)

With `device="cuda"`, passing CuPy arrays or cuDF DataFrames keeps the entire pipeline on GPU without host copies:

```python
import cupy as cp
import xgboost as xgb

X = cp.random.rand(100_000, 50, dtype=cp.float32)
y = (X[:, 0] > 0.5).astype(cp.int32)
clf = xgb.XGBClassifier(device="cuda", n_estimators=200).fit(X, y)
proba = clf.predict_proba(X)   # stays on GPU
```

### SHAP

`shap.TreeExplainer(model)` supports XGBoost models directly (and is GPU-accelerated via `GPUTreeExplainer`). XGBoost's own `pred_contribs=True` / `pred_interactions=True` gives the same values without the dependency.

### Other boosting libraries

The same data and sklearn pipelines can drive LightGBM and CatBoost for comparison. Parameter equivalents: `max_depth` (XGBoost) relates to `num_leaves` (LightGBM, roughly `2**max_depth`); `min_child_weight` corresponds to LightGBM `min_sum_hessian_in_leaf`; `colsample_bytree` to `feature_fraction`; `subsample` to `bagging_fraction` (with `bagging_freq > 0`).

### Distributed and MLOps

- **Dask**: `xgboost.dask` (`DaskXGBClassifier`, `xgb.dask.train`).
- **Spark**: `xgboost.spark.SparkXGBClassifier` / `Regressor` / `Ranker` (PySpark ML pipelines); JVM package `xgboost4j-spark` for Scala.
- **Ray**: Ray Train `XGBoostTrainer`.
- **MLflow**: `mlflow.xgboost.autolog()` and `mlflow.xgboost.log_model(booster, "model")`.
- **Serving**: NVIDIA Triton FIL backend loads XGBoost JSON/UBJ models; ONNX export via `onnxmltools`; Treelite compiles models for fast CPU inference.
- **Optuna**: `optuna-integration` provides `XGBoostPruningCallback` for pruning unpromising trials.

## Cheat Sheet

### Setup and data

| Task | Code |
|---|---|
| Install | `pip install xgboost` |
| Version / CUDA support | `xgb.__version__`, `xgb.build_info()["USE_CUDA"]` |
| DMatrix from DataFrame | `xgb.DMatrix(df, label=y, enable_categorical=True)` |
| Memory-efficient hist data | `xgb.QuantileDMatrix(X, label=y)`; valid: `QuantileDMatrix(Xv, label=yv, ref=dtrain)` |
| Sample weights | `xgb.DMatrix(X, label=y, weight=w)` or `fit(X, y, sample_weight=w)` |
| Offset / exposure | `xgb.DMatrix(X, label=y, base_margin=np.log(exposure))` |
| Custom missing value | `xgb.DMatrix(X, missing=-999)` |
| Silence logs | `xgb.set_config(verbosity=0)` |

### Training

| Task | Code |
|---|---|
| sklearn classifier | `xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6).fit(X, y)` |
| Early stopping (sklearn) | `XGBClassifier(early_stopping_rounds=50, eval_metric="auc").fit(X, y, eval_set=[(Xv, yv)])` |
| Early stopping (native) | `xgb.train(p, dtr, 5000, evals=[(dva, "valid")], early_stopping_rounds=50)` |
| GPU | `device="cuda"` (+ `tree_method="hist"`) |
| Multiclass (native) | `{"objective": "multi:softprob", "num_class": K}` |
| Imbalanced | `scale_pos_weight=neg/pos`, `eval_metric="aucpr"` |
| Quantile regression | `{"objective": "reg:quantileerror", "quantile_alpha": [0.1, 0.5, 0.9]}` |
| Poisson counts | `objective="count:poisson"` |
| Ranking | `xgb.XGBRanker(objective="rank:ndcg").fit(X, y, qid=qid)` |
| Monotone constraint | `monotone_constraints={"price": -1}` |
| Interaction constraint | `interaction_constraints=[["a", "b"], ["c", "d"]]` |
| Leaf-wise growth | `grow_policy="lossguide", max_leaves=63` |
| Random forest | `xgb.XGBRFClassifier(n_estimators=300)` |
| Continue training | `xgb.train(p, dtr, 100, xgb_model=booster)` |
| CV | `xgb.cv(p, dtr, 5000, nfold=5, early_stopping_rounds=50, stratified=True)` |
| Custom objective | `xgb.train(p, dtr, obj=lambda preds, d: (grad, hess))` |

### Prediction and inspection

| Task | Code |
|---|---|
| Probabilities | `clf.predict_proba(X)[:, 1]` |
| Raw margin | `booster.predict(d, output_margin=True)` |
| Use best trees only | `booster.predict(d, iteration_range=(0, booster.best_iteration + 1))` |
| Fast predict | `booster.inplace_predict(X)` |
| SHAP values | `booster.predict(d, pred_contribs=True)` |
| SHAP interactions | `booster.predict(d, pred_interactions=True)` |
| Leaf indices | `booster.predict(d, pred_leaf=True)` / `clf.apply(X)` |
| Importance | `booster.get_score(importance_type="gain")` / `clf.feature_importances_` |
| Plot importance | `xgb.plot_importance(clf, importance_type="gain", max_num_features=20)` |
| Trees as table | `booster.trees_to_dataframe()` |
| Metric history | `clf.evals_result()` |
| Underlying booster | `clf.get_booster()` |

### Persistence

| Task | Code |
|---|---|
| Save (JSON) | `model.save_model("model.json")` |
| Save (binary UBJSON) | `model.save_model("model.ubj")` |
| Load sklearn model | `m = xgb.XGBClassifier(); m.load_model("model.json")` |
| Load booster | `b = xgb.Booster(); b.load_model("model.json")` |
| Bytes | `raw = booster.save_raw("ubj")` |
| Slice trees | `booster[: booster.best_iteration + 1]` |

### Migration from 1.x

| Old | New |
|---|---|
| `tree_method="gpu_hist"` | `tree_method="hist", device="cuda"` |
| `gpu_id=1` | `device="cuda:1"` |
| `predictor="gpu_predictor"` | removed; follows `device` |
| `fit(..., early_stopping_rounds=50)` | `XGBClassifier(early_stopping_rounds=50)` |
| `fit(..., eval_metric="auc")` | `XGBClassifier(eval_metric="auc")` |
| `fit(..., callbacks=[...])` | `XGBClassifier(callbacks=[...])` |
| `feval=` in `xgb.train` | `custom_metric=` |
| `ntree_limit=` in `predict` | `iteration_range=(0, n)` |
| `objective="reg:linear"` | `objective="reg:squarederror"` |
| `use_label_encoder=False` | removed; encode labels yourself |
| `save_model("model.bin")` binary format | `save_model("model.json")` or `.ubj` |

## Further Resources

- Official documentation: https://xgboost.readthedocs.io/en/stable/
- Parameter reference: https://xgboost.readthedocs.io/en/stable/parameter.html
- Python API reference: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- Introduction to boosted trees: https://xgboost.readthedocs.io/en/stable/tutorials/model.html
- GPU support: https://xgboost.readthedocs.io/en/stable/gpu/index.html
- Categorical data: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- Learning to rank: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
- Release notes: https://xgboost.readthedocs.io/en/stable/changes/index.html
- GitHub repository: https://github.com/dmlc/xgboost
- Discussion forum: https://discuss.xgboost.ai/
- Paper: Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016: https://arxiv.org/abs/1603.02754
- Friedman, "Greedy Function Approximation: A Gradient Boosting Machine", Annals of Statistics, 2001: https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full
- SHAP TreeExplainer paper (Lundberg et al., 2020): https://www.nature.com/articles/s42256-019-0138-9
