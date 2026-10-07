# CatBoost

> Gradient boosting on oblivious decision trees with first-class categorical, text, and embedding feature support.

CatBoost is an open-source gradient boosting library that builds ensembles of symmetric (oblivious) decision trees. Its two signature ideas are **ordered target statistics** for encoding categorical features without target leakage, and **ordered boosting**, which reduces the prediction shift that plain gradient boosting suffers from. In practice this means you can hand CatBoost a raw `pandas.DataFrame` with string columns, tell it which columns are categorical, and get a strong model with little tuning.

Covers CatBoost 1.2.x (Python package `catboost`). All examples use the Python API.

## Overview

### What CatBoost is

CatBoost ("Categorical Boosting") is developed and maintained by Yandex and released under the Apache 2.0 license. It was open-sourced in July 2017 and is described in the paper *CatBoost: unbiased boosting with categorical features* (Prokhorenkova et al., NeurIPS 2018). The core is written in C++, with bindings for Python, R, a command-line tool, and appliers for C/C++, Java, .NET, Rust, Node.js and Apache Spark.

Key characteristics:

| Feature | What it means for you |
|---|---|
| Oblivious (symmetric) trees | Every node at a given depth uses the same split. Trees are balanced, fast to evaluate, and less prone to overfitting. |
| Ordered target statistics | Categorical features are converted to numeric statistics using only "past" rows in a random permutation, avoiding target leakage. |
| Ordered boosting | Residuals for each row are computed with models that did not see that row, reducing bias. |
| Native categorical support | No manual one-hot or label encoding required. Strings work directly. |
| Text and embedding features | Built-in tokenization, dictionaries and naive-Bayes / BM25 estimators for text columns; LDA/KNN estimators for embeddings. |
| GPU training | A highly optimized CUDA implementation (`task_type="GPU"`), multi-GPU on one host. |
| Strong defaults | Default hyperparameters (with auto-tuned learning rate) are usually competitive. |
| Model export | CBM (native), JSON, ONNX, CoreML, PMML, C++ and Python source. |

### When to use CatBoost

- Tabular data with many categorical columns, especially high-cardinality ones (user IDs, ZIP codes, product SKUs).
- You want a strong baseline quickly without heavy feature engineering or tuning.
- You need stable, reproducible models that are robust to hyperparameter choices.
- Learning-to-rank problems (`CatBoostRanker` with YetiRank, PairLogit, QueryRMSE, and more).
- Fast inference on CPU: oblivious trees evaluate very efficiently.
- Mixed tabular + short text features.

### When not to use CatBoost

- Images, audio, raw long-form text, or other unstructured data: deep learning (PyTorch, TensorFlow/Keras) is the right tool.
- Extremely large purely numeric datasets where training time on CPU is the main constraint: LightGBM is often faster on CPU for numeric-only data (CatBoost on GPU is competitive).
- When you need a tiny, linear, highly interpretable model: use logistic regression or a GAM.
- Online / incremental learning in the streaming sense: CatBoost supports continuing training from a model (`init_model`), but it is not a streaming learner.

### Where it fits in the ML stack

```text
Data (pandas / polars / numpy / files)
        |
Feature prep (scikit-learn, pandas)  ->  CatBoost handles categoricals/text itself
        |
CatBoostClassifier / Regressor / Ranker  (CPU or GPU)
        |
Evaluation (CatBoost metrics, scikit-learn metrics), explainability (SHAP values built in)
        |
Export (CBM, ONNX, CoreML, JSON, C++/Python code)  ->  serving (Python, C++, Java, ONNX Runtime, Spark)
```

CatBoost sits alongside XGBoost and LightGBM as one of the three dominant gradient-boosted decision tree (GBDT) libraries. It implements the scikit-learn estimator interface, so it plugs into `Pipeline`, `GridSearchCV`, `cross_val_score`, and similar tools.

## Installation

### pip

```bash
pip install catboost
```

Wheels are published for Linux, macOS (x86_64 and arm64) and Windows. The Linux and Windows wheels include GPU (CUDA) support out of the box; there is no separate `catboost-gpu` package. You only need a compatible NVIDIA driver installed.

Optional extras used in this page:

```bash
pip install catboost pandas scikit-learn shap matplotlib ipywidgets
```

`ipywidgets` enables the interactive training plots (`plot=True`) in Jupyter.

### conda

```bash
conda install -c conda-forge catboost
```

### Pinning a version

```bash
pip install "catboost==1.2.8"
```

### Verifying the install

```python
import catboost
print(catboost.__version__)
```

### Checking GPU visibility

```python
from catboost.utils import get_gpu_device_count

n = get_gpu_device_count()
print("CUDA devices visible to CatBoost:", n)
```

A quick end-to-end GPU smoke test:

```python
import numpy as np
from catboost import CatBoostClassifier

X = np.random.rand(10_000, 20)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

model = CatBoostClassifier(iterations=50, task_type="GPU", devices="0", verbose=10)
model.fit(X, y)
```

If no GPU is available you will get an error such as `CatBoostError: ... Environment for task type [GPU] not found` or a message about no CUDA devices. Fall back to `task_type="CPU"`.

## Core Concepts

### Gradient boosting in one paragraph

Gradient boosting builds an additive model `F(x) = f_1(x) + f_2(x) + ... + f_T(x)`. Each new tree `f_t` is fit to the negative gradient of the loss with respect to the current predictions, scaled by a learning rate. For log loss this is (roughly) fitting trees to residuals in log-odds space. The number of trees (`iterations`), the size of each tree (`depth`), and the step size (`learning_rate`) are the three most important knobs.

### Oblivious trees

A CatBoost tree of depth `d` makes exactly `d` split decisions, one per level, and every node on a level uses the same feature and threshold. The tree is therefore a lookup table with `2**d` leaves indexed by `d` binary tests.

```text
depth 0:          [x3 > 0.5 ?]
                  /           \
depth 1:   [x7 > 12 ?]     [x7 > 12 ?]      <- same split on the whole level
             /    \          /    \
leaves:     L0    L1        L2    L3
```

Consequences:

- Very fast inference (bit operations and one table lookup per tree).
- Strong regularization: the tree cannot carve out tiny regions with ad-hoc splits.
- `depth` matters a lot; typical values are 4 to 10, default 6.
- Other growing policies exist (`grow_policy="Depthwise"` or `"Lossguide"`), mainly useful on GPU and for data that benefits from asymmetric trees.

### Categorical features and ordered target statistics

For a categorical column, CatBoost replaces each category value with a statistic computed from the target, such as the smoothed mean target for that category. Naively computing this over the whole training set leaks the target. CatBoost instead:

1. Draws a random permutation of the training rows.
2. For row `i`, computes the statistic using only rows that come before `i` in the permutation.
3. Applies a prior to smooth rare categories.

It also builds **feature combinations** greedily (for example `city x device_type`) and computes statistics on those. The number of categories under which one-hot encoding is used instead is controlled by `one_hot_max_size`.

```python
import pandas as pd
from catboost import CatBoostClassifier

df = pd.DataFrame({
    "city":   ["Paris", "Berlin", "Paris", "Rome", "Berlin", "Rome", "Paris", "Rome"],
    "device": ["ios", "android", "web", "ios", "web", "android", "ios", "web"],
    "age":    [23, 35, 41, 29, 52, 33, 27, 45],
    "bought": [1, 0, 1, 0, 0, 1, 1, 0],
})

X = df.drop(columns="bought")
y = df["bought"]

model = CatBoostClassifier(iterations=50, depth=3, verbose=0)
model.fit(X, y, cat_features=["city", "device"])  # names or integer indices
print(model.predict_proba(X)[:3])
```

Rules for categorical columns:

- Values must be strings or integers. Floats (including `NaN`) are rejected; convert with `.astype(str)` or fill missing values first.
- You can pass `cat_features` as column names (when `X` is a DataFrame) or integer indices.
- Columns with pandas `category` dtype must still be listed in `cat_features`.

### The Pool object

`Pool` is CatBoost's dataset container. It stores features, labels, weights, group IDs (for ranking), baselines, and the feature-type metadata, and it quantizes numeric features once. Passing raw `X, y` to `fit` works too, but `Pool` is useful when you reuse data (training, evaluation, feature importance) or need ranking/group info.

```python
from catboost import Pool

train_pool = Pool(X, y, cat_features=["city", "device"])
print(train_pool.num_row(), train_pool.num_col())
```

### Learning rate auto-selection

If you do not set `learning_rate`, CatBoost picks one based on dataset size and `iterations` for common losses (Logloss, MultiClass, RMSE). The chosen value is printed during training and available via `model.get_all_params()["learning_rate"]`.

### Overfitting detector and best model

Pass an `eval_set` and CatBoost tracks the evaluation metric every iteration. With `early_stopping_rounds=N`, training stops when the metric does not improve for `N` iterations. With `use_best_model=True` (the default when `eval_set` is given), the model is shrunk to the best iteration.

```python
model = CatBoostClassifier(iterations=2000, learning_rate=0.05, eval_metric="AUC")
model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100, verbose=200)
print(model.get_best_iteration(), model.get_best_score())
```

### Missing values

Numeric `NaN` is handled natively. `nan_mode="Min"` (default) treats missing values as smaller than all others, `"Max"` as larger, and `"Forbidden"` raises an error. Categorical missing values must be represented as a string (for example `"NA"`).

### Training artifacts

By default CatBoost writes logs and metrics to a `catboost_info/` directory in the working directory. Disable with `allow_writing_files=False` or redirect with `train_dir="some/path"`.

### Reproducibility

Set `random_seed`. Results are deterministic on CPU for a fixed seed and thread count; GPU results can differ slightly between runs and hardware.

## API Reference

### Model classes

#### CatBoostClassifier

```python
catboost.CatBoostClassifier(
    iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None,
    loss_function=None, eval_metric=None, custom_metric=None,
    random_seed=None, cat_features=None, text_features=None,
    embedding_features=None, class_weights=None, auto_class_weights=None,
    scale_pos_weight=None, one_hot_max_size=None, task_type=None,
    devices=None, early_stopping_rounds=None, use_best_model=None,
    verbose=None, thread_count=None, **kwargs
)
```

Binary and multiclass classification. `loss_function` defaults to `"Logloss"` for two classes and `"MultiClass"` when more than two classes are found. Labels can be integers, strings, or booleans.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `iterations` | int | 1000 | Maximum number of trees. Aliases: `n_estimators`, `num_boost_round`, `num_trees` (use only one). |
| `learning_rate` | float | auto | Step size shrinkage. Alias `eta`. |
| `depth` | int | 6 | Tree depth (max 16 for SymmetricTree). Alias `max_depth`. |
| `l2_leaf_reg` | float | 3.0 | L2 regularization on leaf values. Alias `reg_lambda`. |
| `loss_function` | str | `"Logloss"` / `"MultiClass"` | Objective to optimize. |
| `eval_metric` | str | same as loss | Metric for overfitting detection and best model selection. |
| `custom_metric` | str or list | None | Extra metrics to compute and log (not used for early stopping). |
| `random_seed` | int | 0 | Seed. Alias `random_state`. |
| `cat_features` | list | None | Categorical columns (indices or names). |
| `class_weights` | list/dict | None | Per-class weights. |
| `auto_class_weights` | str | None | `"Balanced"` or `"SqrtBalanced"`. |
| `scale_pos_weight` | float | 1.0 | Weight of positive class (binary only). |
| `task_type` | str | `"CPU"` | `"CPU"` or `"GPU"`. |
| `devices` | str | None | GPU IDs, e.g. `"0"`, `"0:1"`, `"0-3"`. |
| `early_stopping_rounds` | int | None | Stop after N iterations without improvement on `eval_set`. |
| `verbose` | bool/int | True | `False`/`0` silent; integer N logs every N iterations. |
| `thread_count` | int | -1 | CPU threads (-1 = all cores). |

Returns: an unfitted estimator. After `fit`, attributes such as `classes_`, `tree_count_`, `feature_names_`, `best_iteration_`, `best_score_`, `evals_result_`, and `feature_importances_` are populated.

```python
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

clf = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, eval_metric="AUC", verbose=100)
clf.fit(X_tr, y_tr, eval_set=(X_te, y_te), early_stopping_rounds=50)
print(clf.classes_, clf.tree_count_)
```

#### CatBoostRegressor

```python
catboost.CatBoostRegressor(
    iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None,
    loss_function="RMSE", eval_metric=None, random_seed=None,
    cat_features=None, task_type=None, **kwargs
)
```

Regression. Same tree/boosting parameters as the classifier. Common `loss_function` values: `"RMSE"`, `"MAE"`, `"Quantile:alpha=0.9"`, `"Huber:delta=1.0"`, `"Poisson"`, `"Tweedie:variance_power=1.5"`, `"MAPE"`, `"LogLinQuantile"`, `"Expectile:alpha=0.5"`, `"RMSEWithUncertainty"`, `"MultiRMSE"` (multi-target).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss_function` | str | `"RMSE"` | Regression objective. Parameters are passed with `:` syntax. |
| `eval_metric` | str | same as loss | e.g. `"MAE"`, `"R2"`, `"RMSE"`, `"MAPE"`. |
| `iterations`, `learning_rate`, `depth`, ... | | | As for the classifier. |

```python
from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
reg = CatBoostRegressor(iterations=800, learning_rate=0.08, depth=8, loss_function="RMSE", verbose=200)
reg.fit(X, y)
print(reg.predict(X.head()))
```

Quantile / uncertainty example:

```python
q90 = CatBoostRegressor(loss_function="Quantile:alpha=0.9", iterations=300, verbose=0).fit(X, y)
unc = CatBoostRegressor(loss_function="RMSEWithUncertainty", iterations=300, verbose=0).fit(X, y)
mean_var = unc.predict(X.head(), prediction_type="RMSEWithUncertainty")  # columns: mean, variance
print(mean_var)
```

#### CatBoostRanker

```python
catboost.CatBoostRanker(iterations=None, learning_rate=None, depth=None,
                        loss_function="YetiRank", eval_metric=None, **kwargs)
```

Learning to rank. Requires a `group_id` (query ID) for each row; rows of the same group must be contiguous.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss_function` | str | `"YetiRank"` | `"YetiRank"`, `"YetiRankPairwise"`, `"PairLogit"`, `"PairLogitPairwise"`, `"QueryRMSE"`, `"QuerySoftMax"`, `"StochasticRank:metric=NDCG"`. |
| `eval_metric` | str | depends | e.g. `"NDCG:top=10"`, `"PFound"`, `"MAP:top=10"`, `"MRR"`, `"QueryAUC"`. |

```python
import numpy as np
from catboost import CatBoostRanker, Pool

rng = np.random.default_rng(0)
n_queries, docs_per_q = 100, 10
X = rng.normal(size=(n_queries * docs_per_q, 5))
relevance = rng.integers(0, 4, size=n_queries * docs_per_q)
group_id = np.repeat(np.arange(n_queries), docs_per_q)

pool = Pool(X, label=relevance, group_id=group_id)
ranker = CatBoostRanker(loss_function="YetiRank", eval_metric="NDCG:top=5", iterations=200, verbose=50)
ranker.fit(pool)
scores = ranker.predict(X[:10])  # higher score = ranked higher
```

#### CatBoost (generic)

```python
catboost.CatBoost(params=None)
```

The base class. Takes a single `params` dict and is used when you want to load a model of unknown type or use the low-level API. `CatBoostClassifier`, `CatBoostRegressor`, and `CatBoostRanker` are subclasses.

```python
from catboost import CatBoost
model = CatBoost({"loss_function": "Logloss", "iterations": 100, "verbose": 0})
```

### Full hyperparameter reference

The table below lists the training parameters you are most likely to touch. All can be passed to the constructor of any model class.

#### Boosting and trees

| Parameter | Type | Default | Description |
|---|---|---|---|
| `iterations` | int | 1000 | Max trees. |
| `learning_rate` | float | auto (else 0.03) | Shrinkage. |
| `depth` | int | 6 | Tree depth. |
| `grow_policy` | str | `"SymmetricTree"` | `"SymmetricTree"`, `"Depthwise"`, `"Lossguide"`. |
| `max_leaves` | int | 31 | Max leaves; only for `Lossguide`. |
| `min_data_in_leaf` | int | 1 | Minimum samples per leaf; only for `Depthwise`/`Lossguide`. |
| `boosting_type` | str | `"Plain"` (often) | `"Ordered"` or `"Plain"`. Ordered is slower but better on small data. Default depends on dataset size and device. |
| `l2_leaf_reg` | float | 3.0 | L2 regularization. |
| `random_strength` | float | 1.0 | Noise added to split scores, reduces overfitting. |
| `model_size_reg` | float | 0.5 | Penalizes large models (affects categorical combinations). |
| `leaf_estimation_method` | str | depends | `"Newton"`, `"Gradient"`, `"Exact"`. |
| `leaf_estimation_iterations` | int | depends | Gradient steps when computing leaf values. |
| `score_function` | str | `"Cosine"` | Split scoring: `"Cosine"`, `"L2"`, `"NewtonCosine"`, `"NewtonL2"`. |
| `monotone_constraints` | list/dict | None | e.g. `{"price": -1, "quality": 1}`. |
| `feature_weights` | list/dict | None | Multipliers for split scores per feature. |
| `penalties_coefficient` | float | 1.0 | Scales feature penalties. |
| `first_feature_use_penalties` | list/dict | None | Penalty for first use of a feature in the model. |

#### Sampling

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bootstrap_type` | str | `"MVS"` on CPU, `"Bayesian"` on GPU | `"Bayesian"`, `"Bernoulli"`, `"MVS"`, `"Poisson"` (GPU), `"No"`. |
| `bagging_temperature` | float | 1.0 | Intensity of Bayesian bootstrap (0 = no bootstrap). |
| `subsample` | float | 0.8 (Bernoulli/MVS on large data) | Row sampling rate for Bernoulli/MVS/Poisson. |
| `rsm` | float | 1.0 | Random subspace method: fraction of features per split. Alias `colsample_bylevel`. |
| `sampling_frequency` | str | `"PerTree"` | `"PerTree"` or `"PerTreeLevel"`. |

#### Categorical features

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cat_features` | list | None | Categorical columns. |
| `one_hot_max_size` | int | 2 (varies) | Use one-hot for features with at most this many values. |
| `max_ctr_complexity` | int | 4 (CPU) | Max number of categorical features combined. Set 1 to disable combinations. |
| `simple_ctr` | list | auto | CTR types for single features, e.g. `["Borders", "Counter"]`. |
| `combinations_ctr` | list | auto | CTR types for combinations. |
| `counter_calc_method` | str | `"Full"` | How `Counter` CTR is computed. |

#### Numeric quantization

| Parameter | Type | Default | Description |
|---|---|---|---|
| `border_count` | int | 254 (CPU), 128 (GPU) | Number of histogram bins per numeric feature. Alias `max_bin`. |
| `feature_border_type` | str | `"GreedyLogSum"` | Binarization method. |
| `per_float_feature_quantization` | list | None | e.g. `["0:border_count=1024"]`. |
| `nan_mode` | str | `"Min"` | `"Min"`, `"Max"`, `"Forbidden"`. |

#### Overfitting detection

| Parameter | Type | Default | Description |
|---|---|---|---|
| `early_stopping_rounds` | int | None | Shortcut for `od_type="Iter"` with `od_wait=N`. |
| `od_type` | str | `"IncToDec"` | `"IncToDec"` or `"Iter"`. |
| `od_wait` | int | 20 | Iterations to wait after the best. |
| `od_pval` | float | 0 | Threshold for `IncToDec` (try 1e-10 to 1e-2). |
| `use_best_model` | bool | True if eval set | Shrink to best iteration. |
| `eval_metric` | str | loss | Metric watched by detector. |

#### Class imbalance

| Parameter | Type | Default | Description |
|---|---|---|---|
| `auto_class_weights` | str | None | `"Balanced"` or `"SqrtBalanced"`. |
| `class_weights` | list/dict | None | Explicit weights, e.g. `{0: 1, 1: 10}`. |
| `scale_pos_weight` | float | 1.0 | Binary only. |

#### Hardware and runtime

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task_type` | str | `"CPU"` | `"CPU"` or `"GPU"`. |
| `devices` | str | all | GPU device IDs. |
| `gpu_ram_part` | float | 0.95 | Fraction of GPU memory to use. |
| `thread_count` | int | -1 | CPU threads. |
| `used_ram_limit` | str | None | e.g. `"8gb"`; limits CTR memory on CPU. |
| `train_dir` | str | `"catboost_info"` | Output directory for logs. |
| `allow_writing_files` | bool | True | Set False in read-only environments. |
| `save_snapshot` | bool | None | Enable crash-safe snapshots. |
| `snapshot_file` | str | `"experiment.cbsnapshot"` | Snapshot path. |
| `snapshot_interval` | int | 600 | Seconds between snapshots. |
| `metric_period` | int | 1 | Compute metrics every N iterations (speeds up training). |
| `logging_level` | str | `"Verbose"` | `"Silent"`, `"Verbose"`, `"Info"`, `"Debug"`. |

#### Text and embedding features

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_features` | list | None | Text columns. |
| `text_processing` | dict | auto | Tokenizers, dictionaries and feature calcers (`"BoW"`, `"NaiveBayes"`, `"BM25"`). |
| `embedding_features` | list | None | Columns containing numeric vectors (lists/arrays). |

### Training methods

#### fit

```python
model.fit(X, y=None, cat_features=None, text_features=None, embedding_features=None,
          sample_weight=None, baseline=None, use_best_model=None, eval_set=None,
          verbose=None, logging_level=None, plot=False, plot_file=None,
          column_description=None, metric_period=None, silent=None,
          early_stopping_rounds=None, save_snapshot=None, snapshot_file=None,
          snapshot_interval=None, init_model=None, callbacks=None,
          log_cout=None, log_cerr=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | Pool, DataFrame, ndarray, list, file path | required | Training data. If a `Pool`, `y` must be None. |
| `y` | array-like | None | Labels. |
| `cat_features` | list | None | Overrides constructor value. |
| `sample_weight` | array-like | None | Per-row weights. |
| `baseline` | array-like | None | Initial raw predictions (for stacking/residual boosting). |
| `eval_set` | Pool, tuple, or list | None | Validation data; multiple sets allowed (last one is used for early stopping). |
| `early_stopping_rounds` | int | None | Overrides constructor value. |
| `plot` | bool | False | Interactive training plot in Jupyter. |
| `init_model` | model or path | None | Continue training from an existing model. |
| `callbacks` | list | None | Objects with `after_iteration(info)`. |

Returns: `self`.

```python
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, plot=False)
```

#### Continuing training

```python
base = CatBoostRegressor(iterations=200, verbose=0).fit(X, y)
more = CatBoostRegressor(iterations=100, verbose=0)
more.fit(X, y, init_model=base)  # 300 trees total
print(more.tree_count_)
```

#### Callbacks

```python
import time

class TimeLimit:
    def __init__(self, seconds):
        self.deadline = time.time() + seconds

    def after_iteration(self, info):
        # info.iteration, info.metrics -> {"learn": {...}, "validation": {...}}
        return time.time() < self.deadline  # return False to stop

model = CatBoostClassifier(iterations=100_000, verbose=0)
model.fit(X_tr, y_tr, callbacks=[TimeLimit(30)])
```

### Prediction methods

#### predict

```python
model.predict(data, prediction_type=None, ntree_start=0, ntree_end=0,
              thread_count=-1, verbose=None, task_type="CPU")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | Pool, DataFrame, ndarray, list | required | Input features. |
| `prediction_type` | str | `"Class"` (clf) / `"RawFormulaVal"` (reg) | `"Class"`, `"Probability"`, `"LogProbability"`, `"RawFormulaVal"`, `"Exponent"`, `"RMSEWithUncertainty"`. |
| `ntree_start`, `ntree_end` | int | 0, 0 | Use a sub-range of trees (0 = all). |

Returns: `numpy.ndarray`. For classifiers with `"Class"`, an array of class labels (shape `(n,)` or `(n, 1)` depending on version and label type; use `.ravel()` to be safe).

#### predict_proba

```python
model.predict_proba(X, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU")
```

Returns an array of shape `(n_samples, n_classes)`. Columns are ordered like `model.classes_`.

#### predict_log_proba, staged_predict, staged_predict_proba

```python
for i, proba in enumerate(model.staged_predict_proba(X_val, eval_period=100)):
    print(i, proba[:2])
```

`staged_*` methods yield predictions after every `eval_period` trees; useful for plotting quality versus number of trees.

### Evaluation methods

| Method | Signature | Returns |
|---|---|---|
| `score` | `score(X, y)` | Accuracy (classifier) or R^2 (regressor). |
| `get_best_iteration` | `get_best_iteration()` | int or None. |
| `get_best_score` | `get_best_score()` | dict `{"learn": {...}, "validation": {...}}`. |
| `get_evals_result` | `get_evals_result()` | dict of metric histories per dataset. |
| `eval_metrics` | `eval_metrics(data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, plot=False)` | dict metric -> list of values per eval period. |
| `get_all_params` | `get_all_params()` | dict of every resolved parameter, including defaults. |
| `get_params` | `get_params(deep=True)` | dict of user-set parameters (scikit-learn API). |
| `set_params` | `set_params(**params)` | self. |
| `is_fitted` | `is_fitted()` | bool. |

```python
from catboost import Pool
val_pool = Pool(X_te, y_te)
history = clf.eval_metrics(val_pool, metrics=["AUC", "Logloss"], eval_period=50)
print(history["AUC"][-1])
```

Standalone metric computation:

```python
from catboost.utils import eval_metric
import numpy as np

labels = np.array([0, 1, 1, 0])
raw = np.array([-1.2, 0.8, 2.0, 0.1])  # raw scores (log-odds)
print(eval_metric(labels, raw, "AUC"))
print(eval_metric(labels, raw, "Logloss"))
```

### Pool

```python
catboost.Pool(data, label=None, cat_features=None, text_features=None,
              embedding_features=None, column_description=None, pairs=None,
              delimiter="\t", has_header=False, weight=None, group_id=None,
              group_weight=None, subgroup_id=None, pairs_weight=None,
              baseline=None, timestamp=None, feature_names=None, thread_count=-1)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | DataFrame, ndarray, list, str (file path), FeaturesData | required | Features, or path to a TSV/libsvm file. |
| `label` | array-like | None | Target. |
| `cat_features` | list | None | Categorical columns. |
| `text_features` | list | None | Text columns. |
| `embedding_features` | list | None | Embedding columns. |
| `column_description` | str | None | Path to a `.cd` file describing columns (file input). |
| `weight` | array-like | None | Row weights. |
| `group_id` | array-like | None | Query IDs for ranking. |
| `pairs` | array-like or path | None | Explicit pairs for pairwise losses. |
| `baseline` | array-like | None | Initial raw predictions. |
| `feature_names` | list | None | Names when `data` is a numpy array. |

Useful methods: `num_row()`, `num_col()`, `get_label()`, `get_weight()`, `get_feature_names()`, `get_cat_feature_indices()`, `get_text_feature_indices()`, `set_weight(w)`, `set_baseline(b)`, `set_feature_names(names)`, `slice(indices)`, `quantize(...)`, `save(fname)`.

```python
from catboost import Pool
pool = Pool(data=X_tr, label=y_tr, weight=None, feature_names=list(X_tr.columns))
print(pool.num_row(), pool.num_col(), pool.get_feature_names()[:3])
```

Loading from files:

```python
# train.tsv: label<TAB>f1<TAB>f2<TAB>city
# train.cd:  0<TAB>Label
#            3<TAB>Categ
pool = Pool("train.tsv", column_description="train.cd", delimiter="\t", has_header=False)
```

Column description files can be generated with `catboost.utils.create_cd(label=0, cat_features=[3], output_path="train.cd")`.

### Cross-validation

#### cv

```python
catboost.cv(pool=None, params=None, dtrain=None, iterations=None,
            num_boost_round=None, fold_count=3, nfold=None, inverted=False,
            partition_random_seed=0, seed=None, shuffle=True,
            logging_level=None, stratified=None, as_pandas=True,
            metric_period=None, verbose=None, verbose_eval=None, plot=False,
            early_stopping_rounds=None, folds=None, type="Classical",
            return_models=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pool` | Pool | required | Full dataset. |
| `params` | dict | required | Training params; must include `loss_function`. |
| `iterations` | int | from params | Number of trees. |
| `fold_count` | int | 3 | Number of folds. Alias `nfold`. |
| `stratified` | bool | True for classification | Stratified split. |
| `shuffle` | bool | True | Shuffle before splitting. |
| `partition_random_seed` | int | 0 | Seed for the split. Alias `seed`. |
| `early_stopping_rounds` | int | None | Stop when mean test metric stops improving. |
| `folds` | generator or sklearn splitter | None | Custom splits (e.g. `GroupKFold`, `TimeSeriesSplit`). |
| `type` | str | `"Classical"` | `"Classical"`, `"Inverted"`, `"TimeSeries"`. |
| `return_models` | bool | False | Also return the fold models. |

Returns: a `pandas.DataFrame` with columns `iterations`, `test-<Metric>-mean`, `test-<Metric>-std`, `train-<Metric>-mean`, `train-<Metric>-std` (or `(df, models)` with `return_models=True`).

```python
from catboost import Pool, cv

pool = Pool(X, y)
params = {"loss_function": "Logloss", "eval_metric": "AUC", "learning_rate": 0.05,
          "depth": 6, "verbose": False}
scores = cv(pool, params, iterations=1000, fold_count=5, early_stopping_rounds=50, stratified=True)
best = scores["test-AUC-mean"].idxmax()
print(best, scores.loc[best, ["test-AUC-mean", "test-AUC-std"]])
```

### Hyperparameter search

#### grid_search / randomized_search

```python
model.grid_search(param_grid, X, y=None, cv=3, partition_random_seed=0,
                  calc_cv_statistics=True, search_by_train_test_split=True,
                  refit=True, shuffle=True, stratified=None, train_size=0.8,
                  verbose=True, plot=False)

model.randomized_search(param_distributions, X, y=None, cv=3, n_iter=10,
                        partition_random_seed=0, calc_cv_statistics=True,
                        search_by_train_test_split=True, refit=True,
                        shuffle=True, stratified=None, train_size=0.8,
                        verbose=True, plot=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `param_grid` / `param_distributions` | dict | required | Lists of values (or scipy distributions for randomized). |
| `cv` | int or splitter | 3 | Folds for final CV statistics. |
| `search_by_train_test_split` | bool | True | Search with a single split (fast) instead of full CV. |
| `refit` | bool | True | Refit the model on all data with best params. |
| `n_iter` | int | 10 | Randomized search draws. |

Returns: dict with `"params"` (best params) and `"cv_results"`.

```python
model = CatBoostClassifier(iterations=300, verbose=0)
grid = {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1], "l2_leaf_reg": [1, 3, 9]}
result = model.grid_search(grid, X_tr, y_tr, cv=3, verbose=False)
print(result["params"])
```

The model is refit with the best parameters, so `model` is ready to use afterwards.

### Feature importance and explainability

#### get_feature_importance

```python
model.get_feature_importance(data=None, type="FeatureImportance", prettified=False,
                             thread_count=-1, verbose=False, shap_mode="Auto",
                             interaction_indices=None, shap_calc_type="Regular")
```

| `type` value | Needs `data`? | Returns |
|---|---|---|
| `"PredictionValuesChange"` | No | Average change in prediction when a feature value changes. Default for non-ranking losses (what `"FeatureImportance"` resolves to). |
| `"LossFunctionChange"` | Yes | Change in loss when feature is excluded. Default for ranking. More expensive. |
| `"FeatureImportance"` | No | Alias choosing one of the two above. |
| `"ShapValues"` | Yes | Array `(n, n_features + 1)`; last column is expected value. For multiclass: `(n, n_classes, n_features + 1)`. |
| `"ShapInteractionValues"` | Yes | Array `(n, n_features + 1, n_features + 1)`. |
| `"Interaction"` | No | Pairwise feature interaction strengths `[[i, j, score], ...]`. |
| `"PredictionDiff"` | Yes (2 rows) | Feature contribution to the difference between two predictions. |

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | Pool or array-like | None | Required for SHAP and LossFunctionChange. |
| `prettified` | bool | False | Return a DataFrame with `Feature Id` and `Importances`. |
| `shap_mode` | str | `"Auto"` | `"Auto"`, `"UsePreCalc"`, `"NoPreCalc"`. |
| `shap_calc_type` | str | `"Regular"` | `"Approximate"`, `"Regular"`, `"Exact"`. |
| `interaction_indices` | list | None | Restrict SHAP interaction calc to two features. |

```python
from catboost import Pool

fi = clf.get_feature_importance(prettified=True)
print(fi.head(10))

shap_vals = clf.get_feature_importance(Pool(X_te, y_te), type="ShapValues")
contrib, expected = shap_vals[:, :-1], shap_vals[:, -1]
print(contrib.shape, expected[0])
```

`model.feature_importances_` is a shortcut for the default importance as a numpy array, aligned with `model.feature_names_`.

#### Using the shap package

```python
import shap

explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(X_te)
shap.summary_plot(sv, X_te)
```

`shap.TreeExplainer` supports CatBoost models directly, including categorical features.

#### plot_tree

```python
model.plot_tree(tree_idx=0, pool=None)
```

Returns a `graphviz.Digraph` (requires the `graphviz` package). Pass `pool` to show feature names and categorical values.

### Feature selection

#### select_features

```python
model.select_features(X, y=None, eval_set=None, features_for_select=None,
                      num_features_to_select=None, algorithm=None, steps=1,
                      shap_calc_type="Regular", train_final_model=True,
                      verbose=None, logging_level=None, plot=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eval_set` | Pool/tuple | required in practice | Data used to evaluate eliminations. |
| `features_for_select` | list or str | required | Candidate features, e.g. `"0-29"` or list of names/indices. |
| `num_features_to_select` | int | required | How many to keep. |
| `algorithm` | `EFeaturesSelectionAlgorithm` | `RecursiveByShapValues` | Also `RecursiveByPredictionValuesChange`, `RecursiveByLossFunctionChange`. |
| `steps` | int | 1 | Number of elimination rounds. |
| `train_final_model` | bool | True | Refit on selected features. |

Returns: dict with `selected_features`, `selected_features_names`, `eliminated_features`, `loss_graph`.

```python
from catboost import CatBoostClassifier, Pool, EFeaturesSelectionAlgorithm, EShapCalcType

model = CatBoostClassifier(iterations=300, verbose=0)
summary = model.select_features(
    Pool(X_tr, y_tr), eval_set=Pool(X_te, y_te),
    features_for_select=list(range(X_tr.shape[1])), num_features_to_select=10,
    steps=3, algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular, train_final_model=True,
)
print(summary["selected_features_names"])
```

### Saving, loading, and exporting

#### save_model

```python
model.save_model(fname, format="cbm", export_parameters=None, pool=None)
```

| `format` | Notes |
|---|---|
| `"cbm"` | Native binary format. Fully supported, includes categorical feature handling. |
| `"json"` | Human-readable JSON; usable by the JSON applier. |
| `"onnx"` | ONNX-ML; numeric features only (no categorical features in the model). |
| `"coreml"` | Apple CoreML; numeric features only. |
| `"pmml"` | PMML 4.3. |
| `"cpp"` | C++ source code for standalone inference. |
| `"python"` | Python source code for standalone inference. |

`pool` is required for `"cpp"`, `"python"`, and `"json"` exports when the model has categorical features (it is used to save the category hash mapping).

#### load_model

```python
model = CatBoostClassifier()
model.load_model("model.cbm")             # native
model.load_model("model.json", format="json")
```

A generic `CatBoost().load_model(path)` works when you do not know the model type; convert it with `to_classifier` / `to_regressor` if needed.

#### Pickle and joblib

CatBoost models are picklable, so `joblib.dump(model, "m.joblib")` works and is convenient inside scikit-learn pipelines. For long-term storage prefer `.cbm`, which is stable across CatBoost versions.

```python
clf.save_model("clf.cbm")
clf.save_model("clf.onnx", format="onnx",
               export_parameters={"onnx_domain": "ai.catboost", "onnx_model_version": 1,
                                  "onnx_graph_name": "CatBoostModel"})
```

### Model utilities

| Function / method | Description |
|---|---|
| `model.shrink(ntree_end, ntree_start=0)` | Keep only trees in the range. |
| `model.copy()` | Deep copy. |
| `model.get_borders()` | Quantization borders per feature. |
| `model.set_feature_names(names)` | Rename features. |
| `model.get_cat_feature_indices()` | Indices of categorical features. |
| `model.set_scale_and_bias(scale, bias)` / `get_scale_and_bias()` | Linear transform applied to raw predictions. |
| `catboost.sum_models(models, weights=None, ctr_merge_policy="IntersectingCountersAverage")` | Blend multiple models into one. |
| `catboost.to_classifier(model)` / `to_regressor(model)` | Convert a generic `CatBoost` to the sklearn-style class. |

```python
from catboost import sum_models, to_regressor

m1 = CatBoostRegressor(iterations=100, random_seed=1, verbose=0).fit(X, y)
m2 = CatBoostRegressor(iterations=100, random_seed=2, verbose=0).fit(X, y)
blend = to_regressor(sum_models([m1, m2], weights=[0.5, 0.5]))
print(blend.predict(X.head()))
```

### catboost.utils

| Function | Signature | Returns |
|---|---|---|
| `get_gpu_device_count` | `get_gpu_device_count()` | int |
| `eval_metric` | `eval_metric(label, approx, metric, weight=None, group_id=None, thread_count=-1)` | list of float |
| `get_roc_curve` | `get_roc_curve(model, data, thread_count=-1, plot=False)` | `(fpr, tpr, thresholds)` |
| `get_fpr_curve` | `get_fpr_curve(model=None, data=None, curve=None, thread_count=-1, plot=False)` | `(thresholds, fpr)` |
| `get_fnr_curve` | `get_fnr_curve(model=None, data=None, curve=None, thread_count=-1, plot=False)` | `(thresholds, fnr)` |
| `select_threshold` | `select_threshold(model=None, data=None, curve=None, FPR=None, FNR=None, thread_count=-1)` | float threshold |
| `create_cd` | `create_cd(label=None, cat_features=None, text_features=None, weight=None, group_id=None, ..., output_path="train.cd")` | writes file |
| `get_confusion_matrix` | `get_confusion_matrix(model, data, thread_count=-1)` | ndarray |

```python
from catboost import Pool
from catboost.utils import get_roc_curve, select_threshold

val = Pool(X_te, y_te)
fpr, tpr, thr = get_roc_curve(clf, val)
t = select_threshold(clf, val, FPR=0.05)  # threshold giving FPR <= 5%
print("threshold:", t)
```

### Custom losses and metrics

#### Custom metric

```python
import numpy as np

class F1Metric:
    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes: list of containers, one per output dimension (raw values)
        raw = np.array(approxes[0])
        pred = (raw > 0).astype(int)
        y = np.array(target).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        return f1, 1.0  # (error_sum, weight_sum)

    def get_final_error(self, error, weight):
        return error / weight

model = CatBoostClassifier(iterations=200, eval_metric=F1Metric(), verbose=50)
model.fit(X_tr, y_tr, eval_set=(X_te, y_te))
```

Note: CatBoost has a built-in `"F1"` metric; the example shows the protocol. Python metrics are slower than built-ins and force CPU evaluation.

#### Custom objective

```python
import numpy as np

class LoglossObjective:
    def calc_ders_range(self, approxes, targets, weights):
        result = []
        for i in range(len(targets)):
            p = 1.0 / (1.0 + np.exp(-approxes[i]))
            der1 = targets[i] - p          # first derivative of log-likelihood
            der2 = -p * (1 - p)            # second derivative
            w = 1.0 if weights is None else weights[i]
            result.append((der1 * w, der2 * w))
        return result

model = CatBoostClassifier(loss_function=LoglossObjective(), eval_metric="Logloss",
                           iterations=100, verbose=0)
model.fit(X_tr, y_tr)
```

CatBoost expects derivatives of the quantity being **maximized** (log-likelihood), i.e. the negative of the loss gradient. Custom objectives are CPU-only and slower than built-ins.

### Built-in losses and metrics (selected)

| Task | Losses | Metrics |
|---|---|---|
| Binary | `Logloss`, `CrossEntropy`, `Focal:focal_alpha=0.25;focal_gamma=2` | `AUC`, `Logloss`, `Accuracy`, `Precision`, `Recall`, `F1`, `BalancedAccuracy`, `MCC`, `PRAUC`, `BrierScore`, `Kappa` |
| Multiclass | `MultiClass`, `MultiClassOneVsAll` | `MultiClass`, `Accuracy`, `TotalF1`, `AUC` (OvA), `MCC`, `HammingLoss` |
| Multilabel | `MultiLogloss`, `MultiCrossEntropy` | `HammingLoss`, `Accuracy`, `Precision`, `Recall` |
| Regression | `RMSE`, `MAE`, `Quantile`, `MultiQuantile`, `Expectile`, `Huber`, `Poisson`, `Tweedie`, `MAPE`, `LogCosh`, `Lq`, `RMSEWithUncertainty`, `LogLinQuantile` | `RMSE`, `MAE`, `R2`, `MAPE`, `MSLE`, `MedianAbsoluteError`, `SMAPE` |
| Multi-regression | `MultiRMSE` | `MultiRMSE` |
| Ranking | `YetiRank`, `YetiRankPairwise`, `PairLogit`, `PairLogitPairwise`, `QueryRMSE`, `QuerySoftMax`, `QueryCrossEntropy`, `StochasticRank` | `NDCG`, `DCG`, `PFound`, `MAP`, `MRR`, `ERR`, `PrecisionAt`, `RecallAt`, `QueryAUC` |

Metric parameters use `:` and `;`: `"NDCG:top=10;type=Exp"`, `"Quantile:alpha=0.1"`, `"Tweedie:variance_power=1.3"`.

## Tutorials

### Tutorial 1: Binary classification with raw categorical data (Adult income)

Goal: predict whether income exceeds 50K from census data containing many string columns, without any manual encoding.

```python
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from catboost import CatBoostClassifier, Pool

# 1. Load data. as_frame=True keeps string columns as pandas categories.
adult = fetch_openml("adult", version=2, as_frame=True)
X = adult.data.copy()
y = (adult.target == ">50K").astype(int)

# 2. Identify categorical columns and make them clean strings.
cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype(object).fillna("NA").astype(str)  # NaN is not allowed in categoricals

# 3. Train / validation / test split.
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool = Pool(X_val, y_val, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

# 4. Model with early stopping on AUC.
model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    custom_metric=["Logloss", "Accuracy"],
    random_seed=42,
    od_type="Iter",
    od_wait=150,
    verbose=250,
)
model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# 5. Evaluate on the untouched test set.
proba = model.predict_proba(test_pool)[:, 1]
print("Test AUC:", roc_auc_score(y_test, proba))
print(classification_report(y_test, (proba > 0.5).astype(int)))

# 6. Inspect what the model relies on.
print(model.get_feature_importance(prettified=True).head(10))

# 7. Save for production.
model.save_model("adult_income.cbm")
```

Step by step:

1. `fetch_openml` returns a DataFrame; categorical columns arrive as pandas `category` dtype.
2. Casting to `object` first lets `fillna("NA")` work regardless of the category list; then every value is a string.
3. Three-way split: the validation set drives early stopping, so it must not be used for the final score.
4. `od_wait=150` with `use_best_model=True` trims the model to the best iteration.
5. `predict_proba` returns two columns; column 1 is the positive class because `classes_` is `[0, 1]`.
6. `prettified=True` returns a sorted DataFrame.
7. `.cbm` keeps categorical encoders inside the model, so at inference time you pass raw strings again.

### Tutorial 2: Regression with cross-validation, tuning, and SHAP

Goal: predict California house prices, estimate generalization with CV, tune a few parameters, and explain predictions.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool, cv

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1. Baseline CV.
params = {"loss_function": "RMSE", "learning_rate": 0.08, "depth": 6,
          "random_seed": 0, "verbose": False}
cv_df = cv(Pool(X_train, y_train), params, iterations=2000, fold_count=5,
           early_stopping_rounds=100, shuffle=True)
best_iter = int(cv_df["test-RMSE-mean"].idxmin())
print("CV RMSE:", cv_df["test-RMSE-mean"].min(), "at iteration", best_iter)

# 2. Randomized search over a few key parameters.
search_model = CatBoostRegressor(iterations=best_iter + 1, loss_function="RMSE",
                                 random_seed=0, verbose=False)
dist = {
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 10],
    "learning_rate": [0.03, 0.05, 0.08],
    "bagging_temperature": [0, 0.5, 1],
}
res = search_model.randomized_search(dist, X_train, y_train, n_iter=12, cv=3, verbose=False)
print("Best params:", res["params"])

# 3. Evaluate the refit model.
pred = search_model.predict(X_test)
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("Test R2:", r2_score(y_test, pred))

# 4. SHAP values: per-feature contributions for each prediction.
shap_values = search_model.get_feature_importance(Pool(X_test, y_test), type="ShapValues")
contrib = pd.DataFrame(shap_values[:, :-1], columns=X_test.columns)
base_value = shap_values[0, -1]
print("Base value:", base_value)
print(contrib.abs().mean().sort_values(ascending=False))

# The sum of contributions plus base value equals the raw prediction.
row = 0
print(contrib.iloc[row].sum() + base_value, pred[row])
```

Explanation:

1. `cv` returns per-iteration mean/std metrics; `idxmin` gives the best iteration index (the DataFrame index equals the iteration number).
2. `randomized_search` uses a train/test split internally to pick params, then refits on all of `X_train` because `refit=True`.
3. Report metrics on the held-out test set only.
4. SHAP values for trees are additive and sum to the raw model output; the last column is the expected value.

### Tutorial 3: Imbalanced fraud-style classification with threshold selection

Goal: detect a rare positive class (about 2 percent), handle imbalance, and choose a decision threshold that controls false positives.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, confusion_matrix
from catboost import CatBoostClassifier, Pool
from catboost.utils import select_threshold

X, y = make_classification(n_samples=60_000, n_features=25, n_informative=10,
                           weights=[0.98, 0.02], flip_y=0.005, random_state=7)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=7)

train_pool, val_pool, test_pool = Pool(X_tr, y_tr), Pool(X_val, y_val), Pool(X_te, y_te)

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="PRAUC",
    auto_class_weights="SqrtBalanced",
    random_seed=7,
    early_stopping_rounds=200,
    verbose=200,
)
model.fit(train_pool, eval_set=val_pool)

# Pick a threshold on the validation set that keeps FPR at or below 1 percent.
threshold = select_threshold(model, val_pool, FPR=0.01)
print("Chosen threshold:", threshold)

proba = model.predict_proba(test_pool)[:, 1]
pred = (proba >= threshold).astype(int)
print("Test average precision:", average_precision_score(y_te, proba))
print(confusion_matrix(y_te, pred))
```

Notes:

- `auto_class_weights="SqrtBalanced"` is gentler than `"Balanced"`. Class weights distort probabilities; if you need calibrated probabilities, train without weights and tune the threshold instead.
- The threshold is chosen on validation data, never on the test set.
- `PRAUC` focuses on the minority class; `AUC` can look excellent even when precision is poor.

### Tutorial 4: Text and categorical features together (review classification)

Goal: classify short product reviews as positive or negative using the review text plus a categorical product category.

```python
import pandas as pd
from catboost import CatBoostClassifier, Pool

train = pd.DataFrame({
    "review": [
        "great quality and fast shipping", "terrible, broke after one day",
        "works as expected, happy", "waste of money do not buy",
        "excellent value would recommend", "poor build quality and slow delivery",
        "love it, perfect gift", "stopped working, very disappointed",
    ] * 50,
    "category": ["electronics", "toys", "home", "electronics",
                 "home", "toys", "toys", "electronics"] * 50,
    "label": [1, 0, 1, 0, 1, 0, 1, 0] * 50,
})

train_pool = Pool(
    train[["review", "category"]], train["label"],
    text_features=["review"], cat_features=["category"],
)

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    eval_metric="Accuracy",
    text_processing={
        "tokenizers": [{"tokenizer_id": "Space", "separator_type": "ByDelimiter",
                        "lowercasing": "true"}],
        "dictionaries": [{"dictionary_id": "Word", "max_dictionary_size": "5000",
                          "occurrence_lower_bound": "2", "gram_order": "1"},
                         {"dictionary_id": "BiGram", "max_dictionary_size": "5000",
                          "occurrence_lower_bound": "2", "gram_order": "2"}],
        "feature_processing": {"default": [
            {"dictionaries_names": ["Word", "BiGram"], "feature_calcers": ["BoW"],
             "tokenizers_names": ["Space"]},
            {"dictionaries_names": ["Word"], "feature_calcers": ["NaiveBayes"],
             "tokenizers_names": ["Space"]},
        ]},
    },
    verbose=100,
)
model.fit(train_pool)

test = pd.DataFrame({"review": ["fast shipping, great value", "broke immediately, waste"],
                     "category": ["home", "toys"]})
print(model.predict(Pool(test, text_features=["review"], cat_features=["category"])))
```

How it works:

- Text columns are tokenized and turned into dictionaries of unigrams/bigrams.
- Feature calcers (`BoW`, `NaiveBayes`, `BM25`) generate numeric features from tokens, which the trees then split on.
- Omitting `text_processing` uses sensible defaults; the explicit version is shown so you know what to tune.
- `NaiveBayes` and `BM25` calcers are classification-only; `BoW` also works for regression.
- Text features are supported on CPU and GPU, but not every calcer is available on both; check the docs for your device.

## Performance & Best Practices

### Getting a strong model quickly

1. Start with defaults plus an `eval_set` and `early_stopping_rounds`. Let CatBoost choose the learning rate.
2. Make sure all categorical columns are in `cat_features` instead of one-hot encoding them yourself.
3. Tune in this order: `learning_rate` with `iterations` (via early stopping), then `depth` (4 to 10), then `l2_leaf_reg` (1 to 10), then `random_strength` and `bagging_temperature`, then `border_count`.
4. Lower the learning rate and increase iterations for the final model; this almost always helps a little.

### Speed

| Situation | Recommendation |
|---|---|
| Large dataset, NVIDIA GPU available | `task_type="GPU"`; GPU training is often many times faster for large data. |
| Training slow due to metrics | `metric_period=50` or more; avoid expensive `custom_metric`s on every iteration. |
| Many high-cardinality categorical features | Lower `max_ctr_complexity` (1 or 2) to limit feature combinations. |
| Slow on small data | `boosting_type="Plain"` instead of `"Ordered"`. |
| Need faster training at slight accuracy cost | Reduce `border_count` (32 or 64), use `bootstrap_type="Bernoulli"` with `subsample=0.66`. |
| Many features | `rsm=0.1` to `0.5` (CPU) samples features per split. |
| Reusing data across fits | Build one `Pool` and reuse it; quantize once with `pool.quantize()` for repeated experiments. |
| Memory pressure on CPU | `used_ram_limit="16gb"`; on GPU lower `gpu_ram_part`. |

### Inference

- Oblivious trees are fast; batch predictions are much faster than per-row calls in a loop.
- `model.predict(..., thread_count=-1)` uses all cores.
- For low-latency services consider the C/C++ applier, exported C++ code, or ONNX Runtime (numeric features only).
- Use `ntree_end` to trade accuracy for speed without retraining.

### Correctness and robustness

- Keep the same column order and dtypes at inference as in training; pass DataFrames with the same column names.
- Do not compute target encodings yourself before CatBoost; you would reintroduce leakage that CatBoost avoids.
- For time-ordered data, set `has_time=True` so CatBoost uses the data order instead of random permutations for target statistics, and validate with a time-based split.
- Use `random_seed` and log `model.get_all_params()` for reproducibility.
- Set `allow_writing_files=False` in serverless or read-only containers.
- Use `save_snapshot=True` for long trainings on preemptible machines.

### Regularization checklist

| Symptom | Try |
|---|---|
| Train metric far better than validation | Increase `l2_leaf_reg`, lower `depth`, increase `random_strength`, add `bagging_temperature` or `subsample`. |
| Validation keeps improving at the last iteration | Increase `iterations` or `learning_rate`. |
| Noisy validation curve | Lower `learning_rate`; use a larger validation set. |
| Overfitting on high-cardinality categoricals | Lower `max_ctr_complexity`, raise `model_size_reg`. |

## Common Errors & Troubleshooting

### Invalid type for cat_feature

```text
CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=2]=nan : cat_features must be integer or string, real number values and NaN values should be converted to string.
```

Cause: a categorical column contains floats or `NaN`.

Fix:

```python
for c in cat_cols:
    X[c] = X[c].astype(object).fillna("NA").astype(str)
```

If an integer column became float because of missing values, convert after filling: `X[c].fillna(-1).astype(int).astype(str)`.

### Cannot convert string to float

```text
CatBoostError: Bad value for num_feature[non-default value idx=0,feature_idx=3]="red": Cannot convert 'b'red'' to float
```

Cause: a string column was not declared in `cat_features`, so CatBoost treats it as numeric.

Fix: add the column to `cat_features` (or `text_features`):

```python
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
```

### Float numpy array with cat_features

```text
CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
```

Cause: a float numpy array cannot hold categorical values.

Fix: use a pandas DataFrame (with string/int categorical columns) or a numpy array of `dtype=object`.

### Only one unique target value

```text
CatBoostError: Target contains only one unique value
```

Cause: the label column is constant (often after a bad filter or split).

Fix: check `y.value_counts()`; ensure both classes are present in each training split.

### Multiple iteration aliases

```text
CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized.
```

Cause: for example `iterations` passed to the constructor and `n_estimators` set in a search grid.

Fix: use only one alias consistently.

### GPU not available

```text
CatBoostError: ... Environment for task type [GPU] not found
```

or errors mentioning no CUDA devices or an old driver.

Fix: check `catboost.utils.get_gpu_device_count()`, update the NVIDIA driver, verify `nvidia-smi`. macOS builds have no CUDA support; use `task_type="CPU"`.

### Writing catboost_info fails

```text
CatBoostError: ... Can't create train working dir: catboost_info
```

Cause: the current directory is read-only (serverless, some notebooks, containers).

Fix: `allow_writing_files=False` or `train_dir="/tmp/catboost_info"`.

### Feature mismatch at prediction time

Messages mention that a feature is categorical in the model but not in the dataset, or that the number of features differs from the model.

Cause: the prediction data has different columns, order, or types than training.

Fix: select columns as `X_new[model.feature_names_]` and apply the same type conversions used for training.

### Unknown class label in eval set

Cause: validation or test labels include a class not seen in training (common with rare classes and random splits).

Fix: stratify the split, or pass `class_names=[...]` to the classifier constructor to fix the class list.

### use_best_model without eval set

```text
Warning: You should provide test set for use best model. use_best_model parameter swiched to false value.
```

Cause: `use_best_model=True` without `eval_set`. Provide `eval_set` or drop the parameter. (The misspelling is in the actual message.)

### scikit-learn compatibility

With newer scikit-learn releases you may see `AttributeError: 'CatBoostClassifier' object has no attribute '__sklearn_tags__'` when using CatBoost inside scikit-learn meta-estimators. This comes from scikit-learn 1.6 changing its estimator-tag API. Fix: upgrade CatBoost to the latest 1.2.x release, or pin scikit-learn below 1.6 if you cannot upgrade.

### Slow training on CPU

Check for: very high `depth` (cost grows exponentially), `boosting_type="Ordered"` on large data, many categorical combinations (`max_ctr_complexity`), Python custom metrics, and `metric_period=1` with many eval sets.

## Interoperability

### scikit-learn

CatBoost estimators implement `fit`, `predict`, `predict_proba`, `get_params`, `set_params`, and `score`, so they work with scikit-learn utilities.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier

num_cols = ["age", "hours_per_week"]
cat_cols = ["workclass", "occupation"]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", SimpleImputer(strategy="constant", fill_value="NA"), cat_cols),
])

# After ColumnTransformer, categorical columns are at positions 2 and 3.
pipe = Pipeline([
    ("pre", pre),
    ("model", CatBoostClassifier(iterations=300, verbose=0, cat_features=[2, 3])),
])
# scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
```

When a `ColumnTransformer` outputs a numpy object array, refer to categorical features by position. Using `pre.set_output(transform="pandas")` lets you refer to them by name.

### ONNX Runtime

```python
import numpy as np
import onnxruntime as ort

clf.save_model("model.onnx", format="onnx")
sess = ort.InferenceSession("model.onnx")
inp = sess.get_inputs()[0].name
outputs = sess.run(None, {inp: X_te.to_numpy().astype(np.float32)})
print(outputs[0][:5])  # labels; outputs[1] holds probabilities
```

ONNX export supports numeric features only.

### XGBoost and LightGBM

All three are GBDT libraries with similar concepts. Rough parameter equivalents:

| CatBoost | XGBoost | LightGBM |
|---|---|---|
| `iterations` | `n_estimators` | `n_estimators` |
| `learning_rate` | `learning_rate` | `learning_rate` |
| `depth` | `max_depth` | `max_depth` / `num_leaves` |
| `l2_leaf_reg` | `reg_lambda` | `reg_lambda` |
| `rsm` | `colsample_bylevel` | `feature_fraction_bynode` |
| `subsample` | `subsample` | `bagging_fraction` |
| `cat_features` | `enable_categorical=True` + category dtype | `categorical_feature` |
| `early_stopping_rounds` | `early_stopping_rounds` | `callbacks=[lgb.early_stopping(n)]` |

Averaging CatBoost with XGBoost/LightGBM predictions, or stacking them with scikit-learn's `StackingClassifier`, often gives a small boost.

### Deep learning frameworks

CatBoost is complementary to PyTorch and TensorFlow: use a neural network to produce embeddings from images or text, then pass them to CatBoost as `embedding_features` or as plain numeric columns alongside tabular features.

## Cheat Sheet

### Setup and data

| Task | Code |
|---|---|
| Install | `pip install catboost` |
| Version | `catboost.__version__` |
| GPU count | `from catboost.utils import get_gpu_device_count; get_gpu_device_count()` |
| Build pool | `Pool(X, y, cat_features=cats, text_features=texts)` |
| Pool from file | `Pool("train.tsv", column_description="train.cd")` |
| Ranking pool | `Pool(X, y, group_id=qid)` |
| Categorical cols | `X.select_dtypes(["object", "category"]).columns.tolist()` |
| Fix NaN in categoricals | `X[c] = X[c].astype(object).fillna("NA").astype(str)` |

### Training

| Task | Code |
|---|---|
| Classifier | `CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6)` |
| Regressor | `CatBoostRegressor(loss_function="RMSE")` |
| Ranker | `CatBoostRanker(loss_function="YetiRank")` |
| Early stopping | `fit(X, y, eval_set=(Xv, yv), early_stopping_rounds=100)` |
| Silent | `verbose=0` or `logging_level="Silent"` |
| Log every 100 | `verbose=100` |
| GPU | `task_type="GPU", devices="0"` |
| Imbalance | `auto_class_weights="Balanced"` |
| Monotonic | `monotone_constraints={"price": -1}` |
| Quantile | `loss_function="Quantile:alpha=0.9"` |
| Multi-target regression | `loss_function="MultiRMSE"` |
| Continue training | `fit(X, y, init_model=old_model)` |
| Time-ordered data | `has_time=True` |
| No files written | `allow_writing_files=False` |

### Evaluation and tuning

| Task | Code |
|---|---|
| Best iteration | `model.get_best_iteration()` |
| Best scores | `model.get_best_score()` |
| Metric history | `model.get_evals_result()` |
| All params | `model.get_all_params()` |
| Cross-validation | `cv(Pool(X, y), params, fold_count=5, early_stopping_rounds=50)` |
| Grid search | `model.grid_search({"depth": [4, 6, 8]}, X, y)` |
| Random search | `model.randomized_search(dist, X, y, n_iter=20)` |
| Metrics on data | `model.eval_metrics(pool, ["AUC"], eval_period=10)` |
| ROC curve | `get_roc_curve(model, pool)` |
| Threshold at FPR | `select_threshold(model, pool, FPR=0.01)` |

### Prediction and explainability

| Task | Code |
|---|---|
| Labels | `model.predict(X)` |
| Probabilities | `model.predict_proba(X)` |
| Raw scores | `model.predict(X, prediction_type="RawFormulaVal")` |
| Use first N trees | `model.predict(X, ntree_end=N)` |
| Feature importance | `model.get_feature_importance(prettified=True)` |
| Loss-based importance | `model.get_feature_importance(pool, type="LossFunctionChange")` |
| SHAP values | `model.get_feature_importance(pool, type="ShapValues")` |
| Interactions | `model.get_feature_importance(type="Interaction")` |
| Feature selection | `model.select_features(train, eval_set=val, features_for_select=feats, num_features_to_select=10)` |
| Plot tree | `model.plot_tree(tree_idx=0, pool=pool)` |

### Persistence

| Task | Code |
|---|---|
| Save native | `model.save_model("m.cbm")` |
| Load native | `CatBoostClassifier().load_model("m.cbm")` |
| Export ONNX | `model.save_model("m.onnx", format="onnx")` |
| Export CoreML | `model.save_model("m.mlmodel", format="coreml")` |
| Export JSON | `model.save_model("m.json", format="json", pool=pool)` |
| Export C++ / Python | `model.save_model("m.cpp", format="cpp")` / `format="python"` |
| Blend models | `sum_models([m1, m2], weights=[0.5, 0.5])` |

## Further Resources

- Official documentation: https://catboost.ai/docs/
- Training parameters reference: https://catboost.ai/docs/en/references/training-parameters/
- Loss functions and metrics: https://catboost.ai/docs/en/concepts/loss-functions
- GitHub repository: https://github.com/catboost/catboost
- Tutorials (Jupyter notebooks): https://github.com/catboost/tutorials
- Release notes: https://github.com/catboost/catboost/releases
- PyPI: https://pypi.org/project/catboost/
- Paper: Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features", NeurIPS 2018. https://arxiv.org/abs/1706.09516
- Paper: Dorogush, Ershov, Gulin, "CatBoost: gradient boosting with categorical features support", 2018. https://arxiv.org/abs/1810.11363
- SHAP library: https://github.com/shap/shap
- Optuna: https://optuna.org/
