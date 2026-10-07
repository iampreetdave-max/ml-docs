# LightGBM

> Fast, memory-efficient gradient boosting with leaf-wise trees, histogram binning and native categorical support.

LightGBM (Light Gradient Boosting Machine) is a gradient-boosted decision tree framework from Microsoft that is designed for speed and scale. Histogram-based split finding, leaf-wise tree growth, Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) let it train on millions of rows in seconds to minutes on a laptop CPU, while matching or beating other boosting libraries in accuracy.

Covers LightGBM 4.x (examples verified against 4.7). Version 4.0 brought breaking changes that most online examples predate: early stopping and logging are configured through **callbacks** (`lgb.early_stopping`, `lgb.log_evaluation`) instead of `early_stopping_rounds` / `verbose_eval` arguments, `boosting="goss"` became `data_sample_strategy="goss"`, and pip builds use `--config-settings` instead of `--install-option`.

## Overview

### What it is

LightGBM provides:

- **Gradient boosting** of decision trees (`gbdt`), plus random forest (`rf`) and DART (`dart`) modes.
- **Histogram algorithm**: continuous features are bucketed into at most `max_bin` (255) bins once, before training. Split finding then works on histograms rather than sorted values.
- **Leaf-wise (best-first) growth**: each step splits the leaf with the largest loss reduction, producing deeper, asymmetric trees with fewer leaves for the same loss.
- **GOSS** (Gradient-based One-Side Sampling): keeps all large-gradient rows and samples small-gradient rows.
- **EFB** (Exclusive Feature Bundling): merges mutually exclusive sparse features (such as one-hot columns) into single bundles.
- **Native categorical features** using an optimal-partition split on sorted gradient statistics.
- **Missing values** handled natively, with a learned default direction.
- **Objectives** for regression (L2, L1, Huber, Fair, quantile, MAPE, Poisson, Gamma, Tweedie), binary and multiclass classification, cross-entropy, and learning to rank (LambdaRank, XE-NDCG).
- **Linear trees** (`linear_tree=True`): leaves contain linear models instead of constants.
- **Monotone and interaction constraints**, path smoothing, forced splits, cost-efficient gradient boosting.
- **Distributed training**: data-, feature- and voting-parallel tree learners; Dask integration; SynapseML for Spark.
- **GPU**: OpenCL (`device_type="gpu"`) and CUDA (`device_type="cuda"`) builds.

### History and maintainers

LightGBM was released by Microsoft Research (DMTK team) in late 2016 and described in the NeurIPS 2017 paper "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" by Guolin Ke and colleagues. It is maintained on GitHub under the `microsoft/LightGBM` organisation by a small core team of Microsoft and community maintainers, under the MIT license. Bindings exist for Python, R, C, and via third parties for Java, .NET (ML.NET), Julia and more.

| Version | Notable changes |
|---|---|
| 3.0 (2020) | Python API overhaul, `force_col_wise` / `force_row_wise`. |
| 3.1 to 3.3 | Linear trees (`linear_tree`), Dask estimators (`lightgbm.dask`). |
| 4.0 (2023) | Removed `early_stopping_rounds`, `verbose_eval`, `evals_result` from `train()` / `cv()` (use callbacks); removed `early_stopping_rounds`, `verbose` from sklearn `fit()`; `data_sample_strategy` parameter (GOSS moved out of `boosting`); new CUDA tree learner; quantised gradient training (`use_quantized_grad`); scikit-build-core based Python packaging. |
| 4.1 to 4.6 | Position-bias-aware ranking (`position` in `Dataset`), PyArrow input, newer Python versions. |
| 4.7 | sklearn `fit()` gains `eval_X` / `eval_y`; `eval_set` is deprecated (still works, with a warning). |

### When to use it

- Large tabular datasets (hundreds of thousands to hundreds of millions of rows) where training speed matters.
- Many features, especially sparse or one-hot ones (EFB helps).
- Rapid experimentation and hyperparameter search: each fit is cheap.
- Ranking problems (LambdaRank) and high-cardinality categoricals.
- Memory-constrained environments: the binned `Dataset` is compact.

### When not to use it

- Small datasets (under a few thousand rows): leaf-wise growth overfits easily; use stronger regularisation or a simpler model.
- Unstructured data (images, audio, text): use deep learning.
- When you need out-of-the-box GPU acceleration from `pip install` on every platform: XGBoost's standard wheels ship with CUDA, while LightGBM GPU support usually requires building from source.
- Extrapolation beyond the training range (all tree models predict constants outside it; `linear_tree=True` helps somewhat).

### Where it fits

```text
pandas / Polars / NumPy / Arrow / SciPy sparse
            |
   lgb.Dataset (binned, bundled)  <- or the sklearn API does this for you
            |
   lgb.train / LGBMClassifier     <- tuned with Optuna or sklearn search
            |
   pred_contrib (SHAP), feature_importance, plot_* helpers
            |
   model.txt / model_to_string, MLflow, ONNX (onnxmltools), Treelite
```

## Installation

### pip

```bash
python -m pip install -U lightgbm
```

Pre-built wheels exist for Linux, macOS (Intel and Apple Silicon) and Windows. They are CPU-only builds with OpenMP. Optional dependencies:

```bash
python -m pip install "lightgbm[pandas,scikit-learn]"   # DataFrame and sklearn API support
python -m pip install "lightgbm[dask]"                   # distributed training
python -m pip install matplotlib graphviz                # plotting helpers
```

### conda

```bash
conda install -c conda-forge lightgbm
```

### macOS: OpenMP

LightGBM needs the OpenMP runtime. If `import lightgbm` fails with `OSError: dlopen(... lib_lightgbm.dylib ...): Library not loaded: @rpath/libomp.dylib`:

```bash
brew install libomp
```

### GPU builds

The OpenCL GPU build (`device_type="gpu"`) and the CUDA build (`device_type="cuda"`) are compiled from source. In 4.x the Python package uses scikit-build-core, so CMake options are passed with `--config-settings`:

```bash
# OpenCL GPU version (needs OpenCL headers/ICD and Boost on Linux)
python -m pip install --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON lightgbm

# CUDA version (Linux, NVIDIA GPU, CUDA toolkit installed)
python -m pip install --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON lightgbm
```

The old `pip install lightgbm --install-option=--gpu` syntax no longer works with modern pip and LightGBM 4.x.

### Verifying the install

```python
import lightgbm as lgb
print(lgb.__version__)
```

Smoke test:

```python
import numpy as np
import lightgbm as lgb

X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
clf = lgb.LGBMClassifier(n_estimators=50, verbose=-1).fit(X, y)
print("accuracy:", clf.score(X, y))

# GPU check: raises LightGBMError "GPU Tree Learner was not enabled in this build" on CPU-only builds
try:
    lgb.train({"objective": "binary", "device_type": "gpu", "verbose": -1}, lgb.Dataset(X, y), 5)
    print("GPU build OK")
except lgb.basic.LightGBMError as e:
    print("GPU not available:", str(e).splitlines()[0])
```

## Core Concepts

### Gradient boosting recap

LightGBM builds an additive model of regression trees. At each iteration it computes, for every row, the gradient `g_i` and Hessian `h_i` of the loss with respect to the current prediction, then fits a tree whose leaves minimise a second-order approximation of the loss. With L2 regularisation `lambda_l2`, the optimal value of leaf j and the gain of a split are:

```text
w_j  = - G_j / (H_j + lambda_l2)
Gain = G_L^2/(H_L + lambda_l2) + G_R^2/(H_R + lambda_l2) - (G_L+G_R)^2/(H_L + H_R + lambda_l2)
```

where `G` and `H` are sums of gradients and Hessians of the rows in a node. (LightGBM omits the constant 0.5 factor; `lambda_l1` applies soft-thresholding to `G`.) A split is kept only if its gain exceeds `min_gain_to_split`, and children must satisfy `min_data_in_leaf` and `min_sum_hessian_in_leaf`. The final prediction is `init_score + learning_rate * sum of tree outputs`, passed through the link function (sigmoid, softmax, exp) for classification and count objectives.

### Histogram-based split finding

Before training, LightGBM bins each feature into at most `max_bin` buckets using quantiles sampled from `bin_construct_sample_cnt` rows. Training then only needs, for each node, a histogram of summed gradients and Hessians per bin, so evaluating all splits of a feature costs O(#bins) rather than O(#rows). Two tricks make it fast:

- **Histogram subtraction**: a child's histogram equals parent minus sibling, so only the smaller child is computed.
- **Compact storage**: bins are stored as 8-bit or 16-bit integers.

```python
import numpy as np
import lightgbm as lgb

X = np.random.default_rng(0).normal(size=(10_000, 3))
y = X[:, 0] ** 2 + np.random.default_rng(1).normal(scale=0.1, size=10_000)
ds = lgb.Dataset(X, y, params={"max_bin": 63}).construct()
print("rows:", ds.num_data(), "features:", ds.num_feature())
```

### Leaf-wise growth

Level-wise (depth-wise) growth splits every node at the current depth. Leaf-wise growth picks the single leaf with the largest gain anywhere in the tree:

```text
level-wise (XGBoost default)        leaf-wise (LightGBM)
          o                                  o
        /   \                              /   \
       o     o                            o     o
      / \   / \                          / \
     o   o o   o                        o   o
                                           / \
                                          o   o
```

With the same number of leaves, leaf-wise trees reach lower training loss, but they can grow very deep on small data and overfit. The primary complexity control is therefore `num_leaves` (default 31), not `max_depth` (default -1, unlimited). A useful rule: keep `num_leaves` below `2 ** max_depth` if you set both; for example `max_depth=7` with `num_leaves` around 70 to 100.

### GOSS: Gradient-based One-Side Sampling

Rows with small gradients are already well fitted. GOSS keeps the top `top_rate` (default 0.2) fraction of rows by gradient magnitude, randomly samples `other_rate` (default 0.1) of the rest, and up-weights the sampled small-gradient rows by `(1 - top_rate) / other_rate` to keep the gradient estimate unbiased. Enable with `data_sample_strategy="goss"` (4.0+). The previous `boosting="goss"` still works but logs a deprecation warning.

### EFB: Exclusive Feature Bundling

Many sparse features are rarely non-zero at the same time (one-hot columns are the extreme case). EFB greedily bundles such features into a single feature with offset bins, reducing the effective feature count and histogram cost. It is on by default (`enable_bundle=True`).

### Categorical features

Pass categorical columns as pandas `category` dtype (picked up automatically with `categorical_feature="auto"`) or list them via `categorical_feature=[...]` with non-negative integer codes. For each categorical feature LightGBM sorts categories by `sum(gradient) / (sum(hessian) + cat_smooth)` and finds the best split point along that order, which is equivalent to an optimal two-way partition. Related parameters: `max_cat_to_onehot` (4), `max_cat_threshold` (32), `cat_l2` (10), `cat_smooth` (10), `min_data_per_group` (100). This is usually better than one-hot encoding for high-cardinality features, but can overfit rare categories; increase `cat_smooth` and `min_data_per_group` if so.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "city": pd.Categorical(rng.choice(["paris", "lyon", "nice", "lille"], 5000)),
    "x": rng.normal(size=5000),
})
y = (df["city"].isin(["paris", "nice"]).astype(int) + (df["x"] > 0)).clip(0, 1)
model = lgb.LGBMClassifier(n_estimators=50, verbose=-1).fit(df, y)   # city used as categorical
print(model.booster_.dump_model()["feature_infos"]["city"])
```

### Missing values and zeros

`use_missing=True` (default) treats `NaN` as missing and learns which side missing values go at each split. `zero_as_missing=True` additionally treats zeros as missing, which is useful for sparse data where 0 means "absent".

### Learning rate, iterations and early stopping

`learning_rate` (default 0.1) and `num_iterations` (default 100) trade off. Set a small learning rate (0.01 to 0.05), a large number of iterations, and use **early stopping** on a validation set via the `lgb.early_stopping` callback.

### Two APIs

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# Native API: Dataset + params dict + lgb.train
dtrain = lgb.Dataset(X_tr, label=y_tr)
dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)   # reuse training bins
params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
          "num_leaves": 15, "verbose": -1}
booster = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
print("native best iteration:", booster.best_iteration)

# scikit-learn API: same engine, estimator interface
clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=15, verbose=-1)
clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
print("sklearn best iteration:", clf.best_iteration_)
```

Parameter aliases: LightGBM accepts many aliases for each parameter. The sklearn API uses scikit-learn-style names that map to native ones:

| sklearn constructor | Native parameter |
|---|---|
| `n_estimators` | `num_iterations` (`num_boost_round` argument to `train`) |
| `boosting_type` | `boosting` |
| `min_child_samples` | `min_data_in_leaf` |
| `min_child_weight` | `min_sum_hessian_in_leaf` |
| `min_split_gain` | `min_gain_to_split` |
| `subsample` | `bagging_fraction` |
| `subsample_freq` | `bagging_freq` |
| `colsample_bytree` | `feature_fraction` |
| `reg_alpha` | `lambda_l1` |
| `reg_lambda` | `lambda_l2` |
| `subsample_for_bin` | `bin_construct_sample_cnt` |
| `random_state` | `seed` |
| `n_jobs` | `num_threads` |

Important gotcha: `subsample` (bagging) has **no effect** unless `subsample_freq` (`bagging_freq`) is greater than 0.

Never pass the same parameter twice under two aliases (for example `min_child_samples` and `min_data_in_leaf`); LightGBM keeps one and warns.

### Verbosity

LightGBM logs a lot at the default `verbosity=1`. Pass `verbose=-1` (sklearn) or `"verbose": -1` (native params) to silence info and warnings. Training progress printing is controlled separately by the `lgb.log_evaluation(period)` callback.

## API Reference

### lightgbm.Dataset

```python
lgb.Dataset(data, label=None, reference=None, weight=None, group=None, init_score=None,
            feature_name="auto", categorical_feature="auto", params=None, free_raw_data=True,
            position=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | array, DataFrame, sparse, str/Path | required | NumPy, pandas, SciPy sparse, PyArrow, a list of `lgb.Sequence`, or a path to a CSV/TSV/LibSVM/binary file. |
| `label` | array-like | `None` | Target values. Binary: 0/1. Multiclass: 0..num_class-1. |
| `reference` | Dataset | `None` | For validation sets: reuse the training set's bin boundaries. Always set it. |
| `weight` | array-like | `None` | Per-row weights. |
| `group` | array-like | `None` | Ranking: sizes of consecutive query groups (sum equals rows). |
| `init_score` | array-like | `None` | Initial raw scores (offsets), e.g. log exposure or another model's margin. |
| `feature_name` | list or `"auto"` | `"auto"` | Names; taken from DataFrame columns when `"auto"`. |
| `categorical_feature` | list or `"auto"` | `"auto"` | Names or indices of categorical columns; `"auto"` uses pandas `category` dtype. |
| `params` | dict | `None` | Dataset parameters such as `max_bin`, `min_data_in_bin`, `linear_tree`. |
| `free_raw_data` | bool | `True` | Free the raw data after constructing bins. Set `False` to reuse the Dataset with changed parameters. |
| `position` | array-like | `None` | Ranking: display position for unbiased LambdaMART (4.1+). |

The Dataset is **lazy**: bins are built on first use (`construct()`). Useful methods: `construct()`, `num_data()`, `num_feature()`, `get_label()`, `set_weight()`, `set_init_score()`, `set_group()`, `save_binary(path)` (fast reload), `subset(used_indices)`, `create_valid(data, label)`, `get_feature_name()`.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb

df = pd.DataFrame({"x1": [1.0, np.nan, 3.0, 4.0] * 50,
                   "color": pd.Categorical(["r", "g", "b", "r"] * 50)})
y = np.tile([0, 1, 0, 1], 50)
train = lgb.Dataset(df, label=y, params={"max_bin": 63}, free_raw_data=False)
train.construct()
print(train.num_data(), train.num_feature(), train.get_feature_name())
train.save_binary("train.bin")          # reload later with lgb.Dataset("train.bin")
```

### lightgbm.train

```python
lgb.train(params, train_set, num_boost_round=100, valid_sets=None, valid_names=None,
          feval=None, init_model=None, keep_training_booster=False, callbacks=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | dict | required | Booster parameters (see the hyperparameter reference). |
| `train_set` | Dataset | required | Training data. |
| `num_boost_round` | int | `100` | Number of iterations (overridden by `num_iterations` in `params`). |
| `valid_sets` | list of Dataset | `None` | Datasets evaluated each iteration. Include `train_set` to also see training metrics. |
| `valid_names` | list of str | `None` | Names shown in logs and in recorded results (default `valid_0`, ...; the training set is named `training`). |
| `feval` | callable or list | `None` | Custom metric(s): `f(preds, eval_data) -> (name, value, is_higher_better)`. |
| `init_model` | str, Path, Booster | `None` | Continue training from an existing model. |
| `keep_training_booster` | bool | `False` | Keep the returned booster trainable (otherwise it is converted for prediction). |
| `callbacks` | list | `None` | `lgb.early_stopping`, `lgb.log_evaluation`, `lgb.record_evaluation`, `lgb.reset_parameter`, or custom. |

Returns a `Booster`. With early stopping, `booster.best_iteration` is set and `predict` uses it by default.

Removed in 4.0: `early_stopping_rounds`, `verbose_eval`, `evals_result`, `learning_rates` arguments. Use callbacks instead (or `"early_stopping_round"` inside `params`, which LightGBM turns into the callback automatically).

```python
import lightgbm as lgb

history = {}
booster = lgb.train(
    {"objective": "binary", "metric": ["auc", "binary_logloss"], "learning_rate": 0.05,
     "num_leaves": 31, "verbose": -1},
    dtrain, num_boost_round=5000,
    valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
               lgb.log_evaluation(period=200),
               lgb.record_evaluation(history)],
)
print(booster.best_iteration, booster.best_score["valid"]["auc"])
print(len(history["valid"]["auc"]))
```

Early stopping considers every metric on every validation set (except the training set) unless `first_metric_only=True`, in which case only the first metric in `metric` is used.

### lightgbm.cv

```python
lgb.cv(params, train_set, num_boost_round=100, folds=None, nfold=5, stratified=True,
       shuffle=True, metrics=None, feval=None, init_model=None, fpreproc=None, seed=0,
       callbacks=None, eval_train_metric=False, return_cvbooster=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `folds` | splitter or iterable | `None` | Any sklearn splitter (`GroupKFold`, `TimeSeriesSplit`) or explicit index pairs. |
| `nfold` | int | `5` | Folds when `folds` is None. |
| `stratified` | bool | `True` | Stratify folds; set `False` for regression. |
| `metrics` | str or list | `None` | Override metrics. |
| `eval_train_metric` | bool | `False` | Also report training metrics. |
| `return_cvbooster` | bool | `False` | Return the fold models as a `CVBooster` under key `"cvbooster"`. |

Returns a dict of lists keyed like `"valid binary_logloss-mean"` and `"valid binary_logloss-stdv"` (the `"valid "` prefix was added in 4.0; `"train ..."` keys appear with `eval_train_metric=True`). With early stopping, lists are truncated at the best iteration.

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
res = lgb.cv({"objective": "binary", "metric": "auc", "learning_rate": 0.05, "verbose": -1},
             lgb.Dataset(X, y), num_boost_round=2000, nfold=5, stratified=True, seed=0,
             callbacks=[lgb.early_stopping(100, verbose=False)], return_cvbooster=True)
print("best rounds:", len(res["valid auc-mean"]), "CV AUC:", round(res["valid auc-mean"][-1], 4))
cvb = res["cvbooster"]
fold_preds = cvb.predict(X[:5])        # list with one prediction array per fold
```

### lightgbm.Booster

```python
lgb.Booster(params=None, train_set=None, model_file=None, model_str=None)
```

| Method | Signature | Description |
|---|---|---|
| `predict` | `(data, start_iteration=0, num_iteration=None, raw_score=False, pred_leaf=False, pred_contrib=False, data_has_header=False, validate_features=False, **kwargs)` | Predictions. `num_iteration=None` uses `best_iteration` if set. `raw_score=True` gives margins. `pred_contrib=True` gives SHAP values (last column = expected value). `pred_leaf=True` gives leaf indices. Takes raw data, not a `Dataset`. |
| `save_model` | `(filename, num_iteration=None, start_iteration=0, importance_type="split")` | Save as a text model file. |
| `model_to_string` / `model_from_string` | | Serialise to / from a string. |
| `dump_model` | `(num_iteration=None, start_iteration=0, importance_type="split", object_hook=None)` | JSON-like dict of the full model. |
| `feature_importance` | `(importance_type="split", iteration=None)` | `"split"` (count) or `"gain"` (total gain). |
| `feature_name` | `()` | Feature names. |
| `num_trees`, `current_iteration`, `num_feature` | `()` | Model size info. |
| `best_iteration`, `best_score` | attributes | Set by early stopping. |
| `trees_to_dataframe` | `()` | All nodes as a pandas DataFrame. |
| `refit` | `(data, label, decay_rate=0.9, ...)` | Refit leaf values on new data, keeping tree structure. |
| `update` | `(train_set=None, fobj=None)` | Run one more boosting iteration manually. |
| `rollback_one_iter` | `()` | Remove the last iteration. |
| `reset_parameter` | `(params)` | Change parameters (e.g. learning rate) mid-training. |
| `get_leaf_output`, `get_split_value_histogram` | | Introspection. |

```python
# continuing with booster and X from the train example above
import numpy as np
import lightgbm as lgb

proba = booster.predict(X[:5])                          # uses best_iteration
margin = booster.predict(X[:5], raw_score=True)
contrib = booster.predict(X[:5], pred_contrib=True)      # (5, n_features + 1)
assert np.allclose(contrib.sum(axis=1), margin)

booster.save_model("model.txt")                           # best_iteration is saved by default
loaded = lgb.Booster(model_file="model.txt")
assert np.allclose(loaded.predict(X[:5]), proba)
```

### scikit-learn API

#### LGBMClassifier / LGBMRegressor / LGBMRanker

```python
lgb.LGBMClassifier(*, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1,
                   n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None,
                   min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20,
                   subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0,
                   reg_lambda=0.0, random_state=None, n_jobs=None, importance_type="split",
                   **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `boosting_type` | str | `"gbdt"` | `"gbdt"`, `"dart"`, `"rf"`. |
| `num_leaves` | int | `31` | Maximum leaves per tree: the main capacity knob. |
| `max_depth` | int | `-1` | Depth limit; -1 means none. |
| `learning_rate` | float | `0.1` | Shrinkage. |
| `n_estimators` | int | `100` | Boosting iterations. |
| `objective` | str or callable | `None` | Inferred: `"regression"`, `"binary"` / `"multiclass"`, `"lambdarank"`. |
| `class_weight` | dict, `"balanced"`, None | `None` | Classifier only. |
| `min_split_gain` | float | `0.0` | Minimum gain to split. |
| `min_child_weight` | float | `1e-3` | Minimum Hessian sum in a leaf. |
| `min_child_samples` | int | `20` | Minimum rows in a leaf. |
| `subsample` | float | `1.0` | Row fraction; needs `subsample_freq > 0`. |
| `subsample_freq` | int | `0` | Bagging every k iterations. |
| `colsample_bytree` | float | `1.0` | Feature fraction per tree. |
| `reg_alpha` / `reg_lambda` | float | `0.0` | L1 / L2 on leaf values. |
| `random_state` | int | `None` | Seed. |
| `n_jobs` | int | `None` | Threads; `None` or -1 uses all (OpenMP default). |
| `importance_type` | str | `"split"` | For `feature_importances_`: `"split"` or `"gain"`. |
| `**kwargs` | | | Any native parameter, e.g. `verbose=-1`, `max_bin=127`, `cat_smooth=20`. |

`fit` signature (classifier; regressor is the same without `eval_class_weight`):

```python
LGBMClassifier.fit(X, y, sample_weight=None, init_score=None, eval_set=None, eval_names=None,
                   eval_sample_weight=None, eval_class_weight=None, eval_init_score=None,
                   eval_metric=None, feature_name="auto", categorical_feature="auto",
                   callbacks=None, init_model=None)
# 4.7+: also keyword-only eval_X=..., eval_y=... (eval_set deprecated)
```

| `fit` parameter | Description |
|---|---|
| `eval_set` | List of `(X, y)` tuples for evaluation and early stopping. In 4.7+ prefer `eval_X=(X_val,)`, `eval_y=(y_val,)`. |
| `eval_metric` | Extra metric name(s) or callable(s) `f(y_true, y_pred) -> (name, value, is_higher_better)`. |
| `categorical_feature` | Names or indices of categorical columns (or `"auto"` for pandas category). |
| `callbacks` | `[lgb.early_stopping(50), lgb.log_evaluation(100)]`. |
| `init_model` | Continue training. |

Removed in 4.0: `fit(..., early_stopping_rounds=..., verbose=...)`. Use callbacks.

Attributes after fit: `booster_`, `best_iteration_`, `best_score_`, `evals_result_`, `feature_importances_`, `feature_name_`, `n_features_in_`, `feature_names_in_`, `classes_`, `n_classes_`, `objective_`.

`predict(X, raw_score=False, start_iteration=0, num_iteration=None, pred_leaf=False, pred_contrib=False, validate_features=False)` and `predict_proba(X, ...)` use `best_iteration_` automatically.

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=0)

clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=15,
                         min_child_samples=10, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
print("best iteration:", clf.best_iteration_, dict(clf.best_score_["valid_0"]))
print("test AUC:", round(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]), 4))
```

`LGBMRanker.fit` adds `group` (sizes of consecutive query groups), `eval_group`, and `eval_at` (default `(1, 2, 3, 4, 5)`, the NDCG cut-offs). See Tutorial 3.

### Callbacks

| Callback | Signature | Purpose |
|---|---|---|
| `lgb.early_stopping` | `(stopping_rounds, first_metric_only=False, verbose=True, min_delta=0.0)` | Stop when no validation metric improves for `stopping_rounds` iterations. `min_delta` (4.0+) sets a minimum improvement. |
| `lgb.log_evaluation` | `(period=1, show_stdv=True)` | Print metrics every `period` iterations. Replaces `verbose_eval`. |
| `lgb.record_evaluation` | `(eval_result)` | Store metric history into the given dict. Replaces `evals_result`. |
| `lgb.reset_parameter` | `(**kwargs)` | Per-iteration parameter schedules: a list or a function of the iteration index. |

```python
import lightgbm as lgb

schedule = lgb.reset_parameter(learning_rate=lambda it: 0.1 * (0.995 ** it))
booster = lgb.train({"objective": "binary", "verbose": -1}, dtrain, 500, valid_sets=[dvalid],
                    callbacks=[schedule, lgb.early_stopping(50, verbose=False)])
```

A custom callback is any callable receiving a `CallbackEnv` named tuple (`model`, `params`, `iteration`, `begin_iteration`, `end_iteration`, `evaluation_result_list`). Raise `lgb.callback.EarlyStopException(best_iteration, best_score)` to stop.

```python
def print_every_100(env):
    if env.iteration % 100 == 0:
        print(env.iteration, [(name, metric, round(val, 4)) for name, metric, val, _ in env.evaluation_result_list])
```

### Plotting

```python
lgb.plot_importance(booster, ax=None, height=0.2, importance_type="auto", max_num_features=None,
                    ignore_zero=True, figsize=None, precision=3, ...)
lgb.plot_metric(booster_or_evals_result, metric=None, dataset_names=None, ...)
lgb.plot_tree(booster, ax=None, tree_index=0, show_info=None, ...)          # needs graphviz
lgb.create_tree_digraph(booster, tree_index=0, show_info=None, ...)          # graphviz Digraph
lgb.plot_split_value_histogram(booster, feature, bins=None, ...)
```

```python
import matplotlib.pyplot as plt
lgb.plot_importance(clf, importance_type="gain", max_num_features=15)
lgb.plot_metric(clf.evals_result_, metric="auc")
plt.show()
```

### Dask (distributed)

```python
from dask.distributed import Client, LocalCluster
import dask.array as da
import lightgbm as lgb

with LocalCluster(n_workers=2) as cluster, Client(cluster) as client:
    X = da.random.random((200_000, 20), chunks=(50_000, 20))
    y = (X[:, 0] > 0.5).astype(int)
    model = lgb.DaskLGBMClassifier(n_estimators=100, client=client, verbose=-1)
    model.fit(X, y)
    preds = model.predict(X).compute()
    local_model = model.to_local()          # plain LGBMClassifier
```

`DaskLGBMRegressor` and `DaskLGBMRanker` are also available. For Spark, use SynapseML's `LightGBMClassifier`.

### Full hyperparameter reference

Parameters are passed in the `params` dict (native) or as constructor keyword arguments (sklearn). Names below are canonical; common aliases are listed.

#### Core parameters

| Parameter (aliases) | Type | Default | Description |
|---|---|---|---|
| `objective` (`application`) | str | `"regression"` | See the objectives table below. |
| `boosting` (`boosting_type`) | str | `"gbdt"` | `"gbdt"`, `"rf"` (random forest, needs bagging), `"dart"`. |
| `data_sample_strategy` | str | `"bagging"` | `"bagging"` or `"goss"` (4.0+). |
| `num_iterations` (`n_estimators`, `num_boost_round`) | int | `100` | Boosting rounds. |
| `learning_rate` (`eta`, `shrinkage_rate`) | float | `0.1` | Shrinkage. |
| `num_leaves` (`max_leaves`) | int | `31` | Max leaves per tree (2 to 131072). |
| `tree_learner` | str | `"serial"` | `"serial"`, `"feature"`, `"data"`, `"voting"` (distributed). |
| `num_threads` (`n_jobs`, `nthread`) | int | `0` | 0 = OpenMP default (all cores). Use physical core count for best speed. |
| `device_type` (`device`) | str | `"cpu"` | `"cpu"`, `"gpu"` (OpenCL), `"cuda"`. |
| `seed` (`random_state`) | int | `None` | Master seed for all other seeds. |
| `deterministic` | bool | `False` | With `force_col_wise` / `force_row_wise`, gives reproducible results across runs (CPU). |

#### Learning control parameters

| Parameter (aliases) | Type | Default | Description |
|---|---|---|---|
| `force_col_wise` | bool | `False` | Force column-wise histogram building (good when many features or threads). |
| `force_row_wise` | bool | `False` | Force row-wise (good with many rows, few features). |
| `max_depth` | int | `-1` | Depth limit; <= 0 means no limit. |
| `min_data_in_leaf` (`min_child_samples`) | int | `20` | Minimum rows per leaf. Very important for small data. |
| `min_sum_hessian_in_leaf` (`min_child_weight`) | float | `1e-3` | Minimum Hessian sum per leaf. |
| `bagging_fraction` (`subsample`) | float | `1.0` | Row sampling fraction. |
| `pos_bagging_fraction`, `neg_bagging_fraction` | float | `1.0` | Class-specific bagging for imbalanced binary. |
| `bagging_freq` (`subsample_freq`) | int | `0` | Perform bagging every k iterations; 0 disables bagging. |
| `bagging_seed` | int | `3` | Bagging seed. |
| `feature_fraction` (`colsample_bytree`) | float | `1.0` | Feature fraction per tree. |
| `feature_fraction_bynode` (`colsample_bynode`) | float | `1.0` | Feature fraction per split. |
| `extra_trees` | bool | `False` | Random split thresholds (one per feature), more regularisation, faster. |
| `early_stopping_round` (`early_stopping_rounds`, `n_iter_no_change`) | int | `0` | If > 0 and validation sets exist, enables early stopping via an automatically added callback. |
| `first_metric_only` | bool | `False` | Early stop on the first metric only. |
| `max_delta_step` | float | `0.0` | Caps leaf output; <= 0 means no cap. |
| `lambda_l1` (`reg_alpha`) | float | `0.0` | L1 regularisation. |
| `lambda_l2` (`reg_lambda`) | float | `0.0` | L2 regularisation. |
| `linear_lambda` | float | `0.0` | L2 for linear tree leaf models. |
| `min_gain_to_split` (`min_split_gain`) | float | `0.0` | Minimum gain for a split. |
| `drop_rate` | float | `0.1` | DART: fraction of trees dropped. |
| `max_drop` | int | `50` | DART: max trees dropped per iteration. |
| `skip_drop` | float | `0.5` | DART: probability of skipping dropout. |
| `xgboost_dart_mode`, `uniform_drop` | bool | `False` | DART variants. |
| `top_rate` | float | `0.2` | GOSS: retained large-gradient fraction. |
| `other_rate` | float | `0.1` | GOSS: sampled small-gradient fraction. |
| `min_data_per_group` | int | `100` | Minimum rows per categorical group. |
| `max_cat_threshold` | int | `32` | Max categories in one side of a categorical split. |
| `cat_l2` | float | `10.0` | L2 regularisation in categorical splits. |
| `cat_smooth` | float | `10.0` | Smoothing of category statistics; higher reduces overfitting on rare categories. |
| `max_cat_to_onehot` | int | `4` | Use one-vs-rest splits when categories <= this. |
| `monotone_constraints` | list of int | `None` | One of -1, 0, 1 per feature. |
| `monotone_constraints_method` | str | `"basic"` | `"basic"`, `"intermediate"`, `"advanced"` (less restrictive, slower). |
| `monotone_penalty` | float | `0.0` | Penalise monotone splits near the root. |
| `interaction_constraints` | list of lists | `None` | Feature index groups allowed to interact, e.g. `[[0, 1, 2], [2, 3]]`. |
| `path_smooth` | float | `0.0` | Shrinks leaf values toward the parent; helps with small leaves. |
| `forcedsplits_filename` | str | `""` | JSON file with forced top splits. |
| `refit_decay_rate` | float | `0.9` | Decay for `Booster.refit`. |
| `cegb_tradeoff`, `cegb_penalty_split`, `cegb_penalty_feature_lazy`, `cegb_penalty_feature_coupled` | | | Cost-efficient gradient boosting penalties. |
| `use_quantized_grad` | bool | `False` | Quantised gradients for faster training (4.0+); see `num_grad_quant_bins`. |
| `verbosity` (`verbose`) | int | `1` | < 0 fatal only, 0 errors/warnings, 1 info, > 1 debug. |

#### Dataset (IO) parameters

| Parameter (aliases) | Type | Default | Description |
|---|---|---|---|
| `linear_tree` | bool | `False` | Fit linear models in leaves. Dataset parameter: set it on the `Dataset` or in params before construction. Categorical features are used for splits but not inside the leaf models; not supported with the `regression_l1` objective. |
| `max_bin` | int | `255` | Max bins per feature. Smaller is faster and more regularised; larger is more precise. |
| `max_bin_by_feature` | list | `None` | Per-feature bin limits. |
| `min_data_in_bin` | int | `3` | Minimum rows per bin. |
| `bin_construct_sample_cnt` (`subsample_for_bin`) | int | `200000` | Rows sampled to build bins. |
| `data_random_seed` | int | `1` | Seed for bin sampling. |
| `use_missing` | bool | `True` | Treat NaN as missing. |
| `zero_as_missing` | bool | `False` | Also treat zeros as missing. |
| `feature_pre_filter` | bool | `True` | Drop unsplittable features at Dataset construction based on `min_data_in_leaf`. Set `False` if you tune `min_data_in_leaf` on a reused Dataset. |
| `enable_bundle` | bool | `True` | Exclusive Feature Bundling. |
| `categorical_feature` (`cat_feature`) | list | `""` | Categorical columns. |
| `two_round` | bool | `False` | Two-pass file loading for large text files. |
| `header`, `label_column`, `weight_column`, `group_column`, `ignore_column` | | | Options for loading from text files. |

#### Objective parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_class` | int | `1` | Required for `multiclass` / `multiclassova` in the native API. |
| `is_unbalance` (`unbalance`) | bool | `False` | Re-weight classes automatically for binary. Do not combine with `scale_pos_weight`. |
| `scale_pos_weight` | float | `1.0` | Positive class weight for binary. |
| `sigmoid` | float | `1.0` | Sigmoid steepness for binary / lambdarank. |
| `boost_from_average` | bool | `True` | Start from the label mean (regression, binary, cross-entropy). |
| `reg_sqrt` | bool | `False` | Fit `sqrt(label)` for regression with large ranges. |
| `alpha` | float | `0.9` | Huber delta and quantile level. |
| `fair_c` | float | `1.0` | Fair loss parameter. |
| `poisson_max_delta_step` | float | `0.7` | Safeguard for Poisson. |
| `tweedie_variance_power` | float | `1.5` | In [1, 2). |
| `lambdarank_truncation_level` | int | `30` | Top positions used in LambdaRank pairs. |
| `lambdarank_norm` | bool | `True` | Normalise lambdas per query. |
| `label_gain` | list | `0,1,3,7,15,...` | Gain per relevance label (`2^i - 1`). Extend for labels > 30. |
| `lambdarank_position_bias_regularization` | float | `0.0` | Regularisation for position-bias estimation (4.1+). |

Objectives:

| Objective (aliases) | Task |
|---|---|
| `regression` (`l2`, `mse`, `regression_l2`) | Squared loss. |
| `regression_l1` (`l1`, `mae`) | Absolute loss. |
| `huber` | Huber loss (`alpha` = delta). |
| `fair` | Fair loss. |
| `poisson` | Counts (log link). |
| `quantile` | Quantile regression (`alpha` = quantile). |
| `mape` | Mean absolute percentage error. |
| `gamma` | Positive skewed targets (log link). |
| `tweedie` | Zero-inflated positive targets (insurance claims). |
| `binary` | Binary log loss; labels 0/1. |
| `multiclass` (`softmax`) | Multiclass softmax; set `num_class`. |
| `multiclassova` (`ova`) | One-vs-all multiclass. |
| `cross_entropy` (`xentropy`) | Labels are probabilities in [0, 1]. |
| `cross_entropy_lambda` (`xentlambda`) | Alternative parameterisation. |
| `lambdarank` | LambdaMART with NDCG. |
| `rank_xendcg` (`xendcg`) | XE-NDCG ranking objective, faster than lambdarank. |

#### Metric parameters

| Parameter | Default | Description |
|---|---|---|
| `metric` | `""` (objective's default) | One or a list: `l1`, `l2`, `rmse`, `quantile`, `mape`, `huber`, `fair`, `poisson`, `gamma`, `gamma_deviance`, `tweedie`, `ndcg`, `map`, `auc`, `average_precision`, `binary_logloss`, `binary_error`, `auc_mu`, `multi_logloss`, `multi_error`, `cross_entropy`, `kullback_leibler`, or `"None"` to disable. |
| `metric_freq` | `1` | Output frequency. |
| `is_provide_training_metric` | `False` | Report metrics on training data. |
| `eval_at` | `1,2,3,4,5` | NDCG/MAP cut-offs. |
| `multi_error_top_k` | `1` | Top-k for `multi_error`. |

#### GPU parameters

| Parameter | Default | Description |
|---|---|---|
| `gpu_platform_id` | `-1` | OpenCL platform. |
| `gpu_device_id` | `-1` | Device within the platform (or CUDA device). |
| `gpu_use_dp` | `False` | Double precision on GPU (slower). |
| `num_gpu` | `1` | Number of GPUs (CUDA). |

## Tutorials

### Tutorial 1: Binary classification with categorical features, early stopping and SHAP

Goal: predict income above 50K on the Adult census dataset with native categoricals, the native API, callbacks and SHAP contributions.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# 1. Load and prepare: string columns -> pandas category dtype
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
y = (y == ">50K").astype(int)
cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
X[cat_cols] = X[cat_cols].astype("category")

# 2. Three-way split
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=0.2,
                                                      stratify=y_tmp, random_state=0)

# 3. Datasets: the validation set references the training set (shared bins and categories)
dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "cat_smooth": 20,
    "min_data_per_group": 50,
    "seed": 42,
    "verbose": -1,
}

# 4. Train with early stopping on the FIRST metric (auc) and periodic logging
evals = {}
booster = lgb.train(
    params, dtrain, num_boost_round=10_000,
    valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(200, first_metric_only=True),
               lgb.log_evaluation(500),
               lgb.record_evaluation(evals)],
)
print("best iteration:", booster.best_iteration,
      "valid AUC:", round(booster.best_score["valid"]["auc"], 4))

# 5. Test-set evaluation (predict uses best_iteration by default)
proba = booster.predict(X_test)
print("test ROC AUC:", round(roc_auc_score(y_test, proba), 4),
      "PR AUC:", round(average_precision_score(y_test, proba), 4))
print(classification_report(y_test, (proba >= 0.5).astype(int), digits=3))

# 6. Importance: gain vs. SHAP
gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
contrib = booster.predict(X_test, pred_contrib=True)
shap_imp = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=booster.feature_name())
print(pd.DataFrame({"gain": gain / gain.sum(), "mean_abs_shap": shap_imp})
        .sort_values("mean_abs_shap", ascending=False).head(10).round(4))

# 7. Save and reload as text
booster.save_model("adult_lgb.txt")
reloaded = lgb.Booster(model_file="adult_lgb.txt")
assert np.allclose(reloaded.predict(X_test), proba)
```

Step by step:

- **Step 1**: LightGBM reads pandas `category` columns directly. Category-to-code mappings are stored in the model (`pandas_categorical`), so the same categories at predict time are mapped consistently. Unseen categories at predict time are treated like missing.
- **Step 3**: `reference=dtrain` makes the validation set reuse the training bin boundaries; omitting it is a common source of subtle bugs.
- **Step 4**: early stopping via callbacks is the LightGBM 4.x way. `first_metric_only=True` makes AUC (first in the `metric` list) the stopping criterion; the training set is ignored for early stopping automatically.
- **Step 6**: `pred_contrib=True` returns TreeSHAP values; the last column is the expected value (bias).
- **Step 7**: text model files are human-readable and portable across platforms.

### Tutorial 2: Regression with cross-validation, monotone constraints and quantile intervals

Goal: house-price regression with `lgb.cv` to choose the number of rounds, a monotone constraint on income, and 80% prediction intervals from quantile models.

```python
import numpy as np
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mono = [1 if c == "MedInc" else 0 for c in X.columns]       # +1: increasing in MedInc
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "monotone_constraints": mono,
    "monotone_constraints_method": "intermediate",
    "seed": 0,
    "verbose": -1,
}

# 1. CV picks the number of rounds (regression: stratified=False, explicit KFold)
dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
cv = lgb.cv(params, dtrain, num_boost_round=5000,
            folds=KFold(n_splits=5, shuffle=True, random_state=0),
            callbacks=[lgb.early_stopping(100, verbose=False)])
n_rounds = len(cv["valid rmse-mean"])
print("rounds:", n_rounds, "CV RMSE:", round(cv["valid rmse-mean"][-1], 4),
      "+/-", round(cv["valid rmse-stdv"][-1], 4))

# 2. Refit on all training data with that many rounds
model = lgb.train(params, dtrain, num_boost_round=n_rounds)
pred = model.predict(X_test)
print("test RMSE:", round(root_mean_squared_error(y_test, pred), 4),
      "MAE:", round(mean_absolute_error(y_test, pred), 4))

# 3. Verify the monotone constraint
probe = X_test.iloc[:200]
low, high = model.predict(probe.assign(MedInc=2.0)), model.predict(probe.assign(MedInc=8.0))
print("monotone for all rows:", bool(np.all(high >= low)))

# 4. Quantile models for an 80% prediction interval (one model per quantile)
quantiles = {}
for a in (0.1, 0.5, 0.9):
    qp = {**params, "objective": "quantile", "alpha": a, "metric": "quantile"}
    qp.pop("monotone_constraints"); qp.pop("monotone_constraints_method")
    quantiles[a] = lgb.train(qp, dtrain, num_boost_round=n_rounds).predict(X_test)
coverage = np.mean((y_test.values >= quantiles[0.1]) & (y_test.values <= quantiles[0.9]))
print("empirical coverage of 80% interval:", round(coverage, 3))
```

Notes: `lgb.cv` defaults to `stratified=True`, which fails for continuous targets, so pass `folds` (or `stratified=False`). The result keys carry the `"valid "` prefix in 4.x. `monotone_constraints_method="intermediate"` is less restrictive than `"basic"` and usually more accurate. LightGBM trains one model per quantile.

### Tutorial 3: Learning to rank with LGBMRanker

Goal: rank documents per query with LambdaRank and evaluate NDCG@10.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score

rng = np.random.default_rng(0)
n_queries, docs_per_query, n_features = 300, 30, 12
qid = np.repeat(np.arange(n_queries), docs_per_query)
X = rng.normal(size=(len(qid), n_features))
signal = 1.5 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + rng.normal(scale=0.5, size=len(qid))
y = np.clip(np.digitize(signal, [-1, 0.5, 1.5, 2.5]), 0, 4)   # graded relevance 0..4

train_mask = qid < 240                                            # split by query
X_tr, y_tr, q_tr = X[train_mask], y[train_mask], qid[train_mask]
X_te, y_te, q_te = X[~train_mask], y[~train_mask], qid[~train_mask]

def group_sizes(q):
    # rows are sorted by qid; group = number of rows per consecutive query
    _, counts = np.unique(q, return_counts=True)
    return counts

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    lambdarank_truncation_level=20,
    verbose=-1,
)
ranker.fit(
    X_tr, y_tr, group=group_sizes(q_tr),
    eval_set=[(X_te, y_te)], eval_group=[group_sizes(q_te)], eval_at=[5, 10],
    callbacks=[lgb.early_stopping(50, first_metric_only=False, verbose=False), lgb.log_evaluation(100)],
)
print("best iteration:", ranker.best_iteration_, dict(ranker.best_score_["valid_0"]))

scores = ranker.predict(X_te)
df = pd.DataFrame({"qid": q_te, "y": y_te, "score": scores})
ndcg10 = np.mean([ndcg_score([g["y"].to_numpy()], [g["score"].to_numpy()], k=10)
                  for _, g in df.groupby("qid")])
print("mean NDCG@10:", round(float(ndcg10), 4))
```

Key points: `group` is a list of group sizes, not query ids, and rows must be contiguous per query. Relevance labels must be non-negative integers no larger than the length of `label_gain` minus one (31 by default). Use `objective="rank_xendcg"` for a faster alternative.

### Tutorial 4: Hyperparameter tuning with Optuna and pruning

Goal: tune a multiclass model efficiently with Optuna, using early stopping inside each trial and pruning bad trials.

```python
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

X, y = make_classification(n_samples=60_000, n_features=40, n_informative=15, n_classes=4,
                           n_clusters_per_class=2, random_state=0)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=0)

dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False, params={"feature_pre_filter": False})
dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

def objective(trial):
    params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "verbose": -1,
        "seed": 0,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    }

    def prune(env):
        # report the validation loss every 20 iterations so Optuna can stop bad trials early
        if env.iteration % 20 == 0:
            value = env.evaluation_result_list[0][2]
            trial.report(value, env.iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()

    booster = lgb.train(params, dtrain, num_boost_round=5000, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(100, verbose=False), prune])
    trial.set_user_attr("n_rounds", booster.best_iteration)
    return booster.best_score["valid_0"]["multi_logloss"]

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=0),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=100))
study.optimize(objective, n_trials=40)
print("best valid logloss:", round(study.best_value, 4))
print(study.best_params)

# Final model with the best params and tuned rounds
best = {**study.best_params, "objective": "multiclass", "num_class": 4, "bagging_freq": 1,
        "verbose": -1, "seed": 0}
final = lgb.train(best, dtrain, num_boost_round=study.best_trial.user_attrs["n_rounds"])
proba = final.predict(X_test)
print("test logloss:", round(log_loss(y_test, proba), 4),
      "accuracy:", round(accuracy_score(y_test, proba.argmax(axis=1)), 4))
```

Why it is built this way: `feature_pre_filter=False` is required because `min_data_in_leaf` changes between trials while the `Dataset` is reused (otherwise LightGBM raises an error about changing `min_data_in_leaf` after construction). Early stopping removes `n_estimators` from the search space, and the custom callback lets the median pruner kill unpromising trials early. The `optuna-integration` package also ships a ready-made `LightGBMPruningCallback` and an `optuna.integration.lightgbm.LightGBMTuner` that tunes parameters stepwise.

### Tutorial 5: LGBMRegressor in a scikit-learn pipeline with a custom objective and metric

Goal: use LightGBM inside sklearn tooling, and show how to write a custom loss for the sklearn API. The example is an asymmetric squared error where under-prediction costs three times more than over-prediction (useful for inventory or capacity planning).

```python
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

X, y = make_regression(n_samples=20_000, n_features=20, noise=10, random_state=0)
y[::50] += 500                                   # inject a few outliers

UNDER_WEIGHT = 3.0

def asymmetric_l2(y_true, y_pred):
    # LightGBM inspects the signature: (y_true, y_pred[, weight[, group]]), so no extra args
    r = y_pred - y_true
    w = np.where(r < 0, UNDER_WEIGHT, 1.0)        # r < 0 means under-prediction
    grad = 2.0 * w * r
    hess = 2.0 * w
    return grad, hess

def under_rate(y_true, y_pred):
    return "under_rate", float(np.mean(y_pred < y_true)), False   # (name, value, higher_is_better)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)
model = lgb.LGBMRegressor(objective=asymmetric_l2, n_estimators=3000, learning_rate=0.05,
                          num_leaves=31, verbose=-1)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric=["l2", under_rate],
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
pred = model.predict(X_va)
print("best iteration:", model.best_iteration_,
      "share of under-predictions:", round(float(np.mean(pred < y_va)), 3))

# A built-in robust objective in a pipeline with sklearn cross-validation
pipe = make_pipeline(FunctionTransformer(np.tanh),   # any sklearn preprocessing step
                     lgb.LGBMRegressor(objective="huber", alpha=20.0, n_estimators=400,
                                       learning_rate=0.05, verbose=-1))
scores = cross_val_score(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0),
                         scoring="neg_mean_absolute_error", n_jobs=1)
print("CV MAE:", round(-scores.mean(), 3))
```

Notes: with a custom objective LightGBM starts from a raw score of 0 (no `boost_from_average`) and `predict` returns raw scores; for losses with a link function (for example a custom logistic loss) apply the inverse link yourself. With a custom objective, the built-in metric named in `eval_metric` (`"l2"`) is still computed, and `first_metric_only=True` makes early stopping use it. Prefer built-in objectives (`"huber"`, `"fair"`, `"quantile"`) when they fit the problem.

## Performance & Best Practices

### Hyperparameter tuning strategy

1. **Set up early stopping** on a validation set (or `lgb.cv`) and fix a moderately high learning rate (0.05 to 0.1) for search speed.
2. **Tree structure first**: `num_leaves` (15 to 255, log scale), `min_data_in_leaf` (10 to 500, log scale), optionally `max_depth` (-1 or 5 to 12). These dominate.
3. **Sampling**: `feature_fraction` (0.4 to 1.0), `bagging_fraction` (0.5 to 1.0) with `bagging_freq=1`.
4. **Regularisation**: `lambda_l1`, `lambda_l2` (1e-8 to 10, log scale), `min_gain_to_split` (0 to 1), `path_smooth` (0 to 10).
5. **Categoricals**: `cat_smooth` (1 to 100), `min_data_per_group`, `max_cat_threshold`.
6. **Final model**: lower `learning_rate` to 0.01 to 0.03 and rely on early stopping for the new number of rounds.
7. Use **Optuna (TPE)** or `LightGBMTuner` rather than grid search; 30 to 100 trials is usually enough.

Official tuning guidance in one table:

| Goal | Parameters |
|---|---|
| Better accuracy | Larger `max_bin`, smaller `learning_rate` with more rounds, larger `num_leaves`, more data, `boosting="dart"`. |
| Faster training | `data_sample_strategy="goss"` or bagging, `feature_fraction` < 1, smaller `max_bin`, `save_binary` for reuse, `force_col_wise=True`, `use_quantized_grad=True`, set `num_threads` to physical cores. |
| Less overfitting | Smaller `num_leaves`, set `max_depth`, larger `min_data_in_leaf` and `min_sum_hessian_in_leaf`, `lambda_l1` / `lambda_l2`, `min_gain_to_split`, `feature_fraction` and bagging, `extra_trees=True`, `path_smooth`, smaller `max_bin`, more data. |

### Speed and memory

- Construct a `Dataset` once and reuse it across experiments; `save_binary` makes reloading near-instant.
- Pass `free_raw_data=False` only when you need to reuse the raw data (it costs memory).
- Set `num_threads` to the number of **physical** cores; hyperthreads rarely help. When running several models in parallel, divide threads among them.
- Silence the col-wise/row-wise probe by setting `force_col_wise=True` (many features) or `force_row_wise=True` (many rows, few features).
- Keep sparse data in CSR/CSC format; EFB bundles sparse columns automatically.
- Use `float32` NumPy arrays to halve memory before Dataset construction.
- For distributed data, use `lightgbm.dask` or SynapseML on Spark.

### Correctness and reproducibility

- Always pass `reference=dtrain` for validation Datasets.
- Set `seed`, and for exact reproducibility on CPU also `deterministic=True` together with `force_col_wise=True` or `force_row_wise=True`.
- Keep column order, names and categorical dtypes identical at prediction time; pass `validate_features=True` to `predict` to check names.
- With `is_unbalance=True` or `scale_pos_weight`, predicted probabilities are no longer calibrated; calibrate afterwards if probabilities matter.
- Early stopping makes the validation score optimistic; report a separate test score.
- Save models with `save_model` (text) or `model_to_string`; pickling `LGBMClassifier` works but ties you to library versions.

## Common Errors & Troubleshooting

| Error / warning | Cause | Fix |
|---|---|---|
| `TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'` | Removed in 4.0. | `callbacks=[lgb.early_stopping(50)]` or `"early_stopping_round": 50` in params. |
| `TypeError: train() got an unexpected keyword argument 'verbose_eval'` | Removed in 4.0. | `callbacks=[lgb.log_evaluation(100)]`. |
| `TypeError: train() got an unexpected keyword argument 'evals_result'` | Removed in 4.0. | `callbacks=[lgb.record_evaluation(my_dict)]`. |
| `TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'` (or `'verbose'`) | Removed from sklearn `fit` in 4.0. | `fit(..., callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])`; silence logs with `verbose=-1` in the constructor. |
| `LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead.` | LightGBM 4.7+. | `fit(X, y, eval_X=(X_val,), eval_y=(y_val,))`, or ignore if you must support older 4.x. |
| `[LightGBM] [Warning] No further splits with positive gain, best gain: -inf` | Leaves cannot be split under the constraints (tiny data, high `min_data_in_leaf`), often harmless near the end of training. | Lower `min_data_in_leaf` / `min_sum_hessian_in_leaf` on small data, or silence with `verbose=-1`. |
| `[LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements` | Same as above at the start: model cannot grow. | Check target variance, constraints, and data size. |
| `ValueError: pandas dtypes must be int, float or bool. Fields with bad pandas dtypes: city: object` (wording varies by version) | String columns in a DataFrame. | `df["city"] = df["city"].astype("category")` or encode. |
| `LightGBMError: Do not support special JSON characters in feature name.` | Column names contain characters such as `[`, `]`, `{`, `"`, `:`, `,`. | Rename: `df.columns = df.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)`. |
| `[LightGBM] [Warning] Found whitespace in feature_names, replace with underlines` | Spaces in column names. | Harmless; rename columns to avoid confusion. |
| `LightGBMError: Reducing min_data_in_leaf with feature_pre_filter=true may cause unexpected behaviour for features that were pre-filtered by the larger min_data_in_leaf.` | Reusing a Dataset while tuning `min_data_in_leaf`. | Create the Dataset with `params={"feature_pre_filter": False}`. |
| `LightGBMError: GPU Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_GPU=1` | `device_type="gpu"` on a CPU-only wheel. | Build with `--config-settings=cmake.define.USE_GPU=ON`, or use CPU. |
| `LightGBMError: CUDA Tree Learner was not enabled in this build.` | `device_type="cuda"` on a non-CUDA build. | Build with `cmake.define.USE_CUDA=ON`. |
| `OSError: ... Library not loaded: @rpath/libomp.dylib` (macOS) / `libgomp.so.1: cannot open shared object file` (Linux) | OpenMP runtime missing. | `brew install libomp` / `apt install libgomp1`. |
| `LightGBMError: Number of classes should be specified and greater than 1 for multiclass training` | `objective="multiclass"` without `num_class` in the native API. | Add `"num_class": K`. |
| `LightGBMError: Label must be in [0, num_class), but found ...` | Multiclass labels not encoded 0..K-1. | `LabelEncoder` the target. |
| `ValueError: ... stratified ... ` / errors from `lgb.cv` on regression | `stratified=True` default with a continuous target. | `lgb.cv(..., stratified=False)` or pass `folds=KFold(...)`. |
| `LightGBMError: The number of features in data (N) is not the same as it was in training data (M).` | Feature mismatch at predict time. | Use the same columns; `predict_disable_shape_check=True` only if you know what you are doing. |
| `UserWarning: categorical_feature in Dataset is overridden.` | Categorical features specified in two places. | Specify them once (Dataset or `fit`). |
| `[LightGBM] [Warning] min_data_in_leaf is set=40, min_child_samples=20 will be ignored.` | Parameter given under two aliases. | Use one name per parameter. |
| `[LightGBM] [Warning] bagging_fraction is set=0.8 ... ` but no effect | `bagging_freq` left at 0. | Set `bagging_freq=1` (`subsample_freq=1`). |
| Model overfits badly on small data | Leaf-wise growth with default `num_leaves=31` and `min_data_in_leaf=20`. | Lower `num_leaves`, raise `min_data_in_leaf`, set `max_depth`, add `lambda_l2`. |
| `ValueError: Input contains NaN` from sklearn wrappers around LightGBM | The error comes from another pipeline step; LightGBM itself accepts NaN. | Fix the upstream transformer or remove it. |

## Interoperability

### scikit-learn

`LGBMClassifier`, `LGBMRegressor` and `LGBMRanker` implement the sklearn estimator API: `Pipeline`, `ColumnTransformer`, `GridSearchCV`, `RandomizedSearchCV`, `cross_validate`, `StackingClassifier`, `CalibratedClassifierCV`, `permutation_importance`, `PartialDependenceDisplay` all work. Pass `verbose=-1` to keep logs quiet during searches, and avoid double parallelism (`n_jobs=-1` on both LightGBM and the search).

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform
import lightgbm as lgb

search = RandomizedSearchCV(
    lgb.LGBMClassifier(n_estimators=300, verbose=-1, n_jobs=4),
    {"num_leaves": randint(15, 128), "learning_rate": loguniform(0.02, 0.2),
     "min_child_samples": randint(5, 100), "colsample_bytree": [0.6, 0.8, 1.0]},
    n_iter=20, cv=5, scoring="roc_auc", random_state=0, n_jobs=1,
)
```

### pandas, Polars and Arrow

pandas DataFrames are first-class (column names and `category` dtype are respected). PyArrow tables and arrays are accepted by `Dataset` in recent 4.x releases, and Polars DataFrames can be passed via `.to_pandas()` / `.to_arrow()` (newer releases accept them directly).

### SHAP

`booster.predict(X, pred_contrib=True)` gives exact TreeSHAP values. The `shap` package (`shap.TreeExplainer(model)`) supports LightGBM boosters and sklearn wrappers for plots such as beeswarm and dependence plots.

### Other boosting libraries

| Concept | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| Tree size | `num_leaves` | `max_depth` (or `max_leaves` with lossguide) | `depth` (symmetric trees) |
| Min leaf size | `min_data_in_leaf` | `min_child_weight` (Hessian) | `min_data_in_leaf` |
| Row sampling | `bagging_fraction` + `bagging_freq` | `subsample` | `subsample` |
| Column sampling | `feature_fraction` | `colsample_bytree` | `rsm` |
| L2 | `lambda_l2` | `lambda` | `l2_leaf_reg` |
| Early stopping | `lgb.early_stopping` callback | `early_stopping_rounds` | `early_stopping_rounds` |
| Categoricals | pandas `category` / `categorical_feature` | `enable_categorical=True` | `cat_features` |

### Distributed and MLOps

- **Dask**: `lightgbm.dask.DaskLGBMClassifier` / `Regressor` / `Ranker`.
- **Spark**: SynapseML `LightGBMClassifier` / `LightGBMRegressor` / `LightGBMRanker`.
- **Ray**: `lightgbm_ray` / Ray Train `LightGBMTrainer`.
- **MLflow**: `mlflow.lightgbm.autolog()` and `mlflow.lightgbm.log_model(booster, "model")`.
- **ONNX**: `onnxmltools.convert_lightgbm(model, initial_types=...)` then `onnxruntime`.
- **Treelite / NVIDIA Triton FIL**: compile or serve LightGBM text models for low-latency inference.
- **Optuna**: `optuna-integration` provides `LightGBMPruningCallback` and `LightGBMTuner`.

## Cheat Sheet

### Data

| Task | Code |
|---|---|
| Install | `pip install lightgbm` |
| Version | `lgb.__version__` |
| Training Dataset | `dtrain = lgb.Dataset(X, label=y)` |
| Validation Dataset | `dvalid = lgb.Dataset(Xv, label=yv, reference=dtrain)` |
| Categorical columns | `df[c] = df[c].astype("category")` or `lgb.Dataset(X, y, categorical_feature=["c"])` |
| Weights / offsets | `lgb.Dataset(X, y, weight=w, init_score=offset)` |
| Ranking groups | `lgb.Dataset(X, y, group=sizes)` |
| Save / load binary Dataset | `dtrain.save_binary("t.bin")`, `lgb.Dataset("t.bin")` |
| Reuse Dataset while tuning leaf size | `lgb.Dataset(X, y, params={"feature_pre_filter": False})` |
| Silence logs | `"verbose": -1` / `verbose=-1` |

### Training

| Task | Code |
|---|---|
| Native train | `lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid])` |
| Early stopping | `callbacks=[lgb.early_stopping(100)]` |
| Log every N rounds | `callbacks=[lgb.log_evaluation(100)]` |
| Record history | `h = {}; callbacks=[lgb.record_evaluation(h)]` |
| LR schedule | `callbacks=[lgb.reset_parameter(learning_rate=lambda i: 0.1 * 0.99 ** i)]` |
| sklearn classifier | `lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)` |
| sklearn early stopping | `clf.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100)])` |
| Cross-validation | `lgb.cv(params, dtrain, 5000, nfold=5, callbacks=[lgb.early_stopping(100)])` |
| Binary imbalanced | `"is_unbalance": True` or `"scale_pos_weight": neg / pos` |
| Multiclass | `{"objective": "multiclass", "num_class": K}` |
| Quantile | `{"objective": "quantile", "alpha": 0.9}` |
| Poisson / Tweedie | `{"objective": "poisson"}` / `{"objective": "tweedie", "tweedie_variance_power": 1.3}` |
| Ranking | `lgb.LGBMRanker(objective="lambdarank").fit(X, y, group=sizes)` |
| GOSS | `"data_sample_strategy": "goss"` |
| Bagging | `"bagging_fraction": 0.8, "bagging_freq": 1` |
| DART | `"boosting": "dart", "drop_rate": 0.1` |
| Random forest | `"boosting": "rf", "bagging_fraction": 0.63, "bagging_freq": 1, "feature_fraction": 0.8` |
| Linear trees | `"linear_tree": True` |
| Monotone constraint | `"monotone_constraints": [1, 0, -1]` |
| GPU | `"device_type": "gpu"` or `"cuda"` (GPU build required) |
| Continue training | `lgb.train(params, dtrain, 100, init_model=booster)` |

### Prediction and inspection

| Task | Code |
|---|---|
| Predict (best iteration) | `booster.predict(X)` |
| Raw scores | `booster.predict(X, raw_score=True)` |
| SHAP values | `booster.predict(X, pred_contrib=True)` |
| Leaf indices | `booster.predict(X, pred_leaf=True)` |
| First N trees only | `booster.predict(X, num_iteration=N)` |
| Importance | `booster.feature_importance("gain")`, `clf.feature_importances_` |
| Plot importance | `lgb.plot_importance(booster, importance_type="gain", max_num_features=20)` |
| Plot learning curve | `lgb.plot_metric(evals_dict_or_model)` |
| Trees as DataFrame | `booster.trees_to_dataframe()` |
| Best iteration | `booster.best_iteration` / `clf.best_iteration_` |
| Underlying booster | `clf.booster_` |

### Persistence

| Task | Code |
|---|---|
| Save text model | `booster.save_model("model.txt")` / `clf.booster_.save_model("model.txt")` |
| Load | `lgb.Booster(model_file="model.txt")` |
| To / from string | `s = booster.model_to_string()`; `lgb.Booster(model_str=s)` |
| JSON dump | `booster.dump_model()` |

### Migration from 3.x

| 3.x | 4.x |
|---|---|
| `lgb.train(..., early_stopping_rounds=50)` | `callbacks=[lgb.early_stopping(50)]` |
| `lgb.train(..., verbose_eval=100)` | `callbacks=[lgb.log_evaluation(100)]` |
| `lgb.train(..., evals_result=d)` | `callbacks=[lgb.record_evaluation(d)]` |
| `clf.fit(..., early_stopping_rounds=50, verbose=False)` | `clf.fit(..., callbacks=[lgb.early_stopping(50, verbose=False)])` |
| `"boosting": "goss"` | `"data_sample_strategy": "goss"` |
| cv keys `"auc-mean"` | `"valid auc-mean"` |
| `pip install lightgbm --install-option=--gpu` | `pip install --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON lightgbm` |

## Further Resources

- Official documentation: https://lightgbm.readthedocs.io/en/stable/
- Parameters reference: https://lightgbm.readthedocs.io/en/stable/Parameters.html
- Parameters tuning guide: https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
- Python API: https://lightgbm.readthedocs.io/en/stable/Python-API.html
- Features (leaf-wise growth, GOSS, EFB, categorical splits): https://lightgbm.readthedocs.io/en/stable/Features.html
- Installation guide (GPU, CUDA builds): https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html
- GPU tutorial: https://lightgbm.readthedocs.io/en/stable/GPU-Tutorial.html
- GitHub repository: https://github.com/microsoft/LightGBM
- Release notes: https://github.com/microsoft/LightGBM/releases
- Python examples: https://github.com/microsoft/LightGBM/tree/master/examples/python-guide
- Paper: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- Paper: Meng et al., "A Communication-Efficient Parallel Algorithm for Decision Tree", NeurIPS 2016 (voting parallel): https://arxiv.org/abs/1611.01276
- SynapseML (LightGBM on Spark): https://microsoft.github.io/SynapseML/
