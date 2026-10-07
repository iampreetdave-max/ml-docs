# pandas

> Fast, flexible, labeled data structures for tabular and time-series data in Python.

pandas gives Python two core data structures, `Series` (a labeled 1-D array) and `DataFrame` (a labeled 2-D table of columns with possibly different types), plus a large toolkit for reading, cleaning, reshaping, joining, aggregating, and analyzing data. It is the standard tool for the "data wrangling" phase of almost every machine learning project.

Covers pandas 3.0 (examples verified against pandas 3.0.0). Behavior changes from pandas 2.x are called out explicitly; most examples also run on pandas 2.2.

## Overview

### What pandas is

- **`Series`**: a one-dimensional array of values with an `Index` of labels and a single dtype.
- **`DataFrame`**: a dictionary-like collection of `Series` sharing one row `Index`, each column with its own dtype.
- **`Index`**: immutable, hashable labels for rows and columns, with specialized subclasses (`RangeIndex`, `DatetimeIndex`, `MultiIndex`, `CategoricalIndex`, `IntervalIndex`).
- **I/O**: readers and writers for CSV, Excel, Parquet, Feather, JSON, SQL, HTML, HDF5, pickle, and more.
- **Split-apply-combine** (`groupby`), relational joins (`merge`), reshaping (`pivot`, `melt`, `stack`), and rich time-series support (`resample`, `rolling`, time zones, offsets).

### History and maintainers

- 2008: Wes McKinney starts pandas at AQR Capital for financial data analysis; open-sourced in 2009.
- 2012: First edition of *Python for Data Analysis* popularizes it.
- 2020: pandas 1.0.
- April 2023: pandas 2.0 adds optional PyArrow-backed dtypes and non-nanosecond datetimes.
- pandas 3.0: Copy-on-Write is always on, a dedicated string dtype (`str`) is the default for text, and many 2.x deprecations are enforced.
- pandas is a NumFOCUS sponsored project maintained by a core team of volunteers and funded contributors, BSD-3-Clause licensed.

### When to use pandas

- Tabular data that fits in memory (rule of thumb: up to a few GB, ideally under a third of RAM).
- Exploratory data analysis, feature engineering, joining multiple sources, cleaning messy data.
- Time-series with calendars, resampling, rolling windows, and time zones.
- Preparing feature matrices for scikit-learn, XGBoost, LightGBM, statsmodels.

### When not to use pandas

| Situation | Better tool |
|-----------|-------------|
| Data much larger than RAM | Polars (lazy/streaming), DuckDB, Dask, PySpark |
| Heavy multi-core speed on large tables | Polars, DuckDB |
| Pure numerical arrays / tensors | NumPy, PyTorch, JAX |
| N-dimensional labeled arrays | xarray |
| Transactional, concurrent writes | A database (PostgreSQL, SQLite) |

### Where it fits in the ML stack

```text
  raw files / databases / APIs
            |
        pandas  (load, clean, join, feature engineering, EDA)
            |
   .to_numpy() / DataFrame passed directly
            |
  scikit-learn / XGBoost / statsmodels / PyTorch Dataset
            |
  Matplotlib / seaborn / plotly (DataFrame.plot uses Matplotlib)
```

pandas is built on NumPy (and optionally PyArrow) and is consumed directly by scikit-learn, seaborn, statsmodels, and most gradient-boosting libraries.

## Installation

### pip

```bash
pip install pandas
# common optional extras
pip install "pandas[parquet,excel,performance,plot]"
# everything
pip install "pandas[all]"
```

Selected extras:

| Extra | Installs | Enables |
|-------|----------|---------|
| `pyarrow` / `parquet` / `feather` | pyarrow | Parquet/Feather I/O, Arrow-backed dtypes, faster CSV engine |
| `excel` | openpyxl, xlrd, odfpy, pyxlsb, python-calamine, xlsxwriter | `read_excel` / `to_excel` |
| `performance` | numexpr, bottleneck, numba | Faster `eval`/`query`, nan-reductions, numba engines |
| `plot` | matplotlib | `DataFrame.plot` |
| `postgresql`, `mysql`, `sql-other` | SQLAlchemy and drivers | `read_sql` / `to_sql` |
| `fss`, `aws`, `gcp` | fsspec, s3fs, gcsfs | Reading from `s3://`, `gs://` URLs |
| `compression` | zstandard | Zstandard compression |

### conda

```bash
conda install -c conda-forge pandas pyarrow
```

### GPU variant

pandas runs on the CPU. NVIDIA's RAPIDS **cuDF** provides a GPU DataFrame and a zero-code-change accelerator mode for pandas:

```bash
pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
python -m cudf.pandas my_script.py
```

### Verifying the install

```python
import pandas as pd

print(pd.__version__)     # e.g. 3.0.0
pd.show_versions()        # versions of pandas, NumPy, PyArrow and all optional deps
```

Requirements for pandas 3.0: Python 3.11+, NumPy 1.26+, python-dateutil. PyArrow is optional but strongly recommended (it backs the default string dtype when installed).

## Core Concepts

### Series and DataFrame

```python
import numpy as np
import pandas as pd

s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="score")
print(s["b"])          # 20
print(s.index)         # Index(['a', 'b', 'c'], dtype='str')
print(s.dtype)         # int64

df = pd.DataFrame({
    "name": ["Ada", "Ben", "Cy"],
    "age": [36, 41, 29],
    "salary": [120_000.0, 95_000.0, np.nan],
})
print(df.shape)        # (3, 3)
print(df.dtypes)
# name          str
# age         int64
# salary    float64
print(df.columns.tolist())   # ['name', 'age', 'salary']
print(df.index)              # RangeIndex(start=0, stop=3, step=1)
```

A DataFrame column is a Series: `df["age"]` returns a Series named `"age"` sharing the DataFrame's index.

### The Index and alignment

Operations between pandas objects **align on labels**, not positions. Labels missing on either side produce `NaN`.

```python
a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20, 30], index=["y", "z", "w"])
print(a + b)
# w     NaN
# x     NaN
# y    12.0
# z    23.0
print(a.add(b, fill_value=0))   # treat missing as 0
```

This is the most common source of surprise for NumPy users. Use `.to_numpy()` or `reset_index(drop=True)` when you deliberately want positional arithmetic.

### dtypes

| dtype | Example | Notes |
|-------|---------|-------|
| `int64`, `float64`, `bool` | NumPy-backed | Fast; `int64`/`bool` cannot hold NaN. |
| `Int64`, `Float64`, `boolean` | Nullable extension types | Use `pd.NA` for missing; note the capital letter. |
| `str` | Default text dtype in 3.0 | `StringDtype` backed by PyArrow (or Python if PyArrow is missing); missing value is `NaN`. In 2.x text was `object`. |
| `string` | `pd.StringDtype()` | Opt-in string dtype using `pd.NA` as missing value. |
| `category` | `pd.CategoricalDtype` | Few unique values; saves memory, speeds up groupby. |
| `datetime64[ns]`, `datetime64[us]` | Timestamps | 3.0 infers resolution; strings parse to microseconds (`us`) by default. |
| `datetime64[us, UTC]` | tz-aware | Timezone-aware timestamps. |
| `timedelta64[ns]` | Durations | |
| `object` | Arbitrary Python objects | Slow; avoid. |
| `int64[pyarrow]` etc. | `pd.ArrowDtype` | PyArrow-backed; select with `dtype_backend="pyarrow"`. |

```python
df = pd.DataFrame({"n": [1, None, 3]})
print(df["n"].dtype)                  # float64 (NaN forces float)
print(df["n"].astype("Int64"))        # 1, <NA>, 3 with dtype Int64

df = pd.DataFrame({"city": ["NY", "SF", "NY"]})
df["city"] = df["city"].astype("category")
print(df["city"].cat.categories)      # Index(['NY', 'SF'], dtype='str')
```

### Missing data

pandas uses `NaN` (float), `NaT` (datetime), and `pd.NA` (nullable dtypes) to represent missing values. `None` is converted to one of these.

```python
s = pd.Series([1.0, None, 3.0])
s.isna()        # [False, True, False]
s.sum()         # 4.0  -- reductions skip NaN by default (skipna=True)
s.mean()        # 2.0
s.fillna(0)
s.dropna()
```

### Selection: [], .loc, .iloc

| Accessor | Selects by | Slices include end? |
|----------|------------|---------------------|
| `df["col"]`, `df[["a", "b"]]` | Column label(s) | n/a |
| `df[bool_series]` | Boolean row mask | n/a |
| `df.loc[rows, cols]` | **Labels** (or boolean arrays) | Yes, label slices are inclusive |
| `df.iloc[rows, cols]` | **Integer positions** | No, like Python |
| `df.at[row, col]` / `df.iat[i, j]` | Single scalar, fast | n/a |

```python
df = pd.DataFrame(
    {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]},
    index=["w", "x", "y", "z"],
)
df.loc["x", "b"]            # 20
df.loc["x":"y", ["a"]]      # rows x and y (inclusive)
df.iloc[0:2, 1]             # first two rows of column 1
df.loc[df["a"] > 2, "b"]    # boolean mask + column
df.at["z", "a"]             # 4
```

### Copy-on-Write (always on in pandas 3.0)

In pandas 3.0, every indexing result and every method result **behaves as a copy**. Modifying a derived object never modifies the parent. Internally pandas shares memory lazily and copies only when you write.

Consequences:

```python
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

# 1. Chained assignment never works (and raises ChainedAssignmentError warning)
df["a"][df["b"] > 4] = 100        # does NOT modify df

# 2. Do it in one step with .loc
df.loc[df["b"] > 4, "a"] = 100     # modifies df

# 3. A subset is independent
sub = df[df["a"] > 1]
sub["b"] = 0                       # fine, no SettingWithCopyWarning, df unchanged
```

`SettingWithCopyWarning` no longer exists in 3.0. Setting `pd.options.mode.copy_on_write` has no effect (it is deprecated). In pandas 2.x you can opt in with `pd.options.mode.copy_on_write = True` to prepare.

### Vectorized operations and method chaining

Like NumPy, pandas is fast when you operate on whole columns. Methods return new objects, so they can be chained:

```python
df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "qty": [1, 0, 5]})
result = (
    df
    .assign(revenue=lambda d: d["price"] * d["qty"])
    .query("qty > 0")
    .sort_values("revenue", ascending=False)
    .reset_index(drop=True)
)
print(result)
#    price  qty  revenue
# 0   30.0    5    150.0
# 1   10.0    1     10.0
```

### Split-apply-combine

`groupby` splits rows into groups by key, applies a function to each group, and combines the results.

```python
sales = pd.DataFrame({
    "store": ["A", "A", "B", "B", "B"],
    "units": [3, 5, 2, 8, 1],
})
sales.groupby("store")["units"].sum()
# store
# A     8
# B    11
```

### Notable deprecations and removals (2.x to 3.0)

| Removed / deprecated | Use instead |
|----------------------|-------------|
| `df.append(other)` (removed 2.0) | `pd.concat([df, other])` |
| `df.iteritems()` (removed 2.0) | `df.items()` |
| `df.applymap(f)` (deprecated 2.1, removed 3.0) | `df.map(f)` |
| `fillna(method="ffill")` (removed 3.0) | `ffill()`, `bfill()` |
| `df.swapaxes`, `df.bool()`, `df.first("3D")`, `df.last(...)` (removed 3.0) | `transpose`, `.item()`, `.loc` with date ranges |
| Frequency aliases `"M"`, `"Q"`, `"Y"` | `"ME"`, `"QE"`, `"YE"` (month/quarter/year end); `"MS"` etc. unchanged |
| `"H"`, `"T"`, `"S"`, `"L"`, `"U"` | `"h"`, `"min"`, `"s"`, `"ms"`, `"us"` |
| `groupby(..., axis=1)` | Transpose first |
| `observed=False` default in categorical groupby | Default is now `observed=True` |
| `copy=` keyword on many methods | Ignored under Copy-on-Write; deprecated |
| `inplace=True` | Still works for many methods but discouraged; prefer reassignment |
| Text columns as `object` | Default `str` dtype in 3.0 |
| `pytz` required | Optional in 3.0; `zoneinfo` used by default |

## API Reference

All examples assume:

```python
import numpy as np
import pandas as pd
```

### Constructors

#### pd.Series

```python
pd.Series(data=None, index=None, dtype=None, name=None, copy=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array-like, dict, scalar | `None` | Values. A dict's keys become the index. |
| `index` | array-like or Index | `None` | Labels; defaults to `RangeIndex(0..n-1)`. |
| `dtype` | str, NumPy or extension dtype | `None` | Inferred if `None`. |
| `name` | hashable | `None` | Name of the Series (becomes the column name in a DataFrame). |

```python
pd.Series({"a": 1, "b": 2})
pd.Series(5, index=["x", "y"])          # broadcast scalar
pd.Series([1, 2, None], dtype="Int64")  # nullable integer
```

#### pd.DataFrame

```python
pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | dict, list of dicts, 2-D ndarray, DataFrame | `None` | Source data. |
| `index` | Index or array-like | `None` | Row labels. |
| `columns` | Index or array-like | `None` | Column labels (selects/orders keys of a dict). |
| `dtype` | dtype | `None` | Force a single dtype. |

```python
pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
pd.DataFrame([{"a": 1, "b": 2}, {"a": 3}])             # missing key -> NaN
pd.DataFrame(np.arange(6).reshape(3, 2), columns=["c1", "c2"])
pd.DataFrame.from_dict({"r1": {"a": 1}, "r2": {"a": 2}}, orient="index")
```

#### Index helpers

```python
pd.date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, inclusive='both', *, unit=None)
pd.RangeIndex(start=None, stop=None, step=None, dtype=None, copy=False, name=None)
pd.MultiIndex.from_product(iterables, sortorder=None, names=<no_default>)
pd.MultiIndex.from_tuples(tuples, sortorder=None, names=None)
```

```python
pd.date_range("2024-01-01", periods=3, freq="D")
pd.date_range("2024-01-01", "2024-06-30", freq="ME")   # month ends
pd.MultiIndex.from_product([["A", "B"], [2023, 2024]], names=["store", "year"])
```

### Input / output

#### pd.read_csv

```python
pd.read_csv(filepath_or_buffer, *, sep=',', header='infer', names=None, index_col=None, usecols=None, dtype=None, engine=None, converters=None, skiprows=None, nrows=None, na_values=None, keep_default_na=True, parse_dates=None, date_format=None, chunksize=None, compression='infer', thousands=None, decimal='.', encoding=None, on_bad_lines='error', low_memory=True, dtype_backend=<no_default>, ...)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath_or_buffer` | str, path, URL, file-like | required | Local path, `http(s)://`, `s3://`, or buffer. |
| `sep` | str | `','` | Delimiter; `None` sniffs it with the Python engine; `'\t'` for TSV. |
| `header` | int, list, None | `'infer'` | Row number(s) of column names; `None` if no header. |
| `names` | list | `None` | Column names to use. |
| `index_col` | int, str, list | `None` | Column(s) to use as the row index. |
| `usecols` | list or callable | `None` | Subset of columns to read (saves memory). |
| `dtype` | dtype or dict | `None` | Per-column dtypes, e.g. `{"zip": "str"}`. |
| `parse_dates` | list | `None` | Columns to parse as datetimes. |
| `date_format` | str or dict | `None` | Explicit format, e.g. `"%Y-%m-%d"` (faster, unambiguous). |
| `na_values` | scalar, list, dict | `None` | Extra strings to treat as NaN. |
| `nrows` | int | `None` | Number of rows to read. |
| `chunksize` | int | `None` | Return an iterator of DataFrames of this many rows. |
| `engine` | {'c','python','pyarrow'} | `None` (c) | `'pyarrow'` is multithreaded and fastest for large files. |
| `dtype_backend` | {'numpy_nullable','pyarrow'} | NumPy | Use nullable or Arrow-backed dtypes for all columns. |
| `on_bad_lines` | {'error','warn','skip'} or callable | `'error'` | Handling of rows with too many fields. |
| `encoding` | str | `None` (utf-8) | e.g. `'latin-1'` for legacy files. |

Returns: `DataFrame` (or `TextFileReader` if `chunksize`/`iterator` is set).

```python
from io import StringIO

csv = StringIO("date,store,sales\n2024-01-01,A,10\n2024-01-02,B,NA\n")
df = pd.read_csv(csv, parse_dates=["date"], dtype={"store": "category"})
print(df.dtypes)
# date     datetime64[us]
# store          category
# sales           float64

# process a huge file in chunks
# total = sum(chunk["sales"].sum() for chunk in pd.read_csv("big.csv", chunksize=100_000))
```

#### DataFrame.to_csv

```python
DataFrame.to_csv(path_or_buf=None, *, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, mode='w', encoding=None, compression='infer', date_format=None, ...)
```

```python
df.to_csv("out.csv", index=False)           # usually you do not want the RangeIndex
df.to_csv("out.csv.gz", index=False)        # compression inferred from extension
text = df.to_csv(index=False)               # returns a string when no path given
```

#### Parquet, Feather, Excel, JSON, SQL, pickle

```python
pd.read_parquet(path, engine='auto', columns=None, filters=None, dtype_backend=<no_default>, ...)
DataFrame.to_parquet(path=None, *, engine='auto', compression='snappy', index=None, partition_cols=None, ...)
pd.read_excel(io, sheet_name=0, *, header=0, usecols=None, dtype=None, engine=None, ...)
pd.read_json(path_or_buf, *, orient=None, lines=False, ...)
pd.read_sql(sql, con, index_col=None, params=None, parse_dates=None, chunksize=None, ...)
DataFrame.to_sql(name, con, *, schema=None, if_exists='fail', index=True, chunksize=None, method=None, ...)
```

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| Parquet | `pd.read_parquet` | `df.to_parquet` | Columnar, compressed, preserves dtypes. Best default for ML datasets. Needs pyarrow. |
| Feather | `pd.read_feather` | `df.to_feather` | Very fast local Arrow IPC format. |
| Excel | `pd.read_excel` | `df.to_excel` | Needs openpyxl (xlsx). `sheet_name=None` reads all sheets into a dict. |
| JSON | `pd.read_json` | `df.to_json` | `lines=True` for JSON Lines. |
| SQL | `pd.read_sql`, `read_sql_query` | `df.to_sql` | Needs SQLAlchemy engine or a `sqlite3` connection. |
| Pickle | `pd.read_pickle` | `df.to_pickle` | Python-only; never unpickle untrusted data. |
| HTML | `pd.read_html` | `df.to_html` | Needs lxml or bs4+html5lib. |
| Clipboard | `pd.read_clipboard` | `df.to_clipboard` | Handy for quick copy-paste. |

```python
import sqlite3

df = pd.DataFrame({"id": [1, 2], "score": [0.5, 0.9]})
df.to_parquet("scores.parquet", index=False)
back = pd.read_parquet("scores.parquet", columns=["score"])

con = sqlite3.connect(":memory:")
df.to_sql("scores", con, index=False)
pd.read_sql("SELECT * FROM scores WHERE score > ?", con, params=(0.6,))
#    id  score
# 0   2    0.9

df.to_json("scores.jsonl", orient="records", lines=True)
pd.read_json("scores.jsonl", lines=True)
```

### Inspection

```python
DataFrame.head(n=5); DataFrame.tail(n=5); DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None)
DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None)
DataFrame.describe(percentiles=None, include=None, exclude=None)
DataFrame.memory_usage(index=True, deep=False)
```

| Attribute / method | Returns |
|--------------------|---------|
| `df.shape`, `df.ndim`, `df.size` | Dimensions |
| `df.dtypes` | Series of column dtypes |
| `df.columns`, `df.index` | Index objects |
| `df.info()` | Prints dtypes, non-null counts, memory |
| `df.describe()` | Count, mean, std, min, quartiles, max for numeric columns; `include="all"` for every column |
| `df.nunique()` | Unique count per column |
| `s.value_counts(normalize=False, dropna=True)` | Frequency table |
| `s.unique()` | Unique values in order of appearance |
| `df.memory_usage(deep=True)` | Bytes per column including string contents |

```python
df = pd.DataFrame({"x": [1, 2, 2, 3], "c": ["a", "b", "b", "b"]})
df.describe()
df["c"].value_counts()
# c
# b    3
# a    1
# Name: count, dtype: int64
df["c"].value_counts(normalize=True)   # proportions, named "proportion"
df.sample(frac=0.5, random_state=0)
```

### Selection and filtering

#### loc, iloc, at, iat

```python
df.loc[row_indexer, col_indexer]
df.iloc[row_positions, col_positions]
```

Indexers accept a single label/position, a list, a slice, a boolean array, or a callable `lambda d: ...`.

```python
df = pd.DataFrame({"a": range(5), "b": list("vwxyz")})
df.loc[1:3, "b"]                  # labels 1..3 inclusive
df.iloc[1:3]                      # positions 1, 2
df.loc[lambda d: d["a"] % 2 == 0] # callable
df.iloc[-1]                       # last row as a Series
df.loc[df.index[-1], "a"] = 99    # set a single value
```

#### Boolean filtering, isin, between, query

```python
DataFrame.query(expr, *, parser='pandas', engine=None, local_dict=None, global_dict=None, resolvers=None, level=0, inplace=False)
Series.between(left, right, inclusive='both')
DataFrame.isin(values)
```

```python
people = pd.DataFrame({"name": ["Ada", "Ben", "Cy", "Di"],
                       "age": [36, 17, 29, 52],
                       "city": ["NY", "SF", "LA", "NY"]})
people[(people["age"] > 18) & (people["city"] == "NY")]   # parentheses required
people[people["city"].isin(["NY", "LA"])]
people[people["age"].between(20, 40)]
min_age = 30
people.query("age > @min_age and city != 'SF'")           # @ refers to Python variables
people[~people["name"].str.startswith("A")]               # negation
```

#### where, mask, nlargest, nsmallest

```python
s = pd.Series([5, -2, 7, -1])
s.where(s > 0, 0)        # keep where True, else 0 -> [5, 0, 7, 0]
s.mask(s > 0, 0)         # replace where True     -> [0, -2, 0, -1]
people.nlargest(2, "age")
people.nsmallest(1, "age")
```

### Adding, removing, and transforming columns

#### assign, insert, drop, rename

```python
DataFrame.assign(**kwargs)
DataFrame.drop(labels=None, *, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
DataFrame.rename(mapper=None, *, index=None, columns=None, axis=None, inplace=False, level=None, errors='ignore')
DataFrame.insert(loc, column, value, allow_duplicates=<no_default>)
```

```python
df = pd.DataFrame({"w_kg": [70, 80], "h_m": [1.75, 1.80]})
df["bmi"] = df["w_kg"] / df["h_m"] ** 2                 # direct assignment
df = df.assign(obese=lambda d: d["bmi"] > 30)            # chain-friendly
df = df.rename(columns={"w_kg": "weight", "h_m": "height"})
df = df.drop(columns=["obese"])
df.insert(0, "id", [101, 102])
df.columns = df.columns.str.upper()                      # vectorized rename
```

#### astype, to_numeric, to_datetime

```python
DataFrame.astype(dtype, copy=<no_default>, errors='raise')
pd.to_numeric(arg, errors='raise', downcast=None, dtype_backend=<no_default>)
pd.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=False, format=None, exact=<no_default>, unit=None, origin='unix', cache=True)
```

| Parameter | Description |
|-----------|-------------|
| `errors='coerce'` | Unparseable values become NaN / NaT instead of raising. (`errors='ignore'` was removed in 3.0.) |
| `downcast` | `'integer'`, `'signed'`, `'unsigned'`, `'float'` to shrink memory. |
| `format` | strftime format; `'ISO8601'` or `'mixed'` also accepted. |
| `utc` | Return tz-aware UTC timestamps. |
| `unit` | Unit for numeric epoch input, e.g. `'s'`, `'ms'`. |

```python
raw = pd.DataFrame({"n": ["1", "2", "x"], "d": ["2024-01-05", "2024-02-10", "bad"]})
raw["n"] = pd.to_numeric(raw["n"], errors="coerce")             # 1.0, 2.0, NaN
raw["d"] = pd.to_datetime(raw["d"], format="%Y-%m-%d", errors="coerce")
raw = raw.astype({"n": "Float64"})
pd.to_datetime(1_700_000_000, unit="s")                          # Timestamp('2023-11-14 22:13:20')
```

#### map, apply, replace, case_when

```python
Series.map(func=None, na_action=None, engine=None, **kwargs)
DataFrame.map(func, na_action=None, **kwargs)                # element-wise (formerly applymap)
DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), by_row='compat', engine=None, engine_kwargs=None, **kwargs)
DataFrame.replace(to_replace=None, value=<no_default>, *, inplace=False, regex=False)
Series.case_when(caselist)                                    # 2.2+
```

| Method | Operates on | Typical use |
|--------|-------------|-------------|
| `Series.map` | Each element | Lookup via dict/Series, simple function |
| `DataFrame.map` | Each element of all columns | Formatting every cell |
| `DataFrame.apply(axis=0)` | Each column | Column-wise custom reductions |
| `DataFrame.apply(axis=1)` | Each row | Row-wise logic (slow; vectorize if possible) |
| `replace` | Values | Recode sentinel values |

```python
s = pd.Series(["low", "high", "mid"])
s.map({"low": 0, "mid": 1, "high": 2})          # ordinal encoding
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
df.apply(lambda col: col.max() - col.min())      # per column
df.apply(lambda row: row["a"] * row["b"], axis=1)
df.map(lambda v: f"{v:.1f}")
pd.Series([-999, 3]).replace(-999, np.nan)
x = pd.Series([5, 15, 25])
x.case_when([(x < 10, "small"), (x < 20, "medium"), (pd.Series(True, index=x.index), "large")])
# ['small', 'medium', 'large']
```

#### Binning and encoding: cut, qcut, get_dummies, factorize

```python
pd.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True)
pd.qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise')
pd.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
pd.factorize(values, sort=False, use_na_sentinel=True, size_hint=None)
```

```python
ages = pd.Series([5, 17, 30, 65, 80])
pd.cut(ages, bins=[0, 18, 64, 120], labels=["child", "adult", "senior"])
pd.qcut(ages, q=2, labels=["young", "old"])     # equal-frequency bins

colors = pd.DataFrame({"color": ["red", "blue", "red"], "v": [1, 2, 3]})
pd.get_dummies(colors, columns=["color"], dtype=int)
#    v  color_blue  color_red
# 0  1           0          1
# 1  2           1          0
# 2  3           0          1
codes, uniques = pd.factorize(colors["color"])   # [0 1 0], Index(['red', 'blue'])
```

`get_dummies` returns `bool` columns by default; pass `dtype=int` or `float` for numeric models.

### Missing data

```python
DataFrame.isna(); DataFrame.notna()
DataFrame.dropna(*, axis=0, how=<no_default>, thresh=<no_default>, subset=None, inplace=False, ignore_index=False)
DataFrame.fillna(value, *, axis=None, inplace=False, limit=None)
DataFrame.ffill(*, axis=None, inplace=False, limit=None, limit_area=None)
DataFrame.bfill(*, axis=None, inplace=False, limit=None, limit_area=None)
DataFrame.interpolate(method='linear', *, axis=0, limit=None, inplace=False, limit_direction=None, limit_area=None, **kwargs)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `how` | `'any'` | Drop if any (`'any'`) or all (`'all'`) values are missing. |
| `thresh` | none | Keep rows with at least this many non-NA values. |
| `subset` | `None` | Only consider these columns. |
| `value` (fillna) | required | Scalar, or dict/Series mapping column to fill value. |
| `limit` | `None` | Maximum consecutive fills. |

```python
df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})
df.isna().sum()                       # missing count per column: a 1, b 2
df.isna().mean()                      # missing fraction
df.dropna()                           # only row 2 survives
df.dropna(subset=["a"])
df.fillna({"a": df["a"].median(), "b": 0})
df.ffill()                            # forward fill (replaces fillna(method="ffill"))
df.interpolate()                      # linear interpolation
```

### Sorting, ranking, duplicates

```python
DataFrame.sort_values(by, *, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)
DataFrame.sort_index(*, axis=0, level=None, ascending=True, ...)
DataFrame.rank(axis=0, method='average', numeric_only=False, na_option='keep', ascending=True, pct=False)
DataFrame.drop_duplicates(subset=None, *, keep='first', inplace=False, ignore_index=False)
DataFrame.duplicated(subset=None, keep='first')
```

```python
df = pd.DataFrame({"team": ["x", "y", "x", "y"], "pts": [10, 30, 20, 30]})
df.sort_values(["team", "pts"], ascending=[True, False])
df.sort_values("team", key=lambda s: s.str.lower())
df["rank"] = df["pts"].rank(method="dense", ascending=False)
df.drop_duplicates(subset=["team"], keep="last")
df.duplicated(subset=["pts"]).sum()   # 1
```

### Index manipulation

```python
DataFrame.set_index(keys, *, drop=True, append=False, inplace=False, verify_integrity=<no_default>)
DataFrame.reset_index(level=None, *, drop=False, inplace=False, col_level=0, col_fill='', allow_duplicates=<no_default>, names=None)
DataFrame.reindex(labels=None, *, index=None, columns=None, axis=None, method=None, fill_value=None, limit=None, tolerance=None, ...)
```

```python
df = pd.DataFrame({"id": [3, 1, 2], "v": [30, 10, 20]})
df = df.set_index("id").sort_index()
df.reindex([1, 2, 3, 4], fill_value=0)   # conform to new labels
df.reset_index()                          # id back to a column
```

### Combining: merge, join, concat

#### pd.merge / DataFrame.merge

```python
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), indicator=False, validate=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `how` | {'inner','left','right','outer','cross','left_anti','right_anti'} | `'inner'` | Join type (anti joins added in 3.0). |
| `on` | label or list | `None` | Key column(s) present in both frames. |
| `left_on` / `right_on` | label or list | `None` | Keys with different names. |
| `left_index` / `right_index` | bool | `False` | Join on the index. |
| `suffixes` | tuple | `('_x', '_y')` | Suffixes for overlapping non-key columns. |
| `indicator` | bool or str | `False` | Add `_merge` column: `left_only`, `right_only`, `both`. |
| `validate` | str | `None` | Assert key relationship: `'one_to_one'`, `'one_to_many'`, `'many_to_one'`, `'many_to_many'`. |

```python
users = pd.DataFrame({"user_id": [1, 2, 3], "name": ["Ada", "Ben", "Cy"]})
orders = pd.DataFrame({"user_id": [1, 1, 3, 4], "amount": [50, 20, 70, 10]})

pd.merge(users, orders, on="user_id")                     # inner: 3 rows
users.merge(orders, on="user_id", how="left")             # Ben has NaN amount
m = users.merge(orders, on="user_id", how="outer", indicator=True)
m["_merge"].value_counts()
users.merge(orders, on="user_id", how="left", validate="one_to_many")
```

#### pd.concat

```python
pd.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=<no_default>, copy=<no_default>)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `axis` | `0` | `0` stacks rows, `1` places side by side (aligned on index). |
| `join` | `'outer'` | `'inner'` keeps only shared columns/labels. |
| `ignore_index` | `False` | Discard old index and create a new RangeIndex. |
| `keys` | `None` | Build a MultiIndex level identifying the source. |

```python
a = pd.DataFrame({"x": [1, 2]})
b = pd.DataFrame({"x": [3], "y": [9]})
pd.concat([a, b], ignore_index=True)                      # y is NaN for a's rows
pd.concat([a, b], keys=["first", "second"])
pd.concat([a, a.rename(columns={"x": "z"})], axis=1)
```

#### DataFrame.join and merge_asof

```python
left = pd.DataFrame({"v": [1, 2]}, index=["a", "b"])
right = pd.DataFrame({"w": [10]}, index=["a"])
left.join(right, how="left")           # join on index

trades = pd.DataFrame({"t": pd.to_datetime(["10:00:01", "10:00:05"], format="%H:%M:%S"), "px": [100, 101]})
quotes = pd.DataFrame({"t": pd.to_datetime(["10:00:00", "10:00:04"], format="%H:%M:%S"), "bid": [99, 100]})
pd.merge_asof(trades, quotes, on="t")  # nearest earlier quote for each trade (both sorted by key)
```

### GroupBy

#### DataFrame.groupby

```python
DataFrame.groupby(by=None, level=None, *, as_index=True, sort=True, group_keys=True, observed=True, dropna=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `by` | label, list, function, dict, Series | `None` | Grouping keys. |
| `level` | int or name | `None` | Group by index level(s). |
| `as_index` | bool | `True` | `False` returns keys as columns (SQL style). |
| `sort` | bool | `True` | Sort group keys. |
| `observed` | bool | `True` | Only show observed categories for categorical keys (default changed to True in 3.0). |
| `dropna` | bool | `True` | Exclude NaN keys; `False` keeps them as a group. |

Returns: `DataFrameGroupBy`.

| GroupBy method | Result |
|----------------|--------|
| `sum`, `mean`, `median`, `min`, `max`, `std`, `var`, `count`, `size`, `nunique`, `first`, `last` | One row per group |
| `agg` / `aggregate` | One row per group, multiple functions |
| `transform` | Same shape as input (broadcast group result back) |
| `filter` | Subset of original rows from groups passing a test |
| `apply` | Arbitrary function per group (slowest; last resort) |
| `cumsum`, `cumcount`, `rank`, `shift`, `diff`, `pct_change` | Same shape, within-group |
| `rolling`, `resample` | Windowed operations per group |
| `head(n)`, `nth(n)` | Rows per group |

```python
df = pd.DataFrame({
    "dept": ["eng", "eng", "ops", "ops", "ops"],
    "level": ["jr", "sr", "jr", "jr", "sr"],
    "salary": [100, 150, 70, 80, 120],
})

df.groupby("dept")["salary"].mean()
df.groupby(["dept", "level"], as_index=False)["salary"].sum()

# named aggregation: output_column=(input_column, function)
df.groupby("dept").agg(
    n=("salary", "size"),
    avg=("salary", "mean"),
    top=("salary", "max"),
)
#       n    avg  top
# dept
# eng   2  125.0  150
# ops   3   90.0  120

# transform: add group statistics as a feature
df["dept_avg"] = df.groupby("dept")["salary"].transform("mean")
df["salary_vs_dept"] = df["salary"] - df["dept_avg"]

# filter: keep departments with more than 2 employees
df.groupby("dept").filter(lambda g: len(g) > 2)

df.groupby("dept").size()
df.groupby("dept")["salary"].rank(ascending=False)
```

### Reshaping

#### pivot_table, pivot, crosstab

```python
DataFrame.pivot_table(values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=True, sort=True)
DataFrame.pivot(*, columns, index=<no_default>, values=<no_default>)
pd.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False)
```

```python
sales = pd.DataFrame({
    "region": ["N", "N", "S", "S", "S"],
    "q": ["Q1", "Q2", "Q1", "Q1", "Q2"],
    "rev": [10, 20, 5, 15, 30],
})
sales.pivot_table(index="region", columns="q", values="rev", aggfunc="sum", fill_value=0, margins=True)
# q       Q1  Q2  All
# region
# N       10  20   30
# S       20  30   50
# All     30  50   80

pd.crosstab(sales["region"], sales["q"])                       # counts
pd.crosstab(sales["region"], sales["q"], normalize="index")    # row proportions
```

`pivot` reshapes without aggregation and raises `ValueError: Index contains duplicate entries, cannot reshape` if index/column pairs repeat; use `pivot_table` in that case.

#### melt, stack, unstack, explode

```python
DataFrame.melt(id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True)
DataFrame.stack(level=-1, dropna=<no_default>, sort=<no_default>, future_stack=True)
DataFrame.unstack(level=-1, fill_value=None, sort=True)
DataFrame.explode(column, ignore_index=False)
```

```python
wide = pd.DataFrame({"id": [1, 2], "math": [90, 80], "art": [70, 60]})
long = wide.melt(id_vars="id", var_name="subject", value_name="score")
#    id subject  score
# 0   1    math     90
# 1   2    math     80
# 2   1     art     70
# 3   2     art     60
back = long.pivot(index="id", columns="subject", values="score")

s = long.set_index(["id", "subject"])["score"]
s.unstack()                  # subject levels become columns
s.unstack().stack()          # and back

tags = pd.DataFrame({"doc": [1, 2], "tags": [["a", "b"], ["c"]]})
tags.explode("tags")         # one row per tag
```

### Descriptive statistics and window operations

```python
DataFrame.corr(method='pearson', min_periods=1, numeric_only=False)
Series.rolling(window, min_periods=None, center=False, win_type=None, on=None, closed=None, step=None, method='single')
Series.expanding(min_periods=1, method='single')
Series.ewm(com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, times=None, method='single')
Series.shift(periods=1, freq=None, axis=0, fill_value=<no_default>, suffix=None)
Series.pct_change(periods=1, fill_method=None, freq=None, **kwargs)
```

| Method | Description |
|--------|-------------|
| `sum`, `mean`, `median`, `std` (ddof=1), `var`, `min`, `max`, `quantile`, `mode` | Reductions (skip NaN) |
| `cumsum`, `cumprod`, `cummax`, `cummin` | Cumulative |
| `diff`, `pct_change`, `shift` | Lags and changes (key for time-series features) |
| `corr`, `cov`, `corrwith` | Pairwise relationships; `method='spearman'` or `'kendall'` |
| `rolling(w).mean()` | Moving window statistics |
| `expanding().mean()` | All-history statistics |
| `ewm(span=w).mean()` | Exponentially weighted statistics |

```python
s = pd.Series([1, 2, 4, 7, 11, 16])
s.diff()                         # NaN, 1, 2, 3, 4, 5
s.pct_change()
s.shift(1)                       # lag-1 feature
s.rolling(window=3).mean()       # NaN, NaN, 2.33, 4.33, 7.33, 11.33
s.rolling(3, min_periods=1).max()
s.expanding().mean()
s.ewm(span=3, adjust=False).mean()

df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 5, 9], "c": [4, 3, 2, 1]})
df.corr()                        # Pearson correlation matrix
df.corr(method="spearman")
```

Note: pandas `std`/`var` default to `ddof=1` (sample statistics), while NumPy defaults to `ddof=0`.

### Time series

```python
pd.Timestamp("2024-03-10 14:30")
pd.Timedelta(days=2, hours=3)
DataFrame.resample(rule, closed=None, label=None, convention='start', on=None, level=None, origin='start_day', offset=None, group_keys=False)
Series.tz_localize(tz, ambiguous='raise', nonexistent='raise')
Series.tz_convert(tz)
```

| Frequency alias | Meaning |
|-----------------|---------|
| `"s"`, `"min"`, `"h"`, `"D"` | Second, minute, hour, calendar day |
| `"B"` | Business day |
| `"W"`, `"W-MON"` | Weekly (Sunday end by default) |
| `"ME"`, `"MS"` | Month end, month start |
| `"QE"`, `"QS"` | Quarter end, quarter start |
| `"YE"`, `"YS"` | Year end, year start |

```python
idx = pd.date_range("2024-01-01", periods=96, freq="h")
ts = pd.Series(np.arange(96, dtype=float), index=idx)

ts.resample("D").mean()                  # daily averages (3 rows)
ts.resample("6h").agg(["min", "max"])
ts.loc["2024-01-02"]                      # partial string indexing: one whole day
ts.loc["2024-01-01 06:00":"2024-01-01 09:00"]
ts.rolling("12h").sum()                   # time-based window

df = pd.DataFrame({"when": pd.to_datetime(["2024-03-10 14:30", "2024-12-25 08:00"])})
df["year"] = df["when"].dt.year
df["dow"] = df["when"].dt.dayofweek       # Monday = 0
df["is_weekend"] = df["when"].dt.dayofweek >= 5
df["month_name"] = df["when"].dt.month_name()
df["when_utc"] = df["when"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")
(df["when"].iloc[1] - df["when"].iloc[0]).days   # 289
```

### String methods (.str)

```python
s = pd.Series(["  Alice Smith", "bob JONES ", None, "Cara-Lee Wu"])
s.str.strip().str.title()          # 'Alice Smith', 'Bob Jones', NaN, 'Cara-Lee Wu'
s.str.lower().str.contains("jones", na=False)
s.str.split(" ", expand=True)       # split into columns
s.str.len()
s.str.replace(r"[^A-Za-z ]", "", regex=True)
s.str.extract(r"(?P<first>\w+)\s+(?P<last>\w+)")   # regex groups -> columns
s.str.startswith("C", na=False)
s.str[:3]                           # slice each string
```

The `.str` accessor is vectorized over the column and propagates missing values.

### Categorical (.cat)

```python
size = pd.Series(["M", "S", "L", "M"]).astype(pd.CategoricalDtype(["S", "M", "L"], ordered=True))
size.cat.codes             # 1, 0, 2, 1
size.sort_values()         # S, M, M, L (by category order)
size > "S"                 # comparisons respect the order
size.cat.add_categories(["XL"])
```

### Output to NumPy and other structures

```python
DataFrame.to_numpy(dtype=None, copy=False, na_value=<no_default>)
DataFrame.to_dict(orient='dict', *, into=dict, index=True)
DataFrame.to_records(index=True, column_dtypes=None, index_dtypes=None)
```

```python
df = pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]})
df.to_numpy()                          # float64 array (2, 2)
df.to_numpy(dtype="float32", na_value=0.0)
df.to_dict(orient="records")           # [{'a': 1, 'b': 0.5}, {'a': 2, 'b': nan}]
df.to_dict(orient="list")              # {'a': [1, 2], 'b': [0.5, nan]}
```

### Utilities: pipe, eval, options, plotting

```python
def add_ratio(d, num, den):
    return d.assign(ratio=d[num] / d[den])

df = pd.DataFrame({"x": [1, 2], "y": [4, 8]})
df.pipe(add_ratio, "x", "y")

df.eval("z = x * 2 + y")                       # expression on columns, returns new frame

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.3f}".format)
with pd.option_context("display.max_rows", 10):
    print(df)
pd.reset_option("display.float_format")

# plotting (requires matplotlib)
# df.plot(x="x", y="y", kind="line"); df["y"].plot.hist(bins=20); df.plot.scatter(x="x", y="y")
```

## Tutorials

All tutorials generate their own data so they run offline.

### Tutorial 1: Cleaning a messy customer dataset

```python
import numpy as np
import pandas as pd
from io import StringIO

raw = StringIO("""customer_id,signup_date,country,age,monthly_spend,plan
1,2023-01-15,us,34,49.99,Pro
2,2023-02-30,US,,19.99,basic
3,2023-03-10,  Canada ,29,n/a,Basic
3,2023-03-10,  Canada ,29,n/a,Basic
4,2023-04-01,uk,212,99.0,PRO
5,,UK,41,0,basic
""")

# 1. Load, treating "n/a" as missing
df = pd.read_csv(raw, na_values=["n/a"])

# 2. Remove exact duplicate rows
df = df.drop_duplicates()

# 3. Normalize text columns
df["country"] = df["country"].str.strip().str.upper().replace({"CANADA": "CA"})
df["plan"] = df["plan"].str.lower().astype("category")

# 4. Parse dates; invalid dates (Feb 30) become NaT
df["signup_date"] = pd.to_datetime(df["signup_date"], format="%Y-%m-%d", errors="coerce")

# 5. Treat impossible ages as missing, then impute with the median
df.loc[~df["age"].between(0, 120), "age"] = np.nan
df["age"] = df["age"].fillna(df["age"].median()).astype(int)

# 6. Impute spend with the per-plan median
df["monthly_spend"] = df["monthly_spend"].fillna(
    df.groupby("plan", observed=True)["monthly_spend"].transform("median")
)

print(df)
#    customer_id signup_date country  age  monthly_spend   plan
# 0            1  2023-01-15      US   34         49.990    pro
# 1            2         NaT      US   34         19.990  basic
# 2            3  2023-03-10      CA   29          9.995  basic
# 4            4  2023-04-01      UK   34         99.000    pro
# 5            5         NaT      UK   41          0.000  basic
print(df.isna().sum())     # only signup_date still has missing values (2)
```

Each step is a vectorized column operation, and the `.loc[mask, col] = value` pattern sets values safely under Copy-on-Write.

### Tutorial 2: Exploratory data analysis and group statistics

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 1_000
df = pd.DataFrame({
    "region": rng.choice(["north", "south", "east", "west"], size=n),
    "channel": rng.choice(["web", "store"], size=n, p=[0.6, 0.4]),
    "units": rng.poisson(3, size=n),
    "price": rng.normal(20, 4, size=n).round(2),
})
df["revenue"] = df["units"] * df["price"]

# overall shape and summary
print(df.shape)                                # (1000, 5)
print(df.describe().loc[["mean", "std"]].round(2))

# revenue by region and channel
summary = (
    df.groupby(["region", "channel"])
      .agg(orders=("units", "size"), revenue=("revenue", "sum"), avg_price=("price", "mean"))
      .round(1)
      .sort_values("revenue", ascending=False)
)
print(summary.head())

# share of revenue by region (sums to 1)
share = df.groupby("region")["revenue"].sum() / df["revenue"].sum()
print(share.round(3))

# cross-tab of channel mix per region, as row percentages
print(pd.crosstab(df["region"], df["channel"], normalize="index").round(2))

# correlation between numeric columns
print(df[["units", "price", "revenue"]].corr().round(2))
```

### Tutorial 3: Time-series feature engineering for forecasting

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.date_range("2024-01-01", periods=365, freq="D")
trend = np.linspace(100, 150, 365)
weekly = 10 * np.sin(2 * np.pi * idx.dayofweek / 7)
sales = pd.DataFrame({"sales": trend + weekly + rng.normal(0, 5, 365)}, index=idx)

feat = sales.copy()
# calendar features
feat["dow"] = feat.index.dayofweek
feat["month"] = feat.index.month
feat["is_weekend"] = (feat["dow"] >= 5).astype(int)
# lag and rolling features (shift first so no future data leaks into features)
for lag in (1, 7, 14):
    feat[f"lag_{lag}"] = feat["sales"].shift(lag)
feat["roll_mean_7"] = feat["sales"].shift(1).rolling(7).mean()
feat["roll_std_7"] = feat["sales"].shift(1).rolling(7).std()
feat["ewm_14"] = feat["sales"].shift(1).ewm(span=14).mean()
feat = feat.dropna()

# chronological split (never shuffle time series)
train = feat.loc[:"2024-10-31"]
test = feat.loc["2024-11-01":]
print(train.shape, test.shape)     # (291, 10) (60, 10)

# monthly aggregate report
monthly = sales["sales"].resample("ME").agg(["mean", "min", "max"]).round(1)
print(monthly.head(3))
```

Key points: `shift` before `rolling` prevents target leakage, `resample("ME")` aggregates to month end (the old alias `"M"` raises in pandas 3.0), and string slicing on a `DatetimeIndex` makes chronological splits easy.

### Tutorial 4: From DataFrame to a scikit-learn model

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(1)
n = 2_000
df = pd.DataFrame({
    "tenure_months": rng.integers(1, 72, n),
    "monthly_charge": rng.normal(70, 20, n).round(2),
    "contract": rng.choice(["monthly", "one_year", "two_year"], n, p=[0.5, 0.3, 0.2]),
    "support_calls": rng.poisson(1.5, n),
})
logit = (-1.5 + 0.04 * df["monthly_charge"] - 0.06 * df["tenure_months"]
         + 0.5 * df["support_calls"] + np.where(df["contract"] == "monthly", 1.0, -1.0))
df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

X = df.drop(columns="churn")
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = ["contract"]

model = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)          # DataFrames go straight in; columns selected by name
print(f"test accuracy: {model.score(X_test, y_test):.2f}")   # about 0.78

# attach predictions back to the frame for error analysis
results = X_test.assign(y_true=y_test, p_churn=model.predict_proba(X_test)[:, 1])
print(results.groupby("contract")[["y_true", "p_churn"]].mean().round(2))
```

Passing DataFrames (not NumPy arrays) into scikit-learn lets `ColumnTransformer` select columns by name, and `set_output(transform="pandas")` on transformers returns DataFrames instead of arrays.

### Tutorial 5: Joining and reshaping multiple sources

```python
import pandas as pd

products = pd.DataFrame({"sku": ["A1", "B2", "C3"], "category": ["toys", "books", "toys"]})
sales = pd.DataFrame({
    "sku": ["A1", "B2", "A1", "C3", "D4"],
    "month": ["2024-01", "2024-01", "2024-02", "2024-02", "2024-02"],
    "units": [5, 3, 7, 2, 9],
})

merged = sales.merge(products, on="sku", how="left", validate="many_to_one", indicator=True)
print(merged[merged["_merge"] == "left_only"])   # D4 has no product record

report = merged.pivot_table(index="category", columns="month", values="units",
                            aggfunc="sum", fill_value=0)
print(report)
# month     2024-01  2024-02
# category
# books           3        0
# toys            5        9

tidy = report.reset_index().melt(id_vars="category", var_name="month", value_name="units")
print(tidy.sort_values(["category", "month"]).to_string(index=False))
```

## Performance & Best Practices

1. **Vectorize; avoid `iterrows` and row-wise `apply`.** Column arithmetic, `.str`, `.dt`, `np.where`, `np.select`, and `Series.map` are 10x-1000x faster than Python loops.
2. **Use efficient dtypes.** Convert low-cardinality strings to `category`, downcast numbers (`pd.to_numeric(..., downcast="integer")`), use `float32` where precision allows. Check with `df.memory_usage(deep=True)`.
3. **Read only what you need.** `usecols=`, `dtype=`, `nrows=`, and `chunksize=` in `read_csv`; `columns=` and `filters=` in `read_parquet`.
4. **Prefer Parquet over CSV** for intermediate datasets: smaller, typed, and much faster to load.
5. **Use the PyArrow engine and backend** for big CSVs: `pd.read_csv(path, engine="pyarrow", dtype_backend="pyarrow")`.
6. **Build once, do not grow.** Collect pieces in a list and call `pd.concat` once; repeated concatenation in a loop is quadratic.
7. **Use `groupby` built-ins** (`"sum"`, `"mean"`, `transform("mean")`) rather than `apply` with a lambda; they run in Cython.
8. **Sort the index** (`sort_index()`) before repeated label-based slicing, and use `set_index` for repeated lookups.
9. **Method chaining** with `assign`, `pipe`, and `query` keeps transformations readable and avoids intermediate variables; Copy-on-Write makes it cheap.
10. **Install the `performance` extra** (numexpr, bottleneck) for faster `eval`, `query`, and NaN reductions.
11. **Outgrowing memory?** Switch to Polars, DuckDB (`duckdb.sql("SELECT ... FROM df")` queries a DataFrame in place), or Dask.

```python
import numpy as np
import pandas as pd

n = 200_000
df = pd.DataFrame({"a": np.random.default_rng(0).random(n), "g": np.random.default_rng(1).choice(list("abcd"), n)})

# slow: row-wise apply
# df["b"] = df.apply(lambda r: r["a"] * 2 if r["g"] == "a" else r["a"], axis=1)
# fast: vectorized
df["b"] = np.where(df["g"] == "a", df["a"] * 2, df["a"])

before = df.memory_usage(deep=True).sum()
df["g"] = df["g"].astype("category")
df["a"] = df["a"].astype("float32")
after = df.memory_usage(deep=True).sum()
print(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
```

## Common Errors & Troubleshooting

| Error message | Cause | Fix |
|---------------|-------|-----|
| `KeyError: 'col'` | Column name misspelled, has whitespace, or different case. | Check `df.columns.tolist()`; clean with `df.columns = df.columns.str.strip()`. |
| `ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series through chained assignment.` (3.0 warning) | `df["a"][mask] = v` never updates `df` under Copy-on-Write. | `df.loc[mask, "a"] = v`. |
| `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame` (2.x) | Assigning to a filtered subset. | Use `.loc` on the original, or `.copy()` the subset. Gone in 3.0. |
| `ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().` | `and`/`or`/`if` with a Series. | Use `&`, `|`, `~` with parentheses: `(a > 1) & (b < 2)`. |
| `ValueError: cannot reindex on an axis with duplicate labels` | Assigning or aligning with a duplicated index. | `df.index.duplicated().any()`; `reset_index(drop=True)` or deduplicate. |
| `ValueError: Index contains duplicate entries, cannot reshape` | `pivot`/`unstack` with repeated keys. | Use `pivot_table(aggfunc=...)` or deduplicate first. |
| `MergeError: Merge keys are not unique in right dataset; not a many-to-one merge` | `validate=` caught a key problem. | Deduplicate keys or rethink the join; prevents silent row explosion. |
| `ValueError: You are trying to merge on int64 and str columns` | Key dtypes differ. | Cast both: `df["id"] = df["id"].astype(str)`. |
| `TypeError: unsupported operand type(s) for -: 'str' and 'int'` | Numeric column read as text. | `pd.to_numeric(col, errors="coerce")`. |
| `IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer` | `astype(int)` on a column with NaN. | Fill first, or use nullable `astype("Int64")`. |
| `ParserError: Error tokenizing data. C error: Expected 5 fields in line 12, saw 6` | Malformed CSV row / wrong delimiter. | Set `sep=`, check `quotechar`, or `on_bad_lines="skip"`/`"warn"`. |
| `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 ...` | File not UTF-8. | `encoding="latin-1"` or `"cp1252"`. |
| `ValueError: time data "13/01/2024" doesn't match format "%m/%d/%Y"` | Wrong or mixed date formats. | Pass the right `format=`, `dayfirst=True`, `format="mixed"`, or `errors="coerce"`. |
| `ValueError: Invalid frequency: M ... 'M' is no longer supported for offsets. Please use 'ME' instead.` | Old frequency alias in 3.0. | `"ME"`, `"QE"`, `"YE"`, `"h"`, `"min"`, `"s"`. |
| `TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'` | Removed in 3.0. | `df.ffill()` / `df.bfill()`. |
| `AttributeError: 'DataFrame' object has no attribute 'append'` | Removed in 2.0. | `pd.concat([df, new_rows], ignore_index=True)`. |
| `AttributeError: 'DataFrame' object has no attribute 'applymap'` | Removed in 3.0. | `df.map(func)`. |
| `AttributeError: Can only use .str accessor with string values, not integer` | Column is not text (numbers, mixed). | `.astype(str)` first. |
| `PerformanceWarning: DataFrame is highly fragmented` | Inserting many columns one at a time. | Build new columns in a dict and `pd.concat([df, pd.DataFrame(new)], axis=1)`. |
| `ImportError: Missing optional dependency 'pyarrow'` | Parquet/Arrow features need PyArrow. | `pip install pyarrow` (same pattern for `openpyxl`, `sqlalchemy`, `tabulate`). |

## Interoperability

| Library | Integration |
|---------|-------------|
| NumPy | `df.to_numpy()`, `s.to_numpy()`; NumPy ufuncs apply directly to Series (`np.log(s)`) and keep the index. |
| scikit-learn | Accepts DataFrames; `feature_names_in_` records columns; `set_output(transform="pandas")` returns DataFrames. |
| XGBoost / LightGBM / CatBoost | Accept DataFrames directly, including `category` dtype columns (`enable_categorical=True` in XGBoost). |
| statsmodels | Formula API: `smf.ols("y ~ x1 + C(group)", data=df).fit()`. |
| PyTorch | `torch.tensor(df[cols].to_numpy(dtype="float32"))`, or wrap a DataFrame in a `torch.utils.data.Dataset`. |
| TensorFlow | `tf.data.Dataset.from_tensor_slices(dict(df))`. |
| Matplotlib / seaborn / plotly | `df.plot(...)`; `sns.scatterplot(data=df, x=..., y=...)`; `px.line(df, ...)`. |
| PyArrow | `pa.Table.from_pandas(df)`, `table.to_pandas()`; Arrow-backed dtypes via `pd.ArrowDtype`. |
| Polars | `pl.from_pandas(df)`, `polars_df.to_pandas()`. |
| DuckDB | `duckdb.sql("SELECT g, avg(a) FROM df GROUP BY g").df()` queries a DataFrame in place. |
| SQL databases | `pd.read_sql(query, engine)`, `df.to_sql(name, engine)` via SQLAlchemy. |
| Dask | `dask.dataframe` mirrors the pandas API in parallel partitions. |
| cuDF | GPU DataFrame; `python -m cudf.pandas script.py` accelerates unchanged pandas code. |

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.1, 5.9]})
X = df[["x"]].to_numpy()                    # (3, 1) for scikit-learn style APIs
y = df["y"].to_numpy()
coef = np.polyfit(df["x"], df["y"], 1)       # NumPy functions accept Series
df["log_y"] = np.log(df["y"])                # ufunc keeps the index
```

## Cheat Sheet

### Load and inspect

| Task | Code |
|------|------|
| Read CSV | `pd.read_csv("f.csv")` |
| Read selected columns with dtypes | `pd.read_csv("f.csv", usecols=["a", "b"], dtype={"a": "int32"})` |
| Read Parquet | `pd.read_parquet("f.parquet")` |
| First / last rows | `df.head()`, `df.tail(3)` |
| Shape, dtypes, memory | `df.shape`, `df.dtypes`, `df.info()` |
| Summary stats | `df.describe(include="all")` |
| Value frequencies | `df["c"].value_counts(normalize=True)` |
| Missing per column | `df.isna().sum()` |

### Select and filter

| Task | Code |
|------|------|
| Column(s) | `df["a"]`, `df[["a", "b"]]` |
| Row by label / position | `df.loc["r1"]`, `df.iloc[0]` |
| Filter rows | `df[df["a"] > 0]`, `df.query("a > 0 and b == 'x'")` |
| In a set | `df[df["c"].isin(["x", "y"])]` |
| Set values conditionally | `df.loc[df["a"] < 0, "a"] = 0` |
| Columns by dtype | `df.select_dtypes(include="number")` |
| Top n | `df.nlargest(5, "score")` |

### Transform

| Task | Code |
|------|------|
| New column | `df["c"] = df["a"] * df["b"]` or `df.assign(c=lambda d: d.a * d.b)` |
| Rename | `df.rename(columns={"old": "new"})` |
| Drop columns | `df.drop(columns=["x"])` |
| Change dtype | `df.astype({"a": "float32", "c": "category"})` |
| Map values | `df["c"].map({"lo": 0, "hi": 1})` |
| Conditional value | `np.where(df["a"] > 0, "pos", "neg")` |
| Bin | `pd.cut(df["age"], [0, 18, 65, 120])`, `pd.qcut(df["x"], 4)` |
| One-hot | `pd.get_dummies(df, columns=["c"], dtype=int)` |
| Fill / drop missing | `df.fillna({"a": 0})`, `df.dropna(subset=["b"])`, `df.ffill()` |
| Deduplicate | `df.drop_duplicates(subset=["id"])` |
| Sort | `df.sort_values(["a", "b"], ascending=[True, False])` |

### Aggregate and reshape

| Task | Code |
|------|------|
| Group sum | `df.groupby("g")["v"].sum()` |
| Multiple aggregations | `df.groupby("g").agg(n=("v", "size"), avg=("v", "mean"))` |
| Group stat as feature | `df["g_mean"] = df.groupby("g")["v"].transform("mean")` |
| Pivot table | `df.pivot_table(index="r", columns="c", values="v", aggfunc="sum")` |
| Wide to long | `df.melt(id_vars="id")` |
| Long to wide | `df.pivot(index="id", columns="var", values="val")` |
| Frequency table | `pd.crosstab(df["a"], df["b"])` |

### Combine

| Task | Code |
|------|------|
| SQL-style join | `left.merge(right, on="key", how="left")` |
| Different key names | `left.merge(right, left_on="id", right_on="user_id")` |
| Stack rows | `pd.concat([df1, df2], ignore_index=True)` |
| Side by side | `pd.concat([df1, df2], axis=1)` |
| Join on index | `df1.join(df2)` |

### Time series

| Task | Code |
|------|------|
| Parse dates | `pd.to_datetime(df["d"], format="%Y-%m-%d")` |
| Date parts | `df["d"].dt.year`, `.dt.month`, `.dt.dayofweek` |
| Date range | `pd.date_range("2024-01-01", periods=30, freq="D")` |
| Resample | `ts.resample("W").sum()`, `ts.resample("ME").mean()` |
| Rolling mean | `s.rolling(7).mean()` |
| Lag | `s.shift(1)` |
| Percent change | `s.pct_change()` |
| Time zones | `s.dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")` |

### Export

| Task | Code |
|------|------|
| CSV without index | `df.to_csv("out.csv", index=False)` |
| Parquet | `df.to_parquet("out.parquet")` |
| NumPy array | `df.to_numpy()` |
| List of dicts | `df.to_dict(orient="records")` |
| Markdown table | `df.to_markdown()` (needs tabulate) |

## Further Resources

- Official documentation: https://pandas.pydata.org/docs/
- Getting started tutorials: https://pandas.pydata.org/docs/getting_started/index.html
- 10 minutes to pandas: https://pandas.pydata.org/docs/user_guide/10min.html
- User guide: https://pandas.pydata.org/docs/user_guide/index.html
- API reference: https://pandas.pydata.org/docs/reference/index.html
- Copy-on-Write guide: https://pandas.pydata.org/docs/user_guide/copy_on_write.html
- Release notes (what's new): https://pandas.pydata.org/docs/whatsnew/index.html
- Comparison with SQL: https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html
- Cheat sheet (PDF): https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
- GitHub: https://github.com/pandas-dev/pandas
- Book: Wes McKinney, *Python for Data Analysis*, 3rd ed. (free online): https://wesmckinney.com/book/
- Book: Jake VanderPlas, *Python Data Science Handbook* (free online): https://jakevdp.github.io/PythonDataScienceHandbook/
- Tom Augspurger, *Modern Pandas* blog series: https://tomaugspurger.net/posts/modern-1-intro/
- Kaggle Learn, Pandas course: https://www.kaggle.com/learn/pandas
