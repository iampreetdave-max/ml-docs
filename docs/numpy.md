# NumPy

> The fundamental package for array computing in Python.

NumPy provides the `ndarray`, a fast, memory-efficient, N-dimensional array of homogeneous data, together with a large library of vectorized mathematical functions, linear algebra, random number generation, and Fourier transforms. Almost every numerical and machine learning library in Python (pandas, SciPy, scikit-learn, Matplotlib, PyTorch, JAX, TensorFlow) either builds on NumPy or speaks its array protocol.

Covers NumPy 2.x (examples verified against NumPy 2.4). Differences from NumPy 1.x are called out where relevant.

## Overview

### What NumPy is

NumPy (Numerical Python) is an open-source library that supplies:

- **`ndarray`**: a contiguous (or strided) block of memory interpreted as an N-dimensional grid of values of a single data type (`dtype`).
- **Universal functions (ufuncs)**: element-wise operations implemented in C that run over whole arrays without Python loops (`np.add`, `np.exp`, `np.maximum`, ...).
- **Broadcasting**: a set of rules that lets arrays of different shapes participate in the same operation without copying data.
- **Submodules**: `numpy.linalg` (linear algebra), `numpy.random` (random sampling), `numpy.fft` (Fourier transforms), `numpy.polynomial` (polynomials), `numpy.strings` (vectorized string operations, new in 2.0), `numpy.ma` (masked arrays), and `numpy.testing`.

### History and maintainers

- 1995: **Numeric** created by Jim Hugunin and others.
- 2001: **Numarray** written as a faster alternative for large arrays.
- 2005-2006: Travis Oliphant merged both into **NumPy 1.0**.
- 2020: NumPy paper published in *Nature* ("Array programming with NumPy").
- June 2024: **NumPy 2.0**, the first major release in 18 years: cleaned-up namespace, new type-promotion rules (NEP 50), a new string dtype (`StringDType`), and Array API standard support.
- NumPy is a NumFOCUS fiscally sponsored project, developed by a community of volunteers and paid maintainers, licensed under the BSD-3-Clause license.

### When to use NumPy

- Dense numerical data: vectors, matrices, images, time-series windows, tensors on the CPU.
- Vectorized math where Python loops would be too slow.
- As the interchange format between libraries (most ML libraries accept and return NumPy arrays).
- Feature matrices for scikit-learn (`X` of shape `(n_samples, n_features)`, `y` of shape `(n_samples,)`).
- Implementing algorithms from scratch (gradient descent, k-means, PCA) for learning and prototyping.

### When not to use NumPy

| Situation | Better tool |
|-----------|-------------|
| Tabular data with labeled, heterogeneous columns | pandas, Polars |
| GPU computation or automatic differentiation | PyTorch, JAX, CuPy |
| Data larger than memory | Dask, Zarr, memory-mapped arrays, Polars lazy |
| Sparse matrices | `scipy.sparse` |
| Ragged / variable-length nested data | Python lists, Awkward Array, PyArrow |
| Labeled N-D arrays (climate, geoscience) | xarray |

### Where it fits in the ML stack

```text
          Deep learning: PyTorch / TensorFlow / JAX
          Classical ML:  scikit-learn, XGBoost, LightGBM
          Scientific:    SciPy, statsmodels
          Data frames:   pandas, Polars
          Plotting:      Matplotlib, seaborn
  ---------------------------------------------------------
                         NumPy (ndarray, ufuncs, dtypes)
  ---------------------------------------------------------
          BLAS / LAPACK (OpenBLAS, MKL, Accelerate), C runtime
```

## Installation

### pip

```bash
pip install numpy
# pin to the 2.x series
pip install "numpy>=2,<3"
# upgrade
pip install --upgrade numpy
```

Wheels on PyPI ship with OpenBLAS for linear algebra (Accelerate on recent macOS arm64 wheels), so no compiler is required.

### conda

```bash
conda install -c conda-forge numpy
# choose the BLAS implementation on conda-forge
conda install -c conda-forge numpy "libblas=*=*mkl"
conda install -c conda-forge numpy "libblas=*=*openblas"
```

### GPU variants

NumPy itself is CPU-only. For a GPU array library with a NumPy-compatible API, use CuPy (NVIDIA CUDA / AMD ROCm):

```bash
pip install cupy-cuda12x
```

### Verifying the install

```python
import numpy as np

print(np.__version__)        # e.g. 2.4.2
np.show_config()             # BLAS/LAPACK, compiler, SIMD extensions
print(np.show_runtime())     # SIMD features and threadpool info detected at runtime
```

```bash
python -c "import numpy; print(numpy.__version__)"
```

### Version compatibility notes

- NumPy 2.x requires Python 3.10+ (2.3+ requires 3.11+).
- Packages compiled against NumPy 1.x may fail with `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`. Upgrade those packages, or temporarily pin `numpy<2`.

## Core Concepts

### The ndarray

An `ndarray` is a pointer to a block of memory plus metadata describing how to interpret it.

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float32)

print(a.ndim)      # 2            number of axes
print(a.shape)     # (2, 3)       size along each axis
print(a.size)      # 6            total elements
print(a.dtype)     # float32      element type
print(a.itemsize)  # 4            bytes per element
print(a.nbytes)    # 24           total bytes
print(a.strides)   # (12, 4)      bytes to step along each axis
print(a.flags['C_CONTIGUOUS'])  # True
```

The **strides** tell NumPy how many bytes to jump to move one step along each axis. Transposing an array just swaps the strides; no data is copied:

```python
b = a.T
print(b.shape, b.strides)   # (3, 2) (4, 12)
print(np.shares_memory(a, b))  # True
```

### dtypes

Every array has a single dtype. Common ones:

| Kind | dtypes | Notes |
|------|--------|-------|
| Boolean | `np.bool` | 1 byte per element. `np.bool_` still works. |
| Signed integer | `np.int8`, `int16`, `int32`, `int64` | Default int is `int64` on all 64-bit platforms in NumPy 2 (was `int32` on Windows in 1.x). |
| Unsigned integer | `np.uint8` ... `uint64` | `uint8` is the standard image pixel type. |
| Float | `np.float16`, `float32`, `float64` | `float64` is the default. ML frameworks usually prefer `float32`. |
| Complex | `np.complex64`, `complex128` | |
| Datetime | `np.datetime64`, `np.timedelta64` | With units, e.g. `datetime64[ns]`. |
| String | `np.str_` (fixed width `<U10`), `np.dtypes.StringDType()` | `StringDType` is variable-width (2.0+). |
| Object | `object` | Python objects; slow, avoid where possible. |

```python
x = np.arange(5)                 # int64
y = x.astype(np.float32)         # new array, converted
z = np.zeros(3, dtype="uint8")   # dtype from string
print(np.dtype("float32").itemsize)  # 4
print(np.iinfo(np.int8))         # min -128, max 127
print(np.finfo(np.float32).eps)  # 1.1920929e-07
```

### Type promotion (NEP 50, NumPy 2)

NumPy 2 changed how Python scalars interact with arrays. Python `int`/`float` are "weak": they adopt the array's dtype instead of upcasting it.

```python
a = np.array([1, 2, 3], dtype=np.float32)
print((a + 3.0).dtype)                 # float32 (same in 1.x)
print((np.float32(3) + 3.0).dtype)     # float32 in 2.x (float64 in 1.x)

u = np.array([200], dtype=np.uint8)
# u + 300 -> OverflowError: Python integer 300 out of bounds for uint8
print(np.result_type(np.int8, np.float32))  # float32
```

### Vectorization and ufuncs

A ufunc applies an operation element-wise in compiled code. Replace Python loops with whole-array expressions:

```python
x = np.linspace(0, 1, 1_000_000)

# slow: Python loop
# y = [xi**2 + 2*xi + 1 for xi in x]

# fast: vectorized
y = x**2 + 2*x + 1
```

Ufuncs also have methods: `reduce`, `accumulate`, `outer`, `at`, `reduceat`.

```python
np.add.reduce([1, 2, 3, 4])        # 10   (same as sum)
np.add.accumulate([1, 2, 3, 4])    # [ 1  3  6 10]
np.multiply.outer([1, 2], [10, 20, 30])
# [[10 20 30]
#  [20 40 60]]
counts = np.zeros(4, dtype=int)
np.add.at(counts, [0, 1, 1, 3], 1)  # unbuffered in-place, handles repeats
print(counts)                       # [1 2 0 1]
```

### Axes

Reductions take an `axis` argument. Think of `axis=k` as "the axis that gets collapsed".

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])
m.sum()            # 21
m.sum(axis=0)      # [5 7 9]   collapse rows -> one value per column
m.sum(axis=1)      # [ 6 15]   collapse columns -> one value per row
m.sum(axis=1, keepdims=True)  # [[ 6] [15]] shape (2, 1), handy for broadcasting
```

### Broadcasting

When two arrays have different shapes, NumPy compares their shapes from the trailing (rightmost) dimension backwards. Two dimensions are compatible if they are equal or one of them is 1. Missing leading dimensions are treated as 1.

```text
A:      (5, 4, 3)
B:         (4, 1)
Result: (5, 4, 3)

A:      (3,)
B:      (4,)
Error:  operands could not be broadcast together
```

```python
X = np.random.default_rng(0).normal(size=(100, 3))
mu = X.mean(axis=0)            # shape (3,)
sigma = X.std(axis=0)          # shape (3,)
Z = (X - mu) / sigma           # (100, 3) op (3,) -> (100, 3)

# pairwise differences with np.newaxis (alias for None)
a = np.array([1, 2, 3])
diff = a[:, np.newaxis] - a[np.newaxis, :]   # shape (3, 3)

# pairwise squared Euclidean distances between rows of P (n, d) and Q (m, d)
P = np.random.default_rng(1).random((5, 2))
Q = np.random.default_rng(2).random((4, 2))
D2 = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(axis=-1)   # (5, 4)
```

### Views vs copies

Basic slicing returns a **view** that shares memory. Fancy (integer-array) and boolean indexing return a **copy**.

```python
a = np.arange(10)
v = a[2:5]       # view
v[0] = 100
print(a[2])      # 100  -- original modified

c = a[[2, 3, 4]] # copy (fancy indexing)
c[0] = -1
print(a[2])      # 100  -- unchanged

print(v.base is a)        # True
safe = a[2:5].copy()      # explicit copy when you need independence
```

Functions like `reshape`, `ravel`, and `transpose` return views when possible; `flatten` always copies.

### Indexing

```python
a = np.arange(24).reshape(2, 3, 4)

a[0]            # first block, shape (3, 4)
a[0, 1]         # row 1 of block 0, shape (4,)
a[0, 1, 2]      # scalar 6
a[:, 1, :]      # shape (2, 4)
a[..., -1]      # last element along last axis, shape (2, 3)
a[0, ::-1]      # reverse rows of block 0

# boolean mask
x = np.array([3, -1, 4, -1, 5])
x[x < 0] = 0    # [3 0 4 0 5]

# fancy indexing
rows = np.array([0, 1])
cols = np.array([2, 3])
m = np.arange(12).reshape(3, 4)
m[rows, cols]   # [2 7] -> elements (0,2) and (1,3)
m[np.ix_([0, 2], [1, 3])]  # submatrix rows {0,2} x cols {1,3}
```

### Random numbers: the Generator API

Use `np.random.default_rng()` rather than the legacy global functions (`np.random.rand`, `np.random.seed`):

```python
rng = np.random.default_rng(seed=42)
rng.random(3)                 # uniform [0, 1)
rng.normal(0, 1, size=(2, 2))
rng.integers(0, 10, size=5)   # high is exclusive
```

### NumPy 2 namespace cleanup (deprecations and removals)

| Removed / deprecated (1.x) | Use instead |
|----------------------------|-------------|
| `np.float_`, `np.complex_` | `np.float64`, `np.complex128` |
| `np.NaN`, `np.Inf`, `np.NINF`, `np.PINF` | `np.nan`, `np.inf`, `-np.inf` |
| `np.product`, `np.cumproduct` | `np.prod`, `np.cumprod` |
| `np.alltrue`, `np.sometrue` | `np.all`, `np.any` |
| `np.round_` | `np.round` |
| `np.in1d` (removed in 2.4) | `np.isin` |
| `np.trapz` (removed in 2.4) | `np.trapezoid` |
| `np.row_stack` (deprecated) | `np.vstack` |
| `np.asfarray` | `np.asarray(x, dtype=np.float64)` |
| `np.find_common_type` | `np.result_type`, `np.promote_types` |
| `np.msort` | `np.sort(a, axis=0)` |
| `np.cast`, `np.source`, `np.lookfor` | removed |
| `np.array(x, copy=False)` silently copied | now raises if a copy is needed; use `np.asarray(x)` |
| `np.core` (private) | `np._core`; use public APIs |

The `ruff` linter rule `NPY201` automatically flags most of these.

## API Reference

### Array creation

#### np.array

```python
np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `object` | array_like | required | List, tuple, nested sequence, or object exposing the array interface. |
| `dtype` | dtype | `None` | Desired type; inferred if `None`. |
| `copy` | bool or None | `True` | `True` always copies; `None` copies only if needed; `False` never copies and raises `ValueError` if it would have to. |
| `order` | {'K','A','C','F'} | `'K'` | Memory layout of the result. |
| `ndmin` | int | `0` | Minimum number of dimensions (prepends 1s). |

Returns: `ndarray`.

```python
np.array([1, 2, 3])                  # array([1, 2, 3])
np.array([[1, 2], [3, 4]], dtype=float)
np.array([1, 2], ndmin=2)            # array([[1, 2]])
```

#### np.asarray

```python
np.asarray(a, dtype=None, order=None, *, device=None, copy=None, like=None)
```

Converts input to an array without copying if it is already an ndarray of the right dtype. Prefer this at function boundaries.

```python
def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x)

normalize([3, 4])   # array([0.6, 0.8])
```

#### np.zeros, np.ones, np.empty, np.full

```python
np.zeros(shape, dtype=float, order='C', *, device=None, like=None)
np.ones(shape, dtype=None, order='C', *, device=None, like=None)
np.empty(shape, dtype=float, order='C', *, device=None, like=None)
np.full(shape, fill_value, dtype=None, order='C', *, device=None, like=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | int or tuple of ints | required | Output shape. |
| `dtype` | dtype | `float64` | Element type. |
| `fill_value` | scalar | required (`full`) | Value to fill. |

`np.empty` does not initialize memory; values are garbage until written.

```python
np.zeros((2, 3))
np.ones(4, dtype=np.int32)
np.full((2, 2), np.nan)
buf = np.empty(1000)   # fill before reading
```

The `*_like` variants (`np.zeros_like`, `np.ones_like`, `np.empty_like`, `np.full_like`) copy shape and dtype from an existing array.

#### np.arange

```python
np.arange([start,] stop[, step,], dtype=None, *, device=None, like=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | number | `0` | Start of interval (inclusive). |
| `stop` | number | required | End of interval (exclusive). |
| `step` | number | `1` | Spacing. |

```python
np.arange(5)          # [0 1 2 3 4]
np.arange(2, 10, 3)   # [2 5 8]
np.arange(0, 1, 0.25) # [0.   0.25 0.5  0.75]
```

For non-integer steps prefer `linspace`, since floating-point rounding can make `arange` include or exclude the endpoint unpredictably.

#### np.linspace, np.logspace, np.geomspace

```python
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None)
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
np.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num` | int | `50` | Number of samples. |
| `endpoint` | bool | `True` | Whether `stop` is included. |
| `retstep` | bool | `False` | Also return the spacing. |
| `base` | float | `10.0` | Base for `logspace` (exponents `start`..`stop`). |

```python
np.linspace(0, 1, 5)       # [0.   0.25 0.5  0.75 1.  ]
np.logspace(-3, 0, 4)      # [0.001 0.01  0.1   1.   ]  -- e.g. regularization grid
np.geomspace(1, 1000, 4)   # [   1.   10.  100. 1000.]
```

#### np.eye, np.identity, np.diag

```python
np.eye(N, M=None, k=0, dtype=float, order='C', *, device=None, like=None)
np.identity(n, dtype=None, *, like=None)
np.diag(v, k=0)
```

```python
np.eye(3)                  # 3x3 identity
np.eye(3, k=1)             # ones on the first superdiagonal
np.diag([1, 2, 3])         # diagonal matrix
np.diag(np.arange(9).reshape(3, 3))  # extract diagonal: [0 4 8]
```

#### np.meshgrid

```python
np.meshgrid(*xi, copy=True, sparse=False, indexing='xy')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*xi` | 1-D arrays | required | Coordinate vectors. |
| `indexing` | {'xy','ij'} | `'xy'` | Cartesian (`xy`) or matrix (`ij`) indexing. |
| `sparse` | bool | `False` | Return broadcastable sparse grids to save memory. |

```python
x = np.linspace(-1, 1, 3)
y = np.linspace(0, 1, 2)
XX, YY = np.meshgrid(x, y)   # each shape (2, 3)
Z = XX**2 + YY**2            # evaluate a function on a grid
```

### Shape manipulation

#### reshape

```python
np.reshape(a, /, shape, order='C', *, copy=None)
ndarray.reshape(shape, /, *, order='C', copy=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | int or tuple | required | New shape; one dimension may be `-1` (inferred). |
| `order` | {'C','F','A'} | `'C'` | Read/write order. |
| `copy` | bool or None | `None` | `None`: view if possible; `True`: always copy; `False`: raise if a copy is required. |

In NumPy 2.1+ the second argument is named `shape` (the old `newshape` keyword is deprecated).

```python
a = np.arange(12)
a.reshape(3, 4)
a.reshape(-1, 2)     # shape (6, 2)
a.reshape(2, 2, -1)  # shape (2, 2, 3)
```

#### ravel, flatten

```python
a = np.arange(6).reshape(2, 3)
a.ravel()     # view when possible: [0 1 2 3 4 5]
a.flatten()   # always a copy
a.ravel(order="F")  # [0 3 1 4 2 5]
```

#### transpose, swapaxes, moveaxis

```python
np.transpose(a, axes=None)
np.swapaxes(a, axis1, axis2)
np.moveaxis(a, source, destination)
```

```python
img = np.zeros((32, 32, 3))          # HWC
chw = np.transpose(img, (2, 0, 1))   # CHW (PyTorch layout), shape (3, 32, 32)
back = np.moveaxis(chw, 0, -1)       # HWC again
np.swapaxes(np.zeros((2, 3, 4)), 0, 2).shape  # (4, 3, 2)
```

`np.matrix_transpose(x)` (and `x.mT`) transposes only the last two axes of a stack of matrices.

#### expand_dims, squeeze

```python
np.expand_dims(a, axis)
np.squeeze(a, axis=None)
```

```python
v = np.array([1, 2, 3])
np.expand_dims(v, 0).shape   # (1, 3)
v[:, None].shape             # (3, 1)
np.squeeze(np.zeros((1, 3, 1))).shape  # (3,)
```

#### concatenate, stack, vstack, hstack, column_stack

```python
np.concatenate(arrays, /, axis=0, out=None, *, dtype=None, casting='same_kind')
np.stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind')
np.vstack(tup, *, dtype=None, casting='same_kind')
np.hstack(tup, *, dtype=None, casting='same_kind')
np.column_stack(tup)
```

| Function | Behavior |
|----------|----------|
| `concatenate` | Join along an existing axis. `np.concat` is an Array API alias. |
| `stack` | Join along a new axis (all inputs same shape). |
| `vstack` | Stack row-wise (1-D inputs become rows). |
| `hstack` | Stack column-wise (along axis 1, or axis 0 for 1-D). |
| `column_stack` | 1-D arrays become columns of a 2-D array. |

```python
a = np.array([1, 2]); b = np.array([3, 4])
np.concatenate([a, b])     # [1 2 3 4]
np.stack([a, b])           # [[1 2] [3 4]]  shape (2, 2)
np.stack([a, b], axis=1)   # [[1 3] [2 4]]
np.vstack([a, b])          # [[1 2] [3 4]]
np.hstack([a, b])          # [1 2 3 4]
np.column_stack([a, b])    # [[1 3] [2 4]]
```

#### split, array_split

```python
np.split(ary, indices_or_sections, axis=0)
np.array_split(ary, indices_or_sections, axis=0)
```

`split` requires equal division; `array_split` allows unequal pieces.

```python
x = np.arange(10)
np.split(x, 2)           # two arrays of length 5
np.split(x, [3, 7])      # [0..2], [3..6], [7..9]
np.array_split(x, 3)     # lengths 4, 3, 3
```

#### tile, repeat

```python
np.tile([1, 2], 3)           # [1 2 1 2 1 2]
np.repeat([1, 2], 3)         # [1 1 1 2 2 2]
np.repeat([[1, 2]], 2, axis=0)  # [[1 2] [1 2]]
```

#### np.pad

```python
np.pad(array, pad_width, mode='constant', **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pad_width` | int or sequence | required | `((before_1, after_1), ...)` per axis. |
| `mode` | str | `'constant'` | `'constant'`, `'edge'`, `'reflect'`, `'symmetric'`, `'wrap'`, `'mean'`, ... |
| `constant_values` | scalar | `0` | Fill value for `'constant'`. |

```python
img = np.ones((2, 2))
np.pad(img, 1)                          # zero border, shape (4, 4)
np.pad([1, 2, 3], (2, 1), mode="edge")  # [1 1 1 2 3 3]
```

### Mathematical functions (ufuncs)

#### Arithmetic and elementary functions

```python
np.add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None)
```

All ufuncs share the keyword arguments `out`, `where`, `dtype`, and `casting`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `out` | ndarray | `None` | Write result into an existing array (avoids allocation). |
| `where` | array_like of bool | `True` | Compute only where True; elsewhere `out` keeps its value. |
| `dtype` | dtype | `None` | Override computation dtype. |
| `casting` | str | `'same_kind'` | Casting rule for `out`. |

| Operation | Function | Operator |
|-----------|----------|----------|
| Add / subtract | `np.add`, `np.subtract` | `+`, `-` |
| Multiply / divide | `np.multiply`, `np.divide` | `*`, `/` |
| Floor divide / modulo | `np.floor_divide`, `np.mod` | `//`, `%` |
| Power | `np.power` (`np.pow`) | `**` |
| Matrix multiply | `np.matmul` | `@` |
| Exponential / log | `np.exp`, `np.log`, `np.log2`, `np.log10`, `np.log1p`, `np.expm1` | |
| Roots | `np.sqrt`, `np.cbrt`, `np.square` | |
| Trig | `np.sin`, `np.cos`, `np.tan`, `np.arcsin` (`np.asin`), `np.arctan2` (`np.atan2`) | |
| Hyperbolic | `np.sinh`, `np.cosh`, `np.tanh` | |
| Rounding | `np.round`, `np.floor`, `np.ceil`, `np.trunc`, `np.rint` | |
| Absolute / sign | `np.abs`, `np.sign` | |
| Elementwise min/max | `np.minimum`, `np.maximum`, `np.fmin`, `np.fmax` | |

```python
x = np.array([1.0, 2.0, 0.0])
np.divide(1.0, x, out=np.zeros_like(x), where=x != 0)  # [1.  0.5 0. ] no warning
np.log1p(1e-10)          # accurate log(1 + x) for small x
np.maximum(x - 1, 0)     # ReLU-like: [0. 1. 0.]
```

#### np.clip

```python
np.clip(a, a_min=<no value>, a_max=<no value>, out=None, *, min=<no value>, max=<no value>, **kwargs)
```

```python
np.clip([-2, 0.5, 3], 0, 1)         # [0.  0.5 1. ]
np.clip(np.array([1, 5, 9]), max=6) # [1 5 6]  (min/max keywords added in 2.1)
```

#### np.where

```python
np.where(condition, [x, y], /)
```

With three arguments, choose from `x` where `condition` is True and `y` elsewhere. With one argument, equivalent to `np.nonzero(condition)`.

```python
z = np.array([-1.5, 0.2, 3.0])
np.where(z > 0, z, 0.01 * z)   # leaky ReLU: [-0.015  0.2    3.   ]
np.where(z > 0)                # (array([1, 2]),)
```

#### np.isclose, np.allclose, np.array_equal

```python
np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
np.array_equal(a1, a2, equal_nan=False)
```

```python
0.1 + 0.2 == 0.3                 # False
np.isclose(0.1 + 0.2, 0.3)       # True
np.allclose([1.0, 2.0], [1.0, 2.0 + 1e-9])  # True
np.array_equal([1, np.nan], [1, np.nan], equal_nan=True)  # True
```

### Reductions and statistics

#### sum, prod, mean, std, var

```python
np.sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)
np.mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True)
np.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=None, correction=None)
np.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=None, correction=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `axis` | int, tuple, or None | `None` | Axis or axes to reduce; `None` reduces everything. |
| `dtype` | dtype | `None` | Accumulator type (use `float64` to avoid `float16` overflow). |
| `keepdims` | bool | `False` | Keep reduced axes with size 1. |
| `ddof` | int | `0` | Delta degrees of freedom. `ddof=1` gives the sample standard deviation (pandas' default). |
| `where` | bool array | `True` | Include only these elements. |

```python
x = np.array([[1, 2], [3, 4]], dtype=float)
x.sum(), x.mean(), x.std(), x.std(ddof=1)
# (np.float64(10.0), np.float64(2.5), np.float64(1.118...), np.float64(1.290...))
np.prod([1, 2, 3, 4])    # 24
```

NumPy 2 prints scalars with their type, e.g. `np.float64(10.0)`. Use `float(x)` or `.item()` to get a plain Python number.

#### min, max, argmin, argmax, ptp

```python
a = np.array([[3, 7, 1], [9, 2, 8]])
a.max()               # 9
a.max(axis=0)         # [9 7 8]
a.argmax()            # 3 (index into the flattened array)
np.unravel_index(a.argmax(), a.shape)  # (np.int64(1), np.int64(0))
a.argmax(axis=1)      # [1 0]
np.ptp(a, axis=1)     # [6 7]  max - min
```

Note: `ndarray.ptp()` the method was removed in 2.0; use the function `np.ptp`.

#### NaN-aware reductions

`np.nansum`, `np.nanmean`, `np.nanstd`, `np.nanvar`, `np.nanmin`, `np.nanmax`, `np.nanargmin`, `np.nanargmax`, `np.nanmedian`, `np.nanpercentile`, `np.nanquantile`.

```python
x = np.array([1.0, np.nan, 3.0])
np.mean(x)       # nan
np.nanmean(x)    # 2.0
```

#### median, percentile, quantile

```python
np.quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, weights=None)
np.percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, weights=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | float or array | required | Quantile(s) in [0, 1] (`quantile`) or [0, 100] (`percentile`). |
| `method` | str | `'linear'` | `'linear'`, `'lower'`, `'higher'`, `'nearest'`, `'midpoint'`, `'inverted_cdf'`, ... (renamed from `interpolation` in 1.22). |
| `weights` | array | `None` | Per-sample weights (2.0+, requires `method='inverted_cdf'`). |

```python
data = np.array([1, 2, 3, 4, 100])
np.median(data)                      # 3.0
np.percentile(data, [25, 75])        # [2. 4.]
np.quantile(data, 0.9)               # 61.6
```

#### cumsum, cumprod, diff, cumulative_sum

```python
np.cumsum([1, 2, 3, 4])            # [ 1  3  6 10]
np.cumulative_sum([1, 2, 3], include_initial=True)  # [0 1 3 6]  (2.1+)
np.cumprod([1, 2, 3, 4])           # [ 1  2  6 24]
np.diff([1, 4, 9, 16])             # [3 5 7]
np.diff([1, 4, 9, 16], n=2)        # [2 2]
```

#### Correlation, covariance, histogram

```python
np.corrcoef(x, y=None, rowvar=True)
np.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None)
np.histogram(a, bins=10, range=None, density=None, weights=None)
np.bincount(x, /, weights=None, minlength=0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rowvar` | bool | `True` | If True, each **row** is a variable. Set `False` for the usual `(n_samples, n_features)` layout. |
| `bins` | int, sequence, or str | `10` | Number of bins, bin edges, or a rule such as `'auto'`, `'fd'`. |
| `density` | bool | `None` | Normalize so the histogram integrates to 1. |

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
C = np.cov(X, rowvar=False)        # (3, 3)
R = np.corrcoef(X, rowvar=False)   # (3, 3), diagonal == 1
counts, edges = np.histogram(X[:, 0], bins=20)
np.bincount([0, 1, 1, 3])          # [1 2 0 1]
```

### Sorting, searching, and set operations

#### sort, argsort

```python
np.sort(a, axis=-1, kind=None, order=None, *, stable=None)
np.argsort(a, axis=-1, kind=None, order=None, *, stable=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `axis` | int or None | `-1` | Axis to sort; `None` flattens first. |
| `kind` | str | `None` (quicksort/introsort) | `'quicksort'`, `'mergesort'`, `'heapsort'`, `'stable'`. |
| `stable` | bool | `None` | `True` requests a stable sort (Array API keyword). |

```python
a = np.array([3, 1, 2])
np.sort(a)              # [1 2 3]
np.argsort(a)           # [1 2 0]
np.sort(a)[::-1]        # descending
idx = np.argsort(-a)    # descending indices
```

#### partition, argpartition

Find the k smallest elements in O(n) without fully sorting:

```python
scores = np.array([0.1, 0.9, 0.4, 0.8, 0.3])
top2 = np.argpartition(scores, -2)[-2:]          # indices of 2 largest (unordered)
top2 = top2[np.argsort(scores[top2])[::-1]]      # order them
print(top2)   # [1 3]
```

#### searchsorted

```python
np.searchsorted(a, v, side='left', sorter=None)
```

```python
edges = np.array([0, 10, 20, 30])
np.searchsorted(edges, [5, 10, 25])               # [1 1 3]
np.searchsorted(edges, [5, 10, 25], side="right") # [1 2 3]
```

`np.digitize(x, bins)` is a similar binning helper.

#### nonzero, argwhere, flatnonzero, count_nonzero

```python
m = np.array([[0, 1], [2, 0]])
np.nonzero(m)         # (array([0, 1]), array([1, 0]))
np.argwhere(m)        # [[0 1] [1 0]]
np.flatnonzero(m)     # [1 2]
np.count_nonzero(m)   # 2
```

#### unique and set operations

```python
np.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True, sorted=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_index` | bool | `False` | Indices of first occurrences. |
| `return_inverse` | bool | `False` | Indices to reconstruct the input. |
| `return_counts` | bool | `False` | Number of occurrences of each unique value. |
| `axis` | int | `None` | Unique rows/slices along this axis. |

```python
labels = np.array(["cat", "dog", "cat", "bird"])
classes, y = np.unique(labels, return_inverse=True)
print(classes)  # ['bird' 'cat' 'dog']
print(y)        # [1 2 1 0]  -- label encoding
vals, counts = np.unique(labels, return_counts=True)

# Array API style helpers (2.0+)
np.unique_values(labels)
np.unique_counts(labels)      # namedtuple (values, counts)
np.unique_inverse(labels)     # namedtuple (values, inverse_indices)
```

Set operations: `np.isin(element, test_elements)`, `np.intersect1d`, `np.union1d`, `np.setdiff1d`, `np.setxor1d`.

```python
np.isin([1, 2, 3, 4], [2, 4])    # [False  True False  True]
np.intersect1d([1, 2, 3], [2, 3, 4])  # [2 3]
np.setdiff1d([1, 2, 3], [2])     # [1 3]
```

### Logic

```python
x = np.array([1, np.nan, np.inf, -2])
np.isnan(x)        # [False  True False False]
np.isinf(x)        # [False False  True False]
np.isfinite(x)     # [ True False False  True]
np.any(x > 0), np.all(x > -5)
np.logical_and(x > 0, np.isfinite(x))   # [ True False False False]
```

Use `&`, `|`, `~` for element-wise boolean logic on arrays, and wrap each comparison in parentheses: `(x > 0) & (x < 5)`.

### Linear algebra (numpy.linalg and products)

#### np.matmul (@), np.dot, np.vecdot, np.outer

```python
np.matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None)
np.dot(a, b, out=None)
```

| Function | Behavior |
|----------|----------|
| `a @ b` / `np.matmul` | Matrix product; N-D inputs are treated as stacks of matrices in the last two axes (batched). |
| `np.dot` | For 2-D equals matmul; for N-D sums over last axis of `a` and second-to-last of `b`. |
| `np.vecdot` | Dot product over the last axis with broadcasting (2.0+). |
| `np.inner`, `np.outer`, `np.tensordot` | Inner, outer, and general tensor contractions. |

```python
A = np.arange(6).reshape(2, 3)
B = np.arange(12).reshape(3, 4)
(A @ B).shape                                  # (2, 4)
batch = np.random.default_rng(0).random((10, 3, 3))
(batch @ batch).shape                          # (10, 3, 3)
np.vecdot(np.ones((5, 3)), np.ones(3))         # [3. 3. 3. 3. 3.]
np.outer([1, 2], [3, 4])                       # [[3 4] [6 8]]
```

#### np.einsum

```python
np.einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subscripts` | str | required | Einstein summation spec, e.g. `'ij,jk->ik'`. |
| `optimize` | bool or str | `False` | `True`/`'optimal'` picks a cheaper contraction order for 3+ operands. |

```python
A = np.ones((3, 4)); B = np.ones((4, 5))
np.einsum("ij,jk->ik", A, B).shape     # (3, 5) matmul
np.einsum("ii->", np.eye(3))           # 3.0 trace
np.einsum("i,i->", [1, 2, 3], [4, 5, 6])  # 32 dot
np.einsum("bij,bjk->bik", np.ones((2, 3, 4)), np.ones((2, 4, 5))).shape  # (2, 3, 5)
```

#### solve, inv, lstsq, pinv

```python
np.linalg.solve(a, b)
np.linalg.inv(a)
np.linalg.lstsq(a, b, rcond=None)
np.linalg.pinv(a, rcond=None, hermitian=False, *, rtol=<no value>)
```

| Function | Returns |
|----------|---------|
| `solve(a, b)` | `x` with `a @ x == b`. Faster and more stable than `inv(a) @ b`. |
| `inv(a)` | Inverse; raises `LinAlgError: Singular matrix` if not invertible. |
| `lstsq(a, b)` | `(x, residuals, rank, singular_values)` minimizing the squared error. |
| `pinv(a)` | Moore-Penrose pseudo-inverse via SVD. |

```python
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])
x = np.linalg.solve(A, b)        # [2. 3.]

rng = np.random.default_rng(0)
X = np.column_stack([np.ones(50), rng.random(50)])
y = 2 + 3 * X[:, 1] + rng.normal(0, 0.1, 50)
coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
print(coef.round(1))             # approximately [2. 3.]
```

#### eig, eigh, svd

```python
np.linalg.eig(a)
np.linalg.eigh(a, UPLO='L')
np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)
```

These return named tuples in NumPy 2: `EigResult(eigenvalues, eigenvectors)`, `EighResult(...)`, `SVDResult(U, S, Vh)`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `full_matrices` | bool | `True` | `False` returns the reduced ("economy") SVD; use it for tall data matrices. |
| `compute_uv` | bool | `True` | `False` returns only singular values (see also `np.linalg.svdvals`). |
| `UPLO` | {'L','U'} | `'L'` | Triangle used by `eigh`. |

```python
S = np.array([[2.0, 1.0], [1.0, 2.0]])
w, V = np.linalg.eigh(S)         # w ascending: [1. 3.]

M = np.random.default_rng(0).random((100, 5))
U, s, Vt = np.linalg.svd(M, full_matrices=False)
U.shape, s.shape, Vt.shape       # (100, 5) (5,) (5, 5)
np.allclose(M, U @ np.diag(s) @ Vt)   # True
```

Use `eigh` for symmetric matrices such as covariance matrices: it is faster, returns real eigenvalues in ascending order, and orthonormal eigenvectors.

#### norm, det, slogdet, matrix_rank, cholesky, qr

```python
np.linalg.norm(x, ord=None, axis=None, keepdims=False)
np.linalg.cholesky(a, /, *, upper=False)
np.linalg.qr(a, mode='reduced')
```

| `ord` | Vector norm | Matrix norm |
|-------|-------------|-------------|
| `None` | L2 | Frobenius |
| `1` | L1 (sum of abs) | max column sum |
| `np.inf` | max abs | max row sum |
| `'nuc'` | n/a | nuclear (sum of singular values) |

```python
np.linalg.norm([3.0, 4.0])             # 5.0
np.linalg.norm([3.0, 4.0], ord=1)      # 7.0
X = np.random.default_rng(0).random((4, 3))
np.linalg.norm(X, axis=1)              # row norms, shape (4,)

A = np.array([[4.0, 2.0], [2.0, 3.0]])
np.linalg.det(A)                       # 8.0 (up to rounding)
sign, logdet = np.linalg.slogdet(A)    # stable for large matrices
L = np.linalg.cholesky(A)              # A == L @ L.T
Q, R = np.linalg.qr(A)
np.linalg.matrix_rank(A), np.trace(A)  # 2, 7.0
```

### Random sampling (numpy.random)

#### np.random.default_rng

```python
np.random.default_rng(seed=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int, SeedSequence, BitGenerator, Generator, or None | `None` | `None` pulls fresh entropy from the OS. |

Returns a `Generator` backed by PCG64.

| Method | Signature | Description |
|--------|-----------|-------------|
| `random` | `random(size=None, dtype=np.float64, out=None)` | Uniform [0, 1). |
| `integers` | `integers(low, high=None, size=None, dtype=np.int64, endpoint=False)` | Integers in [low, high). |
| `uniform` | `uniform(low=0.0, high=1.0, size=None)` | Uniform [low, high). |
| `normal` | `normal(loc=0.0, scale=1.0, size=None)` | Gaussian. |
| `standard_normal` | `standard_normal(size=None, dtype=np.float64, out=None)` | N(0, 1). |
| `choice` | `choice(a, size=None, replace=True, p=None, axis=0, shuffle=True)` | Sample from an array or `range(a)`. |
| `shuffle` / `permutation` | `shuffle(x, axis=0)` / `permutation(x, axis=0)` | In-place / copy. |
| `binomial`, `poisson`, `exponential`, `beta`, `gamma`, `dirichlet`, `multinomial` | various | Standard distributions. |
| `multivariate_normal` | `multivariate_normal(mean, cov, size=None, ...)` | Correlated Gaussians. |
| `spawn` | `spawn(n_children)` | Independent child generators for parallel workers. |

```python
rng = np.random.default_rng(42)
rng.integers(1, 7, size=10)                       # dice rolls
rng.choice(["a", "b", "c"], size=5, p=[0.5, 0.3, 0.2])
idx = rng.permutation(100)                        # shuffled indices
rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=3)
workers = rng.spawn(4)
```

Legacy functions (`np.random.seed`, `rand`, `randn`, `randint`, `RandomState`) still work but are frozen and use global state. Mapping: `rand(3)` to `rng.random(3)`, `randn(2, 3)` to `rng.standard_normal((2, 3))`, `randint(0, 10, 5)` to `rng.integers(0, 10, 5)`. The same seed gives different numbers in the two APIs.

### FFT (numpy.fft)

```python
np.fft.rfft(a, n=None, axis=-1, norm=None, out=None)
np.fft.rfftfreq(n, d=1.0, device=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | `None` | Transform length (pads or crops). |
| `norm` | {'backward','ortho','forward'} | `'backward'` | Normalization. |
| `d` | float | `1.0` | Sample spacing (1 / sampling rate). |

```python
fs = 100
t = np.arange(0, 1, 1 / fs)
sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
spec = np.fft.rfft(sig)
freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
print(sorted(freqs[np.argsort(np.abs(spec))[-2:]]))   # [5.0, 20.0]
recon = np.fft.irfft(spec, n=len(sig))
```

Also `fft`, `ifft`, `fft2`, `fftn`, `fftshift`. `scipy.fft` adds worker threads and real-to-real transforms.

### Input / output

```python
np.save(file, arr, allow_pickle=True)
np.load(file, mmap_mode=None, allow_pickle=False, ...)
np.savez_compressed(file, *args, allow_pickle=True, **kwds)
np.loadtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0, usecols=None, ...)
np.savetxt(fname, X, fmt='%.18e', delimiter=' ', header='', comments='# ', ...)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mmap_mode` | {None,'r+','r','w+','c'} | `None` | Memory-map instead of loading into RAM. |
| `allow_pickle` | bool | `False` on load | Needed for object arrays; never enable for untrusted files. |

```python
X = np.random.default_rng(0).random((1000, 10))
y = np.arange(1000)
np.save("X.npy", X)
X2 = np.load("X.npy")
big = np.load("X.npy", mmap_mode="r")       # lazy, larger-than-RAM friendly

np.savez_compressed("data.npz", X=X, y=y)
with np.load("data.npz") as d:
    print(d.files)                           # ['X', 'y']

np.savetxt("m.csv", np.eye(3), delimiter=",", fmt="%.2f", header="a,b,c", comments="")
m = np.loadtxt("m.csv", delimiter=",", skiprows=1)
```

For messy CSVs with mixed types, use `pandas.read_csv`.

### Strings (numpy.strings, 2.0+)

```python
s = np.array(["Apple", "banana ", " Cherry"], dtype=np.dtypes.StringDType())
np.strings.lower(np.strings.strip(s))   # ['apple' 'banana' 'cherry']
np.strings.str_len(s)                   # [5 7 7]
np.strings.startswith(s, "A")           # [ True False False]
```

`np.strings` supersedes the legacy `np.char` module.

### Functional and windowing helpers

```python
np.vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None)
np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
numpy.lib.stride_tricks.sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False)
```

```python
vb = np.vectorize(lambda v: "low" if v < 0.5 else "high")
vb(np.array([0.1, 0.7]))                      # ['low' 'high']  (convenience, not speed)

m = np.array([[3, 1, 2], [9, 7, 8]])
np.apply_along_axis(lambda r: r.max() - r.min(), 1, m)   # [2 2]

from numpy.lib.stride_tricks import sliding_window_view
w = sliding_window_view(np.arange(6), 3)      # shape (4, 3), read-only view
w.mean(axis=-1)                               # moving average: [1. 2. 3. 4.]
```

### Polynomials, calculus, interpolation

```python
from numpy.polynomial import Polynomial

x = np.linspace(0, 1, 20)
y = 1 + 2 * x + 3 * x**2
p = Polynomial.fit(x, y, deg=2)
print(p.convert().coef.round(6))   # [1. 2. 3.]
p(0.5)                             # 2.75

coef = np.polyfit(x, y, 2)         # legacy API, highest power first: [3. 2. 1.]
np.polyval(coef, 0.5)              # 2.75

xs = np.linspace(0, np.pi, 1001)
np.trapezoid(np.sin(xs), xs)       # about 2.0
np.gradient(xs**2, xs)             # approx 2 * xs
np.interp(2.5, [1, 2, 3], [10, 20, 30])   # 25.0
np.convolve([1, 2, 3], [0, 1, 0.5])       # [0.  1.  2.5 4.  1.5]
```

## Tutorials

### Tutorial 1: Linear regression with gradient descent

Fit `y = Xw + b` from scratch, comparing against the closed-form least-squares answer.

```python
import numpy as np

rng = np.random.default_rng(0)
n, d = 500, 3
X = rng.normal(size=(n, d))
true_w = np.array([1.5, -2.0, 0.5])
y = X @ true_w + 4.0 + rng.normal(0, 0.1, n)

# 1. initialise parameters
w = np.zeros(d)
b = 0.0
lr = 0.1

# 2. full-batch gradient descent on mean squared error
for epoch in range(200):
    pred = X @ w + b                 # (n,)
    err = pred - y
    grad_w = 2 * X.T @ err / n       # (d,)
    grad_b = 2 * err.mean()
    w -= lr * grad_w
    b -= lr * grad_b

print(w.round(2), round(b, 2))       # [ 1.5  -2.    0.51] 3.99

# 3. closed form via lstsq on [1, X]
A = np.column_stack([np.ones(n), X])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
print(coef.round(2))                 # [ 3.99  1.5  -2.    0.51]
```

Every step is a whole-array operation: one matrix-vector product for predictions and one for the gradient.

### Tutorial 2: PCA from scratch with SVD

```python
import numpy as np

rng = np.random.default_rng(1)
# 2-D latent data embedded in 5-D with noise
Z = rng.normal(size=(300, 2))
W = rng.normal(size=(2, 5))
X = Z @ W + 0.05 * rng.normal(size=(300, 5))

# 1. centre the data
mu = X.mean(axis=0)
Xc = X - mu

# 2. SVD of the centred data
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

# 3. explained variance ratio
var = s**2 / (len(X) - 1)
ratio = var / var.sum()
print(ratio.round(3))        # [0.796 0.203 0.    0.    0.   ]

# 4. project onto the top-k components and reconstruct
k = 2
X_proj = Xc @ Vt[:k].T       # (300, 2)
X_rec = X_proj @ Vt[:k] + mu
print(np.sqrt(((X - X_rec) ** 2).mean()).round(3))   # 0.039
```

`Vt[:k]` holds the principal axes (same as scikit-learn's `PCA.components_`, up to sign).

### Tutorial 3: k-means clustering with broadcasting

```python
import numpy as np

rng = np.random.default_rng(7)
centers_true = np.array([[0, 0], [5, 5], [0, 5]])
X = np.vstack([c + rng.normal(0, 0.6, size=(100, 2)) for c in centers_true])

def kmeans(X, k, n_iter=100, n_init=10, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):                                  # restarts avoid bad local minima
        C = X[rng.choice(len(X), size=k, replace=False)]     # init from data points
        for _ in range(n_iter):
            # (n, 1, 2) - (1, k, 2) -> (n, k, 2) -> squared distances (n, k)
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
            labels = d2.argmin(axis=1)
            new_C = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
                              for j in range(k)])
            if np.allclose(new_C, C):
                break
            C = new_C
        inertia = d2.min(axis=1).sum()                       # within-cluster sum of squares
        if best is None or inertia < best[0]:
            best = (inertia, C, labels)
    return best[1], best[2]

C, labels = kmeans(X, 3)
print(np.round(C[np.lexsort((C[:, 0], C[:, 1]))], 1))
# [[-0.1 -0.1]
#  [ 0.   4.8]
#  [ 5.   5. ]]
print(np.sort(np.bincount(labels)))   # [100 100 100]
```

A single random initialization can converge to a poor local minimum (two centers in one blob), which is why the loop keeps the best of `n_init` runs, exactly like scikit-learn's `KMeans(n_init=...)`. The distance matrix is computed in one broadcasted expression. For very large `n`, compute it in chunks, or use `scipy.spatial.distance.cdist`.

### Tutorial 4: Softmax classifier (multinomial logistic regression)

```python
import numpy as np

rng = np.random.default_rng(3)
n_per, k, d = 200, 3, 2
means = np.array([[0, 0], [3, 0], [0, 3]])
X = np.vstack([m + rng.normal(size=(n_per, d)) for m in means])
y = np.repeat(np.arange(k), n_per)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)    # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

Y = np.eye(k)[y]                 # one-hot encoding via fancy indexing, (600, 3)
W = np.zeros((d, k)); b = np.zeros(k)
for _ in range(500):
    P = softmax(X @ W + b)
    G = (P - Y) / len(X)         # gradient of cross-entropy w.r.t. logits
    W -= 0.5 * X.T @ G
    b -= 0.5 * G.sum(axis=0)

pred = (X @ W + b).argmax(axis=1)
loss = -np.log(softmax(X @ W + b)[np.arange(len(y)), y]).mean()
print(f"accuracy={np.mean(pred == y):.2f} loss={loss:.3f}")   # accuracy=0.91 loss=0.250
```

Key idioms: `np.eye(k)[y]` for one-hot, subtracting the row max before `exp`, and `P[np.arange(n), y]` to pick each row's true-class probability.

## Performance & Best Practices

1. **Vectorize.** Replace Python loops with ufuncs, broadcasting, and reductions. A typical speedup is 10x-100x.
2. **Pick dtypes deliberately.** `float32` halves memory and is what GPUs and most deep-learning code expect. Use `float64` for numerically sensitive statistics.
3. **Avoid unnecessary copies.** Use views (slicing, `reshape`, `transpose`), in-place operators (`x *= 2`), and the `out=` argument: `np.multiply(a, b, out=a)`.
4. **Preallocate** instead of appending. `np.append` and repeated `np.concatenate` in a loop copy the whole array each time (quadratic). Collect pieces in a list and concatenate once, or fill a preallocated `np.empty`.
5. **Prefer `solve` over `inv`**, `eigh` over `eig` for symmetric matrices, and `lstsq` for regression.
6. **Mind memory layout.** Reduce along contiguous axes when possible; `np.ascontiguousarray` before passing to C extensions.
7. **Watch broadcasting blow-ups.** `X[:, None, :] - Y[None, :, :]` creates an `(n, m, d)` temporary. Chunk it, or use the identity `|x-y|^2 = |x|^2 + |y|^2 - 2 x.y`.
8. **Control BLAS threads** with `threadpoolctl` or environment variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) to avoid oversubscription when running many processes.
9. **Use `np.random.default_rng(seed)`** and pass the generator around; avoid global seeding.
10. **Profile** with `%timeit` in IPython before optimizing; reach for Numba, Cython, or JAX only after vectorization.


```python
# distance matrix without a 3-D temporary
def sq_dists(A, B):
    return (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
```

## Common Errors & Troubleshooting

| Error message | Cause | Fix |
|---------------|-------|-----|
| `ValueError: operands could not be broadcast together with shapes (3,) (4,)` | Trailing dimensions differ and neither is 1. | Check `.shape`; add axes with `[:, None]` or `reshape`. |
| `ValueError: cannot reshape array of size 10 into shape (3,4)` | Element count does not match new shape. | Make the product of dims equal `a.size`, or use `-1` for one dim. |
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 ...` | Inner dimensions of `@` disagree. | Ensure `(n, k) @ (k, m)`; transpose one operand. |
| `IndexError: index 5 is out of bounds for axis 0 with size 5` | Zero-based indexing past the end. | Valid indices are `0 .. n-1` (or negative). |
| `IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed` | Using `a[i, j]` on a 1-D array. | Reshape to 2-D or index once. |
| `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()` | `if arr:` or `and`/`or` on arrays. | Use `.any()`, `.all()`, or `&`, `|` with parentheses. |
| `numpy.linalg.LinAlgError: Singular matrix` | Matrix not invertible. | Use `lstsq` or `pinv`, or regularize (`A + lam * np.eye(n)`). |
| `LinAlgError: Matrix is not positive definite` | `cholesky` on a non-PD matrix. | Add jitter to the diagonal, or use `eigh`. |
| `RuntimeWarning: divide by zero encountered in divide` / `invalid value encountered` | Division by zero or `0/0`, `log(0)`. | Guard with `np.where`, `np.divide(..., where=...)`, `np.errstate`, or add epsilon. |
| `RuntimeWarning: overflow encountered in exp` | Large exponent. | Subtract the max (softmax trick), use `np.logaddexp`, or `scipy.special.expit`. |
| `OverflowError: Python integer 300 out of bounds for uint8` | NEP 50: Python ints must fit the array dtype. | Cast first: `a.astype(np.int16) + 300`. |
| `AttributeError: np.float_ was removed in the NumPy 2.0 release. Use np.float64 instead.` | Old 1.x alias. | Use the replacement named in the message; run `ruff --select NPY201`. |
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` | Binary extension built against NumPy 1. | Upgrade the package; or `pip install "numpy<2"` temporarily. |
| `ValueError: Unable to avoid copy while creating an array as requested.` | `np.array(x, copy=False)` in NumPy 2 when a copy is needed. | Use `np.asarray(x)`. |
| `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape` | Ragged nested lists. | Pad to equal lengths, or explicitly use `dtype=object`. |
| `MemoryError: Unable to allocate 74.5 GiB for an array with shape ...` | Huge intermediate (often broadcasting). | Chunk the computation, use `float32`, or memory-map. |
| `ValueError: assignment destination is read-only` | Writing to a read-only view (e.g. `sliding_window_view`, `mmap_mode='r'`, `np.broadcast_to`). | `.copy()` first. |

Silent pitfalls: integer overflow wraps without warning in arrays (`np.array([2**62]) * 4`); `uint8` subtraction wraps (`np.uint8(0) - np.uint8(1)` is `255`); `a = b` aliases rather than copies.

## Interoperability

### Protocols

- **`__array__` / buffer protocol**: lets `np.asarray` consume pandas, PyArrow, PIL images, and more.
- **`__array_ufunc__` / `__array_function__`**: let other array types (CuPy, Dask, xarray, pandas) override NumPy functions.
- **DLPack (`np.from_dlpack`, `x.__dlpack__`)**: zero-copy exchange with PyTorch, JAX, CuPy, TensorFlow.
- **Array API standard**: NumPy 2 implements it in the main namespace; libraries such as SciPy and scikit-learn can dispatch on it (`array_api_compat`).

### With specific libraries

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
arr = df.to_numpy()                       # (2, 2) float64
df2 = pd.DataFrame(arr, columns=["a", "b"])
s = pd.Series(np.arange(3))
```

```python
import torch

a = np.arange(6, dtype=np.float32).reshape(2, 3)
t = torch.from_numpy(a)                   # shares memory with a (CPU)
back = t.numpy()                          # shares memory
t_copy = torch.tensor(a)                  # copies
gpu_back = t.cuda().cpu().numpy() if torch.cuda.is_available() else back
```

```python
from sklearn.linear_model import LogisticRegression

X = np.random.default_rng(0).normal(size=(100, 4))
y = (X[:, 0] > 0).astype(int)
clf = LogisticRegression().fit(X, y)      # sklearn takes and returns ndarrays
proba = clf.predict_proba(X)              # ndarray (100, 2)
```

| Library | To NumPy | From NumPy |
|---------|----------|------------|
| pandas | `df.to_numpy()`, `s.to_numpy()` | `pd.DataFrame(arr)` |
| PyTorch | `t.detach().cpu().numpy()` | `torch.from_numpy(a)` (shared), `torch.tensor(a)` (copy) |
| TensorFlow | `t.numpy()` | `tf.convert_to_tensor(a)` |
| JAX | `np.asarray(x)` | `jnp.asarray(a)` |
| CuPy | `cp.asnumpy(x)` or `x.get()` | `cp.asarray(a)` |
| SciPy sparse | `m.toarray()` | `scipy.sparse.csr_array(a)` |
| PIL | `np.asarray(img)` | `Image.fromarray(a)` |
| PyArrow | `arr.to_numpy()` | `pa.array(a)` |

## Cheat Sheet

### Creation and inspection

| Task | Code |
|------|------|
| From list | `np.array([1, 2, 3])` |
| Zeros / ones / constant | `np.zeros((2, 3))`, `np.ones(5)`, `np.full((2, 2), 7)` |
| Range / evenly spaced | `np.arange(0, 10, 2)`, `np.linspace(0, 1, 11)` |
| Identity | `np.eye(3)` |
| Random (seeded) | `rng = np.random.default_rng(0); rng.normal(size=(3, 3))` |
| Shape / dims / dtype | `a.shape`, `a.ndim`, `a.dtype` |
| Convert dtype | `a.astype(np.float32)` |

### Reshaping and combining

| Task | Code |
|------|------|
| Reshape | `a.reshape(3, -1)` |
| Flatten | `a.ravel()` |
| Add axis | `a[:, None]`, `np.expand_dims(a, 0)` |
| Remove size-1 axes | `a.squeeze()` |
| Transpose | `a.T`, `np.transpose(a, (2, 0, 1))` |
| Join along axis | `np.concatenate([a, b], axis=0)` |
| Stack as new axis | `np.stack([a, b])` |
| Split | `np.split(a, 3)`, `np.array_split(a, 3)` |

### Indexing

| Task | Code |
|------|------|
| Last element | `a[-1]` |
| Every other | `a[::2]` |
| Reverse | `a[::-1]` |
| Column j | `M[:, j]` |
| Boolean filter | `a[a > 0]` |
| Replace by condition | `np.where(a > 0, a, 0)` |
| Pick per-row element | `M[np.arange(n), idx]` |
| Top-k indices | `np.argsort(a)[-k:][::-1]` |
| One-hot | `np.eye(k)[labels]` |

### Math and stats

| Task | Code |
|------|------|
| Column means | `X.mean(axis=0)` |
| Standardize | `(X - X.mean(0)) / X.std(0)` |
| Min-max scale | `(X - X.min(0)) / np.ptp(X, axis=0)` |
| Ignore NaN | `np.nanmean(a)` |
| Percentiles | `np.percentile(a, [5, 95])` |
| Counts of values | `np.unique(a, return_counts=True)` |
| Correlation matrix | `np.corrcoef(X, rowvar=False)` |
| Cumulative sum | `np.cumsum(a)` |
| Approx equality | `np.allclose(a, b)` |

### Linear algebra

| Task | Code |
|------|------|
| Matrix product | `A @ B` |
| Solve Ax = b | `np.linalg.solve(A, b)` |
| Least squares | `np.linalg.lstsq(A, b, rcond=None)[0]` |
| Inverse / pseudo-inverse | `np.linalg.inv(A)`, `np.linalg.pinv(A)` |
| Eigen (symmetric) | `w, V = np.linalg.eigh(S)` |
| SVD | `U, s, Vt = np.linalg.svd(X, full_matrices=False)` |
| Norm / row norms | `np.linalg.norm(v)`, `np.linalg.norm(X, axis=1)` |
| Einstein sum | `np.einsum("ij,jk->ik", A, B)` |

### I/O

| Task | Code |
|------|------|
| Save / load one array | `np.save("a.npy", a)`, `np.load("a.npy")` |
| Save many (compressed) | `np.savez_compressed("d.npz", X=X, y=y)` |
| Memory-map | `np.load("a.npy", mmap_mode="r")` |
| Text | `np.savetxt("a.csv", a, delimiter=",")`, `np.loadtxt("a.csv", delimiter=",")` |

## Further Resources

- Official documentation: https://numpy.org/doc/stable/
- Absolute beginner's guide: https://numpy.org/doc/stable/user/absolute_beginners.html
- User guide: https://numpy.org/doc/stable/user/index.html
- API reference: https://numpy.org/doc/stable/reference/index.html
- NumPy 2.0 migration guide: https://numpy.org/doc/stable/numpy_2_0_migration_guide.html
- Release notes: https://numpy.org/doc/stable/release.html
- NumPy for MATLAB users: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
- NumPy tutorials: https://numpy.org/numpy-tutorials/
- GitHub: https://github.com/numpy/numpy
- NEP index (enhancement proposals): https://numpy.org/neps/
- Paper: Harris et al., "Array programming with NumPy", Nature 585, 2020: https://www.nature.com/articles/s41586-020-2649-2
- Book: Jake VanderPlas, *Python Data Science Handbook* (free online): https://jakevdp.github.io/PythonDataScienceHandbook/
- Book: Wes McKinney, *Python for Data Analysis*, 3rd ed. (free online): https://wesmckinney.com/book/
- From Python to NumPy (Nicolas Rougier): https://www.labri.fr/perso/nrougier/from-python-to-numpy/
- 100 NumPy exercises: https://github.com/rougier/numpy-100
