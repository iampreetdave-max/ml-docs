# TensorFlow

> An end-to-end open-source platform for numerical computing and machine learning, from research to production and edge devices.

TensorFlow is Google's library for building and running computations on tensors (multi-dimensional arrays) across CPUs, GPUs and TPUs. It provides automatic differentiation, a graph compiler (`tf.function` with optional XLA), a high-performance input pipeline (`tf.data`), distribution strategies for multi-device training, and a deployment story that spans servers (TensorFlow Serving), browsers (TensorFlow.js) and mobile/embedded devices (TensorFlow Lite, now LiteRT). The high-level modeling API is Keras.

Covers TensorFlow 2.16 and later (the 2.x line). Since TF 2.16, `tf.keras` resolves to **Keras 3**; see the Keras page for the modeling API in depth.

## Overview

### What TensorFlow is

TensorFlow was released by the Google Brain team in November 2015 under the Apache 2.0 license. TensorFlow 1.x was built around static computation graphs and sessions. TensorFlow 2.0 (September 2019) made **eager execution** the default, adopted Keras as the official high-level API, and introduced `tf.function` to recover graph performance. Development is led by Google with a large open-source community.

Key components:

| Component | Purpose |
|---|---|
| `tf.Tensor`, `tf.Variable` | Immutable and mutable n-dimensional arrays with device placement. |
| `tf.GradientTape` | Reverse-mode automatic differentiation. |
| `tf.function` | Traces Python into optimized graphs; optional XLA compilation with `jit_compile=True`. |
| `tf.data` | Scalable, parallel, streaming input pipelines. |
| `tf.distribute` | Data-parallel training across GPUs, machines and TPUs. |
| `tf.saved_model` | Language-neutral serialized format for serving and conversion. |
| `tf.lite` / LiteRT | On-device inference with quantization. |
| `keras` | High-level layers, models, training loops (Keras 3). |
| TensorBoard | Visualization of metrics, graphs, profiles. |

### Notable changes in recent releases

| Version | Change |
|---|---|
| 2.11 | Last version with native Windows GPU support was 2.10; from 2.11 use WSL2 for GPU on Windows. |
| 2.13 | Apple Silicon wheels published as `tensorflow` (no more separate `tensorflow-macos` needed). |
| 2.14 | `pip install tensorflow[and-cuda]` installs CUDA/cuDNN via pip on Linux. |
| 2.16 | Keras 3 becomes the default `tf.keras`. Legacy Keras 2 is available as the `tf_keras` package with `TF_USE_LEGACY_KERAS=1`. Clang is the default compiler for Linux builds. |
| 2.17+ | Continued CUDA/cuDNN updates; TFLite components are moving to the standalone LiteRT project (`ai-edge-litert`). |

### When to use TensorFlow

- You need a mature production path: TensorFlow Serving, TFX pipelines, Vertex AI, TensorFlow.js, or on-device deployment via TFLite/LiteRT.
- Training on TPUs (TensorFlow and JAX are the first-class TPU frameworks).
- Large-scale input pipelines where `tf.data` performance and TFRecord sharding matter.
- Existing codebases and pretrained models built on TensorFlow/Keras.

### When not to use it

- Classical tabular ML (use scikit-learn, XGBoost, LightGBM, CatBoost).
- Most new research code: the community has largely moved to PyTorch and JAX; many new model releases ship PyTorch weights first.
- Native Windows GPU training (use WSL2 or another framework).
- If you only need the Keras API, consider Keras 3 with the JAX or PyTorch backend and keep TensorFlow only for `tf.data` or export.

### Where it fits in the ML stack

```text
Raw data -> tf.data (parse, augment, batch, prefetch)
         -> keras.Model or custom tf.Module (training with fit() or GradientTape)
         -> tf.distribute (multi-GPU / multi-worker / TPU)
         -> tf.saved_model  -> TF Serving / Vertex AI
                            -> TFLite / LiteRT (mobile, edge)
                            -> TensorFlow.js (browser)
         -> TensorBoard (metrics, profiling)
```

## Installation

### pip (CPU or GPU, Linux / macOS / Windows CPU)

```bash
python -m pip install --upgrade pip
pip install tensorflow
```

Requires a 64-bit Python in the range supported by the release (check the install page; TF 2.16 supports Python 3.9 to 3.12).

### GPU on Linux (CUDA via pip)

```bash
pip install "tensorflow[and-cuda]"
```

This pulls matching NVIDIA CUDA and cuDNN wheels from PyPI; you only need a recent NVIDIA driver on the host (check with `nvidia-smi`).

### GPU on Windows

Native Windows GPU support ended with TF 2.10. Options:

1. Install WSL2 with Ubuntu, install the NVIDIA Windows driver (WSL support is built in), then inside WSL run `pip install "tensorflow[and-cuda]"`.
2. Use the CPU build natively: `pip install tensorflow`.

### macOS

```bash
pip install tensorflow           # CPU, Apple Silicon and Intel
pip install tensorflow-metal     # optional: GPU acceleration on Apple Silicon via Metal
```

`tensorflow-metal` is maintained by Apple and may lag behind the newest TensorFlow release; check compatibility before upgrading.

### conda

```bash
conda create -n tf python=3.11 -y
conda activate tf
pip install "tensorflow[and-cuda]"   # pip inside conda is the officially documented route
```

conda-forge also ships `tensorflow` packages, but the official instructions use pip.

### Legacy Keras 2 (tf_keras)

```bash
pip install tf_keras
export TF_USE_LEGACY_KERAS=1   # must be set before TensorFlow is imported
```

### Verifying the install

```python
import tensorflow as tf
import keras

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)   # 3.x for TF 2.16+
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
```

Build information (CUDA/cuDNN versions the wheel was built against):

```python
import tensorflow as tf
info = tf.sysconfig.get_build_info()
print(info.get("cuda_version"), info.get("cudnn_version"))
```

## Core Concepts

### Tensors

A `tf.Tensor` is an immutable, typed, n-dimensional array with a shape and a device. Tensors behave much like NumPy arrays and support broadcasting.

```python
import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])            # int32, shape (2, 2)
b = tf.constant([[1.0, 0.0], [0.0, 1.0]])    # float32
print(a.shape, a.dtype, a.device)
print(tf.cast(a, tf.float32) @ b)            # matrix multiply
print(a.numpy())                             # to NumPy (eager mode only)
```

Important properties:

- **dtype**: TensorFlow does not silently promote types. `int32 + float32` raises an error; use `tf.cast`.
- **shape**: may be partially unknown inside graphs (`(None, 28, 28)`), known in eager mode.
- **rank**: `tf.rank(x)` or `len(x.shape)`.
- Special types: `tf.RaggedTensor` (variable-length rows), `tf.sparse.SparseTensor`, string tensors.

### Variables

A `tf.Variable` holds mutable state such as model weights. Gradients are tracked automatically for trainable variables.

```python
w = tf.Variable(tf.random.normal([3, 1]), name="w")
w.assign_add(tf.ones_like(w))
print(w.trainable, w.shape)
```

### Eager execution and graphs

Operations execute immediately (eager) by default, which is easy to debug. Decorating a function with `tf.function` traces it into a graph that TensorFlow can optimize, run without Python overhead, serialize to a SavedModel, and compile with XLA.

```python
@tf.function
def step(x):
    return tf.reduce_sum(x * x)

print(step(tf.constant([1.0, 2.0, 3.0])))      # traces once, then runs graph
print(step.get_concrete_function(tf.TensorSpec([None], tf.float32)))
```

Tracing rules to keep in mind:

- A new trace happens for each new input *signature* (dtype/shape for tensors, value for Python scalars).
- Python side effects (`print`, list appends) run only during tracing. Use `tf.print` for runtime printing.
- Python `if`/`for` on tensors is converted by AutoGraph into `tf.cond` / `tf.while_loop`.
- Variables must be created only on the first trace (or outside the function).

### Automatic differentiation

`tf.GradientTape` records operations on watched tensors and computes gradients with `tape.gradient`.

```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x
dy_dx = tape.gradient(y, x)
print(dy_dx)  # 2*3 + 2 = 8.0
```

### Modules, layers and models

- `tf.Module`: the base container that tracks variables and sub-modules; supports checkpointing and SavedModel export.
- `keras.layers.Layer` and `keras.Model`: Keras 3 classes that add build logic, training loops (`fit`), metrics, and serialization.

```python
class Linear(tf.Module):
    def __init__(self, in_dim, out_dim, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.normal([in_dim, out_dim]), name="w")
        self.b = tf.Variable(tf.zeros([out_dim]), name="b")

    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
    def __call__(self, x):
        return x @ self.w + self.b

m = Linear(3, 2)
print(m(tf.ones([4, 3])).shape, [v.name for v in m.trainable_variables])
```

### Input pipelines

`tf.data.Dataset` represents a (possibly infinite) sequence of elements and supports parallel map, shuffle, batch, cache, and prefetch to keep accelerators busy.

```python
ds = (tf.data.Dataset.from_tensor_slices((tf.range(10), tf.range(10) * 2))
      .shuffle(10).batch(4).prefetch(tf.data.AUTOTUNE))
for x, y in ds:
    print(x.numpy(), y.numpy())
```

### Keras in TensorFlow 2.16+

`import keras` and `tf.keras` both give you Keras 3 when TensorFlow 2.16+ is installed (the `keras` package is a dependency). Keras 3 code written against `keras.ops` runs on TensorFlow, JAX, or PyTorch backends; the default backend is TensorFlow. Models are saved in the `.keras` format; SavedModel export uses `model.export(path)`.

```python
import keras
from keras import layers

model = keras.Sequential([keras.Input(shape=(4,)), layers.Dense(8, activation="relu"), layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
print(keras.config.backend())  # "tensorflow"
```

## API Reference

### Tensor creation

#### tf.constant

```python
tf.constant(value, dtype=None, shape=None, name="Const")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `value` | scalar, list, ndarray | required | Initial values. |
| `dtype` | `tf.DType` | inferred | e.g. `tf.float32`, `tf.int64`, `tf.string`. |
| `shape` | list/tuple | None | Reshape or broadcast a scalar to this shape. |
| `name` | str | `"Const"` | Op name. |

Returns: `tf.Tensor`.

```python
t = tf.constant(0.5, shape=[2, 3])         # filled 2x3
s = tf.constant(["a", "bb"])               # string tensor
```

#### tf.convert_to_tensor

```python
tf.convert_to_tensor(value, dtype=None, dtype_hint=None, name=None)
```

Converts NumPy arrays, Python lists, and tensor-likes into a `tf.Tensor`. Use `dtype_hint` to suggest a dtype without forcing it.

#### Filled and range tensors

| Function | Signature | Example |
|---|---|---|
| `tf.zeros` | `tf.zeros(shape, dtype=tf.float32)` | `tf.zeros([2, 3])` |
| `tf.ones` | `tf.ones(shape, dtype=tf.float32)` | `tf.ones([4])` |
| `tf.zeros_like` / `tf.ones_like` | `tf.zeros_like(x, dtype=None)` | `tf.ones_like(w)` |
| `tf.fill` | `tf.fill(dims, value)` | `tf.fill([2, 2], 7)` |
| `tf.eye` | `tf.eye(num_rows, num_columns=None, dtype=tf.float32)` | `tf.eye(3)` |
| `tf.range` | `tf.range(start, limit=None, delta=1, dtype=None)` | `tf.range(0, 10, 2)` |
| `tf.linspace` | `tf.linspace(start, stop, num)` | `tf.linspace(0.0, 1.0, 5)` |

#### Random tensors

```python
tf.random.set_seed(seed)
tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None)
tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None)
tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None)
tf.random.shuffle(value, seed=None)
tf.random.categorical(logits, num_samples, dtype=None, seed=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shape` | list/tuple | required | Output shape. |
| `mean`, `stddev` | float | 0.0, 1.0 | Normal distribution params. |
| `minval`, `maxval` | number | 0, None | Range for uniform (`maxval` required for ints). |
| `seed` | int | None | Op-level seed; combined with the global seed. |

For reproducible, splittable streams prefer `tf.random.Generator`:

```python
g = tf.random.Generator.from_seed(42)
print(g.normal([2, 2]))
print(tf.random.stateless_uniform([3], seed=[1, 2]))  # deterministic given seed pair
```

`keras.utils.set_random_seed(42)` seeds Python, NumPy and TensorFlow at once. For bit-exact GPU results also call `tf.config.experimental.enable_op_determinism()`.

### Tensor manipulation

| Function | Signature | Description |
|---|---|---|
| `tf.cast` | `tf.cast(x, dtype)` | Change dtype. |
| `tf.reshape` | `tf.reshape(tensor, shape)` | New shape; one dim may be `-1`. |
| `tf.transpose` | `tf.transpose(a, perm=None)` | Permute axes. |
| `tf.expand_dims` | `tf.expand_dims(input, axis)` | Insert a size-1 axis. |
| `tf.squeeze` | `tf.squeeze(input, axis=None)` | Remove size-1 axes. |
| `tf.concat` | `tf.concat(values, axis)` | Join along existing axis. |
| `tf.stack` | `tf.stack(values, axis=0)` | Join along new axis. |
| `tf.unstack` | `tf.unstack(value, num=None, axis=0)` | Split into list along axis. |
| `tf.split` | `tf.split(value, num_or_size_splits, axis=0)` | Split into pieces. |
| `tf.tile` | `tf.tile(input, multiples)` | Repeat tensor. |
| `tf.pad` | `tf.pad(tensor, paddings, mode="CONSTANT", constant_values=0)` | Pad. |
| `tf.gather` | `tf.gather(params, indices, axis=None, batch_dims=0)` | Select slices by index. |
| `tf.gather_nd` | `tf.gather_nd(params, indices, batch_dims=0)` | Select by multi-dim index. |
| `tf.where` | `tf.where(condition, x=None, y=None)` | Elementwise select, or indices of True. |
| `tf.boolean_mask` | `tf.boolean_mask(tensor, mask, axis=None)` | Keep entries where mask is True. |
| `tf.one_hot` | `tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None)` | One-hot encode. |
| `tf.tensor_scatter_nd_update` | `tf.tensor_scatter_nd_update(tensor, indices, updates)` | Functional update of a tensor. |
| `tf.shape` | `tf.shape(input)` | Dynamic shape as a tensor (works with unknown dims in graphs). |

```python
x = tf.reshape(tf.range(12), [3, 4])
print(tf.gather(x, [0, 2]))                 # rows 0 and 2
print(tf.where(x > 5, x, tf.zeros_like(x)))  # keep >5 else 0
print(tf.one_hot([0, 2, 1], depth=3))
print(x[1:, ::2])                           # NumPy-style slicing works
```

Tensors are immutable: `x[0, 0] = 1` fails. Use a `tf.Variable` with `assign` or `tf.tensor_scatter_nd_update`.

### Math and reductions

| Function | Description |
|---|---|
| `tf.add`, `tf.subtract`, `tf.multiply`, `tf.divide` | Elementwise arithmetic (operators `+ - * /` also work). |
| `tf.matmul(a, b, transpose_a=False, transpose_b=False)` | Matrix multiply (batched for rank > 2). Operator `@`. |
| `tf.einsum(equation, *inputs)` | Einstein summation, e.g. `"bij,bjk->bik"`. |
| `tf.reduce_sum(x, axis=None, keepdims=False)` | Sum. Similarly `reduce_mean`, `reduce_max`, `reduce_min`, `reduce_prod`, `reduce_any`, `reduce_all`, `reduce_logsumexp`. |
| `tf.argmax(x, axis=None)` / `tf.argmin` | Index of max/min. |
| `tf.math.top_k(x, k=1, sorted=True)` | Top-k values and indices. |
| `tf.sort(x, axis=-1, direction="ASCENDING")` / `tf.argsort` | Sorting. |
| `tf.math.exp`, `log`, `sqrt`, `square`, `abs`, `pow`, `sigmoid`, `tanh` | Elementwise math. |
| `tf.math.cumsum(x, axis=0)` | Cumulative sum. |
| `tf.math.unsorted_segment_sum(data, segment_ids, num_segments)` | Group-by sums. |
| `tf.clip_by_value(t, clip_value_min, clip_value_max)` | Clip values. |
| `tf.clip_by_global_norm(t_list, clip_norm)` | Gradient clipping. |
| `tf.linalg.inv`, `det`, `svd`, `eigh`, `qr`, `cholesky`, `solve`, `norm` | Linear algebra. |
| `tf.nn.relu`, `tf.nn.softmax`, `tf.nn.log_softmax`, `tf.nn.gelu` | Activations. |
| `tf.nn.softmax_cross_entropy_with_logits(labels, logits)` | Loss from logits. |
| `tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)` | Integer labels variant. |
| `tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)` | Binary loss from logits. |
| `tf.nn.conv2d(input, filters, strides, padding)` | Low-level convolution. |
| `tf.nn.dropout(x, rate, seed=None)` | Dropout. |
| `tf.nn.l2_normalize(x, axis)` | Normalize vectors. |

```python
logits = tf.constant([[2.0, 1.0, 0.1]])
labels = tf.constant([0])
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
print(loss.numpy(), tf.nn.softmax(logits).numpy())
```

### Ragged, sparse, and string tensors

```python
rt = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
print(rt.shape, rt.row_lengths())       # (3, None)
print(rt.to_tensor(default_value=0))    # pad to dense

st = tf.sparse.SparseTensor(indices=[[0, 1], [2, 3]], values=[10, 20], dense_shape=[3, 4])
print(tf.sparse.to_dense(st))

words = tf.strings.split(tf.constant(["hello world", "tf data"]))  # RaggedTensor of tokens
print(words)
print(tf.strings.lower("ABC"), tf.strings.to_number("3.5"), tf.strings.length("abc"))
```

| Function | Description |
|---|---|
| `tf.ragged.constant(pylist)` | Create a ragged tensor. |
| `tf.RaggedTensor.from_row_lengths(values, row_lengths)` | Build from flat values. |
| `rt.to_tensor(default_value)` | Pad to dense. |
| `tf.sparse.SparseTensor(indices, values, dense_shape)` | COO sparse tensor. |
| `tf.sparse.to_dense`, `tf.sparse.from_dense`, `tf.sparse.sparse_dense_matmul` | Sparse ops. |
| `tf.strings.split`, `join`, `regex_replace`, `substr`, `to_number`, `to_hash_bucket_fast` | String ops. |

### Variables

```python
tf.Variable(initial_value, trainable=True, name=None, dtype=None, shape=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `initial_value` | tensor-like or callable | required | Initial contents. |
| `trainable` | bool | True | Whether `GradientTape` watches it automatically and it appears in `trainable_variables`. |
| `name` | str | None | Name used in checkpoints. |
| `dtype` | DType | inferred | Data type. |
| `shape` | TensorShape | from value | Use `tf.TensorShape(None)` to allow shape changes on assign. |

Methods: `assign(value)`, `assign_add(delta)`, `assign_sub(delta)`, `read_value()`, `numpy()`, `scatter_nd_update(indices, updates)`.

```python
step = tf.Variable(0, trainable=False, dtype=tf.int64)
step.assign_add(1)
```

### tf.GradientTape

```python
tf.GradientTape(persistent=False, watch_accessed_variables=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `persistent` | bool | False | Allow multiple `gradient` calls (free with `del tape`). |
| `watch_accessed_variables` | bool | True | Auto-watch trainable variables read in the context. |

Methods:

| Method | Description |
|---|---|
| `tape.watch(tensor)` | Track a constant tensor (variables are tracked automatically). |
| `tape.gradient(target, sources, output_gradients=None, unconnected_gradients=tf.UnconnectedGradients.NONE)` | Gradients of `target` with respect to `sources` (tensor, variable, list or nested structure). Returns `None` for unconnected sources unless `unconnected_gradients=tf.UnconnectedGradients.ZERO`. |
| `tape.jacobian(target, sources)` | Full Jacobian. |
| `tape.batch_jacobian(target, source)` | Per-example Jacobian. |
| `tape.stop_recording()` | Context manager to pause recording. |

```python
x = tf.constant([1.0, 2.0])
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = tf.reduce_sum(x ** 3)
    z = tf.reduce_sum(tf.sin(x))
print(tape.gradient(y, x))   # 3x^2
print(tape.gradient(z, x))   # cos(x)
del tape
```

Use `tf.stop_gradient(x)` to block gradient flow through a value, and `tf.custom_gradient` to define your own gradient.

### tf.function

```python
tf.function(func=None, input_signature=None, autograph=True, jit_compile=None,
            reduce_retracing=False, experimental_follow_type_hints=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `func` | callable | None | Function to compile (or use as decorator). |
| `input_signature` | list of `tf.TensorSpec` | None | Fixes the signature; prevents retracing and is required for some exports. |
| `autograph` | bool | True | Convert Python control flow on tensors into graph ops. |
| `jit_compile` | bool | None | Compile the whole function with XLA. |
| `reduce_retracing` | bool | False | Generalize shapes to reduce the number of traces. |

Returns: a `PolymorphicFunction` (callable). Useful methods: `get_concrete_function(*specs)`, `experimental_get_tracing_count()`.

```python
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)],
             jit_compile=True)
def predict(x):
    return tf.nn.softmax(x @ W + b)

W = tf.Variable(tf.random.normal([784, 10]) * 0.01)
b = tf.Variable(tf.zeros([10]))
print(predict(tf.random.normal([32, 784])).shape)
```

Debugging: `tf.config.run_functions_eagerly(True)` runs every `tf.function` eagerly so you can use breakpoints and `print`.

#### tf.TensorSpec

```python
tf.TensorSpec(shape, dtype=tf.float32, name=None)
```

Describes a tensor's shape and dtype. `None` dimensions match any size. Used in `input_signature`, `tf.data` `output_signature`, and SavedModel signatures.

### tf.data

#### Creating datasets

| Constructor | Signature | Use |
|---|---|---|
| `from_tensor_slices` | `tf.data.Dataset.from_tensor_slices(tensors)` | One element per slice along axis 0 (arrays, tuples, dicts). |
| `from_tensors` | `tf.data.Dataset.from_tensors(tensors)` | Single element containing the whole tensor. |
| `range` | `tf.data.Dataset.range(start, stop=None, step=1)` | Integer sequence. |
| `from_generator` | `tf.data.Dataset.from_generator(generator, output_signature=...)` | Wrap a Python generator. |
| `list_files` | `tf.data.Dataset.list_files(file_pattern, shuffle=None, seed=None)` | Glob file paths. |
| `TextLineDataset` | `tf.data.TextLineDataset(filenames, compression_type=None, buffer_size=None, num_parallel_reads=None)` | Lines of text files. |
| `TFRecordDataset` | `tf.data.TFRecordDataset(filenames, compression_type=None, buffer_size=None, num_parallel_reads=None)` | Serialized records. |
| `experimental.make_csv_dataset` | `tf.data.experimental.make_csv_dataset(file_pattern, batch_size, label_name=None, ...)` | Batched dicts from CSV files. |
| `zip` | `tf.data.Dataset.zip(datasets)` | Combine datasets elementwise. |
| `load` | `tf.data.Dataset.load(path)` | Load a dataset saved with `dataset.save(path)`. |

```python
import numpy as np
import tensorflow as tf

features = np.random.rand(100, 4).astype("float32")
labels = np.random.randint(0, 2, size=100)
ds = tf.data.Dataset.from_tensor_slices((features, labels))
print(ds.element_spec)

def gen():
    for i in range(5):
        yield np.full((i + 1,), i, dtype=np.int32), i

gds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(tf.TensorSpec(shape=(None,), dtype=tf.int32),
                      tf.TensorSpec(shape=(), dtype=tf.int32)),
)
```

#### Transformations

| Method | Signature | Description |
|---|---|---|
| `map` | `map(map_func, num_parallel_calls=None, deterministic=None)` | Apply a function to each element. Use `num_parallel_calls=tf.data.AUTOTUNE`. |
| `filter` | `filter(predicate)` | Keep elements where predicate is True. |
| `shuffle` | `shuffle(buffer_size, seed=None, reshuffle_each_iteration=None)` | Random order using a buffer. |
| `batch` | `batch(batch_size, drop_remainder=False, num_parallel_calls=None)` | Stack consecutive elements. |
| `padded_batch` | `padded_batch(batch_size, padded_shapes=None, padding_values=None, drop_remainder=False)` | Batch variable-length elements. |
| `ragged_batch` | `ragged_batch(batch_size, drop_remainder=False)` | Batch into RaggedTensors. |
| `unbatch` | `unbatch()` | Split batches into elements. |
| `repeat` | `repeat(count=None)` | Repeat (forever if `None`). |
| `take` / `skip` | `take(count)` / `skip(count)` | Slice the stream. |
| `cache` | `cache(filename="")` | Cache in memory or to a file after the first epoch. |
| `prefetch` | `prefetch(buffer_size)` | Overlap producer and consumer; use `tf.data.AUTOTUNE`. |
| `interleave` | `interleave(map_func, cycle_length=None, block_length=None, num_parallel_calls=None)` | Read many files concurrently. |
| `flat_map` | `flat_map(map_func)` | Map and flatten. |
| `window` | `window(size, shift=None, stride=1, drop_remainder=False)` | Sliding windows (nested datasets). |
| `bucket_by_sequence_length` | `bucket_by_sequence_length(element_length_func, bucket_boundaries, bucket_batch_sizes, ...)` | Group similar lengths to reduce padding. |
| `enumerate` | `enumerate(start=0)` | Add an index. |
| `shard` | `shard(num_shards, index)` | Take every n-th element (manual sharding). |
| `apply` | `apply(transformation_func)` | Apply a dataset-to-dataset function. |
| `save` | `save(path, compression=None, shard_func=None)` | Persist a dataset to disk. |
| `cardinality` | `cardinality()` | Number of elements, or `INFINITE`/`UNKNOWN` constants. |
| `as_numpy_iterator` | `as_numpy_iterator()` | Iterate as NumPy values. |
| `take_while`, `scan`, `rejection_resample`, `group_by_window` | | Advanced transformations. |

Canonical high-performance pipeline:

```python
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()                      # cache after expensive deterministic work
    .shuffle(10_000)              # shuffle after cache so each epoch differs
    .batch(128)
    .prefetch(AUTOTUNE)           # always last
)
```

Order of operations matters: `map` (deterministic) then `cache`, then `shuffle` (random augmentations go after `cache`), then `batch`, then `prefetch`.

### File I/O, TFRecord, and images

#### tf.io basics

| Function | Description |
|---|---|
| `tf.io.read_file(filename)` | Read whole file as a scalar string tensor. |
| `tf.io.write_file(filename, contents)` | Write a string tensor to a file. |
| `tf.io.decode_image(contents, channels=None, expand_animations=True)` | Decode JPEG/PNG/GIF/BMP. Set `expand_animations=False` to always get rank 3. |
| `tf.io.decode_jpeg(contents, channels=0)` / `tf.io.decode_png` | Format-specific decoders. |
| `tf.io.encode_jpeg(image)` / `tf.io.encode_png` | Encoders. |
| `tf.io.serialize_tensor(t)` / `tf.io.parse_tensor(s, out_type)` | Tensor to/from bytes. |
| `tf.io.gfile.glob`, `exists`, `makedirs`, `GFile` | File system API (also supports GCS paths like `gs://`). |

#### Writing and reading TFRecords

```python
import numpy as np
import tensorflow as tf

def _bytes(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def _int64(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

def _floats(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

with tf.io.TFRecordWriter("data.tfrecord") as w:
    for i in range(100):
        example = tf.train.Example(features=tf.train.Features(feature={
            "id": _int64(i),
            "name": _bytes(f"item-{i}".encode()),
            "vec": _floats(np.random.rand(4).tolist()),
        }))
        w.write(example.SerializeToString())

feature_spec = {
    "id": tf.io.FixedLenFeature([], tf.int64),
    "name": tf.io.FixedLenFeature([], tf.string),
    "vec": tf.io.FixedLenFeature([4], tf.float32),
}

def parse(record):
    return tf.io.parse_single_example(record, feature_spec)

ds = tf.data.TFRecordDataset(["data.tfrecord"]).map(parse, num_parallel_calls=tf.data.AUTOTUNE)
for ex in ds.take(2):
    print(ex["id"].numpy(), ex["name"].numpy(), ex["vec"].numpy())
```

| Feature spec | Use |
|---|---|
| `tf.io.FixedLenFeature(shape, dtype, default_value=None)` | Fixed shape fields. |
| `tf.io.VarLenFeature(dtype)` | Variable length; parsed as SparseTensor. |
| `tf.io.RaggedFeature(dtype, ...)` | Variable length parsed as RaggedTensor. |
| `tf.io.parse_example(serialized, features)` | Batched parsing (faster: batch first, then parse). |

#### tf.image

| Function | Description |
|---|---|
| `tf.image.resize(images, size, method="bilinear", preserve_aspect_ratio=False, antialias=False)` | Resize. |
| `tf.image.resize_with_pad(image, target_height, target_width)` | Resize keeping aspect ratio, pad. |
| `tf.image.central_crop(image, central_fraction)` | Center crop. |
| `tf.image.random_flip_left_right(image, seed=None)` | Random flip. Stateless variants: `tf.image.stateless_random_flip_left_right(image, seed)`. |
| `tf.image.random_brightness(image, max_delta)` / `random_contrast` | Color jitter. |
| `tf.image.convert_image_dtype(image, dtype)` | Convert and rescale (uint8 to float in [0, 1]). |
| `tf.image.per_image_standardization(image)` | Zero mean, unit variance. |
| `tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold=0.5)` | Detection post-processing. |

```python
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    return img / 255.0, label
```

### tf.distribute

#### Strategies

| Strategy | Use case |
|---|---|
| `tf.distribute.MirroredStrategy(devices=None, cross_device_ops=None)` | Synchronous data parallel on multiple GPUs in one machine. |
| `tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=None, communication_options=None)` | Synchronous data parallel across machines (configured via `TF_CONFIG`). |
| `tf.distribute.TPUStrategy(tpu_cluster_resolver)` | TPUs and TPU Pods. |
| `tf.distribute.OneDeviceStrategy(device)` | Single device; useful for testing strategy code. |
| `tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)` | Asynchronous training with parameter servers. |
| `tf.distribute.get_strategy()` | Current (or default no-op) strategy. |

Key attributes/methods:

| Member | Description |
|---|---|
| `strategy.scope()` | Context in which variables (models, optimizers, metrics) must be created. |
| `strategy.num_replicas_in_sync` | Number of replicas; scale global batch size by this. |
| `strategy.experimental_distribute_dataset(dataset)` | Split a batched dataset across replicas. |
| `strategy.distribute_datasets_from_function(dataset_fn)` | Per-worker input pipelines. |
| `strategy.run(fn, args=(), kwargs=None)` | Run `fn` on each replica. |
| `strategy.reduce(reduce_op, value, axis)` | Combine per-replica values (`tf.distribute.ReduceOp.SUM` or `MEAN`). |

With Keras `fit`, you only need the scope:

```python
import keras
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
print("Replicas:", strategy.num_replicas_in_sync)

with strategy.scope():
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(10),
    ])
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

global_batch = 64 * strategy.num_replicas_in_sync
# model.fit(train_ds.batch(global_batch), epochs=5)
```

Custom training loop with a strategy:

```python
GLOBAL_BATCH = 64 * strategy.num_replicas_in_sync

with strategy.scope():
    model = keras.Sequential([keras.Input(shape=(784,)), keras.layers.Dense(10)])
    optimizer = keras.optimizers.Adam()
    loss_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)

def compute_loss(labels, logits):
    per_example = loss_obj(labels, logits)
    return tf.nn.compute_average_loss(per_example, global_batch_size=GLOBAL_BATCH)

@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            loss = compute_loss(y, model(x, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    per_replica = strategy.run(step_fn, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica, axis=None)

# dist_ds = strategy.experimental_distribute_dataset(train_ds.batch(GLOBAL_BATCH))
# for batch in dist_ds: train_step(batch)
```

`tf.nn.compute_average_loss` divides by the global batch size so that summing across replicas gives the correct mean.

### Checkpoints

#### tf.train.Checkpoint and CheckpointManager

```python
tf.train.Checkpoint(root=None, **kwargs)
tf.train.CheckpointManager(checkpoint, directory, max_to_keep, keep_checkpoint_every_n_hours=None,
                           checkpoint_name="ckpt", step_counter=None, checkpoint_interval=None)
```

| Member | Description |
|---|---|
| `ckpt.save(file_prefix)` | Save; returns path. |
| `ckpt.write(file_prefix)` | Save without the save counter. |
| `ckpt.restore(save_path)` | Restore (deferred for variables not yet created). Returns a status object with `assert_consumed()`, `expect_partial()`. |
| `manager.save(checkpoint_number=None)` | Save and rotate old checkpoints. |
| `manager.latest_checkpoint` | Path of newest checkpoint or None. |
| `tf.train.latest_checkpoint(dir)` | Newest checkpoint in a directory. |

```python
model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
optimizer = keras.optimizers.Adam()
step = tf.Variable(0, dtype=tf.int64)

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step)
manager = tf.train.CheckpointManager(ckpt, "./ckpts", max_to_keep=3)

ckpt.restore(manager.latest_checkpoint)   # no-op if None
print("Restored from", manager.latest_checkpoint or "scratch")
step.assign_add(1)
manager.save()
```

Checkpoints store variable values only (no code). For Keras-only workflows prefer `model.save("model.keras")` or `keras.callbacks.ModelCheckpoint`/`BackupAndRestore`.

### SavedModel

#### tf.saved_model.save / load

```python
tf.saved_model.save(obj, export_dir, signatures=None, options=None)
tf.saved_model.load(export_dir, tags=None, options=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `obj` | `tf.Module` (or trackable) | required | Object whose variables and `tf.function`s are saved. |
| `export_dir` | str | required | Output directory (`saved_model.pb` + `variables/`). |
| `signatures` | concrete function or dict | None | Serving signatures, e.g. `{"serving_default": f.get_concrete_function(...)}`. |
| `options` | `tf.saved_model.SaveOptions` | None | Advanced options. |

`load` returns a trackable object exposing saved `tf.function`s, variables and `signatures`.

```python
class Scaler(tf.Module):
    def __init__(self):
        super().__init__()
        self.factor = tf.Variable(2.0)

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def __call__(self, x):
        return {"scaled": x * self.factor}

tf.saved_model.save(Scaler(), "scaler_model")
loaded = tf.saved_model.load("scaler_model")
print(loaded(tf.constant([1.0, 2.0])))
infer = loaded.signatures["serving_default"]
print(infer(x=tf.constant([3.0])))
```

Inspect from the command line:

```bash
saved_model_cli show --dir scaler_model --all
```

#### Exporting Keras 3 models to SavedModel

In Keras 3, `model.save()` only writes `.keras` (or legacy `.h5`) files. To produce a SavedModel for TF Serving or TFLite conversion use `model.export`:

```python
model.export("exported_model")                      # SavedModel with a serving endpoint
reloaded = tf.saved_model.load("exported_model")
print(reloaded.serve(tf.random.normal([1, 4])))
```

For multiple endpoints or custom signatures, use `keras.export.ExportArchive`.

To use an exported SavedModel as a layer in a new Keras model: `keras.layers.TFSMLayer("exported_model", call_endpoint="serving_default")`.

### TensorFlow Lite / LiteRT

#### tf.lite.TFLiteConverter

```python
tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=None, tags=None)
tf.lite.TFLiteConverter.from_keras_model(model)
tf.lite.TFLiteConverter.from_concrete_functions(funcs, trackable_obj=None)
```

| Attribute | Description |
|---|---|
| `optimizations` | `[tf.lite.Optimize.DEFAULT]` enables quantization. |
| `representative_dataset` | Generator yielding sample inputs, enables full integer quantization. |
| `target_spec.supported_ops` | e.g. `[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]` or add `tf.lite.OpsSet.SELECT_TF_OPS` for unsupported ops. |
| `target_spec.supported_types` | e.g. `[tf.float16]` for float16 quantization. |
| `inference_input_type` / `inference_output_type` | `tf.int8` or `tf.uint8` for fully integer I/O. |

`converter.convert()` returns the model as `bytes`.

```python
converter = tf.lite.TFLiteConverter.from_saved_model("exported_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]       # dynamic-range quantization
tflite_bytes = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_bytes)
```

Full integer quantization:

```python
import numpy as np

def representative_data():
    for _ in range(100):
        yield [np.random.rand(1, 4).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model("exported_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
int8_model = converter.convert()
```

#### Running a .tflite model

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

interpreter.set_tensor(inp["index"], np.random.rand(1, 4).astype(np.float32))
interpreter.invoke()
print(interpreter.get_tensor(out["index"]))
```

The on-device runtime is now developed as **LiteRT**. Recent TensorFlow releases warn that `tf.lite.Interpreter` is deprecated; the drop-in replacement is `from ai_edge_litert.interpreter import Interpreter` after `pip install ai-edge-litert`. The `.tflite` file format is unchanged.

### tf.config (devices and memory)

| Function | Description |
|---|---|
| `tf.config.list_physical_devices(device_type=None)` | Physical devices, e.g. `"GPU"`, `"CPU"`, `"TPU"`. |
| `tf.config.list_logical_devices(device_type=None)` | Logical devices after configuration. |
| `tf.config.set_visible_devices(devices, device_type=None)` | Restrict which devices TF uses (call before GPUs are initialized). |
| `tf.config.experimental.set_memory_growth(device, enable)` | Allocate GPU memory on demand instead of all at once. |
| `tf.config.set_logical_device_configuration(device, logical_devices)` | Split a GPU or cap memory with `tf.config.LogicalDeviceConfiguration(memory_limit=MB)`. |
| `tf.config.run_functions_eagerly(run_eagerly)` | Debug `tf.function`s eagerly. |
| `tf.config.threading.set_intra_op_parallelism_threads(n)` / `set_inter_op_parallelism_threads(n)` | CPU threading. |
| `tf.config.optimizer.set_jit(enabled)` | Global XLA auto-clustering. |
| `tf.config.experimental.enable_op_determinism()` | Deterministic ops (slower). |
| `tf.config.experimental.get_memory_info(device)` | Current and peak memory usage, e.g. `"GPU:0"`. |

```python
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Or cap GPU 0 to 4 GB:
# tf.config.set_logical_device_configuration(
#     gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
```

Environment alternative: `TF_FORCE_GPU_ALLOW_GROWTH=true`. Hide GPUs completely with `CUDA_VISIBLE_DEVICES=""`.

### Mixed precision

```python
import keras
keras.mixed_precision.set_global_policy("mixed_float16")    # GPUs with compute capability >= 7.0
# keras.mixed_precision.set_global_policy("mixed_bfloat16") # TPUs and recent GPUs/CPUs
```

Keep the final activation (e.g. softmax) in float32: `keras.layers.Activation("softmax", dtype="float32")`. In Keras 3 `Model.fit` handles loss scaling automatically for `mixed_float16`; in custom loops wrap the optimizer with `keras.optimizers.LossScaleOptimizer`.

### tf.summary and TensorBoard

| Function | Description |
|---|---|
| `tf.summary.create_file_writer(logdir)` | Create a writer. Use `with writer.as_default():`. |
| `tf.summary.scalar(name, data, step=None)` | Log a scalar. |
| `tf.summary.histogram(name, data, step=None)` | Log a histogram. |
| `tf.summary.image(name, data, step=None, max_outputs=3)` | Log images (`[k, h, w, c]`). |
| `tf.summary.text(name, data, step=None)` | Log text. |
| `tf.profiler.experimental.start(logdir)` / `stop()` | Profiling. |

```python
writer = tf.summary.create_file_writer("logs/run1")
with writer.as_default():
    for step in range(100):
        tf.summary.scalar("loss", 1.0 / (step + 1), step=step)
```

```bash
tensorboard --logdir logs
```

## Tutorials

### Tutorial 1: Linear regression from scratch with GradientTape

This shows the core mechanics with no Keras: variables, a loss, gradients, and an optimizer update inside a compiled `tf.function`.

```python
import tensorflow as tf

tf.random.set_seed(0)

# 1. Synthetic data: y = 3x - 2 + noise
N = 1000
X = tf.random.normal([N, 1])
y = 3.0 * X - 2.0 + tf.random.normal([N, 1], stddev=0.3)

# 2. Parameters
w = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))
lr = 0.1

# 3. One training step, compiled to a graph
@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        pred = xb @ w + b
        loss = tf.reduce_mean(tf.square(pred - yb))
    dw, db = tape.gradient(loss, [w, b])
    w.assign_sub(lr * dw)
    b.assign_sub(lr * db)
    return loss

# 4. Mini-batch loop over a tf.data pipeline
ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(N).batch(64)
for epoch in range(20):
    for xb, yb in ds:
        loss = train_step(xb, yb)
    if epoch % 5 == 0:
        tf.print("epoch", epoch, "loss", loss)

print("w =", w.numpy().ravel(), "b =", b.numpy())  # close to 3 and -2
```

Explanation:

1. Data is created as tensors; no NumPy needed.
2. `tf.Variable`s hold the trainable state.
3. Inside the tape, the forward pass is recorded; `tape.gradient` returns gradients in the same order as the sources list. `assign_sub` updates in place. The function is traced once per input signature (the last batch may have a different size and cause one extra trace).
4. `tf.print` prints at graph run time, unlike Python `print`.

### Tutorial 2: Image classification with tf.data and Keras (Fashion-MNIST)

```python
import tensorflow as tf
import keras
from keras import layers

# 1. Load data (NumPy arrays) and build tf.data pipelines
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train[..., None]   # add channel axis -> (60000, 28, 28, 1)
x_test = x_test[..., None]

AUTOTUNE = tf.data.AUTOTUNE
BATCH = 128

def normalize(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    return x, y

val_size = 5000
train_ds = (tf.data.Dataset.from_tensor_slices((x_train[val_size:], y_train[val_size:]))
            .map(normalize, num_parallel_calls=AUTOTUNE)
            .cache()
            .shuffle(20_000)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .batch(BATCH)
            .prefetch(AUTOTUNE))
val_ds = (tf.data.Dataset.from_tensor_slices((x_train[:val_size], y_train[:val_size]))
          .map(normalize, num_parallel_calls=AUTOTUNE).batch(BATCH).cache().prefetch(AUTOTUNE))
test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
           .map(normalize, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE))

# 2. Model
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(10),                       # logits
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 3. Train with callbacks
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    keras.callbacks.TensorBoard(log_dir="logs/fashion"),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

# 4. Evaluate and save
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)
model.save("fashion_cnn.keras")
model.export("fashion_cnn_savedmodel")   # for TF Serving / TFLite
```

Notes:

- Normalization runs before `cache()` (deterministic and cached); augmentation runs after, so each epoch sees fresh random flips.
- The model outputs logits; `from_logits=True` is more numerically stable than softmax plus cross-entropy.
- `restore_best_weights=True` returns the best epoch's weights, not the last.

### Tutorial 3: Custom training loop with metrics, tf.function and checkpoints

When you need full control (GANs, multiple optimizers, unusual losses), write the loop yourself.

```python
import tensorflow as tf
import keras
from keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60_000).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10),
])
optimizer = keras.optimizers.Adam(1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss = keras.metrics.Mean(name="train_loss")
train_acc = keras.metrics.SparseCategoricalAccuracy(name="train_acc")
test_acc = keras.metrics.SparseCategoricalAccuracy(name="test_acc")

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(ckpt, "./mnist_ckpts", max_to_keep=2)
ckpt.restore(manager.latest_checkpoint)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)          # training=True enables dropout
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(loss)
    train_acc.update_state(y, logits)

@tf.function
def test_step(x, y):
    test_acc.update_state(y, model(x, training=False))

writer = tf.summary.create_file_writer("logs/custom_loop")
for epoch in range(5):
    for m in (train_loss, train_acc, test_acc):
        m.reset_state()
    for x, y in train_ds:
        train_step(x, y)
    for x, y in test_ds:
        test_step(x, y)
    with writer.as_default():
        tf.summary.scalar("loss", train_loss.result(), step=epoch)
        tf.summary.scalar("test_acc", test_acc.result(), step=epoch)
    manager.save()
    print(f"epoch {epoch}: loss={float(train_loss.result()):.4f} "
          f"train_acc={float(train_acc.result()):.4f} test_acc={float(test_acc.result()):.4f}")
```

Key points:

- Keras metrics are stateful: call `update_state` per batch, `result()` at the end, `reset_state()` each epoch.
- `model(x, training=True)` toggles dropout/batch-norm behavior.
- The tape only needs to cover the forward pass and loss.
- Checkpoints let you resume after a crash; restoring from `None` is a no-op.

### Tutorial 4: Text classification with TextVectorization and a SavedModel that accepts raw strings

```python
import tensorflow as tf
import keras
from keras import layers

# 1. Load IMDB reviews as (text, label) pairs (pip install tensorflow-datasets)
import tensorflow_datasets as tfds

raw_train, raw_val, raw_test = tfds.load(
    "imdb_reviews", split=["train[:80%]", "train[80%:]", "test"], as_supervised=True)
raw_train, raw_val, raw_test = (d.batch(64) for d in (raw_train, raw_val, raw_test))

# 2. Vectorization layer learns a vocabulary from training text
max_tokens, seq_len = 20_000, 250
vectorize = layers.TextVectorization(max_tokens=max_tokens, output_mode="int",
                                     output_sequence_length=seq_len)
vectorize.adapt(raw_train.map(lambda text, label: text))

# 3. Apply vectorization inside tf.data (fast, runs on CPU in parallel)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train.map(lambda t, l: (vectorize(t), l), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_ds = raw_val.map(lambda t, l: (vectorize(t), l), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_ds = raw_test.map(lambda t, l: (vectorize(t), l), num_parallel_calls=AUTOTUNE)

# 4. Model on integer token IDs
model = keras.Sequential([
    keras.Input(shape=(seq_len,), dtype="int64"),
    layers.Embedding(max_tokens, 128),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.3),
    layers.Dense(1),   # logit
])
model.compile(optimizer="adam",
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(threshold=0.0)])
model.fit(train_ds, validation_data=val_ds, epochs=3)
print(model.evaluate(test_ds))

# 5. End-to-end model that takes raw strings, for serving
raw_input = keras.Input(shape=(1,), dtype="string")
x = vectorize(raw_input)            # (batch, seq_len) int64
probs = layers.Activation("sigmoid")(model(x))
e2e = keras.Model(raw_input, probs)
print(e2e.predict(tf.constant([["A wonderful, moving film."], ["Dull and far too long."]])))
e2e.export("sentiment_savedmodel")
```

Explanation:

- `tfds.load(..., as_supervised=True)` yields `(text, label)` tuples; split slicing creates a validation set from the training split. For your own folders of text files, `keras.utils.text_dataset_from_directory` does the same job.
- `adapt` computes the vocabulary once from training data only.
- Vectorizing inside `tf.data` keeps the GPU busy; wrapping the vectorizer into an end-to-end model makes deployment accept raw text.
- `BinaryAccuracy(threshold=0.0)` because the model outputs logits.
- `model.export` writes a SavedModel; string inputs work because `TextVectorization` is TensorFlow-backed.

## Performance & Best Practices

### Input pipeline

- Always end pipelines with `.prefetch(tf.data.AUTOTUNE)`.
- Use `num_parallel_calls=tf.data.AUTOTUNE` in `map` and `interleave`.
- `cache()` after expensive deterministic preprocessing, before random augmentation.
- Batch before cheap vectorized `map` functions (vectorized mapping is faster than per-element).
- Shard large datasets into many TFRecord files (100 to 200 MB each) and read with `interleave`.
- Profile with the TensorBoard Profiler; the "input pipeline analyzer" tells you if you are input bound.

### Graph compilation

- Wrap training steps in `tf.function`; Keras `fit` does this automatically.
- Avoid retracing: pass tensors (not Python numbers) to `tf.function`s, use `input_signature` or `reduce_retracing=True` for variable shapes.
- Try `jit_compile=True` (XLA) for compute-heavy models; Keras `compile(jit_compile="auto")` enables it where supported.
- Do not call `.numpy()` or Python-side logic inside `tf.function`.

### GPU utilization

- Use mixed precision (`mixed_float16` on NVIDIA Volta or newer, `mixed_bfloat16` on TPU/Ampere+).
- Use layer sizes and batch sizes that are multiples of 8 for Tensor Cores.
- Increase batch size until memory is near full; scale the learning rate accordingly.
- Enable memory growth if sharing the GPU with other processes.
- Use `MirroredStrategy` for multiple GPUs on one machine.

### Numerical stability and correctness

- Prefer losses with `from_logits=True` over applying softmax/sigmoid first.
- Use `tf.debugging.check_numerics` or `tf.debugging.enable_check_numerics()` to find the first NaN/Inf.
- Clip gradients with `tf.clip_by_global_norm` or the optimizer's `clipnorm`/`global_clipnorm`.
- Set seeds with `keras.utils.set_random_seed` and, if needed, `tf.config.experimental.enable_op_determinism()`.

### Deployment

- Export a SavedModel with explicit signatures for serving.
- Put preprocessing inside the exported model (e.g. `TextVectorization`, `Normalization`) to avoid training/serving skew.
- Quantize for mobile with TFLite/LiteRT; validate accuracy after conversion.

## Common Errors & Troubleshooting

### GPU libraries not found

```text
Could not load dynamic library 'libcudnn.so.8'; dlerror: libcudnn.so.8: cannot open shared object file: No such file or directory
W tensorflow/core/common_runtime/gpu/gpu_device.cc] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Skipping registering GPU devices...
```

Cause: CUDA/cuDNN not installed or versions do not match the TensorFlow build.

Fix: on Linux, `pip install "tensorflow[and-cuda]"` in a fresh virtual environment. Check `tf.sysconfig.get_build_info()` and `nvidia-smi`. On Windows, use WSL2 (native GPU support ended at 2.10).

### Out of memory

```text
ResourceExhaustedError: OOM when allocating tensor with shape[256,512,56,56] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
```

Fix: reduce batch size or input resolution, enable mixed precision, enable memory growth, and make sure no other process holds the GPU. Calling `keras.backend.clear_session()` between experiments in a notebook releases graph memory.

### dtype mismatch

```text
InvalidArgumentError: cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:AddV2]
```

Cause: TensorFlow does not auto-promote dtypes.

Fix: `tf.cast(x, tf.float32)`, or create constants with the right dtype (`tf.constant(1.0)` not `tf.constant(1)`). NumPy arrays default to float64; cast with `.astype("float32")`.

### Incompatible shapes

```text
InvalidArgumentError: Incompatible shapes: [32,10] vs. [32]
```

Cause: label/prediction shapes differ, often using `CategoricalCrossentropy` with integer labels or vice versa.

Fix: integer labels go with `SparseCategoricalCrossentropy`; one-hot labels with `CategoricalCrossentropy`. Print `ds.element_spec` and `model.output_shape`.

### Using a tensor as a Python bool

```text
OperatorNotAllowedInGraphError: Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function.
```

Cause: Python control flow on a tensor in a context AutoGraph cannot convert (for example inside a Keras layer called symbolically, or `assert tensor`).

Fix: use `tf.cond`, `tf.where`, or `tf.debugging.assert_*` functions; make sure the function is decorated with `tf.function` so AutoGraph converts `if`/`while`.

### Variables created inside tf.function

```text
ValueError: tf.function only supports singleton tf.Variables created on the first call. Make sure the tf.Variable is only created once or created outside tf.function.
```

Fix: create variables in `__init__` or `build`, or outside the function; guard creation with `if self.v is None:`.

### Excessive retracing

```text
WARNING:tensorflow:5 out of the last 5 calls to <function train_step at 0x...> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors.
```

Fix: define the `tf.function` once outside loops, pass tensors instead of Python scalars, use `input_signature` with `None` dimensions or `reduce_retracing=True`.

### No gradients

```text
ValueError: No gradients provided for any variable: ...
```

Cause: the loss is not connected to the variables (computed outside the tape, used `.numpy()` or NumPy ops in the forward pass, or passed the wrong variable list). In Keras `fit`, it can also happen when the model output is not used by the loss (e.g. wrong output name).

Fix: keep the whole forward pass in TensorFlow ops inside the tape; check `tape.gradient` returns non-`None` values.

### TF1 APIs in TF2

```text
AttributeError: module 'tensorflow' has no attribute 'placeholder'
AttributeError: module 'tensorflow' has no attribute 'Session'
```

Cause: TF 1.x code.

Fix: rewrite to eager code with `tf.function`, or as a stopgap use `tf.compat.v1.placeholder` with `tf.compat.v1.disable_eager_execution()`.

### Keras 3 changes in TF 2.16+

```text
ValueError: Invalid filepath extension for saving. Please add either a `.keras` extension for the native Keras format (recommended) or a `.h5` extension.
```

Cause: `model.save("my_model")` (SavedModel directory) is no longer supported by Keras 3.

Fix: `model.save("my_model.keras")` for Keras, `model.export("my_model")` for SavedModel. If an old codebase depends on Keras 2 behavior, `pip install tf_keras` and set `TF_USE_LEGACY_KERAS=1` before importing TensorFlow.

Other Keras 3 migration errors include `ImportError` for `keras.layers.experimental.preprocessing` (layers now live directly in `keras.layers`) and the deprecated `ImageDataGenerator` (replace with `keras.utils.image_dataset_from_directory` and preprocessing layers).

### oneDNN and CPU instruction messages

```text
I tensorflow/core/util/port.cc] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I tensorflow/core/platform/cpu_feature_guard.cc] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
```

These are informational, not errors. Hide with `TF_CPP_MIN_LOG_LEVEL=2`.

## Interoperability

### NumPy

Tensors convert to and from NumPy arrays automatically; most NumPy functions accept tensors, and `tensor.numpy()` returns an array in eager mode. Watch out for float64 defaults in NumPy.

### pandas

```python
import pandas as pd
import tensorflow as tf

df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "label": [0, 1]})
labels = df.pop("label")
ds = tf.data.Dataset.from_tensor_slices((dict(df), labels.values)).batch(2)
for features, y in ds:
    print(features["a"], y)
```

### Keras 3 with other backends

Keras 3 models written with `keras.ops` and Keras layers can run on JAX or PyTorch by setting `KERAS_BACKEND`. `tf.data` pipelines can feed Keras models on any backend: `model.fit(tf_dataset)` works with the JAX and PyTorch backends too.

### PyTorch

There is no direct model converter in TensorFlow. Common paths:

- Data: convert via NumPy (`tensor.numpy()` to `torch.from_numpy`) or DLPack: `tf.experimental.dlpack.to_dlpack(t)` and `torch.utils.dlpack.from_dlpack(capsule)`.
- Models: export PyTorch to ONNX, or use Keras 3 with the PyTorch backend, or re-implement and port weights.

### ONNX

`tf2onnx` converts SavedModels to ONNX:

```bash
pip install tf2onnx
python -m tf2onnx.convert --saved-model exported_model --output model.onnx --opset 17
```

## Cheat Sheet

### Tensors

| Task | Code |
|---|---|
| Constant | `tf.constant([[1., 2.], [3., 4.]])` |
| Zeros / ones | `tf.zeros([2, 3])`, `tf.ones_like(x)` |
| Random normal | `tf.random.normal([3, 3], seed=1)` |
| Seed everything | `keras.utils.set_random_seed(42)` |
| Cast | `tf.cast(x, tf.float32)` |
| Reshape | `tf.reshape(x, [-1, 28 * 28])` |
| Add axis | `tf.expand_dims(x, -1)` or `x[..., None]` |
| Concatenate | `tf.concat([a, b], axis=0)` |
| Stack | `tf.stack([a, b], axis=1)` |
| Select rows | `tf.gather(x, [0, 2])` |
| Conditional | `tf.where(x > 0, x, 0.0)` |
| One-hot | `tf.one_hot(labels, depth=10)` |
| Matrix multiply | `a @ b` or `tf.matmul(a, b)` |
| Reduce | `tf.reduce_mean(x, axis=1)` |
| Argmax | `tf.argmax(logits, axis=-1)` |
| To NumPy | `x.numpy()` |
| Dynamic shape | `tf.shape(x)[0]` |

### Autodiff and graphs

| Task | Code |
|---|---|
| Gradient | `with tf.GradientTape() as t: y = f(x)` then `t.gradient(y, x)` |
| Watch constant | `t.watch(x)` |
| Multiple gradients | `tf.GradientTape(persistent=True)` |
| Stop gradient | `tf.stop_gradient(x)` |
| Compile function | `@tf.function` |
| Fixed signature | `@tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])` |
| XLA | `@tf.function(jit_compile=True)` |
| Debug eagerly | `tf.config.run_functions_eagerly(True)` |
| Runtime print | `tf.print(x)` |

### tf.data

| Task | Code |
|---|---|
| From arrays | `tf.data.Dataset.from_tensor_slices((x, y))` |
| From generator | `Dataset.from_generator(gen, output_signature=...)` |
| Map in parallel | `ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)` |
| Shuffle | `ds.shuffle(10_000)` |
| Batch | `ds.batch(64, drop_remainder=True)` |
| Pad batch | `ds.padded_batch(32)` |
| Cache | `ds.cache()` |
| Prefetch | `ds.prefetch(tf.data.AUTOTUNE)` |
| Read TFRecord | `tf.data.TFRecordDataset(files)` |
| Parse example | `tf.io.parse_single_example(rec, spec)` |
| Inspect | `ds.element_spec`, `next(iter(ds))` |
| Save / load | `ds.save(path)`, `tf.data.Dataset.load(path)` |

### Devices and distribution

| Task | Code |
|---|---|
| List GPUs | `tf.config.list_physical_devices("GPU")` |
| Memory growth | `tf.config.experimental.set_memory_growth(gpu, True)` |
| Force CPU | `with tf.device("/CPU:0"):` |
| Multi-GPU | `with tf.distribute.MirroredStrategy().scope(): model = ...` |
| Mixed precision | `keras.mixed_precision.set_global_policy("mixed_float16")` |
| Deterministic ops | `tf.config.experimental.enable_op_determinism()` |

### Saving and deployment

| Task | Code |
|---|---|
| Keras save | `model.save("m.keras")` |
| Keras load | `keras.models.load_model("m.keras")` |
| SavedModel from Keras | `model.export("dir")` |
| SavedModel from Module | `tf.saved_model.save(module, "dir")` |
| Load SavedModel | `tf.saved_model.load("dir")` |
| Checkpoint | `tf.train.Checkpoint(model=m, optimizer=o).save("ckpt/x")` |
| TFLite convert | `tf.lite.TFLiteConverter.from_saved_model("dir").convert()` |
| Quantize | `converter.optimizations = [tf.lite.Optimize.DEFAULT]` |
| TensorBoard | `tensorboard --logdir logs` |

## Further Resources

- Official site and guides: https://www.tensorflow.org/
- Install guide (pip, GPU, WSL2): https://www.tensorflow.org/install/pip
- API reference: https://www.tensorflow.org/api_docs/python/tf
- Guide to tf.function: https://www.tensorflow.org/guide/function
- tf.data performance guide: https://www.tensorflow.org/guide/data_performance
- Distributed training guide: https://www.tensorflow.org/guide/distributed_training
- SavedModel guide: https://www.tensorflow.org/guide/saved_model
- LiteRT (formerly TensorFlow Lite): https://ai.google.dev/edge/litert
- GitHub repository: https://github.com/tensorflow/tensorflow
- Release notes: https://github.com/tensorflow/tensorflow/releases
- Keras 3 documentation: https://keras.io/
- TensorFlow Datasets: https://www.tensorflow.org/datasets
- TensorBoard: https://www.tensorflow.org/tensorboard
- Paper: Abadi et al., "TensorFlow: A System for Large-Scale Machine Learning", OSDI 2016. https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi
- Course: "Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning" (DeepLearning.AI, Coursera). https://www.coursera.org/learn/introduction-tensorflow
