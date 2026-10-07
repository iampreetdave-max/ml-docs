# JAX

> Composable transformations of NumPy programs: differentiate, vectorize, JIT-compile to GPU and TPU.

JAX is a Python library for high-performance array computing and machine learning research. It pairs a NumPy-compatible API (`jax.numpy`) with a set of composable function transformations: `jax.grad` for automatic differentiation, `jax.jit` for compilation with XLA, `jax.vmap` for automatic vectorization, and sharding-based parallelism for scaling across many accelerators. Neural network libraries such as Flax and optimizer libraries such as Optax are built on top of it.

Covers JAX 0.4.30+ through the 0.7 series, Flax 0.10+ (NNX and Linen), and Optax 0.2+. Notable changes called out on this page include the typed key API `jax.random.key` (preferred over the legacy `jax.random.PRNGKey`), the `jax.tree` module (alias for common `jax.tree_util` functions), `jax.make_mesh` and `NamedSharding` as the primary scaling model (replacing most uses of `pmap`), `shard_map` moving from `jax.experimental` to `jax.shard_map`, and Flax NNX becoming the recommended Flax API with an `nnx.Optimizer(model, tx, wrt=nnx.Param)` signature in Flax 0.11+.

## Overview

### What it is

JAX lets you write numerical code in a functional, NumPy-like style and then transform it:

- **`jax.numpy`** (`jnp`): almost the full NumPy API, running on CPU, GPU, or TPU.
- **`jax.grad`, `jax.value_and_grad`, `jax.jacfwd`, `jax.jacrev`, `jax.hessian`, `jax.jvp`, `jax.vjp`**: forward- and reverse-mode automatic differentiation of arbitrary order.
- **`jax.jit`**: traces a Python function into an intermediate representation (a *jaxpr*) and compiles it with **XLA** into fused, optimized kernels.
- **`jax.vmap`**: turns a function written for one example into one that processes a batch, without manual batching code.
- **Sharding** (`jax.sharding.Mesh`, `NamedSharding`, `PartitionSpec`): annotate how arrays are laid out across devices, and the compiler inserts the necessary communication (GSPMD).
- **`jax.random`**: explicit, splittable, reproducible pseudo-random number generation.
- **`jax.lax`**: lower-level primitives, including structured control flow (`cond`, `scan`, `while_loop`) that compiles.

### History and maintainers

JAX was created at Google Research (now Google DeepMind) and open-sourced in late 2018. It grew out of the Autograd project (Maclaurin, Duvenaud, Johnson) combined with the XLA compiler used by TensorFlow. The paper "Compiling machine learning programs via high-level tracing" (SysML 2018) describes the original design. It is developed in the open on GitHub under the `jax-ml` organization and is used to train large models at Google DeepMind (Gemini), Anthropic, xAI, Apple, and many research groups. The surrounding ecosystem includes **Flax** (neural networks), **Optax** (optimizers), **Orbax** (checkpointing), **Grain** (data loading), **Equinox** (PyTorch-like modules), and **Keras 3** (which can use JAX as a backend).

### When to use it

- You need **TPUs**, or want first-class multi-device and multi-host scaling with compiler-driven sharding.
- Research that benefits from **composable transforms**: per-example gradients, higher-order derivatives, meta-learning, implicit differentiation, Jacobians.
- Scientific computing: differentiable simulators, physics, probabilistic programming (NumPyro, BlackJAX), optimization (JAXopt/Optimistix).
- Large-scale LLM training where XLA compilation and sharding annotations give excellent hardware utilization.

### When not to use it

- You depend on a large set of PyTorch-only libraries, checkpoints, or deployment tools (most of the open model ecosystem is PyTorch-first).
- Highly dynamic programs whose shapes change every step: JAX recompiles for each new shape, so ragged or data-dependent control flow is painful.
- Code that relies on in-place mutation, Python side effects inside compiled functions, or stateful objects; JAX expects pure functions.
- Windows GPU support: GPU wheels are Linux-only (Windows users use WSL2).

### Where it fits in the ML stack

```text
Models / research code
        |
High-level libraries:  Flax (NNX, Linen), Equinox, Keras 3, Haiku (maintenance), MaxText, Levanter
Training utilities:    Optax (optimizers), Orbax (checkpointing), Grain / tf.data / TFDS (data)
        |
JAX core:  jax.numpy, grad / jit / vmap, jax.random, jax.lax, sharding
        |
Compiler:  XLA (via jaxlib / PJRT plugins); custom kernels via Pallas
        |
Hardware:  TPU, NVIDIA GPU (CUDA), AMD GPU (ROCm), CPU
```

## Installation

### pip

JAX is split into `jax` (pure Python) and `jaxlib` (compiled XLA runtime), plus optional hardware plugins. Install with the extra that matches your hardware:

```bash
# CPU only (Linux, macOS, Windows)
pip install -U jax

# NVIDIA GPU, CUDA 12 (Linux x86_64 / aarch64). CUDA and cuDNN come as pip wheels.
pip install -U "jax[cuda12]"

# NVIDIA GPU with CUDA 13 (newer releases)
pip install -U "jax[cuda13]"

# Google Cloud TPU VM
pip install -U "jax[tpu]"

# AMD GPU (ROCm) - follow the ROCm instructions for the exact plugin version
pip install -U "jax[rocm]"   # or use the AMD-provided docker images / wheels
```

You need a recent NVIDIA driver (check the JAX install page for the minimum version for each CUDA major). The `cuda12` extra pulls in `jax-cuda12-plugin` and the NVIDIA runtime libraries; you do not need a system CUDA toolkit. To use a locally installed CUDA instead, use `"jax[cuda12-local]"`.

### Ecosystem packages

```bash
pip install -U flax optax orbax-checkpoint
```

### conda

```bash
conda create -n jax python=3.11 -y
conda activate jax
pip install -U "jax[cuda12]"        # recommended: pip inside conda

# or the community conda-forge build
conda install -c conda-forge jax
```

### Apple Silicon and Windows

- macOS: `pip install -U jax` gives a fast CPU build. An experimental Metal plugin (`jax-metal`) exists but lags behind and is not recommended for serious work.
- Windows: CPU wheels are available natively; for NVIDIA GPU support use WSL2 with the Linux instructions.

### Verifying the install and GPU/TPU visibility

```python
import jax
import jax.numpy as jnp

print(jax.__version__)             # e.g. 0.7.0
print(jax.default_backend())       # 'gpu', 'tpu', or 'cpu'
print(jax.devices())               # [CudaDevice(id=0), ...] or [TpuDevice(...), ...]
print(jax.device_count(), jax.local_device_count())

x = jnp.ones((1000, 1000))
print(x.devices())                  # where the array lives
print((x @ x).block_until_ready()[0, 0])   # forces execution

jax.print_environment_info()        # versions of jax, jaxlib, numpy, python, and nvidia-smi output
```

If JAX cannot see your GPU, it logs a warning such as `An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.` See Troubleshooting.

### Useful environment variables

| Variable | Effect |
|---|---|
| `JAX_PLATFORMS=cpu` | Force a backend (`cpu`, `cuda`, `tpu`). |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Do not preallocate 75 percent of GPU memory at startup. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=.50` | Change the preallocated fraction. |
| `JAX_ENABLE_X64=1` | Enable 64-bit floats and ints. |
| `XLA_FLAGS=--xla_force_host_platform_device_count=8` | Simulate 8 CPU devices (for testing sharding). |
| `JAX_TRACEBACK_FILTERING=off` | Show full internal tracebacks. |
| `JAX_COMPILATION_CACHE_DIR=/path` | Persistent compilation cache across runs. |

## Core Concepts

### 1. jax.numpy and immutable arrays

`jax.numpy` mirrors NumPy, but JAX arrays (`jax.Array`) are **immutable** and may live on an accelerator. Instead of in-place assignment, use the functional `.at[]` syntax, which returns a new array (inside `jit`, XLA usually performs it in place anyway).

```python
import jax.numpy as jnp
import numpy as np

x = jnp.arange(10.0)
# x[0] = 100.0   # TypeError: JAX arrays are immutable
y = x.at[0].set(100.0)
z = y.at[2:5].add(1.0)
w = z.at[jnp.array([1, 3])].multiply(10.0)
print(x[0], y[0], w)

# NumPy interop
a = np.asarray(w)        # device -> host NumPy array
b = jnp.asarray(a)       # host -> default device
```

Other differences from NumPy:

- Default dtypes are 32-bit (`float32`, `int32`) unless `jax_enable_x64` is set.
- Out-of-bounds indexing does not raise; reads are clamped and out-of-bounds updates are dropped.
- Random numbers live in `jax.random` with explicit keys, not global state.
- Operations dispatch **asynchronously**; use `.block_until_ready()` when timing.

### 2. Pure functions

JAX transformations assume **functionally pure** functions: outputs depend only on inputs, with no side effects. Side effects (printing, appending to a global list, mutating objects) run only once, during *tracing*, and are not part of the compiled program.

```python
import jax
import jax.numpy as jnp

counter = []

@jax.jit
def impure(x):
    counter.append(1)          # side effect: runs only at trace time
    print("tracing!")          # prints once per compilation
    return x * 2

impure(jnp.ones(3))
impure(jnp.ones(3))            # cached: no print, no append
print(len(counter))            # 1
impure(jnp.ones(4))            # new shape -> retrace -> prints again
```

Use `jax.debug.print` to print runtime values inside compiled code.

### 3. Tracing and static vs traced values

When you call a jitted function, JAX replaces array arguments with **tracers** carrying only shape and dtype (an abstract value such as `f32[3]`). It records the operations into a jaxpr, compiles it, and caches the result keyed on the input shapes/dtypes and static arguments.

Consequences:

- Python control flow on traced *values* fails (`if x > 0:` with a traced `x`), because the value is unknown at trace time. Use `jnp.where`, `jax.lax.cond`, or mark the argument static.
- Python control flow on *shapes* and static arguments is fine (shapes are known at trace time).
- New shapes or new static values trigger recompilation.

```python
import jax
import jax.numpy as jnp
from functools import partial

print(jax.make_jaxpr(lambda x: jnp.sin(x) * 2)(1.0))
# { lambda ; a:f32[]. let b:f32[] = sin a; c:f32[] = mul b 2.0 in (c,) }

@partial(jax.jit, static_argnames="n")
def power(x, n):
    for _ in range(n):         # fine: n is static (a Python int)
        x = x * x
    return x

print(power(jnp.float32(1.1), n=3))
```

### 4. Transformations compose

`grad`, `jit`, and `vmap` are higher-order functions that take a function and return a new one; they can be nested arbitrarily.

```python
import jax
import jax.numpy as jnp

def loss(w, x, y):
    return jnp.mean((x @ w - y) ** 2)

grad_loss = jax.grad(loss)                              # d loss / d w
per_example = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))   # per-example gradients
fast = jax.jit(per_example)

w = jnp.ones(3)
x = jnp.ones((8, 1, 3))      # 8 examples, each a (1, 3) batch
y = jnp.zeros((8, 1))
print(fast(w, x, y).shape)   # (8, 3)

# Higher-order derivatives
d2 = jax.grad(jax.grad(jnp.tanh))
print(d2(0.5))
```

### 5. Explicit, splittable random keys

JAX has no global random state. You create a **key**, and each random function consumes one. To get more randomness, **split** the key. Never reuse a key for two different random draws you expect to be independent.

```python
import jax

key = jax.random.key(0)                 # new-style typed key (recommended)
key, sub = jax.random.split(key)
x = jax.random.normal(sub, (3,))
k1, k2, k3 = jax.random.split(key, 3)

legacy = jax.random.PRNGKey(0)          # legacy uint32[2] key; still supported
print(key.dtype, legacy.dtype)          # key<fry> uint32
```

`jax.random.key` returns a scalar array with a special key dtype; `jax.random.PRNGKey` returns a raw `uint32[2]` array. Both work with all `jax.random` functions; the typed key is safer (it cannot be accidentally used in arithmetic) and is the documented recommendation. Convert with `jax.random.key_data(key)` and `jax.random.wrap_key_data(raw)`.

### 6. Pytrees

A **pytree** is any nested structure of lists, tuples, dicts, `None`, and registered classes whose leaves are arrays. All JAX transformations accept pytrees, which is how model parameters are represented as nested dicts.

```python
import jax
import jax.numpy as jnp

params = {"dense1": {"w": jnp.ones((3, 4)), "b": jnp.zeros(4)},
          "dense2": {"w": jnp.ones((4, 1)), "b": jnp.zeros(1)}}

print(jax.tree.map(lambda a: a.shape, params))
leaves, treedef = jax.tree.flatten(params)
print(len(leaves), treedef)
n_params = sum(x.size for x in jax.tree.leaves(params))
scaled = jax.tree.map(lambda p, g: p - 0.1 * g, params, params)   # SGD-style update
```

### 7. Devices, async dispatch, and sharding

Arrays are placed on devices; computations run where their inputs live. With multiple devices, a `jax.sharding.Mesh` names device axes, and a `NamedSharding(mesh, PartitionSpec(...))` describes how each array dimension is split across those axes. Passing sharded inputs to a `jit`-compiled function lets XLA automatically partition the computation and insert collectives (see Tutorial 4).

Dispatch is **asynchronous**: a JAX call returns as soon as the work is enqueued, and the device computes in the background. Converting to NumPy, printing, or calling `.block_until_ready()` waits for the result, so always use the latter when benchmarking.

64-bit types are disabled by default; enable them at program start with `jax.config.update("jax_enable_x64", True)` (or `JAX_ENABLE_X64=1`).

## API Reference

Signatures show the commonly used arguments. Import conventions used throughout:

```python
import jax
import jax.numpy as jnp
from jax import lax, random
```

### jax.numpy

`jax.numpy` implements most of the NumPy API with the same names and semantics. The table lists the most-used functions; anything not listed generally behaves as in NumPy.

| Category | Functions |
|---|---|
| Creation | `array`, `asarray`, `zeros`, `ones`, `full`, `empty`, `eye`, `identity`, `arange`, `linspace`, `zeros_like`, `ones_like`, `full_like`, `meshgrid`, `tri`, `tril`, `triu` |
| Shape | `reshape`, `transpose`, `swapaxes`, `moveaxis`, `expand_dims`, `squeeze`, `ravel`, `concatenate`, `stack`, `hstack`, `vstack`, `split`, `tile`, `repeat`, `broadcast_to`, `pad` |
| Math | `add`, `multiply`, `exp`, `log`, `log1p`, `sqrt`, `power`, `abs`, `sin`, `cos`, `tanh`, `maximum`, `minimum`, `clip`, `round`, `floor` |
| Linear algebra | `dot`, `matmul` (`@`), `einsum`, `tensordot`, `outer`, `inner`, `vdot`, `linalg.norm`, `linalg.inv`, `linalg.solve`, `linalg.svd`, `linalg.eigh`, `linalg.cholesky`, `linalg.qr`, `linalg.det` |
| Reductions | `sum`, `mean`, `std`, `var`, `max`, `min`, `argmax`, `argmin`, `prod`, `cumsum`, `cumprod`, `any`, `all`, `median`, `percentile` |
| Logic / selection | `where`, `select`, `take`, `take_along_axis`, `argsort`, `sort`, `nonzero` (needs `size=` under jit), `unique` (needs `size=` under jit), `isnan`, `isfinite`, `allclose` |

#### jnp.array / jnp.asarray

```python
jnp.array(object, dtype=None, copy=True, order="K", ndmin=0) -> jax.Array
jnp.asarray(a, dtype=None, order=None, copy=None) -> jax.Array
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `object` / `a` | array-like | required | Python scalars, lists, NumPy arrays, JAX arrays. |
| `dtype` | dtype | `None` | Inferred; floats default to `float32` without x64. |
| `copy` | `bool` | `True` / `None` | Whether to force a copy. |

Returns a `jax.Array` on the default device.

```python
import jax.numpy as jnp
a = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
b = jnp.linspace(0.0, 1.0, 5)
c = jnp.einsum("ij,jk->ik", a, a)
d = jnp.where(a > 2, a, 0.0)
```

#### The .at property (functional updates)

```python
x.at[idx].set(values, *, indices_are_sorted=False, unique_indices=False, mode=None)
x.at[idx].add(values) / .multiply(values) / .divide(values) / .power(values)
x.at[idx].min(values) / .max(values)
x.at[idx].get(mode=None, fill_value=None)
```

| `mode` value | Behavior for out-of-bounds indices |
|---|---|
| `"promise_in_bounds"` | Assume valid; undefined otherwise (fastest). |
| `"clip"` | Clamp indices into range. |
| `"drop"` | Ignore out-of-bounds updates (default for updates). |
| `"fill"` | For `get`: return `fill_value` (default NaN for floats). |

Returns a new array. `.at[...].add` accumulates duplicates, making it the JAX equivalent of `np.add.at` (scatter-add).

```python
import jax.numpy as jnp
counts = jnp.zeros(5).at[jnp.array([0, 1, 1, 4])].add(1.0)
print(counts)        # [1. 2. 0. 0. 1.]
```

### Automatic differentiation

#### jax.grad

```python
jax.grad(fun, argnums=0, has_aux=False, holomorphic=False, allow_int=False) -> Callable
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fun` | callable | required | Function returning a **scalar** (or `(scalar, aux)` with `has_aux`). |
| `argnums` | `int` or tuple of `int` | `0` | Which positional arguments to differentiate with respect to. |
| `has_aux` | `bool` | `False` | `fun` returns `(output, aux)`; grad returns `(grads, aux)`. |
| `holomorphic` | `bool` | `False` | For complex holomorphic functions. |
| `allow_int` | `bool` | `False` | Allow integer inputs (gradient is float0 zeros). |

Returns a function computing the gradient, with the same pytree structure as the selected argument(s).

```python
import jax
import jax.numpy as jnp

def loss(params, x, y):
    pred = x @ params["w"] + params["b"]
    return jnp.mean((pred - y) ** 2)

params = {"w": jnp.ones((3,)), "b": 0.0}
x, y = jnp.ones((4, 3)), jnp.zeros(4)
g = jax.grad(loss)(params, x, y)
print(g["w"].shape, g["b"])
gw, gx = jax.grad(loss, argnums=(0, 1))(params, x, y)   # grads w.r.t. params and x
```

#### jax.value_and_grad

```python
jax.value_and_grad(fun, argnums=0, has_aux=False, holomorphic=False, allow_int=False) -> Callable
```

Returns a function giving `(value, grad)`, or `((value, aux), grad)` with `has_aux=True`. This is the standard choice in training steps (one forward pass yields both loss and gradients).

```python
import jax
import jax.numpy as jnp

def loss_with_metrics(w, x, y):
    pred = x @ w
    loss = jnp.mean((pred - y) ** 2)
    return loss, {"mae": jnp.mean(jnp.abs(pred - y))}

(loss, metrics), grads = jax.value_and_grad(loss_with_metrics, has_aux=True)(
    jnp.ones(3), jnp.ones((5, 3)), jnp.zeros(5))
```

#### Jacobians, Hessians, JVP/VJP

```python
jax.jacfwd(fun, argnums=0, has_aux=False, holomorphic=False) -> Callable   # forward mode, good for tall Jacobians
jax.jacrev(fun, argnums=0, has_aux=False, holomorphic=False, allow_int=False) -> Callable   # reverse mode, good for wide
jax.hessian(fun, argnums=0, has_aux=False, holomorphic=False) -> Callable  # jacfwd(jacrev(fun))
jax.jvp(fun, primals, tangents, has_aux=False) -> (primals_out, tangents_out)
jax.vjp(fun, *primals, has_aux=False) -> (primals_out, vjp_fun)
jax.linearize(fun, *primals) -> (primals_out, jvp_fun)
```

#### Stopping gradients and custom rules

```python
jax.lax.stop_gradient(x) -> Array                # treat x as a constant for differentiation
jax.custom_jvp(fun, nondiff_argnums=())          # decorator; define rule with fun.defjvp
jax.custom_vjp(fun, nondiff_argnums=())          # decorator; define rule with fun.defvjp(fwd, bwd)
```

#### jax.checkpoint (rematerialization)

```python
jax.checkpoint(fun, *, prevent_cse=True, policy=None, static_argnums=()) -> Callable   # alias: jax.remat
```

Saves memory by recomputing intermediate activations during the backward pass instead of storing them. `policy` (e.g. `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`) controls what is saved.

### Compilation: jax.jit

```python
jax.jit(fun, in_shardings=..., out_shardings=..., static_argnums=None, static_argnames=None,
        donate_argnums=None, donate_argnames=None, keep_unused=False, backend=None,
        inline=False) -> Callable
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fun` | callable | required | Pure function of arrays/pytrees. Usable as a decorator. |
| `static_argnums` | `int` or sequence | `None` | Positional args treated as compile-time constants (must be hashable). |
| `static_argnames` | `str` or sequence | `None` | Keyword args treated as static. |
| `donate_argnums` / `donate_argnames` | `int`/`str` or sequence | `None` | Inputs whose buffers may be reused for outputs (saves memory; the donated arrays become invalid). |
| `in_shardings` / `out_shardings` | `Sharding` pytree | unspecified | Required/forced layouts of inputs/outputs across devices. |
| `keep_unused` | `bool` | `False` | Keep unused arguments in the compiled computation. |
| `backend` | `str` | `None` | Deprecated in favor of placing inputs on devices. |

Returns a compiled wrapper with the same signature. The first call per (shape, dtype, static values) signature traces and compiles; later calls hit the cache.

```python
import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1_000_000.0)
selu(x).block_until_ready()          # compile + run
selu(x).block_until_ready()          # cached

@partial(jax.jit, static_argnames=("training",), donate_argnames=("state",))
def step(state, x, training: bool):
    scale = 0.5 if training else 1.0  # Python branch on a static value is fine
    return state + scale * x

# Ahead-of-time lowering and inspection
lowered = jax.jit(selu).lower(x)
compiled = lowered.compile()
print(lowered.as_text()[:300])        # StableHLO
print(compiled.cost_analysis())       # FLOPs / bytes estimates (backend dependent)
```

`jax.disable_jit()` is a context manager that runs everything eagerly, which helps debugging.

### Vectorization and parallelism

#### jax.vmap

```python
jax.vmap(fun, in_axes=0, out_axes=0, axis_name=None, axis_size=None, spmd_axis_name=None) -> Callable
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fun` | callable | required | Function written for a single example. |
| `in_axes` | `int`, `None`, or pytree | `0` | Which axis of each input to map over; `None` broadcasts that input. |
| `out_axes` | `int`, `None`, or pytree | `0` | Where the mapped axis appears in outputs. |
| `axis_name` | hashable | `None` | Name for collectives (`lax.psum(x, axis_name)`) inside `fun`. |
| `axis_size` | `int` | `None` | Needed only if no input is mapped. |

Returns a batched function.

```python
import jax
import jax.numpy as jnp

def predict(w, x):           # x: single example (features,)
    return jnp.tanh(w @ x)

w = jnp.ones((4, 3))
xs = jnp.ones((32, 3))
batched = jax.vmap(predict, in_axes=(None, 0))
print(batched(w, xs).shape)  # (32, 4)

# Pairwise distances via nested vmap
dist = jax.vmap(jax.vmap(lambda a, b: jnp.linalg.norm(a - b), (None, 0)), (0, None))
print(dist(xs[:5], xs[:7]).shape)     # (5, 7)

# Map over axis 1 of the input, put the result on axis 1 of the output
print(jax.vmap(jnp.sum, in_axes=1, out_axes=0)(jnp.ones((2, 6))).shape)   # (6,)
```

#### jax.pmap (legacy)

```python
jax.pmap(fun, axis_name=None, in_axes=0, out_axes=0, static_broadcasted_argnums=(),
         devices=None, donate_argnums=()) -> Callable
```

`pmap` compiles `fun` and runs one copy per device, mapping over the leading axis (whose size must equal the number of local devices). It is still supported but **`jax.jit` with sharded inputs, or `shard_map`, is now the recommended approach** for multi-device code.

### Random numbers: jax.random

| Function | Signature | Description |
|---|---|---|
| `key` | `key(seed, *, impl=None)` | New-style typed key (recommended). |
| `PRNGKey` | `PRNGKey(seed, *, impl=None)` | Legacy raw `uint32[2]` key. |
| `split` | `split(key, num=2)` | Return `num` new independent keys (array of keys). |
| `fold_in` | `fold_in(key, data)` | Derive a key from an integer (e.g. step or process index). |
| `normal` | `normal(key, shape=(), dtype=float)` | Standard normal. |
| `uniform` | `uniform(key, shape=(), dtype=float, minval=0.0, maxval=1.0)` | Uniform. |
| `randint` | `randint(key, shape, minval, maxval, dtype=int)` | Integers in `[minval, maxval)`. |
| `bernoulli` | `bernoulli(key, p=0.5, shape=None)` | Booleans. |
| `categorical` | `categorical(key, logits, axis=-1, shape=None)` | Sample from logits. |
| `choice` | `choice(key, a, shape=(), replace=True, p=None, axis=0)` | Sample elements. |
| `permutation` | `permutation(key, x, axis=0, independent=False)` | Shuffle or random permutation of `range(x)`. |
| `truncated_normal` | `truncated_normal(key, lower, upper, shape=None, dtype=float)` | Truncated normal. |
| `gumbel`, `beta`, `gamma`, `dirichlet`, `multivariate_normal`, `exponential`, `laplace`, `poisson` | | Other distributions. |
| `key_data` / `wrap_key_data` | | Convert between typed keys and raw `uint32` data. |

```python
import jax
import jax.numpy as jnp

key = jax.random.key(42)
key, k_init, k_drop, k_perm = jax.random.split(key, 4)
w = jax.random.normal(k_init, (128, 64)) * jnp.sqrt(2 / 128)
mask = jax.random.bernoulli(k_drop, p=0.9, shape=(64,))
idx = jax.random.permutation(k_perm, 1000)            # shuffled indices 0..999
tokens = jax.random.categorical(key, jnp.log(jnp.array([0.1, 0.6, 0.3])), shape=(5,))

# Per-step keys without carrying state
step_key = jax.random.fold_in(jax.random.key(0), 1234)

# Many keys at once
keys = jax.random.split(key, 8)                       # shape (8,) array of keys
samples = jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys)   # (8, 3)
```

### Control flow: jax.lax

Python `if`/`for`/`while` over traced values do not compile. `lax` provides compiled alternatives.

#### lax.cond and lax.switch

```python
jax.lax.cond(pred, true_fun, false_fun, *operands) -> Any
jax.lax.switch(index, branches, *operands) -> Any
```

| Parameter | Type | Description |
|---|---|---|
| `pred` | scalar bool | Selects the branch. |
| `true_fun`, `false_fun` | callables | Must return outputs with identical structure, shapes, and dtypes. |
| `index` | scalar int | Branch index (clamped into range). |
| `branches` | sequence of callables | Candidate branches. |

Under `vmap`, `cond` with a batched predicate is converted to `select`, so both branches execute.

#### lax.fori_loop and lax.while_loop

```python
jax.lax.fori_loop(lower, upper, body_fun, init_val, *, unroll=None) -> Any   # body_fun(i, val) -> val
jax.lax.while_loop(cond_fun, body_fun, init_val) -> Any                      # cond_fun(val) -> bool
```

`while_loop` is not reverse-mode differentiable (forward-mode only). `fori_loop` with static bounds lowers to `scan` and is differentiable.

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def newton_sqrt(a):
    def cond(state):
        x, i = state
        return (jnp.abs(x * x - a) > 1e-6) & (i < 50)
    def body(state):
        x, i = state
        return 0.5 * (x + a / x), i + 1
    x, _ = lax.while_loop(cond, body, (a, 0))
    return x

print(newton_sqrt(2.0))
print(lax.fori_loop(0, 10, lambda i, acc: acc + i, 0))   # 45
```

#### lax.scan

```python
jax.lax.scan(f, init, xs=None, length=None, reverse=False, unroll=1) -> (carry, ys)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `f` | callable | required | `f(carry, x) -> (new_carry, y)`. |
| `init` | pytree | required | Initial carry. Shapes/dtypes must stay constant. |
| `xs` | pytree of arrays | `None` | Sliced along the leading axis, one slice per step. |
| `length` | `int` | `None` | Number of steps if `xs` is `None`. |
| `reverse` | `bool` | `False` | Scan from the end. |
| `unroll` | `int` or `bool` | `1` | Unroll factor (trades compile time for speed). |

Returns the final carry and the stacked per-step outputs. `scan` compiles the body once, so it is the standard way to write RNNs, layer stacks, and long loops (compile time stays constant regardless of length).

```python
import jax
import jax.numpy as jnp
from jax import lax

def cumulative_sum(xs):
    def step(carry, x):
        carry = carry + x
        return carry, carry
    return lax.scan(step, 0.0, xs)

total, running = cumulative_sum(jnp.arange(5.0))
print(total, running)         # 10.0 [ 0.  1.  3.  6. 10.]
```

#### Other lax primitives

| Function | Description |
|---|---|
| `lax.dynamic_slice(operand, start_indices, slice_sizes)` | Slice with traced start indices (static sizes). |
| `lax.dynamic_update_slice(operand, update, start_indices)` | Write a block at traced offsets (e.g. KV cache). |
| `lax.dot_general(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None)` | General contraction. |
| `lax.conv_general_dilated(lhs, rhs, window_strides, padding, ...)` | General convolution. |
| `lax.select(pred, on_true, on_false)` | Elementwise select (all same shape). |
| `lax.map(f, xs, batch_size=None)` | Sequential map (lower memory than vmap). |
| `lax.associative_scan(fn, elems)` | Parallel prefix scan. |
| `lax.psum`, `lax.pmean`, `lax.pmax`, `lax.all_gather`, `lax.ppermute`, `lax.axis_index` | Collectives over a named axis (inside `vmap`/`pmap`/`shard_map`). |
| `lax.stop_gradient(x)` | Block gradients. |
| `lax.rsqrt`, `lax.erf`, `lax.top_k(operand, k)` | Numerics. |

### Pytree utilities: jax.tree and jax.tree_util

The `jax.tree` module (JAX 0.4.25+) offers short aliases for the most common `jax.tree_util` functions.

| Function | Description |
|---|---|
| `jax.tree.map(f, tree, *rest, is_leaf=None)` | Apply `f` leaf-wise across one or more trees with the same structure. |
| `jax.tree.leaves(tree, is_leaf=None)` | List of leaves. |
| `jax.tree.structure(tree)` | The `PyTreeDef`. |
| `jax.tree.flatten(tree)` / `jax.tree.unflatten(treedef, leaves)` | Round-trip. |
| `jax.tree.reduce(f, tree, initializer)` | Fold over leaves. |
| `jax.tree.transpose(outer, inner, tree)` | Swap nesting levels. |
| `jax.tree_util.tree_map_with_path(f, tree)` | `f(path, leaf)`; useful for masks such as "no weight decay on biases". |
| `jax.tree_util.keystr(path)` | Human-readable path string. |
| `jax.tree_util.register_pytree_node(cls, flatten, unflatten)` | Register a custom container. |
| `jax.tree_util.register_dataclass(cls, data_fields, meta_fields)` | Register a dataclass (fields can also be inferred). |

### jax.nn and jax.scipy

| Function | Description |
|---|---|
| `jax.nn.relu`, `gelu(x, approximate=True)`, `silu`/`swish`, `sigmoid`, `softplus`, `elu`, `leaky_relu`, `tanh` (via jnp) | Activations. |
| `jax.nn.softmax(x, axis=-1)`, `log_softmax(x, axis=-1)` | Normalized exponentials. |
| `jax.nn.logsumexp(a, axis=None)` | Stable log-sum-exp (also `jax.scipy.special.logsumexp`). |
| `jax.nn.one_hot(x, num_classes, dtype=float, axis=-1)` | One-hot encoding. |
| `jax.nn.standardize(x, axis=-1)` | Zero-mean unit-variance. |
| `jax.nn.dot_product_attention(query, key, value, bias=None, mask=None, *, is_causal=False, implementation=None)` | Fused attention (cuDNN/XLA). Shapes `(B, T, N, H)`. |
| `jax.nn.initializers.glorot_uniform()`, `he_normal()`, `lecun_normal()`, `normal(stddev)`, `zeros`, `ones` | Initializer factories: `init(key, shape, dtype)`. |
| `jax.scipy.special.erf`, `gammaln`, `logit`, `expit` | Special functions. |
| `jax.scipy.stats.norm.logpdf`, ... | Distributions. |
| `jax.scipy.linalg.solve_triangular`, `cho_solve`, `expm` | Linear algebra. |
| `jax.scipy.optimize.minimize(fun, x0, method="BFGS")` | Simple optimizer. |

### Devices and memory

| Function | Description |
|---|---|
| `jax.devices(backend=None)` | All devices (across hosts). |
| `jax.local_devices()` | Devices attached to this process. |
| `jax.device_count()` / `jax.local_device_count()` | Counts. |
| `jax.process_index()` / `jax.process_count()` | Multi-host identity. |
| `jax.device_put(x, device_or_sharding=None, *, donate=False)` | Transfer / reshard a pytree. |
| `jax.device_get(x)` | Fetch a pytree to host NumPy arrays. |
| `jax.block_until_ready(x)` / `x.block_until_ready()` | Wait for async computation. |
| `x.devices()`, `x.sharding` | Where an array lives. |
| `jax.clear_caches()` | Drop compilation caches. |
| `jax.distributed.initialize()` | Multi-host (multi-process) setup; auto-configures on TPU pods and SLURM. |
| `jax.live_arrays()` | Arrays currently alive (memory debugging). |

### Sharding and multi-device computation

#### Mesh, PartitionSpec, NamedSharding

```python
jax.sharding.Mesh(devices, axis_names)                    # devices: ndarray of devices
jax.make_mesh(axis_shapes, axis_names, *, devices=None)   # JAX 0.4.35+ convenience constructor
jax.sharding.PartitionSpec(*partitions)                   # usually imported as P
jax.sharding.NamedSharding(mesh, spec)
```

| Concept | Meaning |
|---|---|
| `Mesh` | N-dimensional grid of devices with named axes, e.g. `("data", "model")`. |
| `P("data", None)` | Split array dim 0 across the `data` mesh axis; replicate dim 1. |
| `P(None, "model")` | Split dim 1 across `model`. |
| `P(("data", "model"))` | Split dim 0 across both axes jointly. |
| `P()` | Fully replicated. |

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = np.array(jax.devices())
mesh = Mesh(devices.reshape(-1, 1), ("data", "model"))   # e.g. (8, 1) on 8 GPUs
batch = jax.device_put(jnp.ones((64, 512)), NamedSharding(mesh, P("data", None)))
weights = jax.device_put(jnp.ones((512, 256)), NamedSharding(mesh, P(None, "model")))

@jax.jit
def layer(x, w):
    return jax.nn.relu(x @ w)

out = layer(batch, weights)
print(out.sharding)            # NamedSharding(... spec=PartitionSpec('data', 'model'))
```

#### with_sharding_constraint

```python
jax.lax.with_sharding_constraint(x, shardings) -> Array
```

Inside `jit`, pins the layout of an intermediate value, guiding the compiler's partitioning decisions.

#### shard_map

```python
jax.shard_map(f, *, mesh, in_specs, out_specs, check_vma=True)          # newer releases
jax.experimental.shard_map.shard_map(f, mesh, in_specs, out_specs, check_rep=True)   # older releases
```

`shard_map` runs `f` per device on its local shard (SPMD, like `pmap` but mesh-aware), letting you write explicit collectives. Use it when automatic partitioning does not produce the communication pattern you want.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

try:
    from jax import shard_map                          # newer JAX
except ImportError:
    from jax.experimental.shard_map import shard_map   # older JAX

mesh = Mesh(np.array(jax.devices()), ("data",))

def local_mean(x):                         # x is this device's shard
    return jax.lax.pmean(x.mean(keepdims=True), "data")

f = jax.jit(shard_map(local_mean, mesh=mesh, in_specs=P("data"), out_specs=P()))
print(f(jnp.arange(8.0 * len(jax.devices()))))
```

### Debugging utilities

| Tool | Use |
|---|---|
| `jax.debug.print("x={x}", x=x)` | Print runtime values inside `jit`/`vmap`/`grad`. |
| `jax.debug.breakpoint()` | Interactive breakpoint in compiled code. |
| `jax.debug.visualize_array_sharding(x)` | Show how a 2-D array is sharded. |
| `jax.make_jaxpr(f)(*args)` | Show the traced program. |
| `jax.disable_jit()` | Run eagerly with real values. |
| `jax.config.update("jax_debug_nans", True)` | Raise at the first op producing NaN. |
| `jax.config.update("jax_log_compiles", True)` | Log every compilation (find recompiles). |
| `jax.profiler.trace("/tmp/trace")` | Context manager writing a TensorBoard/Perfetto profile. |
| `checkify.checkify(f)` (`from jax.experimental import checkify`) | Functional runtime assertions. |

### Flax NNX (recommended Flax API)

Flax NNX (`from flax import nnx`) represents models as regular Python objects with mutable state, similar to PyTorch modules, while remaining compatible with JAX transforms via `nnx.jit`, `nnx.grad`, `nnx.value_and_grad`, `nnx.vmap`, `nnx.scan`, and `nnx.remat`.

Key classes and functions:

| API | Description |
|---|---|
| `nnx.Module` | Base class; assign layers and `nnx.Variable`s as attributes in `__init__`. |
| `nnx.Rngs(seed, **streams)` | Holds named RNG streams (e.g. `params`, `dropout`); falls back to `default`. |
| `nnx.Linear(in_features, out_features, *, use_bias=True, dtype=None, param_dtype=float32, kernel_init=..., rngs)` | Dense layer. |
| `nnx.Conv(in_features, out_features, kernel_size, strides=1, padding="SAME", *, rngs)` | Convolution (NHWC inputs). |
| `nnx.Embed(num_embeddings, features, *, rngs)` | Embedding table. |
| `nnx.LayerNorm(num_features, *, rngs)`, `nnx.RMSNorm(num_features, *, rngs)` | Normalization. |
| `nnx.BatchNorm(num_features, *, momentum=0.99, epsilon=1e-5, use_running_average=False, rngs)` | Batch norm with `nnx.BatchStat` state. |
| `nnx.Dropout(rate, *, rngs)` | Dropout; disabled in `model.eval()`. |
| `nnx.MultiHeadAttention(num_heads, in_features, *, qkv_features=None, decode=None, rngs)` | Attention block. |
| `nnx.Param`, `nnx.BatchStat`, `nnx.Variable`, `nnx.Cache` | State containers. |
| `nnx.Optimizer(model, tx, wrt=nnx.Param)` | Wraps an Optax transform; `optimizer.update(model, grads)` (Flax 0.11+). |
| `nnx.jit`, `nnx.value_and_grad`, `nnx.grad`, `nnx.vmap`, `nnx.scan`, `nnx.remat` | Transform wrappers that understand NNX objects. |
| `nnx.split(model)` / `nnx.merge(graphdef, state)` | Convert between objects and pure (graphdef, state) pytrees. |
| `nnx.state(model, filter)` / `nnx.update(model, state)` | Extract / write state. |
| `nnx.eval_shape(fn)` | Build an abstract model without allocating memory. |
| `nnx.MultiMetric`, `nnx.metrics.Accuracy`, `nnx.metrics.Average` | Metric accumulators. |
| `model.train()` / `model.eval()` | Toggle Dropout and BatchNorm behavior. |
| `nnx.relu`, `nnx.gelu`, `nnx.softmax`, ... | Re-exported activations. |

Version note: before Flax 0.11, the signature was `nnx.Optimizer(model, tx)` and updates were `optimizer.update(grads)`. Flax 0.11 added the required `wrt` filter and changed the call to `optimizer.update(model, grads)`.

```python
import jax.numpy as jnp
import optax
from flax import nnx

class MLP(nnx.Module):
    def __init__(self, din: int, dhidden: int, dout: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.drop = nnx.Dropout(0.1, rngs=rngs)
        self.fc2 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x):
        return self.fc2(self.drop(nnx.relu(self.fc1(x))))

model = MLP(4, 32, 3, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)          # mutates model parameters in place
    return loss

x = jnp.ones((16, 4))
y = jnp.zeros((16,), dtype=jnp.int32)
print(train_step(model, optimizer, x, y))
model.eval()                                # disable dropout for inference
print(model(x).shape)                       # (16, 3)
```

### Flax Linen (classic API)

Linen (`import flax.linen as nn`) uses immutable dataclass modules and explicit `init`/`apply`. It remains widely used in existing code (many checkpoints, MaxText history, Hugging Face Flax models).

```python
flax.linen.Module                     # dataclass-style; define fields as class attributes
@nn.compact                           # define submodules inline in __call__
module.init(rngs, *args, **kwargs) -> variables          # {'params': ..., 'batch_stats': ...}
module.apply(variables, *args, rngs=None, mutable=False, **kwargs) -> output or (output, updates)
```

| `apply` parameter | Type | Default | Description |
|---|---|---|---|
| `variables` | dict | required | Collections such as `{'params': ..., 'batch_stats': ...}`. |
| `rngs` | dict of keys | `None` | e.g. `{'dropout': key}` for stochastic layers. |
| `mutable` | bool, str, list | `False` | Collections allowed to change (e.g. `['batch_stats']`); then returns `(out, updated_vars)`. |

Common layers: `nn.Dense(features)`, `nn.Conv(features, kernel_size)`, `nn.Embed(num_embeddings, features)`, `nn.LayerNorm()`, `nn.RMSNorm()`, `nn.BatchNorm(use_running_average=...)`, `nn.Dropout(rate, deterministic=...)`, `nn.MultiHeadDotProductAttention(num_heads)`, `nn.scan`, `nn.remat`, pooling `nn.max_pool` / `nn.avg_pool`.

`flax.training.train_state.TrainState` bundles `apply_fn`, `params`, `tx`, and `opt_state`, with `state.apply_gradients(grads=grads)` for updates.

### Optax (optimizers and losses)

Optax optimizers are `GradientTransformation`s: pairs of pure functions `init(params) -> opt_state` and `update(grads, opt_state, params=None) -> (updates, new_opt_state)`.

#### Core optimizers

```python
optax.sgd(learning_rate, momentum=None, nesterov=False)
optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None, nesterov=False)
optax.adamw(learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None,
            weight_decay=0.0001, mask=None, nesterov=False)
optax.lion(learning_rate, b1=0.9, b2=0.99, weight_decay=0.001, mask=None)
optax.adafactor(learning_rate=None, ...)
optax.lamb(learning_rate, ...), optax.rmsprop(learning_rate, ...), optax.adagrad(learning_rate, ...)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` or schedule | required | Constant or `step -> lr` callable. |
| `b1`, `b2` | `float` | `0.9`, `0.999` | Moment decay rates. |
| `weight_decay` | `float` | `1e-4` (adamw) | Decoupled weight decay. |
| `mask` | pytree of bools or callable | `None` | Which params get weight decay. |
| `mu_dtype` | dtype | `None` | Store first moment in lower precision (e.g. bfloat16). |

#### Combinators and utilities

| API | Description |
|---|---|
| `optax.chain(*transforms)` | Compose transforms in order. |
| `optax.clip_by_global_norm(max_norm)` | Gradient clipping. |
| `optax.add_decayed_weights(weight_decay, mask=None)` | Weight decay as a separate transform. |
| `optax.apply_updates(params, updates)` | `params + updates`, leaf-wise. |
| `optax.MultiSteps(opt, every_k_schedule)` | Gradient accumulation over k micro-batches. |
| `optax.inject_hyperparams(optimizer_fn)(learning_rate=...)` | Make hyperparameters part of the state (inspect or change LR). |
| `optax.masked(inner, mask)` / `optax.multi_transform(transforms, param_labels)` | Different optimizers per parameter group. |
| `optax.ema(decay)` | Exponential moving average of updates. |

#### Schedules

| Schedule | Signature |
|---|---|
| `optax.constant_schedule(value)` | Constant. |
| `optax.linear_schedule(init_value, end_value, transition_steps, transition_begin=0)` | Linear ramp. |
| `optax.cosine_decay_schedule(init_value, decay_steps, alpha=0.0)` | Cosine decay. |
| `optax.warmup_cosine_decay_schedule(init_value, peak_value, warmup_steps, decay_steps, end_value=0.0)` | Linear warmup then cosine (decay_steps includes warmup). |
| `optax.exponential_decay(init_value, transition_steps, decay_rate)` | Exponential. |
| `optax.join_schedules(schedules, boundaries)` | Piecewise. |

#### Losses

| Loss | Description |
|---|---|
| `optax.softmax_cross_entropy_with_integer_labels(logits, labels)` | Multi-class with int labels; per-example output. |
| `optax.softmax_cross_entropy(logits, labels)` | With one-hot / soft labels. |
| `optax.sigmoid_binary_cross_entropy(logits, labels)` | Binary / multi-label. |
| `optax.l2_loss(predictions, targets)` | 0.5 * squared error. |
| `optax.squared_error(predictions, targets)` | Squared error. |
| `optax.huber_loss(predictions, targets, delta=1.0)` | Robust regression. |
| `optax.cosine_similarity(predictions, targets)` | Similarity. |
| `optax.smooth_labels(labels, alpha)` | Label smoothing for one-hot labels. |

These are also available under `optax.losses`. All return per-element values; take `.mean()` yourself.

```python
import jax
import jax.numpy as jnp
import optax

params = {"w": jnp.zeros((3,)), "b": jnp.zeros(())}
schedule = optax.warmup_cosine_decay_schedule(0.0, 1e-2, warmup_steps=100, decay_steps=1000)
tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
opt_state = tx.init(params)

def loss_fn(p, x, y):
    return jnp.mean(optax.l2_loss(x @ p["w"] + p["b"], y))

@jax.jit
def step(p, s, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
    updates, s = tx.update(grads, s, p)
    return optax.apply_updates(p, updates), s, loss

x, y = jnp.ones((32, 3)), jnp.ones(32)
for _ in range(5):
    params, opt_state, loss = step(params, opt_state, x, y)
print(loss)
```

## Tutorials

### Tutorial 1: Logistic regression from scratch in pure JAX

This example uses only `jax` and `jax.numpy`: parameter initialization with explicit keys, a loss function, `value_and_grad`, `jit`, mini-batching with `jax.random.permutation`, and plain gradient descent written with `jax.tree.map`.

```python
import jax
import jax.numpy as jnp

# 1. Synthetic, linearly separable-ish data.
key = jax.random.key(0)
key, k_x, k_w, k_noise = jax.random.split(key, 4)
n, d = 2000, 5
X = jax.random.normal(k_x, (n, d))
true_w = jax.random.normal(k_w, (d,))
y = (X @ true_w + 0.3 * jax.random.normal(k_noise, (n,)) > 0).astype(jnp.float32)

# 2. Parameters as a pytree.
params = {"w": jnp.zeros(d), "b": jnp.array(0.0)}

# 3. A pure loss function: binary cross-entropy computed from logits for stability.
def loss_fn(params, x, y):
    logits = x @ params["w"] + params["b"]
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -jnp.mean(y * log_p + (1 - y) * log_not_p)

# 4. One jitted SGD step. value_and_grad returns the loss and grads in one pass.
@jax.jit
def sgd_step(params, x, y, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, loss

@jax.jit
def accuracy(params, x, y):
    return jnp.mean(((x @ params["w"] + params["b"]) > 0) == (y == 1))

# 5. Training loop with a fresh shuffle each epoch.
batch_size = 100
for epoch in range(20):
    key, k_perm = jax.random.split(key)
    perm = jax.random.permutation(k_perm, n)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        params, loss = sgd_step(params, X[idx], y[idx], 0.5)
    if epoch % 5 == 0:
        print(f"epoch {epoch}: loss {loss:.4f} acc {accuracy(params, X, y):.3f}")

# 6. Compare learned direction with the true one.
cos = jnp.dot(params["w"], true_w) / (jnp.linalg.norm(params["w"]) * jnp.linalg.norm(true_w))
print("cosine(learned, true) =", float(cos))
```

Explanation:

1. Every random draw uses its own sub-key from `jax.random.split`.
2. Parameters are a dict, so `jax.grad` returns gradients with the same dict structure.
3. `log_sigmoid` avoids `log(0)` for confident predictions.
4. The update is written functionally (new params returned), which is what lets `jit` compile the whole step.
5. Each batch has the same shape (2000 is divisible by 100), so the step compiles only once.
6. The learned weight vector should point in nearly the same direction as `true_w` (cosine near 1).

### Tutorial 2: MLP classifier with Flax NNX and Optax, plus checkpointing

Classify the scikit-learn handwritten digits (8x8 images) with an NNX model, track metrics with `nnx.MultiMetric`, and save/restore with Orbax.

```bash
pip install -U jax flax optax orbax-checkpoint scikit-learn
```

```python
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1. Data (NumPy on host; JAX moves batches to the device when used).
digits = load_digits()
X = (digits.data / 16.0).astype(np.float32)          # (1797, 64) in [0, 1]
y = digits.target.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 2. Model.
class MLP(nnx.Module):
    def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.bn = nnx.BatchNorm(dhidden, rngs=rngs)
        self.drop = nnx.Dropout(0.2, rngs=rngs)
        self.fc2 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.bn(self.fc1(x)))
        return self.fc2(self.drop(x))

model = MLP(64, 128, 10, rngs=nnx.Rngs(0))
nnx.display(model)                                   # prints the module tree and shapes

# 3. Optimizer and metrics.
steps_per_epoch = len(X_train) // 64
schedule = optax.cosine_decay_schedule(3e-3, decay_steps=30 * steps_per_epoch)
optimizer = nnx.Optimizer(model, optax.adamw(schedule, weight_decay=1e-4), wrt=nnx.Param)
metrics = nnx.MultiMetric(accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss"))

def loss_fn(model, x, y):
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, logits

@nnx.jit
def train_step(model, optimizer, metrics, x, y):
    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, x, y)
    metrics.update(loss=loss, logits=logits, labels=y)
    optimizer.update(model, grads)

@nnx.jit
def eval_step(model, metrics, x, y):
    loss, logits = loss_fn(model, x, y)
    metrics.update(loss=loss, logits=logits, labels=y)

# 4. Training loop.
rng = np.random.default_rng(0)
for epoch in range(30):
    model.train()                                    # dropout on, batch stats updated
    perm = rng.permutation(len(X_train))
    for i in range(steps_per_epoch):
        idx = perm[i * 64:(i + 1) * 64]
        train_step(model, optimizer, metrics, X_train[idx], y_train[idx])
    train_m = metrics.compute()
    metrics.reset()

    model.eval()                                     # dropout off, running stats used
    eval_step(model, metrics, X_test, y_test)
    test_m = metrics.compute()
    metrics.reset()
    if epoch % 5 == 0 or epoch == 29:
        print(f"epoch {epoch}: train acc {train_m['accuracy']:.3f} "
              f"test acc {test_m['accuracy']:.3f} test loss {test_m['loss']:.3f}")

# 5. Save the model state with Orbax.
ckpt_dir = Path("nnx_ckpt").absolute()
_, state = nnx.split(model)
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(ckpt_dir / "state", state, force=True)
checkpointer.wait_until_finished()

# 6. Restore into a fresh model built abstractly (no memory allocated for random init).
abstract_model = nnx.eval_shape(lambda: MLP(64, 128, 10, rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(abstract_model)
restored_state = checkpointer.restore(ckpt_dir / "state", abstract_state)
restored = nnx.merge(graphdef, restored_state)
restored.eval()
preds = restored(jnp.asarray(X_test)).argmax(-1)
print("restored test accuracy:", float((preds == y_test).mean()))
```

Explanation: NNX modules are mutable objects, so `optimizer.update(model, grads)` and `metrics.update(...)` change state in place even inside `nnx.jit`, which handles propagating the updates. `model.train()`/`model.eval()` flip Dropout and BatchNorm modes. Checkpointing uses the pure `(graphdef, state)` split; `nnx.eval_shape` builds the target structure without computing anything. Expect around 97 percent test accuracy.

### Tutorial 3: CNN with Flax Linen, BatchNorm, Dropout, and TrainState

This is the classic Linen pattern still used in many codebases: explicit `init`/`apply`, mutable `batch_stats`, dropout RNGs, and a custom `TrainState`.

```python
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

class CNN(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(32, kernel_size=(3, 3))(x)                 # NHWC layout
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(64, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not train)(x)
        return nn.Dense(self.num_classes)(x)

class TrainState(train_state.TrainState):
    batch_stats: Any
    key: jax.Array

def create_state(key, lr=1e-3):
    model = CNN()
    k_params, k_dropout = jax.random.split(key)
    variables = model.init(k_params, jnp.ones((1, 28, 28, 1)), train=False)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=optax.adam(lr),
        key=k_dropout,
    )

@jax.jit
def train_step(state, images, labels):
    dropout_key = jax.random.fold_in(state.key, state.step)    # new dropout mask each step

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            images, train=True, rngs={"dropout": dropout_key}, mutable=["batch_stats"])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, (logits, updates)

    (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return state, loss, acc

@jax.jit
def eval_step(state, images, labels):
    logits = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats},
                            images, train=False)
    return jnp.mean(jnp.argmax(logits, -1) == labels)

# Synthetic data: class k images have a bright k-th row band, so the task is learnable.
def make_batch(key, batch=128):
    k1, k2 = jax.random.split(key)
    labels = jax.random.randint(k1, (batch,), 0, 10)
    images = 0.3 * jax.random.normal(k2, (batch, 28, 28, 1))
    rows = jnp.arange(28)[None, :, None, None]
    band = (rows // 2.8).astype(jnp.int32) == labels[:, None, None, None]
    return images + band.astype(jnp.float32), labels

key = jax.random.key(0)
state = create_state(key)
for step in range(300):
    key, sub = jax.random.split(key)
    images, labels = make_batch(sub)
    state, loss, acc = train_step(state, images, labels)
    if step % 50 == 0:
        print(f"step {step}: loss {loss:.3f} acc {acc:.3f}")

test_images, test_labels = make_batch(jax.random.key(123), 1000)
print("eval accuracy:", float(eval_step(state, test_images, test_labels)))
```

Explanation: `model.init` returns two collections, `params` and `batch_stats`. During training, `mutable=["batch_stats"]` makes `apply` return updated statistics, which are stored back on the state. Dropout needs an RNG passed through `rngs={"dropout": ...}`; folding in the step number gives a different mask every step without threading a key through the loop. Flax Linen layers use NHWC layout for images (PyTorch uses NCHW).

### Tutorial 4: Data-parallel training with sharding

Use `NamedSharding` to split each batch across all devices while replicating parameters; `jax.jit` automatically partitions the computation and averages gradients. The first line simulates 8 devices on a CPU-only machine so the example runs anywhere; remove it on real multi-GPU/TPU hosts.

```python
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")   # must precede `import jax`

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

print("devices:", jax.devices())
mesh = Mesh(np.array(jax.devices()), axis_names=("data",))
data_sharding = NamedSharding(mesh, P("data"))     # split leading (batch) dim
replicated = NamedSharding(mesh, P())              # full copy on every device

# 1. A small MLP written as pure functions over a params pytree.
def init_params(key, sizes):
    params = []
    for k, (din, dout) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append({"w": jax.random.normal(k, (din, dout)) * jnp.sqrt(2.0 / din),
                       "b": jnp.zeros(dout)})
    return params

def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]

def loss_fn(params, x, y):
    return optax.softmax_cross_entropy_with_integer_labels(forward(params, x), y).mean()

tx = optax.adamw(1e-3)

# 2. Place params and optimizer state replicated.
params = jax.device_put(init_params(jax.random.key(0), [32, 256, 256, 4]), replicated)
opt_state = jax.device_put(tx.init(params), replicated)

# 3. The step is ordinary single-device code. Sharded inputs make XLA partition it.
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# 4. Synthetic dataset; global batch must be divisible by the number of devices.
rng = np.random.default_rng(0)
X = rng.normal(size=(8192, 32)).astype(np.float32)
y = (X[:, :4].argmax(-1)).astype(np.int32)
global_batch = 256

for step in range(200):
    idx = rng.integers(0, len(X), global_batch)
    xb = jax.device_put(X[idx], data_sharding)
    yb = jax.device_put(y[idx], data_sharding)
    params, opt_state, loss = train_step(params, opt_state, xb, yb)
    if step % 50 == 0:
        print(f"step {step}: loss {float(loss):.4f}")

print("input sharding:", xb.sharding)
print("param sharding:", params[0]["w"].sharding)
jax.debug.visualize_array_sharding(xb[:, :8])
```

What happens: each device holds 256 / 8 = 32 examples; parameters are replicated. Because the loss is a mean over the *global* batch, XLA inserts an all-reduce on the gradients automatically, so all replicas apply identical updates. To move to tensor or FSDP-style parallelism, add a second mesh axis (e.g. `("data", "model")`) and shard weight matrices with specs like `P(None, "model")`; the training step code does not change. For multi-host training, call `jax.distributed.initialize()` at startup and build global arrays from per-host data with `jax.make_array_from_process_local_data(sharding, local_batch)`.

## Performance & Best Practices

### Compilation

- **`jit` the largest unit possible** (the whole training step), not tiny functions. Calling many small jitted functions from Python loses fusion opportunities and adds dispatch overhead.
- **Keep shapes static.** Every new input shape triggers a recompile. Pad variable-length data to a few fixed bucket sizes, and use `drop_remainder`-style batching so the last batch has the same shape.
- **Mark configuration as static** (`static_argnames`) only for values that rarely change; each distinct value compiles a new program.
- **Use `lax.scan` over layers** (or `nnx.scan` / `nn.scan`) for deep stacks of identical blocks to cut compile time drastically.
- **Enable the persistent compilation cache** for repeated runs: `jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")`.
- **Detect recompiles** with `jax.config.update("jax_log_compiles", True)`.

### Memory

- **Donate buffers** of state you replace each step: `jax.jit(step, donate_argnums=(0, 1))` lets XLA reuse parameter/optimizer memory for outputs.
- **Rematerialize** with `jax.checkpoint` (or `nnx.remat` / `nn.remat`) to trade compute for activation memory.
- **Use bfloat16** for activations and, where acceptable, parameters; keep optimizer moments in float32 (or set `mu_dtype=jnp.bfloat16` in Optax to save memory).
- **Shard** parameters and optimizer state (FSDP-style `P("data")` on weight dims) when they do not fit on one device.
- **GPU preallocation**: by default JAX reserves 75 percent of GPU memory. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing a GPU, or `XLA_PYTHON_CLIENT_MEM_FRACTION` to adjust.

### Throughput

- **Avoid host round-trips** in the training loop: `float(loss)`, `print(loss)`, or `np.asarray(x)` each step forces synchronization. Log every N steps.
- **Overlap data loading** with computation: prepare the next batch (NumPy) while the device runs the current step; JAX's async dispatch makes this natural. Grain or `tf.data` with prefetching help.
- **Set matmul precision** explicitly if needed: `jax.config.update("jax_default_matmul_precision", "bfloat16")` or `"float32"` for exactness. On TPUs, default float32 matmuls use bfloat16 passes.
- **Prefer `vmap`** over Python loops for batching, and `jnp` vectorized ops over element-wise Python.
- **Fused attention**: `jax.nn.dot_product_attention(..., implementation="cudnn")` on NVIDIA GPUs; Pallas for custom kernels.
- **Benchmark correctly**: warm up once (compilation), then time with `block_until_ready()`.

### Code organization

- Keep model state in pytrees (or NNX objects) and write pure step functions `state, batch -> state, metrics`.
- Split RNG keys at the top level and pass them down explicitly; never reuse a key.
- Use `jax.tree.map` for anything you would otherwise loop over parameters to do.
- Validate with small shapes and `jax.disable_jit()` before scaling up.
- Pin versions of `jax`, `jaxlib`, plugin packages, `flax`, and `optax` together; mismatched `jax`/`jaxlib` versions are a common source of import errors.

## Common Errors & Troubleshooting

### Concretization and tracer conversion errors

```text
jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
```

```text
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[]
```

Cause: Python control flow or Python functions that need a concrete value (`if x > 0`, `range(n)`, `int(x)`, slicing with a traced size) applied to a traced argument inside `jit`. Fixes:

- Replace `if` with `jnp.where` or `lax.cond`; replace loops with `lax.fori_loop` / `lax.scan`.
- Mark the argument static: `jax.jit(f, static_argnames="n")`.
- Use `lax.dynamic_slice` with a static size instead of `x[i:i + n]` with traced `n`.

```python
import jax
import jax.numpy as jnp
from functools import partial

# Broken: @jax.jit def relu(x): return x if x > 0 else 0.0
@jax.jit
def relu(x):
    return jnp.where(x > 0, x, 0.0)

@partial(jax.jit, static_argnames="n")
def first_n(x, n):
    return x[:n]
```

```text
jax.errors.TracerArrayConversionError: The numpy.ndarray conversion method __array__() was called on traced array with shape float32[3]
```

Cause: calling NumPy (`np.sum`, `np.asarray`) or another library on a traced value. Fix: use `jnp` functions inside transformed code.

```text
jax.errors.TracerIntegerConversionError: The __index__() method was called on traced array with shape int32[]
```

Cause: using a traced integer as a Python index or in `range`. Fix: static argument, `lax.dynamic_slice`, or `jnp.take`.

### Boolean indexing under jit

```text
jax.errors.NonConcreteBooleanIndexError: Array boolean indices must be concrete; got bool[10]
```

Cause: `x[mask]` produces a data-dependent shape, which XLA cannot compile. Fix: `jnp.where(mask, x, 0)` and masked reductions (`jnp.sum(x, where=mask)`), or `jnp.nonzero(mask, size=k, fill_value=0)` with a static maximum size.

### Immutable arrays

```text
TypeError: JAX arrays are immutable and do not support in-place item assignment. Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method
```

Fix: `x = x.at[idx].set(y)` (or `.add`, `.multiply`, ...).

### Leaked tracers

```text
jax.errors.UnexpectedTracerError: Encountered an unexpected tracer. A function transformed by JAX had a side effect, allowing for a reference to an intermediate value with type float32[] wrapped in a DynamicJaxprTracer to escape the scope of the transformation.
```

Cause: storing a traced value in a global, a Python list, or an object attribute inside a jitted function (common with stateful classes or caching). Fix: return values instead of storing them; keep functions pure; with NNX objects use `nnx.jit` rather than `jax.jit`.

### Non-hashable static arguments

```text
ValueError: Non-hashable static arguments are not supported. An error occurred while trying to hash an object of type <class 'jax.Array'> ...
```

Cause: marking an array, list, or dict as static. Fix: only mark hashable Python values (ints, strings, tuples, frozen dataclasses) as static; pass arrays as regular arguments.

### Out of memory

```text
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4294967296 bytes.
```

(Newer versions print `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: ...`.) Fixes: reduce batch size; use `jax.checkpoint`; use bfloat16; donate buffers; shard across devices; check that another process (or another framework such as TensorFlow or PyTorch in the same process) is not holding GPU memory; tune `XLA_PYTHON_CLIENT_MEM_FRACTION`. If `nvidia-smi` shows 75 percent memory used right after import, that is normal preallocation, not a leak.

### GPU not found

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

Cause: CPU-only install. Fix: `pip install -U "jax[cuda12]"` (Linux/WSL2 only). If it is installed but still not used, check driver version (`nvidia-smi`), conflicting `LD_LIBRARY_PATH` CUDA libraries, `JAX_PLATFORMS`, and that `jax`, `jaxlib`, and `jax-cuda12-plugin` versions match (`jax.print_environment_info()`).

### float64 silently truncated

```text
UserWarning: Explicitly requested dtype float64 requested in zeros is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable.
```

Fix: `jax.config.update("jax_enable_x64", True)` at program start, or accept float32.

### Shape/structure mismatch in tree operations

```text
ValueError: Custom node type mismatch; expected type: <class 'dict'>; value: ...
ValueError: Dict key mismatch; expected keys: ['b', 'w']; dict: {'w': ...}.
```

Cause: `jax.tree.map` over trees with different structures, or loading a checkpoint into a mismatched model. Fix: print `jax.tree.structure(a)` and `jax.tree.structure(b)` and compare.

### Random key misuse

Symptoms: identical "random" numbers across steps, or identical dropout masks. Cause: reusing the same key. Fix: split (`key, sub = jax.random.split(key)`) or `fold_in(key, step)` every step. Mixing legacy `uint32[2]` keys and typed keys in the same pytree (for example when restoring old checkpoints) causes dtype errors; convert with `jax.random.wrap_key_data` / `jax.random.key_data`.

### Slow first step / constant recompilation

Cause: compilation on the first call is expected; repeated slowness means recompiles due to changing shapes, dtypes, or static values (including weakly-typed Python scalars switching to arrays). Fix: enable `jax_log_compiles`, keep shapes fixed (pad batches), and pass hyperparameters like learning rates as arrays rather than static values.

### Debugging NaNs

Enable `jax.config.update("jax_debug_nans", True)` to re-run the failing primitive eagerly and raise `FloatingPointError: invalid value (nan) encountered in ...` at the source. Common culprits: `log(0)`, `sqrt` at 0 in gradients (use `jnp.sqrt(x + eps)`), `jnp.where` with a NaN in the untaken branch's gradient (apply the "double where" trick).

## Interoperability

### NumPy

`jnp.asarray(np_array)` uploads to the device; `np.asarray(jax_array)` downloads. Most NumPy code ports by changing the import to `jax.numpy` and removing in-place mutations.

### PyTorch

Zero-copy exchange via DLPack, plus a common pattern of using the PyTorch `DataLoader` for input pipelines:

```python
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# DLPack: torch <-> JAX without copies (same device)
t = torch.arange(4.0)
j = jnp.from_dlpack(t)
t_back = torch.from_dlpack(j)

# PyTorch DataLoader feeding NumPy batches to JAX
def numpy_collate(batch):
    xs, ys = zip(*batch)
    return np.stack([x.numpy() for x in xs]), np.array([int(y) for y in ys])

ds = TensorDataset(torch.randn(100, 8), torch.randint(0, 2, (100,)))
for xb, yb in DataLoader(ds, batch_size=32, collate_fn=numpy_collate):
    xb = jnp.asarray(xb)       # ready for a jitted JAX step
```

Weights can be converted by mapping state-dict tensors to NumPy and building the corresponding pytree (remember to transpose Linear weights: PyTorch stores `(out, in)`, Flax `Dense`/`nnx.Linear` kernels are `(in, out)`).

### TensorFlow and data pipelines

- `tf.data` and TensorFlow Datasets (`tfds`) are widely used with JAX: iterate with `tfds.as_numpy(ds)`.
- **Grain** (`grain`) is the JAX-native, deterministic data loader used in MaxText.
- `jax2tf` (`from jax.experimental import jax2tf`) converts JAX functions to TensorFlow graphs for TF Serving / TFLite.

### Keras 3

Keras 3 runs on JAX by setting `KERAS_BACKEND=jax` before importing Keras; Keras layers and models then execute as JAX code and can be combined with JAX training loops via `model.stateless_call`.

### Hugging Face

- **Transformers** historically shipped Flax model classes (`FlaxAutoModel...`). Flax support has been deprecated and is removed in Transformers v5; for JAX work today prefer JAX-native model code such as MaxText, KerasHub (with the JAX backend), or Google's `gemma` JAX library, or convert PyTorch weights yourself.
- `safetensors.flax` (`from safetensors.flax import save_file, load_file`) saves and loads dicts of JAX arrays.
- The Hub (`huggingface_hub`) stores Orbax/safetensors JAX checkpoints like any other files.

### Other JAX ecosystem libraries

| Library | Purpose |
|---|---|
| Equinox | PyTorch-like modules as pytrees; filtered transforms. |
| Optax | Optimizers and losses. |
| Orbax | Checkpointing (async, sharded, multi-host). |
| Grain | Data loading. |
| Chex | Testing utilities and assertions. |
| NumPyro, BlackJAX | Probabilistic programming and MCMC. |
| Diffrax, Optimistix, Lineax | Differential equations, root finding, linear solvers. |
| Pallas (`jax.experimental.pallas`) | Write custom GPU/TPU kernels in Python. |
| MaxText | Reference LLM training/inference at scale. |

## Cheat Sheet

### Arrays and NumPy

| Task | Code |
|---|---|
| Import | `import jax, jax.numpy as jnp` |
| Create | `jnp.array([1., 2.])`, `jnp.zeros((3, 4))`, `jnp.arange(10)` |
| Update element | `x = x.at[i].set(v)` |
| Scatter add | `x = x.at[idx].add(v)` |
| Conditional | `jnp.where(c, a, b)` |
| To NumPy | `np.asarray(x)` or `jax.device_get(x)` |
| Wait for result | `x.block_until_ready()` |
| Enable float64 | `jax.config.update("jax_enable_x64", True)` |
| Devices | `jax.devices()`, `jax.default_backend()` |
| Put on device | `jax.device_put(x, jax.devices()[0])` |

### Transformations

| Task | Code |
|---|---|
| Gradient | `jax.grad(f)(x)` |
| Loss and gradient | `loss, g = jax.value_and_grad(f)(params, batch)` |
| With aux outputs | `(loss, aux), g = jax.value_and_grad(f, has_aux=True)(p)` |
| Grad w.r.t. several args | `jax.grad(f, argnums=(0, 1))` |
| Compile | `jax.jit(f)` or `@jax.jit` |
| Static argument | `jax.jit(f, static_argnames="n")` |
| Donate buffers | `jax.jit(step, donate_argnums=(0,))` |
| Batch a function | `jax.vmap(f, in_axes=(None, 0))` |
| Per-example grads | `jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))` |
| Jacobian / Hessian | `jax.jacrev(f)`, `jax.hessian(f)` |
| Stop gradient | `jax.lax.stop_gradient(x)` |
| Remat | `jax.checkpoint(f)` |
| Show jaxpr | `jax.make_jaxpr(f)(x)` |
| Print inside jit | `jax.debug.print("{x}", x=x)` |

### Random, control flow, pytrees

| Task | Code |
|---|---|
| New key | `key = jax.random.key(0)` |
| Split | `key, sub = jax.random.split(key)` |
| Many keys | `keys = jax.random.split(key, 8)` |
| Per-step key | `jax.random.fold_in(key, step)` |
| Normal / uniform | `jax.random.normal(k, (3,))`, `jax.random.uniform(k, (3,))` |
| Shuffle | `jax.random.permutation(k, n)` |
| If / else | `lax.cond(pred, f_true, f_false, x)` |
| Loop | `lax.fori_loop(0, n, body, init)` |
| Scan | `carry, ys = lax.scan(f, init, xs)` |
| Map over leaves | `jax.tree.map(lambda x: x * 2, tree)` |
| Count params | `sum(x.size for x in jax.tree.leaves(params))` |

### Sharding

| Task | Code |
|---|---|
| Mesh | `Mesh(np.array(jax.devices()), ("data",))` or `jax.make_mesh((8,), ("data",))` |
| Shard batch | `jax.device_put(x, NamedSharding(mesh, P("data")))` |
| Replicate | `jax.device_put(params, NamedSharding(mesh, P()))` |
| Constrain inside jit | `jax.lax.with_sharding_constraint(x, sharding)` |
| Inspect | `x.sharding`, `jax.debug.visualize_array_sharding(x)` |
| Simulate devices on CPU | `XLA_FLAGS=--xla_force_host_platform_device_count=8` |
| Multi-host init | `jax.distributed.initialize()` |

### Flax and Optax

| Task | Code |
|---|---|
| NNX layer | `nnx.Linear(din, dout, rngs=nnx.Rngs(0))` |
| NNX optimizer | `opt = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)` |
| NNX step | `loss, g = nnx.value_and_grad(loss_fn)(model); opt.update(model, g)` |
| NNX train/eval | `model.train()`, `model.eval()` |
| NNX to pytree | `graphdef, state = nnx.split(model)` |
| Linen init | `variables = model.init(key, x)` |
| Linen apply | `model.apply(variables, x, rngs={"dropout": k}, mutable=["batch_stats"])` |
| Optax init / update | `s = tx.init(p); u, s = tx.update(g, s, p); p = optax.apply_updates(p, u)` |
| Clip + AdamW | `optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))` |
| Warmup cosine | `optax.warmup_cosine_decay_schedule(0.0, 1e-3, 1000, 100_000)` |
| CE loss | `optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()` |
| Grad accumulation | `optax.MultiSteps(tx, every_k_schedule=4)` |

## Further Resources

- Official documentation: https://docs.jax.dev/en/latest/
- Quickstart: https://docs.jax.dev/en/latest/quickstart.html
- "The Sharp Bits" (common gotchas): https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
- Installation guide: https://docs.jax.dev/en/latest/installation.html
- Distributed arrays and automatic parallelization: https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- Autodidax (build JAX core from scratch): https://docs.jax.dev/en/latest/autodidax.html
- GitHub repository: https://github.com/jax-ml/jax
- Flax documentation (NNX and Linen): https://flax.readthedocs.io/
- Flax GitHub: https://github.com/google/flax
- Optax documentation: https://optax.readthedocs.io/
- Orbax documentation: https://orbax.readthedocs.io/
- Equinox: https://github.com/patrick-kidger/equinox
- "How to Scale Your Model" (JAX/TPU scaling book): https://jax-ml.github.io/scaling-book/
- MaxText (reference LLM implementation): https://github.com/AI-Hypercomputer/maxtext
- Paper: Frostig, Johnson, Leary, "Compiling machine learning programs via high-level tracing", SysML 2018: https://mlsys.org/Conferences/doc/2018/146.pdf
- Course: UvA Deep Learning Tutorials (JAX versions): https://uvadlc-notebooks.readthedocs.io/
