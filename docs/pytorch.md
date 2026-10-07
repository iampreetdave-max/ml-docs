# PyTorch

> Tensors and dynamic neural networks in Python with strong GPU acceleration.

PyTorch is an open-source deep learning framework built around an n-dimensional `Tensor` type, a tape-based automatic differentiation engine (`torch.autograd`), and a large library of neural network building blocks (`torch.nn`). Its define-by-run ("eager") execution model makes models ordinary Python programs: you can print, branch, loop, and debug them with standard tools. Since PyTorch 2.0, the `torch.compile` compiler stack lets you keep that eager programming model while getting graph-level optimizations.

Covers PyTorch 2.x (examples tested against the 2.4 to 2.8 series). Notable 2.x changes called out on this page include `torch.compile`, the device-generic `torch.amp` API replacing `torch.cuda.amp`, the `weights_only=True` default for `torch.load` (2.6+), `torch.export`, and FSDP2 (`fully_shard`).

## Overview

### What it is

PyTorch provides:

- **`torch.Tensor`**: a strided, typed, device-placed multi-dimensional array, very similar to a NumPy `ndarray` but able to live on GPUs (CUDA, ROCm), Apple Silicon (MPS), Intel GPUs (XPU), and other accelerators.
- **Autograd**: reverse-mode automatic differentiation that records operations on tensors with `requires_grad=True` and computes gradients via `.backward()` or `torch.autograd.grad`.
- **`torch.nn`**: modules (layers), loss functions, initializers, and functional ops for building networks.
- **`torch.optim`**: optimizers (SGD, Adam, AdamW, ...) and learning-rate schedulers.
- **`torch.utils.data`**: `Dataset`, `DataLoader`, samplers, and collation for efficient input pipelines.
- **`torch.amp`**: automatic mixed precision (autocast and gradient scaling).
- **`torch.compile`**: a JIT compiler (TorchDynamo + AOTAutograd + TorchInductor) that speeds up models with one line.
- **`torch.distributed`**: collective communication, DistributedDataParallel (DDP), FSDP, tensor/pipeline parallel primitives, and distributed checkpointing.
- **`torch.export`** and **ONNX export** for deployment.

### History and maintainers

PyTorch was released by Facebook AI Research (now Meta AI) in 2016-2017 as a Python successor to the Lua-based Torch7 library, borrowing ideas from Chainer's define-by-run approach. In September 2022, governance moved to the **PyTorch Foundation** under the Linux Foundation, with members including Meta, AMD, AWS, Google, Microsoft, and NVIDIA. PyTorch 2.0 (March 2023) introduced `torch.compile`. Today PyTorch is the dominant framework in published research and is the backend for most of the open model ecosystem (Hugging Face Transformers, vLLM, Lightning, timm, and many more).

### When to use it

- Research and prototyping, where debuggability and flexibility matter.
- Training and fine-tuning modern deep learning models (transformers, diffusion models, CNNs, GNNs).
- When you want access to the largest ecosystem of pretrained models and third-party libraries.
- Production inference via `torch.compile`, `torch.export`, TorchScript (legacy), ONNX, or serving engines that consume PyTorch checkpoints.

### When not to use it

- Classical tabular ML (gradient-boosted trees, linear models) is usually better served by scikit-learn, XGBoost, or LightGBM.
- If you want a purely functional, compiler-first approach with first-class TPU support and composable transforms (`vmap`, `grad`, `jit`) on pure functions, JAX may fit better (although `torch.func` covers much of this).
- Tiny embedded targets may need specialized runtimes (ExecuTorch is PyTorch's answer for on-device; TFLite/LiteRT is an alternative).

### Where it fits in the ML stack

```text
Applications / research code
        |
High-level libraries:  Hugging Face Transformers, Lightning, timm, torchvision, torchaudio, PEFT, Accelerate
        |
PyTorch: torch.nn, torch.optim, torch.utils.data, torch.distributed, torch.compile
        |
Kernels / compilers:  ATen, cuBLAS, cuDNN, MIOpen, Triton (via Inductor), oneDNN
        |
Hardware:  NVIDIA GPU, AMD GPU, Apple Silicon (MPS), Intel GPU (XPU), CPU, (TPU via torch_xla)
```

## Installation

### pip (recommended)

The official selector at https://pytorch.org/get-started/locally/ generates the right command. Typical commands:

```bash
# Linux / Windows, CUDA 12.x wheels (pick the CUDA version your driver supports)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Newer CUDA builds (example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# macOS (Apple Silicon has MPS support in the default wheel)
pip install torch torchvision torchaudio

# AMD ROCm (Linux only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

On Linux, the default PyPI `torch` wheel already bundles a CUDA build, so `pip install torch` gives you GPU support on most NVIDIA machines. You only need a compatible NVIDIA **driver**; the CUDA toolkit itself is shipped inside the wheel (as `nvidia-*` pip packages).

### conda

PyTorch stopped publishing to its own `pytorch` conda channel starting with 2.6. Use pip inside a conda environment, or the community-maintained `conda-forge` package:

```bash
conda create -n torch python=3.11 -y
conda activate torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# or conda-forge build
conda install -c conda-forge pytorch
```

### TPU and other accelerators

```bash
# Google Cloud TPU VMs
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Intel GPUs (XPU) ship in dedicated wheels
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

### Verifying the install

```python
import torch

print(torch.__version__)               # e.g. 2.5.1+cu124
print(torch.version.cuda)              # CUDA version PyTorch was built with, None for CPU builds
print(torch.cuda.is_available())       # True if an NVIDIA/ROCm GPU is usable
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))   # e.g. (8, 0) for A100
print(torch.backends.cudnn.version())
print(torch.backends.mps.is_available())          # Apple Silicon
```

On ROCm builds, `torch.cuda.*` APIs work against AMD GPUs and `torch.version.hip` is set.

For a full environment report (useful for bug reports):

```bash
python -m torch.utils.collect_env
nvidia-smi   # driver version and GPU memory usage
```

### Choosing a device generically

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# PyTorch 2.5+ also offers a generic accelerator API
# device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
```

## Core Concepts

### 1. Tensors

A tensor has a **dtype** (e.g. `torch.float32`), a **shape** (`torch.Size`), a **device** (`cpu`, `cuda:0`, `mps`), a **layout** (strided, sparse), and optionally **`requires_grad`**. Tensors are views over a contiguous block of memory (the *storage*) described by a shape, strides, and an offset.

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.dtype, x.shape, x.device, x.stride())   # torch.float32 torch.Size([2, 2]) cpu (2, 1)

y = x.t()                     # transpose is a view, no copy
print(y.stride())             # (1, 2) -> not contiguous
print(y.is_contiguous())      # False
z = y.contiguous()            # materializes a new contiguous copy

# Views share memory
v = x.view(4)
v[0] = 100.0
print(x[0, 0])                # tensor(100.)
```

Key rules:

- Floating-point default dtype is `torch.float32`; integer literals become `torch.int64`.
- Operations between tensors on different devices raise an error; you move data explicitly with `.to(device)`.
- Most ops return new tensors; methods ending in `_` (e.g. `add_`, `zero_`) mutate in place.
- Broadcasting follows NumPy semantics: trailing dimensions are aligned and size-1 dimensions are stretched.

```python
a = torch.randn(8, 1, 5)
b = torch.randn(   3, 5)
print((a + b).shape)          # torch.Size([8, 3, 5])
```

### 2. Autograd: the dynamic computation graph

When a tensor has `requires_grad=True`, every operation on it records a node in a graph (each result has a `grad_fn`). Calling `.backward()` on a scalar walks the graph in reverse and accumulates gradients into `.grad` of leaf tensors. The graph is rebuilt on every forward pass, so Python control flow just works.

```python
import torch

w = torch.tensor([2.0, -1.0], requires_grad=True)
x = torch.tensor([3.0, 4.0])
y = (w * x).sum() ** 2       # y = (2*3 + -1*4)^2 = 4
print(y.grad_fn)             # <PowBackward0 ...>

y.backward()
print(w.grad)                # dy/dw = 2*(w.x)*x = 2*2*[3,4] = tensor([12., 16.])

# Gradients ACCUMULATE: call .zero_() / optimizer.zero_grad() between steps
w.grad.zero_()

# Disable tracking for inference
with torch.no_grad():
    y2 = w * 2
print(y2.requires_grad)      # False

# Detach a tensor from the graph
d = (w * 3).detach()
```

`torch.inference_mode()` is a stricter, faster alternative to `no_grad()` for pure inference.

### 3. Modules

`nn.Module` is the base class for all layers and models. A module:

- registers `nn.Parameter`s and sub-modules assigned as attributes,
- defines `forward(...)`,
- can be moved with `.to(device)`, switched with `.train()` / `.eval()`,
- exposes `parameters()`, `named_parameters()`, `state_dict()`, `load_state_dict()`.

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = MLP(784, 256, 10)
print(model)
print(sum(p.numel() for p in model.parameters()))   # 203530

logits = model(torch.randn(32, 784))   # calls __call__ -> hooks + forward
print(logits.shape)                    # torch.Size([32, 10])
```

Always call the module (`model(x)`), not `model.forward(x)`, so hooks run.

### 4. The training loop

PyTorch does not hide the training loop. The canonical shape is:

```python
import torch
from torch import nn

model = nn.Linear(10, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X = torch.randn(256, 10)
y = X @ torch.randn(10, 1) + 0.1 * torch.randn(256, 1)

for epoch in range(100):
    model.train()
    optimizer.zero_grad(set_to_none=True)   # 1. clear old gradients
    pred = model(X)                         # 2. forward
    loss = loss_fn(pred, y)                 # 3. compute loss
    loss.backward()                         # 4. backprop
    optimizer.step()                        # 5. update parameters

print(loss.item())
```

### 5. Train vs eval mode

`model.train()` and `model.eval()` toggle the `training` flag that changes the behavior of layers like `Dropout` and `BatchNorm`. They do **not** disable gradient tracking; combine `model.eval()` with `torch.no_grad()` or `torch.inference_mode()` for evaluation.

### 6. Devices and data movement

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)                   # moves parameters and buffers in place, returns self
x = torch.randn(4, 10, device=device)      # allocate directly on device (faster than .to)
out = model(x)
out_cpu = out.detach().cpu().numpy()       # back to NumPy for plotting/metrics
```

CUDA operations are **asynchronous**: Python returns before the kernel finishes. Calls like `.item()`, `.cpu()`, `print(tensor)`, or `torch.cuda.synchronize()` force synchronization, which matters for timing and performance.

### 7. Eager vs compiled

By default every op runs immediately ("eager mode"). `torch.compile(model)` captures Python bytecode into FX graphs with TorchDynamo, then lowers them to fused Triton (GPU) or C++/OpenMP (CPU) kernels with TorchInductor. Unsupported Python constructs cause a *graph break*, falling back to eager for that portion, so compiled code remains correct.

```python
model = MLP(784, 256, 10)
compiled = torch.compile(model)            # same interface, first call triggers compilation
out = compiled(torch.randn(32, 784))
```

### 8. Functional transforms (torch.func)

`torch.func` (formerly functorch) provides JAX-like composable transforms: `grad`, `vmap`, `jacrev`, `jacfwd`, `hessian`, and `functional_call`.

```python
import torch
from torch.func import grad, vmap

def f(x):
    return torch.sin(x).sum()

x = torch.randn(5)
print(grad(f)(x))                          # cos(x)

batched = vmap(lambda v: v @ v)(torch.randn(8, 3))
print(batched.shape)                       # torch.Size([8])
```

## API Reference

This section covers the most-used parts of the public API. Signatures are simplified to the commonly used arguments; see the official docs for every keyword.

### Tensor creation

#### torch.tensor

```python
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | array-like | required | List, tuple, NumPy array, scalar. Always copies. |
| `dtype` | `torch.dtype` | `None` | Inferred from data if not given. |
| `device` | `str` or `torch.device` | `None` | Target device (current default device if `None`). |
| `requires_grad` | `bool` | `False` | Record operations for autograd. |
| `pin_memory` | `bool` | `False` | Allocate in page-locked memory (CPU only). |

Returns a new tensor that owns a copy of `data`.

```python
import torch
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="cpu")
```

#### Factory functions

```python
torch.zeros(*size, dtype=None, device=None, requires_grad=False) -> Tensor
torch.ones(*size, dtype=None, device=None, requires_grad=False) -> Tensor
torch.full(size, fill_value, dtype=None, device=None) -> Tensor
torch.empty(*size, dtype=None, device=None) -> Tensor
torch.eye(n, m=None, dtype=None, device=None) -> Tensor
torch.arange(start=0, end, step=1, dtype=None, device=None) -> Tensor
torch.linspace(start, end, steps, dtype=None, device=None) -> Tensor
torch.rand(*size, generator=None, dtype=None, device=None) -> Tensor      # U[0, 1)
torch.randn(*size, generator=None, dtype=None, device=None) -> Tensor     # N(0, 1)
torch.randint(low=0, high, size, generator=None, dtype=torch.int64, device=None) -> Tensor
torch.randperm(n, generator=None, dtype=torch.int64, device=None) -> Tensor
torch.zeros_like(input, dtype=None, device=None) -> Tensor   # also ones_like, empty_like, rand_like, randn_like, full_like
```

| Function | Description |
|---|---|
| `zeros`, `ones`, `full` | Constant-filled tensors. |
| `empty` | Uninitialized memory (fastest; values are garbage). |
| `arange` | Like Python `range`, end-exclusive. |
| `linspace` | `steps` evenly spaced values, end-inclusive. |
| `rand`, `randn`, `randint` | Random tensors from the global or a given `torch.Generator`. |
| `*_like` | Same shape (and by default dtype/device) as another tensor. |

```python
import torch
torch.manual_seed(0)
a = torch.zeros(2, 3)
b = torch.arange(0, 10, 2)            # tensor([0, 2, 4, 6, 8])
c = torch.linspace(0, 1, 5)           # tensor([0.00, 0.25, 0.50, 0.75, 1.00])
d = torch.randn(3, 4, device="cpu")
e = torch.randint(0, 10, (2, 2))
f = torch.full((2, 2), 7.0)
g = torch.randn_like(d)
```

### Tensor attributes and conversion

| Attribute / method | Description |
|---|---|
| `t.shape` / `t.size()` | Shape as `torch.Size`. `t.size(1)` gives one dim. |
| `t.dtype` | Data type. |
| `t.device` | Device. |
| `t.ndim` / `t.dim()` | Number of dimensions. |
| `t.numel()` | Total number of elements. |
| `t.requires_grad` | Whether autograd tracks it. |
| `t.grad` | Accumulated gradient (leaf tensors). |
| `t.item()` | Python scalar from a 1-element tensor (syncs GPU). |
| `t.tolist()` | Nested Python lists. |
| `t.to(device=None, dtype=None, non_blocking=False, copy=False)` | Move and/or cast. |
| `t.float()`, `t.half()`, `t.bfloat16()`, `t.long()`, `t.int()`, `t.bool()` | Cast shortcuts. |
| `t.cpu()`, `t.cuda(device=None)` | Device shortcuts. |
| `t.detach()` | New tensor sharing storage but no grad history. |
| `t.clone()` | Copy that keeps grad history. |
| `t.contiguous()` | Contiguous copy if needed. |

```python
import torch
x = torch.randn(2, 3)
y = x.to(torch.float16)
if torch.cuda.is_available():
    z = x.pin_memory().to("cuda", non_blocking=True)
```

### Shape manipulation

```python
Tensor.view(*shape) -> Tensor              # requires compatible strides, never copies
Tensor.reshape(*shape) -> Tensor           # view if possible, otherwise copies
Tensor.permute(*dims) -> Tensor
Tensor.transpose(dim0, dim1) -> Tensor
Tensor.unsqueeze(dim) -> Tensor
Tensor.squeeze(dim=None) -> Tensor
Tensor.flatten(start_dim=0, end_dim=-1) -> Tensor
Tensor.expand(*sizes) -> Tensor            # broadcast view, no copy
Tensor.repeat(*sizes) -> Tensor            # real copy
torch.cat(tensors, dim=0) -> Tensor
torch.stack(tensors, dim=0) -> Tensor
torch.split(tensor, split_size_or_sections, dim=0) -> tuple[Tensor, ...]
torch.chunk(input, chunks, dim=0) -> tuple[Tensor, ...]
```

| Function | Description |
|---|---|
| `view` / `reshape` | Change shape; `-1` infers one dimension. |
| `permute` | Reorder all dims (e.g. NHWC to NCHW). |
| `transpose` / `t` | Swap two dims / 2-D transpose. |
| `unsqueeze` / `squeeze` | Add / remove size-1 dims. |
| `cat` | Join along an existing dim. |
| `stack` | Join along a new dim. |
| `split` / `chunk` | Split by size / by number of chunks. |

```python
import torch
x = torch.arange(24).reshape(2, 3, 4)
print(x.permute(2, 0, 1).shape)         # torch.Size([4, 2, 3])
print(x.flatten(1).shape)               # torch.Size([2, 12])
print(x.unsqueeze(0).shape)             # torch.Size([1, 2, 3, 4])
a, b = torch.split(x, [1, 2], dim=1)
print(a.shape, b.shape)                 # [2, 1, 4] [2, 2, 4]
print(torch.stack([x, x]).shape)        # torch.Size([2, 2, 3, 4])
print(torch.cat([x, x], dim=-1).shape)  # torch.Size([2, 3, 8])
```

### Indexing, selection, and masking

```python
torch.where(condition, input, other) -> Tensor
torch.gather(input, dim, index) -> Tensor
Tensor.scatter_(dim, index, src) -> Tensor
torch.index_select(input, dim, index) -> Tensor
torch.masked_select(input, mask) -> Tensor
Tensor.masked_fill(mask, value) -> Tensor
torch.topk(input, k, dim=-1, largest=True, sorted=True) -> (values, indices)
torch.sort(input, dim=-1, descending=False) -> (values, indices)
torch.argmax(input, dim=None, keepdim=False) -> Tensor
torch.nonzero(input, as_tuple=False) -> Tensor
```

```python
import torch
x = torch.tensor([[1., -2., 3.], [-4., 5., -6.]])
print(x[:, 1])                           # tensor([-2., 5.])
print(x[x > 0])                          # tensor([1., 3., 5.])
print(torch.where(x > 0, x, torch.zeros_like(x)))
print(x.masked_fill(x < 0, float("-inf")))

logits = torch.randn(4, 10)
labels = torch.tensor([1, 0, 9, 3])
picked = logits.gather(1, labels.unsqueeze(1)).squeeze(1)   # logit of the true class
vals, idx = logits.topk(3, dim=-1)
```

### Math and reductions

```python
torch.matmul(input, other) -> Tensor       # also the @ operator
torch.bmm(input, mat2) -> Tensor           # batched (b, n, m) @ (b, m, p)
torch.einsum(equation, *operands) -> Tensor
torch.sum(input, dim=None, keepdim=False, dtype=None) -> Tensor
torch.mean(input, dim=None, keepdim=False) -> Tensor
torch.std(input, dim=None, correction=1, keepdim=False) -> Tensor
torch.max(input, dim, keepdim=False) -> (values, indices)
torch.clamp(input, min=None, max=None) -> Tensor
torch.softmax(input, dim) -> Tensor
torch.log_softmax(input, dim) -> Tensor
torch.cumsum(input, dim) -> Tensor
torch.allclose(input, other, rtol=1e-05, atol=1e-08) -> bool
```

```python
import torch
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B                                   # (3, 5)
batch = torch.bmm(torch.randn(8, 3, 4), torch.randn(8, 4, 5))   # (8, 3, 5)

# Attention scores with einsum
q = torch.randn(2, 8, 16, 64)               # (batch, heads, seq, dim)
k = torch.randn(2, 8, 16, 64)
scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / 64 ** 0.5

x = torch.randn(4, 5)
print(x.sum(dim=1, keepdim=True).shape)     # torch.Size([4, 1])
values, indices = x.max(dim=1)
print(torch.clamp(x, -0.5, 0.5).abs().max() <= 0.5)
```

`torch.linalg` holds linear algebra: `norm`, `inv`, `solve`, `svd`, `qr`, `eigh`, `cholesky`, `matrix_rank`, `pinv`, `det`.

```python
import torch
A = torch.randn(3, 3)
A = A @ A.T + 3 * torch.eye(3)              # symmetric positive definite
b = torch.randn(3)
x = torch.linalg.solve(A, b)
print(torch.allclose(A @ x, b, atol=1e-5))
U, S, Vh = torch.linalg.svd(torch.randn(5, 3), full_matrices=False)
```

### Autograd

#### Tensor.backward

```python
Tensor.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None) -> None
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gradient` | `Tensor` | `None` | Upstream gradient; required if the tensor is not a scalar. |
| `retain_graph` | `bool` | `None` (= `create_graph`) | Keep the graph to call backward again. |
| `create_graph` | `bool` | `False` | Build a graph of the backward pass (for higher-order grads). |
| `inputs` | sequence of Tensors | `None` | Only accumulate into these tensors. |

Accumulates gradients into the `.grad` of leaves. Returns `None`.

#### torch.autograd.grad

```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None,
                    create_graph=False, allow_unused=None) -> tuple[Tensor, ...]
```

Returns gradients instead of accumulating them, which is useful for gradient penalties and meta-learning.

```python
import torch
x = torch.tensor(3.0, requires_grad=True)
y = x ** 3
(dy,) = torch.autograd.grad(y, x, create_graph=True)   # 3x^2 = 27
(d2y,) = torch.autograd.grad(dy, x)                     # 6x = 18
print(dy.item(), d2y.item())
```

#### Gradient-mode context managers

| API | Description |
|---|---|
| `torch.no_grad()` | Disable graph recording (context manager or decorator). |
| `torch.enable_grad()` | Re-enable inside a `no_grad` block. |
| `torch.inference_mode(mode=True)` | Faster `no_grad`; outputs can never be used in autograd later. |
| `torch.set_grad_enabled(mode)` | Toggle with a boolean (handy for train/eval in one function). |
| `torch.autograd.set_detect_anomaly(True)` | Report the forward op that produced NaN/inf in backward (slow, debug only). |

```python
import torch
model = torch.nn.Linear(4, 2)

@torch.inference_mode()
def predict(x):
    return model(x).argmax(-1)

print(predict(torch.randn(3, 4)))
```

### torch.nn: module basics

#### nn.Module

| Method | Description |
|---|---|
| `forward(*args)` | Defines computation. Override this. |
| `parameters(recurse=True)` / `named_parameters()` | Iterate parameters. |
| `buffers()` / `register_buffer(name, tensor, persistent=True)` | Non-trainable state (e.g. BatchNorm running stats). |
| `children()` / `modules()` / `named_modules()` | Iterate sub-modules. |
| `to(device=None, dtype=None)` | Move/cast parameters and buffers. |
| `train(mode=True)` / `eval()` | Set training flag recursively. |
| `zero_grad(set_to_none=True)` | Clear gradients. |
| `state_dict()` / `load_state_dict(state_dict, strict=True, assign=False)` | Serialize / restore. |
| `requires_grad_(requires_grad=True)` | Freeze or unfreeze all parameters. |
| `apply(fn)` | Apply `fn` to every sub-module (e.g. custom init). |
| `register_forward_hook(hook)` / `register_full_backward_hook(hook)` | Inspect activations / gradients. |

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

# Freeze the first layer
model[0].requires_grad_(False)
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(trainable)                     # ['2.weight', '2.bias']

# Custom init
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
model.apply(init_weights)

# Capture an activation with a hook
acts = {}
h = model[1].register_forward_hook(lambda mod, inp, out: acts.__setitem__("relu", out.detach()))
model(torch.randn(2, 4))
h.remove()
print(acts["relu"].shape)            # torch.Size([2, 8])
```

#### Containers

| Class | Description |
|---|---|
| `nn.Sequential(*modules)` | Runs modules in order. Also accepts an `OrderedDict`. |
| `nn.ModuleList(modules)` | List that registers modules (use instead of a Python list). |
| `nn.ModuleDict(modules)` | Dict that registers modules. |
| `nn.ParameterList`, `nn.ParameterDict` | Same, for parameters. |
| `nn.Identity()` | Returns input unchanged (handy placeholder). |

```python
from torch import nn
blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])   # parameters ARE registered
# blocks = [nn.Linear(16, 16) for _ in range(4)]               # parameters NOT registered (bug)
```

#### nn.Parameter

```python
torch.nn.Parameter(data=None, requires_grad=True)
```

A tensor subclass that is automatically registered when assigned as a module attribute.

```python
import torch
from torch import nn

class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * self.gamma
```

### torch.nn: layers

#### nn.Linear

```python
nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_features` | `int` | required | Size of the last input dimension. |
| `out_features` | `int` | required | Size of the last output dimension. |
| `bias` | `bool` | `True` | Learn an additive bias. |

Computes `y = x @ W.T + b`. Input shape `(*, in_features)`, output `(*, out_features)`.

```python
import torch
from torch import nn
fc = nn.Linear(128, 64)
print(fc(torch.randn(32, 10, 128)).shape)     # torch.Size([32, 10, 64])
```

#### nn.Conv2d

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
          dilation=1, groups=1, bias=True, padding_mode="zeros")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_channels` | `int` | required | Input channels C_in. |
| `out_channels` | `int` | required | Number of filters C_out. |
| `kernel_size` | `int` or tuple | required | Filter size. |
| `stride` | `int` or tuple | `1` | Step size. |
| `padding` | `int`, tuple, `"same"`, `"valid"` | `0` | Implicit padding. `"same"` requires stride 1. |
| `dilation` | `int` or tuple | `1` | Spacing between kernel taps. |
| `groups` | `int` | `1` | `groups=in_channels` gives a depthwise conv. |

Input `(N, C_in, H, W)`, output `(N, C_out, H_out, W_out)` where `H_out = floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)`.

```python
import torch
from torch import nn
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
print(conv(torch.randn(8, 3, 32, 32)).shape)   # torch.Size([8, 16, 32, 32])
```

Related: `nn.Conv1d`, `nn.Conv3d`, `nn.ConvTranspose2d`, `nn.MaxPool2d(kernel_size, stride=None, padding=0)`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d(output_size)`, `nn.Upsample`, `nn.PixelShuffle`.

#### Normalization

```python
nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True)
nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)
nn.RMSNorm(normalized_shape, eps=None, elementwise_affine=True)    # PyTorch 2.4+
```

| Layer | Normalizes over | Typical use |
|---|---|---|
| `BatchNorm2d` | batch + spatial, per channel | CNNs with decent batch sizes |
| `LayerNorm` | last `normalized_shape` dims, per sample | Transformers, RNNs |
| `GroupNorm` | groups of channels, per sample | CNNs with small batches |
| `RMSNorm` | last dims, no mean-centering | Modern LLMs (Llama-style) |

```python
import torch
from torch import nn
ln = nn.LayerNorm(512)
print(ln(torch.randn(2, 10, 512)).shape)
```

#### Activations and regularization

| Module | Functional | Notes |
|---|---|---|
| `nn.ReLU(inplace=False)` | `F.relu` | Default choice for CNNs. |
| `nn.GELU(approximate="none")` | `F.gelu` | Transformers; `approximate="tanh"` for speed. |
| `nn.SiLU()` | `F.silu` | Swish; used in modern CNNs/LLMs (SwiGLU). |
| `nn.LeakyReLU(negative_slope=0.01)` | `F.leaky_relu` | |
| `nn.Sigmoid()`, `nn.Tanh()` | `torch.sigmoid`, `torch.tanh` | |
| `nn.Softmax(dim)` | `F.softmax` | Do not apply before `CrossEntropyLoss`. |
| `nn.Dropout(p=0.5)` | `F.dropout(x, p, training)` | Active only in `train()` mode. |

#### nn.Embedding

```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, sparse=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_embeddings` | `int` | required | Vocabulary size. |
| `embedding_dim` | `int` | required | Vector size. |
| `padding_idx` | `int` | `None` | Index whose vector stays zero and gets no gradient. |

Input: `LongTensor` of indices of any shape; output appends `embedding_dim`.

```python
import torch
from torch import nn
emb = nn.Embedding(10_000, 256, padding_idx=0)
tokens = torch.randint(0, 10_000, (4, 20))
print(emb(tokens).shape)                       # torch.Size([4, 20, 256])
```

#### Attention and Transformer layers

```python
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False)
nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                           activation="relu", batch_first=False, norm_first=False)
nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None,
                           dropout_p=0.0, is_causal=False, scale=None) -> Tensor
```

`F.scaled_dot_product_attention` (SDPA) automatically dispatches to FlashAttention, memory-efficient attention, or a math fallback depending on hardware, dtype, and inputs. Use it in custom attention code instead of a hand-written `softmax(Q K^T) V`.

| Parameter (SDPA) | Type | Default | Description |
|---|---|---|---|
| `query`, `key`, `value` | `Tensor` | required | Shapes `(..., L, E)`, `(..., S, E)`, `(..., S, Ev)`. |
| `attn_mask` | `Tensor` | `None` | Boolean (True = attend) or additive float mask. |
| `dropout_p` | `float` | `0.0` | Attention dropout; set 0 in eval. |
| `is_causal` | `bool` | `False` | Apply a causal mask (do not combine with `attn_mask`). |
| `scale` | `float` | `None` | Defaults to `1/sqrt(E)`. |

```python
import torch
import torch.nn.functional as F
from torch import nn

q = k = v = torch.randn(2, 8, 128, 64)              # (batch, heads, seq, head_dim)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(out.shape)                                     # torch.Size([2, 8, 128, 64])

layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True, norm_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=4)
print(encoder(torch.randn(4, 50, 256)).shape)        # torch.Size([4, 50, 256])
```

### Loss functions

| Loss | Constructor | Input / target | Use |
|---|---|---|---|
| `nn.CrossEntropyLoss` | `(weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0)` | logits `(N, C)` or `(N, C, d1, ...)`; class indices `(N)` or probabilities `(N, C)` | Multi-class classification. Applies log-softmax internally. |
| `nn.NLLLoss` | `(weight=None, ignore_index=-100, reduction="mean")` | log-probs `(N, C)`; indices `(N)` | After your own `log_softmax`. |
| `nn.BCEWithLogitsLoss` | `(weight=None, reduction="mean", pos_weight=None)` | logits and float targets, same shape | Binary / multi-label. Numerically stable. |
| `nn.BCELoss` | `(weight=None, reduction="mean")` | probabilities in [0, 1] | Prefer `BCEWithLogitsLoss`. |
| `nn.MSELoss` | `(reduction="mean")` | same shape | Regression (L2). |
| `nn.L1Loss` | `(reduction="mean")` | same shape | Regression (L1). |
| `nn.SmoothL1Loss` / `nn.HuberLoss` | `(beta=1.0)` / `(delta=1.0)` | same shape | Robust regression, box regression. |
| `nn.KLDivLoss` | `(reduction="mean", log_target=False)` | input log-probs, target probs | Distillation. Use `reduction="batchmean"`. |
| `nn.CosineEmbeddingLoss`, `nn.TripletMarginLoss` | | | Metric learning. |
| `nn.CTCLoss` | `(blank=0, zero_infinity=False)` | | Speech/OCR alignment. |

`reduction` is `"mean"`, `"sum"`, or `"none"` (per-element losses).

```python
import torch
from torch import nn

ce = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
logits = torch.randn(4, 5)
targets = torch.tensor([1, 0, 4, -100])        # last sample ignored
print(ce(logits, targets))

bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
print(bce(torch.randn(8, 1), torch.randint(0, 2, (8, 1)).float()))

# Token-level LM loss: flatten (batch, seq, vocab) -> (batch*seq, vocab)
lm_logits = torch.randn(2, 16, 1000)
lm_labels = torch.randint(0, 1000, (2, 16))
loss = nn.functional.cross_entropy(lm_logits.view(-1, 1000), lm_labels.view(-1))
```

### Optimizers (torch.optim)

All optimizers share the interface `Optimizer(params, lr=..., ...)` with `step()`, `zero_grad(set_to_none=True)`, `state_dict()`, `load_state_dict()`, and `param_groups`.

#### torch.optim.SGD

```python
torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | required | Parameters or list of param-group dicts. |
| `lr` | `float` | `1e-3` | Learning rate. |
| `momentum` | `float` | `0` | Momentum factor (0.9 typical). |
| `weight_decay` | `float` | `0` | L2 penalty added to the gradient. |
| `nesterov` | `bool` | `False` | Nesterov momentum. |

#### torch.optim.Adam / AdamW

```python
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                 amsgrad=False, foreach=None, fused=None)
torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                  amsgrad=False, foreach=None, fused=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr` | `float` | `1e-3` | Learning rate. |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | EMA coefficients for first/second moments. |
| `eps` | `float` | `1e-8` | Numerical stability term. |
| `weight_decay` | `float` | `0` (Adam), `0.01` (AdamW) | Adam: L2 in gradient. AdamW: decoupled decay (preferred). |
| `fused` | `bool` | `None` | Use a fused kernel (fastest on GPU). |
| `foreach` | `bool` | `None` | Multi-tensor implementation (default on CUDA). |

Other optimizers: `RMSprop`, `Adagrad`, `Adadelta`, `Adamax`, `NAdam`, `RAdam`, `LBFGS` (requires a closure), `SparseAdam`, `Adafactor` (2.5+).

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10), nn.Linear(10, 2))

# Param groups: no weight decay for biases and norm weights
decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if p.ndim < 2 else decay).append(p)

optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.05},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4, betas=(0.9, 0.95),
)
print(len(optimizer.param_groups))     # 2
```

### Learning-rate schedulers (torch.optim.lr_scheduler)

Call `scheduler.step()` **after** `optimizer.step()` (per epoch or per iteration depending on how you configured it).

| Scheduler | Constructor | Behavior |
|---|---|---|
| `StepLR` | `(optimizer, step_size, gamma=0.1)` | Multiply LR by `gamma` every `step_size` steps. |
| `MultiStepLR` | `(optimizer, milestones, gamma=0.1)` | Decay at listed milestones. |
| `ExponentialLR` | `(optimizer, gamma)` | LR *= gamma each step. |
| `CosineAnnealingLR` | `(optimizer, T_max, eta_min=0.0)` | Cosine decay over `T_max` steps. |
| `CosineAnnealingWarmRestarts` | `(optimizer, T_0, T_mult=1, eta_min=0.0)` | SGDR restarts. |
| `OneCycleLR` | `(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3)` | Warmup then anneal; step every batch. |
| `LinearLR` | `(optimizer, start_factor=1/3, end_factor=1.0, total_iters=5)` | Linear warmup/decay. |
| `LambdaLR` | `(optimizer, lr_lambda)` | LR = base_lr * lr_lambda(step). |
| `SequentialLR` | `(optimizer, schedulers, milestones)` | Chain schedulers (e.g. warmup then cosine). |
| `ReduceLROnPlateau` | `(optimizer, mode="min", factor=0.1, patience=10)` | Call `step(val_loss)`. |

```python
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

model = torch.nn.Linear(4, 1)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
warmup_steps, total_steps = 100, 1000
sched = SequentialLR(
    opt,
    schedulers=[
        LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-5),
    ],
    milestones=[warmup_steps],
)
for step in range(total_steps):
    opt.step()          # (after backward in real code)
    sched.step()
print(sched.get_last_lr())
```

### Data loading (torch.utils.data)

#### Dataset

Map-style datasets implement `__len__` and `__getitem__`. Iterable-style datasets subclass `IterableDataset` and implement `__iter__` (for streams).

```python
import torch
from torch.utils.data import Dataset

class TensorPairs(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

ds = TensorPairs(torch.randn(100, 3), torch.randint(0, 2, (100,)))
x0, y0 = ds[0]
```

Built-ins: `TensorDataset(*tensors)`, `ConcatDataset(datasets)`, `Subset(dataset, indices)`, `random_split(dataset, lengths, generator=None)` (lengths may be fractions like `[0.8, 0.2]`).

#### DataLoader

```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None,
    batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False,
    drop_last=False, timeout=0, worker_init_fn=None, generator=None,
    prefetch_factor=None, persistent_workers=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `Dataset` | required | Source of samples. |
| `batch_size` | `int` | `1` | Samples per batch. |
| `shuffle` | `bool` | `None` (False) | Reshuffle every epoch. Mutually exclusive with `sampler`. |
| `sampler` | `Sampler` | `None` | Custom index order (e.g. `DistributedSampler`, `WeightedRandomSampler`). |
| `num_workers` | `int` | `0` | Subprocesses for loading. 0 loads in the main process. |
| `collate_fn` | callable | `None` | Merges a list of samples into a batch. |
| `pin_memory` | `bool` | `False` | Page-locked host memory for faster host-to-device copies. |
| `drop_last` | `bool` | `False` | Drop the final incomplete batch. |
| `prefetch_factor` | `int` | `None` (2 when workers > 0) | Batches prefetched per worker. |
| `persistent_workers` | `bool` | `False` | Keep workers alive across epochs. |

Returns an iterable over batches.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence

ds = TensorDataset(torch.randn(1000, 20), torch.randint(0, 3, (1000,)))
train_ds, val_ds = random_split(ds, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

# Custom collate for variable-length sequences
def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True, padding_value=0), lengths, torch.tensor(labels)

seq_ds = [(torch.randint(1, 100, (n,)), n % 2) for n in range(5, 25)]
seq_dl = DataLoader(seq_ds, batch_size=4, collate_fn=collate)
x, lengths, y = next(iter(seq_dl))
print(x.shape, lengths)
```

On Windows and macOS (spawn start method), code that creates `DataLoader`s with `num_workers > 0` must run under `if __name__ == "__main__":`.

### Mixed precision (torch.amp)

The device-generic `torch.amp` API replaces the older `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler`, which are deprecated and emit `FutureWarning`s.

```python
torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)
torch.amp.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)   # same thing
torch.amp.GradScaler(device="cuda", init_scale=65536.0, growth_factor=2.0,
                     backoff_factor=0.5, growth_interval=2000, enabled=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `device_type` | `str` | required | `"cuda"`, `"cpu"`, `"mps"`, `"xpu"`. |
| `dtype` | `torch.dtype` | `float16` on CUDA, `bfloat16` on CPU | Lower-precision compute type. |
| `enabled` | `bool` | `True` | Toggle without changing code. |

Autocast runs matmuls/convolutions in low precision and keeps precision-sensitive ops (softmax, losses, reductions) in float32. `GradScaler` is needed for **float16** (to prevent gradient underflow) but not for **bfloat16**.

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
model = nn.Linear(512, 10).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler(device, enabled=(amp_dtype == torch.float16))
x = torch.randn(64, 512, device=device)
y = torch.randint(0, 10, (64,), device=device)

with torch.autocast(device_type=device, dtype=amp_dtype):
    loss = nn.functional.cross_entropy(model(x), y)

scaler.scale(loss).backward()
scaler.unscale_(opt)                                   # unscale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt)
scaler.update()
opt.zero_grad(set_to_none=True)
```

Related: `torch.set_float32_matmul_precision("high")` (or `torch.backends.cuda.matmul.allow_tf32 = True`) enables TF32 tensor cores on Ampere+ GPUs for float32 matmuls.

### torch.compile

```python
torch.compile(model=None, *, fullgraph=False, dynamic=None, backend="inductor",
              mode=None, options=None, disable=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` or callable | required | What to compile; also usable as a decorator. |
| `fullgraph` | `bool` | `False` | Error on any graph break instead of falling back. |
| `dynamic` | `bool` or `None` | `None` | `None`: auto-detect dynamic shapes after a recompile. `True`: compile dynamically up front. `False`: always specialize. |
| `backend` | `str` | `"inductor"` | Also `"cudagraphs"`, `"aot_eager"`, `"eager"` (debugging), or custom. |
| `mode` | `str` | `None` (`"default"`) | `"reduce-overhead"` (CUDA graphs, good for small batches), `"max-autotune"` (slow compile, fastest kernels), `"max-autotune-no-cudagraphs"`. |
| `options` | `dict` | `None` | Backend-specific Inductor options. |

Returns an optimized callable (an `OptimizedModule` when given an `nn.Module`; the original is at `._orig_mod`).

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024))
opt_model = torch.compile(model)                 # or model.compile() in place (2.2+)

@torch.compile(fullgraph=True)
def fused_gelu_bias(x, b):
    return torch.nn.functional.gelu(x + b)

x = torch.randn(64, 1024)
y = opt_model(x)                                 # first call compiles (slow), later calls are fast
z = fused_gelu_bias(x, torch.zeros(1024))
```

Debugging helpers:

| Tool | Purpose |
|---|---|
| `TORCH_LOGS="graph_breaks,recompiles"` env var | Print where and why graphs break / recompile. |
| `torch._dynamo.explain(fn)(*args)` | Summary of graphs and break reasons. |
| `torch.compiler.disable` (decorator) | Exclude a function from compilation. |
| `torch.compiler.reset()` | Clear compilation caches. |
| `TORCH_COMPILE_DEBUG=1` | Dump generated code and IR. |

Note: `state_dict()` on a compiled `OptimizedModule` prefixes keys with `_orig_mod.`; save `model._orig_mod.state_dict()`, keep a reference to the uncompiled model, or use `model.compile()` (in place) to avoid this.

### Saving and loading

```python
torch.save(obj, f, pickle_protocol=2) -> None
torch.load(f, map_location=None, weights_only=True, mmap=None) -> Any
```

| Parameter (`load`) | Type | Default | Description |
|---|---|---|---|
| `f` | path or file-like | required | Checkpoint source. |
| `map_location` | str, device, dict, callable | `None` | Remap storages, e.g. `"cpu"` to load GPU checkpoints on CPU. |
| `weights_only` | `bool` | `True` since 2.6 (`False` before) | Restrict unpickling to tensors, primitive types, and allow-listed classes. Safer. |
| `mmap` | `bool` | `None` | Memory-map the file instead of reading it all into RAM. |

Best practice is to save **state dicts**, not whole pickled modules.

```python
import torch
from torch import nn

model = nn.Linear(4, 2)
opt = torch.optim.AdamW(model.parameters())

torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": 5}, "ckpt.pt")

ckpt = torch.load("ckpt.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])
opt.load_state_dict(ckpt["optimizer"])
start_epoch = ckpt["epoch"] + 1
```

Loading a large model without allocating random weights first:

```python
import torch
from torch import nn

torch.save(nn.Linear(4096, 4096).state_dict(), "big.pt")

with torch.device("meta"):
    big = nn.Linear(4096, 4096)                    # shapes only, no memory allocated
sd = torch.load("big.pt", mmap=True, weights_only=True)
big.load_state_dict(sd, assign=True)               # adopt the loaded tensors directly
print(big.weight.device)                           # cpu
```

`safetensors` (`from safetensors.torch import save_file, load_file`) is the common zero-copy, pickle-free format used across the Hugging Face ecosystem.

### torch.export and deployment

```python
torch.export.export(mod, args, kwargs=None, *, dynamic_shapes=None, strict=...) -> ExportedProgram
torch.export.save(ep, f)
torch.export.load(f) -> ExportedProgram
torch.onnx.export(model, args, f, input_names=None, output_names=None, dynamic_axes=None,
                  opset_version=None, dynamo=False)
```

`torch.export` produces a sound, whole-graph `ExportedProgram` (no graph breaks, no Python) for AOT compilation (AOTInductor), ExecuTorch, or other runtimes. TorchScript (`torch.jit.script` / `torch.jit.trace`) still works but is in maintenance mode.

```python
import torch
from torch import nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return torch.relu(self.fc(x))

batch = torch.export.Dim("batch", min=1, max=1024)
ep = torch.export.export(M(), (torch.randn(2, 8),), dynamic_shapes={"x": {0: batch}})
print(ep.module()(torch.randn(16, 8)).shape)     # torch.Size([16, 4])
torch.export.save(ep, "m.pt2")

torch.onnx.export(M(), (torch.randn(1, 8),), "m.onnx",
                  input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "batch"}})
```

### CUDA utilities (torch.cuda)

| API | Description |
|---|---|
| `torch.cuda.is_available()` | GPU usable. |
| `torch.cuda.device_count()` | Number of GPUs. |
| `torch.cuda.set_device(i)` | Default GPU for this process. |
| `torch.cuda.synchronize()` | Wait for all kernels (needed for timing). |
| `torch.cuda.empty_cache()` | Release cached blocks back to the driver (does not free live tensors). |
| `torch.cuda.memory_allocated()` / `max_memory_allocated()` | Bytes held by tensors. |
| `torch.cuda.memory_reserved()` | Bytes held by the caching allocator. |
| `torch.cuda.reset_peak_memory_stats()` | Reset peaks. |
| `torch.cuda.memory_summary()` | Human-readable allocator report. |
| `torch.cuda.Event(enable_timing=True)` | GPU timing events. |
| `torch.cuda.Stream()` / `torch.cuda.stream(s)` | Concurrent execution streams. |

### Distributed basics (torch.distributed)

#### Process group

```python
torch.distributed.init_process_group(backend=None, init_method=None, timeout=None,
                                     world_size=-1, rank=-1, device_id=None)
torch.distributed.get_rank() -> int
torch.distributed.get_world_size() -> int
torch.distributed.barrier()
torch.distributed.destroy_process_group()
```

| Backend | Use |
|---|---|
| `"nccl"` | NVIDIA GPUs (fastest GPU collectives). ROCm uses RCCL under the same name. |
| `"gloo"` | CPU, and a fallback. |
| `"xccl"` | Intel GPUs (recent releases). |

Collectives: `all_reduce(tensor, op=ReduceOp.SUM)`, `all_gather(tensor_list, tensor)`, `all_gather_into_tensor`, `reduce_scatter_tensor`, `broadcast(tensor, src)`, `send` / `recv`.

Launch with `torchrun`, which sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`:

```bash
torchrun --nproc_per_node=4 train.py
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=host0:29500 train.py
```

#### DistributedDataParallel

```python
torch.nn.parallel.DistributedDataParallel(module, device_ids=None, output_device=None,
    broadcast_buffers=True, find_unused_parameters=False, gradient_as_bucket_view=False,
    static_graph=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `module` | `nn.Module` | required | Model already on the right device. |
| `device_ids` | `list[int]` | `None` | `[local_rank]` for single-device-per-process GPU training. |
| `find_unused_parameters` | `bool` | `False` | Needed if some parameters do not receive grads every step (slower). |
| `gradient_as_bucket_view` | `bool` | `False` | Save memory by aliasing grads to communication buckets. |

DDP replicates the model in each process and all-reduces gradients during `backward()`. Pair it with `DistributedSampler` (call `sampler.set_epoch(epoch)` every epoch). `nn.DataParallel` is legacy and slower; do not use it for new code.

#### FSDP

Fully Sharded Data Parallel shards parameters, gradients, and optimizer state across ranks. Two APIs exist:

- **FSDP1**: `torch.distributed.fsdp.FullyShardedDataParallel(module, auto_wrap_policy=..., mixed_precision=..., sharding_strategy=...)`.
- **FSDP2** (recommended for new code; public as `torch.distributed.fsdp.fully_shard` since 2.6): `fully_shard(module, mesh=None, reshard_after_forward=True, mp_policy=...)`, applied per block then to the root, built on `DTensor`.

```python
# Sketch of FSDP2 usage (run under torchrun after init_process_group)
from torch.distributed.fsdp import fully_shard

def shard(model):
    for block in model.layers:        # shard each transformer block
        fully_shard(block)
    fully_shard(model)                # then the root
    return model
```

Device meshes for multi-dimensional parallelism: `torch.distributed.device_mesh.init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))`. Distributed checkpointing: `torch.distributed.checkpoint` (`dcp.save`, `dcp.load`).

## Tutorials

### Tutorial 1: Image classification on FashionMNIST with a CNN

This example trains a small convolutional network end to end: dataset download, transforms, a model, a training loop with mixed precision, evaluation, and checkpointing.

```bash
pip install torch torchvision
```

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # 1. Data: ToTensor scales to [0, 1], Normalize centers the data.
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2,
                          pin_memory=(device == "cuda"), persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=512, num_workers=2)

    # 2. Model: two conv blocks then a classifier head.
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),                                   # 28 -> 14
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),                                   # 14 -> 7
        nn.Flatten(),
        nn.Dropout(0.25),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    # 3. Optimization setup.
    epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=epochs, steps_per_epoch=len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 4. Train.
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()                     # OneCycleLR steps every batch
            running += loss.item() * x.size(0)

        # 5. Evaluate.
        model.eval()
        correct = 0
        with torch.inference_mode():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        print(f"epoch {epoch}: train loss {running / len(train_ds):.4f}, "
              f"test acc {correct / len(test_ds):.4f}")

    # 6. Save weights only.
    torch.save(model.state_dict(), "fashion_cnn.pt")

if __name__ == "__main__":
    main()
```

What each step does:

1. `transforms.ToTensor()` converts PIL images to `(C, H, W)` float tensors; normalization stats are the dataset mean/std.
2. `nn.Sequential` is enough for a linear stack. `nn.Flatten()` turns `(N, 64, 7, 7)` into `(N, 3136)`.
3. `OneCycleLR` warms up then anneals and is stepped every batch.
4. Autocast plus `GradScaler` gives float16 speedups on GPUs; both are no-ops on CPU because `enabled=False`.
5. `model.eval()` switches BatchNorm to running statistics and disables Dropout; `inference_mode` disables autograd.
6. Saving the `state_dict` keeps the checkpoint portable. Expect roughly 91-92 percent test accuracy after 5 epochs.

### Tutorial 2: Transfer learning with a pretrained ResNet

Fine-tune a torchvision ResNet-18 on an `ImageFolder` dataset (one sub-directory per class). The multi-weight API (`weights=...`) replaced the old `pretrained=True` flag.

```text
data/flowers/
  train/daisy/*.jpg
  train/rose/*.jpg
  val/daisy/*.jpg
  val/rose/*.jpg
```

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = models.ResNet18_Weights.IMAGENET1K_V1

    # 1. Use the exact preprocessing the weights were trained with.
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = weights.transforms()          # resize 256, center crop 224, normalize

    train_ds = datasets.ImageFolder("data/flowers/train", transform=train_tfm)
    val_ds = datasets.ImageFolder("data/flowers/val", transform=val_tfm)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=4)
    num_classes = len(train_ds.classes)

    # 2. Load the backbone and replace the classification head.
    model = models.resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False                         # freeze everything
    model.fc = nn.Linear(model.fc.in_features, num_classes)   # new head is trainable
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(train: bool, optimizer=None):
        model.train(train)
        loader = train_dl if train else val_dl
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
        return loss_sum / total, correct / total

    # 3. Stage 1: train only the head.
    head_opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
    for epoch in range(3):
        tl, ta = run_epoch(True, head_opt)
        vl, va = run_epoch(False)
        print(f"[head] epoch {epoch}: train acc {ta:.3f} val acc {va:.3f}")

    # 4. Stage 2: unfreeze the last block and fine-tune with a lower LR.
    for p in model.layer4.parameters():
        p.requires_grad = True
    ft_opt = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)
    best = 0.0
    for epoch in range(5):
        run_epoch(True, ft_opt)
        _, va = run_epoch(False)
        if va > best:
            best = va
            torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, "best_resnet.pt")
        print(f"[finetune] epoch {epoch}: val acc {va:.3f} (best {best:.3f})")

if __name__ == "__main__":
    main()
```

Key ideas: freezing with `requires_grad = False` saves compute and memory because no gradients are computed for frozen weights; discriminative learning rates via param groups protect pretrained features; and `weights.transforms()` guarantees evaluation preprocessing matches what the backbone expects.

### Tutorial 3: A tiny GPT language model from scratch

A character-level decoder-only transformer using `F.scaled_dot_product_attention`, `torch.compile`, bfloat16 autocast, gradient clipping, and sampling.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Data: any plain text. We embed a short sample so the script is self-contained.
text = ("To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer "
        "The slings and arrows of outrageous fortune, "
        "Or to take arms against a sea of troubles ") * 200
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size, batch_size = 64, 32

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size - 1, (batch_size,))
    x = torch.stack([src[i:i + block_size] for i in ix])
    y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# 2. Model.
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))      # pre-norm residual
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight             # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

model = TinyGPT(len(chars)).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

# 3. Optimizer, schedule, compile.
max_steps, warmup = 1000, 100
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
lr_lambda = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, max_steps) / max_steps))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
train_model = torch.compile(model) if device == "cuda" else model

@torch.no_grad()
def estimate_loss(iters=20):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.stack([model(*get_batch(split))[1] for _ in range(iters)])
        out[split] = losses.mean().item()
    model.train()
    return out

# 4. Train with bfloat16 autocast (no GradScaler needed for bf16).
amp_dtype = torch.bfloat16
for step in range(max_steps):
    x, y = get_batch("train")
    with torch.autocast(device_type=device, dtype=amp_dtype):
        _, loss = train_model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    if step % 200 == 0:
        print(step, estimate_loss())

# 5. Sample.
start = torch.tensor([encode("To be")], device=device)
print(decode(model.generate(start, 200, temperature=0.8, top_k=10)[0].tolist()))
```

Notes: the compiled wrapper (`train_model`) shares parameters with `model`, so evaluation and generation can use the uncompiled module (avoiding recompiles on varying sequence lengths). Weight tying shares the embedding and output projection, a standard trick in language models.

### Tutorial 4: Multi-GPU training with DistributedDataParallel

Save as `ddp_train.py` and launch with `torchrun --nproc_per_node=NUM_GPUS ddp_train.py`. It also runs on CPU with the `gloo` backend (`torchrun --nproc_per_node=2 ddp_train.py`).

```python
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def setup():
    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    return dist.get_rank(), dist.get_world_size(), local_rank, device

def main():
    rank, world_size, local_rank, device = setup()
    torch.manual_seed(0)

    # 1. Every rank builds the same dataset; the sampler gives each rank a disjoint shard.
    X = torch.randn(10_000, 32)
    y = (X.sum(dim=1) > 0).long()
    ds = TensorDataset(X, y)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=128, sampler=sampler, pin_memory=device.type == "cuda")

    # 2. Model on this rank's device, wrapped in DDP.
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 2)).to(device)
    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)                 # different shuffle each epoch
        model.train()
        total = torch.zeros(2, device=device)    # [loss_sum, count]
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()                      # gradients are all-reduced here
            optimizer.step()
            total += torch.tensor([loss.item() * len(xb), len(xb)], device=device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)   # aggregate metrics across ranks
        if rank == 0:
            print(f"epoch {epoch}: loss {(total[0] / total[1]).item():.4f}")

    # 3. Save from rank 0 only, using the unwrapped module.
    if rank == 0:
        torch.save(model.module.state_dict(), "ddp_model.pt")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Key points: one process per GPU; `DistributedSampler` plus `set_epoch` for correct sharding and shuffling; `model.module` to reach the wrapped model; collective metric aggregation with `all_reduce`; only rank 0 writes files. Effective batch size is `batch_size * world_size`.

## Performance & Best Practices

### Throughput checklist

| Technique | How | Typical gain |
|---|---|---|
| Mixed precision | `torch.autocast("cuda", dtype=torch.bfloat16)` (Ampere+) or float16 + `GradScaler` | 1.5-3x, half the activation memory |
| TF32 matmuls | `torch.set_float32_matmul_precision("high")` | Large for float32 matmul on Ampere+ |
| Compile | `model = torch.compile(model)` | 1.3-2x for many models |
| Fused optimizer | `torch.optim.AdamW(..., fused=True)` | Fewer kernel launches |
| SDPA attention | `F.scaled_dot_product_attention` | FlashAttention kernels, less memory |
| Data pipeline | `num_workers>0`, `pin_memory=True`, `persistent_workers=True`, `non_blocking=True` | Removes input-bound stalls |
| Avoid host syncs | Do not call `.item()`, `.cpu()`, `print(t)` every step | Keeps the GPU queue full |
| cuDNN autotune | `torch.backends.cudnn.benchmark = True` for fixed input sizes | Faster convolutions |
| Channels last | `model.to(memory_format=torch.channels_last)` for CNNs with AMP | 10-30 percent on tensor cores |
| Larger batches | Use the largest batch that fits; scale LR accordingly | Better utilization |
| CUDA graphs | `torch.compile(mode="reduce-overhead")` | Removes CPU launch overhead for small models |

### Memory checklist

- **Gradient accumulation**: simulate large batches by calling `backward()` on `loss / accum_steps` several times before `optimizer.step()`.
- **Activation checkpointing**: `torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)` trades compute for memory.
- **`zero_grad(set_to_none=True)`** (the default since 2.0) frees gradient memory between steps.
- **`inference_mode()`** for evaluation to avoid storing activations.
- **Lower precision weights**: bfloat16 models for inference; 8-bit optimizers via third-party libraries (bitsandbytes).
- **Shard**: FSDP/FSDP2 for models whose parameters + optimizer state do not fit on one GPU.
- **Delete references** to large tensors (`del loss, logits`) when holding them across iterations, and avoid accumulating `loss` (a graph-holding tensor) into Python lists; use `loss.item()` or `loss.detach()`.

```python
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

model = nn.Sequential(*[nn.Sequential(nn.Linear(512, 512), nn.GELU()) for _ in range(8)])
opt = torch.optim.AdamW(model.parameters())
accum_steps = 4

for i in range(16):
    x = torch.randn(32, 512)
    h = x
    for block in model:
        h = checkpoint(block, h, use_reentrant=False)     # recompute activations in backward
    loss = h.pow(2).mean() / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
```

### Code-quality practices

- Make device a single variable and create tensors directly on it (`torch.zeros(..., device=device)`).
- Save `state_dict`s (or safetensors), never whole pickled models; load with `weights_only=True`.
- Register non-parameter state with `register_buffer` so it moves with `.to()` and is saved.
- Use `nn.ModuleList` / `nn.ModuleDict`, not Python lists/dicts, for sub-modules.
- Always pair `model.eval()` with `torch.no_grad()` / `inference_mode()` during evaluation.
- Seed everything and log library versions for reproducibility.
- Profile before optimizing (`torch.profiler`, `nvidia-smi`, Nsight Systems).
- Prefer `torch.amp` over the deprecated `torch.cuda.amp` namespace.
- For compiled models, avoid data-dependent Python control flow on tensor values in hot paths (it causes graph breaks) and mark dynamic dimensions if shapes vary.

## Common Errors & Troubleshooting

### CUDA out of memory

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 23.65 GiB of which 1.12 GiB is free. Of the allocated memory 20.10 GiB is allocated by PyTorch, and 1.30 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

Older releases raise `torch.cuda.OutOfMemoryError` / `RuntimeError: CUDA out of memory`.

Causes and fixes:

- Batch too large: reduce `batch_size`, use gradient accumulation.
- Activations too large: use autocast (bf16/fp16), activation checkpointing, shorter sequences.
- Evaluation without `no_grad`: wrap eval in `torch.inference_mode()`.
- Accumulating graph-holding tensors: `total_loss += loss` keeps every graph alive; use `loss.item()`.
- Fragmentation: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Another process holds memory: check `nvidia-smi`.
- Debug with `torch.cuda.memory_summary()` or the memory snapshot tool (`torch.cuda.memory._record_memory_history()`).

### Device mismatch

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

Cause: an input, target, mask, or a tensor created inside `forward` is on CPU while the model is on GPU (or vice versa). Fix: move all inputs with `.to(device)`; create new tensors with `device=x.device`; register constant tensors as buffers so `model.to()` moves them.

```python
import torch
from torch import nn

class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(2.0))   # moves with the module
    def forward(self, x):
        mask = torch.ones(x.shape[0], device=x.device)     # created on the input's device
        return x * self.scale * mask[:, None]
```

### Dtype mismatch

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
RuntimeError: expected scalar type Long but found Float
```

Cause: NumPy defaults to float64, so `torch.from_numpy` gives `torch.float64` tensors; class-index targets must be `torch.long` for `CrossEntropyLoss`; BCE targets must be float. Fix: `.float()` inputs, `.long()` class targets, `.float()` BCE targets.

### Shape mismatch

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x3136 and 1568x128)
```

Cause: the `in_features` of a `Linear` after a `Flatten` does not match the actual flattened size. Fix: print shapes, or use `nn.LazyLinear(out_features)` which infers `in_features` on the first call.

```text
ValueError: Expected input batch_size (32) to match target batch_size (2048).
```

Cause: passing `(N, T, C)` logits to `CrossEntropyLoss` without flattening, or a mis-shaped target. Fix: `loss_fn(logits.view(-1, C), targets.view(-1))` or permute to `(N, C, T)`.

### In-place modification breaks autograd

```text
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [10, 10]], which is output 0 of ReluBackward0, is at version 1; expected version 0 instead.
```

Cause: an in-place op (`x += 1`, `relu_`, `nn.ReLU(inplace=True)`, `tensor[idx] = ...`) overwrote a value saved for backward. Fix: use out-of-place versions (`x = x + 1`), and enable `torch.autograd.set_detect_anomaly(True)` to locate the culprit.

### Backward twice

```text
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).
```

Cause: reusing part of a graph across iterations (e.g. a hidden state in an RNN carried over without `.detach()`), or calling `backward()` twice. Fix: `hidden = hidden.detach()` between batches, or pass `retain_graph=True` if a second backward is genuinely intended.

### No gradient

```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

Cause: computing the loss inside `torch.no_grad()`, parameters frozen, or the computation went through `.detach()`, `.item()`, or NumPy. Fix: keep the path in torch ops with grad enabled.

### Loading checkpoints

```text
_pickle.UnpicklingError: Weights only load failed. ... WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default.
```

Cause: since 2.6, `torch.load` defaults to `weights_only=True`, which refuses arbitrary pickled objects. Fix: store only tensors and primitives; or allow-list the type with `torch.serialization.add_safe_globals([...])`; or, only for files you trust, pass `weights_only=False`.

```text
RuntimeError: Error(s) in loading state_dict for Net: Missing key(s) in state_dict: "fc.weight" ... Unexpected key(s) in state_dict: "module.fc.weight" ...
```

Cause: checkpoint saved from a DDP-wrapped (`module.`) or compiled (`_orig_mod.`) model. Fix: save `model.module.state_dict()` / `model._orig_mod.state_dict()`, or strip the prefix:

```python
sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
```

### GPU not detected

`torch.cuda.is_available()` returns `False`. Causes: a CPU-only wheel installed (`torch.version.cuda` is `None`; reinstall with a CUDA index URL), an NVIDIA driver too old for the wheel's CUDA version (update driver; check `nvidia-smi`), `CUDA_VISIBLE_DEVICES` set to empty, or running inside a container without `--gpus all`.

### Device-side assert

```text
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
```

Usually an out-of-range index: a label `>= num_classes` in `CrossEntropyLoss`, or a token id `>= num_embeddings` in `nn.Embedding`. Fix: run with `CUDA_LAUNCH_BLOCKING=1` or on CPU to get a readable error, and validate label/token ranges.

### DataLoader worker issues

```text
RuntimeError: DataLoader worker (pid 12345) is killed by signal: Killed.
RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
```

The first is usually host RAM exhaustion (reduce `num_workers` or per-sample memory) or small shared memory in Docker (`--shm-size=8g` or `--ipc=host`). The second occurs on Windows/macOS: put the training code under `if __name__ == "__main__":`.

### torch.compile issues

- Frequent recompiles: `torch._dynamo hit config.recompile_limit (8)` (named `cache_size_limit` in older releases). Cause: changing shapes or Python values each call. Fix: `dynamic=True`, `torch._dynamo.mark_dynamic(x, 0)`, or avoid passing varying Python scalars.
- `Unsupported: ...` with `fullgraph=True`: a construct Dynamo cannot trace. Remove `fullgraph=True`, move the code outside the compiled region, or wrap it with `torch.compiler.disable`.
- On Windows, the Inductor backend needs a working C++ compiler (MSVC) for CPU and Triton for GPU; if unavailable, use `backend="aot_eager"` to check correctness.

### NaN losses

Causes: learning rate too high, float16 overflow (use bf16 or `GradScaler`), `log(0)` (use `log_softmax`/`BCEWithLogitsLoss` rather than manual log of probabilities), division by zero in normalization. Fixes: gradient clipping, lower LR or warmup, `torch.autograd.set_detect_anomaly(True)`, check data for NaN/inf with `torch.isfinite(x).all()`.

## Interoperability

### NumPy

`torch.from_numpy` and `Tensor.numpy()` share memory on CPU. GPU tensors must be moved first: `t.detach().cpu().numpy()`. Most NumPy functions also accept CPU tensors via the `__array__` protocol.

### DLPack (JAX, CuPy, TensorFlow, others)

Zero-copy exchange of device memory between frameworks:

```python
import torch
import jax.numpy as jnp
import jax

t = torch.arange(6.0)
j = jnp.from_dlpack(t)            # torch -> JAX (zero-copy when devices match)
t2 = torch.from_dlpack(j)         # JAX -> torch
print(j, t2)
```

`torch.utils.dlpack.to_dlpack` / `from_dlpack` provide the capsule-based API. CuPy (`cupy.from_dlpack`) and TensorFlow (`tf.experimental.dlpack`) interoperate the same way.

### Hugging Face ecosystem

Transformers models are `nn.Module`s: they work with your own training loops, `torch.compile`, AMP, DDP, and FSDP. `safetensors` is the standard checkpoint format; `accelerate` wraps device placement, mixed precision, and distributed launch around plain PyTorch loops; `peft` adds LoRA adapters to any `nn.Module`.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
batch = tok(["great movie", "terrible movie"], return_tensors="pt", padding=True)
out = model(**batch, labels=torch.tensor([1, 0]))
out.loss.backward()            # a regular PyTorch loss
```

### Lightning, torchvision, torchaudio, timm

- **PyTorch Lightning / Lightning Fabric** organize training loops (`LightningModule`, `Trainer`) while staying plain PyTorch inside.
- **torchvision** provides datasets, transforms (`torchvision.transforms.v2`), and pretrained models; **torchaudio** does the same for audio; **timm** provides hundreds of image backbones.

### scikit-learn and pandas

Convert DataFrames with `torch.tensor(df.values, dtype=torch.float32)`; use scikit-learn for preprocessing (`StandardScaler`), splitting, and metrics on NumPy outputs. `skorch` wraps PyTorch modules in a scikit-learn estimator API.

### ONNX, TensorRT, and serving

- `torch.onnx.export` produces ONNX for ONNX Runtime, TensorRT, OpenVINO.
- `torch.export` plus AOTInductor yields a compiled shared library for C++ deployment; ExecuTorch targets mobile/edge.
- Serving engines such as vLLM, SGLang, TGI, and NVIDIA Triton Inference Server load PyTorch checkpoints directly.

### TPUs

`torch_xla` runs PyTorch on Cloud TPUs using XLA (the same compiler JAX uses), with `xm.xla_device()` as the device.

## Cheat Sheet

### Tensors

| Task | Code |
|---|---|
| From list | `torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)` |
| From NumPy (shared) | `torch.from_numpy(arr)` |
| Zeros / ones / random | `torch.zeros(2, 3)`, `torch.ones(2, 3)`, `torch.randn(2, 3)` |
| Range | `torch.arange(0, 10, 2)` |
| Same shape as x | `torch.zeros_like(x)` |
| Move to GPU | `x.to("cuda")` or `x.to(device, non_blocking=True)` |
| To NumPy | `x.detach().cpu().numpy()` |
| To Python number | `x.item()` |
| Reshape | `x.view(-1, 4)` / `x.reshape(2, -1)` |
| Swap dims | `x.transpose(0, 1)` / `x.permute(0, 2, 1)` |
| Add / remove dim | `x.unsqueeze(0)` / `x.squeeze(-1)` |
| Concatenate / stack | `torch.cat([a, b], dim=0)` / `torch.stack([a, b])` |
| Batched matmul | `a @ b` or `torch.bmm(a, b)` |
| Einsum | `torch.einsum("bij,bjk->bik", a, b)` |
| Argmax | `logits.argmax(dim=-1)` |
| Boolean mask | `x[x > 0]`, `x.masked_fill(mask, 0)` |
| Conditional | `torch.where(cond, a, b)` |

### Models and training

| Task | Code |
|---|---|
| Define model | `class Net(nn.Module): def __init__...; def forward(self, x): ...` |
| Count parameters | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| Freeze module | `module.requires_grad_(False)` |
| Train / eval mode | `model.train()` / `model.eval()` |
| No grad | `with torch.inference_mode():` |
| Optimizer | `torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)` |
| Step | `opt.zero_grad(); loss.backward(); opt.step()` |
| Clip grads | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |
| Cosine LR | `torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)` |
| Classification loss | `nn.CrossEntropyLoss()(logits, labels)` |
| Binary / multi-label loss | `nn.BCEWithLogitsLoss()(logits, targets.float())` |
| Regression loss | `nn.MSELoss()(pred, y)` |
| Mixed precision | `with torch.autocast("cuda", dtype=torch.bfloat16): ...` |
| fp16 scaler | `scaler = torch.amp.GradScaler("cuda")` |
| Compile | `model = torch.compile(model)` |
| Attention | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |

### Data, I/O, devices

| Task | Code |
|---|---|
| Dataset from tensors | `TensorDataset(X, y)` |
| Split | `random_split(ds, [0.8, 0.2])` |
| Loader | `DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)` |
| Pad sequences | `torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)` |
| Save weights | `torch.save(model.state_dict(), "m.pt")` |
| Load weights | `model.load_state_dict(torch.load("m.pt", map_location="cpu", weights_only=True))` |
| Export | `torch.export.export(model, (example,))` |
| ONNX | `torch.onnx.export(model, (example,), "m.onnx")` |
| Pick device | `device = "cuda" if torch.cuda.is_available() else "cpu"` |
| Seed | `torch.manual_seed(0)` |
| GPU memory | `torch.cuda.max_memory_allocated() / 2**30` |
| Multi-GPU launch | `torchrun --nproc_per_node=8 train.py` |
| DDP wrap | `DDP(model.to(local_rank), device_ids=[local_rank])` |

## Further Resources

- Official website and install selector: https://pytorch.org/get-started/locally/
- Documentation: https://pytorch.org/docs/stable/index.html
- Tutorials: https://pytorch.org/tutorials/
- "Learn the Basics" tutorial series: https://pytorch.org/tutorials/beginner/basics/intro.html
- torch.compile tutorial: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- Automatic mixed precision docs: https://pytorch.org/docs/stable/amp.html
- Distributed overview: https://pytorch.org/tutorials/beginner/dist_overview.html
- FSDP2 / fully_shard docs: https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- torch.export docs: https://pytorch.org/docs/stable/export.html
- GitHub repository: https://github.com/pytorch/pytorch
- PyTorch examples: https://github.com/pytorch/examples
- Forums: https://discuss.pytorch.org/
- Blog and release notes: https://pytorch.org/blog/
- Paper: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS 2019: https://arxiv.org/abs/1912.01703
- Paper: Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation", ASPLOS 2024: https://dl.acm.org/doi/10.1145/3620665.3640366
- Book: "Deep Learning with PyTorch" (Stevens, Antiga, Viehmann), Manning: https://www.manning.com/books/deep-learning-with-pytorch
- Course: Andrej Karpathy, "Neural Networks: Zero to Hero" (built on PyTorch): https://karpathy.ai/zero-to-hero.html
- Course: fast.ai "Practical Deep Learning for Coders": https://course.fast.ai/
