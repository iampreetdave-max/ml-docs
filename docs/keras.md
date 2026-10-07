# Keras

> A high-level, multi-backend deep learning API: write a model once, run it on TensorFlow, JAX, or PyTorch.

Keras is a deep learning API focused on fast experimentation and readable code. You build models from layers, compile them with an optimizer, a loss, and metrics, and train with `fit()`. Keras 3 is a full rewrite that runs on top of **TensorFlow, JAX, or PyTorch** (and, for inference, OpenVINO), introduces a backend-agnostic tensor API (`keras.ops`), and uses a single native file format (`.keras`).

Covers Keras 3 (the 3.x releases). Where Keras 2 (`tf.keras` before TF 2.16, now the `tf_keras` package) behaves differently, it is called out.

## Overview

### What Keras is

Keras was created by Francois Chollet and first released in March 2015. It originally ran on Theano and TensorFlow; from 2017 it was integrated into TensorFlow as `tf.keras`, and Keras 2.x became effectively TensorFlow-only. In November 2023 the Keras team (at Google) released **Keras 3.0**, which restored multi-backend support. Since TensorFlow 2.16 (March 2024), `pip install tensorflow` installs Keras 3 and `tf.keras` points to it.

Key features of Keras 3:

| Feature | Description |
|---|---|
| Multi-backend | Same model code runs on TensorFlow, JAX, or PyTorch; choose with `KERAS_BACKEND`. |
| `keras.ops` | NumPy-like op namespace plus neural network ops that work on every backend. |
| Three model-building styles | `Sequential`, Functional API, and Model subclassing. |
| Built-in training loop | `fit`, `evaluate`, `predict` with callbacks, metrics, and distribution. |
| Data-agnostic | `fit` accepts NumPy arrays, `tf.data.Dataset`, PyTorch `DataLoader`, and `keras.utils.PyDataset`. |
| `.keras` format | Zip archive with config, weights, and metadata; safe loading by default. |
| Ecosystem | KerasHub (pretrained models: Gemma, Llama, BERT, Stable Diffusion, etc.), KerasTuner, `keras.applications`. |
| Distribution API | `keras.distribution` for data and model parallelism (JAX backend). |

### When to use Keras

- You want to build, train, and iterate on neural networks quickly with minimal boilerplate.
- Standard architectures: CNNs, RNNs, Transformers, autoencoders, tabular MLPs.
- You want backend flexibility: prototype with one framework, deploy with another (for example train on JAX/TPU, export via TensorFlow SavedModel).
- Teaching and readable reference implementations.

### When not to use Keras

- Highly unusual research code that fights the layer/model abstraction at every step; plain PyTorch or JAX may be simpler.
- Classical tabular ML on small data: gradient boosting (XGBoost, LightGBM, CatBoost) usually wins.
- If your team's entire stack (data loaders, distributed tooling, model zoo) is native PyTorch and you never need another backend, Keras adds a layer you may not need (although Keras models with the torch backend are `torch.nn.Module`s).

### Where it fits in the ML stack

```text
Data: NumPy / pandas / tf.data / torch DataLoader / PyDataset
      |
Keras (layers, models, losses, optimizers, metrics, callbacks)
      |
Backend: TensorFlow | JAX | PyTorch      (selected by KERAS_BACKEND)
      |
Hardware: CPU / GPU / TPU
      |
Export: .keras file | TF SavedModel (model.export) -> TF Serving, LiteRT/TFLite | ONNX (recent 3.x)
```

### Keras 2 versus Keras 3 at a glance

| Topic | Keras 2 (`tf_keras`) | Keras 3 |
|---|---|---|
| Backends | TensorFlow only | TensorFlow, JAX, PyTorch |
| Ops | `tf.*` functions | `keras.ops.*` (backend-agnostic); `tf.*` still works on the TF backend |
| Save whole model | `.keras`, `.h5`, or SavedModel dir | `.keras` (recommended) or `.h5`; SavedModel only via `model.export()` |
| Save weights | any name | filename must end in `.weights.h5` |
| Preprocessing layers | some under `layers.experimental.preprocessing` | all in `keras.layers` |
| Image augmentation | `ImageDataGenerator` common | preprocessing layers + `image_dataset_from_directory` |
| `fit(workers=..., use_multiprocessing=...)` | supported | moved to `PyDataset` constructor |
| Optimizers | `tf.keras.optimizers` (new and legacy) | `keras.optimizers` (one implementation per optimizer) |

## Installation

### Install Keras plus a backend

```bash
pip install --upgrade keras
```

Keras itself is pure Python; install at least one backend:

```bash
# TensorFlow backend (default). Linux GPU: tensorflow[and-cuda]
pip install tensorflow
pip install "tensorflow[and-cuda]"

# JAX backend. CPU:
pip install jax
# JAX with NVIDIA GPU (CUDA 12):
pip install "jax[cuda12]"

# PyTorch backend: pick the command for your platform/CUDA at https://pytorch.org
pip install torch
```

Installing TensorFlow 2.16+ automatically installs Keras 3. If an older `tensorflow` pins Keras 2, upgrade with `pip install --upgrade keras` after TensorFlow (or upgrade TensorFlow).

### Selecting the backend

The backend is chosen once, at import time.

```bash
export KERAS_BACKEND="jax"        # or "tensorflow", "torch"
python train.py
```

Or in Python, before the first `import keras`:

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
print(keras.config.backend())   # "torch"
```

Or persistently in `~/.keras/keras.json`:

```json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "jax",
    "image_data_format": "channels_last"
}
```

The environment variable overrides the config file. Changing the backend after `import keras` has no effect in the same process.

### Legacy Keras 2 with TensorFlow 2.16+

```bash
pip install tf_keras
export TF_USE_LEGACY_KERAS=1    # set before importing tensorflow
```

Then `tf.keras` resolves to Keras 2 (`tf_keras`). Use this only to keep old code running while migrating.

### Verifying the install and GPU visibility

```python
import keras
print("Keras", keras.__version__, "backend:", keras.config.backend())
```

Backend-specific GPU checks:

```python
# TensorFlow
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))

# JAX
import jax
print(jax.devices())

# PyTorch
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
```

## Core Concepts

### Layers

A layer is a callable object that holds state (weights) and computation (`call`). Layers are created lazily: weights are built the first time the layer sees an input, so their shapes can be inferred.

```python
import numpy as np
import keras
from keras import layers

dense = layers.Dense(3, activation="relu")
x = np.ones((2, 5), dtype="float32")
y = dense(x)                       # builds kernel (5, 3) and bias (3,)
print(y.shape, [w.shape for w in dense.weights])
```

### Models: three ways to build

**Sequential**: a plain stack, one input and one output.

```python
model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
```

**Functional API**: a directed acyclic graph of layers; supports multiple inputs/outputs, shared layers, and residual connections.

```python
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
skip = x
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.Add()([x, skip])                     # residual connection
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs, name="tiny_resnet")
model.summary()
```

**Subclassing**: full control in Python.

```python
class MLP(keras.Model):
    def __init__(self, hidden=64, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.d1 = layers.Dense(hidden, activation="relu")
        self.drop = layers.Dropout(0.2)
        self.out = layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.d1(x)
        x = self.drop(x, training=training)
        return self.out(x)

model = MLP()
print(model(np.ones((2, 20), dtype="float32")).shape)
```

Prefer Sequential or Functional where possible: they give you `summary()` with shapes, `plot_model`, and saving without custom code. Use subclassing when the forward pass has dynamic Python logic.

### The compile / fit workflow

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
model.evaluate(x_test, y_test)
probs = model.predict(x_test[:5])
```

- `compile` configures; it does not train.
- `fit` returns a `History` object with `history.history["loss"]`, `["val_loss"]`, etc.
- Losses, optimizers, and metrics can be given as objects or string identifiers (`"adam"`, `"mse"`, `"accuracy"`).

### keras.ops: backend-agnostic tensor math

`keras.ops` implements the NumPy API plus neural-network ops. Code written with `keras.ops` runs on every backend.

```python
from keras import ops

x = ops.arange(6, dtype="float32")
x = ops.reshape(x, (2, 3))
print(ops.mean(x, axis=1), ops.softmax(x), ops.matmul(x, ops.transpose(x)))
print(ops.convert_to_numpy(x))     # back to NumPy from any backend
```

Tensors returned by Keras are the native tensor type of the backend (`tf.Tensor`, `jax.Array`, or `torch.Tensor`).

### Trainable state and freezing

```python
base = keras.applications.MobileNetV3Small(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base.trainable = False           # freeze all weights (recompile after changing)
print(len(base.trainable_weights), len(base.non_trainable_weights))
```

Setting `trainable = False` on a `BatchNormalization` layer also makes it run in inference mode, which is what you want during fine-tuning.

### Dtype policies and mixed precision

```python
keras.config.set_dtype_policy("mixed_float16")   # compute in float16, variables in float32
# keras.config.set_dtype_policy("mixed_bfloat16") for TPUs / recent GPUs
```

Keep the output layer in float32 (`layers.Activation("softmax", dtype="float32")`). With `mixed_float16`, `compile` automatically applies loss scaling.

### Serialization

Every built-in object has `get_config()`; models rebuild from configs. Custom layers must implement `get_config()` and be registered with `@keras.saving.register_keras_serializable()` to load from a `.keras` file without passing `custom_objects`.

## API Reference

All examples assume:

```python
import numpy as np
import keras
from keras import layers, ops
```

### Models

#### keras.Input

```python
keras.Input(shape=None, batch_size=None, dtype=None, sparse=None, batch_shape=None, name=None, tensor=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shape` | tuple | None | Shape excluding the batch dimension; use `None` for variable dims, e.g. `(None,)` for variable-length sequences. |
| `batch_size` | int | None | Fixed batch size (rarely needed). |
| `dtype` | str | `"float32"` | e.g. `"int32"`, `"string"` (TF backend only for strings). |
| `sparse` | bool | None | Whether the input is sparse. |
| `batch_shape` | tuple | None | Full shape including batch dimension (alternative to `shape`). |
| `name` | str | None | Input name; used to match dict inputs in `fit`. |

Returns: a symbolic `KerasTensor` used to build Functional models.

#### keras.Sequential

```python
keras.Sequential(layers=None, trainable=True, name=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `layers` | list | None | Layers in order. Start with `keras.Input(...)` to build immediately. |
| `name` | str | None | Model name. |

Methods: `add(layer)`, `pop()`, plus everything on `Model`.

```python
model = keras.Sequential(name="mlp")
model.add(keras.Input(shape=(16,)))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(1))
```

#### keras.Model

```python
keras.Model(inputs, outputs, name=None)        # Functional
class MyModel(keras.Model): ...                # Subclassing
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `inputs` | KerasTensor, list, or dict | required | Model inputs from `keras.Input`. |
| `outputs` | KerasTensor, list, or dict | required | Outputs computed from inputs. |
| `name` | str | None | Model name. |

Common attributes and methods:

| Member | Description |
|---|---|
| `summary(line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False)` | Print architecture. |
| `layers` | List of layers. |
| `get_layer(name=None, index=None)` | Retrieve a layer. |
| `weights`, `trainable_weights`, `non_trainable_weights` | Variables. |
| `get_weights()` / `set_weights(weights)` | Weights as NumPy arrays. |
| `count_params()` | Number of parameters. |
| `inputs`, `outputs`, `input_shape`, `output_shape` | Graph info (Functional/Sequential). |
| `trainable` | Freeze/unfreeze (recompile after changing). |
| `to_json()` / `keras.models.model_from_json(json)` | Architecture only. |
| `get_config()` / `from_config(config)` | Serialization. |

#### Model.compile

```python
model.compile(optimizer="rmsprop", loss=None, loss_weights=None, metrics=None,
              weighted_metrics=None, run_eagerly=False, steps_per_execution=1,
              jit_compile="auto", auto_scale_loss=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | str or Optimizer | `"rmsprop"` | Optimizer instance or name. |
| `loss` | str, callable, Loss, list, dict | None | Loss per output (dict keys = output names). |
| `loss_weights` | list or dict | None | Weights for multi-output losses. |
| `metrics` | list or dict | None | Metrics to track. |
| `weighted_metrics` | list | None | Metrics that use `sample_weight`. |
| `run_eagerly` | bool | False | Disable compilation (debugging). |
| `steps_per_execution` | int | 1 | Batches per compiled call; higher reduces Python overhead (useful on TPU). |
| `jit_compile` | bool or `"auto"` | `"auto"` | XLA compilation (TF/JAX); `"auto"` enables it where supported. |
| `auto_scale_loss` | bool | True | Apply loss scaling automatically under `mixed_float16`. |

#### Model.fit

```python
model.fit(x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None,
          validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
          sample_weight=None, initial_epoch=0, steps_per_epoch=None,
          validation_steps=None, validation_batch_size=None, validation_freq=1)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | array, dict, list, `tf.data.Dataset`, `PyDataset`, torch `DataLoader`, generator | None | Inputs (or `(x, y)` / `(x, y, sample_weight)` datasets). |
| `y` | array | None | Targets; omit when `x` is a dataset. |
| `batch_size` | int | 32 | Only for array inputs. |
| `epochs` | int | 1 | Number of epochs. |
| `verbose` | `"auto"`, 0, 1, 2 | `"auto"` | 0 silent, 1 progress bar, 2 one line per epoch. |
| `callbacks` | list | None | Callback instances. |
| `validation_split` | float | 0.0 | Fraction of array data held out (taken from the end, before shuffling). |
| `validation_data` | tuple or dataset | None | Explicit validation data. |
| `shuffle` | bool | True | Shuffle array data each epoch (ignored for datasets). |
| `class_weight` | dict | None | `{class_index: weight}` for imbalance. |
| `sample_weight` | array | None | Per-sample weights. |
| `initial_epoch` | int | 0 | Resume epoch counter. |
| `steps_per_epoch` | int | None | Batches per epoch (required for infinite datasets). |

Returns: `History` (`history.history` dict of lists, `history.epoch`).

#### Model.evaluate and Model.predict

```python
model.evaluate(x=None, y=None, batch_size=None, verbose="auto", sample_weight=None,
               steps=None, callbacks=None, return_dict=False)
model.predict(x, batch_size=None, verbose="auto", steps=None, callbacks=None)
```

`evaluate` returns the loss (scalar) or `[loss, metric1, ...]`, or a dict with `return_dict=True`. `predict` returns NumPy arrays (or a structure of arrays for multi-output models).

Batch-level methods: `train_on_batch(x, y=None, sample_weight=None, class_weight=None, return_dict=False)`, `test_on_batch(...)`, `predict_on_batch(x)`.

For a single small batch in a latency-sensitive loop, calling `model(x, training=False)` directly is faster than `predict`, which is optimized for large inputs.

#### keras.models helpers

| Function | Description |
|---|---|
| `keras.models.load_model(filepath, custom_objects=None, compile=True, safe_mode=True)` | Load a `.keras` (or `.h5`) model. |
| `keras.models.clone_model(model, input_tensors=None, clone_function=None)` | New model with same architecture, fresh weights. |
| `keras.models.model_from_json(json_string, custom_objects=None)` | Rebuild architecture. |

### Core layers

#### Dense

```python
layers.Dense(units, activation=None, use_bias=True, kernel_initializer="glorot_uniform",
             bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None,
             activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | int | required | Output dimension. |
| `activation` | str or callable | None | e.g. `"relu"`, `"gelu"`, `"sigmoid"`, `"softmax"`. |
| `use_bias` | bool | True | Add a bias vector. |
| `kernel_initializer` | str or Initializer | `"glorot_uniform"` | e.g. `"he_normal"` for ReLU nets. |
| `kernel_regularizer` | Regularizer | None | e.g. `keras.regularizers.L2(1e-4)`. |

Applies `activation(inputs @ kernel + bias)` to the last axis.

#### Other core layers

| Layer | Signature | Purpose |
|---|---|---|
| `Activation` | `Activation(activation)` | Apply an activation. |
| `Dropout` | `Dropout(rate, noise_shape=None, seed=None)` | Randomly zero inputs during training. |
| `Flatten` | `Flatten(data_format=None)` | Collapse non-batch dims. |
| `Reshape` | `Reshape(target_shape)` | Reshape non-batch dims. |
| `Permute` | `Permute(dims)` | Reorder axes. |
| `RepeatVector` | `RepeatVector(n)` | Repeat a vector n times. |
| `Embedding` | `Embedding(input_dim, output_dim, embeddings_initializer="uniform", mask_zero=False)` | Integer IDs to dense vectors. |
| `Lambda` | `Lambda(function, output_shape=None)` | Wrap a simple function (avoid for anything you need to save portably). |
| `Masking` | `Masking(mask_value=0.0)` | Skip timesteps equal to `mask_value`. |
| `TimeDistributed` | `TimeDistributed(layer)` | Apply a layer to each timestep. |

#### Merging layers

| Layer | Signature | Purpose |
|---|---|---|
| `Concatenate` | `Concatenate(axis=-1)` | Join tensors. Functional shortcut: `layers.concatenate([a, b])`. |
| `Add` / `Subtract` / `Multiply` / `Average` / `Maximum` / `Minimum` | `Add()` | Elementwise merge. |

### Convolution and pooling layers

#### Conv2D

```python
layers.Conv2D(filters, kernel_size, strides=(1, 1), padding="valid", data_format=None,
              dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
              kernel_initializer="glorot_uniform", bias_initializer="zeros",
              kernel_regularizer=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filters` | int | required | Number of output channels. |
| `kernel_size` | int or tuple | required | e.g. `3` or `(3, 3)`. |
| `strides` | int or tuple | `(1, 1)` | Step size. |
| `padding` | str | `"valid"` | `"valid"` (no padding) or `"same"` (output size = input / stride). |
| `data_format` | str | None | `"channels_last"` (NHWC, default) or `"channels_first"`. |
| `dilation_rate` | int or tuple | `(1, 1)` | Atrous convolution. |
| `groups` | int | 1 | Grouped convolution. |
| `activation` | str | None | Activation. |

Input `(batch, height, width, channels)`; output `(batch, new_h, new_w, filters)`.

| Layer family | Members |
|---|---|
| Convolution | `Conv1D`, `Conv2D`, `Conv3D`, `SeparableConv1D`, `SeparableConv2D`, `DepthwiseConv2D`, `Conv1DTranspose`, `Conv2DTranspose`, `Conv3DTranspose` |
| Pooling | `MaxPooling1D/2D/3D(pool_size, strides=None, padding="valid")`, `AveragePooling1D/2D/3D`, `GlobalMaxPooling1D/2D/3D`, `GlobalAveragePooling1D/2D/3D` |
| Shape helpers | `ZeroPadding2D`, `Cropping2D`, `UpSampling2D(size=(2, 2), interpolation="nearest")` |

### Normalization and regularization layers

| Layer | Signature | Notes |
|---|---|---|
| `BatchNormalization` | `BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)` | Uses batch stats in training, moving averages in inference. |
| `LayerNormalization` | `LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True)` | Standard in Transformers. |
| `GroupNormalization` | `GroupNormalization(groups=32, axis=-1, epsilon=0.001)` | Batch-size independent. |
| `UnitNormalization` | `UnitNormalization(axis=-1)` | L2-normalize. |
| `Dropout`, `SpatialDropout2D`, `GaussianNoise`, `GaussianDropout`, `AlphaDropout` | | Regularization. |

### Recurrent layers

#### LSTM

```python
layers.LSTM(units, activation="tanh", recurrent_activation="sigmoid", use_bias=True,
            dropout=0.0, recurrent_dropout=0.0, return_sequences=False,
            return_state=False, go_backwards=False, stateful=False, unroll=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | int | required | Hidden state size. |
| `return_sequences` | bool | False | Return output for every timestep `(batch, time, units)` instead of last `(batch, units)`. |
| `return_state` | bool | False | Also return final hidden and cell states. |
| `dropout` / `recurrent_dropout` | float | 0.0 | Input / recurrent dropout. Non-zero `recurrent_dropout` disables the fast cuDNN kernel. |
| `stateful` | bool | False | Carry state across batches. |

Related: `GRU(units, ..., reset_after=True)`, `SimpleRNN`, `Bidirectional(layer, merge_mode="concat")`, `ConvLSTM2D`, `RNN(cell)` for custom cells.

```python
model = keras.Sequential([
    keras.Input(shape=(None,), dtype="int32"),
    layers.Embedding(10_000, 64, mask_zero=True),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.LSTM(16),
    layers.Dense(1, activation="sigmoid"),
])
```

### Attention layers

#### MultiHeadAttention

```python
layers.MultiHeadAttention(num_heads, key_dim, value_dim=None, dropout=0.0, use_bias=True,
                          output_shape=None, attention_axes=None)
# call:
mha(query, value, key=None, attention_mask=None, return_attention_scores=False,
    training=None, use_causal_mask=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_heads` | int | required | Number of attention heads. |
| `key_dim` | int | required | Size of each head for query/key. |
| `value_dim` | int | None | Head size for values (defaults to `key_dim`). |
| `dropout` | float | 0.0 | Attention dropout. |
| `use_causal_mask` (call) | bool | False | Prevent attending to future positions (decoders). |

Returns `(batch, target_len, query_dim)` and optionally attention scores.

```python
x = keras.Input(shape=(None, 64))
attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)   # self-attention
x2 = layers.LayerNormalization()(x + attn)
```

Also available: `Attention` (dot-product), `AdditiveAttention`, `GroupedQueryAttention`.

### Preprocessing layers

Preprocessing layers can live inside the model (portable, avoids training/serving skew) or inside a `tf.data` pipeline (runs on CPU in parallel). Stateful ones are fitted with `adapt(data)`.

| Layer | Signature | Purpose |
|---|---|---|
| `Normalization` | `Normalization(axis=-1, mean=None, variance=None, invert=False)` | Standardize features; `adapt` computes mean/variance. |
| `Rescaling` | `Rescaling(scale, offset=0.0)` | e.g. `Rescaling(1/255)`. |
| `Resizing` | `Resizing(height, width, interpolation="bilinear", crop_to_aspect_ratio=False)` | Resize images. |
| `CenterCrop` | `CenterCrop(height, width)` | Crop center. |
| `RandomFlip` | `RandomFlip(mode="horizontal_and_vertical", seed=None)` | Augmentation (training only). |
| `RandomRotation` | `RandomRotation(factor, fill_mode="reflect", seed=None)` | Factor is a fraction of 2 pi. |
| `RandomZoom` | `RandomZoom(height_factor, width_factor=None)` | Augmentation. |
| `RandomTranslation`, `RandomContrast`, `RandomBrightness`, `RandomCrop` | | Augmentation. |
| `TextVectorization` | `TextVectorization(max_tokens=None, standardize="lower_and_strip_punctuation", split="whitespace", ngrams=None, output_mode="int", output_sequence_length=None)` | Strings to token IDs, multi-hot, counts, or TF-IDF. |
| `StringLookup` | `StringLookup(max_tokens=None, num_oov_indices=1, vocabulary=None, output_mode="int")` | String categories to indices / one-hot. |
| `IntegerLookup` | `IntegerLookup(max_tokens=None, num_oov_indices=1, vocabulary=None, output_mode="int")` | Integer categories to indices. |
| `CategoryEncoding` | `CategoryEncoding(num_tokens=None, output_mode="multi_hot")` | Indices to one-hot/multi-hot/count. |
| `Hashing` | `Hashing(num_bins, mask_value=None, salt=None)` | Hash trick for high-cardinality features. |
| `Discretization` | `Discretization(bin_boundaries=None, num_bins=None)` | Bucketize numeric values. |

```python
norm = layers.Normalization()
norm.adapt(np.array([[1.0], [2.0], [3.0], [4.0]]))
print(norm(np.array([[2.5]])))                      # approximately 0

lookup = layers.StringLookup(output_mode="one_hot")
lookup.adapt(np.array(["red", "green", "blue", "red"]))
print(lookup.get_vocabulary())                      # ['[UNK]', 'red', ...]
```

`TextVectorization` (and string lookups on string inputs) rely on TensorFlow internally. With the JAX or PyTorch backend, use them inside a `tf.data` pipeline rather than inside the model.

### keras.ops (selected)

| Category | Functions |
|---|---|
| Creation | `ops.zeros`, `ops.ones`, `ops.full`, `ops.arange`, `ops.linspace`, `ops.eye`, `ops.convert_to_tensor`, `ops.convert_to_numpy` |
| Shape | `ops.shape`, `ops.reshape`, `ops.transpose`, `ops.expand_dims`, `ops.squeeze`, `ops.concatenate`, `ops.stack`, `ops.split`, `ops.tile`, `ops.repeat` |
| Math | `ops.add`, `ops.multiply`, `ops.matmul`, `ops.einsum`, `ops.exp`, `ops.log`, `ops.sqrt`, `ops.abs`, `ops.clip`, `ops.power` |
| Reductions | `ops.sum`, `ops.mean`, `ops.max`, `ops.min`, `ops.argmax`, `ops.argmin`, `ops.logsumexp`, `ops.top_k` |
| Logic and indexing | `ops.where`, `ops.take`, `ops.take_along_axis`, `ops.slice`, `ops.cond`, `ops.any`, `ops.all` |
| NN | `ops.relu`, `ops.gelu`, `ops.sigmoid`, `ops.softmax`, `ops.log_softmax`, `ops.one_hot`, `ops.conv`, `ops.max_pool`, `ops.dot_product_attention`, `ops.binary_crossentropy`, `ops.sparse_categorical_crossentropy` |
| Control | `ops.cast`, `ops.stop_gradient`, `ops.fori_loop`, `ops.while_loop`, `ops.scan`, `ops.vectorized_map` |
| Random | `keras.random.normal(shape, mean=0.0, stddev=1.0, seed=None)`, `keras.random.uniform`, `keras.random.randint`, `keras.random.dropout`, `keras.random.SeedGenerator(seed)` |

```python
logits = ops.convert_to_tensor([[2.0, 1.0, 0.1]])
print(ops.softmax(logits), ops.argmax(logits, axis=-1))
seed = keras.random.SeedGenerator(42)
print(keras.random.normal((2, 2), seed=seed))
```

Inside layers that draw random numbers, store a `SeedGenerator` on the layer (created in `__init__`) so behavior is reproducible and works under JIT compilation.

### Custom layers

```python
class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform", trainable=True, name="w")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros",
                                 trainable=True, name="b")

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
```

`add_weight(shape=None, initializer=None, dtype=None, trainable=True, regularizer=None, constraint=None, name=None)` creates a backend variable. Add losses with `self.add_loss(value)` inside `call`. Use only `keras.ops` in `call` to stay backend-agnostic.

### Losses

| Class | Signature (main args) | Use |
|---|---|---|
| `BinaryCrossentropy` | `(from_logits=False, label_smoothing=0.0, axis=-1)` | Binary / multi-label. |
| `BinaryFocalCrossentropy` | `(apply_class_balancing=False, alpha=0.25, gamma=2.0, from_logits=False)` | Imbalanced binary. |
| `CategoricalCrossentropy` | `(from_logits=False, label_smoothing=0.0, axis=-1)` | One-hot labels. |
| `SparseCategoricalCrossentropy` | `(from_logits=False, ignore_class=None)` | Integer labels. |
| `CategoricalFocalCrossentropy` | `(alpha=0.25, gamma=2.0, from_logits=False)` | Imbalanced multiclass. |
| `MeanSquaredError` | `()` | Regression. |
| `MeanAbsoluteError` | `()` | Robust regression. |
| `Huber` | `(delta=1.0)` | Robust regression. |
| `LogCosh` | `()` | Smooth robust regression. |
| `KLDivergence` | `()` | Distribution matching. |
| `CosineSimilarity` | `(axis=-1)` | Embedding similarity (returns negative similarity). |
| `Hinge`, `SquaredHinge`, `CategoricalHinge` | `()` | Max-margin. |
| `Poisson` | `()` | Count data. |
| `Tversky`, `Dice` | | Segmentation (recent 3.x). |

All losses accept `reduction` (default `"sum_over_batch_size"`; also `"sum"`, `"mean"`, `"mean_with_sample_weight"`, or `None` for per-sample values) and `name`. String aliases: `"mse"`, `"mae"`, `"binary_crossentropy"`, `"categorical_crossentropy"`, `"sparse_categorical_crossentropy"`, `"huber"`.

```python
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(np.array([1, 0]), np.array([[0.1, 2.0], [3.0, 0.2]])))
```

Custom loss: any function `loss(y_true, y_pred)` returning per-sample values, or a `keras.losses.Loss` subclass with `call(self, y_true, y_pred)`.

### Optimizers

#### Adam

```python
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                      amsgrad=False, weight_decay=None, clipnorm=None, clipvalue=None,
                      global_clipnorm=None, use_ema=False, ema_momentum=0.99,
                      gradient_accumulation_steps=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | float or schedule | 0.001 | Step size. |
| `beta_1`, `beta_2` | float | 0.9, 0.999 | Moment decay rates. |
| `epsilon` | float | 1e-7 | Numerical stability. |
| `weight_decay` | float | None | Decoupled weight decay (use `AdamW` for the standard form). |
| `clipnorm` / `clipvalue` / `global_clipnorm` | float | None | Gradient clipping. |
| `use_ema` | bool | False | Keep an exponential moving average of weights. |
| `gradient_accumulation_steps` | int | None | Accumulate gradients over several batches. |

All optimizers share the clipping, EMA, and accumulation arguments.

| Optimizer | Key defaults |
|---|---|
| `SGD` | `learning_rate=0.01, momentum=0.0, nesterov=False` |
| `RMSprop` | `learning_rate=0.001, rho=0.9, momentum=0.0` |
| `AdamW` | `learning_rate=0.001, weight_decay=0.004` |
| `Adagrad`, `Adadelta`, `Adamax`, `Nadam`, `Ftrl` | Classic variants. |
| `Lion` | `learning_rate=0.001, beta_1=0.9, beta_2=0.99` |
| `Adafactor` | Memory-efficient for large models. |
| `LossScaleOptimizer` | `LossScaleOptimizer(inner_optimizer)` for float16 custom loops. |

Methods: `apply(grads, trainable_variables=None)`, `apply_gradients(grads_and_vars)`, `learning_rate` (property), `iterations`.

#### Learning rate schedules

| Schedule | Signature |
|---|---|
| `ExponentialDecay` | `(initial_learning_rate, decay_steps, decay_rate, staircase=False)` |
| `CosineDecay` | `(initial_learning_rate, decay_steps, alpha=0.0, warmup_target=None, warmup_steps=0)` |
| `CosineDecayRestarts` | `(initial_learning_rate, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0)` |
| `PiecewiseConstantDecay` | `(boundaries, values)` |
| `PolynomialDecay` | `(initial_learning_rate, decay_steps, end_learning_rate=0.0001, power=1.0)` |
| `InverseTimeDecay` | `(initial_learning_rate, decay_steps, decay_rate, staircase=False)` |

```python
steps = 10_000
schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0, decay_steps=steps, warmup_target=1e-3, warmup_steps=500)
optimizer = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=0.01, global_clipnorm=1.0)
```

Schedules are evaluated per optimizer step (batch), not per epoch.

### Metrics

| Metric | Signature (main args) | Notes |
|---|---|---|
| `Accuracy` | `()` | Exact match. |
| `BinaryAccuracy` | `(threshold=0.5)` | Use `threshold=0.0` with logits. |
| `CategoricalAccuracy` | `()` | One-hot labels. |
| `SparseCategoricalAccuracy` | `()` | Integer labels. |
| `TopKCategoricalAccuracy` / `SparseTopKCategoricalAccuracy` | `(k=5)` | Top-k. |
| `AUC` | `(num_thresholds=200, curve="ROC", from_logits=False, multi_label=False)` | ROC or PR (`curve="PR"`). |
| `Precision` / `Recall` | `(thresholds=None, top_k=None, class_id=None)` | Binary or per-class. |
| `F1Score` / `FBetaScore` | `(average=None, threshold=None)` | Expects one-hot / multi-hot labels. |
| `MeanSquaredError`, `RootMeanSquaredError`, `MeanAbsoluteError`, `MeanAbsolutePercentageError` | `()` | Regression. |
| `R2Score` | `()` | Coefficient of determination. |
| `MeanIoU` | `(num_classes)` | Segmentation. |
| `Mean`, `Sum` | `()` | Generic running aggregates. |

Methods: `update_state(y_true, y_pred, sample_weight=None)`, `result()`, `reset_state()`.

```python
m = keras.metrics.AUC()
m.update_state([0, 1, 1, 0], [0.1, 0.8, 0.6, 0.3])
print(float(m.result()))
```

Strings in `compile(metrics=[...])`: `"accuracy"` resolves to the right accuracy variant for the loss; also `"mae"`, `"mse"`, `"auc"`.

### Callbacks

| Callback | Signature (main args) | Purpose |
|---|---|---|
| `ModelCheckpoint` | `(filepath, monitor="val_loss", verbose=0, save_best_only=False, save_weights_only=False, mode="auto", save_freq="epoch", initial_value_threshold=None)` | Save model or weights. `filepath` must end in `.keras`, or `.weights.h5` with `save_weights_only=True`. Supports `{epoch:02d}` and `{val_loss:.3f}` placeholders. |
| `EarlyStopping` | `(monitor="val_loss", min_delta=0, patience=0, verbose=0, mode="auto", baseline=None, restore_best_weights=False, start_from_epoch=0)` | Stop when no improvement. |
| `ReduceLROnPlateau` | `(monitor="val_loss", factor=0.1, patience=10, verbose=0, mode="auto", min_delta=0.0001, cooldown=0, min_lr=0.0)` | Lower LR on plateau. |
| `LearningRateScheduler` | `(schedule, verbose=0)` | `schedule(epoch, lr) -> new_lr`. |
| `TensorBoard` | `(log_dir="logs", histogram_freq=0, write_graph=True, write_images=False, update_freq="epoch", profile_batch=0)` | TensorBoard logging. |
| `CSVLogger` | `(filename, separator=",", append=False)` | Epoch results to CSV. |
| `BackupAndRestore` | `(backup_dir, save_freq="epoch", delete_checkpoint=True)` | Resume interrupted training automatically. |
| `TerminateOnNaN` | `()` | Stop when loss becomes NaN. |
| `LambdaCallback` | `(on_epoch_begin=None, on_epoch_end=None, on_train_begin=None, on_train_end=None, on_train_batch_begin=None, on_train_batch_end=None)` | Quick inline callbacks. |
| `ProgbarLogger`, `RemoteMonitor`, `SwapEMAWeights` | | Misc. |

Custom callback:

```python
class PrintLR(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        print(f"epoch {epoch}: lr={float(ops.convert_to_numpy(lr)):.6f} val_loss={logs.get('val_loss')}")
```

Hooks: `on_train_begin/end`, `on_epoch_begin/end`, `on_train_batch_begin/end`, `on_test_*`, `on_predict_*`. Set `self.model.stop_training = True` to stop.

### Saving and serialization

| API | Description |
|---|---|
| `model.save(filepath, overwrite=True)` | Save whole model; `filepath` must end in `.keras` (recommended) or `.h5` (legacy). |
| `keras.models.load_model(filepath, custom_objects=None, compile=True, safe_mode=True)` | Load a saved model. |
| `model.save_weights(filepath, overwrite=True)` | Weights only; filename must end in `.weights.h5`. |
| `model.load_weights(filepath, skip_mismatch=False)` | Load weights into an identical architecture. |
| `model.export(filepath, format="tf_saved_model")` | Inference-only export (SavedModel for TF Serving / TFLite). Recent 3.x releases add other formats such as `"onnx"`. |
| `keras.layers.TFSMLayer(filepath, call_endpoint="serving_default")` | Reload an exported SavedModel as a layer (TF backend). |
| `@keras.saving.register_keras_serializable(package="Custom", name=None)` | Register custom objects for loading. |
| `keras.saving.serialize_keras_object(obj)` / `deserialize_keras_object(config)` | Low-level (de)serialization. |
| `keras.config.enable_unsafe_deserialization()` | Allow loading `Lambda` layers with Python lambdas (trusted files only). |

```python
model.save("model.keras")
restored = keras.models.load_model("model.keras")

model.save_weights("model.weights.h5")
model.load_weights("model.weights.h5")
```

A `.keras` file is a zip containing `config.json`, `metadata.json`, and `model.weights.h5`. It is backend-independent: a model saved with the JAX backend loads with the PyTorch backend if it uses only Keras layers and `keras.ops`.

### Data utilities

| Function | Signature (main args) | Returns |
|---|---|---|
| `image_dataset_from_directory` | `(directory, labels="inferred", label_mode="int", class_names=None, color_mode="rgb", batch_size=32, image_size=(256, 256), shuffle=True, seed=None, validation_split=None, subset=None, interpolation="bilinear", crop_to_aspect_ratio=False)` | `tf.data.Dataset` of `(images, labels)`. `subset="both"` returns `(train, val)`. |
| `text_dataset_from_directory` | `(directory, labels="inferred", label_mode="int", batch_size=32, max_length=None, shuffle=True, seed=None, validation_split=None, subset=None)` | `tf.data.Dataset` of `(strings, labels)`. |
| `audio_dataset_from_directory` | `(directory, ..., sampling_rate=None, output_sequence_length=None)` | Audio dataset. |
| `timeseries_dataset_from_array` | `(data, targets, sequence_length, sequence_stride=1, sampling_rate=1, batch_size=128, shuffle=False, seed=None, start_index=None, end_index=None)` | Windowed dataset. |
| `split_dataset` | `(dataset, left_size=None, right_size=None, shuffle=False, seed=None)` | Two datasets. |
| `to_categorical` | `(x, num_classes=None)` | One-hot NumPy array. |
| `pad_sequences` | `(sequences, maxlen=None, dtype="int32", padding="pre", truncating="pre", value=0.0)` | Padded NumPy array. |
| `normalize` | `(x, axis=-1, order=2)` | Normalized array. |
| `get_file` | `(fname=None, origin=None, extract=False, file_hash=None, cache_dir=None)` | Local path of a downloaded file. |
| `set_random_seed` | `(seed)` | Seeds Python, NumPy, and the backend. |
| `plot_model` | `(model, to_file="model.png", show_shapes=False, show_layer_names=True, expand_nested=False, dpi=200)` | Diagram (needs `pydot` and Graphviz). |

The directory loaders return `tf.data` datasets and therefore require TensorFlow to be installed, but the datasets can feed models on any backend.

#### keras.utils.PyDataset

```python
class MyData(keras.utils.PyDataset):
    def __init__(self, x, y, batch_size=32, **kwargs):
        super().__init__(**kwargs)          # workers=1, use_multiprocessing=False, max_queue_size=10
        self.x, self.y, self.batch_size = x, y, batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        lo = idx * self.batch_size
        hi = min(lo + self.batch_size, len(self.x))
        return self.x[lo:hi], self.y[lo:hi]

train = MyData(np.random.rand(1000, 8), np.random.randint(0, 2, 1000), batch_size=64, workers=4)
```

`PyDataset` replaces Keras 2's `keras.utils.Sequence`; parallelism options moved from `fit()` to its constructor.

### keras.applications

Pretrained ImageNet models with a common signature:

```python
keras.applications.ResNet50(include_top=True, weights="imagenet", input_tensor=None,
                            input_shape=None, pooling=None, classes=1000,
                            classifier_activation="softmax")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_top` | bool | True | Include the ImageNet classifier head. |
| `weights` | str or None | `"imagenet"` | `None` for random init, or a path. |
| `input_shape` | tuple | None | Required when `include_top=False` and non-default size. |
| `pooling` | str | None | `"avg"` or `"max"` global pooling when `include_top=False`. |
| `classes` | int | 1000 | Classes when `include_top=True` and `weights=None`. |

Families: `VGG16/19`, `ResNet50/101/152` (+`V2`), `InceptionV3`, `Xception`, `MobileNet`, `MobileNetV2`, `MobileNetV3Small/Large`, `DenseNet121/169/201`, `EfficientNetB0`-`B7`, `EfficientNetV2B0`-`L`, `ConvNeXtTiny`-`XLarge`, `NASNetMobile/Large`. Each module has `preprocess_input` (e.g. `keras.applications.resnet50.preprocess_input`) and `decode_predictions`. For modern pretrained models (Gemma, Llama, BERT, ViT, Stable Diffusion) use KerasHub.

### Backend configuration (keras.config)

| Function | Description |
|---|---|
| `keras.config.backend()` | Active backend name. |
| `keras.config.floatx()` / `set_floatx(value)` | Default float dtype. |
| `keras.config.set_dtype_policy(policy)` / `dtype_policy()` | Global mixed precision policy. |
| `keras.config.image_data_format()` / `set_image_data_format(fmt)` | `"channels_last"` or `"channels_first"`. |
| `keras.config.enable_unsafe_deserialization()` | Allow unsafe loading. |
| `keras.config.disable_traceback_filtering()` | Show full stack traces when debugging. |
| `keras.backend.clear_session()` | Reset global state (layer name counters, memory in notebooks). |

### Distribution (keras.distribution)

```python
import keras

devices = keras.distribution.list_devices()
data_parallel = keras.distribution.DataParallel(devices=devices)
keras.distribution.set_distribution(data_parallel)
# Models created after this point train data-parallel across all devices.
```

`DataParallel`, `ModelParallel(layout_map=..., batch_dim_name=...)`, `DeviceMesh`, and `LayoutMap` are currently implemented for the JAX backend. With the TensorFlow backend use `tf.distribute.MirroredStrategy().scope()`; with PyTorch use standard PyTorch DDP.

## Tutorials

### Tutorial 1: MNIST digit classifier (Sequential CNN), backend-agnostic

This runs unchanged on TensorFlow, JAX, or PyTorch.

```python
import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(42)

# 1. Data: NumPy arrays, scaled to [0, 1], with a channel axis.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")[..., None] / 255.0     # (60000, 28, 28, 1)
x_test = x_test.astype("float32")[..., None] / 255.0

# 2. Model
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])
model.summary()

# 3. Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 4. Train with checkpointing and early stopping
callbacks = [
    keras.callbacks.ModelCheckpoint("best_mnist.keras", monitor="val_accuracy", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
]
history = model.fit(x_train, y_train, batch_size=128, epochs=10,
                    validation_split=0.1, callbacks=callbacks)

# 5. Evaluate and predict
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"test accuracy: {acc:.4f}")
pred = model.predict(x_test[:5], verbose=0)
print("predicted:", np.argmax(pred, axis=1), "true:", y_test[:5])

# 6. Reload the best model
best = keras.models.load_model("best_mnist.keras")
print(best.evaluate(x_test, y_test, verbose=0))
```

Step by step:

1. Keras expects channels-last images by default, so a trailing axis of size 1 is added.
2. `keras.Input` fixes the input shape so `summary()` can show output shapes immediately.
3. Integer labels pair with `sparse_categorical_crossentropy`; `"accuracy"` is resolved to `SparseCategoricalAccuracy`.
4. `validation_split=0.1` holds out the last 10 percent of the training arrays. `ModelCheckpoint` writes a full `.keras` model whenever validation accuracy improves.
5. `predict` returns class probabilities (softmax output); `argmax` gives the class.
6. The `.keras` file contains architecture, weights, and optimizer state.

Switch backends by running `KERAS_BACKEND=jax python mnist.py` or `KERAS_BACKEND=torch python mnist.py`.

### Tutorial 2: Tabular data with the Functional API and preprocessing layers

Goal: predict a binary target from mixed numeric and categorical columns, with all preprocessing inside the model so it can be served on raw values.

```python
import numpy as np
import pandas as pd
import keras
from keras import layers

# 1. Synthetic dataset (replace with your own DataFrame)
rng = np.random.default_rng(0)
n = 5000
df = pd.DataFrame({
    "age": rng.integers(18, 80, n).astype("float32"),
    "income": rng.lognormal(10, 0.5, n).astype("float32"),
    "city": rng.choice(["paris", "berlin", "rome", "madrid"], n),
    "plan": rng.choice([1, 2, 3], n).astype("int64"),
})
logit = 0.03 * (df["age"] - 45) + 0.5 * (df["city"] == "berlin") - 0.4 * (df["plan"] == 1)
df["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("float32")

train = df.sample(frac=0.8, random_state=0)
valid = df.drop(train.index)

def to_inputs(frame):
    return {
        "age": frame["age"].to_numpy()[:, None],
        "income": np.log(frame["income"].to_numpy())[:, None],
        "city": frame["city"].to_numpy()[:, None],
        "plan": frame["plan"].to_numpy()[:, None],
    }

x_train, y_train = to_inputs(train), train["target"].to_numpy()
x_valid, y_valid = to_inputs(valid), valid["target"].to_numpy()

# 2. Preprocessing layers fitted on training data only
age_norm = layers.Normalization()
age_norm.adapt(x_train["age"])
income_norm = layers.Normalization()
income_norm.adapt(x_train["income"])
city_lookup = layers.StringLookup(output_mode="one_hot")
city_lookup.adapt(x_train["city"])
plan_lookup = layers.IntegerLookup(output_mode="one_hot")
plan_lookup.adapt(x_train["plan"])

# 3. Functional model with named inputs
inputs = {
    "age": keras.Input(shape=(1,), name="age", dtype="float32"),
    "income": keras.Input(shape=(1,), name="income", dtype="float32"),
    "city": keras.Input(shape=(1,), name="city", dtype="string"),
    "plan": keras.Input(shape=(1,), name="plan", dtype="int64"),
}
features = layers.concatenate([
    age_norm(inputs["age"]),
    income_norm(inputs["income"]),
    city_lookup(inputs["city"]),
    plan_lookup(inputs["plan"]),
])
x = layers.Dense(64, activation="relu")(features)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid", name="target")(x)
model = keras.Model(inputs, output)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc"), "accuracy"])

model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
          epochs=20, batch_size=64,
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                   patience=3, restore_best_weights=True)])

# 4. Predict on raw values
sample = {"age": np.array([[35.0]]), "income": np.log(np.array([[30000.0]])),
          "city": np.array([["berlin"]]), "plan": np.array([[2]])}
print(model.predict(sample, verbose=0))
model.save("tabular.keras")
```

Notes:

- Dict inputs are matched to `keras.Input(name=...)`.
- `adapt` is called on training data only, avoiding leakage from validation data.
- String inputs and `StringLookup` inside the model require the TensorFlow backend. With JAX or PyTorch, apply the lookups in a `tf.data` pipeline (or encode categories as integers beforehand and use `IntegerLookup` / `Embedding`).
- `mode="max"` is set explicitly for AUC, although `"auto"` usually infers it.

### Tutorial 3: Transfer learning for image classification

Goal: classify flower photos (5 classes) using a pretrained EfficientNet, first training a new head, then fine-tuning.

```python
import pathlib
import keras
from keras import layers
import tensorflow as tf   # used for tf.data pipeline helpers

# 1. Download and locate the dataset
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = keras.utils.get_file("flower_photos.tgz", origin=url, extract=True)
root = pathlib.Path(archive).parent
data_dir = next(p for p in root.rglob("flower_photos") if p.is_dir() and (p / "roses").exists())

IMG = (224, 224)
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="both", seed=123,
    image_size=IMG, batch_size=32)
class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# 2. Augmentation as layers (active only during training)
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="augment")

# 3. Pretrained base, frozen
base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                         input_shape=IMG + (3,))
base.trainable = False

inputs = keras.Input(shape=IMG + (3,))
x = augment(inputs)
x = base(x, training=False)          # keep BatchNorm in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 4. Fine-tune the top of the base with a low learning rate
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),    # recompile after changing trainable
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5,
          callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
model.save("flowers_effnet.keras")
```

Explanation:

1. `get_file` caches downloads in `~/.keras/datasets`; searching for the folder keeps the code working across Keras versions, which differ in where `extract=True` puts files.
2. `subset="both"` returns train and validation datasets with a consistent split.
3. Keras EfficientNet models include input rescaling, so feed raw `0..255` pixels. Other applications (e.g. ResNet50) need their module's `preprocess_input`.
4. Calling the base with `training=False` keeps BatchNorm statistics frozen even after unfreezing, which prevents fine-tuning from destroying pretrained features. Always recompile after changing `trainable`.

### Tutorial 4: Custom layer, custom model with train_step, saving and reloading

Goal: build a small Transformer encoder text classifier on IMDB (integer-encoded), with a custom serializable layer and a customized training step (TensorFlow backend shown, with notes for others).

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops

vocab_size, maxlen = 20_000, 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.utils.pad_sequences(x_test, maxlen=maxlen)


@keras.saving.register_keras_serializable(package="tutorial")
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        positions = ops.arange(0, ops.shape(x)[-1], 1)
        return self.token_emb(x) + self.pos_emb(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"maxlen": self.maxlen, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return cfg


@keras.saving.register_keras_serializable(package="tutorial")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = embed_dim, num_heads, ff_dim, rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn = self.drop1(self.att(inputs, inputs), training=training)
        x = self.norm1(inputs + attn)
        ffn = self.drop2(self.ffn(x), training=training)
        return self.norm2(x + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim, "rate": self.rate})
        return cfg


@keras.saving.register_keras_serializable(package="tutorial")
class GradNormModel(keras.Model):
    """Functional model whose train_step also reports the global gradient norm."""

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(grads, self.trainable_variables)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["grad_norm"] = tf.linalg.global_norm(grads)
        return logs


embed_dim, num_heads, ff_dim = 32, 2, 32
inputs = keras.Input(shape=(maxlen,), dtype="int32")
x = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)
x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = GradNormModel(inputs, outputs)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_test, y_test))

model.save("imdb_transformer.keras")
reloaded = keras.models.load_model("imdb_transformer.keras")
np.testing.assert_allclose(model.predict(x_test[:8], verbose=0),
                           reloaded.predict(x_test[:8], verbose=0), rtol=1e-5, atol=1e-6)
print("reload OK")
```

Explanation:

- Custom layers create sublayers in `__init__` and use only `keras.ops`, so they are backend-agnostic.
- `get_config` returns constructor arguments; with `register_keras_serializable`, `load_model` finds the classes without `custom_objects`.
- Subclassing `keras.Model` and calling it with `(inputs, outputs)` gives a Functional model with a custom `train_step`.
- `self.compute_loss` applies the compiled loss (plus any `add_loss` terms); `self.optimizer.apply` updates weights.
- The `train_step` above uses `tf.GradientTape` and works only with the TensorFlow backend. With JAX you implement a stateless `train_step(self, state, data)` using `self.stateless_call` and `jax.value_and_grad`; with PyTorch you call `self.zero_grad()`, `loss.backward()`, and then `self.optimizer.apply(...)` under `torch.no_grad()`. See the "Customizing what happens in fit()" guides for each backend.

## Performance & Best Practices

### Training speed

- Use `tf.data` (or a PyTorch `DataLoader`) with prefetching for anything larger than memory; for NumPy arrays, Keras batches for you.
- Leave `jit_compile="auto"` (XLA on TF/JAX). If a custom layer fails under XLA, set `jit_compile=False` for that model.
- Increase `steps_per_execution` (for example 32) for small models where Python overhead dominates, especially on TPU.
- Use mixed precision: `keras.config.set_dtype_policy("mixed_float16")` on NVIDIA GPUs (compute capability 7.0+) or `"mixed_bfloat16"` on TPU / Ampere+ GPUs.
- Choose the backend that is fastest for your model; the Keras team's benchmarks show it varies by model (JAX is often fastest on TPU and for large models).
- Use `model(x, training=False)` rather than `model.predict(x)` for tiny batches in a loop.

### Model quality

- Start from a pretrained model (keras.applications, KerasHub) whenever possible.
- Normalize inputs (`Normalization`, `Rescaling`); unnormalized inputs are the most common cause of slow or failed training.
- Use `EarlyStopping(restore_best_weights=True)` plus `ReduceLROnPlateau`, or a cosine schedule with warmup.
- Prefer `from_logits=True` losses with a linear final layer for numerical stability.
- Add regularization progressively: dropout, weight decay (`AdamW`), data augmentation layers, label smoothing.

### Reproducibility and robustness

- `keras.utils.set_random_seed(seed)` at program start; on TF add `tf.config.experimental.enable_op_determinism()` for exact GPU repeatability.
- Put preprocessing inside the model (or in a saved pipeline) so serving uses the same transforms.
- Save with `.keras`, register custom objects, and implement `get_config` for every custom layer.
- Use `BackupAndRestore` on preemptible machines.
- Write custom layers with `keras.ops` to keep them backend-portable.

### Memory

- Reduce batch size or image resolution first; then try mixed precision.
- On TensorFlow, enable GPU memory growth (`tf.config.experimental.set_memory_growth`). On JAX, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to stop it reserving most of the GPU memory upfront.
- Call `keras.backend.clear_session()` between models in notebooks.

## Common Errors & Troubleshooting

### Incompatible input shape

```text
ValueError: Input 0 of layer "dense" is incompatible with the layer: expected axis -1 of input shape to have value 784, but received input with shape (None, 28, 28)
```

Cause: data shape does not match `keras.Input(shape=...)`.

Fix: reshape data (`x.reshape(-1, 784)`) or add `layers.Flatten()`; check `model.input_shape` against `x.shape[1:]`.

### Target and output shapes differ

```text
ValueError: Arguments `target` and `output` must have the same shape. Received: target.shape=(None, 1), output.shape=(None, 10)
```

Cause: integer labels used with `categorical_crossentropy` (which expects one-hot), or wrong number of output units.

Fix: use `sparse_categorical_crossentropy` for integer labels, or one-hot encode with `keras.utils.to_categorical(y, num_classes)`.

### Label out of range

```text
Received a label value of 10 which is outside the valid range of [0, 10).
```

Cause: labels are 1-based or the output layer has too few units.

Fix: make labels `0 .. num_classes - 1` and set `Dense(num_classes)`.

### Wrong file extension when saving

```text
ValueError: Invalid filepath extension for saving. Please add either a `.keras` extension for the native Keras format (recommended) or a `.h5` extension. Use `model.export(filepath)` if you want to export a SavedModel for use with TFLite/TFServing/etc.
```

Fix: `model.save("model.keras")`; for a SavedModel use `model.export("dir")`.

### Weights filename

```text
ValueError: The filename must end in `.weights.h5`. Received: filepath=model.h5
```

Fix: `model.save_weights("model.weights.h5")`. For `ModelCheckpoint(save_weights_only=True)` the path must also end in `.weights.h5`; otherwise it must end in `.keras`.

### Unknown custom object when loading

```text
TypeError: Could not locate class 'TransformerBlock'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`.
```

Fix: decorate the class with `@keras.saving.register_keras_serializable()` and import the module defining it before loading, or pass `custom_objects={"TransformerBlock": TransformerBlock}` to `load_model`.

### Lambda layer refused on load

```text
ValueError: Requested the deserialization of a `Lambda` layer with a Python `lambda` inside it. This carries a potential risk of arbitrary code execution and thus it is disallowed by default. If you trust the source of the saved model, you can pass `safe_mode=False` to the loading function in order to allow `Lambda` layer loading.
```

Fix: for trusted files, `keras.models.load_model(path, safe_mode=False)`; better, replace `Lambda` with a small registered custom layer.

### Variables missing on load

```text
ValueError: Layer 'my_layer' expected 2 variables, but received 0 variables during loading. Expected: ['kernel', 'bias']
```

Cause: a custom layer creates weights lazily in a way Keras cannot rebuild from the config, or the architecture changed between save and load.

Fix: create weights in `build(input_shape)`, ensure `get_config` captures all constructor arguments, and, if needed, implement `build_from_config` / `get_build_config`.

### Keras 2 arguments removed

```text
TypeError: Model.fit() got an unexpected keyword argument 'workers'
```

Cause: `workers`, `use_multiprocessing`, and `max_queue_size` were removed from `fit` in Keras 3.

Fix: pass them to your `keras.utils.PyDataset` constructor, or use `tf.data`.

Other Keras 2 migration issues:

| Keras 2 code | Keras 3 replacement |
|---|---|
| `from keras.layers.experimental.preprocessing import Normalization` | `from keras.layers import Normalization` |
| `keras.utils.Sequence` | `keras.utils.PyDataset` |
| `ImageDataGenerator(...).flow_from_directory(...)` | `keras.utils.image_dataset_from_directory` + augmentation layers |
| `model.save("dir")` (SavedModel) | `model.save("m.keras")` or `model.export("dir")` |
| `tf.*` ops inside custom layers | `keras.ops.*` (required for JAX/PyTorch backends) |
| `optimizer.lr` | `optimizer.learning_rate` |
| `keras.backend.*` math functions | `keras.ops.*` |

If you cannot migrate yet, install `tf_keras` and set `TF_USE_LEGACY_KERAS=1`.

### Loss becomes NaN

Causes: learning rate too high, unnormalized inputs, `log(0)` in a custom loss, float16 overflow.

Fix: lower the learning rate, normalize inputs, add `global_clipnorm=1.0` to the optimizer, use built-in losses with `from_logits=True`, add `keras.callbacks.TerminateOnNaN()` to stop early, and check that mixed precision keeps the last layer in float32.

## Interoperability

### TensorFlow

- `tf.data.Dataset` objects can be passed straight to `fit`/`evaluate`/`predict` on every backend.
- On the TF backend, Keras tensors are `tf.Tensor`s, so `tf.*` ops can be used inside layers (at the cost of portability).
- `model.export("dir")` produces a SavedModel for TF Serving and LiteRT/TFLite conversion.
- `tf.distribute` strategies work with the TF backend: build and compile inside `strategy.scope()`.

### JAX

- With `KERAS_BACKEND=jax`, Keras compiles training steps with `jax.jit`.
- Stateless APIs (`model.stateless_call`, `optimizer.stateless_apply`, metric `stateless_update_state`) let you write pure-JAX training loops.
- `keras.distribution` provides data and model parallelism on JAX device meshes.

### PyTorch

- With `KERAS_BACKEND=torch`, every Keras layer and model is also a `torch.nn.Module`, so it can be used inside PyTorch code and trained with a PyTorch optimizer loop.
- `fit` accepts `torch.utils.data.DataLoader` objects.
- `keras.layers.TorchModuleWrapper(module)` wraps an existing `torch.nn.Module` as a Keras layer.

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras

model = keras.Sequential([keras.Input(shape=(8,)), keras.layers.Dense(4)])
print(isinstance(model, torch.nn.Module))   # True
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x, y = torch.randn(16, 8), torch.randn(16, 4)   # on a GPU machine, move to "cuda"
loss = torch.nn.functional.mse_loss(model(x), y)
loss.backward()
opt.step()
```

### scikit-learn

Recent Keras 3 releases provide `keras.wrappers.SKLearnClassifier`, `SKLearnRegressor`, and `SKLearnTransformer` for use in scikit-learn pipelines and grid searches. The third-party SciKeras package is an alternative. Typical usage is to pass a function that builds and compiles the model.

### Hugging Face and KerasHub

KerasHub (`pip install keras-hub`) provides pretrained models (Gemma, Llama, Mistral, BERT, ViT, Stable Diffusion 3, and more) as Keras layers and task models, loadable from Kaggle Models and the Hugging Face Hub with `from_preset(...)`. Keras models can also be pushed to and loaded from the Hugging Face Hub as `.keras` files.

### ONNX and TFLite

- TFLite / LiteRT: export a SavedModel with `model.export(...)`, then convert with `tf.lite.TFLiteConverter.from_saved_model`.
- ONNX: recent Keras 3 releases support `model.export("model.onnx", format="onnx")`; otherwise export a SavedModel and use `tf2onnx`, or export via the PyTorch backend with `torch.onnx`.

## Cheat Sheet

### Setup

| Task | Code |
|---|---|
| Install | `pip install keras tensorflow` (or `jax`, `torch`) |
| Choose backend | `export KERAS_BACKEND=jax` (before import) |
| Check backend | `keras.config.backend()` |
| Version | `keras.__version__` |
| Seed | `keras.utils.set_random_seed(42)` |
| Mixed precision | `keras.config.set_dtype_policy("mixed_float16")` |
| Legacy Keras 2 | `pip install tf_keras` + `TF_USE_LEGACY_KERAS=1` |

### Building models

| Task | Code |
|---|---|
| Input | `keras.Input(shape=(28, 28, 1))` |
| Sequential | `keras.Sequential([keras.Input((10,)), layers.Dense(1)])` |
| Functional | `keras.Model(inputs, outputs)` |
| Multi-input | `keras.Model({"a": in_a, "b": in_b}, out)` |
| Residual | `layers.Add()([x, skip])` |
| Concatenate | `layers.concatenate([a, b])` |
| Freeze | `layer.trainable = False` then recompile |
| Summary | `model.summary()` |
| Plot | `keras.utils.plot_model(model, show_shapes=True)` |
| Get layer | `model.get_layer("name")` |
| Feature extractor | `keras.Model(model.input, model.get_layer("x").output)` |

### Training

| Task | Code |
|---|---|
| Compile | `model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])` |
| Logits loss | `keras.losses.SparseCategoricalCrossentropy(from_logits=True)` |
| Fit arrays | `model.fit(x, y, batch_size=64, epochs=10, validation_split=0.1)` |
| Fit dataset | `model.fit(train_ds, validation_data=val_ds, epochs=10)` |
| Class weights | `model.fit(..., class_weight={0: 1.0, 1: 5.0})` |
| Early stop | `keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)` |
| Checkpoint | `keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True)` |
| LR on plateau | `keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)` |
| Cosine schedule | `keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=10_000)` |
| Clip gradients | `keras.optimizers.Adam(global_clipnorm=1.0)` |
| TensorBoard | `keras.callbacks.TensorBoard("logs")` |
| Debug eagerly | `model.compile(..., run_eagerly=True)` |

### Inference and saving

| Task | Code |
|---|---|
| Evaluate | `model.evaluate(x, y, return_dict=True)` |
| Predict | `model.predict(x, batch_size=256)` |
| Single fast call | `model(x, training=False)` |
| Save model | `model.save("m.keras")` |
| Load model | `keras.models.load_model("m.keras")` |
| Save weights | `model.save_weights("m.weights.h5")` |
| Load weights | `model.load_weights("m.weights.h5")` |
| Export SavedModel | `model.export("export_dir")` |
| Register custom | `@keras.saving.register_keras_serializable()` |

### Data

| Task | Code |
|---|---|
| Images from folders | `keras.utils.image_dataset_from_directory(d, image_size=(224, 224), validation_split=0.2, subset="both", seed=1)` |
| Text from folders | `keras.utils.text_dataset_from_directory(d)` |
| Time series windows | `keras.utils.timeseries_dataset_from_array(x, y, sequence_length=60)` |
| One-hot | `keras.utils.to_categorical(y, 10)` |
| Pad sequences | `keras.utils.pad_sequences(seqs, maxlen=200)` |
| Normalize layer | `norm = layers.Normalization(); norm.adapt(x)` |
| Text to ints | `layers.TextVectorization(max_tokens=20000, output_sequence_length=200)` |
| Augment | `keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])` |

## Further Resources

- Official site, guides, and examples: https://keras.io/
- Keras 3 API reference: https://keras.io/api/
- Getting started with Keras 3: https://keras.io/getting_started/
- Developer guides (Functional API, subclassing, customizing fit, transfer learning): https://keras.io/guides/
- Code examples: https://keras.io/examples/
- Migrating Keras 2 code to multi-backend Keras 3: https://keras.io/guides/migrating_to_keras_3/
- GitHub repository: https://github.com/keras-team/keras
- Release notes: https://github.com/keras-team/keras/releases
- KerasHub: https://keras.io/keras_hub/
- KerasTuner: https://keras.io/keras_tuner/
- Legacy Keras 2 package (tf_keras): https://github.com/keras-team/tf-keras
- Book: Francois Chollet, "Deep Learning with Python" (Manning). https://www.manning.com/books/deep-learning-with-python
- TensorFlow Keras guide: https://www.tensorflow.org/guide/keras
