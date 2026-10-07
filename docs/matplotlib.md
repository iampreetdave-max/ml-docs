# Matplotlib

> The foundational plotting library for Python: publication-quality static, animated, and interactive figures.

Matplotlib is the workhorse visualization library of the scientific Python ecosystem. Almost every other Python plotting tool you meet in machine learning work (seaborn, pandas `.plot()`, scikit-learn's `*Display` classes, statsmodels' diagnostic plots, Yellowbrick, SHAP's classic plots) either renders through Matplotlib or returns Matplotlib objects. Learning its object model once pays off across the whole stack.

This page covers **Matplotlib 3.9 and 3.10** (the current stable series at the time of writing). Where behavior differs from older releases, the change is called out explicitly.

## Overview

### What it is

Matplotlib is a 2D (with basic 3D support) plotting library that produces figures in many output formats (PNG, PDF, SVG, EPS, PGF) and embeds in interactive GUI toolkits (Qt, Tk, GTK, wx, macOS) as well as Jupyter notebooks. It exposes two programming interfaces:

- **The explicit, object-oriented (OO) interface**: you create `Figure` and `Axes` objects and call methods on them (`ax.plot(...)`). This is the recommended style for anything non-trivial.
- **The implicit `pyplot` interface**: a MATLAB-inspired state machine (`plt.plot(...)`) that tracks a "current figure" and "current axes" for you. Convenient for quick interactive exploration.

### History and maintainers

Matplotlib was created by John D. Hunter in 2003 to replicate MATLAB's plotting capabilities for analysing electrocorticography data. After Hunter's death in 2012, development continued under a community of core developers. It is a fiscally sponsored project of **NumFOCUS** and is developed in the open at `github.com/matplotlib/matplotlib`. Releases follow a roughly six-month to yearly cadence for minor versions (3.8 in 2023, 3.9 in 2024, 3.10 in late 2024) with bug-fix releases in between.

### When to use it

- You need **full control** over every element of a figure (ticks, spines, annotations, fonts, layout).
- You are producing **publication or report figures** (PDF/SVG vector output, LaTeX-style math text).
- You want **plots of ML diagnostics**: loss curves, confusion matrices, ROC/PR curves, feature importances, residuals, image grids.
- You are building on top of another library that returns Matplotlib `Axes` (seaborn, pandas, scikit-learn) and need to customize the result.
- You need static images generated **headlessly** on servers or CI (Agg backend).

### When not to use it

- **Highly interactive web dashboards** with hover tooltips, zoom, and linked brushing: Plotly, Bokeh, Altair, or HoloViews are better fits.
- **Quick statistical graphics** of tidy DataFrames: seaborn gives you the same result in one line (and still returns Matplotlib objects).
- **Very large point clouds** (tens of millions of points): Datashader or GPU-backed tools (e.g. fastplotlib, vispy) scale better.
- **Grammar-of-graphics style** declarative specs: plotnine or Altair.

### Where it fits in the ML stack

```text
          Data          ->     Modeling        ->   Evaluation / Reporting
  NumPy, pandas, Polars    scikit-learn, PyTorch,   Matplotlib (+ seaborn,
                           TensorFlow, XGBoost,     pandas.plot, sklearn
                           statsmodels              *Display classes)
```

Matplotlib sits at the end of most pipelines: you use it to explore data during EDA, to monitor training (loss/accuracy curves), and to communicate results (metrics plots, model explanations). It depends on NumPy and accepts NumPy arrays, pandas Series/DataFrames, and anything array-like.

## Installation

### pip

```bash
python -m pip install -U matplotlib
```

Matplotlib 3.9 requires Python 3.9+; Matplotlib 3.10 requires Python 3.10+. Binary wheels are published for Windows, macOS (Intel and Apple Silicon), and Linux (manylinux, musllinux), so no compiler is needed.

### conda

```bash
conda install -c conda-forge matplotlib
```

On conda-forge there is also `matplotlib-base`, which installs Matplotlib without pulling in a GUI toolkit (useful in Docker images and servers):

```bash
conda install -c conda-forge matplotlib-base
```

### Optional dependencies

| Dependency | Purpose |
|---|---|
| `PyQt6` / `PySide6` / `PyQt5` | Qt interactive backend (`QtAgg`) |
| `tkinter` | Tk backend (`TkAgg`); ships with most CPython installers |
| `ipympl` | Interactive widget backend in Jupyter (`%matplotlib widget`) |
| `ffmpeg` (system binary) | Saving animations as MP4 via `FFMpegWriter` |
| `Pillow` | Required dependency; also used for GIF writing (`PillowWriter`) |
| A LaTeX install (`latex`, `dvipng`) | `text.usetex: True` full LaTeX rendering |
| `cairocffi` / `pycairo` | Cairo backends |

Install the Jupyter widget backend:

```bash
python -m pip install ipympl
```

### Verifying the install

```python
import matplotlib
import matplotlib.pyplot as plt

print(matplotlib.__version__)          # e.g. '3.10.0'
print(matplotlib.get_backend())        # e.g. 'agg', 'QtAgg', 'module://matplotlib_inline.backend_inline'
print(matplotlib.get_configdir())      # where matplotlibrc and stylelib live
print(matplotlib.get_cachedir())       # font cache location

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
fig.savefig("check.png")               # works even with no display
```

For the installed version and its dependencies:

```bash
python -m pip show matplotlib
```

## Core Concepts

### The anatomy of a figure

Matplotlib's object hierarchy is the single most important thing to understand:

```text
Figure                      the whole canvas (one window / one image file)
 +-- Axes                   one plotting area with its own coordinate system
 |    +-- XAxis / YAxis     each Axis owns ticks, tick labels, and an axis label
 |    +-- Spines            the lines bounding the data area
 |    +-- Artists           Line2D, PathCollection, Patch, Text, AxesImage, ...
 |    +-- Legend
 +-- SubFigure (optional)   nested figure-like containers
 +-- suptitle, supxlabel, figure-level legend, colorbars
```

- A **Figure** is the top-level container. It knows its size in inches (`figsize`) and resolution (`dpi`).
- An **Axes** (note: plural-looking but singular) is a single plot with data limits, an x-axis, a y-axis, and a title. Most plotting methods live on `Axes`.
- An **Axis** is the number-line object that handles ticks and tick labels.
- Everything visible is an **Artist**. Figures, Axes, lines, text, and patches are all Artists. When a figure is rendered, each Artist draws itself onto the **Canvas** via a **Renderer** supplied by the **backend**.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 200)

fig, ax = plt.subplots(figsize=(6, 4), dpi=100)   # Figure + one Axes
line, = ax.plot(x, np.sin(x), label="sin")        # returns list of Line2D
ax.set_title("A sine wave")
ax.set_xlabel("x (radians)")
ax.set_ylabel("amplitude")
ax.legend()

print(type(fig))     # <class 'matplotlib.figure.Figure'>
print(type(ax))      # <class 'matplotlib.axes._axes.Axes'>
print(type(line))    # <class 'matplotlib.lines.Line2D'>
print(ax.get_children()[:3])
fig.savefig("sine.png", dpi=150, bbox_inches="tight")
```

### Explicit (OO) vs implicit (pyplot) interfaces

Both interfaces call the same underlying code. The difference is who keeps track of "which Axes am I drawing on".

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

# Implicit pyplot style: global state
plt.figure(figsize=(5, 3))
plt.plot(x, np.sqrt(x))
plt.title("pyplot style")
plt.xlabel("x")
plt.savefig("implicit.png")
plt.close()

# Explicit OO style: you hold references
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(x, np.sqrt(x))
ax.set_title("OO style")
ax.set_xlabel("x")
fig.savefig("explicit.png")
plt.close(fig)
```

Rules of thumb:

- Use `pyplot` only to **create** figures (`plt.subplots`, `plt.figure`), to **show** them (`plt.show`), and to **close** them (`plt.close`).
- Use methods on `fig` and `ax` for everything else. This makes functions reusable: a plotting helper should accept an `ax` argument and draw on it.

```python
def plot_history(history, ax=None):
    """Plot training/validation loss on a given Axes (or a new one)."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(history["loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    return ax
```

Note the naming differences between the two styles: `plt.title` vs `ax.set_title`, `plt.xlim` vs `ax.set_xlim`, `plt.xlabel` vs `ax.set_xlabel`, `plt.xticks` vs `ax.set_xticks`.

### Backends

The backend decides where output goes.

- **Non-interactive (file) backends**: `agg` (PNG raster), `pdf`, `svg`, `ps`, `pgf`, `cairo`.
- **Interactive backends**: `QtAgg`, `TkAgg`, `MacOSX`, `GTK4Agg`, `WebAgg`, `nbAgg`, and `ipympl` (`widget`).

Select a backend before any figure is created:

```python
import matplotlib
matplotlib.use("Agg")          # headless; safe on servers and in CI
import matplotlib.pyplot as plt
```

Or with an environment variable:

```bash
MPLBACKEND=Agg python train.py
```

In Jupyter, the default inline backend renders static PNGs after each cell. `%matplotlib widget` (needs `ipympl`) gives pan/zoom inside the notebook.

### Coordinate systems and transforms

Matplotlib has several coordinate systems, and many functions accept a `transform=` argument to choose one:

| Coordinate system | Transform object | (0, 0) is | (1, 1) is |
|---|---|---|---|
| Data | `ax.transData` | data origin | depends on data |
| Axes | `ax.transAxes` | lower-left of Axes | upper-right of Axes |
| Figure | `fig.transFigure` | lower-left of Figure | upper-right of Figure |
| Display | `None` / `IdentityTransform()` | lower-left pixel | in pixels |
| Blended | `ax.get_xaxis_transform()` | x in data, y in axes coords | |

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([0, 10], [0, 100])
ax.text(0.02, 0.95, "top-left, axes coords", transform=ax.transAxes, va="top")
ax.text(5, 50, "data coords (5, 50)")
ax.axvspan(2, 4, alpha=0.2)                          # x in data, spans full height
ax.text(3, 0.5, "band", transform=ax.get_xaxis_transform(), ha="center")
fig.text(0.5, 0.01, "figure footer", ha="center")
fig.savefig("transforms.png")
```

### Layout engines

Overlapping labels are the most common cosmetic problem. Use a layout engine:

- `layout="constrained"` (recommended): solves for space for titles, labels, colorbars, and legends. Pass to `plt.subplots`, `plt.figure`, or `plt.subplot_mosaic`.
- `layout="tight"` / `fig.tight_layout()`: older, simpler heuristic.
- `layout="compressed"`: like constrained but removes whitespace between fixed-aspect Axes (images).

```python
fig, axs = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
```

`constrained_layout=True` and `tight_layout=True` keyword arguments still work but the single `layout=` keyword is the modern spelling.

### Styling: rcParams and style sheets

Global defaults live in the dict-like `matplotlib.rcParams`. Change them at runtime, through a `matplotlibrc` file, or with style sheets.

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.figsize"] = (8, 5)
mpl.rcParams["axes.grid"] = True
mpl.rcParams["font.size"] = 11

print(plt.style.available[:8])
plt.style.use("ggplot")

# Temporary changes
with plt.style.context("dark_background"):
    fig, ax = plt.subplots()
    ax.plot([1, 3, 2])
    fig.savefig("dark.png")

with mpl.rc_context({"lines.linewidth": 3, "axes.titlesize": 16}):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    ax.set_title("thick lines")

mpl.rcdefaults()      # reset everything
```

Version note: in Matplotlib 3.6 the bundled seaborn style sheets were renamed from `"seaborn-whitegrid"` and friends to `"seaborn-v0_8-whitegrid"` etc. The old names were removed in 3.8.

### Color, colormaps, and normalization

- Colors can be named (`"tab:blue"`, `"red"`), hex (`"#1f77b4"`), RGB(A) tuples, grayscale strings (`"0.5"`), or cycle references (`"C0"`..`"C9"`).
- Colormaps map scalar values to colors. Access them through the registry `matplotlib.colormaps["viridis"]` (preferred) or `plt.get_cmap("viridis")`.
- A **Norm** maps data values into [0, 1] before the colormap is applied: `Normalize`, `LogNorm`, `TwoSlopeNorm`, `BoundaryNorm`, `CenteredNorm`, `SymLogNorm`, `PowerNorm`.

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

cmap = mpl.colormaps["coolwarm"]
z = np.random.default_rng(0).normal(size=(20, 20))

fig, ax = plt.subplots(layout="constrained")
im = ax.imshow(z, cmap=cmap, norm=mpl.colors.TwoSlopeNorm(vcenter=0))
fig.colorbar(im, ax=ax, label="z-score")
fig.savefig("norm.png")
```

Deprecation note: `matplotlib.cm.get_cmap` was deprecated in 3.7 and removed in 3.9. Use `matplotlib.colormaps[name]` or `plt.get_cmap(name)`. Likewise `cm.register_cmap` was replaced by `matplotlib.colormaps.register(cmap)`.

### The property cycle

Successive calls to `ax.plot` use the next color from `axes.prop_cycle` (default: the ten `tab10` colors, `C0`..`C9`).

```python
from cycler import cycler
import matplotlib.pyplot as plt

plt.rcParams["axes.prop_cycle"] = cycler(color=["k", "tab:red", "tab:blue"]) + \
                                  cycler(linestyle=["-", "--", ":"])
```

### Rendering lifecycle

Nothing is drawn until the figure is rendered: by `fig.savefig`, `plt.show`, `fig.canvas.draw()`, or the notebook's display hook. Before that, you are only building a tree of Artists. That is why you can call `ax.set_xlim` after `ax.plot`, and why measuring text extents requires a draw (`fig.canvas.draw()` then `artist.get_window_extent()`).

## API Reference

All examples assume:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```

### Figure creation and management (pyplot)

#### plt.subplots

```python
plt.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True,
             width_ratios=None, height_ratios=None, subplot_kw=None,
             gridspec_kw=None, **fig_kw)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nrows`, `ncols` | int | 1 | Grid shape |
| `sharex`, `sharey` | bool or `"row"`, `"col"`, `"all"`, `"none"` | False | Share axis limits and ticks |
| `squeeze` | bool | True | If True, return a single Axes for 1x1 and a 1D array for single row/column |
| `width_ratios` | list of float | None | Relative column widths |
| `height_ratios` | list of float | None | Relative row heights |
| `subplot_kw` | dict | None | Passed to each `add_subplot` (e.g. `{"projection": "polar"}`) |
| `gridspec_kw` | dict | None | Passed to `GridSpec` (e.g. `hspace`, `wspace`) |
| `**fig_kw` | | | Passed to `plt.figure`: `figsize`, `dpi`, `layout`, ... |

Returns `(Figure, Axes or ndarray of Axes)`.

```python
fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True, layout="constrained")
for i, ax in enumerate(axs.flat):
    ax.plot(np.random.rand(10))
    ax.set_title(f"panel {i}")
```

#### plt.figure

```python
plt.figure(num=None, figsize=None, dpi=None, *, facecolor=None, edgecolor=None,
           frameon=True, clear=False, layout=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num` | int or str | None | Figure identifier; reuses an existing figure with that number/label |
| `figsize` | (float, float) | `rcParams["figure.figsize"]` = (6.4, 4.8) | Width, height in inches |
| `dpi` | float | `rcParams["figure.dpi"]` = 100 | Dots per inch |
| `facecolor` | color | white | Background color |
| `clear` | bool | False | Clear an existing figure with the same `num` |
| `layout` | `"constrained"`, `"compressed"`, `"tight"`, `"none"`, or LayoutEngine | None | Layout engine |

Returns a `Figure`.

```python
fig = plt.figure(figsize=(8, 4), layout="constrained")
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection="polar")
```

#### plt.subplot_mosaic / Figure.subplot_mosaic

```python
plt.subplot_mosaic(mosaic, *, sharex=False, sharey=False, width_ratios=None,
                   height_ratios=None, empty_sentinel=".", subplot_kw=None,
                   per_subplot_kw=None, gridspec_kw=None, **fig_kw)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mosaic` | list of lists or str | required | Layout spec; repeated labels span cells |
| `empty_sentinel` | str | `"."` | Label meaning "leave this cell empty" |
| `per_subplot_kw` | dict | None | Per-label kwargs, e.g. `{"C": {"projection": "polar"}}` |

Returns `(Figure, dict[label, Axes])`.

```python
fig, axd = plt.subplot_mosaic(
    """
    AAB
    CCB
    """,
    figsize=(9, 5), layout="constrained",
)
axd["A"].plot(np.random.randn(50).cumsum())
axd["B"].hist(np.random.randn(500), orientation="horizontal")
axd["C"].scatter(*np.random.rand(2, 40))
```

#### plt.show, plt.close, plt.gcf, plt.gca, plt.clf, plt.cla

```python
plt.show(*, block=None)
plt.close(fig=None)        # fig: None (current), Figure, int, str, or "all"
plt.gcf()                  # get current Figure
plt.gca()                  # get current Axes
plt.clf()                  # clear current figure
plt.cla()                  # clear current axes
plt.ion(); plt.ioff()      # interactive mode on/off
plt.pause(interval)        # draw and run GUI event loop for interval seconds
```

| Function | Purpose |
|---|---|
| `plt.show()` | Display all open figures; blocks in scripts until windows close |
| `plt.close("all")` | Free memory for all figures (important in loops) |
| `plt.gcf()` / `plt.gca()` | Bridge from pyplot state to OO objects |
| `plt.pause(0.01)` | Update a live plot during training |

```python
for epoch in range(3):
    fig, ax = plt.subplots()
    ax.plot(np.random.rand(10))
    fig.savefig(f"epoch_{epoch}.png")
    plt.close(fig)       # prevents "More than 20 figures have been opened" warning
```

#### Figure.savefig

```python
Figure.savefig(fname, *, transparent=None, dpi="figure", format=None,
               metadata=None, bbox_inches=None, pad_inches=0.1,
               facecolor="auto", edgecolor="auto", backend=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fname` | str, Path, or file-like | required | Output path; format inferred from extension |
| `dpi` | float or `"figure"` | `rcParams["savefig.dpi"]` (`"figure"`) | Resolution for raster output |
| `format` | str | None | Force format (`"png"`, `"pdf"`, `"svg"`, ...) |
| `bbox_inches` | `"tight"` or Bbox | None | `"tight"` crops to content |
| `pad_inches` | float | 0.1 | Padding when `bbox_inches="tight"` |
| `transparent` | bool | False | Transparent background |
| `metadata` | dict | None | e.g. `{"Author": "me"}` |

Returns None.

```python
import io
fig, ax = plt.subplots()
ax.plot([1, 2, 3])
fig.savefig("plot.pdf")                         # vector
fig.savefig("plot.png", dpi=300, bbox_inches="tight", transparent=True)
buf = io.BytesIO()
fig.savefig(buf, format="png")                  # in-memory, e.g. for a web API
png_bytes = buf.getvalue()
```

#### Figure-level methods

| Method | Description |
|---|---|
| `fig.add_subplot(nrows, ncols, index, **kw)` | Add one Axes to a grid |
| `fig.add_axes([left, bottom, width, height])` | Add Axes at an arbitrary figure-fraction rectangle |
| `fig.add_gridspec(nrows, ncols, **kw)` | Create a `GridSpec` for complex layouts |
| `fig.subfigures(nrows, ncols, **kw)` | Nested `SubFigure` objects, each with their own suptitle/colorbar |
| `fig.suptitle(t, **kw)` | Figure title |
| `fig.supxlabel(t)` / `fig.supylabel(t)` | Shared axis labels |
| `fig.colorbar(mappable, ax=None, cax=None, **kw)` | Add colorbar |
| `fig.legend(handles=None, labels=None, loc=..., **kw)` | Figure-level legend |
| `fig.set_size_inches(w, h)` | Resize |
| `fig.get_axes()` / `fig.axes` | List of Axes |
| `fig.align_labels()` | Align x/y labels across subplots |
| `fig.text(x, y, s)` | Text in figure coordinates |

```python
fig = plt.figure(figsize=(10, 4), layout="constrained")
left, right = fig.subfigures(1, 2, width_ratios=[2, 1])
axl = left.subplots(1, 2)
axr = right.subplots()
left.suptitle("Left subfigure")
right.suptitle("Right subfigure")

gs = plt.figure(layout="constrained").add_gridspec(3, 3)
fig2 = gs.figure
ax_main = fig2.add_subplot(gs[1:, :2])
ax_top = fig2.add_subplot(gs[0, :2], sharex=ax_main)
ax_side = fig2.add_subplot(gs[1:, 2], sharey=ax_main)
```

### Basic plotting methods (Axes)

#### Axes.plot

```python
Axes.plot(*args, scalex=True, scaley=True, data=None, **kwargs)
# call forms: plot(y), plot(x, y), plot(x, y, fmt), plot(x1, y1, fmt1, x2, y2, fmt2)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x`, `y` | array-like | | Data; `x` defaults to `range(len(y))` |
| `fmt` | str | None | Shorthand `"[marker][line][color]"`, e.g. `"o--r"` |
| `data` | indexable (dict, DataFrame) | None | Lets you pass column names as `x`/`y` |
| `color` / `c` | color | next in cycle | Line color |
| `linestyle` / `ls` | str | `"-"` | `"-"`, `"--"`, `"-."`, `":"`, `"None"` |
| `linewidth` / `lw` | float | 1.5 | Line width in points |
| `marker` | str | None | `"o"`, `"s"`, `"^"`, `"x"`, `"."`, ... |
| `markersize` / `ms` | float | 6 | Marker size in points |
| `alpha` | float | None | Opacity 0..1 |
| `label` | str | None | Legend entry |
| `zorder` | float | 2 | Draw order |
| `drawstyle` | str | `"default"` | `"steps-pre"`, `"steps-mid"`, `"steps-post"` |

Returns `list[Line2D]`.

```python
import pandas as pd
df = pd.DataFrame({"epoch": range(20), "loss": np.exp(-np.arange(20) / 5)})
fig, ax = plt.subplots()
ax.plot("epoch", "loss", "o-", data=df, label="train loss", ms=4)
ax.plot(df["epoch"], df["loss"] * 1.1, ls="--", color="tab:orange", label="val loss")
ax.legend()
```

#### Axes.scatter

```python
Axes.scatter(x, y, s=None, c=None, *, marker=None, cmap=None, norm=None,
             vmin=None, vmax=None, alpha=None, linewidths=None,
             edgecolors=None, plotnonfinite=False, data=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x`, `y` | array-like | required | Positions |
| `s` | float or array | `rcParams["lines.markersize"] ** 2` (36) | Marker area in points squared |
| `c` | color, array of colors, or array of floats | None | Floats are colormapped |
| `marker` | str | `"o"` | Marker style |
| `cmap` | str or Colormap | `"viridis"` | Used when `c` is numeric |
| `norm` | Normalize | None | Scaling of `c` |
| `vmin`, `vmax` | float | None | Color limits (ignored if `norm` given) |
| `alpha` | float | None | Opacity |
| `edgecolors` | color or `"face"` | `"face"` | Marker edge color |

Returns a `PathCollection` (a "mappable" you can pass to `colorbar`).

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(300, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
fig, ax = plt.subplots(layout="constrained")
sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=20, edgecolors="k", linewidths=0.3)
ax.legend(*sc.legend_elements(), title="class")
```

`PathCollection.legend_elements()` builds legend handles for the unique color (or size, with `prop="sizes"`) values.

#### Axes.bar and Axes.barh

```python
Axes.bar(x, height, width=0.8, bottom=None, *, align="center", data=None, **kwargs)
Axes.barh(y, width, height=0.8, left=None, *, align="center", data=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | array-like of float or str | required | Bar positions or categorical labels |
| `height` | float or array | required | Bar heights |
| `width` | float or array | 0.8 | Bar widths |
| `bottom` | float or array | 0 | Baseline (for stacking) |
| `align` | `"center"` or `"edge"` | `"center"` | Alignment relative to `x` |
| `color`, `edgecolor` | color | | Fill and edge colors |
| `yerr`, `xerr` | float or array | None | Error bars |
| `capsize` | float | 0 | Error bar cap length |
| `tick_label` | str or list | None | Tick labels |
| `label` | str | None | Legend label |

Returns a `BarContainer`.

```python
models = ["LogReg", "RF", "XGB", "MLP"]
acc = [0.81, 0.86, 0.89, 0.87]
std = [0.02, 0.015, 0.01, 0.02]
fig, ax = plt.subplots()
bars = ax.bar(models, acc, yerr=std, capsize=4, color="tab:blue")
ax.bar_label(bars, fmt="%.2f", padding=3)
ax.set_ylim(0.7, 0.95)
ax.set_ylabel("accuracy")
```

#### Axes.bar_label

```python
Axes.bar_label(container, labels=None, *, fmt="%g", label_type="edge",
               padding=0, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `container` | BarContainer | required | Return value of `bar`/`barh` |
| `labels` | list of str | None | Custom labels; default formats the values |
| `fmt` | str or callable | `"%g"` | `%`-format string, `{}`-format string, or function |
| `label_type` | `"edge"` or `"center"` | `"edge"` | Position |
| `padding` | float | 0 | Distance from bar end in points |

Returns a list of `Annotation`.

#### Axes.hist

```python
Axes.hist(x, bins=None, range=None, density=False, weights=None,
          cumulative=False, bottom=None, histtype="bar", align="mid",
          orientation="vertical", rwidth=None, log=False, color=None,
          label=None, stacked=False, *, data=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | array or sequence of arrays | required | Data (multiple arrays = multiple datasets) |
| `bins` | int, sequence, or str | `rcParams["hist.bins"]` = 10 | Number of bins, edges, or a strategy like `"auto"`, `"fd"`, `"sturges"` |
| `range` | (float, float) | None | Lower and upper range of bins |
| `density` | bool | False | Normalize so area = 1 |
| `weights` | array | None | Per-sample weights |
| `cumulative` | bool or -1 | False | Cumulative histogram |
| `histtype` | str | `"bar"` | `"bar"`, `"barstacked"`, `"step"`, `"stepfilled"` |
| `orientation` | str | `"vertical"` | or `"horizontal"` |
| `log` | bool | False | Log-scale count axis |
| `stacked` | bool | False | Stack multiple datasets |

Returns `(n, bins, patches)`: counts, bin edges, and the drawn artists.

```python
rng = np.random.default_rng(0)
a, b = rng.normal(0, 1, 1000), rng.normal(1.5, 0.8, 1000)
fig, ax = plt.subplots()
n, edges, _ = ax.hist([a, b], bins=40, histtype="stepfilled", alpha=0.5,
                      density=True, label=["class 0", "class 1"])
ax.legend()
```

#### Axes.hist2d and Axes.hexbin

```python
Axes.hist2d(x, y, bins=10, range=None, density=False, weights=None,
            cmin=None, cmax=None, *, data=None, **kwargs)
Axes.hexbin(x, y, C=None, gridsize=100, bins=None, xscale="linear",
            yscale="linear", extent=None, cmap=None, norm=None, vmin=None,
            vmax=None, alpha=None, linewidths=None, edgecolors="face",
            reduce_C_function=np.mean, mincnt=None, marginals=False, *,
            data=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gridsize` | int or (int, int) | 100 | Number of hexagons in x (hexbin) |
| `bins` | `"log"`, int, or sequence | None | `"log"` uses log10 color scale (hexbin) |
| `C` | array | None | Values to aggregate per hexagon with `reduce_C_function` |
| `mincnt` | int | None | Only show cells with at least this many points |

`hist2d` returns `(h, xedges, yedges, image)`; `hexbin` returns a `PolyCollection`.

```python
rng = np.random.default_rng(1)
x, y = rng.normal(size=(2, 100_000))
fig, ax = plt.subplots(layout="constrained")
hb = ax.hexbin(x, y, gridsize=50, bins="log", cmap="inferno")
fig.colorbar(hb, ax=ax, label="log10(count)")
```

#### Axes.boxplot

```python
Axes.boxplot(x, *, notch=None, sym=None, vert=None, orientation="vertical",
             whis=None, positions=None, widths=None, patch_artist=None,
             bootstrap=None, usermedians=None, conf_intervals=None,
             meanline=None, showmeans=None, showcaps=None, showbox=None,
             showfliers=None, boxprops=None, tick_labels=None, flierprops=None,
             medianprops=None, meanprops=None, capprops=None,
             whiskerprops=None, manage_ticks=True, autorange=False,
             zorder=None, capwidths=None, label=None, data=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | array or sequence of arrays | required | One box per array |
| `notch` | bool | False | Notched box (median CI) |
| `whis` | float or (float, float) | 1.5 | Whisker reach in IQR units, or percentiles |
| `tick_labels` | list of str | None | Box labels (3.9+; replaces `labels`) |
| `orientation` | `"vertical"` or `"horizontal"` | `"vertical"` | (3.10+; replaces `vert`) |
| `patch_artist` | bool | False | Draw boxes as filled patches |
| `showmeans` | bool | False | Mark the mean |
| `showfliers` | bool | True | Draw outliers |

Returns a dict with keys `"boxes"`, `"medians"`, `"whiskers"`, `"caps"`, `"fliers"`, `"means"`.

Version notes: `labels=` was renamed `tick_labels=` in 3.9 (old name deprecated). `vert=` is deprecated in 3.10 in favor of `orientation=`; on 3.9 use `vert=False` for horizontal boxes.

```python
rng = np.random.default_rng(0)
scores = [rng.normal(m, 0.03, 30) for m in (0.80, 0.85, 0.88)]
fig, ax = plt.subplots()
ax.boxplot(scores, tick_labels=["A", "B", "C"], patch_artist=True, showmeans=True)
ax.set_ylabel("CV accuracy")
```

#### Axes.violinplot

```python
Axes.violinplot(dataset, positions=None, vert=None, orientation="vertical",
                widths=0.5, showmeans=False, showextrema=True,
                showmedians=False, quantiles=None, points=100,
                bw_method=None, side="both", *, data=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | array or sequence | required | Data per violin |
| `showmedians` | bool | False | Draw median line |
| `quantiles` | list of lists | None | Quantile lines per violin |
| `bw_method` | str, float, callable | None (`"scott"`) | KDE bandwidth |
| `side` | `"both"`, `"low"`, `"high"` | `"both"` | Half violins (3.9+) |

Returns a dict of artists (`"bodies"`, `"cmeans"`, `"cmedians"`, ...).

```python
fig, ax = plt.subplots()
ax.violinplot([np.random.randn(200), np.random.randn(200) * 2], showmedians=True)
ax.set_xticks([1, 2], labels=["narrow", "wide"])
```

#### Axes.errorbar

```python
Axes.errorbar(x, y, yerr=None, xerr=None, fmt="", ecolor=None,
              elinewidth=None, capsize=None, barsabove=False, lolims=False,
              uplims=False, xlolims=False, xuplims=False, errorevery=1,
              capthick=None, *, data=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `yerr`, `xerr` | float, array (N,), or (2, N) | None | Symmetric or asymmetric errors |
| `fmt` | str | `""` | Format for data points/lines; `"none"` for errors only |
| `capsize` | float | `rcParams["errorbar.capsize"]` = 0 | Cap length |
| `ecolor` | color | None | Error bar color |

Returns an `ErrorbarContainer`.

```python
x = np.arange(1, 6)
y = x ** 1.5
fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=[0.5 * np.ones(5), np.linspace(0.2, 1, 5)], fmt="o-", capsize=3)
```

#### Axes.fill_between

```python
Axes.fill_between(x, y1, y2=0, where=None, interpolate=False, step=None,
                  *, data=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | array | required | x coordinates |
| `y1`, `y2` | array or scalar | `y2=0` | Curves to fill between |
| `where` | bool array | None | Only fill where True |
| `interpolate` | bool | False | Compute exact crossing points when using `where` |
| `alpha`, `color`, `label` | | | Patch styling |

Returns a `FillBetweenPolyCollection` (3.10) / `PolyCollection` (earlier).

```python
x = np.linspace(0, 10, 100)
mean = np.sin(x)
sd = 0.2 + 0.05 * x
fig, ax = plt.subplots()
ax.plot(x, mean, label="prediction")
ax.fill_between(x, mean - 1.96 * sd, mean + 1.96 * sd, alpha=0.3, label="95% interval")
ax.legend()
```

`Axes.fill_betweenx` is the horizontal counterpart.

#### Axes.step and Axes.stairs

```python
Axes.step(x, y, *args, where="pre", data=None, **kwargs)
Axes.stairs(values, edges=None, *, orientation="vertical", baseline=0,
            fill=False, data=None, **kwargs)
```

`stairs` is ideal for pre-binned histograms (`values` has length N, `edges` length N+1).

```python
counts, edges = np.histogram(np.random.randn(1000), bins=30)
fig, ax = plt.subplots()
ax.stairs(counts, edges, fill=True, alpha=0.4)
```

#### Axes.ecdf (3.8+)

```python
Axes.ecdf(x, weights=None, *, complementary=False, orientation="vertical",
          compress=False, data=None, **kwargs)
```

Draws the empirical cumulative distribution function. Returns a `Line2D`.

```python
fig, ax = plt.subplots()
ax.ecdf(np.random.randn(500), label="ECDF")
ax.ecdf(np.random.randn(500), complementary=True, label="1 - ECDF")
ax.legend()
```

#### Axes.pie

```python
Axes.pie(x, explode=None, labels=None, colors=None, autopct=None,
         pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=0,
         radius=1, counterclock=True, wedgeprops=None, textprops=None,
         center=(0, 0), frame=False, rotatelabels=False, *, normalize=True,
         hatch=None, data=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | array | required | Wedge sizes |
| `labels` | list of str | None | Wedge labels |
| `autopct` | str or callable | None | e.g. `"%1.1f%%"` |
| `startangle` | float | 0 | Rotation in degrees |
| `wedgeprops` | dict | None | e.g. `{"width": 0.4}` for a donut |

Returns `(patches, texts)` or `(patches, texts, autotexts)` when `autopct` is set.

```python
fig, ax = plt.subplots()
ax.pie([60, 25, 15], labels=["train", "val", "test"], autopct="%1.0f%%",
       wedgeprops={"width": 0.4}, startangle=90)
```

#### Axes.stackplot

```python
Axes.stackplot(x, *args, labels=(), colors=None, hatch=None,
               baseline="zero", data=None, **kwargs)
```

```python
x = np.arange(10)
fig, ax = plt.subplots()
ax.stackplot(x, np.random.rand(3, 10), labels=["a", "b", "c"])
ax.legend(loc="upper left")
```

### Images, matrices, and fields

#### Axes.imshow

```python
Axes.imshow(X, cmap=None, norm=None, *, aspect=None, interpolation=None,
            alpha=None, vmin=None, vmax=None, colorizer=None, origin=None,
            extent=None, interpolation_stage=None, filternorm=True,
            filterrad=4.0, resample=None, url=None, data=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array (M, N), (M, N, 3), or (M, N, 4) | required | Scalar data (colormapped) or RGB(A) image (floats 0..1 or uint8 0..255) |
| `cmap` | str or Colormap | `"viridis"` | Ignored for RGB data |
| `vmin`, `vmax` | float | data min/max | Color limits |
| `aspect` | `"equal"`, `"auto"`, float | `"equal"` | Pixel aspect ratio |
| `interpolation` | str | `"antialiased"` (3.9: `"auto"` in 3.10) | `"nearest"` for crisp pixels |
| `origin` | `"upper"` or `"lower"` | `"upper"` | Where `[0, 0]` goes |
| `extent` | (left, right, bottom, top) | None | Data coordinates of the image |

Returns an `AxesImage`.

```python
rng = np.random.default_rng(0)
images = rng.random((8, 28, 28))
fig, axs = plt.subplots(2, 4, figsize=(8, 4), layout="constrained")
for ax, img in zip(axs.flat, images):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
```

#### Axes.matshow

```python
Axes.matshow(Z, **kwargs)
```

Like `imshow` but places x tick labels on top and uses integer ticks; convenient for matrices.

```python
corr = np.corrcoef(np.random.randn(5, 100))
fig, ax = plt.subplots()
ms = ax.matshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
fig.colorbar(ms)
```

#### Axes.pcolormesh

```python
Axes.pcolormesh(*args, alpha=None, norm=None, cmap=None, vmin=None,
                vmax=None, colorizer=None, shading=None, antialiased=False,
                data=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X`, `Y` | array | None | Grid coordinates (optional) |
| `C` | 2D array | required | Values |
| `shading` | `"flat"`, `"nearest"`, `"gouraud"`, `"auto"` | `"auto"` (rcParam) | How grid and values align |

Returns a `QuadMesh`. Preferred over `pcolor` (much faster).

```python
x = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, x)
Z = np.exp(-(X ** 2 + Y ** 2))
fig, ax = plt.subplots(layout="constrained")
pm = ax.pcolormesh(X, Y, Z, shading="auto", cmap="magma")
fig.colorbar(pm, ax=ax)
```

#### Axes.contour and Axes.contourf

```python
Axes.contour(*args, data=None, **kwargs)    # contour([X, Y,] Z, [levels], **kwargs)
Axes.contourf(*args, data=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X`, `Y` | 2D arrays | None | Coordinates |
| `Z` | 2D array | required | Heights |
| `levels` | int or array | None (auto) | Number of levels or explicit values |
| `cmap`, `colors` | | | Colormap or fixed colors |
| `alpha` | float | None | Transparency |

Returns a `QuadContourSet`. Use `ax.clabel(cs)` to label lines.

```python
# Decision surface for a classifier
from sklearn.datasets import make_moons
from sklearn.svm import SVC

X, y = make_moons(300, noise=0.25, random_state=0)
clf = SVC(gamma=2).fit(X, y)
xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(layout="constrained")
cf = ax.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.7)
ax.contour(xx, yy, Z, levels=[0], colors="k", linewidths=2)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=15)
fig.colorbar(cf, ax=ax, label="decision function")
```

#### Axes.quiver and Axes.streamplot

```python
Axes.quiver(*args, data=None, **kwargs)          # quiver([X, Y], U, V, [C], **kwargs)
Axes.streamplot(x, y, u, v, density=1, linewidth=None, color=None, cmap=None,
                norm=None, arrowsize=1, arrowstyle="-|>", minlength=0.1,
                transform=None, zorder=None, start_points=None,
                maxlength=4.0, integration_direction="both",
                broken_streamlines=True, *, data=None)
```

Useful for visualizing gradient fields of a loss surface.

```python
w = np.linspace(-2, 2, 20)
W1, W2 = np.meshgrid(w, w)
loss = W1 ** 2 + 3 * W2 ** 2
g1, g2 = 2 * W1, 6 * W2
fig, ax = plt.subplots()
ax.contour(W1, W2, loss, levels=15, cmap="viridis")
ax.quiver(W1, W2, -g1, -g2, alpha=0.6)
ax.set_aspect("equal")
```

### Axes configuration

#### Titles, labels, limits, scales

```python
Axes.set_title(label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs)
Axes.set_xlabel(xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs)
Axes.set_ylabel(ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs)
Axes.set_xlim(left=None, right=None, *, emit=True, auto=False, xmin=None, xmax=None)
Axes.set_ylim(bottom=None, top=None, *, emit=True, auto=False, ymin=None, ymax=None)
Axes.set_xscale(value, **kwargs)     # "linear", "log", "symlog", "logit", "asinh", "function"
Axes.set_yscale(value, **kwargs)
Axes.set(**kwargs)                   # batch setter: ax.set(title=..., xlabel=..., xlim=...)
```

| Method | Key parameters | Description |
|---|---|---|
| `set_title` | `loc` in `"left"`, `"center"`, `"right"`; `pad` (points) | Axes title |
| `set_xlabel`/`set_ylabel` | `labelpad`, `loc` | Axis labels |
| `set_xlim`/`set_ylim` | `left`, `right` / `bottom`, `top` | Data limits; pass reversed values to invert |
| `set_xscale`/`set_yscale` | `"log"`, `base=2`; `"symlog"`, `linthresh=1` | Axis scale |
| `set_aspect` | `"equal"`, `"auto"`, float | Data aspect ratio |
| `margins` | `x`, `y` (fraction) | Padding around autoscaled data |
| `invert_yaxis()` | | Flip axis direction |
| `autoscale(enable=True, axis="both", tight=None)` | | Re-enable autoscaling |

```python
fig, ax = plt.subplots()
ax.plot(np.logspace(0, 4, 50), np.logspace(0, 2, 50))
ax.set(xscale="log", yscale="log", xlabel="n samples", ylabel="time (s)",
       title="Scaling", xlim=(1, 1e4))
ax.set_title("left-aligned subtitle", loc="left", fontsize=9)
```

#### Ticks and tick labels

```python
Axes.set_xticks(ticks, labels=None, *, minor=False, **kwargs)
Axes.set_yticks(ticks, labels=None, *, minor=False, **kwargs)
Axes.set_xticklabels(labels, *, minor=False, fontdict=None, **kwargs)
Axes.tick_params(axis="both", **kwargs)
Axes.xaxis.set_major_locator(locator)
Axes.xaxis.set_major_formatter(formatter)
```

| `tick_params` kwarg | Description |
|---|---|
| `axis` | `"x"`, `"y"`, or `"both"` |
| `which` | `"major"`, `"minor"`, `"both"` |
| `direction` | `"in"`, `"out"`, `"inout"` |
| `length`, `width` | Tick mark size |
| `labelsize` | Font size of tick labels |
| `labelrotation` | Rotation in degrees |
| `bottom`, `top`, `left`, `right` | Show/hide ticks per side |
| `labelbottom`, `labelleft`, ... | Show/hide tick labels per side |

Common locators and formatters from `matplotlib.ticker`: `MaxNLocator`, `MultipleLocator`, `FixedLocator`, `LogLocator`, `AutoMinorLocator`, `NullLocator`; `PercentFormatter`, `StrMethodFormatter`, `FuncFormatter`, `FormatStrFormatter`, `EngFormatter`, `NullFormatter`.

```python
from matplotlib.ticker import PercentFormatter, MaxNLocator, MultipleLocator

fig, ax = plt.subplots()
ax.plot(range(1, 11), np.linspace(0.6, 0.93, 10))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.set_xticks([1, 5, 10], labels=["start", "mid", "end"])
```

A formatter can also be passed as a format string or a function (Matplotlib wraps it automatically):

```python
ax.yaxis.set_major_formatter("{x:.0%}")
ax.xaxis.set_major_formatter(lambda x, pos: f"ep {int(x)}")
```

#### Grid, spines, axis visibility

```python
Axes.grid(visible=None, which="major", axis="both", **kwargs)
Axes.spines["top"].set_visible(False)
Axes.set_axis_off()
Axes.axhline(y=0, xmin=0, xmax=1, **kwargs)
Axes.axvline(x=0, ymin=0, ymax=1, **kwargs)
Axes.axhspan(ymin, ymax, xmin=0, xmax=1, **kwargs)
Axes.axvspan(xmin, xmax, ymin=0, ymax=1, **kwargs)
Axes.axline(xy1, xy2=None, *, slope=None, **kwargs)
Axes.hlines(y, xmin, xmax, colors=None, linestyles="solid", **kwargs)
Axes.vlines(x, ymin, ymax, colors=None, linestyles="solid", **kwargs)
```

```python
fig, ax = plt.subplots()
ax.plot(np.random.randn(100).cumsum())
ax.grid(True, linestyle=":", alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0, color="k", lw=0.8)
ax.axvline(50, color="tab:red", ls="--", label="intervention")
ax.axline((0, 0), slope=0.1, color="gray", lw=0.8)   # infinite line
ax.legend()
```

#### Axes.twinx / twiny and secondary axes

```python
Axes.twinx()      # new Axes sharing x, independent y on the right
Axes.twiny()
Axes.secondary_xaxis(location, *, functions=None, transform=None, **kwargs)
Axes.secondary_yaxis(location, *, functions=None, transform=None, **kwargs)
```

```python
epochs = np.arange(1, 31)
loss = np.exp(-epochs / 8)
lr = 1e-3 * 0.9 ** epochs
fig, ax = plt.subplots(layout="constrained")
ax.plot(epochs, loss, color="tab:blue")
ax.set_ylabel("loss", color="tab:blue")
ax2 = ax.twinx()
ax2.plot(epochs, lr, color="tab:orange")
ax2.set_ylabel("learning rate", color="tab:orange")

# Secondary axis with a conversion
fig, ax = plt.subplots()
ax.plot([0, 100], [0, 1])
sec = ax.secondary_xaxis("top", functions=(lambda c: c * 9 / 5 + 32, lambda f: (f - 32) * 5 / 9))
sec.set_xlabel("Fahrenheit")
```

#### Axes.inset_axes

```python
Axes.inset_axes(bounds, *, transform=None, zorder=5, **kwargs)
Axes.indicate_inset_zoom(inset_ax, **kwargs)
```

```python
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x) * np.exp(-x / 5))
axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
axins.plot(x, np.sin(x) * np.exp(-x / 5))
axins.set_xlim(7, 9); axins.set_ylim(-0.3, 0.3)
ax.indicate_inset_zoom(axins, edgecolor="black")
```

### Legends, text, and annotations

#### Axes.legend

```python
Axes.legend(*args, **kwargs)     # legend(), legend(labels), legend(handles, labels)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loc` | str or (x, y) | `"best"` | `"upper right"`, `"lower left"`, `"center"`, ... |
| `bbox_to_anchor` | (x, y) or (x, y, w, h) | None | Anchor point, e.g. `(1.02, 1)` to place outside |
| `ncols` | int | 1 | Number of columns (`ncol` is the older alias) |
| `title` | str | None | Legend title |
| `frameon` | bool | True | Draw frame |
| `fontsize` | int or str | rcParam | Label size |
| `markerscale` | float | 1.0 | Relative marker size in legend |
| `handles`, `labels` | lists | None | Explicit entries |

Returns a `Legend`.

```python
fig, ax = plt.subplots(layout="constrained")
for k in range(5):
    ax.plot(np.random.randn(20).cumsum(), label=f"seed {k}")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="run", frameon=False)
```

Artists whose label starts with an underscore (e.g. `"_nolegend_"`) are excluded from automatic legends.

#### Axes.text and Axes.annotate

```python
Axes.text(x, y, s, fontdict=None, **kwargs)
Axes.annotate(text, xy, xytext=None, xycoords="data", textcoords=None,
              arrowprops=None, annotation_clip=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `xy` | (float, float) | required | Point being annotated |
| `xytext` | (float, float) | `xy` | Text location |
| `xycoords` / `textcoords` | str | `"data"` | `"data"`, `"axes fraction"`, `"figure fraction"`, `"offset points"`, ... |
| `arrowprops` | dict | None | e.g. `{"arrowstyle": "->"}` |
| `ha`, `va` | str | `"left"`, `"baseline"` | Alignment |
| `fontsize`, `fontweight`, `color` | | | Text styling |
| `bbox` | dict | None | Box around text, e.g. `{"boxstyle": "round", "fc": "w"}` |

`text` returns `Text`; `annotate` returns `Annotation`.

```python
vals = np.array([0.9, 0.7, 0.55, 0.52, 0.51, 0.505])
fig, ax = plt.subplots()
ax.plot(vals, "o-")
best = vals.argmin()
ax.annotate("early stop", xy=(best, vals[best]), xytext=(2, 0.8),
            arrowprops={"arrowstyle": "->", "color": "k"},
            bbox={"boxstyle": "round", "fc": "lightyellow"})
ax.text(0.98, 0.02, "n=6", transform=ax.transAxes, ha="right", fontsize=8)
```

#### Math text

Matplotlib parses a TeX subset inside dollar signs without needing LaTeX installed. Use raw strings:

```python
fig, ax = plt.subplots()
ax.set_title(r"$\hat{y} = \sigma(w^T x + b)$")
ax.set_xlabel(r"$\lambda$ (regularization)")
ax.set_ylabel(r"$\mathcal{L}(\theta)$")
```

### Colorbars and colormaps

#### Figure.colorbar

```python
Figure.colorbar(mappable, cax=None, ax=None, use_gridspec=True, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mappable` | ScalarMappable | required | Return value of `imshow`, `scatter`, `contourf`, `pcolormesh`, ... |
| `ax` | Axes or list of Axes | None | Axes to steal space from |
| `cax` | Axes | None | Explicit Axes to draw the colorbar into |
| `orientation` | str | `"vertical"` | or `"horizontal"` |
| `label` | str | `""` | Colorbar label |
| `shrink` | float | 1.0 | Fraction of original size |
| `extend` | str | `"neither"` | `"min"`, `"max"`, `"both"` triangles |
| `ticks` | list or Locator | None | Tick positions |
| `format` | str or Formatter | None | Tick label format |

Returns a `Colorbar`.

```python
fig, axs = plt.subplots(1, 2, layout="constrained")
for ax in axs:
    im = ax.imshow(np.random.rand(10, 10), vmin=0, vmax=1)
fig.colorbar(im, ax=axs, shrink=0.8, label="probability")   # one shared colorbar
```

#### Colormap registry and Normalize classes

```python
matplotlib.colormaps[name]                                # -> Colormap
matplotlib.colormaps[name].resampled(lutsize)             # discrete version
matplotlib.colors.ListedColormap(colors, name="from_list")
matplotlib.colors.LinearSegmentedColormap.from_list(name, colors, N=256)
matplotlib.colors.Normalize(vmin=None, vmax=None, clip=False)
matplotlib.colors.LogNorm(vmin=None, vmax=None, clip=False)
matplotlib.colors.TwoSlopeNorm(vcenter, vmin=None, vmax=None)
matplotlib.colors.CenteredNorm(vcenter=0, halfrange=None, clip=False)
matplotlib.colors.BoundaryNorm(boundaries, ncolors, clip=False, *, extend="neither")
matplotlib.colors.to_rgba(c, alpha=None)
matplotlib.colors.to_hex(c, keep_alpha=False)
```

Colormap families:

| Kind | Examples | Use for |
|---|---|---|
| Perceptually uniform sequential | `viridis`, `plasma`, `inferno`, `magma`, `cividis` | Magnitudes, densities |
| Sequential | `Blues`, `Greens`, `Greys`, `YlOrRd` | Counts, confusion matrices |
| Diverging | `RdBu`, `coolwarm`, `PiYG`, `seismic` | Signed values, correlations |
| Cyclic | `twilight`, `hsv` | Angles, phases |
| Qualitative | `tab10`, `tab20`, `Set2`, `Paired` | Categorical labels |

Append `_r` to reverse any colormap (`"RdBu_r"`).

```python
import matplotlib as mpl
cmap = mpl.colors.LinearSegmentedColormap.from_list("bw_red", ["white", "darkred"])
discrete = mpl.colormaps["viridis"].resampled(5)
colors = discrete(np.linspace(0, 1, 5))       # RGBA array (5, 4)
print(mpl.colors.to_hex(colors[0]))
```

### 3D plotting (mpl_toolkits.mplot3d)

```python
fig.add_subplot(projection="3d")      # returns Axes3D
Axes3D.plot_surface(X, Y, Z, *, norm=None, vmin=None, vmax=None, lightsource=None, **kwargs)
Axes3D.scatter(xs, ys, zs=0, zdir="z", s=20, c=None, depthshade=True, **kwargs)
Axes3D.plot_wireframe(X, Y, Z, **kwargs)
Axes3D.view_init(elev=None, azim=None, roll=None)
```

```python
x = np.linspace(-2, 2, 60)
X, Y = np.meshgrid(x, x)
Z = X ** 2 - Y ** 2
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
ax.view_init(elev=30, azim=-60)
ax.set(xlabel="w1", ylabel="w2", zlabel="loss")
```

No separate import is required on 3.x: `projection="3d"` registers `mpl_toolkits.mplot3d` automatically.

### Animation

```python
matplotlib.animation.FuncAnimation(fig, func, frames=None, init_func=None,
                                   fargs=None, save_count=None, *,
                                   cache_frame_data=True, interval=200,
                                   repeat_delay=0, repeat=True, blit=False)
Animation.save(filename, writer=None, fps=None, dpi=None, codec=None,
               bitrate=None, extra_args=None, metadata=None, extra_anim=None,
               savefig_kwargs=None, *, progress_callback=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fig` | Figure | required | Figure to animate |
| `func` | callable | required | Called as `func(frame, *fargs)`; returns iterable of changed artists if `blit=True` |
| `frames` | int, iterable, generator, or None | None | Frame values |
| `interval` | int | 200 | Delay between frames in ms |
| `blit` | bool | False | Only redraw changed artists |

You must keep a reference to the animation object or it is garbage-collected.

```python
from matplotlib.animation import FuncAnimation, PillowWriter

x = np.linspace(0, 2 * np.pi, 200)
fig, ax = plt.subplots()
(line,) = ax.plot(x, np.sin(x))
ax.set_ylim(-1.1, 1.1)

def update(frame):
    line.set_ydata(np.sin(x + frame / 10))
    return (line,)

anim = FuncAnimation(fig, update, frames=60, interval=50, blit=True)
anim.save("wave.gif", writer=PillowWriter(fps=20))
```

### Artist property access

Every Artist has `get_*` / `set_*` methods plus a generic `set(**kwargs)`, and `plt.setp` / `plt.getp` helpers.

```python
fig, ax = plt.subplots()
(line,) = ax.plot([1, 2, 3])
line.set(color="tab:green", linewidth=3, marker="o")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
print(line.get_color(), line.get_linewidth())
print(plt.getp(line, "marker"))
```

## Tutorials

### Tutorial 1: Training curves dashboard

Goal: visualize a training run (loss, accuracy, learning rate) in a single report figure.

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. Simulate a training history (replace with your Keras/PyTorch logs)
rng = np.random.default_rng(0)
epochs = np.arange(1, 51)
train_loss = 1.2 * np.exp(-epochs / 12) + 0.05 + rng.normal(0, 0.01, 50)
val_loss = (1.2 * np.exp(-epochs / 12) + 0.12
            + 0.002 * np.maximum(0, epochs - 25) ** 1.3 + rng.normal(0, 0.015, 50))
train_acc = 1 - 0.5 * np.exp(-epochs / 10)
val_acc = train_acc - 0.04 - 0.001 * np.maximum(0, epochs - 25)
lr = 1e-3 * 0.5 ** (epochs // 10)

# 2. Build a mosaic: big loss panel on top, two small panels below
fig, axd = plt.subplot_mosaic([["loss", "loss"], ["acc", "lr"]],
                              figsize=(10, 7), layout="constrained")

# 3. Loss with best epoch marked
ax = axd["loss"]
ax.plot(epochs, train_loss, label="train")
ax.plot(epochs, val_loss, label="validation")
best = int(np.argmin(val_loss))
ax.axvline(epochs[best], color="gray", ls="--", lw=1)
ax.annotate(f"best epoch = {epochs[best]}\nval loss = {val_loss[best]:.3f}",
            xy=(epochs[best], val_loss[best]), xytext=(epochs[best] + 5, 0.8),
            arrowprops={"arrowstyle": "->"})
ax.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
ax.legend()

# 4. Accuracy formatted as percent
ax = axd["acc"]
ax.plot(epochs, train_acc, label="train")
ax.plot(epochs, val_acc, label="validation")
ax.yaxis.set_major_formatter("{x:.0%}")
ax.set(title="Accuracy", xlabel="epoch")
ax.legend(loc="lower right")

# 5. Learning-rate schedule on a log scale
ax = axd["lr"]
ax.step(epochs, lr, where="post", color="tab:purple")
ax.set(title="Learning rate", xlabel="epoch", yscale="log")

# 6. Clean up and save
for a in axd.values():
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=0.3)
fig.suptitle("Training run #42", fontsize=14)
fig.savefig("training_dashboard.png", dpi=150)
plt.close(fig)
```

Key points: `subplot_mosaic` gives named Axes, `layout="constrained"` prevents overlaps, the string formatter `"{x:.0%}"` turns fractions into percentages, and `step(where="post")` draws piecewise-constant schedules correctly.

### Tutorial 2: Classifier evaluation figure (confusion matrix, ROC, PR)

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Fit a model
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

# 2. Confusion matrix drawn by hand with imshow + text
cm = confusion_matrix(y_te, pred)
ax = axs[0]
im = ax.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
ax.set_xticks([0, 1], labels=["malignant", "benign"])
ax.set_yticks([0, 1], labels=["malignant", "benign"])
ax.set(xlabel="predicted", ylabel="actual", title="Confusion matrix")
fig.colorbar(im, ax=ax, shrink=0.8)

# 3. ROC curve with diagonal reference
fpr, tpr, _ = roc_curve(y_te, proba)
ax = axs[1]
ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_te, proba):.3f}")
ax.axline((0, 0), slope=1, color="gray", ls="--", lw=1)
ax.set(xlabel="false positive rate", ylabel="true positive rate",
       title="ROC", xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
ax.legend(loc="lower right")

# 4. Precision-recall curve with baseline prevalence
prec, rec, _ = precision_recall_curve(y_te, proba)
ax = axs[2]
ax.plot(rec, prec, lw=2, label=f"AP = {average_precision_score(y_te, proba):.3f}")
ax.axhline(y_te.mean(), color="gray", ls="--", lw=1, label="prevalence")
ax.set(xlabel="recall", ylabel="precision", title="Precision-Recall",
       xlim=(0, 1), ylim=(0, 1.01))
ax.legend(loc="lower left")

fig.savefig("classifier_eval.png", dpi=150)
plt.close(fig)
```

Drawing the confusion matrix yourself shows the core pattern for any annotated heatmap: `imshow` the matrix, then loop over cells with `ax.text`, choosing a contrasting text color. scikit-learn's `ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=axs[0])` does the same thing in one line and accepts your Axes.

### Tutorial 3: Visualizing embeddings with categorical colors

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# 1. Project 64-D digit images to 2-D
digits = load_digits()
emb = PCA(n_components=2, random_state=0).fit_transform(digits.data)
labels = digits.target

# 2. One scatter call per class gives automatic legend entries
fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
cmap = plt.get_cmap("tab10")
for k in range(10):
    m = labels == k
    ax.scatter(emb[m, 0], emb[m, 1], s=10, color=cmap(k), alpha=0.7, label=str(k))

# 3. Label each cluster at its median position
for k in range(10):
    cx, cy = np.median(emb[labels == k], axis=0)
    ax.text(cx, cy, str(k), fontsize=16, weight="bold", ha="center", va="center",
            bbox={"boxstyle": "circle", "fc": "white", "alpha": 0.8})

# 4. Legend with larger markers than the plot, placed outside
ax.legend(title="digit", markerscale=2.5, ncols=2, loc="upper left",
          bbox_to_anchor=(1.01, 1), frameon=False)
ax.set(title="PCA of handwritten digits", xlabel="PC1", ylabel="PC2")

# 5. Small insets showing sample images
for i, idx in enumerate([0, 1]):
    ins = ax.inset_axes([0.02 + 0.12 * i, 0.02, 0.1, 0.1])
    ins.imshow(digits.images[idx], cmap="gray_r")
    ins.set_axis_off()

fig.savefig("embedding.png", dpi=150)
plt.close(fig)
```

Custom legend handles are useful when the plot is a single `scatter` call colored by a numeric array:

```python
handles = [Line2D([], [], marker="o", ls="", color=cmap(k), label=str(k)) for k in range(10)]
# ax.legend(handles=handles, title="digit")
```

### Tutorial 4: Animating gradient descent and live-updating plots

```python
import matplotlib
matplotlib.use("Agg")       # remove this line to see a live window (needs a GUI backend)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# 1. Loss surface and gradient descent path
def loss(w):
    return 0.5 * w[0] ** 2 + 2.0 * w[1] ** 2

def grad(w):
    return np.array([w[0], 4.0 * w[1]])

w = np.array([3.5, 2.0])
path = [w.copy()]
for _ in range(40):
    w = w - 0.2 * grad(w)
    path.append(w.copy())
path = np.array(path)

# 2. Static background: contour plot
g = np.linspace(-4, 4, 200)
W0, W1 = np.meshgrid(g, g)
Z = 0.5 * W0 ** 2 + 2.0 * W1 ** 2
fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
ax.contour(W0, W1, Z, levels=np.logspace(-1, 1.6, 15), cmap="viridis")
(trail,) = ax.plot([], [], "o-", color="tab:red", ms=4)
title = ax.set_title("")
ax.set_aspect("equal")

# 3. Update function: only modify existing artists
def update(i):
    trail.set_data(path[: i + 1, 0], path[: i + 1, 1])
    title.set_text(f"step {i}  loss={loss(path[i]):.4f}")
    return trail, title

# 4. Build and save the animation (keep a reference in `anim`)
anim = FuncAnimation(fig, update, frames=len(path), interval=100)
anim.save("gd.gif", writer=PillowWriter(fps=10))
plt.close(fig)
```

For a live plot inside a training loop with an interactive backend, update artist data and call `plt.pause`:

```python
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig, ax = plt.subplots()
(line,) = ax.plot([], [])
losses = []
for step in range(100):
    losses.append(np.exp(-step / 30) + np.random.rand() * 0.05)
    line.set_data(range(len(losses)), losses)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)
plt.ioff()
```

### Tutorial 5: Publication-ready figure with a consistent style

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

style = {
    "figure.figsize": (3.4, 2.6),       # single-column journal width in inches
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,                 # embed TrueType so text stays editable
    "svg.fonttype": "none",
}

n = np.array([100, 300, 1000, 3000, 10000])
with mpl.rc_context(style):
    fig, ax = plt.subplots(layout="constrained")
    for name, rate, marker in [("model A", 0.30, "o"), ("model B", 0.40, "s")]:
        ax.plot(n, 0.5 * n ** -rate, marker=marker, ms=3, label=name)
    ax.set(xscale="log", yscale="log", xlabel="training samples", ylabel="test error")
    ax.legend(frameon=False)
    fig.savefig("learning_curve.pdf")
    fig.savefig("learning_curve.png")
plt.close("all")
```

`rc_context` keeps styling local; saving to PDF with `pdf.fonttype = 42` produces vector output with editable text that journals accept.

## Performance & Best Practices

### Use the OO interface and pass `ax` around

Write plotting helpers as `def plot_x(..., ax=None)` that draw on a supplied Axes and return it. They compose into grids, work with seaborn and pandas, and never fight over global state.

### Close figures you do not need

Each open pyplot figure holds memory until closed. In loops that generate many images:

```python
import matplotlib.pyplot as plt
import numpy as np

for i in range(1000):
    fig, ax = plt.subplots()
    ax.plot(np.random.rand(100))
    fig.savefig(f"out_{i}.png")
    plt.close(fig)
```

Alternatively create figures with `matplotlib.figure.Figure()` directly. They are not registered with pyplot and are garbage-collected normally:

```python
from matplotlib.figure import Figure

fig = Figure(figsize=(4, 3))
ax = fig.subplots()
ax.plot([1, 2, 3])
fig.savefig("no_pyplot.png")      # uses the Agg canvas automatically
```

This is the recommended pattern in web servers (Flask, FastAPI) because pyplot's global state is not thread-safe.

### Large data

| Problem | Technique |
|---|---|
| Millions of scatter points | `hexbin`, `hist2d`, or `ax.plot(x, y, ",")` (pixel markers) instead of `scatter`; downsample |
| Huge vector PDFs | `ax.scatter(..., rasterized=True)` keeps axes and text vector but rasterizes the dots |
| Long line series | `rcParams["path.simplify"]` (on by default); set `rcParams["agg.path.chunksize"] = 10000` to avoid overflow |
| Many lines | `matplotlib.collections.LineCollection` is far faster than many `plot` calls |
| Many rectangles/circles | `PatchCollection` instead of individual patches |
| Animations | `blit=True`, update data with `set_data`, never recreate artists per frame |

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

rng = np.random.default_rng(0)
segs = [np.column_stack([np.arange(100), rng.normal(size=100).cumsum()]) for _ in range(2000)]
fig, ax = plt.subplots()
ax.add_collection(LineCollection(segs, linewidths=0.3, alpha=0.2))
ax.autoscale()
```

### Visual best practices

- Prefer perceptually uniform colormaps (`viridis`, `cividis`) for sequential data and diverging maps centered on a meaningful midpoint (`TwoSlopeNorm`, `CenteredNorm`) for signed data. Avoid `jet`.
- Use the `tab10` cycle or colorblind-friendly palettes for categories; do not encode more than about eight categories by color alone.
- Label axes with units; use `PercentFormatter` or `EngFormatter` instead of manual arithmetic in labels.
- Start bar charts at zero; line charts need not.
- Use `layout="constrained"` by default; `bbox_inches="tight"` at save time is a fallback.
- Set `figsize` to the physical size you need and keep font sizes fixed, rather than scaling the image afterwards.

### Reproducibility

- Pin `matplotlib` in your requirements; minor versions occasionally change defaults (for example image interpolation in 3.10).
- Keep a project `.mplstyle` file and load it with `plt.style.use("path/to/style.mplstyle")`.
- For deterministic SVG output (diffable in git), set `rcParams["svg.hashsalt"] = "fixed"`.

## Common Errors & Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown` | `plt.show()` with the Agg backend (headless server, CI) | Use `fig.savefig(...)`, or install a GUI toolkit and use `QtAgg`/`TkAgg` |
| `RuntimeWarning: More than 20 figures have been opened.` | Figures created in a loop and never closed | `plt.close(fig)` after saving, or use `matplotlib.figure.Figure` |
| `ValueError: x and y must have same first dimension, but have shapes (10,) and (9,)` | Mismatched array lengths | Check lengths; after `np.diff` slice `x[1:]` |
| `ValueError: 'c' argument has 3 elements, which is inconsistent with 'x' and 'y' with size 100.` | `c=` is neither one color nor one value per point | Pass per-point values, or `color=` for a single color |
| `TypeError: 'Axes' object is not subscriptable` | `plt.subplots()` with one panel returns a single Axes | Use `squeeze=False` or do not index |
| `AttributeError: 'numpy.ndarray' object has no attribute 'plot'` | Calling a method on the whole `axs` array | Index it (`axs[0, 1]`) or iterate `axs.flat` |
| `AttributeError: 'Axes' object has no attribute 'xlabel'` | pyplot names used in OO code | Use `ax.set_xlabel`, `ax.set_title`, `ax.set_xlim` |
| `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'` | Removed in 3.9 | `matplotlib.colormaps["name"]` or `plt.get_cmap("name")` |
| `OSError: 'seaborn-whitegrid' is not a valid package style...` | Style renamed in 3.6, old names removed in 3.8 | `plt.style.use("seaborn-v0_8-whitegrid")`, or `sns.set_theme()` |
| `MatplotlibDeprecationWarning` about boxplot `labels` | Renamed to `tick_labels` in 3.9 | Use `tick_labels=` |
| `MatplotlibDeprecationWarning` about `vert` | 3.10 boxplot/violinplot change | Use `orientation="horizontal"` |
| `UserWarning: Glyph 12354 ... missing from font(s) DejaVu Sans.` | Font lacks the characters (CJK, symbols) | Set `rcParams["font.family"]` to a font that has them, e.g. `"Noto Sans CJK JP"` |
| `RuntimeError: Failed to process string with tex because latex could not be found` | `text.usetex=True` without a LaTeX install | Install LaTeX, or keep `usetex=False` and use mathtext |
| `ValueError: '...' is not a valid value for cmap` | Typo in colormap name | Check `list(matplotlib.colormaps)` |
| `OverflowError: In draw_path: Exceeded cell block limit` | Extremely long paths in Agg | `rcParams["agg.path.chunksize"] = 10000`, or downsample |
| Animation shows nothing / `UserWarning: Animation was deleted without rendering anything` | Animation object garbage-collected | Keep a reference: `anim = FuncAnimation(...)` |
| `RuntimeError: Requested MovieWriter (ffmpeg) not available` | ffmpeg not installed | Install ffmpeg, or save GIF with `PillowWriter` |
| Labels cut off in saved file | Decorations outside the figure | `layout="constrained"` or `savefig(..., bbox_inches="tight")` |
| Blank image saved | `savefig` called after `plt.show()` closed the figure | Call `savefig` before `show` |
| No plots in Jupyter | Backend misconfigured | `%matplotlib inline`, or `%matplotlib widget` with ipympl |
| `Tcl_AsyncDelete: async handler deleted by the wrong thread` | pyplot/Tk used from worker threads | `matplotlib.use("Agg")` and the `Figure` API in threads |
| `UserWarning: Tight layout not applied...` | `tight_layout` cannot fit decorations | Switch to `layout="constrained"` |

### Debugging tips

```python
import matplotlib as mpl

print(mpl.matplotlib_fname())          # which matplotlibrc is in effect
print(mpl.rcParams["backend"])
mpl.set_loglevel("info")               # verbose logs, e.g. font lookup
```

After installing new fonts, delete the font cache in the directory reported by `matplotlib.get_cachedir()` so Matplotlib rebuilds it.

## Interoperability

### NumPy

All plotting functions accept NumPy arrays. NaN values and masked arrays (`np.ma`) create gaps in lines. Image functions expect `(H, W)`, `(H, W, 3)`, or `(H, W, 4)` arrays.

### pandas

pandas' `.plot` accessor uses Matplotlib and returns `Axes`:

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({"a": range(10), "b": [x ** 2 for x in range(10)]})
fig, axs = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
df.plot(ax=axs[0], title="lines")
df.plot.scatter(x="a", y="b", ax=axs[1])
```

Matplotlib understands `datetime64` and pandas `Timestamp` values on axes through `matplotlib.dates`. Use `mdates.ConciseDateFormatter(mdates.AutoDateLocator())` for tidy date labels. String categories on an axis are supported natively.

### seaborn

seaborn's axes-level functions accept `ax=` and return the Matplotlib `Axes`; figure-level functions return a `FacetGrid` exposing `.figure` and `.axes`. All Matplotlib customization applies afterwards.

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig, ax = plt.subplots()
sns.histplot(x=np.random.randn(500), kde=True, ax=ax)
ax.set_title("seaborn on a Matplotlib Axes")
```

### scikit-learn

The `sklearn.metrics` and `sklearn.inspection` display classes render with Matplotlib and accept `ax=`: `ConfusionMatrixDisplay`, `RocCurveDisplay`, `PrecisionRecallDisplay`, `DetCurveDisplay`, `PredictionErrorDisplay`, `PartialDependenceDisplay`, `DecisionBoundaryDisplay`, `LearningCurveDisplay`, `ValidationCurveDisplay`.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay

X, y = load_breast_cancer(return_X_y=True)
clf = RandomForestClassifier(random_state=0).fit(X, y)
fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(clf, X, y, ax=ax)
```

### PyTorch and TensorFlow

Convert tensors to NumPy before plotting (`t.detach().cpu().numpy()`). Image tensors in CHW layout must be transposed to HWC (`img.permute(1, 2, 0)`). TensorBoard accepts figures via `torch.utils.tensorboard.SummaryWriter.add_figure(tag, fig, global_step)`; Weights & Biases via `wandb.log({"plot": wandb.Image(fig)})`.

```python
import matplotlib.pyplot as plt
import torch

img = torch.rand(3, 32, 32)
fig, ax = plt.subplots()
ax.imshow(img.permute(1, 2, 0).numpy())
```

### statsmodels

statsmodels' graphics functions (`sm.qqplot`, `plot_acf`, `plot_pacf`, `plot_regress_exog`) accept `ax=` (most of them) and return a `Figure`.

### Exporting to other tools

- `fig.savefig(buf, format="png")` into an `io.BytesIO` buffer for web APIs.
- Streamlit: `st.pyplot(fig)`. Gradio: `gr.Plot` accepts a Matplotlib figure.
- `fig.canvas.draw(); np.asarray(fig.canvas.buffer_rgba())` converts a figure to an RGBA NumPy array (Agg-based canvases).

## Cheat Sheet

### Setup and figures

| Task | Code |
|---|---|
| Import | `import matplotlib.pyplot as plt` |
| One Axes | `fig, ax = plt.subplots(figsize=(6, 4))` |
| Grid of Axes | `fig, axs = plt.subplots(2, 3, layout="constrained")` |
| Named layout | `fig, axd = plt.subplot_mosaic("AB;CC")` |
| Shared axes | `plt.subplots(2, 1, sharex=True)` |
| Uneven widths | `plt.subplots(1, 2, width_ratios=[3, 1])` |
| Headless backend | `matplotlib.use("Agg")` |
| Save | `fig.savefig("f.png", dpi=300, bbox_inches="tight")` |
| Close | `plt.close(fig)` / `plt.close("all")` |
| Figure title | `fig.suptitle("Title")` |

### Plot types

| Task | Code |
|---|---|
| Line | `ax.plot(x, y, "o-", label="a")` |
| Scatter colored by value | `sc = ax.scatter(x, y, c=v, cmap="viridis")` |
| Bar | `ax.bar(cats, vals)` |
| Horizontal bar | `ax.barh(cats, vals)` |
| Bar value labels | `ax.bar_label(bars, fmt="%.2f")` |
| Histogram | `ax.hist(x, bins=30, density=True)` |
| 2D density | `ax.hexbin(x, y, gridsize=40, bins="log")` |
| Box plot | `ax.boxplot(data, tick_labels=names)` |
| Violin | `ax.violinplot(data, showmedians=True)` |
| ECDF | `ax.ecdf(x)` |
| Error bars | `ax.errorbar(x, y, yerr=e, fmt="o", capsize=3)` |
| Confidence band | `ax.fill_between(x, lo, hi, alpha=0.3)` |
| Heatmap | `im = ax.imshow(M, cmap="Blues"); fig.colorbar(im, ax=ax)` |
| Contour | `ax.contourf(X, Y, Z, levels=20)` |
| 3D surface | `ax = fig.add_subplot(projection="3d"); ax.plot_surface(X, Y, Z)` |
| Reference lines | `ax.axhline(0)` / `ax.axvline(5)` / `ax.axline((0, 0), slope=1)` |
| Shaded region | `ax.axvspan(2, 4, alpha=0.2)` |

### Customization

| Task | Code |
|---|---|
| Labels and title | `ax.set(xlabel="x", ylabel="y", title="t")` |
| Limits | `ax.set_xlim(0, 10)` |
| Log scale | `ax.set_yscale("log")` |
| Ticks with labels | `ax.set_xticks([0, 1], labels=["no", "yes"])` |
| Rotate tick labels | `ax.tick_params(axis="x", labelrotation=45)` |
| Percent axis | `ax.yaxis.set_major_formatter(PercentFormatter(1.0))` |
| Integer ticks | `ax.xaxis.set_major_locator(MaxNLocator(integer=True))` |
| Hide spines | `ax.spines[["top", "right"]].set_visible(False)` |
| Grid | `ax.grid(alpha=0.3)` |
| Legend outside | `ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))` |
| Second y-axis | `ax2 = ax.twinx()` |
| Equal aspect | `ax.set_aspect("equal")` |
| Hide axes | `ax.set_axis_off()` |
| Annotate | `ax.annotate("txt", xy=(x, y), xytext=(x2, y2), arrowprops={"arrowstyle": "->"})` |
| Text in axes coords | `ax.text(0.05, 0.95, "s", transform=ax.transAxes, va="top")` |
| Math text | `ax.set_title(r"$\alpha^2$")` |
| Inset | `ins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])` |

### Styling and colors

| Task | Code |
|---|---|
| List styles | `plt.style.available` |
| Apply style | `plt.style.use("ggplot")` |
| Temporary style | `with plt.style.context("dark_background"): ...` |
| Change a default | `plt.rcParams["font.size"] = 12` |
| Temporary rcParams | `with mpl.rc_context({"lines.linewidth": 2}): ...` |
| Reset | `mpl.rcdefaults()` |
| Get colormap | `cmap = mpl.colormaps["viridis"]` |
| N discrete colors | `mpl.colormaps["viridis"].resampled(5)` |
| Diverging norm | `norm=mpl.colors.TwoSlopeNorm(vcenter=0)` |
| Log color scale | `norm=mpl.colors.LogNorm()` |
| Custom colormap | `LinearSegmentedColormap.from_list("m", ["white", "red"])` |

## Further Resources

- Official documentation: https://matplotlib.org/stable/
- User guide: https://matplotlib.org/stable/users/index.html
- Quick start guide: https://matplotlib.org/stable/users/explain/quick_start.html
- API reference: https://matplotlib.org/stable/api/index.html
- Example gallery: https://matplotlib.org/stable/gallery/index.html
- Tutorials: https://matplotlib.org/stable/tutorials/index.html
- Release notes: https://matplotlib.org/stable/users/release_notes.html
- Cheat sheets and handouts: https://matplotlib.org/cheatsheets/
- Choosing colormaps: https://matplotlib.org/stable/users/explain/colors/colormaps.html
- GitHub repository: https://github.com/matplotlib/matplotlib
- Community forum: https://discourse.matplotlib.org/
- Book: Nicolas P. Rougier, "Scientific Visualization: Python + Matplotlib" (free): https://github.com/rougier/scientific-visualization-book
- Book: Jake VanderPlas, "Python Data Science Handbook", chapter "Visualization with Matplotlib" (free online): https://jakevdp.github.io/PythonDataScienceHandbook/
