# Seaborn

> Statistical data visualization built on Matplotlib, designed for tidy pandas DataFrames.

Seaborn is a high-level interface for drawing attractive, informative statistical graphics. You describe *what* you want to see in terms of DataFrame columns (x, y, hue, size, style, row, col), and seaborn handles the aggregation, statistical estimation (means, confidence intervals, kernel density estimates, regression fits), color mapping, legends, and faceting. Every seaborn plot is a Matplotlib plot underneath, so the full Matplotlib API remains available for final touches.

This page covers **seaborn 0.13.x** (0.13.2 is the current stable release at the time of writing), including the `seaborn.objects` interface introduced in 0.12. Notable renames and deprecations from older releases are flagged inline and summarized in the troubleshooting section.

## Overview

### What it is

Seaborn provides three families of plotting functions plus a newer declarative interface:

| Family | Axes-level functions | Figure-level function |
|---|---|---|
| Relational | `scatterplot`, `lineplot` | `relplot` |
| Distributional | `histplot`, `kdeplot`, `ecdfplot`, `rugplot` | `displot` |
| Categorical | `stripplot`, `swarmplot`, `boxplot`, `violinplot`, `boxenplot`, `pointplot`, `barplot`, `countplot` | `catplot` |
| Regression | `regplot`, `residplot` | `lmplot` |
| Matrix | `heatmap` | `clustermap` |
| Multi-plot grids | | `pairplot`, `jointplot`, `FacetGrid`, `PairGrid`, `JointGrid` |

plus **`seaborn.objects`** (`so.Plot`), a composable grammar-of-graphics style API.

### History and maintainer

Seaborn was created by **Michael Waskom** while at Stanford (first release 2012) and remains primarily maintained by him with community contributors. It is developed at `github.com/mwaskom/seaborn` and was described in the Journal of Open Source Software paper "seaborn: statistical data visualization" (Waskom, 2021). Major milestones:

- **0.9 (2018)**: `relplot`, `scatterplot`, `lineplot`; `factorplot` renamed to `catplot`.
- **0.11 (2020)**: new distribution API (`histplot`, `displot`, `ecdfplot`); `distplot` deprecated.
- **0.12 (2022)**: `seaborn.objects` interface; `errorbar=` replaces `ci=`; data variables must be passed as keywords.
- **0.13 (2023)**: categorical functions rewritten (native numeric/date scales, `fill`, `gap`, `density_norm`, `stat=` in `countplot`), `so.KDE` stat.

### When to use it

- **Exploratory data analysis** of tabular data: distributions, relationships, group comparisons in one line each.
- **Plots that need statistics**: means with bootstrapped confidence intervals, KDEs, regression lines with uncertainty bands.
- **Small multiples / faceting** by categorical variables (`col=`, `row=`).
- **Correlation and confusion matrices** with annotations (`heatmap`), hierarchical clustering (`clustermap`).
- Producing consistent, good-looking defaults with minimal styling effort.

### When not to use it

- **Pixel-perfect custom figures** with unusual layouts: use Matplotlib directly (you can still call seaborn on individual Axes).
- **Interactive dashboards** (hover, zoom, linked selection): Plotly, Bokeh, Altair.
- **Huge datasets**: bootstrapped confidence intervals and KDEs get slow beyond a few hundred thousand rows; pre-aggregate or disable estimation (`errorbar=None`).
- **Non-tabular data** such as images or tensors.

### Where it fits in the ML stack

```text
pandas / NumPy data  ->  seaborn (statistical mapping, estimation, faceting)
                          -> Matplotlib (Figure, Axes, rendering, saving)
```

In ML workflows seaborn is mostly used in the EDA phase (feature distributions, target relationships, correlation matrices, class balance) and in model reporting (comparing metrics across models or folds, confusion matrices, residual analysis). Its optional statistics come from SciPy (KDE fallbacks, clustering) and statsmodels (lowess and robust regression in `regplot`).

## Installation

### pip

```bash
python -m pip install -U seaborn
```

Seaborn 0.13 requires Python 3.8+, NumPy, pandas, and Matplotlib (installed automatically). Some features need SciPy or statsmodels; install them with the `stats` extra:

```bash
python -m pip install -U "seaborn[stats]"
```

| Optional dependency | Needed for |
|---|---|
| `scipy` | `clustermap` (hierarchical clustering); seaborn falls back to its own KDE implementation when SciPy is absent |
| `statsmodels` | `regplot`/`lmplot` with `lowess=True`, `robust=True`, or `logistic=True` |

### conda

```bash
conda install -c conda-forge seaborn
```

The conda-forge `seaborn` package includes SciPy and statsmodels; `seaborn-base` installs only the core requirements.

### Verifying the install

```python
import seaborn as sns
import matplotlib
import pandas as pd

print(sns.__version__)          # e.g. '0.13.2'
print(matplotlib.__version__)
print(pd.__version__)

print(sns.get_dataset_names()[:5])    # needs internet access
tips = sns.load_dataset("tips")       # downloads and caches the CSV
ax = sns.scatterplot(data=tips, x="total_bill", y="tip")
ax.figure.savefig("check.png")
```

Note: `load_dataset` and `get_dataset_names` fetch files from the `mwaskom/seaborn-data` GitHub repository and cache them under `~/seaborn-data`. They are for examples only; offline machines should load data from local files.

## Core Concepts

### Tidy (long-form) data and semantic mappings

Seaborn works best with **long-form** ("tidy") data: one row per observation, one column per variable. You then *map* columns to visual properties:

| Semantic parameter | Visual property |
|---|---|
| `x`, `y` | Position |
| `hue` | Color |
| `size` | Marker size or line width |
| `style` | Marker shape or line dash |
| `row`, `col` | Facets (separate subplots), figure-level functions only |
| `units` | Draw separate lines per unit without a legend (lineplot/relplot) |

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
print(tips.head())
#    total_bill   tip     sex smoker  day    time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2

sns.scatterplot(data=tips, x="total_bill", y="tip",
                hue="time", size="size", style="smoker")
plt.savefig("semantics.png")
```

Wide-form data (a DataFrame where each column is a series) is also accepted by many functions; seaborn then plots each column as a separate level:

```python
import numpy as np
import pandas as pd

wide = pd.DataFrame(np.random.randn(100, 3).cumsum(axis=0), columns=["a", "b", "c"])
sns.lineplot(data=wide)          # one line per column

long = wide.reset_index().melt(id_vars="index", var_name="series", value_name="value")
sns.lineplot(data=long, x="index", y="value", hue="series")   # equivalent, long-form
```

Version note: since 0.12, data variables must be passed as **keyword arguments**. `sns.scatterplot(x, y)` positional calls raise a `TypeError`. Only `data` may be positional.

### Axes-level vs figure-level functions

This distinction explains most seaborn behavior.

- **Axes-level** functions (`scatterplot`, `histplot`, `boxplot`, `heatmap`, ...) draw onto a single Matplotlib `Axes` (the current one, or the one you pass via `ax=`) and **return that Axes**. They fit into any Matplotlib layout.
- **Figure-level** functions (`relplot`, `displot`, `catplot`, `lmplot`, `pairplot`, `jointplot`, `clustermap`) create their **own figure** through a seaborn grid object, support faceting with `row=`/`col=`, put the legend outside the plot, and **return the grid** (`FacetGrid`, `PairGrid`, `JointGrid`, `ClusterGrid`). Size them with `height=` and `aspect=` instead of `figsize`.

```python
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins")

# Axes-level: you control the figure
fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
sns.histplot(data=penguins, x="flipper_length_mm", hue="species", ax=axs[0])
sns.boxplot(data=penguins, x="species", y="body_mass_g", ax=axs[1])

# Figure-level: seaborn controls the figure, facets for free
g = sns.displot(data=penguins, x="flipper_length_mm", hue="species",
                col="sex", kind="kde", height=3.5, aspect=1.2)
print(type(g))                    # seaborn.axisgrid.FacetGrid
g.set_axis_labels("Flipper length (mm)", "Density")
g.figure.savefig("facets.png")
```

| | Axes-level | Figure-level |
|---|---|---|
| Returns | `matplotlib.axes.Axes` | `FacetGrid` / `PairGrid` / `JointGrid` |
| Target | `ax=` parameter | creates a new figure |
| Size | Matplotlib `figsize` | `height=` (inches per facet), `aspect=` (width/height) |
| Faceting | no | `row=`, `col=`, `col_wrap=` |
| Legend | inside Axes | outside, on the figure (default) |
| Access figure | `ax.figure` | `g.figure` (`g.fig` is an older alias) |

### Statistical estimation and error bars

Functions that aggregate (`lineplot`, `barplot`, `pointplot`, `relplot(kind="line")`, `catplot(kind="bar"/"point")`) compute an estimator (default: mean) per group and an error bar. Since 0.12 the error bar is controlled by `errorbar=`:

| `errorbar=` value | Meaning | Type |
|---|---|---|
| `("ci", 95)` | 95% bootstrap confidence interval of the estimator (default) | Inferential |
| `("pi", 50)` | 50% percentile interval of the data | Descriptive |
| `("se", 1)` | +/- 1 standard error | Inferential |
| `("sd", 1)` | +/- 1 standard deviation | Descriptive |
| a callable | `f(values) -> (min, max)` | custom |
| `None` | no error bars (fast) | |

```python
fmri = sns.load_dataset("fmri")
sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event",
             errorbar=("sd", 1), estimator="median")
```

Bootstrapping is controlled by `n_boot=1000` and `seed=`. The old `ci=95` / `ci="sd"` parameter is deprecated (0.12) in favor of `errorbar=`, except in `regplot`/`lmplot`, where `ci=` is still the parameter for the regression band.

### Color palettes

Seaborn chooses palettes based on the data type of `hue`:

- **Categorical** `hue`: a qualitative palette (default `"deep"`, the default cycle after `set_theme`).
- **Numeric** `hue`: a sequential cubehelix-based colormap, with `hue_norm=` to control the value range.

```python
import seaborn as sns

print(sns.color_palette())                       # current cycle as RGB tuples
print(sns.color_palette("Set2", 4).as_hex())
pal = sns.color_palette("viridis", as_cmap=True) # Matplotlib Colormap
sns.palplot(sns.color_palette("husl", 8))        # quick visual swatch
```

Named seaborn palettes: `deep`, `muted`, `pastel`, `bright`, `dark`, `colorblind` (qualitative); `rocket`, `mako`, `flare`, `crest` (sequential); `vlag`, `icefire` (diverging). Any Matplotlib colormap name also works. Append `_r` to reverse and `_d` for a dark variant of a Matplotlib map.

Pass a dict to map specific levels to specific colors:

```python
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time",
                palette={"Lunch": "tab:orange", "Dinner": "tab:blue"})
```

### Themes: style, context, palette

`sns.set_theme()` changes Matplotlib's global rcParams. It is not applied on import (it was before 0.8). Three independent parts:

- **style**: `darkgrid` (default for `set_theme`), `whitegrid`, `dark`, `white`, `ticks` - background, grid, spines.
- **context**: `paper`, `notebook` (default), `talk`, `poster` - scales fonts and line widths.
- **palette**: the color cycle.

```python
sns.set_theme(style="whitegrid", context="talk", palette="colorblind", font_scale=0.9)

with sns.axes_style("ticks"), sns.plotting_context("paper"):
    ax = sns.histplot(data=tips, x="tip")
    sns.despine(ax=ax, trim=True)
```

`sns.set()` is an older alias for `set_theme()` and still works. `sns.reset_defaults()` and `sns.reset_orig()` restore Matplotlib settings.

### The objects interface (seaborn.objects)

Introduced in 0.12, `seaborn.objects` builds a plot from composable pieces rather than many function parameters:

- `so.Plot(data, x=..., y=..., color=...)` - data and default mappings.
- `.add(Mark, Stat/Move...)` - a layer: a **mark** (Dot, Line, Bar, Area, ...), optionally transformed by a **stat** (Agg, Hist, KDE, ...) and **moves** (Dodge, Jitter, Stack, ...).
- `.facet()`, `.pair()`, `.scale()`, `.label()`, `.limit()`, `.layout()`, `.theme()`, `.share()`, `.on()`.
- `.show()` / `.save()` / display in a notebook.

```python
import seaborn.objects as so
import seaborn as sns

penguins = sns.load_dataset("penguins").dropna()
(
    so.Plot(penguins, x="bill_length_mm", y="bill_depth_mm", color="species")
    .add(so.Dot(alpha=0.6))
    .add(so.Line(), so.PolyFit(order=1))
    .facet(col="island")
    .label(x="Bill length (mm)", y="Bill depth (mm)", color="Species")
    .layout(size=(10, 3.5))
    .save("objects_demo.png", dpi=120)
)
```

`Plot` objects are immutable: every method returns a new `Plot`, so you can build a base spec and branch from it. The interface is still marked experimental in 0.13, but it is stable enough for regular use.

## API Reference

All examples assume:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

tips = sns.load_dataset("tips")
penguins = sns.load_dataset("penguins")
fmri = sns.load_dataset("fmri")
flights = sns.load_dataset("flights")
titanic = sns.load_dataset("titanic")
```

Common parameters shared by most functions (not repeated in every table):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | DataFrame, dict, array | None | Input data |
| `x`, `y` | str (column) or vector | None | Positional variables |
| `hue` | str or vector | None | Color grouping |
| `palette` | str, list, dict, Colormap | None | Colors for `hue` levels |
| `hue_order` | list | None | Order of categorical `hue` levels |
| `hue_norm` | tuple or Normalize | None | Range for numeric `hue` |
| `legend` | `"auto"`, `"brief"`, `"full"`, bool | `"auto"` | Legend control |
| `ax` | Axes | None | Target Axes (axes-level functions) |
| `**kwargs` | | | Passed to the underlying Matplotlib function |

### Relational plots

#### scatterplot

```python
sns.scatterplot(data=None, *, x=None, y=None, hue=None, size=None, style=None,
                palette=None, hue_order=None, hue_norm=None, sizes=None,
                size_order=None, size_norm=None, markers=True, style_order=None,
                legend="auto", ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `size` | str | None | Variable mapped to marker area |
| `sizes` | list, dict, or (min, max) | None | Size values or range |
| `style` | str | None | Variable mapped to marker shape |
| `markers` | bool, list, dict | True | Markers for `style` levels |
| `**kwargs` | | | `s`, `alpha`, `edgecolor`, `linewidth` passed to `Axes.scatter` |

Returns `Axes`.

```python
ax = sns.scatterplot(data=penguins, x="flipper_length_mm", y="body_mass_g",
                     hue="species", style="sex", sizes=(20, 200), alpha=0.8)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
```

#### lineplot

```python
sns.lineplot(data=None, *, x=None, y=None, hue=None, size=None, style=None,
             units=None, palette=None, hue_order=None, hue_norm=None,
             sizes=None, size_order=None, size_norm=None, dashes=True,
             markers=None, style_order=None, estimator="mean",
             errorbar=("ci", 95), n_boot=1000, seed=None, orient="x",
             sort=True, err_style="band", err_kws=None, legend="auto",
             ci="deprecated", ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `estimator` | str, callable, or None | `"mean"` | Aggregation of `y` at each `x`; `None` draws every observation |
| `errorbar` | str, (str, float), callable, None | `("ci", 95)` | Error representation |
| `units` | str | None | Sampling units; draws one line per unit (requires `estimator=None`) |
| `err_style` | `"band"` or `"bars"` | `"band"` | Shaded band or error bars |
| `markers`, `dashes` | bool, list, dict | None / True | Style per `style` level |
| `orient` | `"x"` or `"y"` | `"x"` | Aggregation direction |
| `sort` | bool | True | Sort by x before drawing |

Returns `Axes`.

```python
ax = sns.lineplot(data=fmri, x="timepoint", y="signal", hue="region",
                  style="event", markers=True, errorbar=("se", 2))

# Individual subject trajectories instead of an aggregate
sns.lineplot(data=fmri.query("region == 'frontal'"), x="timepoint", y="signal",
             units="subject", estimator=None, lw=0.5, alpha=0.5)
```

#### relplot

```python
sns.relplot(data=None, *, x=None, y=None, hue=None, size=None, style=None,
            units=None, row=None, col=None, col_wrap=None, row_order=None,
            col_order=None, palette=None, hue_order=None, hue_norm=None,
            sizes=None, size_order=None, size_norm=None, markers=None,
            dashes=None, style_order=None, legend="auto", kind="scatter",
            height=5, aspect=1, facet_kws=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `kind` | `"scatter"` or `"line"` | `"scatter"` | Underlying axes-level function |
| `row`, `col` | str | None | Facet variables |
| `col_wrap` | int | None | Wrap columns into multiple rows (no `row` allowed) |
| `height` | float | 5 | Height of each facet in inches |
| `aspect` | float | 1 | Width = `aspect * height` |
| `facet_kws` | dict | None | Passed to `FacetGrid` (e.g. `{"sharey": False}`) |

Returns `FacetGrid`.

```python
g = sns.relplot(data=fmri, x="timepoint", y="signal", hue="event",
                col="region", kind="line", height=3.5, aspect=1.3,
                facet_kws={"sharey": False})
g.set_titles("{col_name} cortex")
```

### Distribution plots

#### histplot

```python
sns.histplot(data=None, *, x=None, y=None, hue=None, weights=None,
             stat="count", bins="auto", binwidth=None, binrange=None,
             discrete=None, cumulative=False, common_bins=True,
             common_norm=True, multiple="layer", element="bars", fill=True,
             shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0,
             pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None,
             palette=None, hue_order=None, hue_norm=None, color=None,
             log_scale=None, legend=True, ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stat` | str | `"count"` | `"count"`, `"frequency"`, `"probability"`/`"proportion"`, `"percent"`, `"density"` |
| `bins` | str, int, or sequence | `"auto"` | NumPy bin rule, count, or edges |
| `binwidth` | float | None | Width of bins (overrides `bins`) |
| `binrange` | (float, float) | None | Range for bins |
| `discrete` | bool | None | Center bins on integer values |
| `multiple` | str | `"layer"` | `"layer"`, `"dodge"`, `"stack"`, `"fill"` for hue levels |
| `element` | str | `"bars"` | `"bars"`, `"step"`, `"poly"` |
| `common_norm` | bool | True | Normalize across hue levels jointly |
| `kde` | bool | False | Overlay a KDE curve |
| `log_scale` | bool or number | None | Log-scale the data axis (binning in log space) |
| `cumulative` | bool | False | Cumulative histogram |
| `y` | str | None | If both `x` and `y` given, draws a bivariate heatmap histogram |

Returns `Axes`.

```python
ax = sns.histplot(data=penguins, x="body_mass_g", hue="species",
                  stat="density", common_norm=False, element="step", kde=True)

sns.histplot(data=tips, x="total_bill", y="tip", bins=25, cbar=True)  # 2D
```

#### kdeplot

```python
sns.kdeplot(data=None, *, x=None, y=None, hue=None, weights=None,
            palette=None, hue_order=None, hue_norm=None, color=None,
            fill=None, multiple="layer", common_norm=True, common_grid=False,
            cumulative=False, bw_method="scott", bw_adjust=1,
            warn_singular=True, log_scale=None, levels=10, thresh=0.05,
            gridsize=200, cut=3, clip=None, legend=True, cbar=False,
            cbar_ax=None, cbar_kws=None, ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bw_adjust` | float | 1 | Multiply the bandwidth; < 1 is wigglier, > 1 smoother |
| `bw_method` | str, float, callable | `"scott"` | Base bandwidth rule (`"scott"`, `"silverman"`, or scalar) |
| `fill` | bool | None | Fill under the curve (replaces deprecated `shade`) |
| `multiple` | str | `"layer"` | `"layer"`, `"stack"`, `"fill"` |
| `common_norm` | bool | True | If False, each hue density integrates to 1 separately |
| `cut` | float | 3 | Extend the curve past data extremes by `cut * bw` |
| `clip` | (float, float) | None | Do not evaluate outside these limits (e.g. `(0, None)`) |
| `levels`, `thresh` | int/list, float | 10, 0.05 | Contour levels for bivariate KDE |

Returns `Axes`.

```python
sns.kdeplot(data=tips, x="total_bill", hue="time", fill=True,
            common_norm=False, bw_adjust=0.7, clip=(0, None))
sns.kdeplot(data=penguins, x="bill_length_mm", y="bill_depth_mm",
            hue="species", levels=5, thresh=0.1)
```

Version note: `shade=True` was deprecated in 0.11 in favor of `fill=True`; `bw=` was replaced by `bw_method=` and `bw_adjust=`.

#### ecdfplot

```python
sns.ecdfplot(data=None, *, x=None, y=None, hue=None, weights=None,
             stat="proportion", complementary=False, palette=None,
             hue_order=None, hue_norm=None, log_scale=None, legend=True,
             ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stat` | str | `"proportion"` | `"proportion"`, `"percent"`, or `"count"` |
| `complementary` | bool | False | Plot 1 - ECDF (survival function) |

Returns `Axes`.

```python
sns.ecdfplot(data=penguins, x="bill_length_mm", hue="species")
```

#### rugplot

```python
sns.rugplot(data=None, *, x=None, y=None, hue=None, height=0.025,
            expand_margins=True, palette=None, hue_order=None, hue_norm=None,
            legend=True, ax=None, **kwargs)
```

Draws a tick at each observation along an axis; `height` is a fraction of the Axes. Returns `Axes`.

```python
ax = sns.kdeplot(data=tips, x="tip")
sns.rugplot(data=tips, x="tip", height=0.05, ax=ax)
```

#### displot

```python
sns.displot(data=None, *, x=None, y=None, hue=None, row=None, col=None,
            weights=None, kind="hist", rug=False, rug_kws=None, log_scale=None,
            legend=True, palette=None, hue_order=None, hue_norm=None,
            color=None, col_wrap=None, row_order=None, col_order=None,
            height=5, aspect=1, facet_kws=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `kind` | `"hist"`, `"kde"`, `"ecdf"` | `"hist"` | Underlying axes-level function |
| `rug` | bool | False | Add a rug plot |
| `**kwargs` | | | Passed to `histplot`/`kdeplot`/`ecdfplot` |

Returns `FacetGrid`.

```python
g = sns.displot(data=penguins, x="flipper_length_mm", col="species",
                row="sex", binwidth=3, height=3, facet_kws={"margin_titles": True})
```

Deprecation: `distplot` (the pre-0.11 combined histogram/KDE function) is deprecated and emits a warning that it will be removed. Replace `sns.distplot(x)` with `sns.histplot(x=x, kde=True, stat="density")` (axes-level) or `sns.displot(x=x, kde=True)` (figure-level).

### Categorical plots

The categorical functions were rewritten in 0.13. Notable changes: `native_scale=True` keeps numeric or datetime category positions; `fill=False` draws line-art boxes/violins; `gap=` adds space between dodged elements; `log_scale=` and `formatter=` are supported; passing `palette` without `hue` is deprecated (assign `hue` to the same variable as `x` and set `legend=False`).

#### stripplot and swarmplot

```python
sns.stripplot(data=None, *, x=None, y=None, hue=None, order=None,
              hue_order=None, jitter=True, dodge=False, orient=None,
              color=None, palette=None, size=5, edgecolor=..., linewidth=0,
              hue_norm=None, log_scale=None, native_scale=False,
              formatter=None, legend="auto", ax=None, **kwargs)
sns.swarmplot(data=None, *, x=None, y=None, hue=None, order=None,
              hue_order=None, dodge=False, orient=None, color=None,
              palette=None, size=5, edgecolor=None, linewidth=0,
              hue_norm=None, log_scale=None, native_scale=False,
              formatter=None, legend="auto", warn_thresh=0.05, ax=None,
              **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `order` | list | None | Order of categories |
| `jitter` | float or bool | True | Random spread along the categorical axis (strip only) |
| `dodge` | bool | False | Separate hue levels along the categorical axis |
| `orient` | `"v"`, `"h"`, `"x"`, `"y"` | None | Orientation (inferred from dtypes) |
| `size` | float | 5 | Marker diameter in points |

Both return `Axes`. `swarmplot` places points without overlap (slow for > a few thousand points).

```python
ax = sns.boxplot(data=tips, x="day", y="total_bill", color="0.9", fliersize=0)
sns.stripplot(data=tips, x="day", y="total_bill", hue="sex", dodge=True,
              alpha=0.6, size=4, ax=ax)
```

#### boxplot

```python
sns.boxplot(data=None, *, x=None, y=None, hue=None, order=None,
            hue_order=None, orient=None, color=None, palette=None,
            saturation=0.75, fill=True, dodge="auto", width=0.8, gap=0,
            whis=1.5, linecolor="auto", linewidth=None, fliersize=None,
            hue_norm=None, native_scale=False, log_scale=None,
            formatter=None, legend="auto", ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `whis` | float or (float, float) | 1.5 | Whisker reach in IQRs, or percentiles |
| `fill` | bool | True | Filled boxes vs outline only (0.13+) |
| `gap` | float | 0 | Space between dodged boxes (0.13+) |
| `width` | float | 0.8 | Width allotted per category |
| `saturation` | float | 0.75 | Desaturate fill colors |
| `fliersize` | float | None | Outlier marker size (0 hides them) |
| `showfliers`, `showmeans`, `notch` (via `**kwargs`) | bool | | Passed to `Axes.boxplot` |

Returns `Axes`.

```python
sns.boxplot(data=titanic, x="class", y="age", hue="alive", gap=0.1,
            showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white"})
```

#### violinplot

```python
sns.violinplot(data=None, *, x=None, y=None, hue=None, order=None,
               hue_order=None, orient=None, color=None, palette=None,
               saturation=0.75, fill=True, inner="box", split=False,
               width=0.8, dodge="auto", gap=0, linewidth=None,
               linecolor="auto", cut=2, gridsize=100, bw_method="scott",
               bw_adjust=1, density_norm="area", common_norm=False,
               hue_norm=None, formatter=None, log_scale=None,
               native_scale=False, legend="auto", inner_kws=None,
               ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `inner` | `"box"`, `"quart"`, `"point"`, `"stick"`, None | `"box"` | Interior representation |
| `split` | bool | False | Draw half violins per hue level (two-level hue) |
| `density_norm` | `"area"`, `"count"`, `"width"` | `"area"` | Width scaling (0.13; replaces `scale`) |
| `common_norm` | bool | False | Normalize across all violins (replaces `scale_hue`) |
| `bw_method`, `bw_adjust` | | `"scott"`, 1 | KDE bandwidth (replaces `bw`) |
| `cut` | float | 2 | Extend beyond data by `cut * bw` (0 limits to data range) |
| `inner_kws` | dict | None | Style the inner marks |

Returns `Axes`.

```python
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex",
               split=True, inner="quart", cut=0, density_norm="count")
```

#### boxenplot

```python
sns.boxenplot(data=None, *, x=None, y=None, hue=None, order=None,
              hue_order=None, orient=None, color=None, palette=None,
              saturation=0.75, fill=True, dodge="auto", width=0.8, gap=0,
              linewidth=None, linecolor=None, width_method="exponential",
              k_depth="tukey", outlier_prop=0.007, trust_alpha=0.05,
              showfliers=True, hue_norm=None, log_scale=None,
              native_scale=False, formatter=None, legend="auto",
              box_kws=None, flier_kws=None, line_kws=None, ax=None, **kwargs)
```

Letter-value plot: shows more quantiles than a boxplot; good for large datasets. `k_depth` in `"tukey"`, `"proportion"`, `"trustworthy"`, `"full"`, or int. Returns `Axes`. (Formerly `lvplot`, renamed in 0.9.)

```python
diamonds = sns.load_dataset("diamonds")
sns.boxenplot(data=diamonds, x="clarity", y="carat", k_depth="trustworthy")
```

#### barplot

```python
sns.barplot(data=None, *, x=None, y=None, hue=None, order=None,
            hue_order=None, estimator="mean", errorbar=("ci", 95),
            n_boot=1000, seed=None, units=None, orient=None,
            color=None, palette=None, saturation=0.75, fill=True,
            hue_norm=None, width=0.8, dodge="auto", gap=0, log_scale=None,
            native_scale=False, formatter=None, legend="auto",
            capsize=0, err_kws=None, ci="deprecated", errcolor="deprecated",
            errwidth="deprecated", ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `estimator` | str or callable | `"mean"` | Statistic per bar (`"median"`, `"sum"`, `np.std`, `len`, ...) |
| `errorbar` | | `("ci", 95)` | Error bar spec; `None` to disable |
| `capsize` | float | 0 | Error bar cap width as fraction of bar width |
| `err_kws` | dict | None | Style error bars (replaces `errcolor`/`errwidth`) |

Returns `Axes`. Note that a bar plot of means hides the distribution; consider `pointplot`, `boxplot`, or `stripplot` too.

```python
ax = sns.barplot(data=titanic, x="class", y="survived", hue="sex",
                 errorbar=("ci", 95), capsize=0.1)
ax.bar_label(ax.containers[0], fmt="%.2f")
```

#### countplot

```python
sns.countplot(data=None, *, x=None, y=None, hue=None, order=None,
              hue_order=None, orient=None, color=None, palette=None,
              saturation=0.75, fill=True, hue_norm=None, stat="count",
              width=0.8, dodge="auto", gap=0, log_scale=None,
              native_scale=False, formatter=None, legend="auto", ax=None,
              **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stat` | str | `"count"` | `"count"`, `"percent"`, `"proportion"`, `"probability"` (0.13+) |

Returns `Axes`. Ideal for class balance.

```python
ax = sns.countplot(data=titanic, x="class", hue="survived", stat="percent")
```

#### pointplot

```python
sns.pointplot(data=None, *, x=None, y=None, hue=None, order=None,
              hue_order=None, estimator="mean", errorbar=("ci", 95),
              n_boot=1000, seed=None, units=None, color=None,
              palette=None, hue_norm=None, markers=..., linestyles=...,
              dodge=False, log_scale=None, native_scale=False, orient=None,
              capsize=0, formatter=None, legend="auto", err_kws=None,
              ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `markers`, `linestyles` | str or list | default | Per hue level |
| `dodge` | bool or float | False | Offset hue levels to avoid overlap |
| `linestyle` (kwarg) | str | | `"none"` to drop connecting lines (replaces `join=False`) |

Returns `Axes`. Good for interaction plots (one factor on x, another on hue).

```python
sns.pointplot(data=tips, x="day", y="tip", hue="sex", dodge=0.3,
              markers=["o", "s"], capsize=0.1)
```

#### catplot

```python
sns.catplot(data=None, *, x=None, y=None, hue=None, row=None, col=None,
            kind="strip", estimator="mean", errorbar=("ci", 95), n_boot=1000,
            seed=None, units=None, order=None, hue_order=None,
            row_order=None, col_order=None, col_wrap=None, height=5,
            aspect=1, log_scale=None, native_scale=False, formatter=None,
            orient=None, color=None, palette=None, hue_norm=None,
            legend="auto", legend_out=True, sharex=True, sharey=True,
            margin_titles=False, facet_kws=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `kind` | str | `"strip"` | `"strip"`, `"swarm"`, `"box"`, `"violin"`, `"boxen"`, `"point"`, `"bar"`, `"count"` |
| `sharex`, `sharey` | bool or `"col"`, `"row"` | True | Share axes across facets |
| `legend_out` | bool | True | Draw legend outside the grid |

Returns `FacetGrid`. (Called `factorplot` before 0.9.)

```python
g = sns.catplot(data=titanic, x="class", y="survived", hue="sex",
                col="embark_town", kind="bar", height=3.5, aspect=0.9,
                errorbar=None)
g.set_axis_labels("", "Survival rate")
g.set_titles("{col_name}")
```

### Regression plots

#### regplot

```python
sns.regplot(data=None, *, x=None, y=None, x_estimator=None, x_bins=None,
            x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
            units=None, seed=None, order=1, logistic=False, lowess=False,
            robust=False, logx=False, x_partial=None, y_partial=None,
            truncate=True, dropna=True, x_jitter=None, y_jitter=None,
            label=None, color=None, marker="o", scatter_kws=None,
            line_kws=None, ax=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ci` | int or None | 95 | Confidence band for the regression line (`None` disables, faster) |
| `order` | int | 1 | Polynomial order |
| `logistic` | bool | False | Logistic regression for binary `y` (needs statsmodels) |
| `lowess` | bool | False | Nonparametric lowess smoother (needs statsmodels; no CI) |
| `robust` | bool | False | Robust regression to downweight outliers (needs statsmodels) |
| `logx` | bool | False | Fit `y ~ log(x)` |
| `x_estimator` | callable | None | Aggregate y per unique x (e.g. `np.mean`) |
| `x_bins` | int or array | None | Bin x before aggregating |
| `x_jitter`, `y_jitter` | float | None | Visual jitter only (fit uses original data) |
| `truncate` | bool | True | Limit line to data range |
| `scatter_kws`, `line_kws` | dict | None | Style the points / line |

Returns `Axes`.

```python
sns.regplot(data=tips, x="total_bill", y="tip", ci=95,
            scatter_kws={"alpha": 0.4, "s": 20}, line_kws={"color": "crimson"})
sns.regplot(data=titanic, x="age", y="survived", logistic=True, y_jitter=0.03)
```

#### lmplot

```python
sns.lmplot(data, *, x=None, y=None, hue=None, col=None, row=None,
           palette=None, col_wrap=None, height=5, aspect=1, markers="o",
           sharex=None, sharey=None, hue_order=None, col_order=None,
           row_order=None, legend=True, legend_out=None, x_estimator=None,
           x_bins=None, x_ci="ci", scatter=True, fit_reg=True, ci=95,
           n_boot=1000, units=None, seed=None, order=1, logistic=False,
           lowess=False, robust=False, logx=False, x_partial=None,
           y_partial=None, truncate=True, x_jitter=None, y_jitter=None,
           scatter_kws=None, line_kws=None, facet_kws=None)
```

Figure-level `regplot` with `hue`/`col`/`row` faceting. Returns `FacetGrid`.

```python
g = sns.lmplot(data=penguins, x="bill_length_mm", y="bill_depth_mm",
               hue="species", height=5, aspect=1.2)
g.set_axis_labels("Bill length (mm)", "Bill depth (mm)")
```

This plot illustrates Simpson's paradox: the overall correlation is negative while the within-species correlations are positive.

#### residplot

```python
sns.residplot(data=None, *, x=None, y=None, x_partial=None, y_partial=None,
              lowess=False, order=1, robust=False, dropna=True, label=None,
              color=None, scatter_kws=None, line_kws=None, ax=None)
```

Fits a regression of `y` on `x` and plots the residuals. `lowess=True` adds a smoothed trend to reveal non-linearity. Returns `Axes`.

```python
sns.residplot(data=tips, x="total_bill", y="tip", lowess=True,
              line_kws={"color": "red"})
```

### Matrix plots

#### heatmap

```python
sns.heatmap(data, *, vmin=None, vmax=None, cmap=None, center=None,
            robust=False, annot=None, fmt=".2g", annot_kws=None,
            linewidths=0, linecolor="white", cbar=True, cbar_kws=None,
            cbar_ax=None, square=False, xticklabels="auto",
            yticklabels="auto", mask=None, ax=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | 2D array or DataFrame | required | Rectangular numeric data; DataFrame index/columns become labels |
| `vmin`, `vmax` | float | None | Color limits |
| `cmap` | str or Colormap | None | Colormap (`"rocket"` sequential, `"icefire"` when `center` is set) |
| `center` | float | None | Value at the colormap center (diverging data) |
| `robust` | bool | False | Use 2nd/98th percentiles for limits |
| `annot` | bool or array | None | Write values in cells |
| `fmt` | str | `".2g"` | Format for annotations, e.g. `"d"`, `".1%"` |
| `linewidths` | float | 0 | Cell divider width |
| `square` | bool | False | Square cells |
| `mask` | bool array | None | True cells are not drawn |
| `cbar_kws` | dict | None | Passed to `Figure.colorbar` (e.g. `{"label": "r"}`) |
| `xticklabels`, `yticklabels` | `"auto"`, bool, int, list | `"auto"` | Label control; int N shows every Nth label |

Returns `Axes`.

```python
corr = penguins.select_dtypes("number").corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"}, ax=ax)

pivot = flights.pivot(index="month", columns="year", values="passengers")
sns.heatmap(pivot, cmap="mako", annot=True, fmt="d", annot_kws={"size": 7})
```

#### clustermap

```python
sns.clustermap(data, *, pivot_kws=None, method="average",
               metric="euclidean", z_score=None, standard_scale=None,
               figsize=(10, 10), cbar_kws=None, row_cluster=True,
               col_cluster=True, row_linkage=None, col_linkage=None,
               row_colors=None, col_colors=None, mask=None,
               dendrogram_ratio=0.2, colors_ratio=0.03,
               cbar_pos=(0.02, 0.8, 0.05, 0.18), tree_kws=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | str | `"average"` | Linkage method (`"single"`, `"complete"`, `"ward"`, ...) |
| `metric` | str | `"euclidean"` | Distance metric (`"correlation"`, `"cosine"`, ...) |
| `z_score` | 0, 1, or None | None | Standardize rows (0) or columns (1) |
| `standard_scale` | 0, 1, or None | None | Min-max scale rows or columns |
| `row_cluster`, `col_cluster` | bool | True | Cluster rows / columns |
| `row_colors`, `col_colors` | list, Series, DataFrame | None | Annotation color strips |
| `**kwargs` | | | Passed to `heatmap` (`cmap`, `center`, `annot`, ...) |

Returns a `ClusterGrid` with attributes `.ax_heatmap`, `.ax_row_dendrogram`, `.ax_col_dendrogram`, `.dendrogram_row.reordered_ind`, `.figure`. Requires SciPy.

```python
iris = sns.load_dataset("iris")
species = iris.pop("species")
lut = dict(zip(species.unique(), sns.color_palette("Set2", 3)))
g = sns.clustermap(iris, z_score=1, cmap="vlag", center=0,
                   row_colors=species.map(lut), figsize=(6, 8),
                   col_cluster=False)
order = g.dendrogram_row.reordered_ind
```

### Multi-plot grids

#### pairplot

```python
sns.pairplot(data, *, hue=None, hue_order=None, palette=None, vars=None,
             x_vars=None, y_vars=None, kind="scatter", diag_kind="auto",
             markers=None, height=2.5, aspect=1, corner=False, dropna=False,
             plot_kws=None, diag_kws=None, grid_kws=None, size=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vars` | list | None | Columns to include (default: all numeric) |
| `x_vars`, `y_vars` | list | None | Non-square grids |
| `kind` | str | `"scatter"` | `"scatter"`, `"kde"`, `"hist"`, `"reg"` |
| `diag_kind` | str | `"auto"` | `"hist"`, `"kde"`, or None |
| `corner` | bool | False | Only lower triangle |
| `plot_kws`, `diag_kws` | dict | None | Passed to off-diagonal / diagonal functions |
| `height` | float | 2.5 | Size of each facet |

Returns `PairGrid`. The deprecated `size=` parameter was replaced by `height=` in 0.9.

```python
g = sns.pairplot(penguins, hue="species", corner=True, diag_kind="kde",
                 plot_kws={"alpha": 0.6, "s": 15}, height=2.2)
```

#### jointplot

```python
sns.jointplot(data=None, *, x=None, y=None, hue=None, kind="scatter",
              height=6, ratio=5, space=0.2, dropna=False, xlim=None,
              ylim=None, color=None, palette=None, hue_order=None,
              hue_norm=None, marginal_ticks=False, joint_kws=None,
              marginal_kws=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `kind` | str | `"scatter"` | `"scatter"`, `"kde"`, `"hist"`, `"hex"`, `"reg"`, `"resid"` |
| `ratio` | int | 5 | Joint-to-marginal height ratio |
| `space` | float | 0.2 | Space between joint and marginal axes |
| `joint_kws`, `marginal_kws` | dict | None | Styling for each part |

Returns `JointGrid` (attributes `.ax_joint`, `.ax_marg_x`, `.ax_marg_y`).

```python
g = sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm",
                  hue="species", kind="kde", height=6)
g.ax_joint.set_xlabel("Bill length (mm)")
```

#### FacetGrid

```python
sns.FacetGrid(data, *, row=None, col=None, hue=None, col_wrap=None,
              sharex=True, sharey=True, height=3, aspect=1, palette=None,
              row_order=None, col_order=None, hue_order=None, hue_kws=None,
              dropna=False, legend_out=True, despine=True,
              margin_titles=False, xlim=None, ylim=None, subplot_kws=None,
              gridspec_kws=None)
```

| Method | Description |
|---|---|
| `map(func, *args, **kwargs)` | Call `func` on each facet with column *values* (Matplotlib-style functions) |
| `map_dataframe(func, *args, **kwargs)` | Call `func(data=subset, ...)` with column *names* (seaborn-style functions) |
| `set_axis_labels(x_var, y_var)` | Label outer axes |
| `set_titles(template, row_template, col_template)` | Facet titles, e.g. `"{col_name}"` |
| `add_legend()` | Add a figure legend |
| `refline(x=None, y=None, **kws)` | Reference lines on every facet |
| `set(**kwargs)` | `Axes.set` on every facet |
| `tick_params(axis="both", **kwargs)` | Tick styling on every facet |
| `savefig(*args, **kwargs)` | Save with `bbox_inches="tight"` by default |
| `.figure`, `.axes`, `.axes_dict`, `.ax` | Underlying Matplotlib objects |

```python
g = sns.FacetGrid(tips, col="time", row="smoker", margin_titles=True, height=3)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", hue="sex")
g.refline(y=tips["tip"].median(), color="gray", ls="--")
g.add_legend()
g.set_axis_labels("Total bill ($)", "Tip ($)")
for (row, col), ax in g.axes_dict.items():
    ax.set_title(f"{row} / {col}", fontsize=9)
```

#### PairGrid and JointGrid

```python
sns.PairGrid(data, *, hue=None, vars=None, x_vars=None, y_vars=None,
             hue_order=None, palette=None, hue_kws=None, corner=False,
             diag_sharey=True, height=2.5, aspect=1, layout_pad=0.5,
             despine=True, dropna=False)
sns.JointGrid(data=None, *, x=None, y=None, hue=None, height=6, ratio=5,
              space=0.2, palette=None, hue_order=None, hue_norm=None,
              dropna=False, xlim=None, ylim=None, marginal_ticks=False)
```

`PairGrid` methods: `map`, `map_diag`, `map_offdiag`, `map_lower`, `map_upper`, `add_legend`. `JointGrid` methods: `plot(joint_func, marginal_func)`, `plot_joint`, `plot_marginals`, `refline`, `set_axis_labels`.

```python
g = sns.PairGrid(penguins, hue="species", vars=["bill_length_mm", "bill_depth_mm", "body_mass_g"])
g.map_upper(sns.scatterplot, s=10)
g.map_lower(sns.kdeplot, levels=4)
g.map_diag(sns.histplot, element="step")
g.add_legend()

j = sns.JointGrid(data=tips, x="total_bill", y="tip")
j.plot(sns.regplot, sns.boxplot)
j.refline(y=tips["tip"].mean())
```

### Styling, themes, and utilities

#### set_theme, set_style, set_context

```python
sns.set_theme(context="notebook", style="darkgrid", palette="deep",
              font="sans-serif", font_scale=1, color_codes=True, rc=None)
sns.set_style(style=None, rc=None)
sns.set_context(context=None, font_scale=1, rc=None)
sns.axes_style(style=None, rc=None)          # dict or context manager
sns.plotting_context(context=None, font_scale=1, rc=None)
sns.set_palette(palette, n_colors=None, desat=None, color_codes=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `context` | str or dict | `"notebook"` | `"paper"`, `"notebook"`, `"talk"`, `"poster"` |
| `style` | str or dict | `"darkgrid"` | `"darkgrid"`, `"whitegrid"`, `"dark"`, `"white"`, `"ticks"` |
| `palette` | str or sequence | `"deep"` | Color cycle |
| `font_scale` | float | 1 | Font size multiplier |
| `rc` | dict | None | Extra rcParams overrides |

```python
sns.set_theme(style="ticks", context="paper", rc={"axes.spines.right": False,
                                                  "axes.spines.top": False})
```

#### color_palette and palette builders

```python
sns.color_palette(palette=None, n_colors=None, desat=None, as_cmap=False)
sns.light_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb")
sns.dark_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb")
sns.diverging_palette(h_neg, h_pos, s=75, l=50, sep=1, n=6, center="light", as_cmap=False)
sns.cubehelix_palette(n_colors=6, start=0, rot=0.4, gamma=1.0, hue=0.8,
                      light=0.85, dark=0.15, reverse=False, as_cmap=False)
sns.husl_palette(n_colors=6, h=0.01, s=0.9, l=0.65, as_cmap=False)
sns.hls_palette(n_colors=6, h=0.01, l=0.6, s=0.65, as_cmap=False)
sns.blend_palette(colors, n_colors=6, as_cmap=False, input="rgb")
sns.mpl_palette(name, n_colors=6, as_cmap=False)
sns.palplot(pal, size=1)
```

`color_palette` returns a `_ColorPalette` (a list of RGB tuples with `.as_hex()`), or a Matplotlib `Colormap` with `as_cmap=True`. It also works as a context manager.

```python
cmap = sns.diverging_palette(250, 15, as_cmap=True)
with sns.color_palette("colorblind"):
    sns.lineplot(data=fmri, x="timepoint", y="signal", hue="region")
```

#### despine and move_legend

```python
sns.despine(fig=None, ax=None, top=True, right=True, left=False,
            bottom=False, offset=None, trim=False)
sns.move_legend(obj, loc, **kwargs)
```

`move_legend` works on an `Axes`, `Figure`, or grid; it recreates the legend so you can change `loc`, `bbox_to_anchor`, `ncol`, `title`, and `frameon`.

```python
ax = sns.histplot(data=penguins, x="bill_length_mm", hue="species")
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3,
                title=None, frameon=False)
sns.despine(ax=ax, offset=5, trim=True)
```

#### Datasets

```python
sns.load_dataset(name, cache=True, data_home=None, **kws)   # -> DataFrame
sns.get_dataset_names()                                     # -> list[str]
sns.get_data_home(data_home=None)                           # cache directory
```

Common datasets: `tips`, `penguins`, `iris`, `titanic`, `fmri`, `flights`, `diamonds`, `mpg`, `planets`, `anscombe`, `exercise`, `car_crashes`, `geyser`, `healthexp`, `taxis`, `dots`. Extra keyword arguments go to `pandas.read_csv`.

### seaborn.objects reference

```python
import seaborn.objects as so

so.Plot(data=None, x=None, y=None, color=None, alpha=None, marker=None,
        pointsize=None, linewidth=None, linestyle=None, fill=None,
        text=None, group=None, ...)
Plot.add(mark, *transforms, orient=None, legend=True, label=None,
         data=None, **variables)
Plot.facet(col=None, row=None, order=None, wrap=None)
Plot.pair(x=None, y=None, wrap=None, cross=True)
Plot.scale(**scales)                 # e.g. x="log", color="flare", y=so.Continuous().tick(...)
Plot.share(**shares)                 # e.g. x=False, y="row"
Plot.limit(**limits)                 # e.g. x=(0, 10)
Plot.label(*, title=None, legend=None, **variables)
Plot.layout(*, size=None, engine=None, extent=None)
Plot.theme(config)                   # dict of rcParams, e.g. sns.axes_style("ticks")
Plot.on(target)                      # draw on an existing Axes, Figure, or SubFigure
Plot.save(loc, **kwargs)
Plot.show(**kwargs)
Plot.plot(pyplot=False)              # compile; returns a Plotter
```

| Component | Classes |
|---|---|
| Marks | `Dot`, `Dots`, `Line`, `Lines`, `Path`, `Paths`, `Dash`, `Range`, `Area`, `Band`, `Bar`, `Bars`, `Text` |
| Stats | `Agg(func="mean")`, `Est(func="mean", errorbar=("ci", 95))`, `Count()`, `Hist(stat="count", bins="auto", ...)`, `KDE(bw_adjust=1, ...)` (0.13), `Perc(k=5)`, `PolyFit(order=2)` |
| Moves | `Dodge(gap=0)`, `Jitter(width=0, x=0, y=0)`, `Stack()`, `Shift(x=0, y=0)`, `Norm(func="max")` |
| Scales | `Continuous`, `Nominal`, `Temporal`, `Boolean` (0.13) |

```python
import seaborn.objects as so

(
    so.Plot(tips, x="day", y="total_bill", color="sex")
    .add(so.Bar(), so.Agg(), so.Dodge())
    .add(so.Range(), so.Est(errorbar="se"), so.Dodge())
    .scale(color="Set2")
    .label(y="Mean bill ($)")
)

(
    so.Plot(penguins, x="flipper_length_mm", color="species")
    .add(so.Area(), so.KDE(common_norm=False))
    .add(so.Bars(alpha=0.3), so.Hist(stat="density", common_norm=False))
)

(
    so.Plot(tips, x="total_bill", y="tip")
    .add(so.Dots(), so.Jitter(0.2))
    .scale(x=so.Continuous(trans="log"))
)
```

Use `Plot.on(ax)` to embed an objects plot in a Matplotlib layout:

```python
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
so.Plot(tips, x="total_bill").add(so.Bars(), so.Hist()).on(axs[0]).plot()
so.Plot(tips, x="total_bill", y="tip").add(so.Dot()).on(axs[1]).plot()
fig.savefig("objects_on_axes.png")
```

## Tutorials

### Tutorial 1: Exploratory data analysis of a classification dataset

Goal: understand the Palmer penguins dataset before training a species classifier.

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

# 1. Load and inspect
penguins = sns.load_dataset("penguins")
print(penguins.shape)
print(penguins.isna().sum())
penguins = penguins.dropna()

# 2. Class balance: is the target imbalanced?
fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
sns.countplot(data=penguins, x="species", hue="species", stat="percent",
              legend=False, ax=axs[0])
axs[0].set_title("Class balance (%)")
for c in axs[0].containers:
    axs[0].bar_label(c, fmt="%.1f")

# 3. Does island leak the label? Cross-tab as a heatmap
ct = penguins.groupby(["island", "species"]).size().unstack(fill_value=0)
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[1])
axs[1].set_title("Island vs species counts")
fig.savefig("eda_balance.png", dpi=120)

# 4. Feature distributions per class
num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
long = penguins.melt(id_vars="species", value_vars=num_cols,
                     var_name="feature", value_name="value")
g = sns.displot(data=long, x="value", hue="species", col="feature",
                col_wrap=2, kind="kde", fill=True, common_norm=False,
                facet_kws={"sharex": False, "sharey": False},
                height=3, aspect=1.4)
g.set_titles("{col_name}")
g.savefig("eda_distributions.png", dpi=120)

# 5. Pairwise relationships
g = sns.pairplot(penguins, vars=num_cols, hue="species", corner=True,
                 diag_kind="kde", plot_kws={"s": 12, "alpha": 0.6}, height=2.2)
g.savefig("eda_pairs.png", dpi=120)

# 6. Correlation matrix (lower triangle)
corr = penguins[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title("Feature correlations")
fig.savefig("eda_corr.png", dpi=120)
plt.close("all")
```

What each step tells you: step 2 shows Chinstrap is the minority class (consider stratified splits); step 3 shows `island` almost determines species for some islands (a potential shortcut feature); steps 4-5 show bill length and flipper length separate the classes well; step 6 flags the strong flipper/body-mass correlation (relevant for linear models).

### Tutorial 2: Comparing models across cross-validation folds

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Run cross-validation for several models and several metrics
X, y = load_breast_cancer(return_X_y=True)
models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0),
    "GradBoost": GradientBoostingClassifier(random_state=0),
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
rows = []
for name, model in models.items():
    res = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "roc_auc", "f1"])
    for metric in ["accuracy", "roc_auc", "f1"]:
        for fold, score in enumerate(res[f"test_{metric}"]):
            rows.append({"model": name, "metric": metric, "fold": fold, "score": score})

# 2. Tidy results: one row per (model, metric, fold)
results = pd.DataFrame(rows)
print(results.groupby(["model", "metric"])["score"].agg(["mean", "std"]).round(4))

# 3. Box + strip per metric facet
sns.set_theme(style="ticks")
g = sns.catplot(data=results, x="model", y="score", col="metric", kind="box",
                hue="model", legend=False, fill=False, height=3.5, aspect=1,
                sharey=False)
g.map_dataframe(sns.stripplot, x="model", y="score", color="k", size=3, alpha=0.6)
g.set_titles("{col_name}")
g.set_axis_labels("", "CV score")
g.savefig("cv_comparison.png", dpi=120)

# 4. Paired view: fold-by-fold lines show whether one model wins consistently
acc = results.query("metric == 'roc_auc'")
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
sns.lineplot(data=acc, x="model", y="score", units="fold", estimator=None,
             color="gray", alpha=0.4, lw=1, ax=ax)
sns.pointplot(data=acc, x="model", y="score", color="crimson",
              errorbar=("ci", 95), capsize=0.1, ax=ax)
ax.set_title("ROC AUC per fold (gray) and mean with 95% CI (red)")
fig.savefig("cv_paired.png", dpi=120)
plt.close("all")
```

The key move is reshaping results into long form (step 2): then a single `catplot` call facets by metric, and `units="fold"` with `estimator=None` draws per-fold lines for a paired comparison.

### Tutorial 3: Time series and grouped trends with uncertainty

```python
import matplotlib.pyplot as plt
import seaborn as sns

flights = sns.load_dataset("flights")       # year, month, passengers

sns.set_theme(style="darkgrid")

# 1. Yearly trend with month-to-month variability as the error band
fig, axs = plt.subplots(2, 1, figsize=(9, 8), layout="constrained")
sns.lineplot(data=flights, x="year", y="passengers", errorbar=("pi", 100),
             marker="o", ax=axs[0])
axs[0].set_title("Mean monthly passengers per year (band = min..max month)")

# 2. One line per month, colored by a sequential palette
sns.lineplot(data=flights, x="year", y="passengers", hue="month",
             palette="flare", ax=axs[1])
sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1), ncol=1, title="month")
axs[1].set_title("Seasonality by month")
fig.savefig("flights_lines.png", dpi=120)

# 3. Seasonal structure as a heatmap
pivot = flights.pivot(index="month", columns="year", values="passengers")
fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
sns.heatmap(pivot, cmap="rocket_r", annot=True, fmt="d",
            annot_kws={"size": 7}, linewidths=0.3, ax=ax)
ax.set_title("Passengers (thousands)")
fig.savefig("flights_heatmap.png", dpi=120)

# 4. Experimental data: fMRI signal per region with SE bands, faceted by event
fmri = sns.load_dataset("fmri")
g = sns.relplot(data=fmri, kind="line", x="timepoint", y="signal",
                hue="region", col="event", errorbar="se", height=3.5, aspect=1.3)
g.refline(y=0, color="k", lw=0.8)
g.savefig("fmri.png", dpi=120)
plt.close("all")
```

`errorbar=("pi", 100)` shows the full range of the data (descriptive), whereas `errorbar="se"` shows the uncertainty of the mean (inferential). Choose deliberately and say which in the caption.

### Tutorial 4: Regression diagnostics and residual analysis

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# 1. Fit a model (fetch_california_housing downloads ~400 KB once)
data = fetch_california_housing(as_frame=True)
df = data.frame
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="MedHouseVal"),
                                          df["MedHouseVal"], random_state=0)
model = HistGradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
pred = model.predict(X_te)

res = X_te.copy()
res["actual"] = y_te.values
res["predicted"] = pred
res["residual"] = res["actual"] - res["predicted"]
res["income_band"] = pd.qcut(res["MedInc"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

sns.set_theme(style="whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")

# 2. Predicted vs actual (dense data -> 2D histogram)
sns.histplot(data=res, x="actual", y="predicted", bins=60, cmap="mako",
             cbar=True, ax=axs[0, 0])
axs[0, 0].axline((0, 0), slope=1, color="red", lw=1)
axs[0, 0].set_title("Predicted vs actual")

# 3. Residual distribution
sns.histplot(data=res, x="residual", kde=True, stat="density", ax=axs[0, 1])
axs[0, 1].axvline(0, color="k", lw=1)
axs[0, 1].set_title("Residual distribution")

# 4. Residuals vs a feature, with a lowess trend (requires statsmodels)
sample = res.sample(2000, random_state=0)
sns.regplot(data=sample, x="MedInc", y="residual", lowess=True,
            scatter_kws={"s": 6, "alpha": 0.3}, line_kws={"color": "red"},
            ax=axs[1, 0])
axs[1, 0].axhline(0, color="k", lw=0.8)
axs[1, 0].set_title("Residuals vs median income")

# 5. Error by subgroup (fairness / slice analysis)
res["abs_error"] = res["residual"].abs()
sns.boxplot(data=res, x="income_band", y="abs_error", hue="income_band",
            legend=False, fliersize=1, ax=axs[1, 1])
axs[1, 1].set_title("Absolute error by income quartile")

fig.savefig("regression_diagnostics.png", dpi=120)
plt.close(fig)
```

Patterns to look for: a curved lowess line in step 4 indicates missing non-linear structure, a skewed residual histogram suggests transforming the target, and uneven boxes in step 5 reveal slices where the model underperforms.

### Tutorial 5: The same analysis in seaborn.objects

```python
import seaborn as sns
import seaborn.objects as so

tips = sns.load_dataset("tips")

# Base specification reused by two plots (Plot objects are immutable)
base = so.Plot(tips, x="total_bill", y="tip", color="time")

p1 = (
    base
    .add(so.Dots(alpha=0.5))
    .add(so.Line(linewidth=2), so.PolyFit(order=1))
    .facet(col="smoker")
    .label(x="Total bill ($)", y="Tip ($)", col="Smoker:")
    .layout(size=(9, 4))
)
p1.save("tips_objects.png", dpi=120)

# Tip rate by day with mean and 95% CI, dodged by sex
tips["tip_rate"] = tips["tip"] / tips["total_bill"]
p2 = (
    so.Plot(tips, x="day", y="tip_rate", color="sex")
    .add(so.Dot(pointsize=3, alpha=0.3), so.Jitter(0.3), so.Dodge())
    .add(so.Dot(pointsize=8), so.Agg("mean"), so.Dodge())
    .add(so.Range(linewidth=2), so.Est(errorbar=("ci", 95)), so.Dodge())
    .scale(y=so.Continuous().label(like="{x:.0%}"))
    .label(y="Tip rate")
)
p2.save("tip_rate_objects.png", dpi=120)
```

Each `.add` is one layer: mark plus transforms. Stats (`Agg`, `Est`, `PolyFit`) compute the data; moves (`Jitter`, `Dodge`) adjust positions. Scales accept formatting through `.label(like=...)`.

## Performance & Best Practices

### Speed

| Situation | What to do |
|---|---|
| Slow `lineplot`/`barplot`/`pointplot` on large data | Bootstrapping dominates: set `errorbar=None` or `errorbar="se"` (analytic), or lower `n_boot` |
| Slow `regplot`/`lmplot` | `ci=None` skips the bootstrap band; sample the scatter points |
| `swarmplot` very slow or warns about unplaced points | Use `stripplot` with jitter, `violinplot`, or `boxenplot` for more than a few thousand points |
| `pairplot` on many columns | Select `vars=`, sample rows, use `corner=True`, `kind="hist"` |
| `scatterplot` with 100k+ points | `histplot(x=..., y=...)` bivariate, `kdeplot`, or Matplotlib `hexbin`; `alpha` and small `s` |
| Huge heatmaps with `annot=True` | Annotation text is the bottleneck; disable `annot` beyond about 30x30 |
| `clustermap` on many rows | Precompute linkage with SciPy (`row_linkage=`) and reuse it |

### Correctness and clarity

- Know whether your error bars are **descriptive** (`sd`, `pi`) or **inferential** (`ci`, `se`), and state it in captions.
- Use `common_norm=False` when comparing distribution *shapes* of groups with different sizes; keep `True` when relative sizes matter.
- For bounded variables (prices, counts), set `clip=(0, None)` or `cut=0` on KDEs so density does not leak below zero.
- Prefer `histplot`/`ecdfplot` over KDE for small samples or discrete data (`discrete=True`).
- Fix category order with `order=`/`hue_order=` so colors and positions are stable across plots.
- Pass explicit dict palettes for categories that recur across a report (e.g. model names).
- Use `palette="colorblind"` for accessibility.

### Code organization

- Use axes-level functions with `ax=` when composing complex multi-panel figures; use figure-level functions for faceting.
- Size figure-level plots with `height`/`aspect`, not `plt.figure(figsize=...)` (which they ignore).
- Do not call `plt.figure()` before a figure-level function: it creates an extra empty figure.
- Set themes once at the top of a notebook or script (`sns.set_theme(...)`), or scope them with `with sns.axes_style(...)`.
- Reshape to long form with `DataFrame.melt` rather than looping over columns.
- Save grids with `g.savefig(...)` (tight bounding box by default) or `g.figure.savefig(...)`.

## Common Errors & Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `TypeError: scatterplot() takes from 0 to 1 positional arguments but 3 were given` | Since 0.12, data variables must be keywords | `sns.scatterplot(data=df, x="a", y="b")` or `sns.scatterplot(x=a, y=b)` |
| ``ValueError: Could not interpret value `colname` for `x`. An entry with this name does not appear in `data`.`` | Typo in column name, or column is the index | Check `df.columns`; `df.reset_index()` to use the index |
| `UserWarning: distplot is a deprecated function and will be removed...` | `distplot` deprecated since 0.11 | `histplot(..., kde=True)` or `displot(...)` |
| `AttributeError: module 'seaborn' has no attribute 'histplot'` | seaborn older than 0.11 | `pip install -U seaborn` |
| `AttributeError: module 'seaborn' has no attribute 'factorplot'` / `'tsplot'` / `'lvplot'` | Removed/renamed functions | `catplot`, `lineplot`, `boxenplot` |
| `FutureWarning: Passing palette without assigning hue is deprecated...` | 0.13 categorical change | Add `hue=` equal to the categorical variable and `legend=False` |
| `FutureWarning: The ci parameter is deprecated. Use errorbar=...` | 0.12 change | `errorbar=("ci", 95)`, `errorbar="sd"`, or `errorbar=None` |
| `FutureWarning: shade is now deprecated in favor of fill` | 0.11 change in `kdeplot` | `fill=True` |
| `FutureWarning` about `scale`/`scale_hue`/`bw` in `violinplot` | 0.13 renames | `density_norm=`, `common_norm=`, `bw_method=`/`bw_adjust=` |
| `FutureWarning` about `join` in `pointplot` | 0.13 change | `linestyle="none"` |
| `UserWarning: X% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.` | Too many points for `swarmplot` | Smaller `size=`, or `stripplot` |
| `UserWarning: Dataset has 0 variance; skipping density estimate.` | Constant group in `kdeplot`/`violinplot` | Filter constant groups or use `histplot` |
| `URLError: <urlopen error [Errno 11001] getaddrinfo failed>` from `load_dataset` | No internet / firewall | Download CSVs from the seaborn-data repo and use `pd.read_csv` |
| `TypeError: ufunc 'isnan' not supported for the input types` in `heatmap` | Non-numeric columns in data | `df.select_dtypes("number")`, or pivot first |
| `ValueError: Unknown format code 'd' for object of type 'float'` | `fmt="d"` with float values | `fmt=".0f"`, or cast to int |
| `ModuleNotFoundError: No module named 'scipy'` / `'statsmodels'` | `clustermap`, `lowess`, `robust`, `logistic` need extras | `pip install "seaborn[stats]"` |
| `figsize` ignored, plot is the wrong size | Figure-level functions create their own figure | Use `height=` and `aspect=` |
| Extra blank figure appears | `plt.figure()` called before a figure-level function | Remove it, or use the axes-level equivalent with `ax=` |
| `AttributeError: 'FacetGrid' object has no attribute 'set_title'` | Grids are not Axes | `g.set_titles(...)`, `g.figure.suptitle(...)`, or loop `g.axes.flat` |
| `AttributeError: 'Axes' object has no attribute 'set_axis_labels'` | Axes-level function returns an Axes | `ax.set(xlabel=..., ylabel=...)` |
| Legend overlaps data | Default placement | `sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))` |
| Suptitle overlaps facet titles | Grid layout leaves no room | `g.figure.suptitle("t", y=1.03)` or `g.figure.subplots_adjust(top=0.9)` |
| Categories in unexpected order | Order inferred from data | `order=[...]`, or use a pandas `Categorical` dtype |
| Numeric x treated as categories in `boxplot`/`barplot` | Categorical axis by design | `native_scale=True` (0.13+) |

## Interoperability

### Matplotlib

Every seaborn plot is Matplotlib. Axes-level functions return `Axes`; grids expose `.figure`, `.axes`, and `.ax`. Pass Matplotlib keyword arguments straight through (`linewidth`, `edgecolor`, `alpha`, `marker`), and use any Matplotlib method afterward (`ax.set_xscale("log")`, `ax.annotate(...)`). Seaborn themes are just rcParams, so they also affect pure Matplotlib plots.

### pandas and Polars

Seaborn's native input is the pandas DataFrame. Column names become axis and legend labels, categorical dtypes control order, and datetime columns render as dates. Since 0.13, objects supporting the DataFrame interchange protocol (Polars, PyArrow tables) are accepted and converted to pandas internally (requires pandas 2.0.2+ for the interchange path):

```python
import polars as pl
import seaborn as sns

df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 3, 5]})
sns.lineplot(data=df, x="x", y="y")
```

NumPy arrays and lists work as vectors (`sns.histplot(x=np.random.randn(100))`) or as wide-form 2D data.

### scikit-learn

Common patterns:

- Confusion matrix: `sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues")`.
- Feature importances: build a DataFrame and use `sns.barplot(data=imp, x="importance", y="feature")`.
- CV results: `pd.DataFrame(grid_search.cv_results_)` then `sns.lineplot(x="param_C", y="mean_test_score")` or a pivoted `heatmap` of two hyperparameters.
- Embeddings: `sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, palette="tab10")`.

```python
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
gs = GridSearchCV(SVC(), {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}, cv=5).fit(X, y)
cvr = pd.DataFrame(gs.cv_results_)
grid = cvr.pivot(index="param_C", columns="param_gamma", values="mean_test_score")
sns.heatmap(grid.astype(float), annot=True, fmt=".3f", cmap="viridis")
```

### statsmodels and SciPy

`regplot`/`lmplot`/`residplot` use statsmodels for `lowess=True`, `robust=True`, and `logistic=True`; for full regression summaries fit the model in statsmodels and plot residuals with seaborn. `clustermap` uses `scipy.cluster.hierarchy`; you can pass precomputed linkages.

### Deep learning frameworks

Convert training logs to a DataFrame (e.g. Keras `pd.DataFrame(history.history)`, or a list of dicts from a PyTorch loop), melt to long form, and use `lineplot`:

```python
import pandas as pd
import seaborn as sns

history = {"loss": [0.9, 0.6, 0.45, 0.4], "val_loss": [0.95, 0.7, 0.6, 0.62]}
h = pd.DataFrame(history).rename_axis("epoch").reset_index()
sns.lineplot(data=h.melt(id_vars="epoch", var_name="split", value_name="loss"),
             x="epoch", y="loss", hue="split", marker="o")
```

## Cheat Sheet

### Setup

| Task | Code |
|---|---|
| Import | `import seaborn as sns` / `import seaborn.objects as so` |
| Theme | `sns.set_theme(style="whitegrid", context="notebook", palette="deep")` |
| Bigger fonts for slides | `sns.set_context("talk")` |
| Temporary style | `with sns.axes_style("white"): ...` |
| Example data | `df = sns.load_dataset("tips")` |
| Remove top/right spines | `sns.despine()` |
| Move legend | `sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))` |

### Distributions

| Task | Code |
|---|---|
| Histogram | `sns.histplot(data=df, x="col")` |
| Histogram + KDE | `sns.histplot(data=df, x="col", kde=True)` |
| Normalized per group | `sns.histplot(data=df, x="col", hue="g", stat="density", common_norm=False)` |
| Stacked histogram | `sns.histplot(data=df, x="col", hue="g", multiple="stack")` |
| KDE | `sns.kdeplot(data=df, x="col", hue="g", fill=True)` |
| ECDF | `sns.ecdfplot(data=df, x="col", hue="g")` |
| 2D histogram | `sns.histplot(data=df, x="a", y="b", cbar=True)` |
| Faceted distributions | `sns.displot(data=df, x="col", col="g", kind="kde")` |
| Joint + marginals | `sns.jointplot(data=df, x="a", y="b", kind="hex")` |
| All pairs | `sns.pairplot(df, hue="g", corner=True)` |

### Relationships

| Task | Code |
|---|---|
| Scatter | `sns.scatterplot(data=df, x="a", y="b", hue="g", style="h")` |
| Line with CI | `sns.lineplot(data=df, x="t", y="v", hue="g")` |
| Line with SD band | `sns.lineplot(data=df, x="t", y="v", errorbar="sd")` |
| No aggregation | `sns.lineplot(data=df, x="t", y="v", units="id", estimator=None)` |
| Faceted relational | `sns.relplot(data=df, x="a", y="b", col="g", kind="line")` |
| Regression line | `sns.regplot(data=df, x="a", y="b")` |
| Regression per group | `sns.lmplot(data=df, x="a", y="b", hue="g")` |
| Logistic fit | `sns.regplot(data=df, x="a", y="binary", logistic=True)` |
| Residuals | `sns.residplot(data=df, x="a", y="b", lowess=True)` |

### Categorical

| Task | Code |
|---|---|
| Counts | `sns.countplot(data=df, x="cat")` |
| Percent counts | `sns.countplot(data=df, x="cat", stat="percent")` |
| Mean with CI bars | `sns.barplot(data=df, x="cat", y="v")` |
| Box | `sns.boxplot(data=df, x="cat", y="v", hue="g")` |
| Violin, split | `sns.violinplot(data=df, x="cat", y="v", hue="g", split=True)` |
| Points over box | `sns.stripplot(data=df, x="cat", y="v", color="k", alpha=0.5)` |
| Interaction plot | `sns.pointplot(data=df, x="cat", y="v", hue="g", dodge=True)` |
| Faceted categorical | `sns.catplot(data=df, x="cat", y="v", col="g", kind="box")` |
| Horizontal | `sns.boxplot(data=df, x="v", y="cat")` |

### Matrices and styling

| Task | Code |
|---|---|
| Correlation heatmap | `sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="vlag", center=0)` |
| Confusion matrix | `sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")` |
| Pivot heatmap | `sns.heatmap(df.pivot(index="r", columns="c", values="v"))` |
| Clustered heatmap | `sns.clustermap(df, z_score=1, cmap="vlag")` |
| Palette list | `sns.color_palette("Set2", 5)` |
| Palette as colormap | `sns.color_palette("rocket", as_cmap=True)` |
| Diverging colormap | `sns.diverging_palette(240, 10, as_cmap=True)` |
| Grid titles | `g.set_titles("{col_name}")` |
| Grid axis labels | `g.set_axis_labels("x", "y")` |
| Save grid | `g.savefig("out.png", dpi=150)` |

### Objects interface

| Task | Code |
|---|---|
| Scatter | `so.Plot(df, x="a", y="b").add(so.Dot())` |
| Colored + fit | `so.Plot(df, x="a", y="b", color="g").add(so.Dot()).add(so.Line(), so.PolyFit())` |
| Histogram | `so.Plot(df, x="a").add(so.Bars(), so.Hist())` |
| Mean bars, dodged | `so.Plot(df, x="cat", y="v", color="g").add(so.Bar(), so.Agg(), so.Dodge())` |
| Error bars | `.add(so.Range(), so.Est(errorbar="se"))` |
| Facet | `.facet(col="g", wrap=3)` |
| Log scale | `.scale(x="log")` |
| Size | `.layout(size=(8, 4))` |
| Draw on Axes | `.on(ax).plot()` |
| Save | `.save("p.png", dpi=150)` |

## Further Resources

- Official documentation: https://seaborn.pydata.org/
- Tutorial / user guide: https://seaborn.pydata.org/tutorial.html
- API reference: https://seaborn.pydata.org/api.html
- Example gallery: https://seaborn.pydata.org/examples/index.html
- The objects interface: https://seaborn.pydata.org/tutorial/objects_interface.html
- Release notes (what's new): https://seaborn.pydata.org/whatsnew/index.html
- Choosing color palettes: https://seaborn.pydata.org/tutorial/color_palettes.html
- GitHub repository: https://github.com/mwaskom/seaborn
- Example datasets: https://github.com/mwaskom/seaborn-data
- Paper: Waskom, M. L. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021
- Book: Jake VanderPlas, "Python Data Science Handbook" (visualization chapter covers seaborn): https://jakevdp.github.io/PythonDataScienceHandbook/
- Stack Overflow tag: https://stackoverflow.com/questions/tagged/seaborn
