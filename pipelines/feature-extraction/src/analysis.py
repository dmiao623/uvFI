import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
import umap
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from typing import Any, Callable, Iterable, List, Optional, Sequence Union

def merge_on_transformed(
    df1: pd.DataFrame,
    col1: str,
    f1: Callable[[Any], Any],
    df2: pd.DataFrame,
    col2: str,
    f2: Callable[[Any], Any],
    keep: str = "inner"
) -> pd.DataFrame:

    df1_tmp = df1[[col1]].copy()
    df2_tmp = df2[[col2]].copy()

    df1_tmp["key"] = df1_tmp[col1].map(f1)
    df2_tmp["key"] = df2_tmp[col2].map(f2)

    merged = pd.merge(df1_tmp[["key"]], df2_tmp[["key"]], on="key", how=keep)
    merged = merged.drop_duplicates("key").reset_index(drop=True)
    return merged

def plot_feature_correlations(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    sort_by_abs: bool = True,
    cmap: str = "coolwarm",
    fmt: str = ".2f",
) -> None:
    with np.errstate(invalid='ignore', divide='ignore'):
        r = df[feature_cols].corrwith(df[target_col], method='pearson')
    if sort_by_abs:
        r = r.reindex(r.abs().sort_values(ascending=False).index)

    corr_df = pd.DataFrame(r).T
    corr_df.index = [target_col]

    f_w = max(4, len(r) * 0.35)
    f_h = 2.2
    base = 10

    plt.figure(figsize=(f_w, f_h))
    sns.set(font_scale=1)

    ax = sns.heatmap(
        corr_df, cmap=cmap,
        center=0, vmin=-1, vmax=1,
        linewidths=0.6, linecolor='black',
        annot=True, fmt=fmt, annot_kws={"size": base * 0.8},
        cbar_kws={
            "label": "Pearson r",
            "shrink": 0.6,
            "pad": 0.02,
            "aspect": 15,
        },
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center",
                       fontsize=base * 0.8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=base * 0.9)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=base * 0.8)          # tick number font
    cbar.ax.set_ylabel("Pearson r", fontsize=base * 0.9, rotation=-90,
                       va='center', labelpad=12)

    ax.set_title(f"Correlation with {target_col}",
                 loc="left", weight="bold", fontsize=base * 1.1)

    plt.tight_layout()
    plt.show()

def plot_feature_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    fit_line: bool = True,
) -> None:

    data = df[[x_col, y_col]].dropna()
    r, _ = pearsonr(data[x_col], data[y_col])

    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=x_col, y=y_col, data=data, s=35)

    if fit_line:
        sns.regplot(
            x=x_col, y=y_col, data=data,
            scatter=False, line_kws={"color": "black", "linewidth": 1}
        )

    plt.title(f"{y_col} vs {x_col}  (r = {r:.2f})")
    plt.tight_layout()
    plt.show()

def plot_correlation_grid(
        df: pd.DataFrame,
        feature_cols: list[str],
        *,
        cluster: bool = True,
    ) -> None:

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').replace(
        [np.inf, -np.inf], np.nan
    )
    good = X.nunique(dropna=True) > 1
    X = X.loc[:, good]

    if X.shape[1] < 2:
        raise ValueError("Need at least two valid features to plot a grid.")

    corr = X.corr(method='pearson')
    var  = X.var()
    np.fill_diagonal(corr.values, var)

    if cluster:
        dist_for_linkage = 1 - corr.abs().fillna(0).values
        linkage = sch.linkage(squareform(dist_for_linkage, checks=False),
                              method='average', optimal_ordering=True)
        order   = sch.dendrogram(linkage, no_plot=True)['leaves']
        corr    = corr.iloc[order, order]

    n = len(corr)
    fig_size = max(4, n * 0.35)
    plt.figure(figsize=(fig_size, fig_size))

    sns.heatmap(
        corr,
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4, linecolor='black',
        square=True,
        cbar_kws={'label': 'Pearson r (off-diag) / Variance (diag)'}
    )

    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.title('Feature × Feature Correlation / Variance Grid',
              loc='left', weight='bold')

    plt.tight_layout()
    plt.show()

def plot_umap(
    df: pd.DataFrame,
    feature_cols: List[str],
    color_col: str,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    cmap: Union[str, plt.Colormap] = "viridis",
    figsize: tuple = (6, 5),
    title: Optional[str] = None,
    show: bool = True,
) -> umap.UMAP:

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    embedding = reducer.fit_transform(df[feature_cols].values)

    values = df[color_col].values
    norm = Normalize(vmin=values.min(), vmax=values.max())

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values,
        cmap=cmap,
        norm=norm,
        s=10,
        linewidth=0,
        alpha=0.8,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_col)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title or f"UMAP colored by `{color_col}`")
    ax.set_aspect("equal", "datalim")

    if show:
        plt.tight_layout()
        plt.show()
    return reducer

