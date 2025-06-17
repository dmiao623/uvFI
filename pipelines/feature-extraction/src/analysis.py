import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
import umap
from pathlib import Path
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

def merge_on_transformed(
    df1: pd.DataFrame,
    col1: str,
    f1: Callable[[Any], Any],
    df2: pd.DataFrame,
    col2: str,
    f2: Callable[[Any], Any],
    how: str = "inner",
    *,
    suffixes: tuple[str, str] = ("_x", "_y"),
    rename: Optional[str]="name",
) -> pd.DataFrame:
    df1_tmp = df1.copy()
    df2_tmp = df2.copy()

    key = "__merge_key__"
    df1_tmp[key] = df1_tmp[col1].map(f1)
    df2_tmp[key] = df2_tmp[col2].map(f2)

    merged = pd.merge(df1_tmp, df2_tmp, on=key, how=how, suffixes=suffixes)
    merged = merged.drop(columns=[col1, col2])

    if rename:
        merged = merged.rename(columns={key: rename})
        col = merged.pop(rename)
        merged.insert(0, rename, col)

    return merged

def plot_feature_correlations(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    cols_per_row: int = 64,
    sort_by_abs: bool = True,
    cmap: str = "coolwarm",
    fmt: str = ".2f",
    h_per_row: float = 2.3,
    w_per_col: float = 0.6,
    base_font: float = 10.0,
) -> None:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = df[feature_cols].corrwith(df[target_col], method="pearson")

    if sort_by_abs:
        r = r.reindex(r.abs().sort_values(ascending=False).index)

    chunks = [
        r.iloc[i : i + cols_per_row]
        for i in range(0, len(r), cols_per_row)
    ]
    n_rows = len(chunks)

    fig_w = max(4.0, min(cols_per_row, max(map(len, chunks))) * w_per_col + 1)
    fig_h = max(3.0, n_rows * h_per_row)

    sns.set(font_scale=1)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        squeeze=False,
    )

    vmin, vmax = -1.0, 1.0
    cbar_ax = None

    for idx, (ax, chunk) in enumerate(zip(axes[:, 0], chunks), start=1):
        heat = sns.heatmap(
            pd.DataFrame(chunk).T,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.6,
            linecolor="black",
            annot=True,
            fmt=fmt,
            annot_kws={"size": base_font * 0.8},
            cbar=idx == 1,
            cbar_ax=cbar_ax,
            cbar_kws={
                "label": "Pearson r",
                "shrink": 0.6,
                "pad": 0.02,
                "aspect": 15,
            },
        )
        if idx == 1:
            cbar_ax = heat.collections[0].colorbar.ax

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=90,
            ha="center",
            fontsize=base_font * 0.8,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=base_font * 0.9,
        )

        if idx == 1:
            ax.set_title(
                f"Correlation with {target_col}",
                loc="left",
                weight="bold",
                fontsize=base_font * 1.1,
            )
        else:
            ax.set_title("")
    if cbar_ax is not None:
        cbar_ax.set_ylabel("Pearson r", rotation=-90, labelpad=12)

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
    feature_cols: Sequence[str],
    color_col: str,
    random_state: int=623,
) -> Optional[np.ndarray]:
    X = np.vstack(df[feature_cols].values)
    embedding = umap.UMAP(random_state=random_state).fit_transform(X)

    plt.figure(figsize=(10, 10))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=df[color_col].values)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"UMAP of selected features (colored by {color_col})")
    plt.colorbar(sc, label=color_col)
    plt.tight_layout()

    # return embedding

def plot_pca(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    color_col: str,
    label_col: Optional[str] = None,
    *,
    n_components: int=2,
    random_state: int=623,
) -> Optional[np.ndarray]:
    X = np.vstack(df[feature_cols].values)

    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(X)

    plt.figure(figsize=(10, 10))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=df[color_col].values)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"PCA of selected features (colored by {color_col})")
    plt.colorbar(sc, label=color_col)

    if label_col is not None:
        for (x, y), label in zip(embedding, df[label_col].values):
            plt.text(x, y, str(label), fontsize=7, ha="center", va="center")

    plt.tight_layout()
    plt.show()

    # return embedding
