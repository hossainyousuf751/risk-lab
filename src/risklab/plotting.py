"""Diagnostic plots (requires matplotlib)."""

from __future__ import annotations

import numpy as np
import pandas as pd

C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#222222", "#64B5CD", "#8C8C8C"]
STYLES = ["-", "--", "-.", ":"] * 3


def _finish(fig, ax, path, title):
    ax.set_title(title, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_var_paths(r: pd.Series, forecasts: dict[str, pd.Series], path: str, title: str,
                   start=None, end=None):
    import matplotlib.pyplot as plt

    r = r.loc[start:end]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(r.index, r * 100, width=1.0, color="#B0B0B0", label="daily return")
    for i, (name, v) in enumerate(forecasts.items()):
        v = v.loc[start:end]
        ax.plot(v.index, -v * 100, STYLES[i % len(C)], color=C[i % len(C)], lw=1.6, label=f"−VaR {name}")
        hit = r < -v
        ax.scatter(r.index[hit], r[hit] * 100, s=28, marker="v", color=C[i % len(C)], zorder=3)
    ax.set_ylabel("return (%)")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _finish(fig, ax, path, title)
    plt.close(fig)


def plot_method_lines(table: pd.DataFrame, x: str, y: str, path: str, title: str, ylabel: str,
                      reference: float | None = None, ref_label: str = ""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, g) in enumerate(table.groupby("method", sort=False)):
        ax.plot(g[x], g[y], STYLES[i % len(C)], marker="o", color=C[i % len(C)], lw=1.8, label=m)
    if reference is not None:
        ax.axhline(reference, color="black", lw=1, ls=":", label=ref_label)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(sorted(table[x].unique()), [str(v) for v in sorted(table[x].unique())])
    ax.set_xlabel("estimation window (days)  ·  50 assets")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.spines[["top", "right"]].set_visible(False)
    _finish(fig, ax, path, title)
    plt.close(fig)
