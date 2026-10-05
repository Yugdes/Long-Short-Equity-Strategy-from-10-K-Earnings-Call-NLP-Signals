"""
Plotting utilities — consistent finance-style charts.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ── Global style ─────────────────────────────────────────────────────────────
COLORS = {
    "long": "#2E86AB",
    "short": "#E8333C",
    "spread": "#1B1B2F",
    "benchmark": "#999999",
    "q1": "#2E86AB",
    "q2": "#59A5D8",
    "q3": "#999999",
    "q4": "#E07A5F",
    "q5": "#E8333C",
}

def set_style():
    """Apply a clean finance-style matplotlib theme."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.framealpha": 0.8,
    })
    sns.set_palette("muted")

set_style()


def save_figure(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    """Save a figure to PNG and PDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ── Standard charts ──────────────────────────────────────────────────────────

def plot_cumulative_returns(
    returns_dict: dict,
    title: str = "Cumulative Returns (Log Scale)",
    log_scale: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot cumulative returns for multiple series.
    returns_dict: {label: pd.Series of monthly returns}
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, rets in returns_dict.items():
        cum = (1 + rets).cumprod()
        color = COLORS.get(label.lower(), None)
        ax.plot(cum.index, cum.values, label=label, color=color, linewidth=1.5)

    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Drawdown",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot underwater (drawdown) chart."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color=COLORS["short"])
    ax.plot(drawdown.index, drawdown.values, color=COLORS["short"], linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_quintile_bars(
    quintile_returns: pd.Series,
    title: str = "Average Monthly Return by Quintile",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart of average return by quintile (Q1 to Q5)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS[f"q{i}"] for i in range(1, len(quintile_returns) + 1)]
    ax.bar(quintile_returns.index, quintile_returns.values * 100, color=colors, edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Monthly Return (%)")
    ax.set_xlabel("Quintile (1 = Low Risk, 5 = High Risk)")
    ax.axhline(0, color="black", linewidth=0.5)

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 36,
    title: str = "Rolling 36-Month Sharpe Ratio",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean() * 12
    rolling_std = returns.rolling(window).std() * np.sqrt(12)
    rolling_sharpe = rolling_mean / rolling_std

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=COLORS["spread"], linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("")

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_factor_loadings(
    loadings: pd.Series,
    title: str = "Factor Loadings (FF5 + Momentum)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of factor loadings with significance markers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS["long"] if v > 0 else COLORS["short"] for v in loadings.values]
    ax.barh(loadings.index, loadings.values, color=colors, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Loading")
    ax.axvline(0, color="black", linewidth=0.5)

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_heatmap(
    data: pd.DataFrame,
    title: str = "Heatmap",
    cmap: str = "RdYlGn_r",
    fmt: str = ".2f",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generic heatmap (e.g., industry × year risk scores)."""
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(data, cmap=cmap, annot=True, fmt=fmt, linewidths=0.5, ax=ax)
    ax.set_title(title)

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_coverage_funnel(
    funnel: pd.DataFrame,
    title: str = "Coverage Funnel by Year",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Stacked area chart of sample attrition by year."""
    fig, ax = plt.subplots(figsize=(14, 6))
    funnel.plot.area(ax=ax, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Number of Firms")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", fontsize=9)

    if save_path:
        save_figure(fig, save_path)
    return fig
