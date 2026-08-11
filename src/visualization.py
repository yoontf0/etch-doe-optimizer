"""All matplotlib figures for the DOE pipeline.

Every figure is generated from simulated data by this project's own code —
no external images are used. Axis labels are in English for font
portability; the Korean narrative lives in the notebooks and README.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    DESIRABILITY,
    FACTOR_LABELS,
    FACTOR_NAMES,
    RESPONSE_LABELS,
    SPECS,
)
from .analysis import main_effects
from .spc import i_chart_stats

FIG_DPI = 150


def save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- main effects
def main_effects_plot(df: pd.DataFrame, response: str) -> plt.Figure:
    """One row of subplots: mean response vs. level for each factor."""
    effects = main_effects(df, response)
    fig, axes = plt.subplots(1, len(FACTOR_NAMES), figsize=(11, 3.2), sharey=True)
    for ax, f in zip(axes, FACTOR_NAMES):
        s = effects[f]
        ax.plot(s.index, s.values, "o-", color="tab:blue")
        ax.axhline(df[response].mean(), color="gray", ls="--", lw=0.8)
        ax.set_xlabel(FACTOR_LABELS[f])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(RESPONSE_LABELS[response])
    fig.suptitle(f"Main Effects — {response}", y=1.02)
    fig.tight_layout()
    return fig


def effect_magnitude_plot(std_eff: pd.DataFrame) -> plt.Figure:
    """Bar chart of |effect| per factor, one panel per response."""
    responses = list(std_eff.columns)
    fig, axes = plt.subplots(1, len(responses), figsize=(3.7 * len(responses), 3.4))
    colors = {"CHF3": "tab:green", "P": "tab:orange", "B": "tab:red"}
    for ax, r in zip(np.atleast_1d(axes), responses):
        vals = std_eff[r]
        ax.bar(vals.index, vals.values, color=[colors[f] for f in vals.index])
        ax.set_title(RESPONSE_LABELS[r], fontsize=10)
        ax.set_ylabel("|high mean − low mean|")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Effect magnitude by factor (1st DOE)", y=1.03)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- trade-off
def tradeoff_plot(df: pd.DataFrame) -> plt.Figure:
    """Bias power drives MD up but ER up too: the core conflict.

    MD must go DOWN (<= 3 %) while ER must go UP (>= 300 nm/min), and both
    are dominated by the same knob (bias power) with the same sign.
    """
    g = df.groupby("B")[["ER", "MD"]].mean()
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()

    ax1.plot(g.index, g["ER"], "o-", color="tab:blue", label="ER (want ≥ 300)")
    ax1.axhline(SPECS["ER"]["limit"], color="tab:blue", ls=":", lw=1)
    ax2.plot(g.index, g["MD"], "s-", color="tab:red", label="M/D (want ≤ 3)")
    ax2.axhline(SPECS["MD"]["limit"], color="tab:red", ls=":", lw=1)

    ax1.set_xlabel(FACTOR_LABELS["B"])
    ax1.set_ylabel(RESPONSE_LABELS["ER"], color="tab:blue")
    ax2.set_ylabel(RESPONSE_LABELS["MD"], color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ax1.set_title("Bias power trade-off: productivity (ER) vs. micro-trench (M/D)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- contours
def desirability_contour(
    grid_df: pd.DataFrame,
    best: pd.Series | None = None,
    fixed_label: str = "",
) -> plt.Figure:
    """Pressure x Bias contour of overall desirability D with the
    hard-spec feasible region overlaid as a hatched area."""
    piv_D = grid_df.pivot_table(index="P", columns="B", values="D")
    piv_F = grid_df.pivot_table(index="P", columns="B", values="feasible")
    B, P = np.meshgrid(piv_D.columns.to_numpy(), piv_D.index.to_numpy())

    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(B, P, piv_D.values, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="Overall desirability D")

    # feasible region: hatched overlay + white boundary
    ax.contourf(
        B, P, piv_F.values.astype(float), levels=[0.5, 1.5],
        colors="none", hatches=["///"],
    )
    ax.contour(B, P, piv_F.values.astype(float), levels=[0.5],
               colors="white", linewidths=1.5)

    if best is not None:
        ax.plot(best["B"], best["P"], "r*", ms=16,
                label=f"best (P={best['P']:.0f}, B={best['B']:.0f}, D={best['D']:.2f})")
        ax.legend(loc="upper right")

    ax.set_xlabel(FACTOR_LABELS["B"])
    ax.set_ylabel(FACTOR_LABELS["P"])
    title = "Desirability surface (hatched = all specs satisfied)"
    if fixed_label:
        title += f"  |  {fixed_label}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- SPC
def i_chart(df: pd.DataFrame, response: str, ax: plt.Axes | None = None) -> plt.Figure:
    """Individuals control chart for one response over production lots."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    else:
        fig = ax.figure
    x, y = df["lot"], df[response]
    st = i_chart_stats(y)

    ax.plot(x, y, "o-", color="tab:blue", ms=4)
    ax.axhline(st["center"], color="green", lw=1, label=f"CL={st['center']:.2f}")
    ax.axhline(st["ucl"], color="red", ls="--", lw=1, label=f"UCL={st['ucl']:.2f}")
    ax.axhline(st["lcl"], color="red", ls="--", lw=1, label=f"LCL={st['lcl']:.2f}")

    ooc = (y > st["ucl"]) | (y < st["lcl"])
    if ooc.any():
        ax.plot(x[ooc], y[ooc], "rx", ms=10, label="out of control")

    ax.set_xlabel("Lot")
    ax.set_ylabel(RESPONSE_LABELS[response])
    ax.set_title(f"I-chart — {response}", fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.grid(alpha=0.3)
    if own_fig:
        fig.tight_layout()
    return fig


def i_chart_panel(df: pd.DataFrame, responses: list[str]) -> plt.Figure:
    """Stacked I-charts for several responses at the final recipe."""
    fig, axes = plt.subplots(len(responses), 1, figsize=(8, 2.9 * len(responses)))
    for ax, r in zip(np.atleast_1d(axes), responses):
        i_chart(df, r, ax=ax)
    fig.suptitle("SPC monitoring at the final recipe", y=1.0)
    fig.tight_layout()
    return fig
