"""Multi-response simultaneous optimization via Derringer-Suich
desirability functions (Derringer & Suich, J. Quality Technology, 1980).

Each response y is mapped to a desirability d in [0, 1]:
    larger-is-better (ER)   : d = clip((y - low) / (high - low), 0, 1)
    smaller-is-better (UT,MD): d = clip((high - y) / (high - low), 0, 1)
    target-is-best (THETA)  : linear ramp up to the target, down after it

The overall desirability D is the geometric mean of the individual d's,
so a single unacceptable response (d = 0) kills the whole recipe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DESIRABILITY, FACTOR_NAMES, FACTOR_RANGES, SPECS


def desirability(y: np.ndarray | float, response: str) -> np.ndarray:
    """Individual Derringer-Suich desirability for one response."""
    cfg = DESIRABILITY[response]
    y = np.asarray(y, dtype=float)
    lo, hi = cfg["low"], cfg["high"]
    if cfg["goal"] == "max":
        d = (y - lo) / (hi - lo)
    elif cfg["goal"] == "min":
        d = (hi - y) / (hi - lo)
    elif cfg["goal"] == "target":
        t = cfg["target"]
        d = np.where(y <= t, (y - lo) / (t - lo), (hi - y) / (hi - t))
    else:  # pragma: no cover
        raise ValueError(f"unknown goal {cfg['goal']!r}")
    return np.clip(d, 0.0, 1.0)


def overall_desirability(pred: dict[str, np.ndarray]) -> np.ndarray:
    """Geometric mean of individual desirabilities."""
    ds = np.stack([desirability(pred[r], r) for r in pred], axis=0)
    return np.exp(np.mean(np.log(np.clip(ds, 1e-12, None)), axis=0)) * (ds.min(axis=0) > 0)


def meets_specs(pred: dict[str, float | np.ndarray]) -> np.ndarray:
    """Boolean mask: does the prediction satisfy every hard spec?"""
    ok = None
    for r, spec in SPECS.items():
        if r not in pred:
            continue
        y = np.asarray(pred[r], dtype=float)
        if spec["type"] == "min":
            m = y >= spec["limit"]
        elif spec["type"] == "max":
            m = y <= spec["limit"]
        else:  # target
            m = np.abs(y - spec["target"]) <= spec["tol"]
        ok = m if ok is None else (ok & m)
    return ok


def grid_search(
    models: dict,
    ranges: dict[str, tuple[float, float]] | None = None,
    fixed: dict[str, float] | None = None,
    n_grid: int = 61,
) -> pd.DataFrame:
    """Evaluate desirability over a factor grid using fitted OLS models.

    ``fixed`` pins factors (e.g. {"CHF3": 25.0}); remaining factors are
    swept over ``ranges`` (defaults to the full operating window).
    Returns a long-format DataFrame with factors, predictions, d_* and D.
    """
    from .analysis import design_matrix_for

    ranges = ranges or FACTOR_RANGES
    fixed = fixed or {}
    free = [f for f in FACTOR_NAMES if f not in fixed]
    axes = [np.linspace(*ranges[f], n_grid) for f in free]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = pd.DataFrame({f: m.ravel() for f, m in zip(free, mesh)})
    for f, v in fixed.items():
        grid[f] = v

    pred = {
        r: np.asarray(m.predict(design_matrix_for(m, grid)))
        for r, m in models.items()
    }
    out = grid.copy()
    for r, y in pred.items():
        out[f"pred_{r}"] = y
        out[f"d_{r}"] = desirability(y, r)
    out["D"] = overall_desirability(pred)
    out["feasible"] = meets_specs(pred)
    return out


def best_condition(grid_df: pd.DataFrame, require_feasible: bool = True) -> pd.Series:
    """Row with maximum overall desirability (optionally spec-feasible)."""
    df = grid_df[grid_df["feasible"]] if require_feasible else grid_df
    if len(df) == 0:
        raise ValueError("no feasible point in the searched grid")
    return df.loc[df["D"].idxmax()]


def validation_table(pred: dict[str, float]) -> pd.DataFrame:
    """PASS/FAIL table of one predicted condition against every spec."""
    rows = []
    for r, spec in SPECS.items():
        if r not in pred:
            continue
        y = float(pred[r])
        if spec["type"] == "min":
            crit, ok = f">= {spec['limit']}", y >= spec["limit"]
        elif spec["type"] == "max":
            crit, ok = f"<= {spec['limit']}", y <= spec["limit"]
        else:
            crit = f"{spec['target']} +/- {spec['tol']}"
            ok = abs(y - spec["target"]) <= spec["tol"]
        rows.append(
            {"response": r, "predicted": round(y, 2), "criterion": crit,
             "result": "PASS" if ok else "FAIL"}
        )
    return pd.DataFrame(rows)
