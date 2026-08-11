"""DOE design matrix generators.

1st DOE : 3-factor 3-level full factorial (3^3 = 27 runs) over the whole
          operating window — screening + linear model fitting.
2nd DOE : CHF3 fixed at the value chosen from the 1st DOE, Pressure and
          Bias swept over a narrow window (3x3 = 9 runs) — refinement.
"""

from __future__ import annotations

import itertools

import pandas as pd

from .config import (
    FACTOR_LEVELS,
    FACTOR_NAMES,
    SECOND_DOE_CHF3,
    SECOND_DOE_LEVELS,
)


def full_factorial(levels: dict[str, list[float]]) -> pd.DataFrame:
    """Cartesian product of the given factor levels, in run order."""
    names = list(levels)
    rows = list(itertools.product(*(levels[n] for n in names)))
    df = pd.DataFrame(rows, columns=names)
    df.insert(0, "run", range(1, len(df) + 1))
    return df


def first_doe() -> pd.DataFrame:
    """27-run full factorial over the full factor window."""
    return full_factorial({n: FACTOR_LEVELS[n] for n in FACTOR_NAMES})


def second_doe() -> pd.DataFrame:
    """9-run refinement design with CHF3 fixed (chosen from 1st DOE)."""
    df = full_factorial(SECOND_DOE_LEVELS)
    df.insert(1, "CHF3", SECOND_DOE_CHF3)
    return df[["run", "CHF3", "P", "B"]]


def coded_units(design: pd.DataFrame, levels: dict[str, list[float]]) -> pd.DataFrame:
    """Convert natural units to coded -1/0/+1 units (for effect sizing)."""
    out = design.copy()
    for name, lv in levels.items():
        lo, hi = min(lv), max(lv)
        center, half = (hi + lo) / 2.0, (hi - lo) / 2.0
        out[name] = (out[name] - center) / half
    return out
