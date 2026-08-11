"""Statistical Process Control (SPC) extension for the final recipe.

Individuals chart (I-chart) with moving-range based control limits:
    UCL/LCL = mean +/- 2.66 * MR_bar        (standard I-MR constants)

Also provides a reproducibility check: n replicate runs at the final
recipe, reporting mean, sigma and whether each run stays inside the
regression-predicted expectation band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_gen import simulate_replicates


def i_chart_stats(values: np.ndarray | pd.Series) -> dict:
    """Center line and control limits for an individuals chart."""
    x = np.asarray(values, dtype=float)
    mr = np.abs(np.diff(x))
    mr_bar = mr.mean() if len(mr) else 0.0
    center = x.mean()
    return {
        "center": center,
        "mr_bar": mr_bar,
        "ucl": center + 2.66 * mr_bar,
        "lcl": center - 2.66 * mr_bar,
        "sigma_hat": mr_bar / 1.128,  # d2 for n=2
    }


def monitoring_run(
    condition: dict[str, float],
    n_lots: int = 25,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate ``n_lots`` production lots at the fixed final recipe."""
    df = simulate_replicates(condition, n=n_lots, seed=seed)
    df = df.rename(columns={"run": "lot"})
    return df


def reproducibility_report(
    condition: dict[str, float],
    predicted: dict[str, float],
    n: int = 3,
    seed: int | None = None,
    k_sigma: float = 3.0,
) -> pd.DataFrame:
    """n replicate runs vs. regression prediction +/- k*sigma_noise band."""
    from .config import NOISE_SIGMA, RESPONSE_NAMES

    reps = simulate_replicates(condition, n=n, seed=seed)
    rows = []
    for r in RESPONSE_NAMES:
        vals = reps[r].to_numpy()
        band = k_sigma * NOISE_SIGMA[r]
        rows.append(
            {
                "response": r,
                "predicted": round(float(predicted[r]), 2),
                "mean": round(vals.mean(), 3),
                "sigma": round(vals.std(ddof=1), 3),
                "band": f"+/- {band:g}",
                "all_within_band": bool(np.all(np.abs(vals - predicted[r]) <= band)),
            }
        )
    return pd.DataFrame(rows)
