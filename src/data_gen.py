"""Virtual-fab data generator for the SiO2 plasma etch process.

The ground-truth model is a first-order linear response surface whose
coefficient SIGNS encode well-known plasma etch physics:

ER = -3.00*CHF3 - 1.20*P + 2.50*B + 206.00      (sigma ~ 5 nm/min)
    * B (bias power) up   -> ion bombardment energy up -> ER up (+2.50)
    * P (pressure) up     -> mean free path down, ion flux less directional
                             -> vertical etch component down (-1.20)
    * CHF3 up             -> thicker fluorocarbon polymer passivation
                             consumes F radicals -> ER down (-3.00)

UT = -0.15*CHF3 + 0.08*P - 0.02*B + 4.95        (sigma ~ 0.10 %)
    * CHF3 up -> stronger sidewall passivation blocks lateral attack of
      isotropic F radicals -> undercut down (-0.15)
    * P up    -> ion scattering + radical-dominant regime -> lateral
      chemical etch up -> undercut up (+0.08)
    * B up    -> anisotropic (vertical) etch dominates -> undercut
      slightly down (-0.02)

MD = 0.03*CHF3 - 0.05*P + 0.12*B - 7.55         (sigma ~ 0.15 %)
    * B up    -> ions reflected off the sidewall focus energy at the
      trench bottom corners -> micro-trench up (+0.12, dominant)
    * P up    -> ion angular spread disperses corner bombardment ->
      micro-trench down (-0.05)
    * CHF3 up -> passivated sidewall reflects ions toward corners ->
      micro-trench slightly up (+0.03)

THETA = 90.40 + 0.08*CHF3 - 0.11*P              (sigma ~ 0.15 deg)
    * Simple model: sidewall angle stays near 90 deg and degrades weakly
      as pressure rises (ion directionality loss); CHF3 passivation
      slightly straightens the profile.

Every generated dataset is pure simulation — no measured fab data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FACTOR_NAMES, NOISE_SIGMA, RESPONSE_NAMES, TRUE_COEF


def true_response(design: pd.DataFrame, response: str) -> np.ndarray:
    """Evaluate the noise-free ground-truth model for one response.

    Parameters
    ----------
    design : DataFrame with columns CHF3, P, B (any number of rows)
    response : one of "ER", "UT", "MD", "THETA"
    """
    coef = TRUE_COEF[response]
    y = np.full(len(design), coef["const"], dtype=float)
    for f in FACTOR_NAMES:
        y += coef[f] * design[f].to_numpy(dtype=float)
    return y


def simulate_experiment(
    design: pd.DataFrame,
    seed: int | None = None,
    noise: bool = True,
) -> pd.DataFrame:
    """Run the virtual fab: attach simulated responses to a design matrix.

    Returns a copy of ``design`` with ER, UT, MD, THETA columns appended.
    UT and MD are clipped at 0 (a ratio cannot be negative).
    """
    rng = np.random.default_rng(seed)
    out = design.copy()
    for r in RESPONSE_NAMES:
        y = true_response(design, r)
        if noise:
            y = y + rng.normal(0.0, NOISE_SIGMA[r], size=len(design))
        if r in ("UT", "MD"):
            y = np.clip(y, 0.0, None)
        out[r] = np.round(y, 3)
    return out


def simulate_replicates(
    condition: dict[str, float],
    n: int = 3,
    seed: int | None = None,
) -> pd.DataFrame:
    """Repeat one recipe n times to check run-to-run reproducibility."""
    design = pd.DataFrame([condition] * n)[FACTOR_NAMES]
    df = simulate_experiment(design, seed=seed)
    df.insert(0, "run", np.arange(1, n + 1))
    return df
