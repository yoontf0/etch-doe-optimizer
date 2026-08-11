"""Statistical analysis of DOE results.

- main_effects        : per-factor mean response at each level
- standardized_effects: coded-unit effect magnitudes (|high mean - low mean|)
- fit_ols             : statsmodels OLS fit y ~ CHF3 + P + B
- ols_report          : tidy coefficient / p-value / R^2 table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import FACTOR_NAMES, RESPONSE_NAMES


def main_effects(df: pd.DataFrame, response: str) -> dict[str, pd.Series]:
    """Mean of ``response`` at each level of each factor."""
    return {f: df.groupby(f)[response].mean() for f in FACTOR_NAMES}


def standardized_effects(df: pd.DataFrame, responses: list[str] | None = None) -> pd.DataFrame:
    """Effect magnitude per factor: |mean(high level) - mean(low level)|.

    Computed on natural-unit data but comparing extreme levels, which for a
    balanced full factorial equals the classic coded-unit effect estimate.
    Rows: factors, columns: responses.
    """
    responses = responses or [r for r in RESPONSE_NAMES if r != "THETA"]
    rows = {}
    for f in FACTOR_NAMES:
        lv = sorted(df[f].unique())
        lo, hi = lv[0], lv[-1]
        rows[f] = {
            r: abs(df.loc[df[f] == hi, r].mean() - df.loc[df[f] == lo, r].mean())
            for r in responses
        }
    return pd.DataFrame(rows).T[responses]


def fit_ols(df: pd.DataFrame, response: str):
    """First-order OLS fit: response ~ const + (non-constant factors).

    Factors held constant in the design (e.g. CHF3 in the 2nd DOE) are
    dropped automatically — their effect is absorbed by the intercept.
    """
    factors = [f for f in FACTOR_NAMES if df[f].nunique() > 1]
    X = sm.add_constant(df[factors].astype(float), has_constant="add")
    return sm.OLS(df[response].astype(float), X).fit()


def ols_report(df: pd.DataFrame, responses: list[str] | None = None) -> pd.DataFrame:
    """Coefficients, p-values and R^2 for each response, as a tidy table."""
    responses = responses or RESPONSE_NAMES
    rows = []
    for r in responses:
        m = fit_ols(df, r)
        for term in m.model.exog_names:
            rows.append(
                {
                    "response": r,
                    "term": term,
                    "coef": m.params[term],
                    "p_value": m.pvalues[term],
                    "r_squared": m.rsquared,
                }
            )
    rep = pd.DataFrame(rows)
    rep["coef"] = rep["coef"].round(4)
    rep["p_value"] = rep["p_value"].apply(lambda p: float(f"{p:.3g}"))
    rep["r_squared"] = rep["r_squared"].round(4)
    return rep


def design_matrix_for(model, df: pd.DataFrame) -> pd.DataFrame:
    """Build the exog matrix a fitted model expects from raw factor data."""
    X = pd.DataFrame(index=df.index)
    for name in model.model.exog_names:
        X[name] = 1.0 if name == "const" else df[name].astype(float)
    return X


def predict(models: dict, condition: dict[str, float]) -> dict[str, float]:
    """Predict every response at one condition from fitted OLS models."""
    x = pd.DataFrame([condition])
    return {
        r: float(m.predict(design_matrix_for(m, x)).iloc[0])
        for r, m in models.items()
    }


def fit_all(df: pd.DataFrame, responses: list[str] | None = None) -> dict:
    """Fit OLS models for all responses; returns {response: model}."""
    responses = responses or RESPONSE_NAMES
    return {r: fit_ols(df, r) for r in responses}
