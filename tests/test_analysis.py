"""Regression recovery and optimization sanity tests.

The core scientific check: fitting OLS on the 27-run simulated DOE must
recover the ground-truth coefficients within noise-scaled tolerance.
"""

import numpy as np
import pytest

from src.analysis import fit_all, fit_ols, predict, standardized_effects
from src.config import FACTOR_NAMES, NOISE_SIGMA, TRUE_COEF
from src.data_gen import simulate_experiment
from src.doe_design import first_doe
from src.optimization import (
    best_condition,
    desirability,
    grid_search,
    validation_table,
)

SEED = 42


@pytest.fixture(scope="module")
def doe_data():
    return simulate_experiment(first_doe(), seed=SEED)


@pytest.fixture(scope="module")
def models(doe_data):
    return fit_all(doe_data, ["ER", "UT", "MD", "THETA"])


def test_coefficient_recovery(doe_data):
    """OLS coefficients must land close to ground truth for every response."""
    for r in ["ER", "UT", "MD"]:
        m = fit_ols(doe_data, r)
        for term in FACTOR_NAMES:
            se = float(m.bse[term])
            err = abs(float(m.params[term]) - TRUE_COEF[r][term])
            assert err < max(4 * se, 1e-6), f"{r}/{term}: err={err}, se={se}"


def test_r_squared_high(doe_data):
    for r in ["ER", "UT", "MD"]:
        assert fit_ols(doe_data, r).rsquared > 0.9


def test_effect_ranking_matches_physics(doe_data):
    """Bias must dominate ER and MD; pressure must dominate UT."""
    eff = standardized_effects(doe_data, ["ER", "UT", "MD"])
    assert eff["ER"].idxmax() == "B"
    assert eff["MD"].idxmax() == "B"
    assert eff["UT"].idxmax() == "P"


def test_desirability_bounds():
    assert desirability(380.0, "ER") == pytest.approx(1.0)
    assert desirability(280.0, "ER") == pytest.approx(0.0)
    assert desirability(0.3, "UT") == pytest.approx(1.0)
    assert desirability(2.5, "UT") == pytest.approx(0.0)
    assert desirability(90.0, "THETA") == pytest.approx(1.0)
    assert 0.0 < desirability(88.5, "THETA") < 1.0


def test_grid_search_finds_feasible_recipe(models):
    grid = grid_search(models, fixed={"CHF3": 25.0}, n_grid=41)
    best = best_condition(grid)
    assert best["feasible"]
    assert best["D"] > 0.3
    # optimum should sit at low pressure (physics: UT and ER both prefer it)
    assert best["P"] < 30.0


def test_final_recipe_passes_all_specs(models):
    pred = predict(models, {"CHF3": 25.0, "P": 15.0, "B": 82.0})
    table = validation_table(pred)
    core = table[table["response"].isin(["ER", "UT", "MD"])]
    assert (core["result"] == "PASS").all(), table.to_string()
