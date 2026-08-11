"""Unit tests for the virtual-fab data generator."""

import numpy as np
import pandas as pd
import pytest

from src.config import FACTOR_NAMES, RESPONSE_NAMES, TRUE_COEF
from src.data_gen import simulate_experiment, simulate_replicates, true_response
from src.doe_design import first_doe, second_doe


def test_first_doe_shape():
    d = first_doe()
    assert len(d) == 27
    assert list(d.columns) == ["run"] + FACTOR_NAMES
    # full factorial: every factor takes each level 9 times
    for f in FACTOR_NAMES:
        assert (d[f].value_counts() == 9).all()


def test_second_doe_fixed_chf3():
    d = second_doe()
    assert len(d) == 9
    assert (d["CHF3"] == 25.0).all()


def test_true_response_known_point():
    # hand-computed at CHF3=25, P=15, B=82 (the final recipe)
    x = pd.DataFrame([{"CHF3": 25.0, "P": 15.0, "B": 82.0}])
    assert true_response(x, "ER")[0] == pytest.approx(318.00)
    assert true_response(x, "UT")[0] == pytest.approx(0.76)
    assert true_response(x, "MD")[0] == pytest.approx(2.29)


def test_noise_free_simulation_matches_truth():
    d = first_doe()
    sim = simulate_experiment(d, noise=False)
    for r in RESPONSE_NAMES:
        expected = np.clip(true_response(d, r), 0, None) if r in ("UT", "MD") \
            else true_response(d, r)
        np.testing.assert_allclose(sim[r], expected, atol=1e-3)


def test_simulation_reproducible_with_seed():
    d = first_doe()
    a = simulate_experiment(d, seed=1)
    b = simulate_experiment(d, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_ratios_never_negative():
    d = first_doe()
    sim = simulate_experiment(d, seed=7)
    assert (sim["UT"] >= 0).all()
    assert (sim["MD"] >= 0).all()


def test_replicates():
    reps = simulate_replicates({"CHF3": 25, "P": 15, "B": 82}, n=3, seed=0)
    assert len(reps) == 3
    assert set(RESPONSE_NAMES) <= set(reps.columns)
