"""End-to-end DOE pipeline: simulate -> analyze -> optimize -> verify -> SPC.

Regenerates every figure in results/ and prints the key tables.
Run from the repo root:  python scripts/run_pipeline.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import fit_all, ols_report, predict, standardized_effects
from src.config import RANDOM_SEED, SECOND_DOE_CHF3
from src.data_gen import simulate_experiment
from src.doe_design import first_doe, second_doe
from src.optimization import best_condition, grid_search, validation_table
from src.spc import monitoring_run, reproducibility_report
from src.visualization import (
    desirability_contour,
    effect_magnitude_plot,
    i_chart_panel,
    main_effects_plot,
    save,
    tradeoff_plot,
)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def main() -> None:
    # ---------------------------------------------------------- 1st DOE
    print("=" * 60)
    print("[1] 1st DOE: 3-factor 3-level full factorial (27 runs)")
    doe1 = simulate_experiment(first_doe(), seed=RANDOM_SEED)
    doe1.to_csv(os.path.join(RESULTS, "doe1_data.csv"), index=False)
    print(doe1.head(9).to_string(index=False))

    for r in ["ER", "UT", "MD"]:
        save(main_effects_plot(doe1, r), os.path.join(RESULTS, f"main_effects_{r}.png"))

    eff = standardized_effects(doe1, ["ER", "UT", "MD"])
    print("\nEffect magnitude (|high - low| mean):")
    print(eff.round(3).to_string())
    save(effect_magnitude_plot(eff), os.path.join(RESULTS, "effect_magnitude.png"))

    # --------------------------------------------------------- regression
    print("\n" + "=" * 60)
    print("[2] OLS regression (y ~ CHF3 + P + B)")
    rep = ols_report(doe1, ["ER", "UT", "MD", "THETA"])
    rep.to_csv(os.path.join(RESULTS, "ols_report.csv"), index=False)
    print(rep.to_string(index=False))
    models = fit_all(doe1, ["ER", "UT", "MD", "THETA"])

    # ---------------------------------------------------------- trade-off
    save(tradeoff_plot(doe1), os.path.join(RESULTS, "tradeoff_bias.png"))

    # ------------------------------------------------------- optimization
    print("\n" + "=" * 60)
    print(f"[3] Desirability optimization (CHF3 fixed at {SECOND_DOE_CHF3} sccm)")
    grid1 = grid_search(models, fixed={"CHF3": SECOND_DOE_CHF3}, n_grid=121)
    best1 = best_condition(grid1)
    print(f"grid optimum: P={best1['P']:.1f} mTorr, B={best1['B']:.1f} W, "
          f"D={best1['D']:.3f}")
    save(
        desirability_contour(grid1, best1, fixed_label=f"CHF3={SECOND_DOE_CHF3:g} sccm"),
        os.path.join(RESULTS, "desirability_contour.png"),
    )

    # ------------------------------------------------------------ 2nd DOE
    print("\n" + "=" * 60)
    print("[4] 2nd DOE: refinement around the optimum (9 runs)")
    doe2 = simulate_experiment(second_doe(), seed=RANDOM_SEED + 1)
    doe2.to_csv(os.path.join(RESULTS, "doe2_data.csv"), index=False)
    print(doe2.to_string(index=False))

    models2 = fit_all(doe2, ["ER", "UT", "MD", "THETA"])
    grid2 = grid_search(
        models2,
        ranges={"P": (15.0, 25.0), "B": (60.0, 90.0)},
        fixed={"CHF3": SECOND_DOE_CHF3},
        n_grid=121,
    )
    best2 = best_condition(grid2)
    final = {"CHF3": SECOND_DOE_CHF3, "P": round(best2["P"]), "B": round(best2["B"])}
    print(f"\nFinal recipe: CHF3={final['CHF3']:g} sccm / "
          f"P={final['P']:g} mTorr / B={final['B']:g} W")
    save(
        desirability_contour(grid2, best2,
                             fixed_label=f"2nd DOE model, CHF3={SECOND_DOE_CHF3:g}"),
        os.path.join(RESULTS, "desirability_contour_2nd.png"),
    )

    # ---------------------------------------------------------- validation
    print("\n" + "=" * 60)
    print("[5] Validation: final recipe vs. specs (1st-DOE global model)")
    pred = predict(models, final)
    table = validation_table(pred)
    table.to_csv(os.path.join(RESULTS, "validation_table.csv"), index=False)
    print(table.to_string(index=False))

    print("\nReproducibility (n=3 replicate runs):")
    reps = reproducibility_report(final, pred, n=3, seed=RANDOM_SEED + 2)
    reps.to_csv(os.path.join(RESULTS, "reproducibility.csv"), index=False)
    print(reps.to_string(index=False))

    # ----------------------------------------------------------------- SPC
    print("\n" + "=" * 60)
    print("[6] SPC: I-charts over 25 simulated lots at the final recipe")
    lots = monitoring_run(final, n_lots=25, seed=RANDOM_SEED + 3)
    lots.to_csv(os.path.join(RESULTS, "spc_lots.csv"), index=False)
    save(i_chart_panel(lots, ["ER", "UT", "MD"]), os.path.join(RESULTS, "spc_icharts.png"))

    print("\nAll figures and tables saved under results/")


if __name__ == "__main__":
    main()
