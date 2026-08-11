# -*- coding: utf-8 -*-
"""Build and execute the three Korean storytelling notebooks.

Run from repo root:  python scripts/build_notebooks.py
"""

import os

import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

SETUP = """\
import os, sys
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RANDOM_SEED, SECOND_DOE_CHF3, TRUE_COEF, SPECS
from src.data_gen import simulate_experiment, simulate_replicates
from src.doe_design import first_doe, second_doe
from src.analysis import (fit_all, fit_ols, ols_report, predict,
                          standardized_effects)
from src.optimization import (best_condition, desirability, grid_search,
                              validation_table)
from src.spc import monitoring_run, reproducibility_report
from src.visualization import (desirability_contour, effect_magnitude_plot,
                               i_chart_panel, main_effects_plot,
                               tradeoff_plot)

pd.set_option("display.width", 120)
"""


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


def build(path, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "name": "python3", "display_name": "Python 3", "language": "python",
    }
    NotebookClient(nb, timeout=300, resources={"metadata": {"path": NB_DIR}}).execute()
    nbf.write(nb, path)
    print("built:", os.path.basename(path))


# ============================================================ notebook 01
nb1 = [
    md("""\
# 01. 1차 DOE — SiO₂ 식각 프로파일 문제와 실험 설계

> **주의: 이 프로젝트의 모든 데이터는 물리 모델 기반 시뮬레이션으로 생성한 것이며,
> 실제 장비/팹 데이터가 아닙니다.** (교육 프로그램 팀 프로젝트를 개인 학습용으로 처음부터 재구현)

## 문제 배경

SiO₂ 플라즈마 식각 공정에서 두 가지 프로파일 불량이 동시에 발생하는 상황을 가정한다.

| 불량 | 현상 | 메커니즘 |
|---|---|---|
| **Undercut (U/T)** | 마스크 직하부 측벽이 횡방향으로 파임 | 측벽 보호막(passivation) 부족 시 방향성이 없는 F 라디칼이 등방성으로 측벽을 공격 |
| **Micro-trench (M/D)** | 트렌치 바닥 모서리가 중앙보다 깊게 파임 | 측벽에서 반사된 이온이 바닥 모서리에 집중되어 국부 식각률 상승 |

여기에 생산성 지표인 **Etch Rate(ER)** 와 측벽각 **θ**까지 총 4개 반응을 동시에 만족해야 한다.

## 인자와 목표

- **인자 3개**: CHF₃ 유량 20~30 sccm / Pressure 15~45 mTorr / Bias Power 75~105 W
- **목표**: ER ≥ 300 nm/min, U/T ≤ 2.0 %, M/D ≤ 3.0 %, θ → 90°

세 인자 모두 부족/과다 양쪽에서 서로 다른 불량을 유발하는 **trade-off 구조**이므로,
한 인자씩 바꾸는 방식(OFAT)이 아니라 **DOE 기반 다중 반응 최적화**가 필요하다."""),
    code(SETUP),
    md("""\
## 1차 DOE 설계: 3인자 3수준 full factorial (3³ = 27 runs)

전 범위를 훑는 스크리닝 목적이므로 모든 수준 조합을 실험한다.
(1차 모델이 선형이므로 2수준으로도 충분하지만, 3수준을 쓰면 곡률 여부도 확인할 수 있다.)"""),
    code("""\
design1 = first_doe()
design1.head(9)"""),
    md("""\
## 가상 팹(virtual fab)에서 데이터 생성

`src/data_gen.py`의 ground-truth 선형 모델에 가우시안 노이즈를 더해 실험값을 만든다.
계수의 부호는 플라즈마 식각 물리를 반영한다:

- **CHF₃ ↑** → 폴리머 측벽 보호막 강화 → **U/T ↓**, F 라디칼 소모로 **ER ↓**
- **Pressure ↑** → 평균자유행로(MFP) 감소, 이온 산란 → **U/T ↑**, 모서리 집중 완화로 **M/D ↓**
- **Bias ↑** → 이온 에너지 증가 → **ER ↑**, 모서리 과식각으로 **M/D ↑**"""),
    code("""\
doe1 = simulate_experiment(design1, seed=RANDOM_SEED)
doe1"""),
    md("""\
## Main Effect Plot

인자별 수준 평균을 보면 어떤 인자가 어떤 반응을 어느 방향으로 움직이는지 보인다."""),
    code("""\
for r in ["ER", "UT", "MD"]:
    fig = main_effects_plot(doe1, r)
    plt.show()"""),
    md("""\
## 표준화 효과 크기 — 어느 인자가 지배적인가

균형 잡힌 full factorial에서는 |최고수준 평균 − 최저수준 평균| 이 곧 효과 추정치다."""),
    code("""\
eff = standardized_effects(doe1, ["ER", "UT", "MD"])
display(eff.round(3))
fig = effect_magnitude_plot(eff)
plt.show()"""),
    md("""\
## 1차 DOE 결론

| 반응 | 지배 인자 | 물리적 이유 |
|---|---|---|
| ER | **Bias** | 이온 타격 에너지를 직접 좌우 |
| M/D | **Bias** | 모서리 이온 집중도 결정 |
| U/T | **Pressure** | 이온 산란/라디칼 등방성 결정 |
| (전체) | CHF₃는 효과 최소 | 미세조정용 보조 인자 |

**핵심 발견**: M/D는 Bias를 *내려야* 좋아지고 ER은 Bias를 *올려야* 좋아진다 —
같은 인자에 상반된 요구가 걸리는 상충 구조. → **02 노트북에서 회귀 모델과 trade-off를 정량화한다.**"""),
]

# ============================================================ notebook 02
nb2 = [
    md("""\
# 02. 다중회귀 분석과 Trade-off 구조

1차 DOE 27 runs 데이터로 반응별 1차 회귀 모델을 적합하고,
계수가 ground truth를 잘 복원하는지, 그리고 Bias를 둘러싼 상충 구조를 정량적으로 확인한다."""),
    code(SETUP),
    code("""\
doe1 = simulate_experiment(first_doe(), seed=RANDOM_SEED)  # 01과 동일 시드 → 동일 데이터
report = ols_report(doe1, ["ER", "UT", "MD", "THETA"])
report"""),
    md("""\
## 적합 결과 해석

- 모든 반응에서 **R² > 0.98** — 1차(선형) 모델로 충분히 설명된다.
- THETA에 대한 Bias 계수만 p-value가 크다(참값 자체가 0) — 통계적으로도 올바르게 "효과 없음"으로 나온다.

## 추정 계수 vs. 참값(ground truth)

시뮬레이션 프로젝트의 장점: 정답을 알고 있으므로 분석 파이프라인 자체를 검증할 수 있다."""),
    code("""\
rows = []
for r in ["ER", "UT", "MD"]:
    m = fit_ols(doe1, r)
    for term in ["const", "CHF3", "P", "B"]:
        rows.append({"response": r, "term": term,
                     "true": TRUE_COEF[r]["const" if term == "const" else term],
                     "estimated": round(float(m.params[term]), 4),
                     "std_err": round(float(m.bse[term]), 4)})
comp = pd.DataFrame(rows)
comp["within_2se"] = (comp["true"] - comp["estimated"]).abs() <= 2 * comp["std_err"]
comp"""),
    md("""\
대부분의 계수가 ±2SE 안에서 참값을 복원한다 (27 runs, 노이즈 존재 조건에서 기대되는 수준).

## Bias Power를 둘러싼 상충 구조

- **ER 스펙**: ≥ 300 nm/min → Bias를 **올려야** 함 (+2.5 nm/min per W)
- **M/D 스펙**: ≤ 3 % → Bias를 **내려야** 함 (+0.12 %p per W)

같은 노브에 반대 방향의 요구가 걸려 있으므로 단일 반응 최적화로는 답이 없다."""),
    code("""\
fig = tradeoff_plot(doe1)
plt.show()"""),
    md("""\
## 결론

Bias 한 인자만으로는 ER과 M/D를 동시에 만족시킬 수 없고,
Pressure(U/T 지배)까지 얽혀 있으므로 **4개 반응을 하나의 목적함수로 묶는
Desirability 기반 다중 반응 동시 최적화**가 필요하다. → 03 노트북."""),
]

# ============================================================ notebook 03
nb3 = [
    md("""\
# 03. Desirability 최적화 → 2차 DOE → 검증 → SPC

## Derringer–Suich Desirability Function (1980)

각 반응 y를 만족도 d ∈ [0, 1]로 변환한 뒤 기하평균으로 종합 만족도 D를 만든다:

- **ER (클수록 좋음)**: d = (y − 280) / (380 − 280), [0,1] 클리핑
- **U/T, M/D (작을수록 좋음)**: d = (상한 − y) / (상한 − 하한)
- **θ (목표값 90°)**: 90°에서 d=1, ±3° 밖에서 d=0인 삼각형
- **종합**: D = (d_ER · d_UT · d_MD · d_θ)^(1/4) — 하나라도 0이면 D=0

기하평균을 쓰는 이유: 어떤 반응 하나가 완전히 불만족(d=0)인 레시피는
다른 반응이 아무리 좋아도 탈락시켜야 하기 때문이다."""),
    code(SETUP),
    code("""\
doe1 = simulate_experiment(first_doe(), seed=RANDOM_SEED)
models = fit_all(doe1, ["ER", "UT", "MD", "THETA"])

# 1차 DOE 결론에 따라 CHF3는 중앙값 25 sccm에 고정 (효과가 가장 작은 보조 인자)
grid1 = grid_search(models, fixed={"CHF3": SECOND_DOE_CHF3}, n_grid=121)
best1 = best_condition(grid1)
print(f"1차 모델 기준 최적점: P={best1['P']:.1f} mTorr, B={best1['B']:.1f} W, D={best1['D']:.3f}")
fig = desirability_contour(grid1, best1, fixed_label=f"CHF3={SECOND_DOE_CHF3:g} sccm")
plt.show()"""),
    md("""\
빗금 영역이 **4개 스펙을 모두 만족하는 feasible region**이다.
저압(P≈15)·중간 Bias(B≈75~85) 코너에 최적점이 있다 —
저압은 U/T와 θ를 살리고, Bias는 ER(≥300)과 M/D(≤3)의 타협점을 찾은 결과다.

## 2차 DOE — 최적점 주변 정밀 탐색

1차 최적점이 탐색 범위 경계(P=15) 근처라서, CHF₃=25 고정 후
**Pressure 15~25 mTorr / Bias 60~90 W** 좁은 창에서 3×3 = 9 runs를 추가로 돌린다.
(Bias는 하한을 75 → 60 W까지 내려 경계 바깥도 확인)"""),
    code("""\
doe2 = simulate_experiment(second_doe(), seed=RANDOM_SEED + 1)
doe2"""),
    code("""\
models2 = fit_all(doe2, ["ER", "UT", "MD", "THETA"])
grid2 = grid_search(models2, ranges={"P": (15.0, 25.0), "B": (60.0, 90.0)},
                    fixed={"CHF3": SECOND_DOE_CHF3}, n_grid=121)
best2 = best_condition(grid2)
final = {"CHF3": SECOND_DOE_CHF3, "P": round(best2["P"]), "B": round(best2["B"])}
print(f"최종 레시피: CHF3={final['CHF3']:g} sccm / P={final['P']:g} mTorr / B={final['B']:g} W")
fig = desirability_contour(grid2, best2, fixed_label="2nd DOE local model")
plt.show()"""),
    md("""\
## 최종 레시피 검증 — 회귀식 대입 PASS/FAIL

1차 DOE의 글로벌 모델에 최종 조건을 대입해 스펙 충족 여부를 판정한다."""),
    code("""\
pred = predict(models, final)
validation_table(pred)"""),
    md("""\
## 재현성 확인 — 반복 run (n=3)

같은 레시피로 3회 반복 시뮬레이션하여 산포(σ)와
회귀 예측값 ±3σ_noise 밴드 내 포함 여부를 확인한다."""),
    code("""\
reproducibility_report(final, pred, n=3, seed=RANDOM_SEED + 2)"""),
    md("""\
## SPC 확장 — 양산 관리도 (I-chart)

레시피 확정 후에는 관리도로 공정 드리프트를 조기에 감지한다.
25개 lot을 시뮬레이션하고 개별값 관리도(I-chart)의 UCL/LCL(= CL ± 2.66·MR̄)을 그린다."""),
    code("""\
lots = monitoring_run(final, n_lots=25, seed=RANDOM_SEED + 3)
fig = i_chart_panel(lots, ["ER", "UT", "MD"])
plt.show()"""),
    md("""\
## 최종 결론

| 항목 | 값 |
|---|---|
| **최종 레시피** | CHF₃ 25 sccm / Pressure 15 mTorr / Bias ≈ 79 W |
| ER | ≈ 311 nm/min (≥ 300 **PASS**) |
| U/T | ≈ 0.9 % (≤ 2.0 **PASS**) |
| M/D | ≈ 2.0 % (≤ 3.0 **PASS**) |
| θ | ≈ 90.7° (90 ± 3 **PASS**) |

**총 실험 횟수 36 runs** (27 + 9)로 4개 반응을 동시에 만족하는 조건을 찾았다.
OFAT로 같은 해상도를 얻으려면 인자당 별도 스윕이 필요해 실험 수가 크게 늘고,
인자 간 상충 구조(Bias의 ER↔M/D)는 아예 발견하지 못했을 것이다.

### 한계점
- 모든 데이터는 **1차 선형 ground truth 기반 시뮬레이션**이다. 실제 식각 공정은
  강한 비선형·교호작용(예: polymer 과다 시 etch stop)이 존재한다.
- 노이즈를 i.i.d. 가우시안으로 가정 — 실제 장비는 드리프트/lot 간 상관이 있다.
- 실무에서는 SEM 계측 오차, 챔버 상태(PM 주기) 등이 추가 변수가 된다."""),
]

build(os.path.join(NB_DIR, "01_first_doe.ipynb"), nb1)
build(os.path.join(NB_DIR, "02_regression_tradeoff.ipynb"), nb2)
build(os.path.join(NB_DIR, "03_optimization_and_spc.ipynb"), nb3)
