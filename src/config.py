"""Single source of truth for factor ranges, response targets and the
ground-truth process model coefficients.

Factors (plasma etch of SiO2, CHF3/CF4-based chemistry):
    CHF3  : CHF3 flow rate [sccm], 20-30
    P     : process pressure [mTorr], 15-45
    B     : bias power [W], 75-105

Responses:
    ER    : etch rate [nm/min]            -> maximize, spec >= 300
    UT    : undercut ratio [%]            -> minimize, spec <= 2.0
    MD    : micro-trench ratio [%]        -> minimize, spec <= 3.0
    THETA : sidewall angle [deg]          -> target 90 deg
"""

# ---------------------------------------------------------------- factors
FACTOR_NAMES = ["CHF3", "P", "B"]

FACTOR_RANGES = {
    "CHF3": (20.0, 30.0),   # sccm
    "P": (15.0, 45.0),      # mTorr
    "B": (75.0, 105.0),     # W
}

FACTOR_LABELS = {
    "CHF3": "CHF$_3$ flow [sccm]",
    "P": "Pressure [mTorr]",
    "B": "Bias power [W]",
}

# 3-level full factorial levels for the 1st (screening) DOE
FACTOR_LEVELS = {
    "CHF3": [20.0, 25.0, 30.0],
    "P": [15.0, 30.0, 45.0],
    "B": [75.0, 90.0, 105.0],
}

# ------------------------------------------------------------- responses
RESPONSE_NAMES = ["ER", "UT", "MD", "THETA"]

RESPONSE_LABELS = {
    "ER": "Etch Rate [nm/min]",
    "UT": "Undercut Ratio [%]",
    "MD": "Micro-trench Ratio [%]",
    "THETA": "Sidewall Angle [deg]",
}

# Process specification (acceptance criteria)
SPECS = {
    "ER": {"type": "min", "limit": 300.0},    # ER >= 300 nm/min
    "UT": {"type": "max", "limit": 2.0},      # UT <= 2.0 %
    "MD": {"type": "max", "limit": 3.0},      # MD <= 3.0 %
    "THETA": {"type": "target", "target": 90.0, "tol": 3.0},  # 90 +/- 3 deg
}

# ------------------------------------------------- ground-truth coefficients
# Linear model: y = b0 + b1*CHF3 + b2*P + b3*B  (+ Gaussian noise)
# Signs encode the physics — see data_gen.py docstrings.
TRUE_COEF = {
    "ER": {"const": 206.00, "CHF3": -3.00, "P": -1.20, "B": 2.50},
    "UT": {"const": 4.95, "CHF3": -0.15, "P": 0.08, "B": -0.02},
    "MD": {"const": -7.55, "CHF3": 0.03, "P": -0.05, "B": 0.12},
    # THETA: ~90 deg, weakly decreasing with pressure (ion scattering
    # degrades verticality), weakly increasing with CHF3 (sidewall
    # passivation straightens the profile).
    "THETA": {"const": 90.20, "CHF3": 0.08, "P": -0.11, "B": 0.00},
}
# THETA const chosen so that THETA = 90.2 + 0.08*(CHF3-25) - 0.11*(P-20)
# expressed in absolute terms below:
TRUE_COEF["THETA"] = {
    "const": 90.20 - 0.08 * 25.0 + 0.11 * 20.0,  # = 90.40
    "CHF3": 0.08,
    "P": -0.11,
    "B": 0.00,
}

NOISE_SIGMA = {"ER": 5.0, "UT": 0.10, "MD": 0.15, "THETA": 0.15}

# ------------------------------------------------------------ desirability
# Derringer-Suich desirability bounds (d=0 outside, d=1 fully satisfied)
DESIRABILITY = {
    "ER": {"goal": "max", "low": 280.0, "high": 380.0},
    "UT": {"goal": "min", "low": 0.5, "high": 2.0},
    "MD": {"goal": "min", "low": 1.0, "high": 3.0},
    "THETA": {"goal": "target", "target": 90.0, "low": 87.0, "high": 93.0},
}

# ----------------------------------------------------------------- 2nd DOE
# CHF3 fixed from 1st DOE conclusion; narrow refinement window
SECOND_DOE_CHF3 = 25.0
SECOND_DOE_LEVELS = {
    "P": [15.0, 20.0, 25.0],
    "B": [60.0, 75.0, 90.0],
}

RANDOM_SEED = 42
