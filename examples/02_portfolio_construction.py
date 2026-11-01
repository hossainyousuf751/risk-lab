"""Study 2: how estimation error degrades each portfolio method.

The true covariance is known, so each portfolio's real risk is computed
exactly (w' Sigma w), with no noise from realised returns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from risklab import (
    block_covariance, equal_weight, factor_cov, hrp, hrp_order, inverse_variance, ledoit_wolf, min_variance,
    nonlinear_shrinkage, risk_parity, sample_cov,
)
from risklab.plotting import plot_method_lines

DOCS = Path(__file__).resolve().parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

TRUE = block_covariance(n_blocks=5, block_size=10, rho_within=0.6, rho_between=0.2, seed=0)
N = TRUE.shape[0]
METHODS = {                                     # E maps estimator name -> annualised covariance
    "1/N": lambda E: equal_weight(E["sample"]),
    "inverse variance": lambda E: inverse_variance(E["sample"]),
    "min-var (sample)": lambda E: min_variance(E["sample"]),
    "min-var (Ledoit-Wolf)": lambda E: min_variance(E["lw"]),
    "min-var (nonlinear)": lambda E: min_variance(E["nl"]),
    "min-var (5-factor)": lambda E: min_variance(E["factor"]),
    "min-var LW long-only": lambda E: min_variance(E["lw"], long_only=True),
    "risk parity (LW)": lambda E: risk_parity(E["lw"]),
    "HRP (sample)": lambda E: hrp(E["sample"]),
    "HRP, yearly re-cluster": lambda E: hrp(E["sample"], order=order),     # order refreshed every 12 rebalances
}
order = hrp_order(TRUE)
oracle = {m: fn(dict.fromkeys(("sample", "lw", "nl", "factor"), TRUE)) for m, fn in METHODS.items()}

rng = np.random.default_rng(0)
daily = rng.multivariate_normal(np.zeros(N), TRUE / 252, size=252 * 12)
rows = []
for window in (60, 126, 252, 504):
    prev, stats = {}, {m: {"vol": [], "turnover": [], "eff_n": []} for m in METHODS}
    deltas = []
    for i, t in enumerate(range(504, len(daily), 21)):         # monthly rebalancing
        X = daily[t - window:t]
        S = sample_cov(X) * 252
        if i % 12 == 0:
            order = hrp_order(S)
        L, delta = ledoit_wolf(X)
        deltas.append(delta)
        E = {"sample": S, "lw": L * 252, "nl": nonlinear_shrinkage(X) * 252, "factor": factor_cov(X, 5) * 252}
        for m, fn in METHODS.items():
            w = fn(E)
            stats[m]["vol"].append(np.sqrt(w @ TRUE @ w))
            stats[m]["eff_n"].append(1 / np.sum(w**2))
            if m in prev:
                stats[m]["turnover"].append(np.abs(w - prev[m]).sum())
            prev[m] = w
    for m, s in stats.items():
        own = np.sqrt(oracle[m] @ TRUE @ oracle[m])
        rows.append({"window": window, "method": m, "true_vol": np.mean(s["vol"]),
                     "oracle_vol": own, "estimation_cost_pct": 100 * (np.mean(s["vol"]) / own - 1),
                     "monthly_turnover": np.mean(s["turnover"]), "effective_n": np.mean(s["eff_n"]),
                     "lw_shrinkage": np.mean(deltas)})

table = pd.DataFrame(rows)
with pd.option_context("display.width", 140):
    print(table.round(3).to_string(index=False))
plot_method_lines(table, "window", "true_vol", DOCS / "portfolio_true_vol.png",
                  "True risk of each portfolio vs. length of estimation window",
                  "true annualised volatility", float(np.sqrt(oracle["min-var (sample)"] @ TRUE @ oracle["min-var (sample)"])),
                  "true minimum-variance portfolio")
plot_method_lines(table, "window", "monthly_turnover", DOCS / "portfolio_turnover.png",
                  "Monthly turnover: how much noise each method trades on", "one-way turnover per month")
