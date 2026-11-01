"""Study 5: portfolio VaR from a multivariate model (DCC-GARCH, t-copula) vs portfolio-level models.

Three assets with GARCH volatilities, DCC correlations and multivariate-t shocks.
Because the shocks are elliptical, the true VaR and ES of the portfolio are known
in closed form every day.
"""

import numpy as np
import pandas as pd

from risklab import (
    backtest_var, simulate_dcc_t, true_garch_t_es, true_garch_t_var, var_es_dcc, var_es_fhs,
    var_es_historical,
)

W = np.array([0.5, 0.3, 0.2])
r, H = simulate_dcc_t(3000, seed=1)
p = r @ W
truth = pd.DataFrame({"sigma": np.sqrt(np.einsum("i,tij,j->t", W, H, W))}, index=r.index)
truth.attrs.update(r.attrs)
true_var, true_es = true_garch_t_var(truth, 0.01), true_garch_t_es(truth, 0.01)

burn = 1000
test = p.iloc[burn:]
forecasts = [var_es_historical(p, 0.01), var_es_fhs(p, 0.01),
             var_es_dcc(r, W, 0.01, copula=False), var_es_dcc(r, W, 0.01, copula=True)]
rows = []
for f in forecasts:
    f = f.iloc[burn:]
    bt = backtest_var(test, f, 0.01)
    rows.append({**vars(bt),
                 "var_error_vs_true": ((f["var"] - true_var).abs() / true_var).iloc[burn:].mean(),
                 "es_error_vs_true": ((f["es"] - true_es).abs() / true_es).iloc[burn:].mean()})
bt = backtest_var(test, pd.DataFrame({"var": true_var, "method": "TRUE"}).iloc[burn:], 0.01)
rows.append({**vars(bt), "var_error_vs_true": 0.0, "es_error_vs_true": 0.0})
print("99% one-day portfolio VaR and ES, 2,000 out-of-sample days (20 exceptions expected)")
with pd.option_context("display.width", 160):
    print(pd.DataFrame(rows).round(3).to_string(index=False))
