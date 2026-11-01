"""Study 1: four VaR/ES methods on GARCH returns with fat tails, where the true VaR is known."""

from pathlib import Path

import pandas as pd

from risklab import (
    acerbi_szekely_z2, backtest_var, simulate_garch_t, true_garch_t_es, true_garch_t_var,
    var_es_ewma, var_es_fhs, var_es_historical, var_es_normal,
)
from risklab.plotting import plot_var_paths

DOCS = Path(__file__).resolve().parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

sim = simulate_garch_t(3000, seed=1)
r, burn = sim["r"], 1000
test = r.iloc[burn:]
methods = (var_es_historical, var_es_normal, var_es_ewma, var_es_fhs)

print("99% one-day VaR, 2,000 out-of-sample days (20 exceptions expected)")
true99 = true_garch_t_var(sim, 0.01)
rows, forecasts = [], {}
for fn in methods:
    f = fn(r, 0.01)
    bt = backtest_var(test, f.iloc[burn:], 0.01)
    err = ((f["var"] - true99).abs() / true99).iloc[burn:].mean()
    rows.append({**vars(bt), "mean_abs_error_vs_true": err})
    forecasts[bt.method] = f["var"]
truth = backtest_var(test, pd.DataFrame({"var": true99, "method": "TRUE VaR"}).iloc[burn:], 0.01)
rows.append({**vars(truth), "mean_abs_error_vs_true": 0.0})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n97.5% Expected Shortfall: Acerbi-Szekely Z2 (~0 if right; reject below -0.70)")
true_es = true_garch_t_es(sim, 0.025)
for fn in methods:
    f = fn(r, 0.025).iloc[burn:]
    print(f"  {f['method'].iloc[0]:<11} Z2 = {acerbi_szekely_z2(test, f['es'], f['var'], 0.025):+.3f}")
true_var975 = true_garch_t_var(sim, 0.025).iloc[burn:]
print(f"  {'TRUE':<11} Z2 = {acerbi_szekely_z2(test, true_es.iloc[burn:], true_var975, 0.025):+.3f}")

peak = sim["sigma"].iloc[burn:].idxmax()
window = (peak - pd.Timedelta(days=120), peak + pd.Timedelta(days=180))
plot_var_paths(r, {"historical": forecasts["historical"], "FHS": forecasts["fhs"], "true": true99},
               DOCS / "var_paths.png", "99% VaR through a volatility spike: historical vs filtered HS",
               *window)
