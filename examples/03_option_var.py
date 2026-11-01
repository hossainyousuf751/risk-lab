"""Study 3: delta-normal vs delta-gamma vs full-revaluation VaR for option books."""

import pandas as pd

from risklab import OptionPosition, Portfolio, option_portfolio_var

S, VOL = 100.0, 0.25
books = {
    "long 1 share": Portfolio(S, VOL, shares=1),
    "covered call (share + short 105 call)": Portfolio(S, VOL, shares=1, options=[OptionPosition(105, 0.25, True, -1)]),
    "protective put (share + long 95 put)": Portfolio(S, VOL, shares=1, options=[OptionPosition(95, 0.25, False, 1)]),
    "short ATM straddle": Portfolio(S, VOL, options=[OptionPosition(100, 0.25, True, -1), OptionPosition(100, 0.25, False, -1)]),
}
rows = []
for horizon in (1, 10):
    for name, port in books.items():
        v = option_portfolio_var(port, 0.01, horizon)
        rows.append({"horizon_days": horizon, "book": name, "delta": v["delta"], "gamma": v["gamma"],
                     "delta_normal": v["delta_normal"], "delta_gamma": v["delta_gamma_mc"],
                     "full_reval": v["full_revaluation_mc"],
                     "delta_normal_vs_full": v["delta_normal"] / v["full_revaluation_mc"]})
with pd.option_context("display.width", 160):
    print(pd.DataFrame(rows).round(3).to_string(index=False))
