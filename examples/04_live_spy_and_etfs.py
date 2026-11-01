"""Study 4: real data (requires yfinance). SPY VaR backtest and a multi-asset ETF portfolio."""

import numpy as np
import pandas as pd

from risklab import (
    backtest_var, equal_weight, hrp, inverse_variance, ledoit_wolf, min_variance, risk_parity,
    var_es_ewma, var_es_fhs, var_es_historical, var_es_normal,
)
from risklab.data import load_yfinance

END = "2025-12-31"     # fixed so the numbers in the README reproduce

# Part A: SPY 99% VaR through 2008, 2020 and 2022
spy = load_yfinance(["SPY"], start="2003-01-01", end=END).squeeze().pct_change().dropna()
print("SPY 99% one-day VaR backtest (out-of-sample from 2007):")
for fn in (var_es_historical, var_es_normal, var_es_ewma, var_es_fhs):
    f = fn(spy, 0.01).loc["2007":]
    bt = backtest_var(spy.loc["2007":], f, 0.01)
    print(f"  {bt.method:<11} {bt.exceptions:>3} exceptions (expected {bt.expected:.0f}), "
          f"Kupiec p={bt.kupiec_p:.3f}, independence p={bt.independence_p:.3f}, worst 250d: {bt.worst_250} ({bt.zone})")

# Part B: multi-asset ETF portfolios, monthly rebalanced on a rolling 1-year covariance.
# This loop has no costs, no execution lag and no overfitting control. quant-stack's study S3
# (https://github.com/hossainyousuf751/quant-stack) reruns the same universe and methods through
# backtest-lab's engine with lag, costs, walk-forward and Deflated Sharpe; prefer its numbers.
tickers = ["SPY", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ"]
rets = load_yfinance(tickers, start="2007-06-01", end=END).pct_change().dropna()
methods = {"1/N": equal_weight, "inverse variance": inverse_variance, "min-var LW": None,
           "risk parity": risk_parity, "HRP": hrp}
month_ends = rets.groupby(rets.index.to_period("M")).tail(1).index
out = {m: [] for m in methods}
for start, end in zip(month_ends[12:-1], month_ends[13:]):
    window = rets.loc[:start].iloc[-252:]
    S = window.cov().to_numpy() * 252
    L = ledoit_wolf(window)[0] * 252
    hold = rets.loc[start:end].iloc[1:]
    for m, fn in methods.items():
        w = min_variance(L, long_only=True) if fn is None else fn(L if m == "risk parity" else S)
        out[m].append((1 + hold).cumprod().iloc[-1].to_numpy() @ w - 1)   # buy-and-hold within the month
res = pd.DataFrame(out, index=month_ends[13:])
print("\nMonthly-rebalanced ETF portfolios:")
print(pd.DataFrame({
    "annual_return": (1 + res).prod() ** (12 / len(res)) - 1,
    "annual_vol": res.std() * np.sqrt(12),
    "sharpe_rf0": res.mean() / res.std() * np.sqrt(12),
    "max_drawdown": ((1 + res).cumprod() / (1 + res).cumprod().cummax() - 1).min(),
}).round(3))
