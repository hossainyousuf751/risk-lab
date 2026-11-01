"""Formal VaR and ES backtests.

Kupiec (1995): is the exception rate right?
Christoffersen (1998): are exceptions independent, or do they cluster?
Basel traffic light: exceptions in 250 days at 99% (green 0-4, yellow 5-9, red 10+).
Acerbi & Szekely (2014) Z2: is ES large enough given the losses beyond VaR?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def _loglik_bernoulli(n0: int, n1: int, p: float) -> float:
    return float(xlogy(n0, 1 - p) + xlogy(n1, p))


def kupiec_pof(exceptions, alpha: float) -> tuple[float, float]:
    """Proportion-of-failures likelihood ratio test. Returns (LR, p-value), chi2(1)."""
    e = np.asarray(exceptions, dtype=bool)
    n, x = len(e), int(e.sum())
    lr = -2 * (_loglik_bernoulli(n - x, x, alpha) - _loglik_bernoulli(n - x, x, x / n))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(exceptions) -> tuple[float, float]:
    """First-order Markov test of exception independence. Returns (LR, p-value), chi2(1)."""
    e = np.asarray(exceptions, dtype=int)
    prev, cur = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    lr = -2 * (_loglik_bernoulli(n00 + n10, n01 + n11, pi)
               - _loglik_bernoulli(n00, n01, pi0) - _loglik_bernoulli(n10, n11, pi1))
    return float(lr), float(stats.chi2.sf(lr, 1))


def traffic_light(exceptions, window: int = 250) -> tuple[int, str]:
    """Worst rolling-window exception count and its Basel zone (calibrated for 99% VaR).

    A series shorter than the window is scored on its full length.
    """
    worst = int(pd.Series(np.asarray(exceptions, dtype=int)).rolling(window, min_periods=1).sum().max())
    zone = "green" if worst <= 4 else "yellow" if worst <= 9 else "red"
    return worst, zone


def acerbi_szekely_z2(returns, es, var, alpha: float) -> float:
    """Z2 = sum(X_t * I_t / (T * alpha * ES_t)) + 1; ~0 if ES is right, negative if too small.

    Acerbi & Szekely report approximate critical values of -0.70 (5%) and
    -1.8 (0.01%), stable across a wide range of distributions.
    """
    x, es, var = (np.asarray(a, dtype=float) for a in (returns, es, var))
    hit = x < -var
    return float(np.sum(x[hit] / es[hit]) / (len(x) * alpha) + 1)


@dataclass
class VaRBacktest:
    method: str
    n: int
    exceptions: int
    expected: float
    kupiec_p: float
    independence_p: float
    worst_250: int
    zone: str

    @property
    def passes(self) -> bool:
        return self.kupiec_p > 0.05 and self.independence_p > 0.05


def backtest_var(returns: pd.Series, forecast: pd.DataFrame, alpha: float) -> VaRBacktest:
    df = pd.DataFrame({"r": returns, "var": forecast["var"]}).dropna()
    if df.empty:
        raise ValueError("returns and forecast have no overlapping non-NaN days")
    exc = (df["r"] < -df["var"]).to_numpy()
    worst, zone = traffic_light(exc)
    return VaRBacktest(
        method=str(forecast["method"].iloc[0]), n=len(df), exceptions=int(exc.sum()),
        expected=alpha * len(df), kupiec_p=kupiec_pof(exc, alpha)[1],
        independence_p=christoffersen_independence(exc)[1], worst_250=worst, zone=zone,
    )
