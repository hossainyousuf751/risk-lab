"""Rolling one-day VaR and ES forecasts.

Conventions: VaR and ES are positive loss numbers (fractions of value). The
forecast on day t uses returns strictly before t, so comparing it with the
return on day t is a genuine out-of-sample test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter

from risklab.garch import fit_garch, garch_variance


def _frame(var, es, index, name):
    return pd.DataFrame({"var": var, "es": es}, index=index).rename_axis(None).assign(method=name)


def var_es_historical(r: pd.Series, alpha: float = 0.01, window: int = 500) -> pd.DataFrame:
    x = r.to_numpy()
    var = np.full(len(x), np.nan)
    es = np.full(len(x), np.nan)
    for t in range(window, len(x)):
        w = x[t - window:t]
        q = np.quantile(w, alpha)
        var[t], es[t] = -q, -w[w <= q].mean()
    return _frame(var, es, r.index, "historical")


def var_es_normal(r: pd.Series, alpha: float = 0.01, window: int = 500) -> pd.DataFrame:
    mu = r.rolling(window).mean().shift(1)
    sd = r.rolling(window).std().shift(1)
    z = stats.norm.ppf(alpha)
    return _frame(-(mu + sd * z), -(mu - sd * stats.norm.pdf(z) / alpha), r.index, "normal")


def var_es_ewma(r: pd.Series, alpha: float = 0.01, lam: float = 0.94, warmup: int = 500) -> pd.DataFrame:
    """RiskMetrics: exponentially weighted variance with normal quantiles."""
    x = r.to_numpy()
    u = np.empty(len(x))
    u[0] = np.var(x[:warmup])                       # seed variance
    u[1:] = (1 - lam) * x[:-1] ** 2                 # sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) r^2_{t-1}
    s2 = lfilter([1.0], [1.0, -lam], u)
    sd = np.sqrt(s2)
    z = stats.norm.ppf(alpha)
    var, es = -sd * z, sd * stats.norm.pdf(z) / alpha
    var[:warmup] = es[:warmup] = np.nan
    return _frame(var, es, r.index, "ewma")


def var_es_fhs(r: pd.Series, alpha: float = 0.01, window: int = 1000, refit_every: int = 21) -> pd.DataFrame:
    """Filtered historical simulation (Barone-Adesi et al. 1999).

    GARCH(1,1) captures volatility clustering; the empirical distribution of
    standardised residuals captures fat tails without assuming a distribution.
    """
    x = r.to_numpy()
    var = np.full(len(x), np.nan)
    es = np.full(len(x), np.nan)
    params = None
    for t in range(window, len(x)):
        w = x[t - window:t]
        mu = w.mean()
        if params is None or (t - window) % refit_every == 0:
            params = fit_garch(w)
        s2 = garch_variance(w - mu, params)
        z = (w - mu) / np.sqrt(s2[:-1])
        q = np.quantile(z, alpha)
        sigma_next = np.sqrt(s2[-1])
        var[t] = -(mu + sigma_next * q)
        es[t] = -(mu + sigma_next * z[z <= q].mean())
    return _frame(var, es, r.index, "fhs")
