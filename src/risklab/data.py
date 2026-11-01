"""Synthetic data with a known truth: GARCH-t returns and block covariance matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from risklab.garch import GarchParams

TRADING_DAYS = 252


def simulate_garch_t(n: int = 3000, params: GarchParams = GarchParams(2e-6, 0.08, 0.90), nu: float = 5.0,
                     mu: float = 0.0003, seed: int | None = 0, start: str | None = "2014-01-01") -> pd.DataFrame:
    """Daily returns with GARCH(1,1) volatility and standardised Student-t innovations.

    Returns a DataFrame with columns r (return) and sigma (true conditional vol),
    so the true VaR and ES on every day are known exactly. Business-day index
    from `start`, or a plain integer index if start is None (needed for long
    simulations: pandas < 3 dates end in 2262).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = params.unconditional_var
    for t in range(n):
        if t:
            s2[t] = params.omega + params.alpha * (r[t - 1] - mu) ** 2 + params.beta * s2[t - 1]
        r[t] = mu + np.sqrt(s2[t]) * z[t]
    idx = pd.RangeIndex(n) if start is None else pd.bdate_range(start, periods=n)
    out = pd.DataFrame({"r": r, "sigma": np.sqrt(s2)}, index=idx)
    out.attrs.update(nu=nu, mu=mu)
    return out


def _std_t_quantile(alpha: float, nu: float) -> float:
    return stats.t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)


def true_garch_t_var(sim: pd.DataFrame, alpha: float = 0.01) -> pd.Series:
    nu, mu = sim.attrs["nu"], sim.attrs["mu"]
    return -(mu + sim["sigma"] * _std_t_quantile(alpha, nu))


def true_garch_t_es(sim: pd.DataFrame, alpha: float = 0.025) -> pd.Series:
    """Closed-form ES of a standardised Student-t, scaled by the true conditional vol."""
    nu, mu = sim.attrs["nu"], sim.attrs["mu"]
    q = stats.t.ppf(alpha, nu)
    es_std_t = stats.t.pdf(q, nu) / alpha * (nu + q**2) / (nu - 1) * np.sqrt((nu - 2) / nu)
    return -mu + sim["sigma"] * es_std_t


def block_covariance(n_blocks: int = 5, block_size: int = 10, rho_within: float = 0.6,
                     rho_between: float = 0.2, vol_range=(0.15, 0.40), seed: int | None = 0) -> np.ndarray:
    """Annualised covariance with sector-like correlation blocks."""
    rng = np.random.default_rng(seed)
    n = n_blocks * block_size
    blocks = np.repeat(np.arange(n_blocks), block_size)
    corr = np.where(blocks[:, None] == blocks[None, :], rho_within, rho_between)
    np.fill_diagonal(corr, 1.0)
    vols = rng.uniform(*vol_range, n)
    return corr * np.outer(vols, vols)


def load_yfinance(tickers: list[str], start: str = "2005-01-01", end: str | None = None) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker. Pass `end` to make results reproducible."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("install the data extra: pip install -e '.[data]'") from exc
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all")
