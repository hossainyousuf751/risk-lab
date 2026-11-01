"""Portfolio VaR from a multivariate model: GARCH volatilities, DCC correlations, t-copula.

Engle (2002) DCC: with standardised residuals e_t,
    Q_t = (1 - a - b) Qbar + a e_{t-1} e_{t-1}' + b Q_{t-1},   R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2.
Estimated in two steps: univariate GARCH per asset, then (a, b) by Gaussian
quasi-likelihood with Qbar targeted to the sample covariance of e.

The t-copula variant joins the empirical distribution of each asset's
residuals (as in filtered historical simulation) with a Student-t dependence
structure, so joint crashes are more likely than under a Gaussian copula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

from risklab.garch import fit_garch, garch_variance
from risklab.var import _frame


def dcc_correlations(e: np.ndarray, a: float, b: float) -> np.ndarray:
    """R_0..R_T for residuals e (T x N); the last one is the forecast for the day after the sample."""
    qbar = e.T @ e / len(e)
    x = np.empty((len(e) + 1, *qbar.shape))
    x[0] = qbar
    x[1:] = (1 - a - b) * qbar + a * e[:, :, None] * e[:, None, :]
    Q = lfilter([1.0], [1.0, -b], x, axis=0)
    d = np.sqrt(np.einsum("tii->ti", Q))
    return Q / (d[:, :, None] * d[:, None, :])


def _dcc_nll(theta, e):
    R = dcc_correlations(e, *theta)[:-1]
    _, logdet = np.linalg.slogdet(R)
    quad = np.einsum("ti,ti->t", e, np.linalg.solve(R, e[..., None])[..., 0])
    return 0.5 * np.sum(logdet + quad)


def fit_dcc(e: np.ndarray) -> tuple[float, float]:
    """(a, b) by quasi-maximum likelihood on standardised residuals e (T x N)."""
    e = np.asarray(e, dtype=float)
    best = None
    for x0 in ((0.02, 0.95), (0.05, 0.90), (0.01, 0.98)):
        res = minimize(_dcc_nll, x0, args=(e,), method="SLSQP", bounds=[(0.0, 0.5), (0.0, 0.999)],
                       constraints=[{"type": "ineq", "fun": lambda t: 0.999 - t[0] - t[1]}],
                       options={"ftol": 1e-9, "maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[0]), float(best.x[1])


def fit_t_copula_nu(e: np.ndarray, R: np.ndarray,
                    grid=(3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50)) -> float:
    """Degrees of freedom of a t-copula, by maximum likelihood on the pseudo-observations of e.

    R[t] is the correlation on day t (from DCC). Grid search: the likelihood is
    flat in nu, so finer resolution changes little.
    """
    T, k = e.shape
    u = stats.rankdata(e, axis=0) / (T + 1)
    _, logdet = np.linalg.slogdet(R)
    best_nu, best_ll = None, -np.inf
    for nu in grid:
        x = stats.t.ppf(u, nu)
        quad = np.einsum("ti,ti->t", x, np.linalg.solve(R, x[..., None])[..., 0])
        ll = np.sum(gammaln((nu + k) / 2) - gammaln(nu / 2) - 0.5 * k * np.log(nu * np.pi)
                    - 0.5 * logdet - 0.5 * (nu + k) * np.log1p(quad / nu)) - np.sum(stats.t.logpdf(x, nu))
        if ll > best_ll:
            best_nu, best_ll = nu, ll
    return float(best_nu)


def var_es_dcc(returns: pd.DataFrame, weights, alpha: float = 0.01, window: int = 1000,
               refit_every: int = 21, copula: bool = True, n_sims: int = 20_000,
               seed: int | None = 0) -> pd.DataFrame:
    """Rolling one-day VaR/ES of the portfolio `weights` from a DCC-GARCH model.

    copula=False: normal shocks, closed-form VaR from sqrt(w'Hw).
    copula=True: t-copula over each asset's empirical standardised residuals, by simulation.
    Like `var_es_fhs`, the forecast for day t uses returns strictly before t.
    """
    X = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    n, k = X.shape
    var, es = np.full(n, np.nan), np.full(n, np.nan)
    rng = np.random.default_rng(seed)
    z_a = stats.norm.ppf(alpha)
    for t in range(window, n):
        W = X[t - window:t]
        mu = W.mean(axis=0)
        refit = (t - window) % refit_every == 0
        if refit:
            garch = [fit_garch(W[:, i]) for i in range(k)]
        s2 = np.column_stack([garch_variance(W[:, i] - mu[i], garch[i]) for i in range(k)])
        e = (W - mu) / np.sqrt(s2[:-1])
        sd = np.sqrt(s2[-1])
        if refit:
            a, b = fit_dcc(e)
        R = dcc_correlations(e, a, b)
        if not copula:
            m, s = w @ mu, np.sqrt((w * sd) @ R[-1] @ (w * sd))
            var[t], es[t] = -(m + s * z_a), -(m - s * stats.norm.pdf(z_a) / alpha)
            continue
        if refit:
            nu = fit_t_copula_nu(e, R[:-1])
            # t quantiles of the plotting positions i/(T+1): mapping a simulated t-variate through this
            # grid onto the sorted residuals is the empirical quantile of its copula uniform, without
            # evaluating the t CDF (the bottleneck) on every draw.
            t_grid = stats.t.ppf(np.arange(1, window + 1) / (window + 1), nu)
        g = rng.standard_normal((n_sims, k)) @ np.linalg.cholesky(R[-1]).T
        x = g / np.sqrt(rng.chisquare(nu, (n_sims, 1)) / nu)
        e_sorted = np.sort(e, axis=0)
        shocks = np.column_stack([np.interp(x[:, i], t_grid, e_sorted[:, i]) for i in range(k)])
        pnl = (mu + sd * shocks) @ w
        q = np.quantile(pnl, alpha)
        var[t], es[t] = -q, -pnl[pnl <= q].mean()
    return _frame(var, es, returns.index, "dcc-t-copula" if copula else "dcc-normal")
