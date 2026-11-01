"""GARCH(1,1) by Gaussian quasi-maximum likelihood.

sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

QMLE is consistent for the variance parameters even when innovations are fat
tailed, which is why filtered historical simulation pairs it with the
empirical distribution of standardised residuals instead of a normal quantile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter


@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_var(self) -> float:
        return self.omega / (1 - self.persistence)


def garch_variance(r: np.ndarray, p: GarchParams, v0: float | None = None) -> np.ndarray:
    """In-sample conditional variances sigma^2_0..sigma^2_{n-1}, plus the one-step forecast.

    Returns an array of length n + 1; the last element is the forecast for the
    day after the sample.
    """
    r = np.asarray(r, dtype=float)
    v0 = float(np.var(r)) if v0 is None else v0
    x = np.empty(len(r) + 1)
    x[0] = v0
    x[1:] = p.omega + p.alpha * r**2
    return lfilter([1.0], [1.0, -p.beta], x)


def _nll(theta, r, v0):
    omega, alpha, beta = theta
    s2 = garch_variance(r, GarchParams(omega, alpha, beta), v0)[:-1]
    if np.any(s2 <= 0):
        return 1e10
    return 0.5 * np.sum(np.log(s2) + r**2 / s2)


def fit_garch(r, demean: bool = True) -> GarchParams:
    """Fit on returns scaled to percent, where all parameters are O(0.01-1).

    Daily omega is ~1e-6 in raw units, which leaves the optimiser badly
    conditioned (it can stall at the starting point). Scaling by 100 fixes this;
    omega is rescaled by 1e-4 afterwards. alpha and beta are scale-free.
    """
    r = 100 * np.asarray(r, dtype=float)
    if demean:
        r = r - r.mean()
    v = float(np.var(r))
    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.03, 0.95)):
        x0 = [v * (1 - a0 - b0), a0, b0]
        res = minimize(_nll, x0, args=(r, v), method="SLSQP",
                       bounds=[(1e-8, 10 * v), (0.0, 1.0), (0.0, 1.0)],
                       constraints=[{"type": "ineq", "fun": lambda t: 0.9999 - t[1] - t[2]}],
                       options={"ftol": 1e-10, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = map(float, best.x)
    return GarchParams(omega * 1e-4, alpha, beta)
