"""Portfolio construction from a covariance matrix.

All methods are risk-based (no expected-return inputs), because expected
returns are far harder to estimate than covariances and dominate the error
of mean-variance optimisation.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _require_positive_variance(cov: np.ndarray) -> None:
    # Risk-based weights scale with 1/risk, so a riskless asset (e.g. cash) would get infinite weight.
    if np.any(np.diag(cov) <= 0):
        raise ValueError("cov has an asset with zero variance; risk-based weights are undefined for it")


def equal_weight(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    return np.full(n, 1 / n)


def inverse_variance(cov: np.ndarray) -> np.ndarray:
    _require_positive_variance(cov)
    iv = 1 / np.diag(cov)
    return iv / iv.sum()


def min_variance(cov: np.ndarray, long_only: bool = False) -> np.ndarray:
    n = cov.shape[0]
    if not long_only:
        w = np.linalg.solve(cov, np.ones(n))
        return w / w.sum()
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w,
                   method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-14, "maxiter": 500})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fraction of portfolio variance contributed by each asset (sums to 1)."""
    return w * (cov @ w) / (w @ cov @ w)


def risk_parity(cov: np.ndarray, budget: np.ndarray | None = None) -> np.ndarray:
    """Equal (or budgeted) risk contributions via Spinu's (2013) convex formulation.

    Minimise 0.5 y'Cov y - sum(b_i ln y_i) over y > 0, then w = y / sum(y).
    """
    _require_positive_variance(cov)
    n = cov.shape[0]
    b = np.full(n, 1 / n) if budget is None else np.asarray(budget, dtype=float)
    y0 = 1 / np.sqrt(np.diag(cov))
    res = minimize(lambda y: 0.5 * y @ cov @ y - b @ np.log(y), y0,
                   jac=lambda y: cov @ y - b / y, method="L-BFGS-B",
                   bounds=[(1e-12, None)] * n, options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 5000})
    return res.x / res.x.sum()


def hrp_order(cov: np.ndarray) -> np.ndarray:
    """HRP leaf order: single-linkage clustering on correlation distance sqrt(0.5 * (1 - rho))."""
    vol = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(vol, vol), -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    return leaves_list(linkage(pdist(dist), method="single"))


def hrp(cov: np.ndarray, order: np.ndarray | None = None) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado 2016).

    1. Cluster assets and reorder so similar assets sit together (`hrp_order`).
    2. Split the ordered list recursively in halves, allocating between the
       halves in inverse proportion to their (inverse-variance) cluster risk.
    No matrix inversion is needed, so it works even when cov is singular.

    Almost all of HRP's turnover comes from the order changing between
    rebalances, not from the bisection weights (Study 2). Passing an order
    computed less often than you rebalance removes most of it.
    """
    _require_positive_variance(cov)
    order = hrp_order(cov) if order is None else order
    w = np.ones(len(order))
    clusters = [list(order)]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for left, right in zip(clusters[::2], clusters[1::2]):
            v_l, v_r = (_cluster_var(cov, c) for c in (left, right))
            a = 1 - v_l / (v_l + v_r)
            w[left] *= a
            w[right] *= 1 - a
    return w / w.sum()


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    iv = 1 / np.diag(sub)
    iv /= iv.sum()
    return float(iv @ sub @ iv)
