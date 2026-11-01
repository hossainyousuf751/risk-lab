"""Covariance estimators."""

from __future__ import annotations

import numpy as np


def sample_cov(returns) -> np.ndarray:
    return np.cov(np.asarray(returns, dtype=float), rowvar=False)


def ledoit_wolf(returns) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004) shrinkage towards a scaled identity.

    Sigma = delta * mu * I + (1 - delta) * S, with the optimal delta estimated
    from the data. Returns (covariance, delta). When there are few observations
    per asset, S is noisy and delta is large.
    """
    X = np.asarray(returns, dtype=float)
    X = X - X.mean(axis=0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    b2_bar = (np.sum(np.sum(X**2, axis=1) ** 2) - n * np.sum(S**2)) / (n**2 * p)
    b2 = min(b2_bar, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * mu * np.eye(p) + (1 - delta) * S, float(delta)


def nonlinear_shrinkage(returns) -> np.ndarray:
    """Ledoit & Wolf (2020) analytical nonlinear shrinkage.

    Keeps the sample eigenvectors and replaces each sample eigenvalue with a
    nonparametric estimate of u_i' Sigma u_i, using a kernel estimate of the
    limiting spectral density and its Hilbert transform. Unlike linear
    shrinkage, small eigenvalues are pulled up and large ones down by
    different amounts. Works for p > n.
    """
    X = np.asarray(returns, dtype=float)
    X = X - X.mean(axis=0)
    n, p = X.shape
    n -= 1                                          # one degree of freedom lost to demeaning
    lam, u = np.linalg.eigh(X.T @ X / n)            # ascending
    lam = np.clip(lam[max(0, p - n):], 0, None)     # the min(p, n) informative eigenvalues
    h = n ** (-1 / 3)
    H = h * lam[None, :]                            # locally adaptive bandwidth h * lambda_j
    x = (lam[:, None] - lam[None, :]) / H
    f = 3 / (4 * np.sqrt(5)) * np.mean(np.maximum(1 - x**2 / 5, 0) / H, axis=1)
    with np.errstate(divide="ignore"):
        hf = (-3 / (10 * np.pi)) * x + 3 / (4 * np.sqrt(5) * np.pi) * (1 - x**2 / 5) * np.log(
            np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    edge = np.isclose(np.abs(x), np.sqrt(5))
    hf[edge] = (-3 / (10 * np.pi)) * x[edge]
    Hf = np.mean(hf / H, axis=1)
    c = p / n
    if p <= n:
        d = lam / ((np.pi * c * lam * f) ** 2 + (1 - c - np.pi * c * lam * Hf) ** 2)
    else:
        Hf0 = (1 / np.pi) * (3 / (10 * h**2) + 3 / (4 * np.sqrt(5) * h) * (1 - 1 / (5 * h**2))
                             * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))) * np.mean(1 / lam)
        d0 = 1 / (np.pi * (p - n) / n * Hf0)
        d = np.concatenate([np.full(p - n, d0), lam / (np.pi**2 * lam**2 * (f**2 + Hf**2))])
    return (u * d) @ u.T


def factor_cov(returns, k: int) -> np.ndarray:
    """Statistical factor model: top-k principal components plus diagonal residual variance.

    Sigma = B B' + D, with B from the k largest eigenpairs of the sample
    covariance and D = diag(S - B B'). Estimates p*k + p parameters instead
    of p(p+1)/2, at the cost of bias when the true structure has more factors.
    """
    S = sample_cov(returns)
    lam, u = np.linalg.eigh(S)
    B = u[:, -k:] * np.sqrt(lam[-k:])
    BB = B @ B.T
    return BB + np.diag(np.diag(S - BB))
