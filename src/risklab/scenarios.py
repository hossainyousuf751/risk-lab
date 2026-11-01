"""Scenario-based VaR by full revaluation, and stress tests, for any position that can price itself.

A position only has to answer value(scenario). That covers the option books in
nonlinear.py and, through structural typing, positions defined elsewhere (a
structured note repriced by Monte Carlo). VaR and ES are positive losses of the
P&L across scenarios, so they need no linearisation: this is the full-revaluation
method the delta-normal and delta-gamma approximations try to shortcut.

Scenario generators:
  historical           realised h-day returns applied to today's spots
  gbm                  lognormal moves with a given covariance (e.g. Ledoit-Wolf)
  filtered historical  historical returns rescaled by today's GARCH vol over the vol
                       at the time (Hull & White 1998; Barone-Adesi et al. 1999)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from risklab.garch import fit_garch, garch_variance

TRADING_DAYS = 252


@dataclass(frozen=True, eq=False)
class Scenario:
    """A batch of market scenarios, shocks relative to today.

    spot_mult: multiplier of each underlying's spot, shape (n_scenarios, n_underlyings);
               a scalar or 1-D array (n_scenarios,) applies to every underlying.
    vol_shift: added to implied vols (0.05 = +5 vol points); scalar or (n_scenarios,).
    skew_shift: added implied vol per unit of log-moneyness ln(K / S) after the shock,
                so a negative value steepens the put skew; scalar or (n_scenarios,).
    elapsed:   years that pass (theta and the roll-down of the schedule).
    """

    spot_mult: float | np.ndarray = 1.0
    vol_shift: float | np.ndarray = 0.0
    skew_shift: float | np.ndarray = 0.0
    elapsed: float = 0.0

    def spot(self, i: int) -> np.ndarray:
        """Spot multipliers of underlying i, shape (n_scenarios,)."""
        m = np.asarray(self.spot_mult, dtype=float)
        return np.broadcast_to(m if m.ndim < 2 else m[:, i], (len(self),))

    def __len__(self) -> int:
        """Batch size: the longest of spot_mult (rows), vol_shift and skew_shift; scalars broadcast."""
        m = np.asarray(self.spot_mult, dtype=float)
        return max(m.shape[0] if m.ndim else 1, np.size(self.vol_shift), np.size(self.skew_shift))


class Position(Protocol):
    def value(self, scenario: Scenario) -> float | np.ndarray:
        """Value under each scenario in the batch, in currency units."""
        ...


def _values(positions, scenario: Scenario) -> np.ndarray:
    return np.stack([np.broadcast_to(np.asarray(p.value(scenario), dtype=float), (len(scenario),))
                     for p in positions])


def full_revaluation_var(positions, scenarios: Scenario, alpha: float = 0.01) -> dict:
    """VaR and ES (positive losses) of the summed P&L, revaluing every position in every scenario.

    P&L is value(scenario) - value(today), so with scenarios.elapsed > 0 it
    includes time decay, as option_portfolio_var's full revaluation does.
    Returns {"var", "es", "pnl"} with pnl of shape (n_scenarios,).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    pnl = (_values(positions, scenarios) - _values(positions, Scenario())).sum(axis=0)
    var = -float(np.quantile(pnl, alpha))
    return {"var": var, "es": -float(pnl[pnl <= -var].mean()), "pnl": pnl}


DEFAULT_STRESSES = {
    "spot -10%": Scenario(spot_mult=0.90),
    "spot -20%": Scenario(spot_mult=0.80),
    "spot -30%": Scenario(spot_mult=0.70),
    "vol +5": Scenario(vol_shift=0.05),
    "vol +10": Scenario(vol_shift=0.10),
    "spot -20%, vol +10": Scenario(spot_mult=0.80, vol_shift=0.10),
}


def stress_test(positions, named: dict[str, Scenario] | None = None) -> pd.DataFrame:
    """P&L of each position, and the total, under named single scenarios (default DEFAULT_STRESSES).

    positions is a list, or a dict whose keys name the columns.
    """
    named = DEFAULT_STRESSES if named is None else named
    if isinstance(positions, dict):
        names, pos = list(positions), list(positions.values())
    else:
        names, pos = [f"position_{i}" for i in range(len(positions))], list(positions)
    base = _values(pos, Scenario())[:, 0]
    rows = {k: _values(pos, s)[:, 0] - base for k, s in named.items()}
    out = pd.DataFrame(rows, index=names).T
    out["total"] = out.sum(axis=1)
    return out


def _compound(r: pd.DataFrame, horizon_days: int) -> np.ndarray:
    """Overlapping horizon-day gross returns, shape (n_windows, n_assets)."""
    log_growth = np.log1p(r).rolling(horizon_days).sum().dropna()
    return np.exp(log_growth.to_numpy())


def historical_scenarios(returns: pd.Series | pd.DataFrame, horizon_days: int = 1) -> Scenario:
    """Every realised overlapping horizon-day return (columns = underlyings) applied to today's spots."""
    r = returns.to_frame() if isinstance(returns, pd.Series) else returns
    return Scenario(spot_mult=_compound(r.dropna(), horizon_days), elapsed=horizon_days / TRADING_DAYS)


def gbm_scenarios(cov: np.ndarray, n_scenarios: int = 10_000, horizon_days: int = 1,
                  seed: int | None = 0) -> Scenario:
    """Lognormal spot moves over the horizon, zero drift (martingale), from a daily covariance.

    cov is the covariance of daily simple returns, e.g. sample_cov or ledoit_wolf(returns)[0].
    """
    cov = np.atleast_2d(np.asarray(cov, dtype=float)) * horizon_days
    z = np.random.default_rng(seed).standard_normal((n_scenarios, len(cov))) @ np.linalg.cholesky(cov).T
    return Scenario(spot_mult=np.exp(z - 0.5 * np.diag(cov)), elapsed=horizon_days / TRADING_DAYS)


def filtered_historical_scenarios(returns: pd.Series | pd.DataFrame, horizon_days: int = 1) -> Scenario:
    """Historical returns, each rescaled by today's GARCH(1,1) vol over the vol on its own day.

    A return from a calm year counts as a large one if today is turbulent, and vice
    versa, so the scenarios reflect current volatility while keeping the empirical
    tails and cross-asset dependence. Each asset gets its own GARCH fit.
    """
    r = (returns.to_frame() if isinstance(returns, pd.Series) else returns).dropna()
    scaled = {}
    for c in r.columns:
        x = r[c].to_numpy()
        mu = x.mean()
        s2 = garch_variance(x - mu, fit_garch(x))   # n + 1 values: the last is tomorrow's forecast
        scaled[c] = mu + (x - mu) * np.sqrt(s2[-1] / s2[:-1])
    return Scenario(spot_mult=_compound(pd.DataFrame(scaled, index=r.index), horizon_days),
                    elapsed=horizon_days / TRADING_DAYS)
