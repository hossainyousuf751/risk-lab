"""VaR for option portfolios: delta-normal vs delta-gamma vs full revaluation.

Parametric VaR assumes P&L is linear in the underlying. For options it is not:
a short straddle has almost no delta but large negative gamma, so delta-normal
VaR reports almost no risk on a position that can lose a great deal.

With vol_fn a Portfolio prices each option off a volatility smile instead of one
flat vol. When spot moves, the smile can stay put (sticky strike: an option keeps
the vol of its strike) or move with spot (sticky moneyness: an option takes the vol
of its new moneyness K / S). After a fall on a downward-sloping skew, sticky strike
leaves every option's vol unchanged, while sticky moneyness lowers the vol of puts
(now closer to the money) and raises that of calls; Derman (1999) discusses both.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from risklab.scenarios import Scenario


def _bs(S, K, T, r, vol, is_call):
    S = np.asarray(S, dtype=float)
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if is_call:
        return S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    return K * np.exp(-r * T) * ndtr(-d2) - S * ndtr(-d1)


@dataclass(frozen=True)
class OptionPosition:
    strike: float
    maturity: float
    is_call: bool
    quantity: float          # negative = short


@dataclass
class Portfolio:
    """Shares and European options on one underlying.

    vol_fn(K, T) (optional) gives the implied vol of strike K and maturity T
    given today's spot; flat `vol` otherwise. sticky is "strike" or "moneyness"
    (see the module docstring). underlying is this book's column in a
    multi-asset Scenario. value(S, elapsed) prices at spots S; value(scenario)
    prices a Scenario batch, which can also shift and tilt the vols.
    """

    spot: float
    vol: float
    rate: float = 0.0
    shares: float = 0.0
    options: list[OptionPosition] = field(default_factory=list)
    vol_fn: Callable | None = None
    sticky: str = "strike"
    underlying: int = 0

    def __post_init__(self) -> None:
        if self.sticky not in ("strike", "moneyness"):
            raise ValueError(f"sticky must be 'strike' or 'moneyness', got {self.sticky!r}")

    def value(self, S, elapsed: float = 0.0):
        if isinstance(S, Scenario):
            return self._value(self.spot * S.spot(self.underlying), S.elapsed, S.vol_shift, S.skew_shift)
        if self.vol_fn is None:
            v = self.shares * np.asarray(S, dtype=float)
            for o in self.options:
                v = v + o.quantity * _bs(S, o.strike, o.maturity - elapsed, self.rate, self.vol, o.is_call)
            return v
        return self._value(S, elapsed, 0.0, 0.0)

    def _value(self, S, elapsed, vol_shift, skew_shift):
        S = np.asarray(S, dtype=float)
        v = self.shares * S
        for o in self.options:
            T = o.maturity - elapsed
            if self.vol_fn is None:
                vol = self.vol
            elif self.sticky == "strike":
                vol = self.vol_fn(o.strike, o.maturity)
            else:
                vol = self.vol_fn(o.strike * self.spot / S, o.maturity)
            # ponytail: the smile is read at the original maturity, not rolled down by elapsed; fine for
            # VaR horizons of days, re-read at T if horizons reach a meaningful share of maturity.
            vol = np.maximum(vol + vol_shift + skew_shift * np.log(o.strike / S), 1e-4)
            v = v + o.quantity * _bs(S, o.strike, T, self.rate, vol, o.is_call)
        return v

    def greeks(self, h: float = 1e-3) -> tuple[float, float]:
        dS = self.spot * h
        up, mid, dn = self.value(self.spot + dS), self.value(self.spot), self.value(self.spot - dS)
        return float((up - dn) / (2 * dS)), float((up - 2 * mid + dn) / dS**2)


def option_portfolio_var(port: Portfolio, alpha: float = 0.01, horizon_days: int = 1,
                         n_sims: int = 200_000, seed: int | None = 0) -> dict:
    """VaR (positive loss, currency units) by three methods over the horizon."""
    h = horizon_days / 252
    if any(o.maturity <= h for o in port.options):
        raise ValueError(f"horizon_days={horizon_days} reaches or passes an option's maturity")
    delta, gamma = port.greeks()
    sd_dS = port.spot * port.vol * np.sqrt(h)
    delta_normal = -norm.ppf(alpha) * abs(delta) * sd_dS

    rng = np.random.default_rng(seed)
    S_h = port.spot * np.exp(-0.5 * port.vol**2 * h + port.vol * np.sqrt(h) * rng.standard_normal(n_sims))
    dS = S_h - port.spot
    pnl_dg = delta * dS + 0.5 * gamma * dS**2
    pnl_full = port.value(S_h, elapsed=h) - port.value(port.spot)
    return {
        "delta": delta, "gamma": gamma,
        "delta_normal": float(delta_normal),
        "delta_gamma_mc": float(-np.quantile(pnl_dg, alpha)),
        "full_revaluation_mc": float(-np.quantile(pnl_full, alpha)),
    }
