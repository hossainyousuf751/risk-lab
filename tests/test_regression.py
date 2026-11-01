"""Pin the README headline numbers to the offline examples (same seeds and inputs)."""

import numpy as np
import pandas as pd
import pytest

from risklab import (
    OptionPosition, Portfolio, backtest_var, block_covariance, equal_weight, hrp, ledoit_wolf, min_variance,
    option_portfolio_var, sample_cov, simulate_dcc_t, simulate_garch_t, true_garch_t_var, var_es_dcc,
    var_es_ewma, var_es_fhs, var_es_historical, var_es_normal,
)


def approx(x):
    return pytest.approx(x, rel=1e-10)


def test_study1_var_backtests():
    """examples/01_var_backtesting.py: simulate_garch_t(3000, seed=1), 1,000-day burn-in."""
    sim = simulate_garch_t(3000, seed=1)
    r, burn = sim["r"], 1000
    true99 = true_garch_t_var(sim, 0.01)
    want = {"historical": (36, 0.001226307003358386, 0.028271531068299226, 0.4092864647374442),
            "normal": (49, 3.810734703198621e-08, 0.03668143866273549, 0.311441058600018),
            "ewma": (32, 0.013118145076341198, 0.5388447383132331, 0.10975592740435473),
            "fhs": (28, 0.08997547619700617, 0.40837386335463965, 0.0692953226671849)}
    for fn in (var_es_historical, var_es_normal, var_es_ewma, var_es_fhs):
        f = fn(r, 0.01)
        bt = backtest_var(r.iloc[burn:], f.iloc[burn:], 0.01)
        err = ((f["var"] - true99).abs() / true99).iloc[burn:].mean()
        assert (bt.exceptions, bt.kupiec_p, bt.independence_p, err) == approx(want[bt.method])


def test_study3_option_var():
    """examples/03_option_var.py: delta-normal, delta-gamma and full revaluation, seed 0."""
    books = {
        "share": Portfolio(100.0, 0.25, shares=1),
        "cc": Portfolio(100.0, 0.25, shares=1, options=[OptionPosition(105, 0.25, True, -1)]),
        "pp": Portfolio(100.0, 0.25, shares=1, options=[OptionPosition(95, 0.25, False, 1)]),
        "straddle": Portfolio(100.0, 0.25, options=[OptionPosition(100, 0.25, True, -1),
                                                    OptionPosition(100, 0.25, False, -1)]),
    }
    want = {
        (1, "share"): (3.663653534366386, 3.6261934008883756, 3.6261934008885817),
        (1, "cc"): (2.30251757278766, 2.4778284423869246, 2.441115803236689),
        (1, "pp"): (2.498012062649234, 2.2848326875729894, 2.312573620285141),
        (1, "straddle"): (0.18257357756761303, 0.632698790536794, 0.5450500259521589),
        (10, "share"): (11.585489726323747, 11.099194968393183, 11.099194968393814),
        (10, "cc"): (7.281199882571536, 8.838582951828498, 8.469083780104699),
        (10, "pp"): (7.899407740546806, 5.8099061455060745, 5.923991667059094),
        (10, "straddle"): (0.5773483456790813, 5.57490266515022, 4.572640739324891),
    }
    for (horizon, name), expected in want.items():
        v = option_portfolio_var(books[name], 0.01, horizon)
        assert (v["delta_normal"], v["delta_gamma_mc"], v["full_revaluation_mc"]) == approx(expected)


@pytest.mark.slow
def test_study2_portfolio_construction_window_60():
    """examples/02_portfolio_construction.py, 60-day window (methods independent of HRP re-clustering)."""
    true = block_covariance(n_blocks=5, block_size=10, rho_within=0.6, rho_between=0.2, seed=0)
    daily = np.random.default_rng(0).multivariate_normal(np.zeros(len(true)), true / 252, size=252 * 12)
    vols, deltas = {"1/N": [], "minvar_lw": [], "hrp": []}, []
    for t in range(504, len(daily), 21):
        X = daily[t - 60:t]
        S = sample_cov(X) * 252
        L, d = ledoit_wolf(X)
        deltas.append(d)
        for k, w in (("1/N", equal_weight(S)), ("minvar_lw", min_variance(L * 252)), ("hrp", hrp(S))):
            vols[k].append(np.sqrt(w @ true @ w))
    got = [np.mean(vols[k]) for k in ("1/N", "minvar_lw", "hrp")] + [np.mean(deltas)]
    assert got == approx([0.15134710953396244, 0.08905199743180546, 0.12803270007333925, 0.16341523771591251])


@pytest.mark.slow
def test_study5_multivariate_var():
    """examples/05_multivariate_var.py: simulate_dcc_t(3000, seed=1), weights 0.5 / 0.3 / 0.2."""
    w = np.array([0.5, 0.3, 0.2])
    r, _ = simulate_dcc_t(3000, seed=1)
    p = r @ w
    for f, want in ((var_es_fhs(p, 0.01), (23, 0.510229163520435)),
                    (var_es_dcc(r, w, 0.01, copula=True), (20, 1.0))):
        bt = backtest_var(p.iloc[1000:], f.iloc[1000:], 0.01)
        assert (bt.exceptions, bt.kupiec_p) == approx(want)
    assert isinstance(p, pd.Series)
