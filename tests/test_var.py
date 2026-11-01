import numpy as np
import pandas as pd
import pytest

from risklab import (
    GarchParams, backtest_var, fit_garch, simulate_garch_t, true_garch_t_es, true_garch_t_var,
    var_es_ewma, var_es_fhs, var_es_historical, var_es_normal,
)


@pytest.fixture(scope="module")
def sim():
    return simulate_garch_t(3000, seed=1)


def test_garch_fit_recovers_parameters():
    sim = simulate_garch_t(20_000, params=GarchParams(2e-6, 0.08, 0.90), nu=8, seed=2)
    p = fit_garch(sim["r"].to_numpy())
    assert p.alpha == pytest.approx(0.08, abs=0.02)
    assert p.beta == pytest.approx(0.90, abs=0.03)


def test_garch_rejects_constant_returns():
    with pytest.raises(ValueError, match="zero variance"):
        fit_garch(np.zeros(100))


def test_true_var_and_es_match_monte_carlo():
    sim = simulate_garch_t(400_000, params=GarchParams(1e-4, 0.0, 0.0), nu=5, mu=0.0, seed=3, start=None)
    r = sim["r"]
    assert true_garch_t_var(sim, 0.01).iloc[0] == pytest.approx(-np.quantile(r, 0.01), rel=0.02)
    q = np.quantile(r, 0.025)
    assert true_garch_t_es(sim, 0.025).iloc[0] == pytest.approx(-r[r <= q].mean(), rel=0.03)


def test_forecasts_use_only_past_data(sim):
    r = sim["r"]
    full = var_es_historical(r, 0.01, 250)
    cut = var_es_historical(r.iloc[:1500], 0.01, 250)
    pd.testing.assert_series_equal(full["var"].iloc[:1500], cut["var"])


@pytest.mark.parametrize("method, kwargs", [(var_es_normal, {"window": 250}), (var_es_ewma, {"warmup": 250}),
                                            (var_es_fhs, {"window": 500, "refit_every": 50})])
def test_parametric_forecasts_use_only_past_data(sim, method, kwargs):
    """The same truncation test for the other rolling methods (docs/CONVENTIONS.md, No look-ahead)."""
    r = sim["r"].iloc[:900]
    full = method(r, 0.01, **kwargs)
    cut = method(r.iloc[:700], 0.01, **kwargs)
    pd.testing.assert_frame_equal(full.iloc[:700], cut)


def test_true_var_passes_and_normal_fails(sim):
    r = sim["r"].iloc[1000:]
    truth = pd.DataFrame({"var": true_garch_t_var(sim, 0.01), "method": "true"}).iloc[1000:]
    assert backtest_var(r, truth, 0.01).passes
    assert not backtest_var(r, var_es_normal(sim["r"], 0.01).iloc[1000:], 0.01).passes


def test_fhs_passes_on_garch_t_data(sim):
    f = var_es_fhs(sim["r"], 0.01)
    assert backtest_var(sim["r"].iloc[1000:], f.iloc[1000:], 0.01).passes
