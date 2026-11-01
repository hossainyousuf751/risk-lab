import numpy as np
import pandas as pd
import pytest

from risklab import GarchParams, fit_dcc, simulate_dcc_t, true_garch_t_es, true_garch_t_var, var_es_dcc

W = np.array([0.5, 0.3, 0.2])


def test_dcc_recovers_parameters_from_true_shocks():
    r, H = simulate_dcc_t(6000, a=0.04, b=0.94, seed=1, start=None)
    z = (r.to_numpy() - r.attrs["mu"]) / np.sqrt(np.einsum("tii->ti", H))
    a, b = fit_dcc(z)
    assert a == pytest.approx(0.04, abs=0.015)
    assert b == pytest.approx(0.94, abs=0.03)


def test_true_portfolio_var_and_es_match_monte_carlo():
    # Constant volatility and correlation: returns are iid multivariate t, so the
    # closed-form portfolio VaR/ES must match the empirical tail of w'r.
    r, H = simulate_dcc_t(100_000, params=GarchParams(1e-4, 0.0, 0.0), a=0.0, b=0.0, nu=5,
                          mu=0.0, seed=2, start=None)
    p = r.to_numpy() @ W
    truth = pd.DataFrame({"sigma": np.sqrt(np.einsum("i,tij,j->t", W, H, W))})
    truth.attrs.update(r.attrs)
    assert true_garch_t_var(truth, 0.01).iloc[0] == pytest.approx(-np.quantile(p, 0.01), rel=0.03)
    q = np.quantile(p, 0.025)
    assert true_garch_t_es(truth, 0.025).iloc[0] == pytest.approx(-p[p <= q].mean(), rel=0.03)


@pytest.mark.parametrize("copula", [False, True])
def test_dcc_forecasts_use_only_past_data(copula):
    r, _ = simulate_dcc_t(330, seed=3)
    kw = dict(weights=W, window=300, refit_every=10, copula=copula, n_sims=2000)
    full = var_es_dcc(r, **kw)
    cut = var_es_dcc(r.iloc[:320], **kw)
    pd.testing.assert_series_equal(full["var"].iloc[:320], cut["var"])
