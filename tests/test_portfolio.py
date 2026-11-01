import numpy as np
import pytest

from risklab import (
    Portfolio, OptionPosition, block_covariance, equal_weight, factor_cov, hrp, hrp_order, inverse_variance,
    ledoit_wolf, min_variance, nonlinear_shrinkage, option_portfolio_var, risk_contributions, risk_parity,
)

COV = block_covariance(n_blocks=3, block_size=4, seed=1)


def test_ledoit_wolf_matches_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    X = np.random.default_rng(0).multivariate_normal(np.zeros(12), COV / 252, 80)
    ours, delta = ledoit_wolf(X)
    ref = sk.LedoitWolf().fit(X)
    np.testing.assert_allclose(ours, ref.covariance_, rtol=1e-10)
    assert delta == pytest.approx(ref.shrinkage_)


def _rel_loss(est, truth):
    return np.linalg.norm(est - truth) / np.linalg.norm(truth)


@pytest.mark.parametrize("n", [40, 120])                      # p = 50: both p > n and p < n
def test_nonlinear_shrinkage_beats_linear_and_approaches_oracle(n):
    truth = block_covariance(seed=0) / 252
    losses = []
    for seed in range(10):
        X = np.random.default_rng(seed).multivariate_normal(np.zeros(50), truth, n)
        _, u = np.linalg.eigh(np.cov(X, rowvar=False))
        oracle = (u * np.einsum("ji,jk,ki->i", u, truth, u)) @ u.T   # best estimator with sample eigenvectors
        losses.append([_rel_loss(E, truth) for E in
                       (np.cov(X, rowvar=False), ledoit_wolf(X)[0], nonlinear_shrinkage(X), oracle)])
    sample, lw, nl, oracle = np.mean(losses, axis=0)
    assert oracle < nl < lw <= sample


def test_factor_cov_recovers_exact_factor_model():
    rng = np.random.default_rng(0)
    B = rng.normal(0, 0.02, (12, 3))
    truth = B @ B.T + np.diag(rng.uniform(1e-5, 4e-5, 12))
    X = rng.multivariate_normal(np.zeros(12), truth, 50_000)
    assert _rel_loss(factor_cov(X, 3), truth) < 0.02
    assert _rel_loss(factor_cov(X, 12), np.cov(X, rowvar=False)) < 1e-12   # k = p is the sample covariance


def test_min_variance_is_minimal():
    w = min_variance(COV)
    assert w.sum() == pytest.approx(1)
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = rng.normal(size=12)
        v /= v.sum()
        assert w @ COV @ w <= v @ COV @ v + 1e-12


def test_long_only_min_variance():
    w = min_variance(COV, long_only=True)
    assert (w >= 0).all() and w.sum() == pytest.approx(1)
    assert w @ COV @ w >= min_variance(COV) @ COV @ min_variance(COV) - 1e-12


def test_risk_parity_equalises_contributions():
    rc = risk_contributions(risk_parity(COV), COV)
    np.testing.assert_allclose(rc, 1 / 12, atol=1e-6)


@pytest.mark.parametrize("fn", [equal_weight, inverse_variance, hrp, risk_parity])
def test_long_only_methods_are_valid(fn):
    w = fn(COV)
    assert (w > 0).all() and w.sum() == pytest.approx(1)


@pytest.mark.parametrize("fn", [inverse_variance, risk_parity, hrp])
def test_risk_based_methods_reject_riskless_asset(fn):
    with pytest.raises(ValueError, match="zero variance"):
        fn(np.diag([0.0, 0.04, 0.09]))


def test_hrp_works_on_singular_covariance():
    X = np.random.default_rng(2).normal(size=(10, 20))     # 10 obs, 20 assets: rank-deficient
    w = hrp(np.cov(X, rowvar=False))
    assert np.isfinite(w).all() and w.sum() == pytest.approx(1)


def test_hrp_on_uncorrelated_assets_is_inverse_variance():
    cov = np.diag([0.01, 0.04, 0.09, 0.16])
    np.testing.assert_allclose(hrp(cov), inverse_variance(cov), rtol=1e-10)
    # Bisection of uncorrelated assets gives inverse variance for any order.
    np.testing.assert_allclose(hrp(cov, order=np.array([2, 0, 3, 1])), inverse_variance(cov), rtol=1e-10)


def test_hrp_with_fixed_order_matches_default():
    np.testing.assert_allclose(hrp(COV, order=hrp_order(COV)), hrp(COV))


def test_delta_normal_is_exact_for_linear_and_fails_for_short_straddle():
    stock = option_portfolio_var(Portfolio(100, 0.25, shares=1))
    assert stock["delta_normal"] == pytest.approx(stock["full_revaluation_mc"], rel=0.03)
    straddle = Portfolio(100, 0.25, options=[OptionPosition(100, 0.25, True, -1),
                                             OptionPosition(100, 0.25, False, -1)])
    v = option_portfolio_var(straddle, horizon_days=10)
    assert v["full_revaluation_mc"] > 4 * v["delta_normal"]


def test_option_var_rejects_horizon_past_maturity():
    port = Portfolio(100, 0.25, options=[OptionPosition(100, 0.02, True, 1)])
    with pytest.raises(ValueError):
        option_portfolio_var(port, horizon_days=10)
