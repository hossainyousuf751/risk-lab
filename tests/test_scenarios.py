import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risklab import (
    OptionPosition, Portfolio, Scenario, filtered_historical_scenarios, fit_garch, full_revaluation_var,
    garch_variance, gbm_scenarios, historical_scenarios, simulate_garch_t, stress_test,
)
from risklab.nonlinear import _bs

STRADDLE = [OptionPosition(100, 0.5, True, 1), OptionPosition(100, 0.5, False, 1)]


def skew(K, T):
    return 0.20 - 0.5 * np.log(np.asarray(K, dtype=float) / 100.0)


def test_scenario_path_matches_the_spot_path():
    port = Portfolio(100.0, 0.25, rate=0.03, shares=2, options=STRADDLE)
    S = np.array([80.0, 100.0, 125.0])
    via_scenario = port.value(Scenario(spot_mult=S / 100, elapsed=0.1))
    np.testing.assert_allclose(via_scenario, port.value(S, elapsed=0.1), rtol=1e-14)
    flat_fn = Portfolio(100.0, 0.25, rate=0.03, shares=2, options=STRADDLE, vol_fn=lambda K, T: 0.25)
    np.testing.assert_allclose(flat_fn.value(S, elapsed=0.1), port.value(S, elapsed=0.1), rtol=1e-14)


def test_sticky_strike_and_sticky_moneyness_by_hand():
    put = [OptionPosition(95, 0.5, False, 1)]
    S = 90.0
    strike = Portfolio(100.0, 0.2, options=put, vol_fn=skew, sticky="strike").value(S)
    money = Portfolio(100.0, 0.2, options=put, vol_fn=skew, sticky="moneyness").value(S)
    assert strike == pytest.approx(_bs(S, 95, 0.5, 0.0, skew(95, 0.5), False), rel=1e-14)
    assert money == pytest.approx(_bs(S, 95, 0.5, 0.0, skew(95 * 100 / 90, 0.5), False), rel=1e-14)
    assert money < strike  # after a fall the put is nearer the money, where the skewed vol is lower
    with pytest.raises(ValueError, match="sticky"):
        Portfolio(100.0, 0.2, sticky="delta")


def test_full_revaluation_var_of_a_share_is_the_lognormal_quantile():
    sigma, h, alpha = 0.25, 10, 0.01
    daily = sigma**2 / 252
    res = full_revaluation_var([Portfolio(100.0, sigma, shares=1)],
                               gbm_scenarios(np.array([[daily]]), 400_000, horizon_days=h, seed=1), alpha)
    s = sigma * np.sqrt(h / 252)
    exact = -100 * (np.exp(-0.5 * s**2 + s * norm.ppf(alpha)) - 1)
    assert res["var"] == pytest.approx(exact, rel=0.01)
    assert res["es"] > res["var"] and res["pnl"].shape == (400_000,)


def test_vol_shift_and_default_stresses():
    long_straddle = Portfolio(100.0, 0.2, options=STRADDLE)
    up = long_straddle.value(Scenario(vol_shift=0.05)) - long_straddle.value(Scenario())
    vega = 2 * 100 * np.sqrt(0.5) * norm.pdf(0.5 * 0.2 * np.sqrt(0.5))  # call + put, per unit vol
    assert up[0] == pytest.approx(0.05 * vega, rel=0.01)

    short_put = Portfolio(100.0, 0.2, options=[OptionPosition(90, 0.5, False, -1)])
    table = stress_test({"short put": short_put, "straddle": long_straddle})
    assert list(table.columns) == ["short put", "straddle", "total"]
    assert table.loc["spot -30%", "short put"] < table.loc["spot -10%", "short put"] < 0
    assert table.loc["vol +10", "straddle"] > 0
    np.testing.assert_allclose(table["total"], table["short put"] + table["straddle"])


def test_historical_scenarios_compound_overlapping_windows_and_map_columns():
    r = pd.DataFrame({"a": [0.1, -0.1, 0.2], "b": [0.0, 0.5, -0.5]})
    sc = historical_scenarios(r, horizon_days=2)
    np.testing.assert_allclose(sc.spot_mult, [[1.1 * 0.9, 1.5], [0.9 * 1.2, 0.75]], rtol=1e-14)
    assert sc.elapsed == pytest.approx(2 / 252)
    on_b = Portfolio(10.0, 0.2, shares=1, underlying=1)
    np.testing.assert_allclose(on_b.value(sc), [15.0, 7.5], rtol=1e-14)


def test_filtered_historical_scenarios_scale_to_todays_vol():
    sim = simulate_garch_t(3000, seed=4)
    x = sim["r"].to_numpy()
    s_now = np.sqrt(garch_variance(x - x.mean(), fit_garch(x))[-1])
    moves = filtered_historical_scenarios(sim["r"]).spot_mult[:, 0] - 1
    assert moves.std() == pytest.approx(s_now, rel=0.1)
    plain = historical_scenarios(sim["r"]).spot_mult[:, 0] - 1
    assert abs(moves.std() - plain.std()) > 0.05 * plain.std()  # rescaling did something


def test_any_object_with_value_is_a_position():
    class Digital:
        def value(self, scenario):
            return np.where(scenario.spot(0) > 1.0, 1.0, 0.0)

    res = full_revaluation_var([Digital()], Scenario(spot_mult=np.linspace(0.9, 1.1, 101)), alpha=0.25)
    assert res["var"] == pytest.approx(0.0) and res["pnl"].min() == 0.0


def test_a_vol_only_batch_has_the_length_of_its_shifts():
    sc = Scenario(vol_shift=np.array([0.0, 0.05, 0.10]))
    assert len(sc) == 3 and sc.spot(0).shape == (3,)
    straddle = Portfolio(100.0, 0.2, options=STRADDLE)
    v = straddle.value(sc)
    assert v.shape == (3,) and v[0] < v[1] < v[2]
