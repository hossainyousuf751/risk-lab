import numpy as np
import pandas as pd
import pytest

from risklab import (
    acerbi_szekely_z2, backtest_var, christoffersen_independence, kupiec_pof, traffic_light,
)


def test_kupiec_accepts_correct_rate_and_rejects_wrong():
    exact = np.zeros(1000, dtype=bool)
    exact[::100] = True                               # exactly 1% exceptions
    assert kupiec_pof(exact, 0.01)[0] == pytest.approx(0.0, abs=1e-10)
    too_many = np.zeros(1000, dtype=bool)
    too_many[::25] = True                             # 4%
    assert kupiec_pof(too_many, 0.01)[1] < 0.001


def test_kupiec_rejects_empty_input():
    with pytest.raises(ValueError):
        kupiec_pof([], 0.01)


def test_kupiec_zero_exceptions():
    lr, p = kupiec_pof(np.zeros(1000, dtype=bool), 0.01)
    assert lr == pytest.approx(-2 * 1000 * np.log(0.99))


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(0)
    iid = rng.random(2000) < 0.02
    clustered = np.zeros(2000, dtype=bool)
    for start in rng.choice(1990, 8, replace=False):
        clustered[start:start + 5] = True
    assert christoffersen_independence(iid)[1] > 0.05
    assert christoffersen_independence(clustered)[1] < 0.001


def test_traffic_light_zones():
    e = np.zeros(500, dtype=bool)
    assert traffic_light(e) == (0, "green")
    e[100:106] = True
    assert traffic_light(e) == (6, "yellow")
    e[110:115] = True
    assert traffic_light(e)[1] == "red"


def test_z2_near_zero_when_es_matches_losses():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200_000)
    var, alpha = 1.959964, 0.025
    es = -x[x < -var].mean()
    assert acerbi_szekely_z2(x, np.full_like(x, es), np.full_like(x, var), alpha) == pytest.approx(0, abs=0.03)
    assert acerbi_szekely_z2(x, np.full_like(x, es / 2), np.full_like(x, var), alpha) < -0.7


def test_traffic_light_on_series_shorter_than_window():
    e = np.zeros(100, dtype=bool)
    e[:3] = True
    assert traffic_light(e) == (3, "green")


def test_backtest_var_rejects_no_overlap():
    r = pd.Series([0.01, -0.02], index=[0, 1])
    with pytest.raises(ValueError):
        backtest_var(r, pd.DataFrame({"var": [np.nan, np.nan], "method": "x"}, index=[0, 1]), 0.01)
