# Changelog

## 0.4.1 (2026-11-11)

First release on PyPI, as `marketrisk-lab` (`pip install marketrisk-lab`): the name `risklab` is
taken on PyPI by an unrelated project. The import name is unchanged (`import risklab`). No code
changes; every result is identical to 0.4.0.
- Licence metadata as an SPDX expression with `LICENSE` included, and the repository URL.
- `publish` workflow: a version bump merged to main tags, releases and uploads to PyPI.
- README links and images are absolute, so they work on PyPI.

## 0.4.0 (2026-11-01)

### Added
- M5: the `risklab.scenarios` module.
  - `Scenario` (a vectorised batch of shocks) and the `Position` protocol.
  - `full_revaluation_var(positions, scenarios, alpha)`.
  - Three scenario generators: `historical_scenarios`, `gbm_scenarios` and `filtered_historical_scenarios`.
  - `stress_test` with `DEFAULT_STRESSES`.
- `Portfolio` gains `vol_fn`, `sticky` ("strike" or "moneyness") and `underlying`, and `value()` accepts a
  `Scenario`. The defaults keep flat vol, so **existing results are unchanged** (pinned in
  `tests/test_regression.py`).
- Truncation (no look-ahead) tests for `var_es_normal`, `var_es_ewma` and `var_es_fhs`, as
  `docs/CONVENTIONS.md` requires.
- Regression tests pinning the README numbers of Studies 1, 2, 3 and 5. The studies that take tens of
  seconds are marked `slow` and excluded by default; run them with `pytest -m slow`.
- Example 04 Part B points to quant-stack study S3 for a backtest with costs and lag.

## 0.3.0

- M4: multivariate portfolio VaR with DCC-GARCH and a t-copula (`var_es_dcc`). Tagged retroactively
  at commit edd8b6c.
