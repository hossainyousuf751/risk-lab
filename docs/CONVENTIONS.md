# Technical conventions

Rules every module follows. If code and this file disagree, fix one of them in the same PR.

## Finance conventions

- **VaR and ES are positive loss numbers.** A 1% VaR of 0.02 means "lose more than 2% with 1%
  probability". Exceptions are `return < -VaR`.
- **`alpha` is the tail probability** (0.01 for 99% VaR), never the confidence level.
- **Returns are simple daily fractions** (0.01 = 1%), not percent, not log, unless a docstring says
  otherwise. Internal rescaling (e.g. `fit_garch` works in percent) is undone before returning.
- **Covariances**: functions take whatever frequency they're given and return it unchanged.
  Annualisation (`* 252`, `TRADING_DAYS` in `data.py`) happens in the caller.
- **Portfolio weights sum to 1.** Long-only methods return `w >= 0`.

## No look-ahead

Any rolling forecast indexed at day *t* uses only data strictly before *t*. Every new rolling
method gets a truncation test like `test_forecasts_use_only_past_data`: the forecast computed on
a truncated series must equal the forecast computed on the full one.

## Testing

- Every method has a known-truth test (see CONTRIBUTING). Prefer exact checks; when Monte Carlo
  is unavoidable, fix the seed and state why the tolerance is what it is.
- Tests are plain `pytest` functions with `assert` or `pytest.approx`. Use a module-scoped fixture
  only for expensive shared data.
- The full suite should stay under ~60 s. Long studies belong in `examples/`, not tests.
- Optional dependencies in tests use `pytest.importorskip`.

## Code style

- Python ≥ 3.10, `from __future__ import annotations` at the top of every module.
- NumPy/SciPy vectorisation over Python loops, except where the recursion is inherently
  sequential (rolling refits).
- Module docstring states the model and the paper it comes from. Function docstrings state units
  and return shape when not obvious.
- Lines ≤ 110 characters. Names follow the maths (`S`, `mu`, `sigma`, `delta`) when that's clearer
  than English.
- Validate inputs only at public boundaries, with a `ValueError` that names the bad argument.
- No new abstractions until there is a second use for them.

## Dependencies

- Runtime: `numpy`, `scipy`, `pandas` only. Anything else is an optional extra (`plot`, `data`)
  imported inside the function that needs it, with an `ImportError` that says which extra to install.
- `scikit-learn` and other reference libraries are `dev` only, used to validate our implementations.

## Public API

- Public names are exported from `risklab/__init__.py` and listed in `__all__` (alphabetical).
- Renaming or removing a public name is a breaking change and needs a minor version bump while
  we are on 0.x.

## Examples and figures

- `examples/NN_name.py` is a self-contained study: fixed seeds, prints the table quoted in the
  README, writes figures to `docs/`.
- Only `04_*` touches the network. Synthetic studies must run offline.
