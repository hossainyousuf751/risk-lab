# risk-lab: agent guide

Read [CONTRIBUTING.md](CONTRIBUTING.md) (process, definition of done) and
[docs/CONVENTIONS.md](docs/CONVENTIONS.md) (sign conventions, no look-ahead, style) before changing code.

## Commands

```bash
pip install -e ".[dev]"          # add ,data for the live-data example
pytest -q                        # ~20 s
cd examples && python 01_var_backtesting.py   # 02 takes ~1-2 min; 04 needs network
```

## Rules that are easy to break

- Never hand-edit README numbers. Re-run the example and copy its output.
- VaR/ES are positive losses; `alpha` is the tail probability.
- Rolling forecasts use data strictly before day t. Add a truncation test for any new one.
- No new runtime dependencies.
- Stay inside the module you claimed. If you need a change elsewhere, say so instead of making it.
