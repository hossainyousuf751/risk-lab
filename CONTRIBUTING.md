# Contributing to risk-lab

risk-lab has one rule that everything else follows from: **every method is validated against a
known truth.** A model ships only with a test or study where the right answer is known exactly
(simulated data, closed form, or a reference implementation) and the method is shown to hit it.

Technical rules (sign conventions, no look-ahead, dependencies, style) are in
[docs/CONVENTIONS.md](docs/CONVENTIONS.md). This file covers process and responsibilities.

## Workflow

1. **One milestone item or one fix per branch.** Name it `m3-hrp-dendrogram`, `fix-traffic-light-short-series`.
   Roadmap items (M2, M3, …) in the README are the unit of planning.
2. **Claim before you start.** Open an issue (or comment on the existing one) naming the roadmap
   item and the module you will touch. Two people never edit the same module on parallel branches.
3. **Small commits, imperative subject**: `Add dendrogram bisection to HRP`. Explain *why* in the
   body when the diff doesn't make it obvious.
4. **Open a PR to `main`.** CI must be green. At least one review from the owner of each touched
   module (see below).
5. **Squash-merge.** `main` is always releasable: tests pass and README numbers are current.

## Definition of done

A change is done when every applicable box is ticked:

- [ ] `pytest` passes locally on Python 3.10+.
- [ ] New method has a **known-truth test**: exact (closed form, reference library to ~1e-10) or
      statistical (Monte Carlo with a fixed seed and a tolerance justified in a comment).
- [ ] New public function is exported in `src/risklab/__init__.py` and listed in the README
      components table.
- [ ] Every number quoted in the README was produced by the current code. If you changed a
      method used by a study, re-run that example and update the table **in the same PR**.
- [ ] Regenerated figures in `docs/` are committed when a study's plot changes.
- [ ] Roadmap checkbox ticked when a milestone is complete.
- [ ] No new runtime dependency (see conventions). New optional dependencies go in an extra.

## Responsibilities

Each area has one owner who reviews changes to it and keeps its README section accurate.
Until the team grows, the maintainer owns everything; replace the names as people join.

| Area | Files | Owner responsibilities |
|---|---|---|
| Volatility models | `garch.py`, `var.py`, `multivariate.py` | Estimator correctness, no look-ahead in rolling forecasts, Study 1 and 5 numbers |
| Backtests | `backtests.py` | Test statistics match the cited papers; critical values documented |
| Option risk | `nonlinear.py` | Pricing and Greeks correctness, Study 3 numbers |
| Covariance & portfolios | `covariance.py`, `portfolio.py` | Estimators match references, Study 2 numbers |
| Data | `data.py` | Simulators expose the exact truth; live loaders fail with clear messages |
| Plots & docs | `plotting.py`, `docs/`, `README.md` | Figures regenerated, README tables in sync with code |
| CI & packaging | `pyproject.toml`, `.github/` | Matrix stays green, dependency floors honest |

**Maintainer** (currently: repository owner): merges PRs, cuts releases, decides roadmap order,
breaks ties.

**Reviewer**: checks the definition of done, runs the affected example if numbers changed, and
asks "what is the known truth this is tested against?" on every new method.

**Author**: owns the PR until merge, including README and figure updates.

## Releases

Bump `version` in `pyproject.toml` (semver: new public function → minor, fix → patch), tag
`vX.Y.Z` on `main`, and summarise the milestone in the GitHub release notes.

## Working with AI agents

Agents follow the same rules. [CLAUDE.md](CLAUDE.md) is their entry point. An agent must not
edit README numbers by hand. It re-runs the example and copies the output.
