# AGENTS.md

Instructions for Claude Code and other coding agents contributing to
`embodied-dose-response` (import name `edr`). Read this before making changes —
it covers setup, testing, and the invariants that are easy to break silently.

## What this repo is

Dose-response failure-threshold estimation for embodied driving policies under
graded, physically-grounded degradation of the ego-state input. See
[README.md](README.md) for the research question and a glossary of terms
(`edr`, σ\*, ED_p, meta-action, etc.), and [ROADMAP.md](ROADMAP.md) for the
stage-by-stage plan and what's currently blocking what. Code comments
sometimes cite a planning doc by section (e.g. "PLAN.md 2.1") for deeper
design rationale beyond the roadmap. If a `PLAN.md` exists under a
`docs`-style directory in your checkout, you may check it for that. **It
isn't published with the repo, so it usually won't be there — if it's
absent, don't guess at what it says or assume undocumented rationale.** Code
comments and open issues are written to stand on their own without it.

**Status: pre-Stage-0 scaffold.** Most of `src/edr/{data,runner,metrics,analysis}`
and all of `packages/doseresponse-scorecard` are pinned-signature stubs that
`raise NotImplementedError`. This is deliberate, not unfinished work left for
you to fill in — see "Stubs" below before implementing one.

## The one rule

The GPU plane emits raw artifacts; the analysis plane scores them. GPU jobs
write a predicted trajectory, reasoning text, timings, and hashes — nothing
derived. Meta-action extraction, displacement error, and predicates all run
post-hoc on CPU against those artifacts, so recalibrating the extractor never
costs a re-inference. This is enforced by
`tests/test_stubs.py::test_analysis_plane_imports_without_torch`, which fails
if importing `edr.schema`, `edr.seeding`, `edr.perturb`, `edr.metrics`, or
`edr.analysis` drags in `torch`. Never have an analysis-side module import
`edr.runner` (or anything that imports it).

## Setup

This is a `uv` workspace with two members: the root `edr` package and
`packages/doseresponse-scorecard`.

| Where | Command |
|---|---|
| Default dev environment | `uv sync --extra analysis --group dev` |
| Laptop, job submission only | `uv sync --extra cloud` |
| GPU host (Linux only) | `uv sync --extra gpu --extra analysis` |
| Matches CI | `uv sync --locked --extra analysis --group dev` |

Use the first row unless you're specifically working on the SageMaker
submission path or runner internals. `--extra gpu` installs nothing on macOS
by design — every GPU dependency carries a `sys_platform == 'linux'` marker,
because torch has published no macOS x86_64 wheel since 2.2.2. Don't add a
GPU-plane dependency without that marker; an unmarked one makes `uv lock`
unsolvable on an Intel Mac dev machine.

## Test, lint, format

```bash
uv run pytest              # tests/ and packages/doseresponse-scorecard/tests
uv run ruff check .
uv run ruff format --check .
```

All three run in CI (`.github/workflows/ci.yml`) against `--extra analysis`
only — no GPU, no gated dataset, no model weights. Keep it that way: a test
that needs any of those belongs behind a skip, not in the default run.

Run all three locally before opening a PR. `ruff format` (without `--check`)
will fix formatting for you; `ruff format --check` is what CI enforces.

## Stubs: read before you implement one

**`edr.data`, `edr.runner`, `edr.metrics`, `edr.analysis`** (the
`STUB_PACKAGES` in `tests/test_stubs.py`): every callable has a pinned
signature and a docstring explaining decisions already made about it, but its
body is `raise NotImplementedError(...)`. This shape is enforced directly —
`test_stub_callables_raise_not_implemented` fails if a stub in one of these
packages *doesn't* raise, and `test_all_annotations_resolve` fails on a
typo'd type or broken forward ref, since the signature is the entire value of
an unimplemented stub.

**`packages/doseresponse-scorecard`** is a stub package too (per its own
README and docstrings), but it has no equivalent automated raise-check —
nothing currently fails CI if you fill in one of its functions without
updating anything else. Treat the convention as binding anyway.

Implications for you, both cases:

- **Don't "helpfully" fill in a stub's body as a side effect of an unrelated
  change.** Most are blocked on something explicit (gated dataset access,
  model weights, an open design question named in the docstring) — read the
  docstring, it says what.
- If you *are* asked to implement one, update the surrounding tests in the
  same change: for `edr` stubs, remove that module from `STUB_PACKAGES` (or
  narrow the parametrization) in `tests/test_stubs.py` and add real tests; for
  the scorecard, add real tests, since none will fail on your behalf. A PR
  that leaves a stub half-implemented but still nominally "a stub" is worse
  than leaving it alone.
- Stub docstrings often encode a decided design, not just a TODO — e.g.
  `edr/analysis/fit.py` specifies clustering by `scenario_id` and a
  directional threshold convention before any fitting code exists. Preserve
  that reasoning if you implement it; don't quietly substitute a different
  design.

## Hard invariants

These are the ones covered by tests that only catch a *changed* value, not a
*wrong* design decision applied consistently. Know them before touching the
relevant files.

- **RNG draw order is part of the reproducibility contract.**
  `edr.seeding.cell_seed` derives every cell's randomness from
  `(scenario_id, axis, level_index, seed_index)`, and each `Perturbation.apply`
  must draw from its `rng` in a fixed, documented order. Adding, removing, or
  reordering a draw changes every downstream output for the same seed and
  silently invalidates already-emitted records. `tests/test_perturb_contract.py`
  pins golden arrays specifically to catch this — a failure there almost
  always means a draw-order change; treat it as load-bearing, not flaky.
- **Never key anything on Python's builtin `hash()`.** It's salted per process
  (`PYTHONHASHSEED`) and would make the sweep unreproducible across shards.
  Use `hashlib.blake2b`, as `edr.seeding` does.
- **`model_id` is deliberately excluded from `cell_seed`.** This is what makes
  the two model arms a *paired* comparison — both see bit-identical
  perturbation draws at every cell. Don't add it.
- **`packages/doseresponse-scorecard` must never import `edr`, and must stay
  free of domain-specific terms** (model names, `scenario_id`, `ego_`-prefixed
  identifiers, etc.). It's an independently publishable package shared outside
  this repo. `tests/test_no_domain_coupling.py` greps for a forbidden-term
  list and `tests/test_import.py` asserts it imports with no `edr` on
  `sys.modules` — both run as part of the normal `uv run pytest`.
- **Severity levels live in `configs/axis/*.yaml`, never hardcoded in
  Python.** A `Perturbation` implementation should contain no severity
  numbers — it only knows how to *apply* a severity it's given. Re-anchoring a
  grid after a pilot should be a config diff, not a code change.
- **`configs/` is deliberately outside `src/edr`** so `pip install edr` gives
  the library without the physical-grid configs riding along. Don't make
  config loading depend on package data.

## Gitignore gotchas

- `calibration/` hand-labels **are** source and must be committed; only
  `calibration/scratch/` is ignored. Don't widen that ignore rule.
- `/data/` and `/weights/` are anchored with a leading slash on purpose — an
  unanchored `data/` would also match `src/edr/data/` and silently drop a
  source package from every commit. Keep the leading slash if you touch
  `.gitignore`.
- Never commit model weights or dataset shards (`*.npz`, `*.safetensors`,
  anything under `/data/` or `/weights/`) — they're gated / large by design,
  not just ignored by convention.

## Code style

- `ruff` config: line length 100, target `py311`, rules `E, F, I, UP, B, SIM`.
  Run `uv run ruff check .`; don't hand-tune around a rule it flags without a
  reason.
- `from __future__ import annotations` at the top of every module.
- Comments and docstrings explain **why**, not what — this repo names the
  specific failure a design choice prevents (see almost any file under
  `src/edr/` or `packages/doseresponse-scorecard/src/` for the pattern). Match
  that: don't restate what a well-named function does; explain a non-obvious
  constraint if one exists. If there's no non-obvious constraint, don't add a
  comment.
- New modules follow the existing per-file docstring pattern: what the module
  is, and — critically — what decision it embodies that isn't visible in the
  code itself.

## Out of scope

Closed-loop simulation, actuation delay, text perturbation, and training
anything. These were cut deliberately. Don't reintroduce one of them through a
"small" helper — this repo is meant to stay small.

## License

Code is Apache-2.0. Model weights are OpenMDW-1.1 with NVIDIA's
non-commercial assertion on top — re-verify licensing per artifact before any
commercial-path change; don't assume Apache-2.0 covers weights or generated
model outputs.
