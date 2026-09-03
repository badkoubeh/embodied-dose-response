# embodied-dose-response

Dose-response threshold estimation for embodied policies under graded input
degradation. Applied to Alpamayo 1.5; open-loop, reproducible, single-GPU.

**Status: pre-Stage-0 scaffold.** Nothing here is validated.

The architecture, experimental protocol, and go/no-go criteria live in a planning
document (`PLAN.md`) that is kept locally and not published with the repo. Code
comments cite it by section — "PLAN.md 2.1", "PLAN.md 3.2" — and the open issues
carry enough context to stand on their own without it.

## The question

Under physically-grounded, benign degradation of the ego-state input, does
**reasoning–action consistency** degrade at a *lower* severity than **trajectory
quality**? That is: does the interpretability guarantee that RL post-training was
trained to enforce fail before the driving fails?

The headline object is the ratio σ\*(consistency) / σ\*(trajectory) with a Fieller
interval. Below 1, consistency breaks first. Above 1, RL bought real robustness.
Either is a result.

## The one rule

**The GPU plane emits raw artifacts; the analysis plane scores them.** GPU jobs
write a predicted trajectory, reasoning text, timings, and hashes — nothing
derived. Meta-action extraction, displacement error, and predicates all run
post-hoc on CPU against those artifacts.

The payoff: recalibrating the meta-action extractor never costs a re-inference.
That is the single most important architectural decision in this repo, and it is
enforced rather than assumed — the packaging keeps torch out of the base install,
and `tests/test_stubs.py::test_analysis_plane_imports_without_torch` fails if it
ever leaks in.

## Quickstart

```bash
uv sync --extra analysis --group dev   # analysis plane; no CUDA wheels
uv run pytest
uv run ruff check .
```

| Where | Command |
|---|---|
| Laptop (analysis, dev) | `uv sync --extra analysis --group dev` |
| Laptop (job submission) | `uv sync --extra cloud` |
| GPU host | `uv sync --extra gpu --extra analysis` |
| CI | `uv sync --locked --extra analysis --group dev` |

Note: `--extra gpu` installs nothing on macOS. torch has published no macOS
x86_64 wheel since 2.2.2, so those dependencies carry `sys_platform == 'linux'`
markers — without them `uv lock` is unsolvable on an Intel Mac. Intended, but
surprising the first time.

`configs/` sits outside the package, so `pip install edr` gives you the library
without them. The runner always executes from a checkout.

## What works today

Real, tested, and CPU-only:

| Path | |
|---|---|
| `src/edr/schema.py` | `Sample`, `RawRecord`, `ScoredRecord`, `Cell` |
| `src/edr/seeding.py` | deterministic per-cell RNG, stable across processes and shards |
| `src/edr/perturb/` | the `Perturbation` protocol, a name registry, and all three axes |
| `configs/` | Hydra groups carrying the physical severity grids |

Stubs with complete signatures — pinned contracts, no implementations — for
`data/`, `runner/`, `metrics/`, `analysis/`, `scripts/`, and the
`doseresponse-scorecard` package. Each names what blocks it: gated dataset
access, model weights, or the calibration set.

## The three axes

All three degrade the ego-state channel only, so one video decode serves an
entire severity sweep.

| Axis | Unit | What it models |
|---|---|---|
| `ego_noise` | m | White noise on the pose history — a noisy but unbiased estimate |
| `ego_drift` | %/distance | Dead-reckoning drift during a GNSS outage. **The differentiator axis** |
| `staleness` | ms | A lagging estimate: perception sees now, localization reports τ ago |

Severity levels live in `configs/axis/*.yaml`, not in code, so re-anchoring a
grid after the pilot is a reviewable config diff. Every level carries a physical
referent. **All current values are provisional** — the reported grid comes from
the Stage-2 pilot.

## Reproducibility

Every cell's randomness derives from `(scenario_id, axis, level_index,
seed_index)` via blake2b, never from a stream. Results are shard-order
independent and any single cell reproduces in isolation. `model_id` is
deliberately excluded so both model arms see identical perturbation draws — the
secondary arm is a paired comparison.

Golden-value tests pin both the seeds and the per-axis draw order: reordering an
`rng` call changes every output while the seed stays the same, which would
silently invalidate already-emitted records.

## Out of scope

Closed-loop simulation, actuation delay, text perturbation, and training anything.
Each was cut deliberately; see PLAN.md for the reasoning. This repo is
deliberately small; keep it that way.

## License

Apache-2.0. Model weights are OpenMDW-1.1 with NVIDIA's non-commercial assertion
on top; publishing evaluation results is unrestricted. Re-verify per artifact
before any commercial path.
