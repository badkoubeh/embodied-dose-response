# ROADMAP.md

The stage-by-stage plan for `embodied-dose-response`, kept in sync with the
project's local planning doc so this part of it travels with the repo. See
[AGENTS.md](AGENTS.md) for contribution mechanics and [README.md](README.md)
for the research question and glossary.

**Status: pre-Stage-0.** Nothing is validated until the Stage 0 go/no-go
passes. This is also why most of `src/edr/{data,runner,metrics,analysis}` and
`packages/doseresponse-scorecard` are pinned-signature stubs rather than
working code — see AGENTS.md's "Stubs" section before treating one as a gap
to fill.

## Stage 0 — go/no-go

Confirm the ego-state channel is actually load-bearing before building
anything else on top of it. Request dataset and weights access, stand up a
minimal single-GPU inference smoke test, produce one real trajectory and
reasoning trace, and measure actual per-inference wall-clock and VRAM
(planning numbers are optimistic; PLAN.md 4's latency reality check is a
one-line reminder not to trust them). Then write a small ego-state noise
hook and run a coarse sweep over a handful of severity levels on a small
scenario subset.

> **Kill criterion.** If minADE does not move under physically extreme
> ego-state corruption, the vision path dominates and ego-state perturbation
> is not a productive axis. Pivot to a consistency-ordering-only framing, or
> stop — do not proceed to Stage 1 on a null result.

Blocked on: gated dataset access, model weights.

## Stage 1 — instrument

Build all three perturbation hooks for real (`ego_noise`, `ego_drift`,
`staleness`), the meta-action extractor, and a hand-labeled calibration set
(~100 pairs, stratified across the severity grid — see
[calibration/README.md](calibration/README.md)) to validate the extractor
against, with precision/recall reported rather than assumed. Wire up Hydra
configs, experiment logging, deterministic seeding, and the
`doseresponse-scorecard` skeleton.

Blocked on: Stage 0 passing.

## Stage 2 — sweep

Run a cheap adaptive pilot to locate the interesting severity range per
axis, then commit to the fixed common grid (`configs/grid/full.yaml`) for
the reported run. Optionally add the confounded secondary model arm, only if
Stage 0 showed clean ego-state sensitivity — see `edr.seeding` for why the
two arms must see bit-identical perturbation draws if that arm runs at all.

Blocked on: Stage 1 landing; a real calibration precision/recall number.

## Stage 3 — analyze and write

Fit the dose-response curves, extract thresholds, compute the Fieller ratio
on the headline σ\*(consistency) / σ\*(trajectory) object, produce the hero
figure, and draft the paper — including an explicit Limitations section.

Blocked on: Stage 2's sweep data.

## Stage 4 — ship

Preprint, plus a clean repo, then submit.

---

This file mirrors the "Stages and schedule" section of the local `PLAN.md`
(see AGENTS.md on why that file may not be in your checkout) at the level of
detail meant to travel with the repo. It deliberately omits `PLAN.md`'s venue
strategy, risk register, and cross-paradigm scoping notes — those are
planning content, not roadmap. If a stage's scope changes, update this file
in the same PR as the code that changes it; don't let it drift into
aspirational fiction.
