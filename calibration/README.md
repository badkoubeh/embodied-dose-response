# Calibration set

Hand-labeled (reasoning text, trajectory) pairs used to validate the meta-action
extractor. PLAN.md 5 (Stage 1) calls for ~100; its precision and recall against
these labels are a reported result, not an internal check, because the extractor
is a reimplementation of a reward NVIDIA did not release (PLAN.md 3.2, 6).

**Nothing is here yet** — the set cannot be built before Stage 0 produces real
reasoning traces.

## Two requirements that are easy to miss

**Stratify across the severity grid.** Labeling only clean records leaves the
project's most dangerous confound unmeasured: if reasoning text grows hedged or
malformed as severity rises, extractor precision falls, measured consistency
falls with it, and that is indistinguishable from the effect this project claims
to measure. Sample labels at every level, and report precision as a function of
severity. A flat curve is the validity check; a sloping one changes the paper.

**~100 pairs is thin for per-class claims.** Over six classes that is ~17 each,
so per-class precision intervals are roughly ±0.18 at best. Either go to 200–300
pairs, or report macro-averages with explicit Wilson intervals and say the
per-class numbers are indicative only.

## Format

One JSONL file per labeling round, `labels_<round>.jsonl`:

```json
{
  "scenario_id": "...",
  "axis": "ego_drift",
  "level_index": 3,
  "seed_index": 0,
  "reasoning_text": "...",
  "label_longitudinal": "decelerate",
  "label_lateral": "straight",
  "label_source": "yield",
  "labeler": "...",
  "notes": ""
}
```

`label_source` keeps the original six-way label where the labeler used one, so
the many-to-one collapse into (longitudinal, lateral) stays auditable — see
`edr.schema.MetaAction` for why the comparison is a pair rather than one label.

Labels are source and belong in version control. Only `calibration/scratch/` is
ignored.
