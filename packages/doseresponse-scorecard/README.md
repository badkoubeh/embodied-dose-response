# doseresponse-scorecard

Domain-neutral dose-response threshold estimation.

Given per-observation outcomes measured across a graded severity axis, fit a
dose-response curve, extract a threshold ED_p, and put an interval on it —
including on the *ratio* of two thresholds, which is the object you need when
asking "does measure A break before measure B?".

**Nothing in this package is specific to any policy, model, or domain.** It sees
severities, outcomes, and cluster ids. That constraint is deliberate: the package
is shared with ZetaBench, and the same protocol is meant to apply to a driving
policy and to a PID loop on a 6-DOF rocket reading a biased state estimate.

## Status

Skeleton. The API surface below is pinned; the implementations are stubs.

| Module | Contents |
|---|---|
| `fitting` | probit / logit / cloglog GLM; 4-parameter log-logistic; isotonic |
| `thresholds` | `ThresholdSpec`, ED_p extraction, right-censored lower bounds |
| `intervals` | profile-likelihood, Fieller ratio, delta method, parametric bootstrap |
| `schema` | the score-card JSON schema |
| `plotting` | the hero figure: two curves, one severity axis, both thresholds, ratio CI |

Fits are validated against R's `drc` (`ED`, `EDcomp`). Do not hand-roll a probit
fitter here — wrap `statsmodels`.

## License

Apache-2.0.
