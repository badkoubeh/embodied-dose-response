"""GPU plane: model wrapper, sharding, raw-record writing (PLAN.md 3.2).

STUB. Needs weights and a >=24 GB GPU, neither available at scaffold time.

The invariant this package exists to protect: it emits `RawRecord`s and nothing
else. No meta-action, no ADE, no predicate. PLAN.md 2.1 -- recalibrating the
extractor must never cost a re-inference.
"""
