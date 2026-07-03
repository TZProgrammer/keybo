# agent-artifacts/

Working artifacts produced while driving the open questions (`OPEN_QUESTIONS.md`) toward
closure, plus design-audit reports. One artifact per open question: each states the best
answer we can currently give, the evidence behind it, and — most importantly — **the concrete
test/experiment that would close the question definitively**, with acceptance criteria.

| Artifact | Question | Status |
|---|---|---|
| `OQ1-frequency-feature.md` | freq as model feature vs. objective weight | 🟡 leaning weight-only; closable with LOLO harness |
| `OQ2-freq-saturation.md` | freq feature saturation / dual role | 🟡 mostly subsumed by OQ-1 |
| `OQ3-frequency-distribution.md` | which freq distribution; per-user corpora | 🔴 needs decision + small tooling |
| `OQ4-objective.md` | is predicted-time the right objective? | 🔴 product decision; experiments proposed |
| `OQ5-generalization-validation.md` | validating on novel layouts | 🔴 build the harness (design inside) |
| `OQ6-geometry.md` | non-row-stagger geometries | 🟢 decided (no, for now) |
| `OQ7-nonqwerty-leverage.md` | leveraging imbalanced non-QWERTY data | 🔴 experiment matrix defined |
| `OQ8-sliced-evaluation.md` | per-layout / per-proficiency evaluation | 🟡 design settled; blocked on schema change |
| `design-audit-2026-07-03.md` | full design audit: gaps, missing features, risks | report |
| `experiments/` | runnable probe scripts backing the artifacts | — |

Conventions: claims are tagged 🟢 VERIFIED (ran it) / 🟡 HIGH (read the code) / 🟠 INFERRED /
🔴 UNCERTAIN. Each "Definitive close" section is written so that executing it produces a
yes/no decision, not more discussion.
