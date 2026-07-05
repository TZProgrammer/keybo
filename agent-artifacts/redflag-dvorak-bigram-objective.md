# Red flag: why does dvorak outrank modern layouts? (resolved 2026-07-05)

The user flagged it: dvorak ranking above semimak/graphite/colemak under the R1W model
smells wrong — modern layouts are more carefully constructed. Three hypotheses were
pre-registered before results; the tests (driver
`agent-artifacts/experiments/redflag_and_buckets.py`, results
`keybo-e2e/runs/redflag_buckets.json`) adjudicated them cleanly.

## Verdict: the ranking is real, but it is the verdict OF A BIGRAM OBJECTIVE

**H-B (weighting confound) — REFUTED.** With layout-balance weights OFF (dvorak's 64
typists back to their natural 9.1% example share), the ordering is unchanged: dvorak
+0.85% > semimak +0.55% > graphite +0.45% > qwerty > colemak −0.75%. The ~800×
per-sample upweight amplifies dvorak's edge slightly (+1.08 vs +0.85) but does not
create it.

**H-A (structural: bigram PP rewards alternation) — CONFIRMED, quantitatively.**
- Corpus-weighted motion-class shares (common 840-bigram subset):

  | layout | alternation | same-hand | same-finger |
  |---|---|---|---|
  | dvorak | **80.2%** | 16.1% | 3.8% |
  | graphite | 78.9% | 18.4% | 2.6% |
  | semimak | 77.4% | 20.1% | 2.4% |
  | colemak | 72.1% | 25.1% | 2.8% |
  | qwerty | 68.2% | 25.4% | 6.3% |

- The model prices the classes at alt **165 ms**, same-hand **185 ms**, SFB **196 ms**
  (means over the position-pair table). Back-of-envelope: dvorak's 2.8 pp alternation
  edge over semimak × the 20 ms alt-vs-same-hand gap ≈ 0.56 ms/bigram ≈ **0.33%** of
  total — the observed scoreboard gap is **0.29 pp**. The alternation arithmetic
  reproduces the ranking almost exactly (semimak claws back some via its lower SFB
  share, which is why it beats graphite despite less alternation).

Dvorak IS the maximum-alternation design — vowels on one hand, consonants on the other.
A bigram press→press objective genuinely, correctly-per-the-data measures alternation as
~20 ms faster than same-hand motion, so it ranks dvorak on top. Modern layouts
deliberately *trade alternation away* for same-hand roll quality, redirect avoidance,
and comfort — properties that live at the **trigram/rhythm level, invisible to a bigram
model**. The disagreement with community rankings is therefore expected, not a bug:
community metrics see rolls; our objective cannot, yet.

**TEST-1 nuance (train with ZERO dvorak rows) — an honest caveat.** Without dvorak's own
data, dvorak drops below semimak and graphite (−1.42 vs −0.90/−1.29), and *every*
alternative sinks below qwerty. Reading: the model's confidence that strongly
alternation-heavy, un-qwerty-like geometry is fast is substantially calibrated from
dvorak's own rows (it is the only high-alternation training layout). Not memorization in
the freq-ID sense — but "dvorak is the top named layout" is partially self-supported
evidence. With only 4 training layouts, that circularity is unavoidable; it is exactly
the data-ceiling caveat OQ-5 carries.

## Consequences

1. **Our optimized layout inherits the same lens.** Its dvorak-like vowel/consonant hand
   split and its +3.49% are conditioned on the bigram objective. Under a trigram- or
   comfort-aware objective, both the layout and the named-layout ordering may shift.
2. **This accelerates the trigram milestone.** The bigram world's structural blind spot
   (rolls) is now measured, not suspected. Remaining bigram-world work (feature/target/
   tuning arms in `bigram-experiment-backlog.md`) is about squeezing transfer quality;
   the *objective's* next step is trigrams (OQ-10) + comfort terms (OQ-4).

## Bucketed-frequency arms (user proposal + reframe) — both REJECTED by the rule

Pre-registered rule: adopt over shipped R1W only if pooled τ ≥ +1.0 AND mean ρ/ceiling
> 0.931 (R1W's anchor, reproduced by this driver at 0.937).

| arm | pooled τ (3 seeds) | mean ρ/ceiling | verdict |
|---|---|---|---|
| **F20W** — 20 equal-count corpus-freq buckets as a FEATURE | +1.0 all | 0.763 | rejected |
| **R3W** — practice term shrunk toward the freq-bucket curve | +1.0 all | 0.921 | rejected |
| R1W (shipped; anchor) | +1.0 all | **0.937** | keeps |

- **The user's instinct was directionally right:** F20W transfers *far* better than raw
  freq ever did (τ +1.0 vs +0.333) and beats the no-practice control (ρ/ceiling 0.763 vs
  0.641) — coarse buckets do capture practice without full ID-memorization. But the
  explicit additive term captures the same signal better (0.937), because it models
  practice per-bigram with principled shrinkage instead of forcing the tree to spend
  splits on a 20-level coarse code and its geometry interactions.
- **R3W's failure is instructive:** shrinking rare bigrams toward their frequency-class
  curve helps the thin dvorak fold (ρ .61 vs .60) but *hurts* the qwerty fold badly
  (.69 vs .77) — in that fold the curve is estimated from the three thin layouts and is
  noisy; anchoring b to it pollutes the well-measured bigrams. Shrink-to-zero stays.
