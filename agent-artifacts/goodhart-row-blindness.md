# Goodhart caught: the C2A5 feature reduction made the optimizer row-blind (2026-07-05)

The most instructive failure of the project so far. Recorded in full because the mechanism
generalizes: **our validation harness and our optimizer query the model on different
distributions, and a feature change can improve one while breaking the other.**

## What happened, in order

1. The feature-arm matrix (13 arms, LOLO-judged) found that dropping the second-key
   row/finger one-hots + capping tree depth at 3 ("C2A5") raised mean held-out ρ/ceiling
   from .937 to **1.000** at layout-τ +1.0. Adopted per the pre-registered rule; the
   prod-path re-verify confirmed the numbers (`runs/lolo_v5_c2a5.json`).
2. The very next layout search produced:

   ```
   d g e a , n r t s c
   k y q ; . f l b w p     <- k y q ; . on the HOME ROW
   j / o u i h m x v z     <- o u i (vowels!) on the BOTTOM row
   ```

   Junk keys on home, the vowel cluster exiled to the bottom row — visibly absurd, and
   its headline (+2.10% vs qwerty) *lower* than the previous model's layout despite the
   "better" model.
3. Diagnosis (one probe): under C2A5 features, `a→s` (home row), `q→w` (top), and `z→x`
   (bottom) produce **byte-identical feature vectors** — same |dx|, dy=0, same relational
   flags — so the model prices them identically (167.13 ms each). The pre-C2A5 model
   priced them 172.6 / 164.6 / **207.3** ms. With no row signal for same-row bigrams, home
   placement is worthless to the optimizer and 30! permutations collapse into massive
   equivalence classes.
4. Cross-check under the row-aware model: the C2A5-optimized layout scores only +0.91% vs
   qwerty (vs the previous best's +2.56%) — the "optimum" was an artifact of the blindness.

## Why LOLO could not catch this

LOLO evaluates on **real layouts** — qwerty/azerty/qwertz/dvorak — and every real layout
uses rows sensibly (common keys on home). Within that distribution, the row one-hots are
nearly redundant with the relational features, so deleting them looked free (even
beneficial: less memorization capacity). But the **optimizer queries the model far off
that distribution** — it actively searches for pricing null spaces, and "rows don't matter
for same-row bigrams" is exactly such a null space. Classic Goodhart: optimizing a proxy
(LOLO ρ) diverged from the target (layout quality) precisely where the proxy has no data.

## Resolution (FEATURE_VERSION 2026-07-05.3)

- **Row/finger one-hots restored** — they are load-bearing for the optimizer even where
  held-out prediction metrics call them redundant. A schema regression test now pins them
  with the incident as its docstring.
- **Depth 3 kept** — arm C2 (depth-3 with FULL features) scored 0.9966 alone, nearly all
  of C2A5's gain, without deleting information. Regularization ≠ feature deletion: the
  first limits how the model *uses* information; the second removes it from the
  *objective*.
- Layout `dgea,...` discarded; `runs/final_layout_c2a5.json` kept as the incident record.

## Standing lessons (added to the backlog as E5)

1. **A feature that looks redundant on-distribution may be load-bearing off-distribution.**
   Feature *deletions* need an optimizer-side sanity gate, not just LOLO: run a quick
   search under the candidate model and eyeball/auto-check the result (e.g. corpus-weighted
   home-row usage of the optimized layout must exceed each named layout's).
2. **Regularization is the safe way to cut memorization capacity; deletion is not.**
3. This is the second time the FEATURE_VERSION/verification discipline caught a subtle
   break within hours instead of weeks (first: the stale-model load guard). The loop
   "adopt → verify → search → inspect" is doing its job; keep the *search* step in it.
