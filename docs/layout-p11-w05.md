# P11-w0.5 — the pure-speed headline (2026-07-11)

> **Note (2026-07-11, user-approved):** the PRIMARY deliverable is now
> **P10-w0.5** (`docs/layout-p10-w05.md`). The two are speed-TIED (+4.00% vs
> +3.95%, inside the ~0.2% plateau), and P10-w0.5 wins every preference gauge:
> genkey Score 33.7 vs 41.1, keymeow sfb 1.18% vs 1.70%, balanced finger loads
> (max 15.9%) vs 20.0% concentrated on the right ring, lower dislocation. Three
> independent multi-gauge searches (P12 dislocation-weighted, P13 genkey-weighted,
> and the combined genkey+oxey search, which reconverged on P10's layout) all
> selected P10-w0.5. This layout remains the argmax of the calibrated speed
> objective alone and is retained as the pure-speed headline artifact; its
> finger-load concentration is a documented calibration-era artifact that buys
> no measured speed (lag-3 probe null).

```
c g l d k   . , y o u
s r t h m   p n i e a
q x w b v   f z j ; /
```

String form: `cgldk.,yousrthmpnieaqxwbvfzj;/` (qwerty slot order, space unchanged).

## What it is

The recommended member of the P11-FINAL speed family — the layout the fully
gate-verified measured-speed model picks with community comfort heuristics at half
weight. Statistically speed-TIED with the family's pure-speed champion (+4.00% vs
+3.99% — inside search noise) while carrying lower sfb and the family's lowest
first-finger-calibrated pattern share.

| metric | value | context |
|---|---|---|
| predicted speed vs qwerty | **+4.00%** | calibrated LOGRAT corrected-trigram objective, wpm 90 |
| speed vs pure-speed champ (w=0) | +0.01% (tie) | inside the ~0.2% plateau |
| sfb | **1.09%** | qwerty ~6% |
| outer-first share | **0.42%** | qwerty 1.08%; uncalibrated-era families 1.2% |
| home row usage | ~55% | |
| certificate (bigram component) | within 3.41% of provable optimum | Gilmore–Lawler |
| skill range | optimal within noise at wpm 70 AND 110 | multi-wpm argmax stage |

## The model stack behind it (every component gate-verified)

- **LOGRAT target space** (`log(ms·wpm/12000)`): cross-layout wmae −37% (bigram),
  −24% (join trigram) — the campaign's largest lever, adopted with rare-ngram guards.
- **Fitted first-finger calibration**: pinky-first +43ms / ring-first +21ms, fitted
  in-pipeline by the matched-cell estimator (nothing hardcoded), stored in the model
  sidecar, applied per position pair at serve. Verified: LOLO non-degrading, E5-v2
  cross-regret −0.003%, served-sign 8/8. Its fingerprint: the family's outer-first
  share collapses ~2.5× vs uncalibrated builds at ~zero speed cost.
- **Join-construction conditioned trigram** (press2→press3 increment), the
  construction re-verified under LOGRAT (rho/ceiling 1.011).
- **Survivorship**: on the final night, six further improvement arms (variance
  correction, rollover mixture, letter-additive practice, per-sample trigram targets,
  pinball quality model, feature/tuning re-runs) ALL failed their preregistered rules
  — the shipping stack is the one dozens of registered challengers could not beat.

## Multi-wpm result

The wpm-90 argmax carries +0.057% regret at wpm 70 and −0.010% at wpm 110: this
layout is optimal (within noise) across the 70–110 band. One layout serves the range.

## Honest caveats

- Model predictions validated cross-layout on 4 layouts (the dataset's ceiling); not
  a human trial of this layout. +4.00% is a within-model statement.
- The plateau is real: a structurally different family shape
  (`uoy,.vlmdgaeinprhtcs;/jkbfwxzq` and variants) ties at +3.99% — near-equivalent
  optima exist; single-swap variants of this layout are equivalent.
- Finger-load concentration: 20.0% of keystrokes land on the right ring (vs 15.9%
  max on P10-w0.5). Measured speed cost of avoiding this: ~zero (lag-3 probe null;
  P12's w_d=0.5 arm rebalanced at −0.025% regret). It is a preference defect, not a
  speed feature — the reason P10-w0.5 was promoted over this layout.
- Primary deliverable: `docs/layout-p10-w05.md` (P10-w0.5 — speed-tied, wins all
  preference gauges, triple-search convergence).
