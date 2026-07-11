# P9LR-w0.5 — the quality deliverable (2026-07-11)

```
c i t h s   n l e a k
. p g v f   w z u o j
q y d b m   r x / , ;
```

String form: `cithsnleak.pgvfwzuojqydbmrx/,;` (qwerty slot order, space unchanged).

## What it is

The recommended member of the P9-LR QUALITY family: optimized for **attainable**
speed — the F5M objective (mean of each pattern's fastest fifth of executions) rather
than the strategy-average. The quality thesis, measured across the campaign: cell
means mix clean executions with fumbles; a layout chosen on the fast tail rewards
patterns that are fast *when executed well*. F5M is the one alternative frame that
certifies layout ranking (decisive-pair tau 1.0).

| metric | value | context |
|---|---|---|
| predicted attainable-speed vs qwerty | **+2.72%** | F5M-LOGRAT corrected-trigram objective, wpm 90 |
| sfb | **1.19%** | family w=0 is 4.58% — the oxey half-weight buys −3.4pp sfb for −0.16% speed |
| family objective purity | PURE (both models F5M-LR) | trigram gate passed at raw-rho 0.69 (bar 0.55) |
| model behind it | DED-LR: −37.7% wmae vs the ms-era quality model | the LOGRAT lever transfers to the F5M target intact |

## Structure

The quality surface puts the letter core on the TOP row (`cithsnleak`) — a stable
signature across both quality-family generations (ms-era P9 did the same). Mechanism:
top and home rows are speed-tied in the measured physics (OQ-14), and the fast-tail
surface exploits that tie differently than the mean surface. The ms-era P9 champion
carries +0.43% regret under this upgraded surface — the −37.7% model improvement
moved the quality argmax beyond plateau noise.

## Speed vs quality: which family to pick

- **P11-w0.5** (`docs/layout-p11-w05.md`): best *expected* speed — the average
  keystroke, all executions included.
- **P9LR-w0.5** (this): best *attainable* speed — what your clean executions reward.
- The two argmaxes genuinely differ (measured D4 cross-regret ~0.2–0.4% both ways).
  For a single pick, P11-w0.5 is the default recommendation (expectation is what
  total typing time integrates); P9LR is the aspirational-ceiling alternative.

## Honest caveats

- Same validation ceiling as all campaign numbers: 4 layouts, model-predicted, not a
  human trial. The F5M target additionally reads the fast tail of small cells, which
  is noisier than the mean — its harness evidence (DED-LR, rho/own-ceiling 1.056,
  taus 1.0) is the licence.
- w=0's 4.58% sfb shows the raw quality surface tolerates same-finger patterns the
  mean surface avoids (fast-tail executions of sfbs can be quick); the w=0.5 oxey
  term is doing real work here, not cosmetics.
