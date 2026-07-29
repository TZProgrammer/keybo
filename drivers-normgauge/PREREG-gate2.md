# NORMGAUGE-GATE-2 PREREG — registered BEFORE the run, 2026-07-29

## Why this run exists

NORMGAUGE-GATE-1 could not separate two hypotheses: (H1) the blend is genuinely less accurate,
(H2) the AALTO-held-out frame favours AALTO because the three `.standardized` surfaces share
AALTO's bigram tensor. The result was perfectly monotone in AALTO weight, which BOTH predict.

I named two routes to separate them. **Route A (`.native` surfaces) is NOT RUNNABLE: they are not
shipped.** `data/surfaces/` contains exactly three files, all `.standardized`; no `.native` artifact
exists anywhere in the repo or on any branch, and no in-tree code builds one (`analysis/surfaces.py`
only *reads* `.standardized`). Recorded so it is not re-proposed.

**Route B IS runnable and is what this does: invert the held-out source.** Hold out COMMUNITY
(`tristrokes_last_community.tsv`, participants 200001-200007) instead of AALTO. The two sources are
DISJOINT, so if H2 is the whole story the direction must FLIP — a frame biased toward AALTO becomes
a frame biased against it.

## Pre-registered predictions (committed before seeing output)

* **P1** — If H2 is the whole story, holding out COMMUNITY makes the blends BEAT solo-AALTO,
  because the bias now runs the other way. **A flip is evidence FOR H2.**
* **P2** — If H1 is the whole story, solo-AALTO wins here TOO (a genuinely better predictor wins on
  any frame). **No flip is evidence FOR H1.**
* **P3** — I expect a PARTIAL flip: solo-COMMUNITY wins its own held-out frame, blends land between.
  Mixed evidence, both mechanisms live.
* **P4 — the falsifier that matters most:** COMMUNITY's frame is 4 participants at median 1 pid/cell.
  **If the fold ceilings come out low or the scoreable-cell count is small, this frame cannot
  adjudicate ANYTHING and the run must be reported as UNINFORMATIVE rather than as a verdict.**
  I commit to reporting that outcome as an answer, not as a failure to be retried on another frame.

## Method — identical to GATE-1 except for the held-out source

Cells = (layout, ngram, wpm bucket). Bucket-centered Spearman rho ÷ that fold's split-half ceiling.
Held-out unit = the layout label. Competitors: solo-AALTO (= ms/char), drop-pool 50/50,
registered (c), and **solo-COMMUNITY added as a positive control** — it MUST do well on its own
held-out source, and if it does not, the harness is not measuring what it claims.

## What this run cannot do

It cannot make the surfaces independent. Both frames read the same three surfaces, whose shared
component is real (each surface's deviation from the 3-way mean is only 37-49% of its own sd). A
flip would show the FRAME drives the ordering; it would not show the blend is better in absolute
terms. **No landing decision follows from this run either way** — that stays the user's.
