# Research Roadmap — toward the best possible keyboard layout

Brainstormed 2026-07-04, after the rewrite + three audit rounds. Organized by what each
path contributes to the end goal. `TODO.md` tracks the committed work items; this doc is
the wider option space with reasoning, so future decisions pick from a considered menu
rather than the first idea available. Tags: 🟢 verified insight · 🟡 solid reasoning ·
🟠 speculative.

The end goal decomposes into four requirements, and every path below serves one:

1. **TRUST** — the objective must rank *novel* layouts correctly (else "optimal" means
   "optimal under a wrong model").
2. **SEARCH** — the optimizer must actually find the global optimum of that objective.
3. **OBJECTIVE** — "best" must mean what the user wants (speed? comfort? their corpus?).
4. **DATA** — everything above is bounded by what the data can support.

---

## A. TRUST — make the model's judgment of unseen layouts credible

### A1. The validation harness (committed plan — the keystone)
Schema change (layout + participant labels) → leave-one-layout-out + {layout × WPM} sliced
metrics → decide OQ-1 (freq feature) with the tightened decision rules (Kendall's τ
decisive, noise-ceiling-anchored thresholds, ≥3 seeds). Everything else in this section
builds on it. Already fully specified in `agent-artifacts/OQ5-*.md`.

### A2. 🟡 Rank, don't regress
The optimizer consumes only the *ordering* of layouts, and within a layout only relative
n-gram costs weighted by frequency. Squared-error regression spends model capacity getting
absolute milliseconds right — capacity that could go to getting *order* right. Options:
XGBoost's pairwise/LambdaMART ranking objectives; or keep regression but *evaluate* on rank
metrics only (already decided). A ranking loss also blunts OQ-1's additive-practice-effect
confound (additive shifts don't change order). Cheap experiment once the harness exists:
same features, `rank:pairwise` vs `reg:squarederror`, compare held-out Kendall's τ.

### A3. 🟡 Uncertainty-aware objective: optimize the ensemble, not the point estimate
A single model's errors become the optimizer's exploits — SA will happily park keys in
configurations where the model is *wrong-low* rather than *truly fast* (Goodhart). Train a
small ensemble (bootstrap over participants × seeds), then optimize a robust statistic:
`fitness = mean + λ·std` (or worst-case over the ensemble). Layouts that are fast under
EVERY plausible model beat layouts that are fast under one. This directly targets "best
possible" vs "best at gaming one model", and the ensemble is nearly free (retrain is
minutes). Evaluate: does the robust-optimum transfer better across ensemble members held
out of its own objective?

### A4. 🟡 Hierarchical / mixed-effects modeling of participants
Rows within a participant are correlated (skill, hardware, rhythm). Pooling ignores this;
the harness's participant-level bootstrap acknowledges it at eval time but not train time.
A participant random effect (or even per-participant normalization: z-score durations
within participant before pooling) removes between-typist variance the model currently has
to explain with geometry features. Likely a large, cheap accuracy win; also the correct
foundation for personalization (D2).

### A5. 🟠 Model the full duration distribution, not the mean
Typing times are heavy-right-tailed (hesitations). The IQR-mean target throws away shape.
Quantile regression (e.g. median + p90) lets the objective optimize "consistently fast"
vs "fast on average, occasionally terrible" — plausibly what typists actually feel.
Low priority until A1–A3 land.

---

## B. SEARCH — actually find the optimum of the objective

### B1. 🟢 The QAP reformulation (verified; the big one)
**Verified 2026-07-04:** with the freq feature pinned (or post-OQ-1 dropped) and wpm fixed,
the bigram feature vector depends ONLY on the two key *positions* — different characters at
the same slots give byte-identical vectors. Therefore:

    fitness(layout) = Σ_bg  w(bg) · T[pos(bg[0]), pos(bg[1])]

where `T` is a **961-entry (31×31 positions) table of model predictions** computable with
ONE batch predict per run. Consequences, in order of importance:

- **Speed:** a swap changes 2 rows/columns of the assignment; delta-evaluation becomes pure
  table arithmetic — no XGBoost call in the loop at all. This subsumes the planned
  "delta-scoring" TODO and is orders of magnitude stronger (~25ms/eval → microseconds).
  Full-corpus optimize with thousands of restarts becomes interactive. Trigram analog:
  T3 = 31³ ≈ 30k entries — same trick, still one batch predict.
- **It IS a Quadratic Assignment Problem** (flows = bigram frequencies, distances = learned
  times) — the literature the paper itself cites. That unlocks 50 years of QAP machinery:
  memetic/tabu heuristics that are SOTA (far stronger than vanilla SA), Gilmore–Lawler
  **lower bounds** (a certificate of "no layout can beat X" — turning "best we found" into
  "provably within ε of optimal"), and exact branch-and-bound feasibility at n=30–31.
- **Caveat (ties to OQ-1):** with the freq *feature* retained, T gains a freq-bin dimension;
  saturation means ~1 bin in practice, but dropping the feature makes the reformulation
  *exact* — one more argument for weight-only.

### B2. 🟡 Optimizer portfolio + restarts at scale
`--attempts N` exists; B1 makes N=10,000 feasible. Add a memetic algorithm (population +
local search) and tabu search over the same table; keep SA as baseline. Report the
best-found distribution across methods — if they all converge to the same optimum, that's
(weak) evidence of global optimality; if not, the landscape is telling us something.

### B3. 🟡 Constraints as first-class citizens
Real users need: pinned keys (keep `zxcv` for shortcuts; keep punctuation placement),
"max distance from QWERTY" (learnability budget), per-hand load bounds. All are trivial in
the QAP frame (restrict the assignment) and turn "the best layout" into "the best layout
*you would actually adopt*" — arguably the most user-valuable search feature.

### B4. 🟠 Symmetry & search-space reduction
The geometry is left/right mirror-symmetric; if the objective is too (check: rolls/hand
features may break it), every layout has a mirror twin and half the space is redundant.
Verify empirically on T; if it holds, fix one high-frequency key's hand to halve the space.

---

## C. OBJECTIVE — make "best" mean the right thing

### C1. 🟡 Comfort/effort composite (OQ-4, designed)
`CompositeScorer = predicted_time + λ·(SFB, scissor, LSB, redirect penalties)`, all λ=0
default. The features exist; ~a day of work. Present the user the 3-point trade-off curve
(pure speed / balanced / comfort-heavy) and let them pick — that choice closes OQ-4.

### C2. 🟡 Personal corpus weighting (OQ-3b)
`keybo corpus-freqs <your-text>` → optimize for YOUR distribution (code, chat, another
language). The weight file is already swappable; only the generator tool is missing.
Likely changes the optimum materially for programmers (symbols, camelCase). This is the
single most user-differentiating feature.

### C3. 🟠 Learnability as an explicit objective term
Switching cost is real; a layout 3% faster but 40 keys moved may be worse *in practice*
than 2.5% faster with 12 keys moved. Term: predicted_time + μ·d(layout, QWERTY) (keys moved
or finger-remap distance). Sweep μ → a Pareto front of "gain vs retraining pain". Pairs
naturally with B3 pinning.

### C4. 🟠 Multi-WPM robustness
We optimize at one target WPM, but a typist traverses speeds while learning. Optimize
`Σ_wpm π(wpm)·fitness(layout, wpm)` over a progression distribution π, or check the single-
WPM optimum's stability across the WPM range (cheap with B1's per-wpm tables).

---

## D. DATA — raise the ceiling everything else is under

### D1. 🟡 Mine the existing dump harder first
136M keystrokes is a lot; we use bigrams/trigrams at ≥ threshold WPM from clean windows.
Cheap extensions: 4-grams for rhythm context 🟠; per-participant normalization (A4); use
*error rates* as a signal (positions that cause typos are bad even when fast) 🟡 — the
correction data we currently discard is an unexploited label.

### D2. 🟠 Personal data collection → personalized layouts
A small opt-in local logger (or typing-test page) collecting a user's own (position, time)
pairs → fine-tune the population model per user (hierarchical: population prior +
individual effects). "The best layout *for you*" is both more achievable and more valuable
than universal-best; it sidesteps some generalization worries (the model is grounded in the
target user).

### D3. 🟠 Crowdsource non-QWERTY data (the Kiakl revival)
The permanent ceiling on TRUST is 4 QWERTY-family training layouts. A lightweight web
typing test targeted at alt-layout communities (Colemak/Dvorak/Workman users are
enthusiastic testers) directly attacks the confound OQ-1 worries about. High effort, high
ceiling-raise; the harness (A1) tells us exactly how much we need it.

### D4. 🟠 Active learning to choose what data to collect
Once A3's ensemble exists, its *disagreement* on candidate layouts identifies which
(position, transition) cells the models are uncertain about → design the D3 typing test to
cover exactly those. Data collection guided by model uncertainty instead of convenience.

---

## Sequencing (what I'd actually do)

1. **A1 harness** (committed; schema → LOLO → OQ-1 decision) — nothing is trustworthy
   without it.
2. **B1 QAP table** — verified feasible, transforms search power AND makes every subsequent
   experiment (A3 ensembles, B2 portfolio, C4 sweeps) cheap. Do immediately after (or
   alongside) A1; it's independent of the schema change.
3. **A2 + A3** (rank objective, ensemble robustness) — cheap once A1 exists; directly
   improve "optimal means optimal".
4. **C1 + C2** (comfort knob, personal corpus) — the user-facing "best for me" features.
5. **B2/B3** (portfolio, constraints, bounds) — squeeze the search; get the optimality
   certificate.
6. **D2/D3/D4** — raise the data ceiling; longest lead time, start D3 socially early if
   the A1 verdict shows weak transfer.
