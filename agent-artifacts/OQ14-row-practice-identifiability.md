# OQ-14 — Is the home-row advantage biomechanics or practice? (opened 2026-07-05)

**Status: 🔴 open — surfaced by the E5 gate firing on the corrected (d3) model's layout.**

## How it surfaced

The post-Goodhart search (full features, depth 3) produced
`dae,yrntscgoipulmfbwq;/.khvxzj` — common letters (`d a e , y r n t s c`) on the **top**
row, home-row corpus share only 33.7% (vs colemak 71.9%, dvorak 68.4%; qwerty itself is
32.0%). The E5 structural gate flagged it. But unlike the row-blindness incident, this
model IS row-aware (bottom-row bigrams priced ~40 ms slower) — so the low home share is
the model's honest preference, not a null space. The discriminating probe (same-row letter
bigrams, wpm 60–100, median of per-bigram medians):

| layout | top | HOME | bottom |
|---|---|---|---|
| qwerty | **159** | 162 | 211 |
| qwertz | **144** | 152 | 217 |
| azerty | 134 | 132 | 133 |
| dvorak | 154 | **120** | 185 |

**Each layout's fastest row is the row where ITS common letters live.** Qwerty's top row
(e r t y u i o p) beats its own home row; dvorak's home row (a o e u i d h t n s) crushes
everything. This is position-level practice — fingers are fast at the positions they use
most — and it is exactly the confound the per-bigram practice term CANNOT absorb: b(ngram)
follows the bigram, but this effect follows the position, pooled across all bigrams
sharing it.

## What is and isn't trustworthy

- **Bottom-row penalty: trustworthy.** All layouts agree (185–217 ms vs 120–162) —
  biomechanical, and the optimized layout correctly banishes rare letters there.
- **Top-vs-home: NOT identifiable from this data.** 98.7% qwerty means "top is fast" and
  "practiced positions are fast" are nearly the same statement. Dvorak's 120 ms home row
  is the strongest counter-evidence (home CAN be much faster) but is itself
  practice-confounded plus population-confounded (64 enthusiasts).
- **Equilibrium argument, cuts both ways:** a new layout's user eventually practices
  whatever positions its common letters occupy. If position speed is mostly practice, the
  top-vs-home choice barely matters at equilibrium (and the model's preference is
  harmless); if partly biomechanical, the data under-credits home. The honest statement:
  the bigram data cannot currently separate these.

## Candidate resolutions (all harness-judged)

1. **Position-level practice term**: p(position) analog of b(ngram), identified from
   cross-layout variation (same position, different letter-occupancy across layouts);
   residualize both. Risk: with 4 layouts, position-occupancy variation is thin.
2. **Within-participant longitudinal signal**: the dump has session ordering per
   participant — practice curves within a session/person could separate familiarity from
   geometry (participants improving on THEIR layout's positions over sessions).
3. **Dvorak-emphasis sensitivity**: train with dvorak upweighted further; if home-row
   preference strengthens smoothly, the home signal lives mainly in dvorak's data (thin);
   if stable, qwerty data itself supports it.
4. Treat as an OQ-4 comfort prior: impose home-row preference as an explicit objective
   term with a user-chosen weight, acknowledging the data can't justify a specific value.

## Interim posture

The d3-best layout stands as the current model's honest optimum (+2.49% vs qwerty;
old best re-scores within 0.11% under this model — the plateau again), with THIS caveat
attached wherever it's published: its top-row-heavy structure reflects a possibly-
practice-confounded row preference. The E5 gate stays, recalibrated: flag when home share
< qwerty's (32%), not when below every named layout (dvorak/colemak are home-maximalist
designs, the wrong bar for a data-driven result).
