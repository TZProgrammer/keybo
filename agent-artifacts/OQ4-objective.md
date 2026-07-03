# OQ-4 — Is predicted typing time the right objective at all?

**Status: 🔴 open — ultimately a product decision (what do YOU want the layout to be best
at?), but the design work to make any answer cheap is concrete and worth doing.**

## Best current answer

**Speed-only is a defensible v1 objective but almost certainly not the end state.** Evidence
and reasoning:

- The original paper itself concedes the speed ceiling is modest (~6% over QWERTY) and that
  comfort/strain matter to real users at least as much; the alt-keyboard community's canon
  (SFB%, scissors, LSBs, redirects, rolls) is essentially a *comfort* vocabulary. We already
  compute all of those as features — they just only influence the objective indirectly,
  through their learned effect on *time*.
- 🟡 HIGH: a pure-speed optimizer will happily accept comfort-hostile placements whenever the
  time model is indifferent (e.g. two candidate layouts within model noise but wildly
  different SFB loads). With ~25ms MAE model error, "within noise" is a big region — meaning
  comfort is currently decided by RNG, not by anyone's intent.
- The honest framing: **time is the only axis we have labels for** (the dataset measures
  when keys went down, not how they felt). Any comfort term will be theory-driven
  (community-standard penalties), not learned. That's fine — but it should be explicit,
  tunable, and OFF by default until validated, not smuggled in.

## Recommended design (cheap to adopt now, decide weights later)

A composite scorer behind the existing seam — no core changes needed:

    fitness(layout) = predicted_time(layout)            # the current learned objective
                    + λ_sfb  · Σ freq(bg)·[sfb(bg)]     # theory-driven comfort penalties
                    + λ_scis · Σ freq(bg)·[scissor(bg)]
                    + λ_lsb  · Σ freq(bg)·[lsb(bg)]
                    + λ_redir· Σ freq(tg)·[redirect(tg)]

- All λ=0 reproduces today's behavior exactly (safe default).
- Implemented as a `CompositeScorer(IScorer)` wrapping the model scorer + a cheap
  penalty scorer — both already computable from existing features. ~1 day of work.
- Optionally expose `--sfb-weight` etc. on `keybo optimize` so users can express preference.
- A true multi-objective (Pareto front of speed vs comfort, NSGA-II-style) is the deluxe
  version; start with weighted-sum, it composes with the existing SA unchanged.

## Definitive close

This closes by DECISION after one experiment makes the trade-off visible:

1. Build `CompositeScorer` (above).
2. Optimize three layouts: λ=0 (pure speed), λ=community-standard weights (moderate), and
   λ→∞ on SFB only (comfort-dominated).
3. Report the trade-off table: each layout's predicted time + SFB% + scissor% + redirect%.
   (All computable today; no new data needed.)
4. The user picks the point on the curve they actually want — that choice IS the closure.
   Record it here and in a design doc; default the CLI to it.

Note: whichever objective is chosen, the OQ-5 harness validates only the *time* term. The
comfort terms are definitionally unvalidatable with this dataset — keep them clearly labeled
as theory-driven priors, and revisit if effort/EMG-style data ever exists (Kiakl, the paper's
crowdsourcing tool, was built for exactly this and never got volume).
