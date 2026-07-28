# K10 — the strongest resurrection candidate

**Finding (killed 2/3, claimed verdict WRONG, confidence VERIFIED):**
> oxey `inroll`/`outroll` credit ZERO same-row rolls — 32–63% of eligible mass silently
> unrewarded, sparing qwerty most.
> file: `src/keybo/scoring/oxey.py` :: `OxeyStyleScorer.pattern_shares` (inroll/outroll branch)

Base for every citation below: **f4c917a** (ledger text quoted from `dec1c3f`, the audit-era
base the finders used, via `git show dec1c3f:PREREGISTRATIONS.md`).

## The mechanics are NOT in dispute
All three votes ran the reproducer and it reproduced. My own independent sweep, built the way
`cli/analyze.py` builds a layout (`Layout(qwerty, ROW_STAGGERED_30)`), reproduces the finder's
census exactly:

| quantity | finder | me |
|---|---|---|
| eligible same-hand two-finger ordered pairs | 324 | **324** |
| of which SAME-ROW | 108 | **108** |
| credited by `is_inwards OR is_outwards` | 216 | **216** |
| SAME-ROW pairs receiving ANY roll credit | 0 | **0** |

Negative control on my own probe: substituting a finger-order predicate credits 108/108
same-row pairs, so the probe CAN report credit when it exists — the 0 is the predicates
gating, not my enumeration failing. (`probe_K10.py` section D.)

The dissenting verifier (vote 2, `a7722e9759429a2c6`, refuted=False) went further and showed
the branch is **load-bearing in frozen registered numbers**: it reproduces
`FROZEN_KEYBO_LSB['oxey-style']`, `FROZEN_ARCHIVE_1843`, `FROZEN_FLAGSHIP_C3` at diff exactly
`0.000e+00`, and zeroing the two roll weights moves them by −10.85..−11.00 on numbers of
magnitude 10–14.

## The grounds for the kill, checked one at a time

### GROUND A — "credited elsewhere: same-row roll credit exists as the separate TRIGRAM gauge `sr-roll`" (vote 1, leg 2) → **FAILS**
This is the only leg that addresses the finding's own question, and it is false as a defence:

- `'sr-roll'` / `'sr_roll'` occurrences in `src/keybo/scoring/oxey.py`: **0**
- `kmstats` imported by `scoring/oxey.py`: **False**
- `sr-roll` is not a key of `DEFAULT_OXEY_WEIGHTS` (keys: alternate, bad_redirect, dsfb,
  imbalance, inroll, lsb, onehand, outroll, redirect, scissor, sfb)
- `sr-roll` lives in `src/keybo/analysis/kmstats.py`, and is a member of
  `_TRIGRAM_METRICS = ('alt','roll','sr-roll','redir')` — a **trigram** metric
- `sr-roll` and `oxey-style` are **separate, co-equal entries** of `GAUGE_NAMES`

So `sr-roll` cannot credit anything inside the `oxey-style` number. **The refutation answers a
different question than the finding asked**: the finding asks "does the oxey-style SCORER
credit same-row rolls?", the refutation answers "does the FRAME price same-row rolls
anywhere?". That is exactly the brief's "a refutation that answers a DIFFERENT question"
failure mode.

### GROUND B — "ALREADY REGISTERED, verbatim, 3x" (vote 3, leg 1) → **FAILS as applied to this site**
The citations resolve on-topic at `dec1c3f`, but every one is about the **feature schema /
model input**, not the oxey-style scorer's weighted terms:

- `dec1c3f:755` — "the driver classified rolls via schema features named *roll* — none exist;
  inwards/outwards fire only on cross-row rolls, so 0 cells were labeled rolls". Context is the
  **D1 quality driver's cell classification** (it VOIDED gates b/c and forced a rerun). About a
  *driver*, not the scorer.
- `dec1c3f:1896` — "dy=0 gates angle/inwards/outwards to 0". Context is `WHY THE COLLISION
  EXISTS`, the **model feature schema's** pinky→ring/middle→ring degeneracy.
- `THEORY-1`/`DIRECTION-1` (`:6968`/`:7115`) — already ACTED ON, and the action was scoped to
  `effect_curves.py`, which **renamed its copies** to `outer_high`/`outer_low` with tests
  pinning it. The rename did NOT touch `scoring/oxey.py`, whose terms are still called
  `inroll`/`outroll`.

Positive check for the negative claim (absence is not disproof, so I looked for the positive
form): grepping the ledger for any line that ties same-row/cross-row roll gating to the
**oxey-style scorer** returns nothing. All 11 `same-row` hits are schema/model/driver context.
So the thing the refutation says is "already registered" — that the *scorer's* rewarded-roll
terms exclude same-row pairs — is **not registered anywhere**.

### GROUND C — "already one of the FOUR KNOWN-DEFECTIVE TERMS (:8188-8192); penaltyaudit's live scope" (vote 3, leg 2 / vote 1's residual) → **VERIFIES, but for a DIFFERENT defect**
`dec1c3f:8192` does register `inroll`/`outroll` as known-defective — verbatim:
> **`inroll` -2.0 vs `outroll` -1.0 asserts a 2x preference where oxeylyzer-1's REAL ported
> weights assert 4%** (+250 vs +240) ... AND the served BIGRAM vector has NO direction channel
> ... so that distinction is UNREPRESENTABLE there

That is a registration about the **relative magnitude of the two weights** (2x vs 4%) and
about the *served feature vector's* lack of a direction channel. It is NOT a registration that
the terms' **population is missing 108 of 324 eligible pairs**. Both votes treat the former as
covering the latter. It does not: you can fix the 2x→4% weight ratio and the same-row
population is still uncredited.

### GROUND D — "the 'should' is invented from a name; the docstrings state the row test AS the definition" (vote 1, leg 1) → **VERIFIES at the predicate, FAILS at the accused site**
`classify.is_inwards`'s docstring does disclose it: *"Rolling toward the index finger (outer key
on the higher row)"*, and `tests/features/test_ngram.py:181` positively pins `'as'` →
inwards==0.0/outwards==0.0. Both confirmed by reading the tree at f4c917a. So at the
**feature-predicate** layer, disclosure matches behaviour and the finder's "should" is weak.

But the finding's `file` is `src/keybo/scoring/oxey.py`, and there:
- `DEFAULT_OXEY_WEIGHTS['inroll']` = *"inward rolls rewarded: community prizes them; our data
  shows same-hand continuation is genuinely sub-additive"* — no row qualification
- `DEFAULT_OXEY_WEIGHTS['outroll']` = *"outward rolls rewarded, less than inward (community
  convention)"* — no row qualification
- the module docstring lists *"rolls (rewarded)"* among the scored classes — no row qualification
- occurrences of the string `row` in the whole of `scoring/oxey.py`: **1**, and it is
  `"scissor": (4.0, "adjacent-finger two-row reaches")`

So the scorer never discloses that its rewarded-roll population excludes a third of eligible
pairs. The disclosure the votes credit lives in a *different module* (`features/classify.py`)
that the scorer calls. That is the campaign's own recurring bug class — a name doing
load-bearing work with the check one layer away.

## What survives, and what does not
The finder's claimed verdict label **WRONG** does not survive: nothing establishes a *correct*
same-row bigram roll credit, so "the number is incorrect" is the wrong label (vote 1's
`verdict_correction` says this, and I agree). Vote 2 — the one non-refuting verifier — also
correctly caught that the finder's `-11.8..-22.3 unit` counterfactual sizes "shipped vs a FULL
finger-order redefinition" (which also relabels 108 of the 216 cross-row pairs), NOT
"shipped vs same-row-mass-added". So the finder's blast-radius magnitude is unsupported.

**But the core claim is intact and unregistered.** Correct label: **UNSUPPORTED** — a shipped
scorer term documented as rewarding "rolls" prices only 216 of 324 eligible pairs, the
exclusion is disclosed nowhere at that site, no ledger entry registers this population gap,
and the branch reproduces three frozen registered `oxey-style` constants at diff 0.0.
Blast radius: rank 4 by the audit's own scale (a shipped contract is false; no registered
number moves *today*, since the frozen constants ARE the current behaviour).
