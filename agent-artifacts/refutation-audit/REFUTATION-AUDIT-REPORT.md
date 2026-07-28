# refaudit — the first audit of a refutation

**Base:** `/tmp/refaudit` @ `f4c917a`, branch `refutation-audit` (drivers committed through
`5313015`). Ledger quotes are from `dec1c3f`, the audit-era base the original finders used.
Nothing pushed, no CR, `PREREGISTRATIONS.md` untouched.

---

## (0) The count — pinned, and the report was wrong

`journal-digest.json` stores 37 findings and 110 verdicts as **two flat lists with no key
between them**: the finding→verdict join was lost, which is why nobody could audit a
refutation. The join is recoverable from the raw per-agent transcripts (which survive in the
session dir): every verify agent's first user message embeds `## THE FINDING UNDER TEST`, and
its `StructuredOutput` call carries the verdict.

**37 distinct findings verified · 23 SURVIVED · 14 KILLED.** 🟢 VERIFIED
- kill votes: **2/3 ×8, 3/3 ×6** · survivor votes: 0/3 ×12, 1/3 ×11
- kill rule (`wf-ultraaudit.js`): `survives = good.length > 0 && nRefuted < 2`
- 111 verdict agents, all returned.

So **the callback's "the 14 REFUTED findings" is RIGHT and the report's "~19 findings died"
is WRONG.** Diagnosis of the "~19": 23 triage agents ran, but the digest carries only **19
triage records** — the report read the digest's truncated triage count as a kill count. A
triage count can never be a kill count in the first place: triage runs only on *survivors*.

Two further digest losses found while pinning this:
- **111 verdict agents but 110 verdict records.** The dropped one is a *non-refuting* vote
  (on `test_score_coverage.py stays 4-PASSED`, agent `aa04228cfad0c3d8f`), so 45-refutations
  and 14-kills are unaffected — but the digest's own 65/45 split should read 66/45.
- **4 lost triage records** (23 agents → 19 records).

Census cross-check: 159 workflow cache keys, 158 with a result, **1 DEAD** (finder
`gauges-community-ports`, 6 attempts, never returned → that remit went unexamined), 168 files
on disk (= attempts including retries). Matches the report's "159 agents (158 done, 1 errored)".

**Harness controls** (`positive_control.py`): my 23/14 matches the prose report and the
profiles-index headline, both written independently; a mutation control MOVES the tally
(flip a refuting vote on a 2/3 kill → 24/13; null a survivor's panel → 22/15); the parser
refuses a renamed header; 37 of 37 titles distinct with no `None`.

---

## (1)–(3) Every kill triaged: 14 of 14 checked, 13 verify, 1 fails

Full per-kill detail with depth codes and exactly how each was checked:
`artifacts/coverage-ledger.txt`. Full vote text: `artifacts/killed-dossier.md`.

| # | finding (killed) | votes | stated ground for the kill | verdict |
|---|---|---|---|---|
| K1 | blend-v1 "prose" is 63–71% one machine-written file | 2/3 | an existing gate DOES bite the fault | **VERIFIES** (live mutation) |
| K2 | manifest's skipgram identity violated on 2854/4094 | 3/3 | residue is documented largest-remainder rounding | **VERIFIES** |
| K3 | `same_hand_other`'s "defining SHAP feature" over-covers | 2/3 | docstring says "over the class's pairs"; metric mixes polarities | **VERIFIES** |
| K4 | SELECT-1 raw-support tiebreak holds at 2 of 23 thresholds | 2/3 | swept the CORRECTED column; registered h2h used HISTORICAL | **VERIFIES** |
| K5 | `FREQ_PRIOR` ships a 2-of-3 panel undisclosed | 2/3 | already registered; the "82×" divides incommensurable units | **VERIFIES** |
| K6 | `comfort` divides by the FULL corpus mass | 3/3 | the headline % is an algebraic identity; comfort cancels | **VERIFIES** |
| K7 | effect-curves' WPM axis runs outside `wpm_range` | 3/3 | `wpm_range` is an unfitted literal; 50/130 are validated midpoints | **VERIFIES** |
| K8 | `KNOWN_LAYOUTS['mtgap']` is not a layout core | 3/3 | the counting proof's premise is false of the real capture | **VERIFIES** |
| K9 | iWeb `1-skip.txt` is charset-truncated, not a different pass | 3/3 | two arithmetic impossibilities | **VERIFIES** |
| **K10** | **oxey `inroll`/`outroll` credit ZERO same-row rolls** | **2/3** | **"credited elsewhere as `sr-roll`" + "already registered"** | **FAILS → RESURRECTED** |
| K11 | `tune_lolo`'s tau gate eliminated 0 of 24 | 2/3 | the docstring self-documents a max-over-candidates bar | **VERIFIES** |
| K12 | pinky guard evaluated on a space-excluded marginal | 2/3 | the bar IS a normalized 8-finger share; the "gap" is a pure rescale | **VERIFIES** |
| K13 | golden npz's trigram half pins `c` to (3,3) | 2/3 | the 3.23% denominator is wrong — most columns are bigram compositions | **VERIFIES** |
| K14 | K31 gate A asserts on `ROW_STAGGERED_30` | 2/3 | G30 and G31 are the same feature-computing object | **VERIFIES** |

### The one resurrection: K10 🟢 grounds verified

> `oxey inroll/outroll credit ZERO same-row rolls` — `src/keybo/scoring/oxey.py`,
> `OxeyStyleScorer.pattern_shares`. Killed 2/3. Full analysis: `artifacts/K10-analysis.md`.

Four grounds, checked one at a time:

- **GROUND A (decisive) FAILS.** Vote 1: *"Same-row roll credit exists in the frame as the
  separate TRIGRAM gauge `sr-roll`."* But `sr-roll` occurs **0×** in `scoring/oxey.py`,
  `kmstats` is **not imported** by it, `sr-roll` is **not** a key of `DEFAULT_OXEY_WEIGHTS`,
  it is a member of `_TRIGRAM_METRICS` in `analysis/kmstats.py`, and `sr-roll` and
  `oxey-style` are **separate co-equal entries** of `GAUGE_NAMES`. It cannot credit anything
  inside the `oxey-style` number. **This is the brief's "a refutation that answers a DIFFERENT
  question"**: the finding asked *does this SCORER credit same-row rolls*, the refutation
  answered *does the FRAME price them anywhere*.
- **GROUND B FAILS as applied.** Vote 3's three "already registered, verbatim" cites resolve
  on-topic at `dec1c3f` but are all about the **feature schema / D1 driver / effect_curves**:
  `:755` is the D1 quality driver's cell classification, `:1896` is the model schema's
  `dy=0` degeneracy, and THEORY-1/DIRECTION-1's rename landed in `effect_curves.py` (whose
  copies became `outer_high`/`outer_low`) while the scorer's terms are still
  `inroll`/`outroll`. Checked positively, not by absence: no ledger line ties same-row roll
  gating to the oxey-style scorer.
- **GROUND C verifies, but registers a DIFFERENT defect.** `dec1c3f:8192` does list these
  terms among "THE FOUR KNOWN-DEFECTIVE TERMS" — for the **weight ratio** (`-2.0` vs `-1.0`
  = 2× where oxeylyzer asserts 4%) and the served vector's missing direction channel. Fix the
  ratio and the 108-pair population gap is still there.
- **GROUND D verifies at the predicate, FAILS at the accused site.** `classify.is_inwards`'s
  docstring does say *"outer key on the higher row"* and `test_ngram.py:181` pins
  `'as'` → 0.0. But in `scoring/oxey.py` the labels are *"inward rolls rewarded"* /
  *"outward rolls rewarded, less than inward"* and the module docstring lists *"rolls
  (rewarded)"* — the string `row` occurs **once** in the whole file, in the scissor label.

Independent re-derivation reproduces the finder's census exactly: **324** eligible same-hand
two-finger ordered pairs, **108** same-row, **216** credited, **0** same-row credited.

**Real status: UNSUPPORTED, rank 4.** Not the finder's `WRONG` (nothing establishes a
*correct* same-row credit), and not refuted. Load-bearing: the branch reproduces
`FROZEN_KEYBO_LSB`/`ARCHIVE_1843`/`FLAGSHIP_C3` `oxey-style` at diff exactly 0.0 — that was
the *non-refuting* verifier's own positive control.

---

## (2) The specific failure mode I was sent to hunt

**A wrong constant supporting a true conclusion — found 3, none fatal.** All three are in
kills whose conclusions independently verify, so none is a false refutation; all three are
quotable-number defects that would propagate if inherited:
1. **K9 vote 1: "3129 of 3474"** — the true figures are **3128 of 3473** (off by one). The
   impossibility argument is unaffected.
2. **K13: votes 2 and 3 say "38 of 46" and "43 of 46"** derived-trigram columns. The exact
   count is **42** (19 `bg1_` + 19 `bg2_` + 4 `sg_`). **Neither vote was right**, though the
   direction (the finder's 3.23% denominator is wrong) holds.
3. **K5 vote 1's own `verdict_correction`** already caught that the finder's 300-permutation
   mean-inflation figure (+0.4944 pp) reproduces at no seed. Recorded because it is the same
   class and it was caught *inside* the audit.

**Ledger line-number citations: the refuting votes are CLEAN, unlike the 23 confirmations.**
All 23 confirmed findings had stale citations (drift +2..+44). I resolved all 25 distinct
line numbers cited by refuting votes at both shas: they resolve **on-topic at `dec1c3f`**
(`artifacts/ledger-cites.txt`). One anomaly — K1v3 cites "50023", out of range at both shas
(8209 / 9277 lines) — a typo in a leg that also cites trap #6, which exists.

**Trap citations:** all 12 distinct traps cited (#1, 6, 9, 14, 17, 19, 23, 31, 41, 44, 45,
48) exist in `TOOLING-TRAPS.md` and their titles match the use.

**Claimed artifacts (`ls`-ed, not trusted):** 65 distinct paths — 36 exist, 19 exist in-repo
at `dec1c3f`, **10 missing**. Nine of the ten are expected-dead (`/tmp/ua-mut-*` mutation
copies, `/tmp/ua-preK31`, `workspaces/keybo-selmethod` = trap #14), per the audit's own
cleanup note. **The real gap: 3 scratch driver scripts** cited under abbreviated
`/local/.../` paths (K7v3 ×2, K10v1 `a1_angle_identity.py`) **are absent** — those legs are
unrecheckable *from the artifact*. I re-derived K7's ground from source instead, and K10's is
attack A3 below.

**Shared-component check:** K13 and K14 rest on ONE shared structural claim (`.slots` is
never read by the feature pipeline). I flagged that both would fall together, then verified
it independently: grep returns nothing, G30/G31 differ in that one dataclass field, and
`max|features(G30) − features(G31)| = 0.0` exactly.

---

## Self-separation: what I killed of my own

`probe_K10_hostile.py` — four attacks on my own resurrection, strongest first:

- **A1 "GROUND A was never load-bearing"** — the strongest attack. A 2/3 kill survives if
  either refuting vote is independently sufficient. I walked every leg of both votes: the
  **surviving** legs (the invented "should", the tautological 50% figure, the finder's
  misquote of the `effect_curves` docstring) all attack the finder's *framing* and its
  *supporting* arguments. Only two legs were aimed at the core factual claim — vote 1's
  `sr-roll` leg and vote 3's already-registered leg — and both fail. The kill does not stand,
  **but the resurrected finding is narrower than the finder wrote.**
- **A3 "my check shares a component with its target"** — I used the accused predicates to
  show same-row pairs get no credit. Re-derived from **row equality alone**, no predicate
  call: reading their source, both return a *strict* row comparison, algebraically False when
  rows are equal. 108 of 324 = 33.3%, established without inheriting anything from the
  predicates.
- **A2 "disclosed one call away"** — conceded in part, and it is why the label is
  UNSUPPORTED/rank-4 rather than the finder's WRONG.
- **A4 scope honesty** — **I resurrect the population/disclosure claim ONLY.** The finder's
  "32–63% of eligible **mass**" and "sparing qwerty most" I did **not** verify, and vote 2 —
  the finding's own *supporter* — showed the counterfactual sizes a *full finger-order
  redefinition* (which also relabels 108 of the 216 cross-row pairs), not
  same-row-mass-added. **The magnitude half of K10 stays dead.**

**Three of my own controls failed first and were fixed** (all recorded in the drivers, since
a control that cannot fail is worthless):
1. The mutation control nulled the panel of an *already-killed* finding — a no-op that
   reported a false "HARNESS BLIND". Retargeted at a survivor.
2. The G30/G31 control perturbed `row_offsets` by a **uniform** constant, which cancels
   (offsets enter only inside differences), reporting a false "COMPARISON BLIND". Fixed to
   perturb one row (features move 1.527e+02); the uniform shift is kept as a second control
   asserting the predicted 0.0.
3. `probe_K08.py` first regexed boards from raw zip bytes and recovered **0** registry
   matches — its control caught that it was reading the shifted-repeat half. Rewritten to use
   the repo's own loader path; qwerty/colemak/colemak-dh then recover exactly.

---

## Coverage, stated plainly

**14 of 14 kills triaged. 13 grounds verify. 1 fails (K10, resurrected as UNSUPPORTED,
narrowed to its population/disclosure claim).** Every probe that reports "no problem" was
first shown able to report one. Every probe asserted `keybo.__file__` under `/tmp/refaudit`
via `preflight.py`, which carries its own negative control.

**What I did NOT independently re-run** (each inside a kill whose other grounds verified):
- K4's 23-threshold *historical* min_cell sweep (needs a ~14 min `rs.pkl` rebuild). I
  verified its ledger basis instead: `dec1c3f:5098` registers "tri-serve 13.9%" = the
  historical 13.8625, and `:5236` states the registered h2h used historical semantics.
- K11's 200k-pool lexicographic-equivalence test. I verified the decisive leg directly: the
  docstring says "the maximum achieved by any candidate" and the code is
  `best_tau = max(r[2] for r in results)` — verbatim.
- K10's corpus-mass figures — deliberately left dead (A4).

**One structural gap this audit cannot close:** the 14 kills are the findings that *reached* a
panel. The DEAD finder (`gauges-community-ports`, 6 attempts, no result) raised nothing, so
its remit was never examined by anyone — that is a coverage hole in the original audit, not a
refutation I can check.

---

## Verdict on the asymmetry the campaign named

The child's stated worry was that *"a refutation rested on a wrong-constant-supporting-a-true-
conclusion"* and killed a real defect invisibly. That worry was **justified in mechanism and
almost right in outcome**:
- The mechanism it feared is real and I found it three times (K9, K13 ×2) — but in all three
  the conclusion held, so no finding died of it.
- The panel actually lost a finding to a **different** mechanism: **a refutation that answers
  a different question than the finding asked** (K10's `sr-roll` leg), reinforced by an
  **"already registered" claim that cites the right words at the wrong component** (K10's
  ledger legs). Both are label-vs-referent errors — the exact bug class the audit existed to
  hunt, committed by the audit's own refuters.
- Base rate: **1 of 14 kills (7%) had a failing decisive ground.**

⚠ **A comparison I drafted and then killed, because it was the very error I was hunting.** I
first wrote that this 1-of-14 beats "the 4 defects the warm self-audit found in its own 23
confirmations", concluding the refutation panel was *more* precise than the confirmation path.
That is **not a valid comparison** and I withdraw it: reading `ULTRAAUDIT-SELFAUDIT` at
`f4c917a`, those four defects are **report-level**, not four refuted confirmations — (a) a
mislabelled constant in the prose (the √SB column), (b) all 23 line citations stale, (c) an
independence caveat on three findings, (d) one remit mislabelled COVERED instead of PARTIAL.
Not one of them is "a confirmed finding that was actually false". So the two rates count
different things, and pairing them would have been a number supporting a true-sounding
conclusion — the exact shape of the three defects I logged above, committed by me. **The
honest statement is the bare one: 1 of 14 kills was wrong. I have no comparable
false-confirmation rate to set against it, because nobody has re-litigated the 23
confirmations as *findings* the way I have now re-litigated the 14 kills.**
