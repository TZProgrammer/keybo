# NORMOPT-1 — what does the normgauge objective actually PRODUCE?

**Branch:** `normopt-layouts` (worktree `/tmp/normopt`, base `main` @ `96e6138`)
**Final SHA:** resolve it — `git -C /local/home/zegertho/repos/keybo rev-parse normopt-layouts` (was `70ab42b` when written; committing this report MOVES HEAD, so the branch name is the durable handle, not any SHA quoted here — HANDOFF-1 DEFECT 2). 3 commits: `1911061` (sweep) → `184536f` (amendment: the corrected control) → the report commit.
**Nothing pushed, no CR, `PREREGISTRATIONS.md` untouched, `data/models/k31/` untouched, no layout adopted.**
Prereg (registered before any cross-arm comparison): `/local/home/zegertho/agent/state/normopt/PREREGISTRATION.md`

---

# VERDICT

**Do the objectives produce materially different layouts? YES on layout identity, and the
normgauge boards are MEASURABLY SLOWER on the shipped gauge — but the number that makes that
true is one I had to discover mid-run, and it inverts my own first answer.**

Four arms, 40 runs, shipped `keybo optimize` machinery, search hyperparameters at DEFAULTS,
seeds 0–9 paired across arms. Best-of-10 per arm on its own objective:

| arm | objective | best ms/char | winner |
|---|---|---|---|
| **A** | shipped `optimize --ngram bigram` | 255.755107 | `phae,gdcnrfuoiybtsml-'j.qkvwzx` |
| **A2** | **the reported gauge** (corrected control) | **254.196774** | `po,u.gdfnlheaiycstmr'qjk-bwvxz` |
| **B** | normgauge `registered (c)` | 255.669213 | `cnrdg,aehymlstf.oiupwzxbvk-qj'` |
| **C** | normgauge 50/50 | 254.757350 | `uyo.,fpsmrhgeaidtcnlk'q-jwbvzx` |

**Against the CORRECT control (A2), the normgauge winners are WORSE on `ms/char`:**

| comparison | Δ ms/char | vs model-seed floor 0.135 | vs search-seed floor 0.883 | vs within-arm sd |
|---|---|---|---|---|
| B − A2 | **+1.472439** (worse) | 10.91× | 1.67× | 1.85× |
| C − A2 | **+1.544987** (worse) | 11.44× | 1.75× | 1.94× |

**And symmetrically, A2 is worse on the blends** — by 0.031494 (registered (c)) and 0.038506
(50/50). Each objective wins on its own ruler; neither dominates. 🟢 VERIFIED

⚠ **Both bootstrap CIs INCLUDE 0** at best-of-10 (B−A2 95% CI [−0.114, +1.627], P(B better)=0.028;
C−A2 [−0.947, +1.315], P=0.101). **This is a point-estimate reversal, not a significance claim.**
🟡 HIGH

**The single most important finding is not about normgauge at all** — see §1.

---

## 1. ⚠ MY FIRST ANSWER WAS WRONG, AND WHY — the ruler confound

I first ran arm A as the brief specified: `keybo optimize` with no `--model-weight`, i.e. the
shipped default. On the campaign's reported `ms/char` that gave **the opposite verdict** (C's
winner appeared 0.998 ms/char *faster* than A's, 7.39× the floor, CI excluding 0).

That was an artifact. 🟢 VERIFIED, two independent ways:

- **My measurement:** `spearman(bigram-table ms/char, analyze-trigram ms/char) = +0.246`
  (p=0.085) over n=50 C30M boards; **523 of 1225 pairs (42.7%) DISCORDANT.**
- **Sibling `searchparams` found the same thing independently while I was running**
  (`SEARCHPARAMS-1`, `PREREGISTRATIONS.md:10540`, landed on main as `3a80332` mid-run). Their
  spearman is 0.6715 on a different board mix — same sign, same conclusion — and they have the
  **mechanism I lacked: the CUBIC (trigram) term carries the reported gauge's variance**
  (sd 0.803 vs 0.274; `spearman(cubic,total)` 0.971 vs `spearman(quadratic,total)` 0.750). So a
  bigram-only search cannot rank optimized layouts *however many restarts you buy*.

So `keybo optimize --ngram bigram` minimizes `TableBigramScorer`, while the campaign *reports*
`analyze`'s TimeSurface (mean-over-3-bigram-seeds `T2` + mean-over-3-trigram-seeds `Tc`,
trigram-frequency weighted, ÷ covered mass). **"arm A = the ms/char control" is under-specified,
and the verdict's SIGN flips with the choice.**

**Arm A2 is the repair.** Same shipped `SimulatedAnnealing` + `two_opt`, same defaults, same seeds
0–9 — only the search scorer changes, set *to* the reported gauge. **Parity-gated at rel 1.2e−14
against `keybo analyze` on 5 known boards before use:**

```
keybo-lsb   mine 254.630749593  analyze 254.630749593   rel 1.1e-15
keybo-c30m  mine 254.590413931  analyze 254.590413931   rel 6.0e-15
arm-B       mine 253.900579104  analyze 253.900579104   rel 7.6e-15
graphite    mine 258.169563130  analyze 258.169563130   rel 1.2e-14
qwerty30m   mine 264.138916579  analyze 264.138916579   rel 7.3e-15
```

**Fixing the ruler is worth −1.558333 ms/char (11.5× the model-seed floor; bootstrap CI
[−1.983, −0.154] EXCLUDES 0, P(A2 better)=0.982) — larger than the entire objective effect I was
sent to measure.** 🟢 VERIFIED

## 2. ⚠ The floor I was told to use is the wrong floor

My brief says "median 0.135 ms/char over 91 board pairs." That is 🟢 correct as stated
(`PREREGISTRATIONS.md:10405`) but `SEARCHPARAMS-1` corrects its **provenance**: 0.135 is a
**MODEL-SEED** floor, while my replicates are **SEARCH-SEED at fixed model**. The design-matched
search-seed scale is **median |d| = 0.883 — 6.5× larger.** My own pooled within-arm
sd(ms/char) = 0.765 lands at the same order independently. Both floors are reported in the table
above. Against the correct 0.883, B−A2 and C−A2 are only **1.67× and 1.75×** — real but not large.
🟢 VERIFIED

## 3. Per-arm variance vs the between-arm effect

| arm | best | median | sd | range |
|---|---|---|---|---|
| A | 255.755107 | 256.301375 | 0.731980 | 2.097600 |
| A2 | 254.196774 | 255.861892 | 0.822215 | 2.303172 |
| B | 255.669213 | 256.064569 | 0.655311 | 1.828910 |
| C | 254.757350 | 255.813184 | 0.906584 | 3.136416 |

**Pooled within-arm sd = 0.794703 over A2/B/C (0.779022 over all four arms) = 5.9× the 0.135
floor.** Per my prereg's UNRESOLVED clause: the
B−A2 and C−A2 gaps are 1.85× and 1.94× the within-arm sd — they *do* clear the spread, but only
just, and the bootstrap CIs on the best-of-10 statistic include 0. **At this search budget (the
shipped default: `--attempts 1`, one restart) the objective choice is detectable but is the same
order as search noise.** 🟢 VERIFIED

Paired per-seed deltas (same seed each arm) on the reported gauge, arm A as the baseline:
B−A favours B on 7/10 seeds, C−A on 9/10 — *this is the inverted-ruler comparison and should not
be read as a normgauge win*; on arm A's own ruler B is worse on **10/10** and C on **9/10**.

## 4. The layouts and the 15-gauge frame

Full 15-gauge frame from the shipped `keybo analyze` (all 40 layouts + 20 field boards in
`artifacts/`; `verdict.json`, `final.json`):

```
board            sfr     sfb     sfs sfb-dst sfs-dst     lsb lsb-dst     alt    roll sr-roll   redir scissor imbal oxey-st comfort   ms/char
A win         2.6596  2.2862  9.3908  3.0337 11.4240  0.7905  1.6780 43.9733 41.3412 10.0782  3.2530  0.1626 1.2897  4.6941  4.6491 255.7551
B win         2.6596  2.1394 10.3264  2.6691 12.5207  0.9519  2.0385 45.4198 39.9501  9.7045  3.4341  0.2526 2.0393  9.8273  4.6569 255.6692
C win         2.6596  2.6436  9.0741  3.2544 11.0497  0.8520  1.7780 45.9456 38.7187  8.9493  2.7786  0.1639 2.8597  9.9713  4.9126 255.7418
keybo-lsb     2.6596  1.6231  7.6488  1.9031  8.9906  0.9219  1.8960 45.1561 41.6249 12.6921  3.3584  0.1429 2.0779 -4.1880  3.7109 254.6307
keybo-c30m    2.6596  1.6799  7.7025  1.9091  9.0440  1.7375  3.7721 44.8196 41.6862 14.0806  3.2801  0.2280 1.8456 -3.7938  3.4472 254.5904
arm-B(field)  2.6596  2.5391  6.7995  3.0423  8.0056  1.1411  2.3227 37.1373 45.4421 17.8131  4.4206  0.2567 4.8754  7.9284  3.4140 253.9006
```

**Arm means over 10 seeds, in sd(arm A) units — only TWO gauges move more than 2 sd:**

| gauge | arm A | arm B | arm C | B−A / sdA | C−A / sdA |
|---|---|---|---|---|---|
| **lsb** | 0.577±0.160 | 1.031±0.437 | 1.115±0.433 | **+2.85** | **+3.37** |
| **scissor** | 0.205±0.079 | 0.374±0.153 | 0.375±0.204 | **+2.15** | **+2.17** |
| roll | 40.53±1.89 | 43.14±4.30 | 40.46±3.02 | +1.38 | −0.03 |
| alt | 43.67±4.31 | 38.88±7.31 | 42.99±6.05 | −1.11 | −0.16 |
| oxey-style | 6.55±5.27 | 12.20±7.82 | 8.66±9.68 | +1.07 | +0.40 |
| sfb | 2.123±0.258 | 2.298±0.430 | 2.162±0.349 | +0.68 | +0.15 |
| sfs | 8.962±0.740 | 9.044±0.802 | 8.813±1.016 | +0.11 | −0.20 |
| redir | 4.396±2.814 | 5.850±2.838 | 4.954±3.006 | +0.52 | +0.20 |

**Answer to "is sfb/roll/alt materially different, or the same board with noise?" — the same
board with noise, on those three.** `sfb` moves +0.68/+0.15 sd, `roll` +1.38/−0.03, `alt`
−1.11/−0.16. **The normgauge objective buys lateral stretch and scissors:** `lsb` +2.9/+3.4 sd and
`scissor` +2.2 sd, i.e. it accepts ~1.8× the lsb mass and ~1.8× the scissor mass of the control.
🟢 VERIFIED

**sg_dist** (⚠ see §7 — the *gauge* is not on main; I computed the same quantity, corpus-weighted
mean `geometry.distance(a,c)` over trigrams, directly from the shipped geometry):
arm A 3.9539±0.0937, arm B 3.9077±0.1077 (−0.49 sd), arm C 3.8654±0.1187 (−0.94 sd) — normgauge
boards have *tighter* skipgram spans, sub-1-sd. 🟡 HIGH (my computation, not the shipped gauge).

### Systematic character — hand, row, finger

Corpus-weighted monogram mass share (mean ± sd over 10 seeds):

| arm | left-hand % | home-row % | \|L−R\| |
|---|---|---|---|
| A | 39.36±1.94 | 31.25±4.53 | 2.94±2.58 |
| B | 37.86±2.64 | 31.12±5.89 | 4.31±3.39 |
| C | 38.59±2.31 | 35.70±6.42 | 3.34±3.06 |

**P4 (registered as weak/nearly-post-hoc) does not survive as stated.** The single-probe hint
(A 28.4% vs B 37.5% home row) does not generalise: over 10 seeds arm B's home-row share is
*identical* to arm A's (31.12 vs 31.25, well inside a 4.5-point sd). Arm C is +4.5 points but that
is 0.99 sd. **No hand/row favouritism above noise.** 🟢 VERIFIED (prediction refuted)

**Per-finger TIME share (% of total predicted ms) is where a real systematic shift shows up:**

| | LI | LM | LP | LR | RI | RM | RP | RR | THUMB |
|---|---|---|---|---|---|---|---|---|---|
| arm A | 15.42 | 10.79 | 8.49 | 8.66 | 16.38 | 10.75 | 8.26 | 6.75 | 14.50 |
| arm B | 13.45 | 13.26 | 7.04 | 8.19 | 16.25 | 13.17 | 5.90 | 8.29 | 14.45 |
| arm C | 13.79 | 14.64 | 6.66 | 7.28 | 15.84 | 12.23 | 6.78 | 8.34 | 14.45 |
| B−A (sdA) | −0.84 | **+0.96** | −0.39 | −0.14 | −0.04 | **+1.13** | **−1.15** | +0.84 | −0.54 |
| C−A (sdA) | −0.70 | **+1.50** | −0.49 | −0.43 | −0.18 | +0.69 | −0.72 | +0.86 | −0.50 |
| *keybo-lsb* | 15.39 | 13.14 | 5.53 | 7.32 | 14.80 | 12.12 | 6.06 | 11.12 | 14.52 |

**Normgauge moves predicted time OFF the index fingers and pinkies and ONTO the middle fingers**
(LM +0.96/+1.50 sd, RM +1.13/+0.69, RP −1.15/−0.72), and in doing so moves *toward* the frozen
field boards' profile (keybo-lsb LM 13.14, RM 12.12 — arm B matches at 13.26/13.17 where arm A
sits at 10.79/10.75). All shifts are ≤1.5 sd individually, but the **pattern is consistent across
both normgauge arms and both hands**, which a single-gauge sd test does not capture. 🟠 INFERRED
(the direction is consistent; no individual finger clears 2 sd).

### Hamming between winners

```
        A    A2    B    C
A       0    22   29   29
A2     22     0   30   30
B      29    30    0   13
C      29    30   13    0
```

Within-arm across seeds: median 27–29 of 30. **Exactly as my prereg predicted, Hamming is large
everywhere — including between the two arms that are within noise of each other — so I do not
read it as evidence of materiality.** The informative number is B-vs-C at **13/30**: the two
normgauge weightings land far closer to each other than either does to any control (29–30).

## 5. Both directions, stated symmetrically

| | on `ms/char` (reported gauge) | on normgauge blend |
|---|---|---|
| **cost of switching TO normgauge** | B is **+1.472** worse, C is **+1.545** worse than A2 (10.9×/11.4× model-seed floor; 1.67×/1.75× search-seed floor) | — |
| **cost of staying on `ms/char`** | — | A2 is **−0.031** worse on registered (c), **−0.039** worse on 50/50 |

**P2 HOLDS: every arm wins on its own objective.** On the reported gauge A2 wins (254.197 vs
255.669/254.757); on blend registered (c) arm B wins (0.946223 vs A2's 0.914728); on blend 50/50
arm B also wins (0.946015). The search is optimizing what it was told to. 🟢 VERIFIED

**P3 CONFIRMED in direction, and it is asymmetric as predicted** — but note the two costs are on
different scales and are not directly comparable (a borrowed-ruler comparison is exactly what
NORMGAUGE-1's self-kill #6 flagged). In *relative* terms: 1.472/254.2 = 0.58% on ms, vs
0.031/0.946 = 3.3% on the blend. 🟡 HIGH

**P1 REFUTED — and this is a finding against the ledger.** I registered that B and C would be
within the floor of each other because NORMGAUGE-1 measured POOL's 0.0612 to do no observable
work. Measured: **the winners differ by 0.072548 ms/char at the winner level but are different
boards at Hamming 13/30, and each arm's best-on-its-own-blend is a distinct layout.** POOL's weight
*does* change what the search converges to, even though it does not change how the gauge *ranks* a
fixed field. ⚠ **NOT a contradiction of NORMGAUGE-1** — that entry measured reordering of an
existing field, which is a different question from what a stochastic search finds. 🟡 HIGH

## 6. TASK 4 — does normgauge reproduce a known board?

**0 of 40 exact reproductions.** P5 confirmed. But the min-Hamming distribution separates the arms
cleanly, and this is the cleanest positive signal in the whole run:

| arm | min Hamming to field | median | closest |
|---|---|---|---|
| A | 15 | 23.0 | `A-s7` is 15 from `ng:registered-best` |
| A2 | 15 | 22.5 | `A2-s8` is 15 from `archive-1843` |
| **B** | **5** | 20.0 | `B-s3` is 5 from `ng:droppool-best` |
| **C** | **4** | 20.5 | `C-s2` is 4 from `ng:droppool-best` |

**The normgauge objective re-finds the prior normgauge campaign's own best board from independent
seeds; neither control ever gets within 15 keys of anything.** And the near-misses are *the same
key multiset rotated among 4 slots*:

```
C-s2              "clndf,aeihrmstpguo.ywzxbvk-qj'"   ms/char 255.741762
ng:droppool-best  "clndf,geihrmstp.aouywzxbvk-qj'"   ms/char 255.602750   Hamming 4/30
   idx  6 (row 0 col 6, R): mine 'a'  theirs 'g'
   idx 15 (row 1 col 5, R): mine 'g'  theirs '.'
   idx 16 (row 1 col 6, R): mine 'u'  theirs 'a'
   idx 18 (row 1 col 8, R): mine '.'  theirs 'u'
   same multiset in the differing slots? TRUE  -> a 4-cycle of {a, g, u, .}
```

`B-s3` is the same picture at Hamming 5 (adds a `,`/`.` swap). 🟢 VERIFIED

**Read this as convergence evidence, not as validation of the objective:** it says the normgauge
objective has a reproducible basin that an independent search re-enters, which is a real property
of the objective. It says nothing about whether that basin is a *good* board — and on the reported
gauge `ng:droppool-best` (255.603) is 1.41 ms/char slower than arm A2's winner (254.197).

**No produced layout reproduces any of the human/community boards** (`graphite`, `semimak`) or the
flagship field (`keybo-c30m`, `keybo-lsb`, `flagship-c3`, `archive-1843/1846`, `BALL-1`,
`p16-balance`, `lsb-sib`) at Hamming < 15.

⚠ **Most of my 40 layouts are beaten on the reported gauge by frozen field boards** — `arm-B`
253.9006, `BALL-1` 253.9664, `keybo-c30m` 254.5904, `keybo-lsb` 254.6307 vs my best 254.1968.
Only arm A2's winner beats `keybo-c30m`/`keybo-lsb`; nothing beats `arm-B`/`BALL-1`. This is
consistent with `SEARCHPARAMS-1`'s finding that the campaign's boards are better-converged on the
reported gauge than the shipped default produces, and with `--attempts 1` leaving ~1.2 ms/char of
search slack. 🟢 VERIFIED

## 7. Three corrections to the brief

1. 🟢 **`normgauge` is ALREADY MERGED into `main`.** `git merge-base --is-ancestor c9e1337 main`
   → 0. `src/keybo/scoring/model_norm.py` and `drivers-normgauge/` are tracked in `main`. TASK 1's
   "rebase or cherry-pick; if it does not compose, report conflicts and STOP" was moot — there was
   nothing to compose, and no conflicts existed. The brief's `9290e9d` is also **3 commits stale**
   (branch tip `c9e1337`).
2. 🟢 **The objective is already wired into the shipped CLI** (`--model-weight` + `--model-anchors`),
   exactly as NORMGAUGE-1's own closing line states (`aba7c69`, "deliverable 2, not a driver path").
   No driver was needed.
3. 🟢 **"the full 15-gauge + sg_dist frame (`keybo analyze` prints it)" is FALSE on current main.**
   `SGDIST-SHIP-1` (`6516348`) is a **`PREREGISTRATIONS.md`-only commit** (`git show --stat`: 1 file,
   +10 lines). The gauge code (`src/keybo/analysis/skipgram_span.py` + the analyze wiring, `bbc2332`)
   is on **unmerged branch `sgdist-ship`** and is NOT an ancestor of main. `analyze` prints 15
   gauges and no sg_dist. I computed sg_dist directly instead and labelled it as mine.

Plus the two substantive corrections in §1 and §2 (the ruler, and the floor's provenance), which
came from my own measurement and from `SEARCHPARAMS-1` converging on it independently.

## 8. What is NOT covered — named, not silently dropped

- **Arm A′ via `keybo optimize --ngram trigram` was NOT run.** Infeasible under host thrash: a
  `--max-outer 200` *capped* run exceeded 550 s (vs 18 s for arms B/C) because
  **`TableTrigramScorer` ships but is wired into no CLI path** (`SEARCHPARAMS-1` verified the same;
  `grep -rn TableTrigramScorer src/keybo/cli/` returns nothing), so `--ngram trigram` falls back to
  the ~1000×-slower per-evaluation model path. **Arm A2 is my substitute and it is arguably the
  better control** (it targets the reported gauge exactly, parity 1.2e−14, rather than a
  single-seed trigram model), but it is **not** the shipped CLI path — it drives the shipped
  `SimulatedAnnealing` + `two_opt` from a ~40-line scorer.
- **No significance claim.** n=10 search seeds per arm; bootstrap CIs on B−A2 and C−A2 include 0.
- **One model seed** for arms A/A2's search (`bigram_reg31_seed0`); arm A2's *gauge* uses the
  shipped 3+3-seed means. Model-seed variance is out of scope.
- **`--attempts` held at the shipped default 1**, per the brief's "hyperparameters at DEFAULTS."
  `SEARCHPARAMS-1` measures that as ~6.1 floors worse than best-of-8, so **all four arms are
  equally under-powered** — matched, but not near converged.
- `p13stab-win` excluded from the field (not C30M charset).

## 9. Standing caveats

Everything here is a property of the **FITTED MODEL** — surfaces fitted on 4 training layouts,
**baked at 90 WPM** — never a claim about realized human typing. The normgauge frame caveat applies
throughout: the three `.standardized` surfaces share AALTO's bigram tensor and are **less
independent** than the `.native` arrays; POOL is fitted on the union of the other two. Every
normalized value is an **UPPER bound** (the `one` anchors are searched optima), so a blend of 0.946
is not "94.6% of optimal."

**This report is descriptive. It contains no recommendation to adopt a layout or to land an
objective — both remain user-gated, and the accuracy question stays where PREREG:10093 left it.**

---

## Appendix — TASK 1 gate (four bit-exact reconciliations)

| check | result |
|---|---|
| corpus + 3 surface sha256 vs anchor provenance | 🟢 all 4 match |
| `assert_direction` / `assert_matches_surfaces` / `assert_batch_invariant` | 🟢 PASS ×3 |
| **RECON A** MODELNORM-1's 10M AALTO champion fit | published `223236317224.4177` vs reproduced `223236317224.41766`, rel **−1.37e−16** |
| **RECON B** each `one` anchor == fit of its `layout_of_record` | rel **exactly 0.0** ×3, each normalizes to `1.000000000` |
| **RECON C** `zero` anchors rebuilt from (n=100, seed=20260728) | rel **exactly 0.0** ×3 |
| **RECON D** `arm-B` / `BALL-1` ms/char via `analyze` | `253.900579` / `253.966426` vs published `253.9006` / `253.966426` |
| qwerty normalizes to ~0.42–0.56, NOT ~0 | aalto-n 0.565032 / comm-n 0.462077 / pool-n 0.545283 |

RECON D is the one that caught the scale trap: my first cross-score used `TableBigramScorer` and
got `arm-B = 118.408891`, a **2.14× scale gap** from the published 253.9006. Testing sub-floor gaps
on that scale would have made every difference look 2.14× smaller than the floor allows.

**Artifacts:** `/local/home/zegertho/agent/state/normopt/artifacts/` (40 run JSONs + `verdict.json`,
`final.json`, `armA2.json`, `sgdist.json`, `crossscore.json`, `names.json`; 236 KB)
**Drivers + full results, committed:** `drivers-normopt/` on branch `normopt-layouts` (resolve the SHA with `git rev-parse normopt-layouts`)
**Recover the branch:** `git -C /local/home/zegertho/repos/keybo worktree add --detach $(git -C /local/home/zegertho/repos/keybo rev-parse normopt-layouts) <path>`
