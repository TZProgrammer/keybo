# Design Audit — 2026-07-03 (post-rewrite, post-bug-fix state)

Scope: the whole project as it stands (17 commits since the rewrite began; 176 tests green).
Purpose: what can still be improved, which gaps remain, what is not yet implemented, which
bugs we may not have caught, and which pre-rewrite features never made it across. Written to
surface the unknown-unknowns — each item says how it was established (🟢 ran it / 🟡 read it /
🟠 inferred) and what would resolve it.

A second independent adversarial audit of the post-fix delta is running in parallel
(subagent `keybo-delta-audit`); its report will land at
`state/keybo-delta-audit/delta-report.md` and may add to the "bugs not yet caught" section.

---

## A. Bugs found DURING this audit (fixed already)

- 🟢 **Control keys did not break n-gram windows (fixed, commit 64f0a32).** The splicing fix
  (cda3d52) caught gaps from flagged-incorrect *characters*, but `group_sessions` silently
  pre-dropped multi-char rows (BKSP/SHIFT/arrows) before extraction — so `a <BKSP> b` still
  spliced into `('ab', 4000ms)` on the real CLI path. Found by attacking my own fix with the
  question "what else creates a gap that the index check can't see?". Control keys now stay
  in the stream (creating index gaps); only single-char rows join the difflib alignment.
  Lesson recorded: *a fix verified at the unit layer can still be bypassed one layer up* —
  the CLI-path repro is the one that counts.

## B. Legacy features NOT carried over (🟢 verified against legacy/)

| Legacy capability | Where it lived | Status in rewrite | Recommendation |
|---|---|---|---|
| **Cross-validation during training** (5-fold KFold, CV R²/MAE printed) | `wpm_conditioned_model.cross_validate_model` | **Missing** — `keybo train` fits and saves, reports nothing | Fold into the OQ-5 harness (better than restoring as-was: CV on pooled rows leaks participants across folds) |
| **`--hyperparams_file`** (train with tuned params from `tune`) | `wpm_conditioned_model.main` | **Missing** — `keybo tune` writes best_hyperparams.json but `keybo train` cannot consume it! The tune→train loop is broken | Add `--hyperparams <json>` to `keybo train`. Small, do soon |
| **Trigram corpus coverage trimming** (`tg_coverage`: use top-N% of trigrams) | `legacy/optimizer.py` | **Missing** — trigram scorer always uses ALL ~100k corpus trigrams | Reintroduce as `--coverage 95` on score/optimize: directly mitigates the trigram perf pain (top ~5k trigrams ≈ 95% of weight); pairs with delta-scoring |
| **Fit plots** (freq-vs-time scatter + fit line) | `plot_results` in both model scripts | **Missing** (matplotlib is a dep but unused) | Low priority; add a `validate --plot` once the harness exists |
| **Multi-attempt optimize + best-of logging** (`num_attempts=10`, appended to logfile.txt) | `legacy/simulated_annealing.py` | **Missing** — `keybo optimize` runs ONE seed and prints | Add `--attempts N` (SA is seeded; trivially parallelizable) + write results to a run log. Restores real search power |
| **Named layouts: pinev4, semimak-JQ, tusk, threa** | `legacy/print_stats.py` | **Partial** — new registry has qwerty/dvorak/colemak/graphite/semimak; the project's OWN legacy layouts (tusk, threa) are gone | Add them back to `layouts.py` (2 lines each; they're this project's history) |
| **2-opt/3-opt as standalone entry points** | `two_opt.py`, `three_opt.py` | **Partial** — two_opt runs inside optimize; three_opt exists but NO CLI reaches it | Add `--local-search {none,2opt,3opt}` to optimize |

## C. Gaps / not-yet-implemented (verified 🟢 unless noted)

1. **No layout label in processed data** — blocks per-layout eval, LOLO, OQ-1/5/7/8. THE
   schema prerequisite. (Also: no participant id retained → participant-level bootstrap and
   leakage-safe splits are impossible. Add BOTH columns in the same change.)
2. **`target_wpm` is not validated against the model's trained `wpm_range`.** Metadata
   carries `wpm_range` but nothing checks it: `keybo score --target-wpm 200` silently
   extrapolates (trees clamp at the training boundary — the same class of bug as the old
   repo's "optimize for 100 WPM" illusion, audit-era finding #8). Cheap fix: warn/error in
   `build_scorer` when target is outside `metadata.wpm_range`.
3. **Uppercase/shifted characters are out of scope but silently so** — 6.1% of corpus weight
   (`' I'`, `'Th'`…) is dropped with no accounting. Correct behavior (a 30-key lowercase
   board can't type them without a shift model), wrong silence. Log a one-line coverage
   report in `score`/`optimize` ("scored 93.9% of corpus weight; dropped: capitals 6.1%").
4. **`time_mode` semantics are undocumented and unvalidated** (`full` = first-press→last-press
   of the window; `last` = inter-key gap of the final pair). Which one the bigram model
   SHOULD train on is actually a modeling question (full window time for a bigram includes
   the first key's own press latency). Legacy defaulted `full`; we kept it. Document, and
   let the OQ-5 harness compare both on held-out error. 🟠
5. **No delta-scoring** (known, TODO P2): full-corpus optimize impractical, trigram worse.
   With B's coverage-trim restored, trigram becomes usable even before delta-scoring lands.
6. **No `corpus-freqs` generator** (OQ-3b): per-user objective corpora need a supported tool.
7. **No composite/comfort objective** (OQ-4): design proposed in the OQ-4 artifact; λ=0
   default keeps it non-breaking.
8. **`keybo tune` output loop is broken** (see B: train can't consume it) — tune currently
   produces a file nothing reads. 🟢
9. **Model artifacts carry no data provenance** — metadata lacks the strokes-file
   hash/participant count/date. When we start comparing models (harness), we won't know what
   data each was trained on. Add `extra={"strokes_sha256": ..., "n_rows": ...}` at train time.
10. **`fetch-data` verifies size, not content** — no checksum of the Aalto zip (server sends
    no ETag-as-hash). Record the SHA-256 of the first good download in the repo and verify
    subsequent downloads against it (supply-chain + corruption guard). 🟡
11. **CI runs unit tests only** — no smoke of the four CLI workflows on fixtures (they're
    tested via pytest, but a `just doctor`-style CLI smoke in CI would catch packaging/entry-
    point breaks the unit layer can't see).

## D. Bugs we may NOT have caught (risk register, honest)

- 🟠 **`mark_correct_flags` alignment quality.** difflib longest-match alignment is a heuristic;
  on heavily-corrected sessions it can mark plausible-but-wrong subsequences "correct" (the
  legacy code had the same property; the paper describes a more careful windowed alignment
  that neither implementation fully realizes). The contiguity fix bounds the damage (no
  cross-gap durations), but flag *assignment* itself is untested against pathological
  sessions. Mitigation: property-test with adversarial typo patterns; compare emitted n-gram
  counts against hand-computed truth on ~20 crafted sessions.
- 🟠 **Duplicate-key `Occurrence` aggregation across layouts.** Aggregation keys on
  `(positions, ngram)`; two layouts can map the same ngram to the same positions (e.g. QWERTY
  and QWERTZ share most letters) — fine — but WPM distributions differ by cohort, and pooling
  hides that. Not a bug per se; becomes visible (and fixable) with the layout column.
- 🟠 **XGBoost nondeterminism across versions/threads.** Reproducibility tests cover the SA
  (seeded RNG) but model *training* determinism across machines is unpinned (hist tree method
  is deterministic per version single-threaded, but `n_jobs` default varies). If exact
  reproducibility of a published layout matters, pin `n_jobs=1` + xgboost version in the
  model metadata.
- 🟠 **The `wpm` feature at serve time is a constant column** (target_wpm) while at train time
  it varies per row — structurally the same class of quirk as the freq feature (a
  train-varying, serve-constant column). Benign for ranking within one run (constant offset)
  but worth a schema note; the OQ-5 harness will quantify whether per-WPM slopes transfer.
- 🔴 **Unknown-unknowns in the real-data path.** Everything data-side is validated on synthetic
  fixtures; the real 136M dump has encodings, malformed rows, and participant quirks fixtures
  don't model. The first real `process-data` run (TODO P1) should be treated as a test: run
  with a row-rejection counter and eyeball the rejection breakdown.

## E. Design improvements worth making (non-bug)

1. **Optimize ergonomics:** `--attempts N` (restore legacy best-of-N), `--out <file>` to save
   the best layout + its score as JSON (today: stdout only — a run's result isn't durable),
   and a `--progress` flag (legacy printed per-iteration fitness; new SA is silent).
2. **Coverage trimming as a first-class knob** (see B) — biggest practical perf lever
   available without touching the scorer.
3. **`Layout` serialization** — `Layout.from_string`/`to_string` exist implicitly (chars),
   but a saved-layout JSON (chars + geometry name + provenance) would make optimize outputs
   self-describing and scoreable later.
4. **Split `keystrokes.py`** (~260 lines, parsing + alignment + windowing + orchestration) if
   it grows further — it's the file every data question touches.
5. **Property-based testing** (hypothesis) for layout swap/undo invariants and feature-vector
   invariants (e.g. one-hots sum to 1, mutual exclusions) — the scissor/same_finger
   contradiction is exactly the class a property test finds and example tests miss.

## F. Prioritized "if you do only five things next"

1. **Schema change: layout + participant columns** (unblocks OQ-1/5/7/8 — everything).
2. **OQ-5 validation harness + real-data run** (the keystone; folds in legacy CV restoration).
3. **`keybo train --hyperparams`** (un-break the tune→train loop; 30 min).
4. **Trigram coverage trim + optimize `--attempts/--out`** (usability of the actual product).
5. **Decide OQ-1 with the harness**, then execute the freq-feature fork (delete vs join).
