# TODO — keybo backlog

Concrete work items. Conceptual/modeling questions live in `OPEN_QUESTIONS.md`; several tasks
below are **blocked on** one of those decisions and say so.

- Priority: **P0** blocks correctness / everything · **P1** correctness, needed soon ·
  **P2** important, not urgent · **P3** nice-to-have.
- State: `[ ]` not started · `[~]` in progress / partial · `[x]` done · `[?]` needs a decision.

---

## ★ North Star

Produce the *best possible* keyboard layout — where "best" means it genuinely types faster/
better **for a real user on a layout they'd master**, not merely "scores well under a model that
was fit to four QWERTY-family layouts." The gap between those two is the whole risk, and it is
what `OPEN_QUESTIONS.md` (esp. OQ-1, OQ-5) is about. **Do not trust any cross-layout speed
number until the generalization harness (P1, OQ-5) exists.**

---

## P0 — Correctness

- [x] **Scorer must include the space key.** Done, commit `dda56fd`. Was dropping 37.4% of
  corpus weight; train/serve parity restored. (audit #2)
- [x] **Don't splice non-adjacent keys across mistypes into n-grams.** Done, commit `cda3d52`.
  (audit #1)
- [x] **`same_finger` bigrams must not be flagged adjacent/scissor.** Done, commit `b4a851e`.
  (audit #3)
- [x] **De-vacuum the bug-#14 plateau regression test.** Done, commit `5c0a93d`; verified it
  fails when the bug is reverted. (audit #4)
- [x] **Harden `rotation_angle` `row_offsets` lookup.** Done, commit `1dc1ea3`. (audit #7)

## P1 — Correctness / needed soon

- [~] **Properly resolve the trigram constituent-frequency features (audit #5).** Interim fix
  landed (commit `1dc1ea3`): scorer and training now both use the 1.0 default, so it's
  *consistent and inert*, not skewed. The **real** fix is **blocked on OQ-1**:
  - If OQ-1 = "frequency is weight-only" → **delete** `bg1_freq`/`bg2_freq`/`sg_freq` (and
    reconsider the top-level `freq` feature) from the schema; bump `FEATURE_VERSION`.
  - If OQ-1 = "keep frequency as a feature" → join real corpus bigram/skipgram frequencies into
    the **training** matrix so both sides see the same values (needs a corpus lookup at train
    time), and fix the distribution mismatch (OQ-3).
  Acceptance: a train/serve parity test still passes, AND the chosen direction is justified by
  the OQ-5 experiment, not by intuition.
- [ ] **Retain the source layout label through processing (schema change).** PREREQUISITE for
  per-layout eval and leave-one-layout-out. Today `Occurrence`/`StrokeRow` keep only
  `(positions, ngram, wpm, duration)` and occurrences are pooled across all participants keyed
  on physical position, so the layout a stroke came from is **discarded**. Add the layout tag
  through `extract_occurrences` → aggregation → the TSV → `StrokeRow`. (OQ-5, OQ-8)
- [ ] **Measure the layout + proficiency distribution of qualifying participants.** One-shot:
  `load_participant_metadata` → Counter by LAYOUT, and a WPM histogram. Grounds OQ-7 (how
  imbalanced?) and OQ-8 (bucket boundaries) with real numbers instead of guesses. Cheap; do
  early. (OQ-7, OQ-8)
- [ ] **Generalization + sliced-eval harness (OQ-5, OQ-8).** Two capabilities:
  (a) **leave-one-layout-out**: train on 3 layouts, predict the 4th, report ranking/time error
  on the held-out one; (b) **sliced metrics**: a {layout × WPM-bucket} → {R², MAE, ranking-err}
  matrix, surfacing the *worst* cell, not just the mean. This is the tool that answers OQ-1 and
  grounds every "X% faster" claim. **Highest-leverage item after the P0 bugs** (needs the schema
  change above for the layout axis; the WPM axis works today). Acceptance: `keybo` reports the
  matrix + held-out ranking; wired into `just`.
- [ ] **Decide how to leverage imbalanced non-QWERTY data (OQ-7).** Compare inverse-frequency
  sample weighting vs. resample-with-replacement vs. none vs. non-QWERTY-as-holdout, judged by
  the *per-layout* held-out metrics from the harness above. Prefer weighting over resampling
  (resampling inflates minority-layout confidence). Interacts with OQ-1 (do it after).
- [ ] **Decide `freq`-as-feature (OQ-1) using the harness above.** Then execute the P1 #5 fork.
- [ ] **Verify the full real-data pipeline end to end.** Run `fetch-data → process-data →
  train → score` on the actual 136M dump (not synthetic fixtures) and sanity-check the trained
  model's per-layout / per-bucket held-out metrics + the layout ranking. Nothing has been run
  on real data yet.

## P2 — Important, not urgent

- [ ] **Delta-scoring for `optimize` (perf).** The scorer re-scores the entire corpus on every
  candidate swap (~25 ms/eval bigram, worse for trigram), so full-corpus `optimize` is very
  slow and currently needs `--max-outer`. Incremental delta-scoring (only re-score n-grams whose
  moved keys participate in) fits behind the existing `IScorer` interface — an optimization, not
  an API change. Biggest usability win; gates comfortable full-corpus runs for BOTH bigram and
  trigram. (audit #10 — hoist the invariant `fitness` out of the 2-opt inner loop — folds in.)
- [ ] **User-configurable objective corpus (OQ-3).** Let a user weight the objective by their
  own text (prose/code/other language) instead of only iWeb. Plausibly the highest-value
  "best layout for ME" capability. Needs a documented way to generate frequency files.
- [ ] **Push the current work + get it on the laptop.** ~7 unpushed commits on `main`
  (trigram-CLI wiring + 5 audit fixes). `git push origin main`, then `git pull` on the laptop.
- [ ] **Validate `nix develop` on the laptop.** `flake.nix` was authored by mirroring gen-ai but
  never run (no nix on the dev box). First `nix develop` → `just doctor` is the real test; if a
  wheel can't find a `.so`, add it to `runtimeLibs` in `flake.nix`.

## P3 — Nice-to-have / after the above

- [ ] **Multi-objective / comfort terms (OQ-4).** Consider optimizing effort/comfort (SFBs,
  scissors, redirects as explicit costs) alongside or instead of pure predicted time; possibly
  Pareto rather than single-scalar.
- [ ] **Remove dead filter branches in keystroke extraction.** `group_sessions` already drops
  multi-char/`SHIFT` rows, so the re-checks in `extract_occurrences` are dead on the CLI path
  (live only when called directly, as unit tests do). Minor. (audit #9)
- [ ] **Assert the global optimum, not just a local one, in the 2-opt test.** The test asserts
  "no single swap improves" (local); the plan claims "reaches the known minimum" (global). They
  happen to coincide on the test landscape — assert the global value to lock it. (audit #8)

---

## Recently completed (this session)

Full rewrite (10 phases) · nix/uv/just tooling mirroring `~/repos/gen-ai` · `src/` layout ·
`fetch-data` downloader (resumable, verified against the real Aalto server) · trigram model
wired into `optimize`/`score` CLIs · independent audit + all 5 confirmed findings fixed
(TDD, test-first). 174 tests, ruff clean. See git log and `docs/`.
