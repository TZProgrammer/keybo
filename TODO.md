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

- [x] **Properly resolve the trigram constituent-frequency features (audit #5).** DONE via
  the OQ-1 measured closure (commit `61a3d5d`): ALL frequency features deleted from the
  schema — the bigram `freq` and the trigram `tg_freq`/`bg1_freq`/`bg2_freq`/`sg_freq` —
  `FEATURE_VERSION` bumped to `2026-07-05.1` (pre-bump models refuse to load). The landmine
  dies structurally: there is no frequency column left to diverge. In its place, `keybo
  train` fits the harness-validated R1W recipe by default: an explicit additive per-bigram
  practice term (shrunk backfit, residualized out of the geometry model's target, stored in
  the model metadata) + layout-balance example weights. Parity tests updated; direction
  justified by the OQ-5 experiment as required (τ +0.667→+1.0, beats-baseline 12/12).
- [x] **Retain the source layout label through processing (schema change).** Done (commits
  `10c1fa9`/`a4e6f7a`/`5e8b618` produce side, `4a936e7` consume side): `Occurrence` and
  `StrokeRow` carry `layout`; samples are `(wpm, duration, pid, hold)` 4-tuples; the TSV is
  layout-first (old-format files are detected by first byte and rejected with the fix named);
  rejection counters print from `process-data`. `hold` carries OQ-11's feature candidate
  forward. (OQ-5, OQ-8, OQ-11)
- [x] **Measure the layout + proficiency distribution of qualifying participants.** Done
  twice over: metadata counter (commit `c5ea7a5`: 98.68% qwerty, dvorak n=77) and the v3
  stroke table's per-layout volumes (qwerty 31.2M samples / 54,690 participants; qwertz 277k /
  485; azerty 92k / 166; dvorak 37k / 64). (OQ-7, OQ-8)
- [~] **Generalization + sliced-eval harness (OQ-5, OQ-8).** LOLO harness DONE (commit
  `3eb7d0f`): `keybo validate` / `just validate` — split-half noise ceiling first
  (participant-level bisection), decisive layout-level Kendall's τ incl. a pooled
  fully-out-of-sample τ, bucket-centered per-cell ρ supplementary, distance+wpm linear
  baseline floor, ≥3 seeds, discrimination-tested against synthetic lawful/lawless worlds.
  Remaining for OQ-8: the full {layout × WPM-bucket} → {R², MAE} worst-cell matrix as a
  first-class report (cells exist; the slicing report doesn't yet).
- [x] **Rejection-counter breakdown in process-data.** Done (commit `5e8b618`); first real
  readout: kept 31.6M | non-contiguous 1.43M | off-layout 439k | banned/multi-char/bad-time 0.
  (Supersedes the 5%-smoke-test item: the counters now make every full run self-auditing, and
  three full-dump passes have completed without incident.)
- [ ] **Decide how to leverage imbalanced non-QWERTY data (OQ-7).** Compare inverse-frequency
  sample weighting vs. resample-with-replacement vs. none vs. non-QWERTY-as-holdout, judged by
  the *per-layout* held-out metrics from the harness above. Prefer weighting over resampling
  (resampling inflates minority-layout confidence). Interacts with OQ-1 (do it after).
- [ ] **Decide `freq`-as-feature (OQ-1) using the harness above.** Then execute the P1 #5 fork.
- [x] **Verify the full real-data pipeline end to end.** Done three times on the dev box
  (2026-07-04): v1 (pre-quote-fix), v2 (quote-clean; optimized layout +5.18% vs qwerty,
  +0.93% vs semimak), v3 (new schema + LOLO validate). Named-layout ordering sane and stable
  across runs: semimak > graphite > dvorak > colemak > qwerty. Notable: v1/v2 optimized
  layouts share only 6/30 positions yet cross-score within ~0.5% under both models — the SA
  optimum is a wide plateau; search + validation beat more restarts.

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
- [ ] **Get the current work on the laptop.** `git pull` on the laptop (main is pushed
  regularly from the dev box). Note the schema change: any laptop-local `bistrokes.tsv` from
  before 2026-07-04 must be regenerated (`just process-data ...`) — train/validate will
  refuse the old format by design.
- [ ] **Validate `nix develop` on the laptop.** `flake.nix` was authored by mirroring gen-ai but
  never run (no nix on the dev box). First `nix develop` → `just doctor` is the real test; if a
  wheel can't find a `.so`, add it to `runtimeLibs` in `flake.nix`.

## P3 — Nice-to-have / after the above

- [ ] **Multi-objective / comfort terms (OQ-4).** Consider optimizing effort/comfort (SFBs,
  scissors, redirects as explicit costs) alongside or instead of pure predicted time; possibly
  Pareto rather than single-scalar.
- [ ] **Remove dead filter branches in keystroke extraction.** Note: after the contiguity
  fixes, `group_sessions` now KEEPS control-key and empty-LETTER rows, so `extract_occurrences`'
  single-char and banned-key window filters are genuinely load-bearing on the CLI path — the
  original "dead branch" observation (audit #9) no longer applies as written. Re-evaluate
  which (if any) checks are still redundant before removing anything.
- [ ] **Assert the global optimum, not just a local one, in the 2-opt test.** The test asserts
  "no single swap improves" (local); the plan claims "reaches the known minimum" (global). They
  happen to coincide on the test landscape — assert the global value to lock it. (audit #8)

---

## Recently completed (this session)

Full rewrite (10 phases) · nix/uv/just tooling mirroring `~/repos/gen-ai` · `src/` layout ·
`fetch-data` downloader (resumable, verified against the real Aalto server) · trigram model
wired into `optimize`/`score` CLIs · independent audit + all 5 confirmed findings fixed
(TDD, test-first). 174 tests, ruff clean. See git log and `docs/`.
