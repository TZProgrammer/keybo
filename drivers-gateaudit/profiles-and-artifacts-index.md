# Artifacts index — gateaudit

Decision audit of the calibration gate on branch `calib` (@ c28b37e), repo `~/repos/keybo`.
Verdict: **ADOPT WITH MODIFICATION** (scope `passed` to `{pooled, bucket_centered}`; drop the
`band` default). Full reasoning: `/local/home/zegertho/agent/state/gateaudit/report.md`.

All artifacts are small JSON/logs (292 K total) — no bulk data, no model weights. Every one is ALSO
committed in-repo at `drivers-gateaudit/` on branch `gateaudit-audit` in the worktree below, so they
survive a workspace loss.

## Durable locations

- **State dir (primary):** `/local/home/zegertho/agent/state/gateaudit/artifacts/`
- **In-repo copy:** `/local/home/zegertho/repos/keybo-wt-gateaudit/drivers-gateaudit/`
  on branch `gateaudit-audit`, commits `f230c4d`, `e776614`, `4288f45` — **UNPUSHED, by design.**
- **Worktree:** `/local/home/zegertho/repos/keybo-wt-gateaudit` (branches `gateaudit-audit`,
  `gateaudit-proposal`)

## Reproduction environment (required — the venv silently resolves `keybo` to the shared checkout)

```bash
export PYTHONPATH=/local/home/zegertho/repos/keybo-wt-gateaudit/src
PY=/local/home/zegertho/repos/keybo/.venv/bin/python
# every driver prints keybo.__file__ and asserts it is under the worktree before measuring
```

LOLO data: `/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv` (2202 rows, 4 layouts).
⚠ **NOT** `data/community/processed/bistrokes_community.tsv` — that has 12 layouts incl. custom
boards and `validate()` raises on them at `min_cell_samples=10`. This cost one wasted run.

---

## Runs

### RUN 1 — full suite on the reconstructed minimal landing (INVARIANT 5)
- **Verdict: PASS.** `1290 passed / 3 skipped / 0 failed`, rc=0, 197.70 s.
- Tree: `gateaudit-audit` @ `3f787bf` = 3 files cherry-picked from `calib` onto `origin/main`
  (`8701c00`): `src/keybo/verdicts.py`, `src/keybo/training/validate.py`, `tests/test_verdicts.py`.
- **Deletions audit: ZERO** (`git diff --diff-filter=D --name-only origin/main` empty).
  260 insertions, 2 deletions — both inside one comment. `tests/analysis/test_los.py` intact @ 228
  lines; `src/keybo/layouts.py` and `src/keybo/training/train.py` NOT in the diff.
- Note 1290 vs the branch's 1276: the branch deletes `test_los.py` (14 tests).
- Log: `/local/home/zegertho/agent/state/gateaudit/artifacts/suite-reconstructed-tree.log`

### RUN 2 — my own LOLO, 3 seeds × 4 folds (INVARIANT 2 base measurement)
- **Verdict: rc=0.** Driver `g01_scope.py`; output
  `/local/home/zegertho/agent/state/gateaudit/artifacts/g01_scope.json`, log `g01.log`.
- **NEGATIVE CONTROL PASSED:** reproduces CALIB-1's registered per-fold bucket-centered slopes to
  **|diff| = 0.0000** on all four folds — azerty 1.0423 / dvorak 0.9248 / qwertz 1.0217 /
  qwerty 1.4067. Pooled also matches `agent-artifacts/results_bigram.json` per-seed.
- ⇒ **CORRECTION: the parent's e2e run is the outlier, not the ledger** (its qwerty 1.3116 is 0.095
  low).
- Records the full `support` map (n_cells, n_participants) per slice — the field the parent's own
  artifact had collapsed to `support_present: true`, and the one INVARIANT 2 turns on.

### RUN 3 — historical replay + negative control (INVARIANT 4 / Q1, Q2)
- **Verdict: the gate WOULD have caught it.** Driver `g02_historical.py`, output
  `/local/home/zegertho/agent/state/gateaudit/artifacts/g02_historical.json`.
- Source: the SHIPPED pre-branch artifact `~/repos/keybo/agent-artifacts/results_bigram.json`.
- `pooled_only` flags **exactly qwerty** (1.2356/1.2199/1.2295, all 3 seeds) and passes the other
  three folds. `every_available_slice` and `buckets_only` fail all four (useless).
- ⚠ **THIS CONTROL CAUGHT A REAL LEDGER ERROR** — see the correction below.

### RUN 4 — estimand probes (INVARIANT 1)
- Drivers `g03_estimand.py` (+ `g04_estimand_fix.py`, which corrects two construction errors of my
  own in g03). Outputs `g03_estimand.json`, `g04_estimand_fix.json`.
- **Identity control PASSED:** MSE-optimal predictor gives `slope(obs~pred) = 1.000000` at
  r = 0.3/0.5/0.657889/0.9, with `slope_fwd × slope_rev = r²` to **1.1e-16**.
- **MEASURED FALSE-FLAG FLOOR** (20,000 trials/cell, truth slope exactly 1, band (0.90,1.10), at the
  repo's own r = 0.657889): n=12 **77.2%** · n=20 70.6% · n=40 59.0% · n=64 **49.2%** · n=100 38.8%
  · n=200 22.1% · n=400 8.0% · n=900 **0.89%**. Measured sd matches analytic `sd_e/√(n−2)` to 3 dp
  (an internal control on the simulation).
- **Pooled-only is BLIND:** constructed 5-bucket ramp with true within-bucket compression 1.45× →
  pooled reads **1.0176 (IN BAND)** while bucket_centered recovers **1.4500**.
- Monotone warp at slope EXACTLY 1.0: in band, ρ=τ=+1.0, local exchange rate wrong **1623×**.

### RUN 5 — scope comparison table (INVARIANT 2)
- Driver `g05_scope_table.py`, output
  `/local/home/zegertho/agent/state/gateaudit/artifacts/g05_scope_table.json`.
  Applies every candidate rule POST-HOC to RUN 2, so no configuration got its own private run.
- Folds passing (fold passes iff all 3 seeds pass): **every_slice 0/4** · buckets_only 0/4 ·
  support_gated_n100 0/4 · n200 1/4 · **bucket_centered_only 3/4** · pooled_only 3/4 ·
  **structural_pair 3/4** · support_gated_n400 3/4. All 3/4 rows fail **qwerty only**.
- **THE DECIDING MEASUREMENT:** the 8 non-qwerty out-of-band slices are **1.30–2.37 sd**
  (thin, n=64–366, false-flag 9.8–49.1%); the 6 qwerty ones are **4.09–12.29 sd** (n=477–2648).
  **NO OVERLAP.**

### RUN 6 — false-flag budget (INVARIANT 4 / Q3)
- Driver `g06_invariant4.py`, output `g06_invariant4.json`.
- Over the 12 fold×seed cells: **every_slice fires 12/12 with 11.74 expected noise-only flags**;
  support_gated_n400 3/12 @ 0.97; **structural_pair 3/12 @ 0.14**; pooled_only 3/12 @ 0.14;
  bucket_centered_only 3/12 @ 0.00.
- Band-width check: a 5% false-flag band needs [0.921,1.079]…[0.956,1.044] at pooled n=799–2648
  ⇒ **(0.90,1.10) is correct-to-conservative THERE**; a thin bucket at n=64 would need [0.715,1.285].
  One band cannot serve both scopes.

### RUN 7 — the modification, coded and verified end-to-end (INVARIANT 5/6)
- Branch `gateaudit-proposal` @ `8ca512d` (**unpushed**). Changes: `band` default removed from
  `calibration_report()`; new `deciding` arg + `CALIBRATION_DECIDING_SLICES_RECOMMENDED`;
  `out_of_band_advisory` so a narrowed scope cannot HIDE a slice; `bucket_centered` support recorded.
- **Full suite: PASS — `1295 passed / 3 skipped / 0 failed`, rc=0, 174.08 s**
  (`/local/home/zegertho/agent/state/gateaudit/artifacts/suite-proposal.log`).
  The 2 initial reds were the 2 intended behaviour changes, not bugs; 5 new tests added.
- **End-to-end through `validate()`: 8/8 checks PASS** (driver `g07_e2e_proposal.py`, output
  `/local/home/zegertho/agent/state/gateaudit/artifacts/g07_e2e_proposal.json`, log `g07.log`):
  arm 1 `band=None` → all four `passed: None` with all 7 slopes still reported; arm 2 band+scope →
  **azerty/dvorak/qwertz PASS, qwerty FAIL**; arm 3 every-slice → all four fail (reproduces the
  branch, making the §(b) comparison apples-to-apples).

---

## Corrections to the parent's brief (the highest-value output)

1. 🟢 **`calibration_report()` DOES default `band`** (`verdicts.py:367`); only `require_calibration()`
   does not. `validate()` passes no band ⇒ every artifact ships a hard `passed: false` against a band
   no human chose. The ledger (:12214) repeats the same wrong claim.
2. 🟢 **A THIRD, previously unreported ledger misreading.** `PREREGISTRATIONS.md:11919` and `:12013`
   quote per-fold pooled **"0.914–0.999 … the surface does not compress"**. Measured range over the
   12 fold×seed cells of that exact file: **0.9138 … 1.2356**. The quote **excludes the qwerty fold**
   (1.2283); excluding qwerty reproduces "0.914–0.999" exactly.
3. 🟢 **The parent's e2e bucket-centered numbers are the outlier**, not CALIB-1's (RUN 2).
4. 🟢 **"1276 passed" is the branch's suite** — 14 tests short because the branch deletes
   `test_los.py`. The reconstructed tree is 1290.
5. 🟠 **H2 understated:** not "inert", but "asserts a failing verdict against an unchosen band".

## Open

🔴 **The trigram surface is unmeasured** — ledger:98 is a *trigram* claim and I did not run the
trigram LOLO, so I cannot say whether the gate would have caught that one. Highest-value follow-up
(~10 min: same `g01_scope.py`, `ngram="trigram"`).
