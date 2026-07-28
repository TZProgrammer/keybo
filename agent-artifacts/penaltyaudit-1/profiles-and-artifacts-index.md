# Artifacts index — penaltyaudit (PENALTYAUDIT-1)

All produced 2026-07-28. Compute only (no hardware). Worktree `/tmp/penaudit` (branch
`penalty-audit`, base `571bfe9`) — **ephemeral**; everything durable was copied here.

## Frame that applies to EVERY number
`time = g(geometry, wpm) + b(ngram)`; **only `g`** (served geometry) is used. Surfaces **baked at
90 WPM**. Corpus **blend-v1** (md5 `c5066fa7bcc46dea1ecbc987fb465b4a`, re-derived 🟢).
MODELLED ONLY — tau saturated at 1.0, Phase-D cancelled; no realized-speed claim anywhere.

## Verdict artifacts (JSON)
| file | what | key result |
|---|---|---|
| `term_battery.json` | per-term matched BIGRAM prices, 9 source-x-family tables, stratum-bootstrap CIs | AALTO/T2native column reproduces THEORY-1 exactly |
| `tri_battery.json` | TRIGRAM terms x 4 strata levels x 6 native surfaces, with disjointness flags | onehand-vs-alt DISJOINT above land(b),land(c); (b,c) strata shared = 0 |
| `identification.json` | corr matrix, eigenvalues, effective dof, VIF, variance shares (random pool n=400) | eff dof **5.69 of 11**; sfb+imbalance = 79.8% of score variance |
| `band_compare.json` | random vs near-optimal pool | eff dof **collapses 5.69 -> 2.50**; VIF(alternate) 8.2 -> **46.3**; random pool misses 75% of near-opt sfb range |
| `functional_form.json` | form, slope + CI95, VALID RANGE, per source (n=891) | every slope CI excludes 0; `scissor` SATURATING 3/3 |
| `cluster_attribution.json` | per-cluster betas + leave-one-CLUSTER-out delta-R2, K=2..6 | {scissor,outroll} delta-R2 0.12-0.16; {bad_redirect} **0.0000-0.0012** |
| `implied_weights.json` | implied weight per term, sfb-anchored at +12.0, per source | onehand implied **+22.5** vs shipped -1.5 |
| `floor_and_identity.json` | identity R2, per-class ms coefficients, paired floor | identity **R2=0.186** (not algebra); PAIRED floor **0.2453 ms/char at n=11** |
| `term_contrasts.json` | FIRST-PASS battery — **SUPERSEDED** (over-controlled: included origin sig, so most rows returned NO SHARED STRATUM). Kept only so the reconciliation is auditable. Do not cite. |

## Tool outputs
`effect-curves-and-shap/`
- `ec-uniform/` — `keybo effect-curves`, 3-seed ensemble, WPMs 60/80/90/100/110/120, uniform pair
  weighting. **POSITIVE CONTROL: 112 cells vs THEORY-1's frozen `curves.json`, max abs diff 0.0** 🟢
- `ec-qwerty-blend/` — same, corpus-weighted by qwerty on blend-v1 bigrams
- `shap-bg-seed{0,1,2}/` — `keybo shap-report --on grid --target-wpm 90`, 961 rows x 20 features.
  `inwards` ranks **LAST of 20** at 0.00-0.05% importance, seed-unstable sign; `bottom` 18%.

## Probe scripts (`probes/`) — all positive-controlled before use
| script | role |
|---|---|
| `matched_prices.py` | THEORY-1's estimator, copied **byte-identical** (md5 `38294e1b26e950adeb37773f069c315b`) |
| `pc_matched.py` | control: **165 cells vs frozen `matched_prices.json`, diff 0.0** 🟢 |
| `pc_curves.py` | control: **112 cells vs frozen `curves.json`, diff 0.0** 🟢 |
| `pc_tri.py` | control: enum map vs `community.FINGERS[SLOT2DOF]`; reproduces THEORY-1's trigram numbers |
| `recon_tri.py` | trap-43 reconciliation of the ONE number that did not reproduce (onehand-vs-redirect) |
| `collin3.py` | **the share instrument**; control: 7 layouts x 11 terms vs `OxeyStyleScorer.pattern_shares`, **diff 0.0** 🟢 |
| `collin.py`, `collin2.py` | superseded drafts. `collin2.py` is the one whose control CAUGHT MY SPACE BUG — kept as the record |
| `verify_structure.py` | independent re-verification: max non-landing swap diff **exactly 0.0**; roll <=> rowspan>0 |
| `term_battery.py`, `tri_battery.py` | the matched batteries |
| `ident.py`, `band.py`, `cluster_attr2.py` | identification, band comparison, per-cluster attribution |
| `marginal.py` | trap-49 marginal vs conditional (the sign evidence) |
| `form.py`, `price_units.py` | functional form + implied weights |
| `entangle.py` | identity check + paired floor |
| `zero_test.py` | drop-each-term ranking test; sign-corrected-vs-shipped agreement |
| `restate.py` | re-derives the cited "R2=0.9937 restatement" (I get 0.9869 / 0.9704) |
| `diag_roll.py`, `chk_space*.py`, `probe_*.py` | diagnostics: roll/rowspan structure, space semantics, native-vs-standardized frames |

## Inputs consumed (read-only; NOT produced here)
- `state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces/` —
  the **NATIVE** surfaces. ⚠ Only reachable by explicit path: `keybo.analysis.surfaces._resolve()`
  has **no `.native` branch**. `AALTO_FREQ_PRIOR.native` is **MISSING**.
- `state/keybo-optimization/artifacts/theory-1/` — `T2_*.npy`, `Tc_*.npy`, seed surfaces,
  `matched_prices.json`, `final_price_table.json`, `matched_prices.py`, `run_prices.py`
- `/tmp/penaudit/data/models/k31/*.json.gz` — ⚠ **not CLI-loadable as shipped** (sidecar naming);
  gunzip artifact + sidecar to a scratch dir first.

## Not done
No hardware. No push, no CR, no ledger edit; `PREREGISTRATIONS.md` untouched; no layout promoted;
`DEFAULT_OXEY_WEIGHTS` unchanged; no optimization run against my recommendations (a separate arm).

---

## Flush addendum (2026-07-28, after parent registered ledger `45ea276`)

Three artifacts added to answer the parent's load-bearing questions. See **report.md APPENDIX A**.

| file | what | key result |
|---|---|---|
| `sign_table.json` | marginal r + marginal rho + conditional beta for **all 11 terms x 3 sources x BOTH pools** (near_optimal_n341, random_n400), + multivariate R2 | the suppression claim is now auditable with no re-run. `onehand` marg +0.627..+0.703 / cond **-0.190..-0.349** = suppression; `inroll`/`outroll` marg AND cond both positive = no suppression, cleanest flips |
| `scissor_conditional.json` | scissor & sfb slopes MARGINAL vs CONDITIONAL, per source, n=891 | **ratio 7.0x is MARGINAL; conditional is 2.2x-4.4x**. Direction survives 3/3, level does not |
| `probes/recon_counts.py` | reconciles the parent's redirect=3240 vs my 2700 over 3 triple universes | 🟢 **3240 is correct for the ledger.** `bad_redirect` is NESTED in `redirect` (oxey.py:143-146), so 2700 (exclusive) + 540 = 3240 (term-firing). 432 oxey.py-only holds at BOTH levels |

New probes (also in `probes/`): `recon_counts.py`, `scissor_prov.py`, `scissor_cond.py`,
`signtable.py`, `verify_report.py`. All gated on the `collin3.py` positive control (diff **0.0**).

### Self-audit of the dossier's own figures
`probes/verify_report.py` reads **95** headline numbers in report.md back out of the JSON artifacts:
**94/95 matched**; the one miss was my own 2-dp rounding of `bad_redirect`'s AALTO slope
(3.7050 written as 3.71), since corrected to **+3.70** in the table. Re-runnable.

### Corrections made to the dossier at flush time
1. `bad_redirect` AALTO slope 3.71 -> **3.70** (match the artifact exactly).
2. The `redirect` class-count prose (2700) is now labelled as the **exclusive-of-bad** subset, with
   **3240** given as the term-firing count. No share, slope, ratio or verdict changed.
3. The `scissor` 7.0x is now explicitly labelled **MARGINAL**, with the conditional 2.2x-4.4x
   alongside and a recommended softer ledger wording. **This is the one verdict-affecting change.**
4. Added a CI on the scissor **ratio** (conservative, propagated from both slope CIs) — the original
   dossier gave a CI on the slope only.

### Cluster-membership caveat now recorded
`scissor` is **outside** the 5-term `{sfb, onehand, redirect, alternate, imbalance}` cluster at the
reported **K=5 / K=6** cut (it sits in `{scissor, outroll}`, in-band VIF **2.78** — one of the
best-identified terms), but it **does merge in at K=4** and below. Stated as conditional on the cut,
not unconditional.

### Added after sibling `scissorprice` raised the saturation objection (01:40)
| file | what | key result |
|---|---|---|
| `scissor_tangent.json` | scissor & sfb TANGENT slopes at the registry-mean / -median operating share, per source, n=891 | curvature c2 is **NEGATIVE** (-0.83..-1.28) and the operating share (registry mean **0.3591%**) is BELOW the form-pool mean (0.6375%), so the tangent is **STEEPER** than the linear slope: ratio **8.0x-8.3x**, not 7.0x. Also records each registry layout's own scissor share |
| `probes/scissor_tangent.py` | the producing script | — |

⚠ **The two corrections to the 7.0x point in OPPOSITE directions and are NOT combined:** conditioning
-> 2.2x-4.4x; saturation/tangent -> ~8x. The **conditional tangent ratio** is the missing cell;
sibling `scissorprice` owns it (its artifacts land in `state/scissorprice/artifacts/`).

### ⚠ Trap-35 warning for anyone copying these probes
**21 of 34 probe scripts contain the hardcoded literal `/tmp/penaudit`.** `collin3.py` does
`np.save('/tmp/penaudit/probe/_X_random.npy', ...)` and most consumers do
`spec_from_file_location('c3','/tmp/penaudit/probe/collin3.py')` — so a naive copy into another
worktree silently **imports from and writes into THIS tree**, breaking isolation in both directions.
Rewrite the literals first: `grep -ln /tmp/penaudit probe/*.py`. The one clean, self-contained file is
`matched_prices.py` (the parent's THEORY-1 estimator, byte-identical, md5
`38294e1b26e950adeb37773f069c315b`) — copy that as-is.
