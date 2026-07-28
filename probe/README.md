# scissorprice probes — pricing the oxey `scissor` term

Branch `scissor-price`, base `45ea276`. Own venv at `/tmp/scissorprice/.venv`
(`keybo.__file__ == /tmp/scissorprice/src/keybo/__init__.py`; trap-35 checked — no
`repos/keybo` literal in any driver, and every `/tmp/penaudit` path was repointed here).

MODELLED ONLY: g-frame (`time = g(geometry,wpm) + b(ngram)`, only `g` served), surfaces baked
at 90 WPM, corpus blend-v1, tau saturated at 1.0. Nothing here is a claim about realized
human typing speed. `DEFAULT_OXEY_WEIGHTS` is NOT edited by any script; candidate weights go
through the shipped `OxeyStyleScorer(weights=...)` public override or a local weight vector.

## Inherited, positive-controlled instruments (not rewritten — trap 28)
| file | provenance | control |
|---|---|---|
| `matched_prices.py` | THEORY-1's matched estimator, **byte-identical**, md5 `38294e1b26e950adeb37773f069c315b` | `pc_matched.py`: **165 cells vs frozen `matched_prices.json`, max abs diff 0.0** 🟢 |
| `collin3.py` | penaltyaudit's vectorized share path | **7 layouts x 11 terms vs `OxeyStyleScorer.pattern_shares`, max abs diff 0.0** 🟢 (re-run at import by every sp* script, asserted) |

`_X_random.npy` produced here is md5-identical (`f02f99a48e9895bc2529dc7367a6e5cb`) to the one
penaltyaudit's own later run produced independently.

## My probes
| script | question | output |
|---|---|---|
| `sp1_identify.py` | is `scissor` in the collinear cluster? BKW variance-decomposition + VIF + leave-one-TERM-out + bootstrap sign stability — all **clustering-free** | `sp1_identification.json` |
| `sp2_absorb.py` | WHICH term absorbs the marginal->conditional drop? honest confound vs suppression by an unidentified regressor | `sp2_absorption.json` |
| `sp3_ratio.py` | the 2x2: {marginal,conditional} x {linear,tangent}, with the ratio bootstrapped jointly | `sp3_ratio_2x2.json` |
| `sp4_calib.py` | which cell calibrates an additive weight? + scoring experiment over 6 speedtie champions + 11 registry | `sp4_calibration_and_scoring.json` |
| `sp5_joint.py` | joint 11-weight refit vs single-weight patch; free-one-weight PLACEBO; form by grouped CV; speed-tied tiebreak | `sp5_joint_placebo_form.json` |
| `sp6_function.py` | the penalty FUNCTION: support audit, per-finger split, the excluded neighbours | `sp6_penalty_function.json` |
| `sp7_robust.py` | four adversarial attacks: cluster bootstrap, leave-one-source-layout-out, perturbation radius, domain coverage | `sp7_robustness.json` |
| `sp8_final.py` | the definitive estimate with all three correction axes applied at once | `sp8_definitive.json` |
| `sp9_indomain_check.py` | does the domain restriction break identification? same-size placebo; form-free concavity bins | `sp9_indomain_check.json` |

## ⚠️ REUSE HAZARD — these scripts hardcode `/tmp/scissorprice/probe`
11 of the 13 files carry that literal (only `matched_prices.py` is clean), and it is an
**ephemeral** path that dies with the workspace. Every `sp*.py` fails at import
(`spec_from_file_location("c3", "/tmp/scissorprice/probe/collin3.py")`) once it is gone. To re-run:

    git worktree add /tmp/<new> scissor-price          # this branch, tip e1a04c7
    sed -i 's#/tmp/scissorprice#/tmp/<new>#g' /tmp/<new>/probe/*.py

`collin3.py:137-138` also *writes* into its own probe dir, so that dir must be writable.

## ⚠️ `collin3.py` IS penaltyaudit's FILE, and the pool seed is ITS seed
`diff` after reversing the path rewrite is **empty**, and the pool uses `random.Random(31337)` —
penaltyaudit's seed. So reproducing its `scissor_conditional.json` / `scissor_tangent.json`
digit-for-digit is **near-tautological** (same instrument + same seed + same surfaces) and evidences
correct transcription, NOT independent corroboration. The genuinely independent parts of this work
are the BKW/eigen route, the matched-estimator controls (against frozen THEORY-1 artifacts), the
cluster bootstrap, leave-one-source-out, the radius sweep, the same-size placebo, the domain-coverage
finding, and `sp6`'s matched decomposition.

## ⚠️ `sp8`'s +32.59 depends on ONE exclusion, worth 2.6x
The "domain of use" drops `qwerty30m` as an outlier. Keeping it: implied w **+12.68** (n=809) vs
**+32.59** (n=503) — and the full pool gives +13.29, so the whole "domain pushes UP" axis is
contingent on that judgment. Direction (ratio>1) survives either way; the level does not.

Run any of them with `/tmp/scissorprice/.venv/bin/python probe/<script>.py`.
Copies of penaltyaudit's own probes are also present for reconciliation; the `sp*` files are mine.
