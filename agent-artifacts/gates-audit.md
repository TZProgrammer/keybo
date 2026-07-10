# Gates audit (2026-07-10, user directive: "audit if our gates are doing the right thing")

Every gate in force, its purpose, calibration status, known misfires, and fix status.
Registered companion: PREREGISTRATIONS.md GATE-AUDIT round (5d4228e); noise floors from
runs/gate_noise.json (gate_noise.py, 10 seeds x 4 folds on the anchor config).

## The inventory

### 1. Rare-ngram guard (umae <= +2%, dec3 <= +2% rel)
- **Purpose:** block adoptions that improve frequent cells by abandoning rare ones —
  rare cells are the only evidence for position pairs the optimizer explores off the
  corpus frequency distribution (user's "wmae is dangerous" rule).
- **Track record:** the campaign's workhorse. Blocked: MED/M5 labels, S1 label (both
  spaces), OCC, W-N/W-SQRT/W-INV, crosseval CAP-adoption. Four independent tests showed
  the same dense-win/rare-loss signature => the guard detects a real, recurring failure
  mode, not noise.
- **Calibration:** thresholds were set by judgment (2%), not by a measured noise floor.
  gate_noise.py measures the 10-seed pairwise noise p95 for each metric; verdict rule:
  a threshold is defensible iff it exceeds that p95. Big rejections (S1 +7.2%, OCC
  +15.6%, W-INV +17.9%) are far above any plausible floor; near-misses (W-SQRT +4.45%,
  MED +3.5%) may be within ~2x noise — annotated, not re-adjudicated.
- **Status: SOUND in direction; thresholds now being calibrated.**

### 2. Adoption bar (wmae > 1% rel better; tune bar 0.5%)
- **Purpose:** don't churn the production recipe for noise-level gains.
- **Calibration:** same gate_noise run. If p95 pairwise seed noise on wmae > 0.5%, the
  tune bar is inside noise and lucky seeds can qualify; the 1% bar has more margin.
- **Status: PENDING gate_noise verdict** (thresholds bumped to ceil(p95) for future
  rounds if flagged; existing verdicts stand).

### 3. tau / decisive-pair tau (must not degrade)
- **Purpose:** magnitude gains must not break layout ORDERING — the one thing the
  optimizer ultimately consumes.
- **Known misfire (fixed):** all-pair tau punished arms for coin-flips on statistical
  ties (azerty-qwertz; dvorak-qwerty). Fix: pair_gap_boot bootstrap CIs => decisive-pair
  tau. This audit's tail_gap_boot extends the same fix to the q=0.2 frame (gate iv of
  QIN-LR judged tail ranking with MEAN-frame decisive pairs — same bug, one level up).
- **Residual gap:** dp-tau on 4 layouts has granularity 1/4-1/6 — one flipped pair is
  0.5 vs 1.0 with nothing between. More layouts (Phase D) is the real fix.
- **Status: SOUND after the decisive-pair fix; frame-matched decisive sets required**
  (each eval frame needs ITS OWN bootstrap, not the mean frame's — now practiced).

### 4. E5 optimizer-side structural gate (feature deletions)
- **Purpose:** LOLO evaluates on real layouts; the optimizer queries far off that
  distribution and exploits pricing null spaces (Goodhart row-blindness, 2026-07-05).
- **Misfire found by THIS audit (G1):** the v1 clause "home share >= every named
  layout's" fails KNOWN-GOOD models — incumbent optimizer outputs measure 53.9% (P10)
  and 31.6% (P8b) vs colemak's 59.8%; OQ-14's top~home speed tie means speed-optimal
  layouts do not maximize home share. The bar encoded community doctrine, not physics:
  uninformative in both directions (a good model could fail it; row-blind junk with
  lucky home placement could pass it).
- **Fix (registered before reading the gated result):** v2 = cross-regret under the
  trusted incumbent surface, bar 0.75% (plateau 0.5% + margin). This is the test the
  original incident would have failed loudly. Home share + distinct-vector counts stay
  as informational diagnostics.
- **First use:** A5-LOGRAT blocked at +0.815% (bar 0.75%; qwerty +4.21% for scale).
  Both v1 and v2 agreed on A5; the correction mattered for legitimacy, not outcome.
- **Status: FIXED (v2 in force).**

### 5. censor_ratio (clean_sweep hesitation-cap guard, <= 3.0)
- **Known defect (registered 2026-07-08, unfixed):** its construction cannot
  distinguish frequency-correlated from geometry-correlated censoring (crosseval
  measured hesitations' distance-rho at -0.01); CAP4 improving the rare decile while
  failing the ratio contradicts the censoring story it infers.
- **Status: KNOWN-DEFECTIVE, registered; any future cap re-test must fix it first.**

### 6. kb_strat cross-stratum spread bar (< 0.25pp)
- **Misfire:** stricter than the measurement noise floor (no CIs on per-stratum
  regrets); failed the letter while the argmax was hardware-invariant everywhere.
- **Status: MISCALIBRATED, acknowledged in the outcome; CI-based re-registration
  required before any future hardware adjudication.**

### 7. Split-half ceilings as rho normalizer
- **Quirk:** pooled models beat single-layout ceilings (rho/ceil > 1.0 on azerty/
  qwertz folds) — the ceiling is one layout's internal agreement, not a bound for a
  model borrowing cross-layout strength. Harmless for A/B (both arms share the
  normalizer); misleading if ever read as an absolute "fraction of achievable."
- **Status: SOUND for comparisons; documented footgun for absolute readings.**

### 8. Leakage gates (stage-1 contract + residual ngram-R2 audit)
- **Purpose:** blind-pace models must not fingerprint content via timing patterns.
- **Track record:** disqualified nothing incorrectly; the model-class cap (linear over
  robust aggregates) held the line when raw per-offset nets would have leaked.
- **Status: SOUND; untested against an adversarial probe (acceptable — the model class
  cap makes the attack surface small).**

## Systemic findings

- **G-A (the pattern behind G1 + kb_strat + censor_ratio):** gates fail when their
  threshold encodes DOCTRINE or JUDGMENT (community home-row lore; 0.25pp; 3.0) rather
  than a measured quantity (noise floor, plateau width, known-good baseline). Fixed
  gates (dp-tau, E5-v2) all moved from doctrine to measurement.
- **G-B:** every gate that compares means needs a noise floor measured ONCE on the
  anchor config (gate_noise.py); thresholds below the p95 pairwise floor are flags.
- **G-C:** decisive-pair sets are FRAME-SPECIFIC — reusing the mean frame's bootstrap
  on a different aggregation (q=0.2, F5M) silently re-introduces the tie bug the
  bootstrap was built to fix. Each frame gets its own pair CIs (tail_gap_boot pattern).
- **G-D (calibration standard):** a gate must pass known-good inputs and fail a known-bad
  one (the original Goodhart layout is the regression case for E5-v2). A gate never
  exercised against either is unvalidated, whatever its intuition.
