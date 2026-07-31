# RETRAIN-DIRECTION — generated result tables

seeds: [0, 1, 2]   high-wpm floor: 80

## trigram

#### trigram: high-wpm non-regression (widened vs narrow), floor >= 80 wpm

| holdout | seed | gated | passed | regressing buckets | per-bucket deltas (all) |
|---|---|---|---|---|---|
| azerty | 0 | True | FAIL | [120] | {40: -0.0054, 60: 0.0017, 80: -0.0031, 100: -0.0046, 120: -0.0357} |
| azerty | 1 | True | FAIL | [120] | {40: 0.0009, 60: 0.0025, 80: 0.0022, 100: 0.0002, 120: -0.0087} |
| azerty | 2 | True | FAIL | [100] | {40: 0.0005, 60: -0.001, 80: 0.0079, 100: -0.0051, 120: 0.0322} |
| dvorak | 0 | True | FAIL | [120] | {40: -0.012, 60: 0.0, 80: 0.0026, 100: -0.001, 120: -0.0326} |
| dvorak | 1 | True | FAIL | [120] | {40: -0.0076, 60: -0.0019, 80: 0.0019, 100: 0.0008, 120: -0.0306} |
| dvorak | 2 | True | FAIL | [120] | {40: -0.0029, 60: 0.0025, 80: -0.0045, 100: 0.0107, 120: -0.0316} |
| qwerty | 0 | True | pass | [] | {40: 0.0009, 60: 0.001, 80: -0.0009, 100: -0.0034, 120: -0.0014} |
| qwerty | 1 | True | pass | [] | {40: 0.0032, 60: 0.0016, 80: 0.0017, 100: 0.0015, 120: 0.001} |
| qwerty | 2 | True | pass | [] | {40: -0.0069, 60: -0.0068, 80: -0.0033, 100: -0.0032, 120: -0.002} |
| qwertz | 0 | True | pass | [] | {40: 0.0031, 60: -0.0015, 80: -0.0015, 100: -0.0022, 120: 0.0072} |
| qwertz | 1 | True | pass | [] | {40: 0.0007, 60: -0.0001, 80: 0.0015, 100: -0.0016, 120: 0.0104} |
| qwertz | 2 | True | pass | [] | {40: -0.0009, 60: -0.0017, 80: -0.0014, 100: -0.0016, 120: -0.0015} |

- **verdict: FAIL — widened regresses a high-wpm bucket**

#### trigram: paired per-fold transfer deltas (rho_widened - rho_narrow)

| holdout | seed | ceiling | rho_narrow | rho_widened | delta | hw-gate |
|---|---|---|---|---|---|---|
| azerty | 0 | 0.8084 | +0.8444 | +0.8416 | -0.0028 | FAIL [120] |
| azerty | 1 | 0.8084 | +0.8427 | +0.8443 | +0.0016 | FAIL [120] |
| azerty | 2 | 0.8084 | +0.8408 | +0.8421 | +0.0013 | FAIL [100] |
| dvorak | 0 | 0.7100 | +0.6203 | +0.6167 | -0.0036 | FAIL [120] |
| dvorak | 1 | 0.7100 | +0.6069 | +0.6071 | +0.0002 | FAIL [120] |
| dvorak | 2 | 0.7100 | +0.6093 | +0.6080 | -0.0013 | FAIL [120] |
| qwerty | 0 | 0.9757 | +0.7447 | +0.7435 | -0.0012 | pass |
| qwerty | 1 | 0.9757 | +0.7472 | +0.7485 | +0.0013 | pass |
| qwerty | 2 | 0.9757 | +0.7472 | +0.7414 | -0.0058 | pass |
| qwertz | 0 | 0.8542 | +0.8813 | +0.8823 | +0.0010 | pass |
| qwertz | 1 | 0.8542 | +0.8816 | +0.8820 | +0.0004 | pass |
| qwertz | 2 | 0.8542 | +0.8827 | +0.8812 | -0.0016 | pass |

- cells: 12  wins(delta>0): 6  losses(delta<0): 6  ties(|delta|<=1e-9): 0
- mean paired delta: -0.0009  min: -0.0058  max: +0.0016
- per-fold sign consistency across seeds:
    - azerty: [-0.0028, 0.0016, 0.0013] -> MIXED
    - dvorak: [-0.0036, 0.0002, -0.0013] -> MIXED
    - qwerty: [-0.0012, 0.0013, -0.0058] -> MIXED
    - qwertz: [0.001, 0.0004, -0.0016] -> MIXED

#### trigram: feature-importance (total gain) of the NEW direction columns, widened full-data model

| new column | mean frac of total gain (over seeds) |
|---|---|
| redirect_sfgated | 0.004552 |
| bad_redirect_sfgated | 0.002338 |
| bg1_inwards_ordered | 0.000915 |
| bg1_outwards_ordered | 0.002392 |
| bg2_inwards_ordered | 0.007846 |
| bg2_outwards_ordered | 0.008766 |

- **new columns' combined share of total gain: 0.026809 (2.681%)**
- per-seed used-columns (any gain > 0):
    - seed 0: new columns used = ['redirect_sfgated', 'bad_redirect_sfgated', 'bg1_inwards_ordered', 'bg1_outwards_ordered', 'bg2_inwards_ordered', 'bg2_outwards_ordered']
    - seed 1: new columns used = ['redirect_sfgated', 'bad_redirect_sfgated', 'bg1_outwards_ordered', 'bg2_inwards_ordered', 'bg2_outwards_ordered']
    - seed 2: new columns used = ['redirect_sfgated', 'bad_redirect_sfgated', 'bg1_inwards_ordered', 'bg1_outwards_ordered', 'bg2_inwards_ordered', 'bg2_outwards_ordered']

#### trigram: layout ordering over the named field, narrow vs widened (seed 0)

- kendall tau(narrow order, widened order): +0.8667
- spearman rho(narrow fitness, widened fitness): +0.9571
- positions moved: 8 of 15  moved=['colemak', 'semimak', 'keybo-c30m', 'p16-balance', 'flagship-c3', 'archive-1846', 'lsb-sib', 'keybo-lsb+lm']
- argmin (best) narrow: p13stab-win   widened: p13stab-win

| rank | narrow order | widened order |
|---|---|---|
| 0 | p13stab-win | p13stab-win |
| 1 | flagship-c3 | lsb-sib  <-- moved |
| 2 | lsb-sib | flagship-c3  <-- moved |
| 3 | archive-1843 | archive-1843 |
| 4 | keybo-lsb | keybo-lsb |
| 5 | archive-1846 | keybo-lsb+lm  <-- moved |
| 6 | keybo-lsb+lm | archive-1846  <-- moved |
| 7 | p16-balance | colemak  <-- moved |
| 8 | semimak | p16-balance  <-- moved |
| 9 | dvorak | dvorak |
| 10 | colemak | keybo-c30m  <-- moved |
| 11 | keybo-c30m | semimak  <-- moved |
| 12 | graphite | graphite |
| 13 | qwerty | qwerty |
| 14 | qwerty30m | qwerty30m |

