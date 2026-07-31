# RETRAIN-DIRECTION — generated result tables

seeds: [0, 1, 2]   high-wpm floor: 80

## bigram

#### bigram: high-wpm non-regression (widened vs narrow), floor >= 80 wpm

| holdout | seed | gated | passed | regressing buckets | per-bucket deltas (all) |
|---|---|---|---|---|---|
| azerty | 0 | True | FAIL | [120] | {40: -0.0002, 60: 0.0004, 80: 0.0009, 100: 0.0071, 120: -0.0078} |
| azerty | 1 | True | pass | [] | {40: 0.0056, 60: -0.0013, 80: 0.0001, 100: -0.0021, 120: 0.0002} |
| azerty | 2 | True | FAIL | [100, 120] | {40: -0.0055, 60: -0.0019, 80: -0.002, 100: -0.0063, 120: -0.0212} |
| dvorak | 0 | True | pass | [] | {40: -0.0084, 60: 0.0048, 80: 0.0036, 100: 0.0016, 120: 0.0097} |
| dvorak | 1 | True | pass | [] | {40: -0.0019, 60: 0.0112, 80: 0.0036, 100: 0.0048, 120: 0.0023} |
| dvorak | 2 | True | FAIL | [80] | {40: -0.0146, 60: 0.0062, 80: -0.0055, 100: -0.0046, 120: 0.0086} |
| qwerty | 0 | True | pass | [] | {40: -0.0089, 60: -0.0069, 80: -0.0044, 100: -0.0016, 120: 0.0001} |
| qwerty | 1 | True | pass | [] | {40: 0.0081, 60: 0.0052, 80: 0.0013, 100: 0.0027, 120: -0.0013} |
| qwerty | 2 | True | pass | [] | {40: -0.0001, 60: -0.0022, 80: 0.0011, 100: -0.0005, 120: -0.0006} |
| qwertz | 0 | True | pass | [] | {40: -0.0038, 60: -0.0035, 80: -0.001, 100: -0.0018, 120: 0.006} |
| qwertz | 1 | True | pass | [] | {40: -0.0046, 60: -0.0026, 80: 0.0004, 100: 0.0001, 120: 0.0091} |
| qwertz | 2 | True | FAIL | [120] | {40: 0.0016, 60: -0.0, 80: -0.0016, 100: -0.0012, 120: -0.0091} |

- **verdict: FAIL — widened regresses a high-wpm bucket**

#### bigram: paired per-fold transfer deltas (rho_widened - rho_narrow)

| holdout | seed | ceiling | rho_narrow | rho_widened | delta | hw-gate |
|---|---|---|---|---|---|---|
| azerty | 0 | 0.8716 | +0.8664 | +0.8671 | +0.0007 | FAIL [120] |
| azerty | 1 | 0.8716 | +0.8675 | +0.8679 | +0.0004 | pass |
| azerty | 2 | 0.8716 | +0.8695 | +0.8654 | -0.0040 | FAIL [100, 120] |
| dvorak | 0 | 0.7888 | +0.7041 | +0.7019 | -0.0021 | pass |
| dvorak | 1 | 0.7888 | +0.6933 | +0.6968 | +0.0035 | pass |
| dvorak | 2 | 0.7888 | +0.6987 | +0.6918 | -0.0069 | FAIL [80] |
| qwerty | 0 | 0.9904 | +0.7789 | +0.7726 | -0.0064 | pass |
| qwerty | 1 | 0.9904 | +0.7793 | +0.7831 | +0.0037 | pass |
| qwerty | 2 | 0.9904 | +0.7763 | +0.7753 | -0.0009 | pass |
| qwertz | 0 | 0.9206 | +0.9173 | +0.9139 | -0.0034 | pass |
| qwertz | 1 | 0.9206 | +0.9165 | +0.9142 | -0.0023 | pass |
| qwertz | 2 | 0.9206 | +0.9167 | +0.9164 | -0.0003 | FAIL [120] |

- cells: 12  wins(delta>0): 4  losses(delta<0): 8  ties(|delta|<=1e-9): 0
- mean paired delta: -0.0015  min: -0.0069  max: +0.0037
- per-fold sign consistency across seeds:
    - azerty: [0.0007, 0.0004, -0.004] -> MIXED
    - dvorak: [-0.0021, 0.0035, -0.0069] -> MIXED
    - qwerty: [-0.0064, 0.0037, -0.0009] -> MIXED
    - qwertz: [-0.0034, -0.0023, -0.0003] -> consistent

#### bigram: feature-importance (total gain) of the NEW direction columns, widened full-data model

| new column | mean frac of total gain (over seeds) |
|---|---|
| inwards_ordered | 0.012825 |
| outwards_ordered | 0.016624 |

- **new columns' combined share of total gain: 0.029449 (2.945%)**
- per-seed used-columns (any gain > 0):
    - seed 0: new columns used = ['inwards_ordered', 'outwards_ordered']
    - seed 1: new columns used = ['inwards_ordered', 'outwards_ordered']
    - seed 2: new columns used = ['inwards_ordered', 'outwards_ordered']

#### bigram: layout ordering over the named field, narrow vs widened (seed 0)

- kendall tau(narrow order, widened order): +0.9429
- spearman rho(narrow fitness, widened fitness): +0.9857
- positions moved: 2 of 15  moved=['colemak', 'semimak']
- argmin (best) narrow: p13stab-win   widened: p13stab-win

| rank | narrow order | widened order |
|---|---|---|
| 0 | p13stab-win | p13stab-win |
| 1 | keybo-lsb | keybo-lsb |
| 2 | keybo-lsb+lm | keybo-lsb+lm |
| 3 | lsb-sib | lsb-sib |
| 4 | archive-1846 | archive-1846 |
| 5 | archive-1843 | archive-1843 |
| 6 | flagship-c3 | flagship-c3 |
| 7 | keybo-c30m | keybo-c30m |
| 8 | p16-balance | p16-balance |
| 9 | dvorak | dvorak |
| 10 | semimak | colemak  <-- moved |
| 11 | graphite | graphite |
| 12 | colemak | semimak  <-- moved |
| 13 | qwerty | qwerty |
| 14 | qwerty30m | qwerty30m |

## trigram

#### trigram: high-wpm non-regression (widened vs narrow), floor >= 80 wpm

| holdout | seed | gated | passed | regressing buckets | per-bucket deltas (all) |
|---|---|---|---|---|---|
| azerty | 0 | True | FAIL | [120] | {40: -0.0031, 60: 0.0017, 80: -0.0044, 100: 0.0078, 120: -0.0313} |
| azerty | 1 | True | pass | [] | {40: -0.0029, 60: 0.0017, 80: 0.0042, 100: 0.0062, 120: 0.0313} |
| azerty | 2 | True | FAIL | [100] | {40: 0.0008, 60: 0.0004, 80: 0.0022, 100: -0.0122, 120: 0.053} |
| dvorak | 0 | True | FAIL | [100, 120] | {40: -0.0148, 60: -0.0153, 80: -0.0039, 100: -0.0217, 120: -0.0069} |
| dvorak | 1 | True | FAIL | [100, 120] | {40: -0.0138, 60: -0.0034, 80: 0.0024, 100: -0.0128, 120: -0.0217} |
| dvorak | 2 | True | FAIL | [120] | {40: -0.0029, 60: 0.011, 80: -0.002, 100: 0.0081, 120: -0.0059} |
| qwerty | 0 | True | pass | [] | {40: 0.0042, 60: 0.0003, 80: 0.0016, 100: -0.0026, 120: -0.0044} |
| qwerty | 1 | True | pass | [] | {40: -0.0072, 60: -0.0057, 80: -0.0033, 100: -0.0009, 120: -0.0013} |
| qwerty | 2 | True | pass | [] | {40: 0.0033, 60: 0.0008, 80: -0.0006, 100: -0.0008, 120: -0.001} |
| qwertz | 0 | True | FAIL | [120] | {40: 0.0006, 60: -0.0014, 80: -0.0015, 100: -0.004, 120: -0.0185} |
| qwertz | 1 | True | FAIL | [120] | {40: 0.0004, 60: 0.0005, 80: -0.002, 100: -0.0029, 120: -0.006} |
| qwertz | 2 | True | pass | [] | {40: -0.0012, 60: -0.0017, 80: -0.002, 100: 0.0025, 120: 0.0176} |

- **verdict: FAIL — widened regresses a high-wpm bucket**

#### trigram: paired per-fold transfer deltas (rho_widened - rho_narrow)

| holdout | seed | ceiling | rho_narrow | rho_widened | delta | hw-gate |
|---|---|---|---|---|---|---|
| azerty | 0 | 0.8084 | +0.8444 | +0.8432 | -0.0012 | FAIL [120] |
| azerty | 1 | 0.8084 | +0.8427 | +0.8424 | -0.0002 | pass |
| azerty | 2 | 0.8084 | +0.8408 | +0.8416 | +0.0008 | FAIL [100] |
| dvorak | 0 | 0.7100 | +0.6203 | +0.6064 | -0.0139 | FAIL [100, 120] |
| dvorak | 1 | 0.7100 | +0.6069 | +0.6014 | -0.0055 | FAIL [100, 120] |
| dvorak | 2 | 0.7100 | +0.6093 | +0.6120 | +0.0027 | FAIL [120] |
| qwerty | 0 | 0.9757 | +0.7447 | +0.7448 | +0.0001 | pass |
| qwerty | 1 | 0.9757 | +0.7472 | +0.7419 | -0.0053 | pass |
| qwerty | 2 | 0.9757 | +0.7472 | +0.7477 | +0.0005 | pass |
| qwertz | 0 | 0.8542 | +0.8813 | +0.8810 | -0.0003 | FAIL [120] |
| qwertz | 1 | 0.8542 | +0.8816 | +0.8817 | +0.0001 | FAIL [120] |
| qwertz | 2 | 0.8542 | +0.8827 | +0.8814 | -0.0013 | pass |

- cells: 12  wins(delta>0): 5  losses(delta<0): 7  ties(|delta|<=1e-9): 0
- mean paired delta: -0.0020  min: -0.0139  max: +0.0027
- per-fold sign consistency across seeds:
    - azerty: [-0.0012, -0.0002, 0.0008] -> MIXED
    - dvorak: [-0.0139, -0.0055, 0.0027] -> MIXED
    - qwerty: [0.0001, -0.0053, 0.0005] -> MIXED
    - qwertz: [-0.0003, 0.0001, -0.0013] -> MIXED

#### trigram: feature-importance (total gain) of the NEW direction columns, widened full-data model

| new column | mean frac of total gain (over seeds) |
|---|---|
| bg1_inwards_ordered | 0.003214 |
| bg1_outwards_ordered | 0.002387 |
| bg2_inwards_ordered | 0.008463 |
| bg2_outwards_ordered | 0.010356 |

- **new columns' combined share of total gain: 0.024420 (2.442%)**
- per-seed used-columns (any gain > 0):
    - seed 0: new columns used = ['bg1_inwards_ordered', 'bg1_outwards_ordered', 'bg2_inwards_ordered', 'bg2_outwards_ordered']
    - seed 1: new columns used = ['bg1_inwards_ordered', 'bg1_outwards_ordered', 'bg2_inwards_ordered', 'bg2_outwards_ordered']
    - seed 2: new columns used = ['bg1_inwards_ordered', 'bg1_outwards_ordered', 'bg2_inwards_ordered', 'bg2_outwards_ordered']

#### trigram: layout ordering over the named field, narrow vs widened (seed 0)

- kendall tau(narrow order, widened order): +0.8286
- spearman rho(narrow fitness, widened fitness): +0.9464
- positions moved: 11 of 15  moved=['dvorak', 'colemak', 'semimak', 'keybo-c30m', 'keybo-lsb', 'p16-balance', 'flagship-c3', 'archive-1843', 'archive-1846', 'lsb-sib', 'keybo-lsb+lm']
- argmin (best) narrow: p13stab-win   widened: p13stab-win

| rank | narrow order | widened order |
|---|---|---|
| 0 | p13stab-win | p13stab-win |
| 1 | flagship-c3 | lsb-sib  <-- moved |
| 2 | lsb-sib | flagship-c3  <-- moved |
| 3 | archive-1843 | keybo-lsb  <-- moved |
| 4 | keybo-lsb | archive-1843  <-- moved |
| 5 | archive-1846 | keybo-lsb+lm  <-- moved |
| 6 | keybo-lsb+lm | archive-1846  <-- moved |
| 7 | p16-balance | colemak  <-- moved |
| 8 | semimak | dvorak  <-- moved |
| 9 | dvorak | p16-balance  <-- moved |
| 10 | colemak | keybo-c30m  <-- moved |
| 11 | keybo-c30m | semimak  <-- moved |
| 12 | graphite | graphite |
| 13 | qwerty | qwerty |
| 14 | qwerty30m | qwerty30m |

