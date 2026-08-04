"""FM4 INVARIANT 2: the ZERO-NUMERICAL-CHANGE checksum harness.

Emits a stable digest of every number a rename could conceivably move, so the SAME script run
before and after the diff produces two files that must be BYTE-IDENTICAL:

  1. the FULL bigram feature matrix over all K30 and K31 ordered position pairs, and the
     trigram matrix over all K30 triples -- as sha256 over the float64 bytes, in CANONICAL
     COLUMN ORDER (so a reordering, not just a value change, would show up);
  2. per-column sha256, keyed by SEMANTIC SLOT INDEX rather than by name, because a rename
     changes the name and comparing name->hash would produce a spurious mismatch. Names are
     recorded alongside so the rename is VISIBLE but not conflated with a value change;
  3. every gauge on every named layout (the analyze board's own gauge dict) -- these must not
     move at all, names included;
  4. the shipped models' predictions on the full serve grid (bigram + trigram), which is what
     actually reaches a published ms/char number.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

import keybo  # noqa: E402
from keybo.features.ngram import (  # noqa: E402
    bigram_features_from_positions,
    trigram_features_from_positions,
)
from keybo.features.schema import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    FEATURE_VERSION,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31  # noqa: E402

print("keybo.__file__ =", keybo.__file__)
print("FEATURE_VERSION =", FEATURE_VERSION)

WPM = 90.0


def sha(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()


out = {"keybo_file": keybo.__file__, "FEATURE_VERSION": FEATURE_VERSION}

# --- 1+2. feature matrices ----------------------------------------------------------------
for gname, g in (("K30", ROW_STAGGERED_30), ("K31", ROW_STAGGERED_31)):
    slots = list(g.slots)
    X = np.array(
        [bigram_features_from_positions(g, (a, b), WPM) for a, b in itertools.product(slots, repeat=2)]
    )
    out[f"bigram_{gname}"] = {
        "shape": list(X.shape),
        "matrix_sha256": sha(X),
        "sum": repr(float(X.sum())),
        "names": list(BIGRAM_FEATURE_NAMES),
        # keyed by SLOT INDEX: a rename must not perturb these
        "col_sha256_by_index": {str(i): sha(X[:, i]) for i in range(X.shape[1])},
    }
    print(f"  bigram {gname}: shape={X.shape} sha={out[f'bigram_{gname}']['matrix_sha256'][:16]}")

slots = list(ROW_STAGGERED_30.slots)
T = np.array(
    [
        trigram_features_from_positions(ROW_STAGGERED_30, (a, b, c), WPM)
        for a, b, c in itertools.product(slots, repeat=3)
    ]
)
out["trigram_K30"] = {
    "shape": list(T.shape),
    "matrix_sha256": sha(T),
    "sum": repr(float(T.sum())),
    "names": list(TRIGRAM_FEATURE_NAMES),
    "col_sha256_by_index": {str(i): sha(T[:, i]) for i in range(T.shape[1])},
}
print(f"  trigram K30: shape={T.shape} sha={out['trigram_K30']['matrix_sha256'][:16]}")

# --- 3. every gauge on every named layout --------------------------------------------------
print("\n=== gauges ===")
from keybo.analysis.kmstats import KmStats  # noqa: E402
from keybo.analysis.lateral_span import LateralSpan  # noqa: E402
from keybo.analysis.redirects import RedirectFamily  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.scoring.comfort import ComfortBigramScorer  # noqa: E402
from keybo.scoring.oxey import OxeyStyleScorer  # noqa: E402

cdir = production_corpus_dir(None)
bg = load_frequencies(str(cdir / "bigrams.txt"))
sk = load_frequencies(str(cdir / "1-skip31.txt"))
tg = load_frequencies(str(cdir / "trigrams.txt"))
print(f"  corpus dir = {cdir}")

km = KmStats(bg, sk, tg)
oxey = OxeyStyleScorer(bg, sk, tg)
comfort = ComfortBigramScorer(bg, skipgram_freqs=sk)
fam = RedirectFamily(tg)
span = LateralSpan(bg)

gauges = {}
for name, lay30 in sorted(NAMED_LAYOUTS.items()):
    if len(lay30) != 30:
        continue
    try:
        layout = Layout(lay30, ROW_STAGGERED_30)
        entry = {
            "kmstats": {k: repr(v) for k, v in km.stats(lay30).items()},
            "oxey_pattern_shares": {k: repr(v) for k, v in oxey.pattern_shares(layout).items()},
            "oxey_fitness": repr(oxey.fitness(layout)),
            "comfort_fitness": repr(comfort.fitness(layout)),
            "redirect_family": {k: repr(v) for k, v in fam.shares(lay30).items()},
            "lat_span_share": repr(span.share(layout)),
        }
        gauges[name] = entry
    except Exception as exc:  # noqa: BLE001
        gauges[name] = {"error": f"{type(exc).__name__}: {exc}"}
out["gauges"] = gauges
print(f"  {len(gauges)} layouts scored on 5 gauge families")

# --- 4. the shipped models' predictions on the full serve grid -----------------------------
print("\n=== shipped model predictions ===")
# Load through the SHIPPED loader (timecard._load_gz_model) rather than a hand-rolled path,
# so the sidecar/version guard is exercised exactly as production does it.
from keybo.analysis.timecard import _load_gz_model  # noqa: E402

preds = {}
slots31 = list(ROW_STAGGERED_31.slots)
for seed in (0, 1, 2):
    m = _load_gz_model(f"bigram_reg31_seed{seed}")
    Xb = np.array(
        [
            bigram_features_from_positions(ROW_STAGGERED_31, (a, b), WPM)
            for a, b in itertools.product(slots31, repeat=2)
        ]
    )
    ms = m.predict_ms(Xb)
    preds[f"bigram_seed{seed}"] = {
        "sidecar_feature_version": m.metadata.feature_version,
        "sidecar_feature_names": list(m.metadata.feature_names),
        "pred_ms_sha256": sha(ms),
        "pred_ms_sum": repr(float(ms.sum())),
    }
    print(f"  bigram seed{seed}: sum(ms)={float(ms.sum()):.10f} sha={sha(ms)[:16]}")
for seed in (0, 1, 2):
    m = _load_gz_model(f"trigram_cond31_seed{seed}")
    ms = m.predict_ms(T)
    preds[f"trigram_seed{seed}"] = {
        "sidecar_feature_version": m.metadata.feature_version,
        "sidecar_feature_names": list(m.metadata.feature_names),
        "pred_ms_sha256": sha(ms),
        "pred_ms_sum": repr(float(ms.sum())),
    }
    print(f"  trigram seed{seed}: sum(ms)={float(ms.sum()):.10f} sha={sha(ms)[:16]}")
out["model_predictions"] = preds

tag = os.environ.get("FM4_TAG", "before")
path = os.path.join(os.path.dirname(__file__), f"checksum_{tag}.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=2, sort_keys=True)
print(f"\nwrote {path}")
