"""FM4 INVARIANT 3: what a served-column RENAME actually does, measured not argued.

Three separate questions, three separate answers:
  Q1  Does the LOAD GUARD (models/base.py:177) reject a shipped model after a rename?
  Q2  Does anything ELSE reject it -- i.e. is the rename free once the load succeeds?
  Q3  Do the attribution READERS even SHOW the new name, or do they read the sidecar?

Method: rename `lateral` -> `RENAMED_PROBE` in the schema lists AND in the row builder, in
memory, then exercise the real production paths. Nothing on disk is touched.
"""

from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

import keybo  # noqa: E402
from keybo.features import ngram as NG  # noqa: E402
from keybo.features import schema as SCH

print("keybo.__file__ =", keybo.__file__)
out = {"keybo_file": keybo.__file__}

OLD, NEW = "lateral", "RENAMED_PROBE"

# --- Q0: where do the attribution readers GET their names from? (read, then confirm) --------
from keybo.analysis.timecard import _load_gz_model  # noqa: E402

m_before = _load_gz_model("bigram_reg31_seed0")
print(f"\nQ0 sidecar names (unchanged on disk): ...{m_before.metadata.feature_names[7]}...")
out["Q0_sidecar_col7_name"] = m_before.metadata.feature_names[7]

# --- perform the in-memory rename, in BOTH places a served name lives --------------------
def patched_placement(geometry, a, b, direction=False, kitchensink=False):
    row = _orig_placement(geometry, a, b, direction=direction, kitchensink=kitchensink)
    return {(NEW if k == OLD else k): v for k, v in row.items()}


_orig_placement = NG._placement_row_from_positions
NG._placement_row_from_positions = patched_placement

for listname in (
    "_BIGRAM_PLACEMENT_NAMES",
    "BIGRAM_FEATURE_NAMES",
    "BIGRAM_DIRECTION_FEATURE_NAMES",
    "BIGRAM_KITCHENSINK_FEATURE_NAMES",
    "TRIGRAM_FEATURE_NAMES",
    "TRIGRAM_DIRECTION_FEATURE_NAMES",
    "TRIGRAM_KITCHENSINK_FEATURE_NAMES",
):
    lst = getattr(SCH, listname)
    setattr(
        SCH,
        listname,
        [n.replace(OLD, NEW) if n == OLD or n.endswith(f"_{OLD}") else n for n in lst],
    )
# the ngram module bound these at import time
NG.BIGRAM_FEATURE_NAMES = SCH.BIGRAM_FEATURE_NAMES
NG.TRIGRAM_FEATURE_NAMES = SCH.TRIGRAM_FEATURE_NAMES
print(f"  renamed in schema: bigram[7] = {SCH.BIGRAM_FEATURE_NAMES[7]!r}")

# --- Q1: does the shipped model still LOAD? ------------------------------------------------
print("\nQ1 does the load guard reject it?")
try:
    m = _load_gz_model("bigram_reg31_seed0")
    print(f"  LOADS FINE. guard compares only feature_version ({m.metadata.feature_version!r}).")
    print(f"  metadata.feature_names[7] is still {m.metadata.feature_names[7]!r} (from the SIDECAR)")
    out["Q1_load"] = {
        "loaded": True,
        "feature_version": m.metadata.feature_version,
        "sidecar_col7": m.metadata.feature_names[7],
    }
except Exception as exc:  # noqa: BLE001
    print(f"  REJECTED: {type(exc).__name__}: {exc}")
    out["Q1_load"] = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}

# --- Q2: does the SCORING path still work, and does it give the same numbers? --------------
print("\nQ2 does scoring still produce identical numbers?")
import itertools  # noqa: E402

from keybo.features.ngram import bigram_features_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402

slots = list(ROW_STAGGERED_31.slots)
X = np.array(
    [bigram_features_from_positions(ROW_STAGGERED_31, (a, b), 90.0) for a, b in itertools.product(slots, repeat=2)]
)
import hashlib  # noqa: E402

sha = hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()
print(f"  bigram K31 matrix sha256 = {sha}")
print("  (baseline was            = 80251a89115976f3...)")
out["Q2_matrix_sha256"] = sha

# --- Q3: does `keybo compare` / shap_diff still work? --------------------------------------
print("\nQ3 does the ATTRIBUTION path (shap_diff / keybo compare) still work?")
try:
    from keybo.analysis.shap_diff import shap_diff

    d = shap_diff("pyou'vgdnmheai.cstrlkjz,-wfbxq", "bldwz'foujnrtsgyhaeiqxmcvkp,.-", channel="t2")
    print(f"  RAN. names seen by the report: {d.feature_names[7]!r}")
    out["Q3_shap_diff"] = {"ran": True, "col7": d.feature_names[7]}
except Exception as exc:  # noqa: BLE001
    print(f"  REJECTED: {type(exc).__name__}: {exc}")
    out["Q3_shap_diff"] = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}

# --- Q4: does shap_report show the new name? -----------------------------------------------
print("\nQ4 does shap-report show the NEW name, or the sidecar's OLD one?")
try:
    pass

    from keybo.analysis.shap_report import compute_shap
    rep = compute_shap(_load_gz_model("bigram_reg31_seed0"), X[:200], interactions_max_rows=50)
    print(f"  report.feature_names[7] = {rep.feature_names[7]!r}   <- source of the printed name")
    out["Q4_shap_report_col7"] = rep.feature_names[7]
except Exception as exc:  # noqa: BLE001
    print(f"  ERROR: {type(exc).__name__}: {exc}")
    out["Q4_shap_report_col7"] = f"ERROR {type(exc).__name__}: {exc}"

with open(os.path.join(os.path.dirname(__file__), "stamp_probe.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote stamp_probe.json")
