"""Train the interp.1 arm models ON MY OWN TREE, into MY OWN scratch dir.

I do NOT reuse /tmp/interpframe_wk/models: those are another agent's scratch on tmpfs, and
"a file with the right name" is not "a model trained on the frame I think it was". This
retrains from the same stroke table the shipped k31 bigram models used, asserts the stamp,
and records the recipe in the artifact.

Seeds 0,1,2 mirror the shipped served surface's 3-seed average, so the two arms' surfaces are
built the same way -- the comparison must differ in the FRAME, not in the seed count.
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-goodhart/agent-artifacts/goodhart")
from _boot import ARTIFACTS, SCRATCH, assert_tree, require  # noqa: E402
assert_tree()
os.makedirs(SCRATCH, exist_ok=True)

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import BIGRAM_INTERP_FEATURE_NAMES, BIGRAM_INTERP_MONOTONE, FEATURE_VERSION_INTERP  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.models.xgboost_model import XGBoostTypingModel  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402

WPM = 90.0
SEEDS = (0, 1, 2)
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} bigram rows; layouts {sorted({r.layout for r in rows})}")

out = {"strokes": STROKES, "n_rows": len(rows), "seeds": list(SEEDS), "wpm": WPM,
       "frame": list(BIGRAM_INTERP_FEATURE_NAMES), "stamp": FEATURE_VERSION_INTERP,
       "monotone": list(BIGRAM_INTERP_MONOTONE), "models": {}}

for s in SEEDS:
    path = f"{SCRATCH}/interp_mono_seed{s}.json"
    if os.path.exists(path):
        m = XGBoostTypingModel.load(path, expected_feature_version=FEATURE_VERSION_INTERP)
        log(f"  REUSED (mine) {path}")
    else:
        log(f"  training interp seed{s} ...")
        m = train_bigram_model(
            rows, target_wpm=WPM, geometry=ROW_STAGGERED_31,
            interp=True, monotone=True, seed=s,
        )
        assert m.metadata.feature_version == FEATURE_VERSION_INTERP, m.metadata.feature_version
        assert list(m.metadata.feature_names) == list(BIGRAM_INTERP_FEATURE_NAMES)
        m.save(path)
        log(f"  trained + saved {path}")
    rec = (m.metadata.extra.get("training") or {}).get("interp_frame") or {}
    out["models"][f"seed{s}"] = {"path": path, "feature_version": m.metadata.feature_version,
                                 "feature_names": list(m.metadata.feature_names),
                                 "interp_frame_record": rec}
    log(f"    stamp={m.metadata.feature_version}  n_cols={len(m.metadata.feature_names)}  rec={rec}")

with open(f"{ARTIFACTS}/train_interp.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
log(f"wrote {ARTIFACTS}/train_interp.json")
