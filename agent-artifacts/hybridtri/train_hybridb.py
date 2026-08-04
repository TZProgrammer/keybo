"""Train the three seeded hybrid-B models the exploit probe searches against.

Retrained on MY OWN tree into MY OWN scratch dir. A filename is not a provenance: EXPLOIT-1
recorded that it deliberately did NOT reuse a sibling's tmpfs scratch for exactly this reason, and
/tmp is tmpfs that gets wiped.

Three seeds (0,1,2) to match the shipped served surface's own seed average, so the two surfaces the
exploit probe compares are built the same way -- mean over three seeded models of the per-cell
predicted ms.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import ARTIFACTS, SCRATCH, assert_tree  # noqa: E402

assert_tree()

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_HYBRIDB_FEATURE_NAMES,
    FEATURE_VERSION_HYBRIDB,
)
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402

WPM = 90.0
SEEDS = (0, 1, 2)
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
SENTINEL = "/tmp/hybridtri_wk/train_hybridb.sentinel"
os.makedirs(SCRATCH, exist_ok=True)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows; layouts {sorted({r.layout for r in rows})}")

out = {"stamp": FEATURE_VERSION_HYBRIDB, "seeds": list(SEEDS), "models": {}}
for s in SEEDS:
    log(f"training hybrid-B seed {s}")
    m = train_bigram_model(
        rows,
        target_wpm=WPM,
        geometry=ROW_STAGGERED_31,
        seed=s,
        n_jobs=8,
        interp="hybridb",
        monotone=True,
    )
    # ASSERT before saving: a model saved under a hybridb_* filename that is not a hybrid-B model
    # is exactly the "filename is not a provenance" trap.
    if m.metadata.feature_version != FEATURE_VERSION_HYBRIDB:
        raise SystemExit(f"ABORT stamp seed{s}: {m.metadata.feature_version}")
    if list(m.metadata.feature_names) != list(BIGRAM_HYBRIDB_FEATURE_NAMES):
        raise SystemExit(f"ABORT columns seed{s}: {m.metadata.feature_names}")
    rec = (m.metadata.extra.get("training") or {}).get("interp_frame") or {}
    want = [1, 1, 1, 1, 1, 1, 1, 1, 1, -1] + [0] * 8
    if rec.get("frame") != "hybrid-b" or list(rec.get("monotone_constraints") or ()) != want:
        raise SystemExit(f"ABORT frame record seed{s}: {rec}")
    path = f"{SCRATCH}/hybridb_mono_seed{s}.json"
    m.save(path)
    out["models"][str(s)] = {
        "path": path,
        "stamp": m.metadata.feature_version,
        "n_columns": len(m.metadata.feature_names),
        "interp_frame": rec,
        "target_space": m.target_space,
    }
    log(f"  saved {path}  stamp={m.metadata.feature_version}  mono={rec.get('monotone_constraints')}")

with open(f"{ARTIFACTS}/train_hybridb.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/train_hybridb.json")
with open(SENTINEL, "w") as fh:
    fh.write("done\n")
log(f"SENTINEL {SENTINEL}")
