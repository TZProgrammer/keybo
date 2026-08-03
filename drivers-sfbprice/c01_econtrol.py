"""INVARIANT E — the negative controls. Must pass before any corrected number is reportable.

E1  DELTA=0 reproduces TOURNAMENT-1's 125 published per-seed ms/char (5 cluster boards x 25 seeds)
    AND the SHIPPED TimeSurface.card() ms_per_char through the reviewed code path.
E2  the 3 shipped seeds, rebuilt by ME from data/models/k31 (read-only), reproduce tournament's
    first-3 per-seed values -- i.e. my independent rebuild agrees, not just the inherited npz.
    Plus sha256 of every rescued table so the artifact I depend on is identified, not just named.
E2b the rescued tables are LAYOUT-INDEPENDENT as advertised: one table set scores every board.
G-FINITE on every vector.
"""
import json
import time

import numpy as np
from _guard import ART, FIELD_ORDER, SEEDS, SEEDTABLES, assert_d5, build_boards, sha

t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

log("D5:")
assert_d5()

import surface  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

BOARDS = build_boards()
TOURN = "/local/home/zegertho/agent/state/tournament/artifacts/tournament.json"

log("loading per-seed tables (0-2 rebuilt from SHIPPED models, 3-24 rescued npz)")
T2s, Tcs = surface.load_all_seed_tables()
assert len(T2s) == 25

log("hashing the rescued tables (provenance: the artifact, not its name)")
SHAS = {f"tables_seed{s}.npz": sha(f"{SEEDTABLES}/tables_seed{s}.npz") for s in range(3, 25)}
log(f"  22 files hashed, first: tables_seed3.npz {SHAS['tables_seed3.npz'][:16]}")

cdir, tri_freq = surface.corpus(None)
log(f"corpus {cdir.name}: {len(tri_freq)} trigrams, mass {sum(tri_freq.values())}")

ARR = {nm: surface.board_arrays(BOARDS[nm], tri_freq) for nm in FIELD_ORDER}
X = {}
for nm in FIELD_ORDER:
    v = np.array([surface.mspc(ARR[nm], T2s[s], Tcs[s]) for s in range(len(SEEDS))])
    require_finite(v.tolist(), f"per-seed ms/char {nm}")
    X[nm] = v
log("scored 13 boards x 25 seeds at DELTA=0")

# ---------------------------------------------------------------------------- E1a vs tournament
pub = json.load(open(TOURN))["mspc"]["all"]
e1a = {}
worst = 0.0
for nm in ("arm-B", "F(2.5)", "BALL-1", "F(2.0)", "candidate"):
    d = np.abs(np.array(pub[nm]) - X[nm])
    e1a[nm] = {"worst_abs": float(d.max()),
               "worst_rel": float((d / np.abs(np.array(pub[nm]))).max())}
    worst = max(worst, float(d.max()))
    log(f"E1a {nm:12s} worst |diff| vs published = {d.max():.3e}  (n=25)")
log(f"E1a WORST ABS DELTA OVER 125 PUBLISHED VALUES = {worst:.3e}")

# ------------------------------------------------------- E1b vs the SHIPPED TimeSurface.card()
from keybo.analysis.timecard import TimeSurface  # noqa: E402
log("building shipped TimeSurface (seed-mean over the 3 shipped seeds) for E1b")
surf = TimeSurface(tri_freq, target_wpm=surface.WPM, keep_seed_tables=True)
T2m = np.mean(T2s[:3], axis=0)
Tcm = np.mean(Tcs[:3], axis=0)
e1b = {}
worst_rel = 0.0
for nm in ("arm-B", "candidate", "keybo-lsb", "qwerty", "graphite"):
    shipped = surf.card(BOARDS[nm]).ms_per_char
    mine = surface.mspc(ARR[nm], T2m, Tcm)
    rel = abs(mine - shipped) / abs(shipped)
    worst_rel = max(worst_rel, rel)
    st_pub = surf.seed_totals(BOARDS[nm])
    st_mine = [float(((T2s[s][ARR[nm][0], ARR[nm][1]] + Tcs[s][ARR[nm][0], ARR[nm][1], ARR[nm][2]])
                      * ARR[nm][3]).sum()) for s in (0, 1, 2)]
    r2 = max(abs(a - b) / a for a, b in zip(st_pub, st_mine))
    e1b[nm] = {"shipped_card": shipped, "mine": mine, "rel": rel, "seed_totals_rel": r2}
    log(f"E1b {nm:12s} shipped={shipped:.9f} mine={mine:.9f} rel={rel:.3e} "
        f"seed_totals rel={r2:.3e}")
log(f"E1b WORST REL DEV vs shipped card() = {worst_rel:.3e}")

# ---------------------------------------------------------------- E2b layout-independence check
# One table set scores every board: if the tables secretly carried a layout, two boards sharing a
# charset could not differ by a corpus-structure amount. Positive signal: the 13 means are all
# distinct AND the sd across boards is >> the sd across seeds within a board.
means = {nm: float(X[nm].mean()) for nm in FIELD_ORDER}
across_boards_sd = float(np.std(list(means.values()), ddof=1))
within_board_sd = float(np.mean([X[nm].std(ddof=1) for nm in FIELD_ORDER]))
log(f"E2b across-board sd {across_boards_sd:.4f} vs mean within-board (seed) sd "
    f"{within_board_sd:.4f}  ratio {across_boards_sd / within_board_sd:.2f}x")
log(f"E2b all 13 board means distinct: {len(set(np.round(list(means.values()), 9))) == 13}")

out = {
    "gate": "INVARIANT E (negative controls) at DELTA=0",
    "e1a_vs_tournament_published": e1a,
    "e1a_worst_abs_over_125": worst,
    "e1b_vs_shipped_card": e1b,
    "e1b_worst_rel": worst_rel,
    "e2_rescued_table_sha256": SHAS,
    "e2b": {"across_board_sd": across_boards_sd, "within_board_seed_sd": within_board_sd,
            "all_means_distinct": bool(len(set(np.round(list(means.values()), 9))) == 13)},
    "board_means_delta0": means,
    "mspc_delta0": {nm: X[nm].tolist() for nm in FIELD_ORDER},
    "corpus": cdir.name, "n_trigrams": len(tri_freq), "corpus_mass": sum(tri_freq.values()),
    "same_finger_cells": int(surface.same_finger_mask().sum()),
    "wall_s": time.time() - t0,
}
json.dump(out, open(f"{ART}/c01_econtrol.json", "w"), indent=1)
log(f"wrote {ART}/c01_econtrol.json")
log("ALL-DONE")
