"""REHUNT preflight — the positive control that must pass before a single eval is spent.

THREE CHECKS, in order. If any fails, everything downstream is void and the run STOPS.

  A. BOARD REPRODUCTION. Rebuilt in a pristine worktree from staged corpora, does this board
     reproduce the FROZEN per-axis numbers of every one of the 42 re-adjudicated rows — the
     nine non-wfd axes on the frozen `best_layout`, plus the frozen `incumbent_axes`? Without
     this, a re-run's "no dominator" could be a rebuilt-board artifact rather than a result.

  B. LEGACY wfd REPRODUCTION. Does `wfd_mode='legacy'` reproduce the frozen `best_axes.wfd`
     exactly? That pins that we are correcting the campaign's actual frame, not some other one.

  C. RE-ADJUDICATION REPRODUCTION. Recomputing the CORRECTED wfd for candidate and incumbent
     and recounting `n_ge`, do we reproduce WFD-FRAMES-1's `n_ge_own_pin` for all 42 rows and
     its 14 dominance flips exactly? This is the brief's requirement 1.

Every axis direction is DERIVED from qwerty-is-worst, never assumed (trap 5). `sfr` is not in
this frame; the 12-axis frame here is the campaign's `AXES12`, none of whose members is a
permutation invariant (asserted).

MODELED/gauge only. Held-layout tau saturated at 1.0; Phase-D cancelled. No realized-speed claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corpus_eval as CE  # noqa: E402
import wscissor_eval as WE  # noqa: E402

ART = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts")
READJ = ART / "wfd-frames-1/readjudication.json"

#: The seven frozen hunt artifacts carrying the 14 flipped verdicts, and how to rebuild each.
#: `frame=None` means the 10-axis frame (`hunt_on_blend`); otherwise a `wscissor_hunt` frame.
FROZEN_HUNTS = {
    "gen-on-blend/hunt-blend-armA-norm.json": dict(corpus="blend", arm="A", frame="ten"),
    "gen-on-blend/hunt-blend-armB-norm.json": dict(corpus="blend", arm="B", frame="ten"),
    "noanchor-1/hunt-noanchor-armA-norm.json": dict(corpus="noanchor", arm="A", frame="ten"),
    "noanchor-1/hunt-noanchor-armB-norm.json": dict(corpus="noanchor", arm="B", frame="ten"),
    "wscissor-gen-1/runs/whunt-blend-twelve.json": dict(corpus="blend", arm="A", frame="twelve"),
    "wscissor-gen-1/runs/whunt-iweb-twelve.json": dict(corpus="iweb", arm="A", frame="twelve"),
    "wscissor-gen-1/runs/whunt-noanchor-twelve.json": dict(
        corpus="noanchor", arm="A", frame="twelve"
    ),
}

#: rel-tol for reproducing a frozen float axis. The community gauges are exact integers; the
#: corpus-tabled axes are float sums whose summation order we do not control across a rebuild.
RTOL = 1e-9


def corpus_manifest() -> dict:
    """sha256 + md5 of every table each corpus actually reads. Named explicitly (CORPUS-SWAP-1:
    `data/corpus` is blend-v1 by default on some branches, so a bare name is not evidence)."""
    out = {}
    for name, directory in sorted(CE.CORPUS_DIRS.items()):
        tables = {}
        for fname in ("bigrams.txt", CE.SKIP_NAME, "trigrams.txt"):
            path = Path(directory) / fname
            blob = path.read_bytes()
            tables[fname] = {
                "path": str(path),
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "md5": hashlib.md5(blob).hexdigest(),
            }
        out[name] = {"label": CE.CORPUS_LABELS[name], "dir": str(directory), "tables": tables}
    return out


def derive_directions(board: WE.WScissorBoard, frame: list[str]) -> dict[str, int]:
    """Axis directions DERIVED from qwerty-is-worst, then checked against the frame's SIGN.

    qwerty30m is the worst layout on this board by construction of the campaign's frame, so for
    each axis the sign that makes qwerty the minimum IS the higher-better direction. Verified
    against a spread of real layouts so a tie cannot silently pick a direction.
    """
    qwerty = "qwertyuiopasdfghjkl'zxcvbnm,.-"
    q = board.axes12(qwerty)
    others = [board.axes12(lay) for lay in CE.INCUMBENTS.values()]
    derived = {}
    for axis in frame:
        vals = [o[axis] for o in others]
        if all(v > q[axis] for v in vals):
            derived[axis] = +1  # qwerty is the minimum => higher is better
        elif all(v < q[axis] for v in vals):
            derived[axis] = -1
        else:
            raise ValueError(
                f"axis {axis!r}: qwerty is neither min nor max vs the incumbents "
                f"(qwerty={q[axis]!r}, others={vals!r}) — direction not derivable"
            )
    mismatched = {a: (derived[a], WE.SIGN12[a]) for a in frame if derived[a] != WE.SIGN12[a]}
    if mismatched:
        raise ValueError(f"DERIVED direction disagrees with the frame's SIGN12: {mismatched}")
    return derived


def assert_no_invariant_axis(board: WE.WScissorBoard, frame: list[str], seed: int = 20260726):
    """No axis in this frame may be a permutation invariant (trap 23: `sfr` was one).

    Shuffle the layout and require every axis to actually move. A `std > 0` filter is NOT the
    test — numpy reports std=1.9e-14 for a true invariant, which such a filter KEEPS.
    """
    rng = np.random.default_rng(seed)
    rows = [board.axes12("".join(rng.permutation(list(CE.C30M)))) for _ in range(60)]
    frozen = [a for a in frame if len({round(r[a], 12) for r in rows}) == 1]
    if frozen:
        raise ValueError(f"permutation-INVARIANT axes in the frame (would tie by construction): {frozen}")
    return {a: int(len({round(r[a], 12) for r in rows})) for a in frame}


def close(a: float, b: float, rtol: float = RTOL) -> bool:
    return bool(np.isclose(a, b, rtol=rtol, atol=0.0)) or a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "runs" / "rehunt-preflight.json"))
    args = ap.parse_args()

    readj = json.loads(READJ.read_text())
    print(f"loaded {len(readj)} re-adjudicated rows from {READJ}", flush=True)

    manifest = corpus_manifest()
    print("\n== corpus manifest (named explicitly; sha256 recorded) ==", flush=True)
    for name, blob in manifest.items():
        print(f"  {name:9s} {blob['label']:20s} {blob['dir']}", flush=True)
        for fname, t in blob["tables"].items():
            print(f"      {fname:14s} md5={t['md5']} sha256={t['sha256'][:16]}…", flush=True)

    # ---- one board per (corpus, arm), in BOTH wfd modes ----------------------------------
    boards: dict[tuple, dict] = {}
    for spec in FROZEN_HUNTS.values():
        key = (spec["corpus"], spec["arm"])
        if key in boards:
            continue
        ceil = CE.SixSurface(spec["corpus"]).ceiling_map
        boards[key] = {
            mode: WE.WScissorBoard(
                corpus=spec["corpus"],
                arm=spec["arm"],
                ceilings=ceil,
                objective="wide",
                wfd_mode=mode,
            )
            for mode in ("corrected", "legacy")
        }
        print(f"built board corpus={spec['corpus']} arm={spec['arm']} (both wfd modes)", flush=True)

    # ---- direction derivation + invariance guard, once per board -------------------------
    print("\n== axis directions DERIVED from qwerty-is-worst (never assumed) ==", flush=True)
    directions = {}
    distinct_values = {}
    for key, pair in boards.items():
        directions[f"{key[0]}/arm{key[1]}"] = derive_directions(pair["corrected"], WE.AXES12)
        distinct_values[f"{key[0]}/arm{key[1]}"] = assert_no_invariant_axis(
            pair["corrected"], WE.AXES12
        )
        print(f"  {key[0]}/arm{key[1]}: matches SIGN12 on all {len(WE.AXES12)} axes ✅", flush=True)

    # ---- the three checks, row by row ----------------------------------------------------
    rows_out = []
    for row in readj:
        fname = row["file"]
        if fname not in FROZEN_HUNTS:
            continue
        spec = FROZEN_HUNTS[fname]
        frame = CE.AXES if spec["frame"] == "ten" else WE.AXES12
        pair = boards[(spec["corpus"], spec["arm"])]
        frozen = json.loads((ART / fname).read_text())
        target = row["target"]
        lay = row["layout"]
        frozen_axes = frozen["per_target_best"][target]["best_axes"]
        frozen_inc = frozen["incumbent_axes"]

        # (A) BOARD REPRODUCTION — the nine/eleven NON-wfd axes on the frozen best_layout.
        ours = pair["corrected"].axes12(lay)
        nonwfd = [a for a in frame if a != "wfd"]
        board_bad = {
            a: (frozen_axes[a], ours[a]) for a in nonwfd if not close(frozen_axes[a], ours[a])
        }
        # …and on every published incumbent row.
        inc_bad = {}
        for iname, iaxes in frozen_inc.items():
            mine = pair["corrected"].axes12(iaxes["layout"])
            for a in nonwfd:
                if a in iaxes and not close(iaxes[a], mine[a]):
                    inc_bad[f"{iname}.{a}"] = (iaxes[a], mine[a])

        # (B) LEGACY wfd REPRODUCTION — exact integer match on candidate and incumbents.
        legacy_cand = pair["legacy"].axes12(lay)["wfd"]
        legacy_bad = {}
        if legacy_cand != frozen_axes["wfd"]:
            legacy_bad["candidate"] = (frozen_axes["wfd"], legacy_cand)
        for iname, iaxes in frozen_inc.items():
            mine = pair["legacy"].axes12(iaxes["layout"])["wfd"]
            if mine != iaxes["wfd"]:
                legacy_bad[iname] = (iaxes["wfd"], mine)

        # (C) RE-ADJUDICATION — recount n_ge with the CORRECTED wfd on both sides.
        sign = np.array([WE.SIGN12[a] for a in frame])
        cand_frozen = np.array([frozen_axes[a] for a in frame]) * sign
        if target.startswith("IDEAL"):
            stack = np.array(
                [[frozen_inc[i][a] for a in frame] for i in frozen_inc]
            ) * sign
            targ_frozen = stack.max(axis=0)
            targ_corrected_wfd = max(
                pair["corrected"].axes12(frozen_inc[i]["layout"])["wfd"] for i in frozen_inc
            )
        else:
            targ_frozen = np.array([frozen_inc[target][a] for a in frame]) * sign
            targ_corrected_wfd = pair["corrected"].axes12(frozen_inc[target]["layout"])["wfd"]
        w = frame.index("wfd")
        n_ge_frozen = int(np.sum(cand_frozen >= targ_frozen - 1e-9))
        cand_corr = cand_frozen.copy()
        targ_corr = targ_frozen.copy()
        cand_corr[w] = WE.SIGN12["wfd"] * ours["wfd"]
        targ_corr[w] = WE.SIGN12["wfd"] * targ_corrected_wfd
        n_ge_corr = int(np.sum(cand_corr >= targ_corr - 1e-9))
        n_gt_corr = int(np.sum(cand_corr > targ_corr + 1e-9))

        # The frozen artifact's OWN verdict is the STRICT one (>= everywhere AND > somewhere).
        # `readjudicate.py` used the LOOSE form `n_ge == n_axes`, which labels a SELF-TIE
        # (best_layout IS the incumbent, n_strict=0) a dominator — see `self_tie` below. Both
        # are recorded; the strict one is authoritative because it is what the hunt reported.
        frozen_strict = bool(frozen["per_target_best"][target]["dominates_target"])
        frozen_n_strict = int(frozen["per_target_best"][target]["best_n_strict_better"])
        self_tie = bool(
            not target.startswith("IDEAL") and lay == frozen_inc[target]["layout"]
        )
        rows_out.append(
            dict(
                file=fname,
                corpus=spec["corpus"],
                arm=spec["arm"],
                frame=spec["frame"],
                frame_size=len(frame),
                target=target,
                layout=lay,
                self_tie=self_tie,
                board_reproduces=not board_bad,
                board_mismatches=board_bad,
                incumbents_reproduce=not inc_bad,
                incumbent_mismatches=inc_bad,
                legacy_wfd_reproduces=not legacy_bad,
                legacy_wfd_mismatches=legacy_bad,
                frozen_n_ge=frozen["per_target_best"][target].get("best_n_ge"),
                frozen_n_strict=frozen_n_strict,
                n_ge_recomputed_frozen=n_ge_frozen,
                n_ge_corrected=n_ge_corr,
                n_gt_corrected=n_gt_corr,
                readj_n_ge_own_pin=row["n_ge_own_pin"],
                readj_n_ge_as_coded=row["n_ge_as_coded"],
                matches_readjudication=bool(n_ge_corr == row["n_ge_own_pin"]),
                dominates_frozen=frozen_strict,
                dominates_frozen_loose=bool(row["dominates_as_coded"]),
                dominates_corrected=bool(n_ge_corr == len(frame) and n_gt_corr >= 1),
                dominates_corrected_loose=bool(n_ge_corr == len(frame)),
                readj_dominates_own_pin=bool(row["dominates_own_pin"]),
                matches_readjudication_verdict=bool(
                    (n_ge_corr == len(frame)) == row["dominates_own_pin"]
                ),
                wfd_frozen=frozen_axes["wfd"],
                wfd_corrected=ours["wfd"],
                wfd_target_corrected=targ_corrected_wfd,
            )
        )

    # ---- verdict --------------------------------------------------------------------------
    nA = sum(r["board_reproduces"] and r["incumbents_reproduce"] for r in rows_out)
    nB = sum(r["legacy_wfd_reproduces"] for r in rows_out)
    nC = sum(r["matches_readjudication"] for r in rows_out)
    nD = sum(r["matches_readjudication_verdict"] for r in rows_out)
    # STRICT verdict flips — the frozen artifacts' own definition, and the brief's 14.
    flips = [r for r in rows_out if r["dominates_frozen"] and not r["dominates_corrected"]]
    reverse = [r for r in rows_out if not r["dominates_frozen"] and r["dominates_corrected"]]
    self_ties = [r for r in rows_out if r["self_tie"]]
    total = len(rows_out)
    print(f"\n== POSITIVE CONTROL over the {total} rows of the 7 frozen hunts ==", flush=True)
    print(f"  A. board reproduces every frozen non-wfd axis (cand + incumbents): {nA}/{total}")
    print(f"  B. legacy wfd reproduces the frozen number EXACTLY:                {nB}/{total}")
    print(f"  C. corrected-wfd n_ge matches WFD-FRAMES-1's n_ge_own_pin:         {nC}/{total}")
    print(f"  D. LOOSE verdict (n_ge==n) matches its dominates_own_pin:          {nD}/{total}")
    print(f"\n  STRICT flips reproduced (frozen dominates -> corrected NOT): {len(flips)}  <- the brief's 14")
    print(f"  reverse flips (NOT -> dominates):                            {len(reverse)}")
    print(
        f"  self-tie rows (best_layout IS the incumbent, n_strict=0): {len(self_ties)} — the frozen\n"
        f"      artifacts call these NOT dominators; readjudicate.py's loose n_ge==n calls them\n"
        f"      dominators, which is why D is read against the LOOSE form."
    )

    for r in rows_out:
        if not (
            r["board_reproduces"]
            and r["incumbents_reproduce"]
            and r["legacy_wfd_reproduces"]
            and r["matches_readjudication"]
            and r["matches_readjudication_verdict"]
        ):
            print(
                f"   FAIL {r['file'][:44]:44s} {r['target']:14s} "
                f"board={r['board_reproduces']} inc={r['incumbents_reproduce']} "
                f"legacy={r['legacy_wfd_reproduces']} readj={r['matches_readjudication']}",
                flush=True,
            )
            for label, bad in (
                ("board", r["board_mismatches"]),
                ("inc", r["incumbent_mismatches"]),
                ("legacy", r["legacy_wfd_mismatches"]),
            ):
                for k, (frz, got) in list(bad.items())[:6]:
                    print(f"        {label}.{k}: frozen={frz!r} ours={got!r}", flush=True)

    ok = (
        nA == total
        and nB == total
        and nC == total
        and nD == total
        and len(flips) == 14
        and len(reverse) == 0
        and total > 0
    )
    out = dict(
        verdict="PASS" if ok else "FAIL",
        n_rows=total,
        board_reproduces=nA,
        legacy_wfd_reproduces=nB,
        matches_readjudication=nC,
        dominance_verdict_matches_loose=nD,
        flips_reproduced=len(flips),
        reverse_flips=len(reverse),
        self_tie_rows=len(self_ties),
        strict_vs_loose_note=(
            "readjudicate.py defined dominance as n_ge==n_axes with no strict-win term, so it "
            "labels the 12 SELF-TIE rows (best_layout IS the incumbent, n_strict=0) as "
            "dominators; the frozen artifacts' own dominates_target is the strict form and "
            "calls them False. Both counts are recorded. The 14 headline flips are unaffected: "
            "every one has n_strict>=1 under the frozen frame."
        ),
        flipped_cells=[
            dict(corpus=r["corpus"], arm=r["arm"], frame=r["frame"], target=r["target"])
            for r in flips
        ],
        corpus_manifest=manifest,
        derived_directions=directions,
        distinct_values_over_60_random_perms=distinct_values,
        rtol=RTOL,
        rows=rows_out,
        note="MODELED/gauge only; tau saturated, Phase-D cancelled; no realized-speed claim.",
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nverdict={out['verdict']}  wrote {args.out}", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
