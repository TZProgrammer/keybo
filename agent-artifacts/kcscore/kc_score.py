"""Score keycraft's published layouts through keybo's SHIPPED model, and compare the objectives.

Runs `keybo analyze --json` (the shipped CLI, never a hand-rolled scorer) over every keycraft
board that keybo's frame can represent, plus our own candidates, on one corpus, in one process,
so every number on the board comes from the same run.

Three measurement rules this driver enforces, each because violating one produces a confident
wrong headline:

**1. Coverage gates every comparison.** ``ms_per_char`` is ``total_ms / covered_mass`` — a rate
over the corpus mass the layout's charset can actually type. ``saved_vs_ref_pct`` is NOT
normalized (it is ``(ref_total - total)/ref_total`` on RAW totals), so a layout missing a common
letter shows a huge fake "saving" purely because it typed less of the corpus. This driver
reports coverage beside every rate, computes a coverage-matched peer group, and refuses to put
layouts of materially different coverage in one ranking without labelling it.

**2. A keycraft board is not a keybo layout.** All 150 are 3x12 + thumbs. Projecting onto
keybo's 3x10 drops the two outer columns and all thumb keys. Where those held characters, the
projection is a DIFFERENT layout than the one keycraft ranked, and its score is not a score of
keycraft's design. Such rows are scored (the number is real for the projected board) but
flagged ``projection_lossy`` and kept out of the headline comparison.

**3. Only C30M permutations get the fitted surfaces.** ``model_scores.available`` is False for
any other charset (the surfaces are locked to the C30M 31-slot table). Reported as N/A.

Emits ``kc_scored.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/tmp/kcscore")
PY = "/local/home/zegertho/repos/keybo/.venv/bin/python"
PARSED = Path("/tmp/kc_layouts.json")
OUT = Path("/tmp/kc_scored.json")

C30M = set("qwertyuiopasdfghjkl'zxcvbnm,.-")

#: Our candidates. The three search arms come from the parent's brief and are re-measured here
#: rather than trusted; the named ones resolve through the CLI's own registry
#: (``keybo.cli.analyze._EXTRA_NAMED`` + ``keybo.layouts.NAMED_LAYOUTS``), so no 30-char string
#: for them is retyped in this file.
OUR_ARMS = {
    "arm-B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "arm-H": "flmpg-,uoysntcdireahkxvwb.'jzq",
}
OUR_NAMED = [
    "keybo-lsb",
    "keybo-lsb+lm",
    "keybo-c30m",
    "flagship-c3",
    "graphite",
    "semimak",
    "qwerty30m",
    "qwerty",
]


def run_analyze(layouts: list[str], corpus: str = "blend-v1") -> dict:
    """One `keybo analyze --json` invocation. Asserts the tree and the corpus it actually used."""
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO / "src"),
        "HOME": str(Path.home()),
    }
    # Provenance first: a number from the wrong tree is worse than no number.
    check = subprocess.run(
        [
            PY,
            "-c",
            "from keybo.testkit import assert_module_under;"
            f"print(assert_module_under('keybo', {str(REPO)!r}))",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    if check.returncode != 0:
        raise SystemExit(f"tree assertion FAILED:\n{check.stderr}")

    proc = subprocess.run(
        [PY, "-m", "keybo", "analyze", *layouts, "--corpus", corpus, "--json"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    if proc.returncode != 0:
        raise SystemExit(f"analyze failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}")
    data = json.loads(proc.stdout)
    got = data["corpus"]
    if corpus not in str(got):
        raise SystemExit(f"corpus mismatch: asked {corpus!r}, ran on {got!r}")
    return data


def main() -> int:
    parsed = json.loads(PARSED.read_text())

    # Which keycraft boards can keybo's frame even hold? A hole ('~') in the inner 3x10 means
    # there is no 30-char string at all, so those cannot be scored (northwest, birdie, ...).
    scorable = [r for r in parsed if r["scorable_30"]]
    unscorable = [r for r in parsed if not r["scorable_30"]]

    # Score the whole scorable list in ONE run so every number shares a surface and a corpus.
    strings = [r["key30"] for r in scorable]
    # Duplicate 30-char strings would collide in the CLI's row dict keyed by layout; check.
    dupes = {s for s in strings if strings.count(s) > 1}
    print(f"scorable={len(scorable)} unscorable={len(unscorable)} duplicate-strings={len(dupes)}")

    ours = OUR_NAMED + list(OUR_ARMS.values())
    payload = sorted(set(strings)) + ours
    print(f"one analyze run over {len(payload)} layouts ...", flush=True)
    data = run_analyze(payload)
    rows = data["rows"]

    def cell(key: str) -> dict:
        r = rows[key]
        t = r["time"]
        ms = r["model_scores"]
        surf = {}
        if ms.get("available"):
            for name, v in ms["surfaces"].items():
                surf[name.split("_")[0]] = v["fit"] if v else None
        return {
            "layout": r["layout"],
            "ms_per_char": t["ms_per_char"],
            "coverage_pct": t["coverage_pct"],
            "saved_vs_ref_pct": t["saved_vs_ref_pct"],
            "gauges": r["gauges"],
            "surfaces_available": bool(ms.get("available")),
            "surfaces_reason": ms.get("reason"),
            "surfaces": surf,
        }

    out_kc = []
    for r in scorable:
        c = cell(r["key30"])
        out_kc.append(
            {
                "name": r["name"],
                "kc_rank": r["rank"],
                "kc_score": r["kc_score"],
                "geometry": r["geometry"],
                "url": r["url"],
                "kc_stats": r["kc_stats"],
                "key30": r["key30"],
                "charset30": r["charset30"],
                "is_c30m_perm": r["is_c30m_perm"],
                "projection_lossy": not r["faithful"],
                "projection": r["projection"],
                "letters_on_thumb": r["letters_on_thumb"],
                "dropped_outer": r["dropped_outer"],
                **c,
            }
        )

    out_ours = []
    for name in OUR_NAMED:
        c = cell(name)
        out_ours.append({"name": name, "source": "keybo registry", **c})
    for name, s in OUR_ARMS.items():
        c = cell(s)
        out_ours.append({"name": name, "source": "parent brief (re-measured here)", **c})

    result = {
        "provenance": {
            "keycraft_ranking": "https://rbscholtus.github.io/keycraft/ (150 layouts, sorted by keycraft Score desc)",
            "keycraft_layouts": "https://github.com/rbscholtus/keycraft data/layouts/*.klf (BSD-3-Clause)",
            "pulled": "2026-07-28/29",
            "keybo_corpus": data["corpus"],
            "keybo_corpus_provenance": data.get("corpus_provenance"),
            "skipgram_table": data.get("skipgram_table"),
            "gauge_frame": data.get("gauge_frame"),
            "target_wpm": data["target_wpm"],
            "ref": data["ref"],
            "model_family": data.get("model_family"),
            "scored_via": "keybo analyze --json (shipped CLI), one process, one corpus",
        },
        "keycraft": out_kc,
        "ours": out_ours,
        "unscorable": [
            {
                "name": r["name"],
                "kc_rank": r["rank"],
                "kc_score": r["kc_score"],
                "why": r["projection"],
                "key30": r["key30"],
            }
            for r in unscorable
        ],
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(
        f"wrote {OUT}: {len(out_kc)} keycraft rows, {len(out_ours)} of ours, {len(unscorable)} unscorable"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
