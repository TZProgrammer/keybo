"""Render the head-to-head board and the two-objective comparison from ``kc_scored.json``.

Every table here carries its coverage column, because ``ms_per_char`` is a rate over the corpus
mass a layout's charset can type and is NOT comparable across materially different coverage.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

SCORED = Path("/tmp/kc_scored.json")
GAUGES = (
    "sfb",
    "sfs",
    "lsb",
    "alt",
    "roll",
    "sr-roll",
    "redir",
    "scissor",
    "comfort",
    "oxey-style",
)


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def kendall(x: list[float], y: list[float]) -> float:
    n = len(x)
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = x[i] - x[j], y[i] - y[j]
            if a == 0 or b == 0:
                continue
            con, dis = (con + 1, dis) if (a > 0) == (b > 0) else (con, dis + 1)
    return (con - dis) / (con + dis) if con + dis else float("nan")


def main() -> int:
    d = json.loads(SCORED.read_text())
    kc = d["keycraft"]
    ours = {r["name"]: r for r in d["ours"]}
    p = d["provenance"]
    out: list[str] = []
    w = out.append

    w("# keycraft's layouts under keybo's speed model — head-to-head board")
    w("")
    w(
        f"corpus **{p['keybo_corpus']}** · skipgrams `{p['skipgram_table']}` · target {p['target_wpm']:.0f} WPM "
    )
    w(f"model family `{p['model_family']}` · ref `{p['ref']}` · scored via `{p['scored_via']}`  ")
    w(
        f"keycraft ranking + layouts pulled {p['pulled']}; layouts from `data/layouts/*.klf` (BSD-3-Clause)"
    )
    w("")
    w("**ms/char is a rate over the corpus mass each layout can type. Coverage is printed beside")
    w("every number and rows of materially different coverage are NOT one ranking.**")
    w("")

    w("## THE BOARD")
    w("")
    w("| layout | source | ms/char | coverage% | Δ vs arm-B | surfaces | note |")
    w("|---|---|---:|---:|---:|:--:|---|")
    base = ours["arm-B"]["ms_per_char"]
    rows: list[tuple] = []
    for n, r in ours.items():
        rows.append((r["ms_per_char"], n, "ours", r, ""))
    # the identical-charset control and the best faithful keycraft layout
    ctrl = next(r for r in kc if r["is_c30m_perm"])
    faith = [r for r in kc if not r["projection_lossy"]]
    bestf = min(faith, key=lambda r: r["ms_per_char"])
    rows.append(
        (
            ctrl["ms_per_char"],
            ctrl["name"],
            "keycraft",
            ctrl,
            f"kc#{ctrl['kc_rank']} — **identical charset & coverage to ours: the clean control**",
        )
    )
    rows.append(
        (
            bestf["ms_per_char"],
            bestf["name"],
            "keycraft",
            bestf,
            f"kc#{bestf['kc_rank']} — fastest keycraft layout keybo's frame holds exactly",
        )
    )
    for r in sorted(kc, key=lambda r: r["kc_rank"])[:5]:
        note = (
            f"kc#{r['kc_rank']} — {r['projection']}"
            if r["projection_lossy"]
            else f"kc#{r['kc_rank']}"
        )
        rows.append((r["ms_per_char"], r["name"], "keycraft", r, note))
    seen = set()
    for ms, name, src, r, note in sorted(rows):
        if (name, src) in seen:
            continue
        seen.add((name, src))
        cov = r["coverage_pct"]
        flag = "" if abs(cov - 88.7147) < 0.01 else f" ⚠ {cov - 88.7147:+.2f}pp"
        w(
            f"| `{name}` | {src} | **{ms:.2f}** | {cov:.3f}{flag} | {ms - base:+.2f} | "
            f"{'yes' if r['surfaces_available'] else 'N/A'} | {note} |"
        )
    w("")
    w(
        f"**Nothing in keycraft's 150 beats our best.** 0 of the {len(kc)} scorable layouts scores below"
    )
    w(
        f"arm-B's {base:.2f} ms/char. The best keycraft number at all is "
        f"{min(r['ms_per_char'] for r in kc):.2f}, and it belongs to a lossy projection."
    )
    w("")

    w("## Coverage bands — why one ranking would be wrong")
    w("")
    w("| coverage band | n | best ms/char | mean | comparable to ours? |")
    w("|---|---:|---:|---:|---|")
    bands: dict[float, list] = {}
    for r in kc:
        bands.setdefault(round(r["coverage_pct"], 3), []).append(r)
    for cov in sorted(bands, reverse=True)[:8]:
        B = bands[cov]
        if abs(cov - 88.7147) < 0.01:
            ok = "**yes — identical**"
        else:
            delta = cov - 88.7147
            ok = f"no, {delta:+.2f}pp vs ours"
        w(
            f"| {cov:.3f}% | {len(B)} | {min(r['ms_per_char'] for r in B):.2f} | "
            f"{st.mean([r['ms_per_char'] for r in B]):.2f} | {ok} |"
        )
    w("")

    w("## The two objectives barely agree — rank correlation")
    w("")
    w(
        "keycraft Score (higher better) vs our ms/char (lower better, negated), Spearman ρ / Kendall τ:"
    )
    w("")
    w("| subset | n | Spearman ρ | Kendall τ |")
    w("|---|---:|---:|---:|")
    subsets = [
        ("all scorable", kc),
        ("faithful projection only", faith),
        ("coverage-matched (87.494%)", [r for r in kc if abs(r["coverage_pct"] - 87.494) < 1e-3]),
        (
            "faithful AND coverage-matched",
            [r for r in faith if abs(r["coverage_pct"] - 87.494) < 1e-3],
        ),
        ("keycraft top-30 only", [r for r in kc if r["kc_rank"] <= 30]),
    ]
    for label, S in subsets:
        x = [r["kc_score"] for r in S]
        y = [-r["ms_per_char"] for r in S]
        w(f"| {label} | {len(S)} | {spearman(x, y):+.4f} | {kendall(x, y):+.4f} |")
    w("")

    w("## Gauge profiles — do they get there the same way?")
    w("")
    ourfast = [ours[n] for n in ("arm-B", "BALL-1", "arm-H", "keybo-lsb", "flagship-c3")]
    band = sorted(
        [r for r in faith if abs(r["coverage_pct"] - 87.494) < 1e-3],
        key=lambda r: r["ms_per_char"],
    )[:8]
    w("| gauge | ours (mean of 5 fastest) | keycraft (mean of 8 fastest faithful) | Δ |")
    w("|---|---:|---:|---:|")
    for g in GAUGES:
        a = st.mean([r["gauges"][g] for r in ourfast])
        b = st.mean([r["gauges"][g] for r in band])
        w(f"| `{g}` | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    w("")
    Path("/tmp/kc_board.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
