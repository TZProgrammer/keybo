"""Merge the bigram-only and trigram-only result JSONs into one, combining the per-ngram
ranking dicts (each run only computes ranking for the ngrams it loaded). Usage:

    python merge_results.py results_bigram.json results_trigram.json merged.json
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        a = json.load(f)
    with open(sys.argv[2], encoding="utf-8") as f:
        b = json.load(f)
    out = dict(a)
    for k, v in b.items():
        if k == "ranking":
            merged = dict(a.get("ranking", {}))
            merged.update(v)  # trigram ranking key(s) added to bigram's
            out["ranking"] = merged
        elif k in ("bigram", "trigram"):
            out[k] = v
        else:
            out.setdefault(k, v)  # seeds / high_wpm_floor: keep, assert-equal below
    # sanity: shared scalars must agree
    for key in ("seeds", "high_wpm_floor"):
        if key in a and key in b and a[key] != b[key]:
            raise SystemExit(f"cannot merge: {key} differs ({a[key]} vs {b[key]})")
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(
        f"merged -> {sys.argv[3]}  (ngrams: {[n for n in ('bigram', 'trigram') if n in out]}, "
        f"ranking: {sorted(out.get('ranking', {}))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
