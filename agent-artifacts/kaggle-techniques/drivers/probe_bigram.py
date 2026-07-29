from collections import defaultdict
from keybo.testkit import assert_module_under
assert_module_under("keybo", "/tmp/kaggle")
from keybo.data.strokes import load_strokes
path = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
rows = load_strokes(path, ngram_len=2, wpm_threshold=0, min_samples=1)
per, nsamp, nrow = defaultdict(set), defaultdict(int), defaultdict(int)
for r in rows:
    nrow[r.layout] += 1
    for wpm, dur, pid, hold in r.samples:
        per[r.layout].add(pid); nsamp[r.layout] += 1
print(f"BIGRAM k31: rows={len(rows)} layouts={len(per)} total_samples={sum(nsamp.values())}", flush=True)
for la in sorted(per):
    print(f"  {la:10s} participants={len(per[la]):6d} samples={nsamp[la]:9d} rows={nrow[la]}", flush=True)
print(f"  participant-count SET = {sorted({len(v) for v in per.values()})}", flush=True)
print(f"  LOLO folds available = {len(per)}", flush=True)
