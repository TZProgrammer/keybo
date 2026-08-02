"""INVARIANT A, last bullet: how much TRAINING DATA actually carries signal about the offsets?"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, json
from collections import Counter, defaultdict
import numpy as np
from keybo.data.strokes import load_strokes
from keybo.training.validate import build_cells

BI  = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
TRI = "/local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv"

# Match the repo's own defaults. Check what validate CLI uses.
rows = load_strokes(BI, ngram_len=2, wpm_threshold=40, min_samples=10)
print(f"bigram rows loaded: {len(rows)}")
cells = build_cells(rows)
print(f"bigram CELLS (layout,ngram,wpm-bucket): {len(cells)}")

def klass(pos):
    (ax,ay),(bx,by) = pos[0], pos[1]
    if ay==0 or by==0: return "space-touching"
    if ay==by: return "same-row"
    return "cross-row"

# ---- ROW level
c_rows = Counter(klass(r.positions) for r in rows)
samp   = Counter(); samp_by = defaultdict(int)
for r in rows: samp[klass(r.positions)] += len(r.samples)
freq   = Counter()
for r in rows: freq[klass(r.positions)] += r.frequency
print("\n=== BIGRAM stroke ROWS by class ===")
tot=sum(c_rows.values())
for k in ("same-row","cross-row","space-touching"):
    print(f"  {k:15s} rows {c_rows[k]:5d} ({100*c_rows[k]/tot:5.1f}%)   raw samples {samp[k]:9d} ({100*samp[k]/sum(samp.values()):5.1f}%)   corpus-freq mass {freq[k]:12d} ({100*freq[k]/sum(freq.values()):5.1f}%)")

# ---- CELL level (the actual unit of evaluation)
c_cells = Counter(klass(c.positions) for c in cells)
n_cells = Counter(); 
for c in cells: n_cells[klass(c.positions)] += c.n
print("\n=== BIGRAM EVAL CELLS by class (the unit LOLO scores) ===")
tc=sum(c_cells.values())
for k in ("same-row","cross-row","space-touching"):
    print(f"  {k:15s} cells {c_cells[k]:5d} ({100*c_cells[k]/tc:5.1f}%)  underlying samples {n_cells[k]:9d}")

# ---- per layout, cross-row cells (what each LOLO fold has to work with)
print("\n=== cross-row + space cells PER LAYOUT (per-fold identifying data) ===")
per = defaultdict(Counter)
for c in cells: per[c.layout][klass(c.positions)] += 1
for lay in sorted(per):
    d = per[lay]; t=sum(d.values())
    print(f"  {lay:8s} total {t:5d} | same-row {d['same-row']:5d} | cross-row {d['cross-row']:5d} | space {d['space-touching']:4d}"
          f"   => identifying (cross-row) = {100*d['cross-row']/t:5.1f}%")

# ---- distinct row-PAIR combos among cross-row cells: which of the 2 params each informs
print("\n=== which OFFSET CONTRAST each cross-row cell informs (row pair) ===")
rp = Counter()
for c in cells:
    (ax,ay),(bx,by)=c.positions[0],c.positions[1]
    if ay==0 or by==0: rp[("space", ay if by==0 else by)] += 1
    elif ay!=by: rp[tuple(sorted((ay,by)))] += 1
for k,v in sorted(rp.items(), key=lambda kv:-kv[1]):
    print(f"  rows {k}: {v} cells")

print("[bigram-only census done]")
