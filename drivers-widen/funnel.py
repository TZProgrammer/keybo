"""ARM-1 DESCRIPTIVE FUNNEL (not a hypothesis test): re-derive the layout-diverse headroom
from the pipeline's ACTUAL filters, correcting the parent's AVG_WPM_15 error.

Counts, per layout:
  (1) raw metadata rows
  (2) pass load_participant_metadata (FINGERS 9-10, AVG_WPM_15>=40 floor, KB full/laptop, layout)
  (3) distinct pids actually PRESENT in the shipped tristroke table (tristrokes31_cond_v1.tsv)
  (4) distinct pids that survive load_strokes(min_samples=10)+build_cells(min_cell_samples=10)
      i.e. participants who actually contribute to the GATE-1 fit.
Also dumps KEYBOARD_TYPE and FINGERS raw distributions for ARM 2 feasibility.
"""
from __future__ import annotations
import csv, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/local/home/zegertho/repos/keybo/src")
from keybo.data.keystrokes import load_participant_metadata, _LAYOUT_ROWS
from keybo.data.strokes import load_strokes
from keybo.training.validate import build_cells

META = "/local/home/zegertho/keybo-e2e/dataset/Keystrokes/files/metadata_participants.txt"
TSV = "/local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

csv.field_size_limit(sys.maxsize)

# (1) raw metadata: LAYOUT / KEYBOARD_TYPE / FINGERS distributions (all rows)
raw_layout = Counter(); kb_all = Counter(); fingers_all = Counter()
# cross-tabs restricted to layout in the 4 supported (what could ever enter)
kb_by_layout = defaultdict(Counter); fingers_by_layout = defaultdict(Counter)
n_rows = 0
with open(META, newline="", encoding="utf-8", errors="replace") as f:
    for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
        n_rows += 1
        lay = (row.get("LAYOUT") or "").strip().lower()
        kb = (row.get("KEYBOARD_TYPE") or "").strip().lower()
        fg = (row.get("FINGERS") or "").strip()
        raw_layout[lay] += 1; kb_all[kb] += 1; fingers_all[fg] += 1
        if lay in _LAYOUT_ROWS:
            kb_by_layout[lay][kb] += 1
            fingers_by_layout[lay][fg] += 1
log(f"metadata rows: {n_rows}")
print("RAW LAYOUT:", dict(raw_layout.most_common()))
print("RAW KEYBOARD_TYPE:", dict(kb_all.most_common()))
print("RAW FINGERS:", dict(fingers_all.most_common()))

# (2) pass load_participant_metadata (the actual process-time filter, min_wpm=40)
meta = load_participant_metadata(META)  # default min_wpm=40
pool = Counter(m["LAYOUT"] for m in meta.values())
log(f"pass filter: {sum(pool.values())} participants")
print("POOL (pass filter) by layout:", dict(pool.most_common()))

# KEYBOARD_TYPE distribution AMONG the passing pool (should be full/laptop only)
kb_pass = Counter((m.get("KEYBOARD_TYPE") or "").strip().lower() for m in meta.values())
print("KEYBOARD_TYPE among passing pool:", dict(kb_pass.most_common()))
# KEYBOARD_TYPE x layout among passing pool
for lay in ("qwerty","qwertz","azerty","dvorak"):
    sub = Counter((m.get("KEYBOARD_TYPE") or "").strip().lower()
                  for m in meta.values() if m["LAYOUT"]==lay)
    print(f"  passing {lay}: KB {dict(sub)}")

# (3) distinct pids present in shipped table, per layout
log("scanning shipped table for distinct pids per layout ...")
pids_in_table = defaultdict(set)
rows_by_layout = Counter()
with open(TSV, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5: continue
        layout = parts[0]
        rows_by_layout[layout] += 1
        for tok in parts[4:]:
            # tok like "(92, 72, 100001, 0)"; pid is 3rd int
            try:
                inner = tok.strip()[1:-1].split(",")
                pid = int(inner[2])
                pids_in_table[layout].add(pid)
            except (IndexError, ValueError):
                continue
log("table scan done")
print("TABLE distinct pids by layout:", {k: len(v) for k,v in pids_in_table.items()})
print("TABLE rows by layout:", dict(rows_by_layout))

# (4) distinct pids surviving load_strokes(min_samples=10)+build_cells(min_cell_samples=10)
log("load_strokes(min_samples=10) ...")
strokes = load_strokes(TSV, ngram_len=3, wpm_threshold=0, min_samples=10)
cells = build_cells(strokes, wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
pids_used = defaultdict(set)
cells_by_layout = Counter()
for c in cells:
    cells_by_layout[c.layout] += 1
    for s in c.samples:
        pids_used[c.layout].add(s[2])
print("USED-IN-FIT distinct pids by layout (min_samples=10,min_cell_samples=10):",
      {k: len(v) for k,v in pids_used.items()})
print("CELLS by layout:", dict(cells_by_layout))
log("done")
