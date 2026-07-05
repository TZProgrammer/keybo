"""How much of R1W's per-cell rho on dvorak is 'b credit' vs geometry transfer?
Score b ALONE (no model) against the dvorak fold's observed cell times with the same
bucket-centered spearman. If b-alone rho is close to R1W's full rho, the practice term
carries most of the per-cell metric (legit prediction, but geometry credit is smaller
than the headline rho suggests). tau is b-invariant either way (proven separately)."""

import json

import numpy as np

from keybo.data.strokes import load_strokes
from keybo.training.validate import _centered_spearman, build_cells

rows = load_strokes("bistrokes_v3.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
cells = [
    c
    for c in build_cells(rows, wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
    if c.layout == "dvorak"
]
obs = np.array([c.obs for c in cells])
for seed in (0, 1, 2):
    b = json.load(open(f"models/bigram_r1w_seed{seed}.practice.json"))
    bvec = np.array([b.get(c.ngram, 0.0) for c in cells])
    rho = _centered_spearman(cells, bvec, obs)
    print(f"seed {seed}: dvorak rho of b ALONE = {rho:+.3f}")
print("reference: R1W full (g+b) dvorak rho +.562/+.589/+.574; B (no b) +.310/+.313/+.301")
