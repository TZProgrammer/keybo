import os, pickle, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
from keybo.data.strokes import load_strokes
t0=time.time()
rows=load_strokes("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
print(f"loaded {len(rows)} rows in {time.time()-t0:.1f}s")
pickle.dump(rows, open("/tmp/stagger-work/bi_rows.pkl","wb"), protocol=5)
print("cached")
