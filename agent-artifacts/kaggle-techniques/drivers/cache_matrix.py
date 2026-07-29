import numpy as np, time
from keybo.testkit import assert_module_under
assert_module_under("keybo", "/tmp/kaggle")
from keybo.data.strokes import load_strokes
from keybo.training.train import _build_matrix_full
from keybo.geometry import ROW_STAGGERED_30
t0=time.time()
rows = load_strokes("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
X,y,ng,lay,cnt = _build_matrix_full(rows, ngram="bigram", geometry=ROW_STAGGERED_30, target_space="LOGRAT")
np.savez_compressed("/tmp/kaggle-work/matrix_bigram_lograt.npz",
                    X=X, y=y, ngrams=ng.astype(str), layouts=lay.astype(str), counts=cnt)
print(f"cached X={X.shape} in {time.time()-t0:.0f}s -> matrix_bigram_lograt.npz", flush=True)
