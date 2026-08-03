"""Q06 — the CORRECTED cross-layout transfer test (registered in ADDENDUM 3, 5fc572a).

q04 compared b_qwerty vs b_nonqwerty (corr 0.6489, rms 0.142384) against a floor built by
split-halving the QWERTY rows (corr 0.991864) and emitted "DISAGREE BEYOND NOISE". THAT FLOOR IS NOT
DESIGN-MATCHED: qwerty is ~98.7% of the SAMPLES, so b_nonqwerty is estimated from ~1.3% of the data
-- far noisier (sd 0.0909 vs 0.1816) and shrunk harder by the (sum(c)+100) denominator (slope
0.3247). A floor from two data-rich estimates cannot bound one data-rich vs one data-poor estimate.

C-1 reliability of BOTH sides (split-half b_nonqwerty too, same samples-within-row method).
C-2 DISATTENUATION: corr_true = corr_obs / sqrt(rel_q * rel_nq).
    RULE: corr_true >= 0.90 => the apparent non-transfer is measurement noise, b DOES transfer.
          corr_true <= 0.80 => a genuine layout-specific component survives.
C-3 the SAMPLE-MATCHED floor: subsample QWERTY to non-qwerty's per-ngram sample counts, fit b on
    that, and split-half IT. Both sides then equally shrunk and equally noisy. THIS is the floor a
    non-transfer claim must clear, not q04's.
C-4 placebo: b_qwerty vs a matched-SIZE qwerty subsample must land at the MATCHED floor, not the
    data-rich floor -- guards against the subsampling itself creating the gap.

REGISTERED PREDICTION (against my own draft finding): corr_true rises substantially toward 1 and the
sample-matched floor lands far below 0.9919 => most of the apparent non-transfer is an artifact.
"""
import copy
import json
import time
from collections import defaultdict

import numpy as np
from _guard import ART, BOOT_SEED, assert_d5, load_rows

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


log("D5:")
assert_d5()

from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import _build_matrix_full, fit_practice_term, train_bigram_model  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

G = ROW_STAGGERED_31
out = {"boot_seed": BOOT_SEED, "geometry": "ROW_STAGGERED_31",
       "registered_in": "PREREGISTRATIONS.md FREQCORRECT-1 ADDENDUM 3 (commit 5fc572a)"}

log("loading rows")
rows = load_rows()
qrows = [r for r in rows if r.layout == "qwerty"]
nqrows = [r for r in rows if r.layout != "qwerty"]
n_q_samples = sum(len(r.samples) for r in qrows)
n_nq_samples = sum(len(r.samples) for r in nqrows)
out["sample_asymmetry"] = {
    "n_rows_qwerty": len(qrows), "n_rows_nonqwerty": len(nqrows),
    "n_samples_qwerty": n_q_samples, "n_samples_nonqwerty": n_nq_samples,
    "qwerty_share_of_samples": float(n_q_samples / (n_q_samples + n_nq_samples)),
}
log(f"  qwerty {len(qrows)} rows / {n_q_samples} samples ; non-qwerty {len(nqrows)} rows / "
    f"{n_nq_samples} samples  => qwerty holds "
    f"{100 * n_q_samples / (n_q_samples + n_nq_samples):.2f}% of SAMPLES")


def fit_b_on(subset, seed=0, k=100.0):
    """b fitted by the shipped estimator on `subset`: a b-free g, then the shrunk residual mean."""
    X, y, ngrams, layouts, counts = _build_matrix_full(subset, ngram="bigram", geometry=G,
                                                       target_space="LOGRAT")
    m = train_bigram_model(subset, target_wpm=90.0, geometry=G, random_state=seed, n_jobs=48,
                           practice_term=False)
    return fit_practice_term(ngrams, y - m.predict(X), counts, k=k)


def split_samples(subset, rng):
    """Two row-lists with the SAME ngrams, each carrying a random half of every row's samples."""
    h1, h2 = [], []
    for r in subset:
        if len(r.samples) < 4:
            continue
        idx = rng.permutation(len(r.samples))
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
        r1, r2 = copy.copy(r), copy.copy(r)
        r1.samples = [r.samples[j] for j in a]
        r2.samples = [r.samples[j] for j in b]
        h1.append(r1)
        h2.append(r2)
    return h1, h2


def agree(b1, b2):
    """(corr, rms, n_shared) between two b maps on their shared ngrams."""
    sh = sorted(set(b1) & set(b2))
    if len(sh) < 20:
        return np.nan, np.nan, len(sh)
    a = np.array([b1[n] for n in sh])
    bb = np.array([b2[n] for n in sh])
    return float(np.corrcoef(a, bb)[0, 1]), float(np.sqrt(((bb - a) ** 2).mean())), len(sh)


def split_half_reliability(subset, label, n_splits, rng):
    """Spearman-Brown-uncorrected split-half agreement -- the reliability of a HALF-SIZE estimate."""
    cs, rs = [], []
    for i in range(n_splits):
        h1, h2 = split_samples(subset, rng)
        c, r, n = agree(fit_b_on(h1, seed=i), fit_b_on(h2, seed=i))
        if np.isfinite(c):
            cs.append(c)
            rs.append(r)
            log(f"  {label} split {i}: n_shared={n} corr={c:.4f} rms={r:.6f}")
    assert cs, f"{label}: every split skipped -- refusing to publish a floorless comparison"
    return np.array(cs), np.array(rs)


# ============================================================ the cross-layout comparison itself
log("fitting b on QWERTY-only and NON-QWERTY-only (the comparison under test)")
b_q = fit_b_on(qrows)
b_nq = fit_b_on(nqrows)
corr_obs, rms_obs, n_sh = agree(b_q, b_nq)
sh = sorted(set(b_q) & set(b_nq))
out["cross_layout"] = {
    "corr_obs": corr_obs, "rms_obs": rms_obs, "n_shared": n_sh,
    "sd_b_qwerty": float(np.std([b_q[n] for n in sh], ddof=1)),
    "sd_b_nonqwerty": float(np.std([b_nq[n] for n in sh], ddof=1)),
    "slope_nq_on_q": float(np.polyfit([b_q[n] for n in sh], [b_nq[n] for n in sh], 1)[0]),
}
log(f"  CROSS-LAYOUT corr = {corr_obs:.4f}  rms = {rms_obs:.6f}  n_shared = {n_sh}")

# ============================================================ C-1 reliability of BOTH sides
NS = 8
log(f"C-1: split-half reliability of BOTH sides ({NS} splits each)")
rng = np.random.default_rng(BOOT_SEED)
rel_q_c, rel_q_r = split_half_reliability(qrows, "qwerty", NS, rng)
rel_nq_c, rel_nq_r = split_half_reliability(nqrows, "non-qwerty", NS, rng)
out["c1_reliability"] = {
    "qwerty": {"corr_mean": float(rel_q_c.mean()), "corr_sd": float(rel_q_c.std(ddof=1)),
               "rms_mean": float(rel_q_r.mean()), "n_splits": int(len(rel_q_c))},
    "nonqwerty": {"corr_mean": float(rel_nq_c.mean()), "corr_sd": float(rel_nq_c.std(ddof=1)),
                  "rms_mean": float(rel_nq_r.mean()), "n_splits": int(len(rel_nq_c))},
}
log(f"  rel_q = {rel_q_c.mean():.4f} +- {rel_q_c.std(ddof=1):.4f} ; "
    f"rel_nq = {rel_nq_c.mean():.4f} +- {rel_nq_c.std(ddof=1):.4f}")

# ============================================================ C-2 disattenuation
rel_q, rel_nq = float(rel_q_c.mean()), float(rel_nq_c.mean())
denom = np.sqrt(max(rel_q, 1e-9) * max(rel_nq, 1e-9)) if rel_q > 0 and rel_nq > 0 else np.nan
corr_true = corr_obs / denom if np.isfinite(denom) and denom > 0 else np.nan
out["c2_disattenuated"] = {
    "corr_obs": corr_obs, "rel_qwerty": rel_q, "rel_nonqwerty": rel_nq,
    "sqrt_rel_product": float(denom), "corr_true": float(corr_true),
    "registered_rule": ">=0.90 => b DOES transfer (noise artifact); <=0.80 => genuine layout component",
    "verdict": ("b DOES TRANSFER -- apparent non-transfer is measurement noise" if corr_true >= 0.90
                else "GENUINE LAYOUT-SPECIFIC COMPONENT SURVIVES" if corr_true <= 0.80
                else "INTERMEDIATE (0.80-0.90) -- neither registered branch fires"),
    "caveat": ("reliabilities are HALF-SIZE estimates; using them undercorrects (a full-size "
               "estimate is more reliable than a half-size one), so corr_true here is a LOWER "
               "BOUND on the disattenuated agreement"),
}
log(f"C-2: corr_true = {corr_obs:.4f} / sqrt({rel_q:.4f} * {rel_nq:.4f}) = {corr_true:.4f} "
    f"=> {out['c2_disattenuated']['verdict']}")

# ============================================================ C-3 the SAMPLE-MATCHED floor
log("C-3: the SAMPLE-MATCHED floor -- qwerty subsampled to non-qwerty per-ngram sample counts")
nq_counts = defaultdict(int)
for r in nqrows:
    nq_counts[r.ngram] += len(r.samples)
# the per-ngram target count: what non-qwerty actually had for that ngram (median over its layouts)
nq_per_ngram = {}
tmp = defaultdict(list)
for r in nqrows:
    tmp[r.ngram].append(len(r.samples))
for ng, v in tmp.items():
    nq_per_ngram[ng] = int(np.median(v))
med_target = int(np.median(list(nq_per_ngram.values())))
log(f"  non-qwerty per-ngram sample counts: median {med_target}, "
    f"range [{min(nq_per_ngram.values())}, {max(nq_per_ngram.values())}]")


def matched_qwerty(rng):
    """QWERTY rows subsampled so each ngram carries non-qwerty's per-ngram sample count."""
    outr = []
    for r in qrows:
        k = nq_per_ngram.get(r.ngram, med_target)
        k = min(k, len(r.samples))
        if k < 4:
            continue
        idx = rng.permutation(len(r.samples))[:k]
        rr = copy.copy(r)
        rr.samples = [r.samples[j] for j in idx]
        outr.append(rr)
    return outr


NM = 8
log(f"  measuring the matched floor over {NM} splits")
rngm = np.random.default_rng(BOOT_SEED + 21)
m_c, m_r = [], []
for i in range(NM):
    mq = matched_qwerty(rngm)
    h1, h2 = split_samples(mq, rngm)
    c, r, n = agree(fit_b_on(h1, seed=i), fit_b_on(h2, seed=i))
    if np.isfinite(c):
        m_c.append(c)
        m_r.append(r)
        log(f"  matched split {i}: n_shared={n} corr={c:.4f} rms={r:.6f}")
assert m_c, "matched floor EMPTY -- refusing to publish a floorless comparison"
m_c, m_r = np.array(m_c), np.array(m_r)
out["c3_sample_matched_floor"] = {
    "n_splits": int(len(m_c)),
    "corr_mean": float(m_c.mean()), "corr_sd": float(m_c.std(ddof=1)),
    "corr_p05": float(np.percentile(m_c, 5)),
    "rms_mean": float(m_r.mean()), "rms_p95": float(np.percentile(m_r, 95)),
    "q04_datarich_floor_corr_for_reference": 0.9918637968546737,
    "q04_datarich_floor_rms_for_reference": 0.020322297194418148,
    "design": ("QWERTY subsampled to non-qwerty's PER-NGRAM sample counts, then split-halved. Both "
               "sides are then equally shrunk by the (sum(c)+k) denominator and equally noisy, so "
               "this floor MATCHES the cross-layout comparison design. q04's floor did not."),
}
log(f"C-3 MATCHED FLOOR: corr {m_c.mean():.4f} +- {m_c.std(ddof=1):.4f} (p05 "
    f"{np.percentile(m_c, 5):.4f}); rms {m_r.mean():.6f} "
    f"[q04's mis-matched floor was corr 0.9919 / rms 0.020322]")

# ============================================================ the corrected verdict
clears = bool(corr_obs < np.percentile(m_c, 5))
out["corrected_verdict"] = {
    "cross_layout_corr": corr_obs,
    "matched_floor_corr_mean": float(m_c.mean()), "matched_floor_corr_p05": float(np.percentile(m_c, 5)),
    "rms_ratio_cross_over_matched_floor": float(rms_obs / m_r.mean()),
    "rms_ratio_cross_over_q04_floor": float(rms_obs / 0.020322297194418148),
    "below_matched_floor_p05": clears,
    "verdict": ("GENUINE NON-TRANSFER -- cross-layout agreement is below even the sample-matched floor"
                if clears else
                "NO DEMONSTRATED NON-TRANSFER -- the apparent gap is within what matched noise and "
                "differential shrinkage produce"),
    "note": ("q04's JSON emitted 'DISAGREE BEYOND NOISE (contamination)' at a 7.01x rms ratio "
             "against a MIS-MATCHED floor. This entry supersedes it."),
}
log(f"CORRECTED VERDICT: {out['corrected_verdict']['verdict']}")
log(f"  rms ratio vs MATCHED floor = {rms_obs / m_r.mean():.2f}x "
    f"(vs q04's mis-matched {rms_obs / 0.020322297194418148:.2f}x)")

# ============================================================ C-4 placebo
log("C-4 placebo: full b_qwerty vs a matched-SIZE qwerty subsample (same layout, so truth = agree)")
rngp = np.random.default_rng(BOOT_SEED + 31)
pl_c, pl_r = [], []
for i in range(4):
    c, r, n = agree(b_q, fit_b_on(matched_qwerty(rngp), seed=100 + i))
    if np.isfinite(c):
        pl_c.append(c)
        pl_r.append(r)
        log(f"  placebo {i}: n_shared={n} corr={c:.4f} rms={r:.6f}")
out["c4_placebo_same_layout_matched_size"] = {
    "n": len(pl_c), "corr_mean": float(np.mean(pl_c)) if pl_c else None,
    "rms_mean": float(np.mean(pl_r)) if pl_r else None,
    "interpretation": ("SAME layout, so any disagreement here is pure noise+shrinkage from the size "
                       "reduction. If this lands near the cross-layout 0.6489, the cross-layout gap "
                       "is explained by size alone."),
}
if pl_c:
    log(f"C-4 placebo: corr {np.mean(pl_c):.4f} rms {np.mean(pl_r):.6f} "
        f"[cross-layout was corr {corr_obs:.4f} rms {rms_obs:.6f}]")

out["wall_s"] = time.time() - t0
path = f"{ART}/q06_transfer.json"
json.dump(out, open(path, "w"), indent=1)
log(f"wrote {path}  ({out['wall_s']:.1f}s)")
