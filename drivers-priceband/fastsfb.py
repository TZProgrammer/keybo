"""Vectorized kmstats gauges over a char-index frequency matrix, so a 435-swap sweep is
milliseconds instead of minutes. Verified against KmStats.stats() in verify()."""
import numpy as np
from keybo.analysis.kmstats import _KEYS, STAT_NAMES, _distance, _is_lsb, _is_roll, _is_redirect, _direction
from keybo.data.corpus import load_frequencies, production_corpus_dir

CHARS = "abcdefghijklmnopqrstuvwxyz',.-"
NC = len(CHARS)

class FastGauges:
    """sfb / sfb-dist / lsb / lsb-dist / sfs / sfs-dist over the 30-slot board.

    Layout-restricted denominators exactly as kmstats: only n-grams fully on the 30 keys.
    Trigram gauges (alt/roll/sr-roll/redir) are cubes — built lazily, they are big."""
    def __init__(self, corpus=None):
        d = production_corpus_dir(corpus)
        bi = load_frequencies(str(d/'bigrams.txt')); sk = load_frequencies(str(d/'1-skip31.txt'))
        ci = {c:i for i,c in enumerate(CHARS)}
        def mat(freqs):
            F = np.zeros((NC,NC))
            for ng,f in freqs.items():
                if len(ng)==2 and ng[0] in ci and ng[1] in ci: F[ci[ng[0]],ci[ng[1]]] += f
            return F
        self.FB = mat(bi); self.FS = mat(sk)
        self.bi_total = self.FB.sum(); self.sk_total = self.FS.sum()
        K = _KEYS  # 30 slot keys, slot order
        n=30
        self.SFB = np.zeros((n,n)); self.SFBD = np.zeros((n,n))
        self.LSB = np.zeros((n,n)); self.LSBD = np.zeros((n,n))
        self.SFR = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                a,b = K[i],K[j]
                if i==j: self.SFR[i,j]=1.0; continue
                if a.finger==b.finger:
                    self.SFB[i,j]=1.0; self.SFBD[i,j]=_distance(a,b)
                if _is_lsb(a,b):
                    self.LSB[i,j]=1.0; self.LSBD[i,j]=abs(a.x-b.x)
        self.ci = ci
    def perm(self, lay30):
        slot={ch:i for i,ch in enumerate(lay30)}
        return np.array([slot[c] for c in CHARS], dtype=np.intp)
    def bigrams(self, p):
        ix = np.ix_(p,p)
        return dict(
            sfb=100*(self.FB*self.SFB[ix]).sum()/self.bi_total,
            **{"sfb-dist":100*(self.FB*self.SFBD[ix]).sum()/self.bi_total},
            **{"lsb":100*(self.FB*self.LSB[ix]).sum()/self.bi_total},
            **{"lsb-dist":100*(self.FB*self.LSBD[ix]).sum()/self.bi_total},
            sfr=100*(self.FB*self.SFR[ix]).sum()/self.bi_total,
            sfs=100*(self.FS*self.SFB[ix]).sum()/self.sk_total,
            **{"sfs-dist":100*(self.FS*self.SFBD[ix]).sum()/self.sk_total},
        )
    def sfb_only(self, p):
        return 100*(self.FB*self.SFB[np.ix_(p,p)]).sum()/self.bi_total

class FastTriGauges:
    """alt / roll / sr-roll / redir. 30^3 cubes over char-index trigram cube."""
    def __init__(self, corpus=None):
        d = production_corpus_dir(corpus)
        tri = load_frequencies(str(d/'trigrams.txt'))
        ci = {c:i for i,c in enumerate(CHARS)}
        F = np.zeros((NC,NC,NC))
        for ng,f in tri.items():
            if len(ng)==3 and all(c in ci for c in ng): F[ci[ng[0]],ci[ng[1]],ci[ng[2]]] += f
        self.F = F; self.total = F.sum()
        K=_KEYS; n=30
        self.ALT=np.zeros((n,n,n)); self.ROLL=np.zeros((n,n,n))
        self.SRROLL=np.zeros((n,n,n)); self.REDIR=np.zeros((n,n,n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    A,B,C_=K[i],K[j],K[k]
                    if A.hand!=B.hand and A.hand==C_.hand: self.ALT[i,j,k]=1.0
                    r=_is_roll(A,B,C_)
                    if r:
                        self.ROLL[i,j,k]=1.0
                        if A.row==B.row==C_.row: self.SRROLL[i,j,k]=1.0
                    if _is_redirect(A,B,C_): self.REDIR[i,j,k]=1.0
    def stats(self, p):
        ix=np.ix_(p,p,p)
        return {"alt":100*(self.F*self.ALT[ix]).sum()/self.total,
                "roll":100*(self.F*self.ROLL[ix]).sum()/self.total,
                "sr-roll":100*(self.F*self.SRROLL[ix]).sum()/self.total,
                "redir":100*(self.F*self.REDIR[ix]).sum()/self.total}

def verify(layouts):
    from keybo.analysis.kmstats import KmStats
    d=production_corpus_dir()
    kms=KmStats(load_frequencies(str(d/'bigrams.txt')),load_frequencies(str(d/'1-skip31.txt')),
                load_frequencies(str(d/'trigrams.txt')))
    fg=FastGauges(); ft=FastTriGauges()
    worst=0.0; print("== fastsfb verification vs KmStats.stats() ==")
    print(f"{'layout':<14}{'gauge':<10}{'kmstats':>12}{'fast':>12}{'abserr':>11}")
    for n,s in layouts.items():
        ref=kms.stats(s); p=fg.perm(s)
        got={**fg.bigrams(p), **ft.stats(p)}
        for k in STAT_NAMES:
            e=abs(ref[k]-got[k]); worst=max(worst,e)
            if e>1e-9: print(f"{n:<14}{k:<10}{ref[k]:>12.6f}{got[k]:>12.6f}{e:>11.2e}")
    print(f"worst abs error over {len(layouts)} layouts x {len(STAT_NAMES)} gauges: {worst:.3e}")
    return worst
