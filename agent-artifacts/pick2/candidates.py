"""pick2: the candidate set — every layout worth considering, with provenance.

Grouped by PROVENANCE because provenance is decision-relevant: a board with a real user
base carries evidence a one-off search output cannot, and vice versa.
"""

#: Real layouts people actually type on (community/historical). Provenance: external.
REAL = {
    "qwerty":       "qwertyuiopasdfghjkl;zxcvbnm,./",   # control / baseline
    "qwerty30m":    "qwertyuiopasdfghjkl'zxcvbnm,.-",   # charset-matched control
    "dvorak":       "',.pyfgcrlaoeuidhtns;qjkxbmwvz",
    "colemak":      "qwfpgjluy;arstdhneiozxcvbkm,./",
    "colemak-dh":   "qwfpbjluy;arstgmneiozxcdvkh,./",
    "canary":       "wlypbzfou;crstgmneiaqjvdkxh/,.",
    "recurva":      "frdpvqjuoysntcb.heaizxkgwml,;/",
    "sturdy":       "vmlcpxfouj;strdy.naeizkqgwbh',",
    "graphite":     "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak":      "flhvz'wuoysrntkcdeaixjbmqpg,.-",
}

#: This campaign's own search outputs. Provenance: internal, n=1 user (nobody).
CAMPAIGN = {
    # the ARMH-1 family (ledger:9423) — the registered lead and its siblings
    "arm-B":        "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "BALL-1":       "flmpg-yuo,sntcdireahkxbwv'.jzq",   # REGISTERED LEAD (ledger:9532)
    "MID":          "flmpg.yuo,sntcdireahkxbwv'-jzq",
    "armH-hdln":    "flmpg-,uoysntcdireahkxvwb.'jzq",
    # the pyuo/lsb family
    "keybo-lsb":    "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "flagship-c3":  "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "lsb-sib":      "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "keybo-c30m":   "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    # other named campaign boards
    "p16-balance":  "frlwg'uyoksntdc.ieahvxmpb,-jqz",
    "p13stab-win":  "rcgkmq.ouylsthd,naeixwbfvpjz;/",
    "p10-w05":      "clgmk.,ouysrthdpnaeiqxwbvfz/;j",
    "p11-w05":      "uoy,.vlmdgaeinprhtcs;/jkbfwxzq",
    "tri-best":     "fdnlmkioheswrtcpuabyjxqgv,.;z/",
    "bigram-d3":    "wae,ylrstfgoiupmncdbq;.kzhvxj/",
}

ALL = {**REAL, **CAMPAIGN}
PROVENANCE = {**dict.fromkeys(REAL, "real"), **dict.fromkeys(CAMPAIGN, "campaign")}

def validate():
    bad = {n: s for n, s in ALL.items() if len(s) != 30 or len(set(s)) != 30}
    if bad:
        raise SystemExit(f"malformed boards: {bad}")
    # every board must be a permutation of one of the two 30-key charsets, or flagged
    return len(ALL)

if __name__ == "__main__":
    n = validate()
    import collections
    cs = collections.Counter(frozenset(s) for s in ALL.values())
    print(f"{n} candidate boards, {len(cs)} distinct charsets")
    for c, k in cs.most_common():
        names = [nm for nm, s in ALL.items() if frozenset(s) == c]
        print(f"  n={k:2d}  {''.join(sorted(c))}  <- {names}")
