"""The 14-board field (C30M charset: ' and - replace ; and /), copied verbatim from
`state/pair-perturb/artifacts/v01_table.json` so my numbers are comparable to the prior arm's.
qwerty30m is the OFF-FRONTIER control; the other 13 are the optimized field."""

FIELD = {
    "keybo-c30m": "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "p16-balance": "frlwg'uyoksntdc.ieahvxmpb,-jqz",
    "qwerty30m": "qwertyuiopasdfghjkl'zxcvbnm,.-",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "arm-A": "udy.,fgpmliheaocsntr-k'qjwzbvx",
    "arm-B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
}
OFF_FRONTIER = "qwerty30m"
OPTIMIZED = [b for b in sorted(FIELD) if b != OFF_FRONTIER]
