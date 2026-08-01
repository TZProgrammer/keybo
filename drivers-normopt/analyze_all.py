"""Drive the SHIPPED `keybo analyze --json` in-process over all produced + field layouts."""
import sys, json, io, contextlib
sys.path.insert(0, "/tmp/normopt/src")
from keybo.cli import analyze as AN
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.layouts import NAMED_LAYOUTS
import argparse

produced = {}
for arm in "ABC":
    for s in range(10):
        produced[f"{arm}-s{s}"] = json.load(open(f"/tmp/normopt/runs/{arm}-s{s}.json"))["layout"]

field = dict(_EXTRA_NAMED)
field["graphite"] = NAMED_LAYOUTS["graphite"]
field["semimak"]  = NAMED_LAYOUTS["semimak"]
field["BALL-1"]   = "flmpg-yuo,sntcdireahkxbwv'.jzq"
field["arm-B"]    = "flmpg-yuo,sntdcireahkxbwv'.jzq"
field["arm-A"]    = "udy.,fgpmliheaocsntr-k'qjwzbvx"
field["ng:anchor-AALTO"]     = "lnfdg-,yehcrstmaoiupxzbwv.kq'j"
field["ng:anchor-COMMUNITY"] = "cstr,kdeaigflnmypo.uwzqbxvh-j'"
field["ng:anchor-POOL"]      = "cyea,krstpguoi-mlndfwj'.qhvxzb"
field["ng:registered-best"]  = "ufio,vdnrmyhea.ptsclkj'-qgbzxw"
field["ng:droppool-best"]    = "clndf,geihrmstp.aouywzxbvk-qj'"
field["ng:10M-AALTO-champ"]  = "lnfdg-,yehcrstmaoiupxqbwv.k'jz"
field.pop("p13stab-win", None)   # not C30M (has ;/ ) — analyze would N/A it

# one analyze call over the DISTINCT layout strings
order, seen = [], set()
for lay in list(produced.values()) + list(field.values()):
    if lay not in seen:
        seen.add(lay); order.append(lay)
print(f"analyzing {len(order)} distinct layouts", file=sys.stderr)

p = argparse.ArgumentParser(); AN.add_arguments(p)
args = p.parse_args([*order, "--json", "--attribution"])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = AN.run(args)
assert rc == 0, f"analyze rc={rc}"
rows = json.loads(buf.getvalue())
json.dump(rows, open("/tmp/normopt/runs/analyze-all.json","w"), indent=1, sort_keys=True)
# name map so the report can talk in names
names = {"produced": produced, "field": field}
json.dump(names, open("/tmp/normopt/runs/names.json","w"), indent=1, sort_keys=True)
print(f"OK rows={len(rows['rows'])} corpus={rows['corpus']}", file=sys.stderr)
