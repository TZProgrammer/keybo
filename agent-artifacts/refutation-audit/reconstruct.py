#!/usr/bin/env python3
"""Reconstruct the finding -> 3-lens-verdict mapping for the ultracode audit.

The digest (journal-digest.json) lost the join: it has 37 findings and 110
verdicts as two flat lists with no key between them. But every verify agent's
FIRST user message embeds the finding under test (title/file/symbol/...), and
its StructuredOutput tool call carries the verdict. So we can re-join from the
raw per-agent transcripts, which survive in the session dir.

Emits refutation-map.json: one record per DISTINCT finding, with its votes.
"""
import json, os, re, sys, glob
from collections import Counter, defaultdict

W = os.path.expanduser(
    "~/.claude/projects/-local-home-zegertho-agent-workspaces-ultracode-audit/"
    "43318f15-a9ab-42db-94c7-199bf3619621/subagents/workflows/wf_32ff2687-938")

def msg_text(m):
    """Flatten a message content into text."""
    if m is None: return ""
    c = m.get("content")
    if isinstance(c, str): return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, dict):
                if b.get("type") == "text": out.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    out.append(json.dumps(b.get("input", {})))
        return "\n".join(out)
    return ""

def first_prompt(recs):
    for r in recs:
        if r.get("type") == "user" and isinstance(r.get("message"), dict):
            t = msg_text(r["message"])
            if t: return t
    return ""

def structured_outputs(recs):
    """Every StructuredOutput tool_use input, in order."""
    outs = []
    for r in recs:
        if r.get("type") != "assistant": continue
        m = r.get("message") or {}
        c = m.get("content")
        if not isinstance(c, list): continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                nm = (b.get("name") or "")
                if "StructuredOutput" in nm or "structured_output" in nm.lower():
                    outs.append(b.get("input"))
    return outs

FIELD_RE = {
    "title":            r"^- \*\*title:\*\* (.*)$",
    "file":             r"^- \*\*file:\*\* (.*)$",
    "symbol":           r"^- \*\*symbol:\*\* (.*)$",
    "verdict":          r"^- \*\*verdict claimed:\*\* (.*)$",
    "confidence":       r"^- \*\*confidence claimed:\*\* (.*)$",
    "label_claim":      r"^- \*\*label_claim \(what the name claims\):\*\* (.*)$",
    "referent_reality": r"^- \*\*referent_reality \(what the finder says it does\):\*\* (.*)$",
    "blast_radius":     r"^- \*\*blast_radius claimed:\*\* (.*)$",
}

def parse_verify_prompt(t):
    """A verify-lens prompt. Returns the finding fields + which lens, else None."""
    if "## THE FINDING UNDER TEST" not in t: return None
    seg = t.split("## THE FINDING UNDER TEST", 1)[1]
    f = {}
    for k, rx in FIELD_RE.items():
        m = re.search(rx, seg, re.M)
        f[k] = (m.group(1).strip() if m else None)
    # reproducer_cmd: the first fenced block after the reproducer_cmd label
    m = re.search(r"- \*\*reproducer_cmd:\*\*\s*\n```\n(.*?)\n```", seg, re.S)
    f["reproducer_cmd"] = m.group(1) if m else None
    m = re.search(r"- \*\*reproducer_output the finder reported:\*\*\s*\n```\n(.*?)\n```", seg, re.S)
    f["reproducer_output"] = m.group(1) if m else None
    return f

def parse_triage_prompt(t):
    if "## THE SURVIVING FINDING" not in t: return None
    m = re.search(r"## THE SURVIVING FINDING \(from finder `([^`]*)`\)", t)
    origin = m.group(1) if m else None
    seg = t.split("## THE SURVIVING FINDING", 1)[1]
    m = re.search(r"^- \*\*title:\*\* (.*)$", seg, re.M)
    return {"origin": origin, "title": m.group(1).strip() if m else None}

def agent_role(t):
    """Classify the agent by its prompt."""
    if "## THE FINDING UNDER TEST" in t: return "verify"
    if "## THE SURVIVING FINDING" in t: return "triage"
    if "## YOUR JOB: TRIAGE" in t: return "triage"
    if "THE NEXT ROUND'S WORK-LIST" in t: return "critic"
    return "finder"

def main():
    files = sorted(glob.glob(os.path.join(W, "agent-*.jsonl")))
    agents = {}
    for p in files:
        aid = os.path.basename(p)[len("agent-"):-len(".jsonl")]
        try:
            recs = [json.loads(l) for l in open(p) if l.strip()]
        except Exception as e:
            print(f"WARN unreadable {p}: {e}", file=sys.stderr); continue
        t = first_prompt(recs)
        outs = structured_outputs(recs)
        agents[aid] = {"path": p, "prompt": t, "role": agent_role(t), "outputs": outs,
                       "n_recs": len(recs)}

    print(f"agents on disk: {len(agents)}")
    print("roles:", Counter(a["role"] for a in agents.values()))

    # --- the verify agents: join finding -> vote
    byfinding = defaultdict(lambda: {"finding": None, "votes": []})
    for aid, a in agents.items():
        if a["role"] != "verify": continue
        f = parse_verify_prompt(a["prompt"])
        if f is None: continue
        # last StructuredOutput wins (retries produce several)
        vote = a["outputs"][-1] if a["outputs"] else None
        # lens: recover from the lens brief text that the prompt embeds
        key = f["title"]
        rec = byfinding[key]
        rec["finding"] = f
        rec["votes"].append({"agent": aid, "vote": vote,
                             "n_outputs": len(a["outputs"])})
    print(f"distinct findings that reached a verify panel: {len(byfinding)}")

    out = []
    for title, rec in byfinding.items():
        votes = rec["votes"]
        good = [v for v in votes if isinstance(v["vote"], dict)]
        nref = sum(1 for v in good if v["vote"].get("refuted") is True)
        nvotes = len(good)
        survives = nvotes > 0 and nref < 2
        out.append({
            "title": title,
            "file": rec["finding"]["file"],
            "symbol": rec["finding"]["symbol"],
            "verdict_claimed": rec["finding"]["verdict"],
            "confidence_claimed": rec["finding"]["confidence"],
            "label_claim": rec["finding"]["label_claim"],
            "referent_reality": rec["finding"]["referent_reality"],
            "blast_radius_claimed": rec["finding"]["blast_radius"],
            "reproducer_cmd": rec["finding"]["reproducer_cmd"],
            "reproducer_output": rec["finding"]["reproducer_output"],
            "n_verify_agents": len(votes),
            "n_votes_returned": nvotes,
            "n_refuted": nref,
            "survives": survives,
            "votes": [{"agent": v["agent"], "refuted": (v["vote"] or {}).get("refuted"),
                       "lens_applicable": (v["vote"] or {}).get("lens_applicable"),
                       "reasoning": (v["vote"] or {}).get("reasoning"),
                       "evidence_cmd": (v["vote"] or {}).get("evidence_cmd"),
                       "evidence_output": (v["vote"] or {}).get("evidence_output"),
                       "verdict_correction": (v["vote"] or {}).get("verdict_correction")}
                      for v in votes],
        })
    out.sort(key=lambda r: (not r["survives"], r["file"] or "", r["title"] or ""))

    killed = [r for r in out if not r["survives"]]
    surv = [r for r in out if r["survives"]]
    print(f"\n=== RECONSTRUCTED TALLY ===")
    print(f"distinct findings verified : {len(out)}")
    print(f"  survived (confirmed)     : {len(surv)}")
    print(f"  KILLED (refuted)         : {len(killed)}")
    print(f"total votes returned       : {sum(r['n_votes_returned'] for r in out)}")
    print(f"total refuted votes        : {sum(r['n_refuted'] for r in out)}")
    print(f"\nvote breakdown of KILLED findings (n_refuted/n_votes):")
    print("  ", Counter(f"{r['n_refuted']}/{r['n_votes_returned']}" for r in killed))
    print(f"vote breakdown of SURVIVORS:")
    print("  ", Counter(f"{r['n_refuted']}/{r['n_votes_returned']}" for r in surv))

    dest = "/local/home/zegertho/agent/state/refaudit/artifacts/refutation-map.json"
    with open(dest, "w") as fh:
        json.dump({"killed": killed, "survived": surv,
                   "meta": {"agents_on_disk": len(agents),
                            "roles": dict(Counter(a["role"] for a in agents.values()))}},
                  fh, indent=1)
    print(f"\nwrote {dest}")

if __name__ == "__main__":
    main()
