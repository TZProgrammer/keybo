import sys, json
sys.path.insert(0,"/tmp/normopt/src")
V=json.load(open("/tmp/normopt/runs/verdict.json"))
P,F=V["produced"],V["field"]
def diff(a,b):
    return [(i,a[i],b[i]) for i in range(30) if a[i]!=b[i]]
print("="*92); print("NEAR-REPRODUCTIONS — what exactly differs?"); print("="*92)
for run,tgt in (("C-s2","ng:droppool-best"),("B-s3","ng:droppool-best")):
    a=P[run]["layout"]; b=F[tgt]["layout"]
    d=diff(a,b)
    print(f"\n{run}  {a!r}   ms/char {P[run]['ms']:.6f}  blend(c) {P[run]['bl_c']:.6f}")
    print(f"{tgt}  {b!r}   ms/char {F[tgt]['ms']:.6f}  blend(c) {F[tgt]['bl_c']:.6f}")
    print(f"  Hamming {len(d)}/30 — differing positions (idx, mine, theirs):")
    for i,x,y in d:
        r,c = i//10, i%10
        print(f"    idx {i:2d} (row {r} col {c}, {'L' if c<5 else 'R'} hand): mine {x!r}  theirs {y!r}")
    # is it a pure permutation of the same key SET in those slots?
    print(f"  same multiset in differing slots? {sorted(x for _,x,_ in d)==sorted(y for _,_,y in d)}")
# also: how close are B and C winners to each other
print("\n"+"="*92); print("B winner vs C winner"); print("="*92)
a,b=V["winners"]["B"],V["winners"]["C"]
d=diff(a,b); print(f"  Hamming {len(d)}/30")
print(f"  B {a!r}\n  C {b!r}")
