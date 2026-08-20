#!/usr/bin/env python3
import csv, sys, glob, os, collections
COL = {n:i for i,n in enumerate(
 "kind op scalar backend shape_class m n k batch chosen_origin chosen_algo calls "
 "native_route_existed native_route_supported library uplo side diag transA transB".split())}
TR={0:"N",1:"T",2:"C"}
d=sys.argv[1]
per_suite=collections.Counter(); per_suite_rows=collections.Counter()
shape_owner=collections.defaultdict(collections.Counter)
out=[]
for f in sorted(glob.glob(os.path.join(d,"*.csv"))):
    suite=os.path.basename(f)[:-4]
    for r in csv.reader(open(f)):
        if not r or r[0]!="reached" or r[1]!="gemm": continue
        if not r[COL["scalar"]].startswith("complex"): continue
        c=int(r[COL["calls"]] or 0)
        per_suite[suite]+=c; per_suite_rows[suite]+=1
        m,n,k,b=(int(r[COL[x]]) for x in ("m","n","k","batch"))
        tf=TR[int(r[COL["transA"]])]+TR[int(r[COL["transB"]])]
        shape_owner[(r[COL["scalar"]],m,n,k,b,tf)][suite]+=c
        out.append([suite,r[COL["scalar"]],m,n,k,b,tf,c,r[COL["chosen_origin"]],r[COL["chosen_algo"]]])
print("suite,complex_gemm_rows,complex_gemm_calls")
for s,c in per_suite.most_common():
    print(f"{s},{per_suite_rows[s]},{c}")
print()
print("top complex shapes with owning suites:")
tot=collections.Counter()
for k2,v in shape_owner.items(): tot[k2]=sum(v.values())
for k2,c in tot.most_common(20):
    print(f"  {k2[0]:>15} m={k2[1]:5d} n={k2[2]:5d} k={k2[3]:4d} b={k2[4]:4d} {k2[5]}  calls={c:6d}  {dict(shape_owner[k2])}")
with open(sys.argv[2],"w",newline="") as fh:
    w=csv.writer(fh); w.writerow("suite scalar m n k batch trans calls origin algo".split())
    for r in sorted(out,key=lambda r:(-r[7])): w.writerow(r)
