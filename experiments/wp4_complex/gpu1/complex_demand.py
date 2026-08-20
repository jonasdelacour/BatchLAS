#!/usr/bin/env python3
"""Complex-specific GEMM demand from a coverage capture, probes subtracted."""
import csv, sys, collections

COL = {n:i for i,n in enumerate(
 "kind op scalar backend shape_class m n k batch chosen_origin chosen_algo calls "
 "native_route_existed native_route_supported library uplo side diag transA transB".split())}
TR = {0:"N",1:"T",2:"C"}

def rows(path, op="gemm"):
    return [r for r in csv.reader(open(path)) if r and r[0]=="reached" and r[1]==op]

def key(r):
    g=lambda n: r[COL[n]]
    return (g("scalar"),g("m"),g("n"),g("k"),g("batch"),g("transA"),g("transB"))

def subtract(rs, probe_path):
    p=collections.Counter()
    for r in rows(probe_path):
        p[key(r)] += int(r[COL["calls"]] or 0)
    out=[];dr=0;dc=0
    for r in rs:
        c=int(r[COL["calls"]] or 0); t=c-p.get(key(r),0)
        if t<=0: dr+=1; dc+=c; continue
        dc+=c-t; r=list(r); r[COL["calls"]]=str(t); out.append(r)
    print(f"# probes subtracted: dropped {dr} rows / {dc} calls; {len(out)} rows remain", file=sys.stderr)
    return out

def main(full, probe, outcsv):
    rs = subtract(rows(full), probe)
    g=lambda r,n: r[COL[n]]
    with open(outcsv,"w",newline="") as fh:
        w=csv.writer(fh)
        w.writerow("scalar m n k batch transA transB calls origin algo square NN min_dim max_dim".split())
        for r in sorted(rs, key=lambda r:(g(r,"scalar"), -int(g(r,"calls") or 0))):
            t=g(r,"scalar")
            if not t.startswith("complex"): continue
            m,n,k,b=(int(g(r,x)) for x in ("m","n","k","batch"))
            tA,tB=int(g(r,"transA")),int(g(r,"transB"))
            w.writerow([t,m,n,k,b,TR[tA],TR[tB],g(r,"calls"),g(r,"chosen_origin"),g(r,"chosen_algo"),
                        int(m==n==k), int(tA==0 and tB==0), min(m,n,k), max(m,n,k)])
    for t in ("complex<float>","complex<double>","float","double"):
        sel=[r for r in rs if g(r,"scalar")==t]
        if not sel: continue
        calls=sum(int(g(r,"calls") or 0) for r in sel)
        M=lambda r,x:int(g(r,x))
        def frac(pred):
            rr=sum(1 for r in sel if pred(r)); cc=sum(int(g(r,"calls") or 0) for r in sel if pred(r))
            return rr,cc
        sq=frac(lambda r: M(r,"m")==M(r,"n")==M(r,"k"))
        nn=frac(lambda r: M(r,"transA")==0 and M(r,"transB")==0)
        md=frac(lambda r: min(M(r,"m"),M(r,"n"),M(r,"k"))>=256)
        big=frac(lambda r: max(M(r,"m"),M(r,"n"))>=128)
        gate=frac(lambda r: min(M(r,"m"),M(r,"n"),M(r,"k"))>=256 and M(r,"transA")==0 and M(r,"transB")==0)
        nat=frac(lambda r: g(r,"chosen_origin")=="native")
        print(f"\n=== {t}: {len(sel)} rows, {calls} calls ===")
        print(f"  square m==n==k : {sq[0]} rows / {sq[1]} calls")
        print(f"  NN (no trans)  : {nn[0]} rows / {nn[1]} calls")
        print(f"  min_dim>=256   : {md[0]} rows / {md[1]} calls")
        print(f"  max(m,n)>=128  : {big[0]} rows / {big[1]} calls")
        print(f"  min>=256 & NN  : {gate[0]} rows / {gate[1]} calls")
        print(f"  routed native  : {nat[0]} rows / {nat[1]} calls")
        tc=collections.Counter(); tcc=collections.Counter()
        for r in sel:
            f=TR[M(r,"transA")]+TR[M(r,"transB")]; tc[f]+=1; tcc[f]+=int(g(r,"calls") or 0)
        print("  transpose forms rows/calls:", {k2:(v,tcc[k2]) for k2,v in tc.most_common()})
        kc=collections.Counter()
        for r in sel: kc[M(r,"k")] += int(g(r,"calls") or 0)
        print("  k histogram (calls):", dict(sorted(kc.items())))
        top=sorted(sel,key=lambda r:-int(g(r,"calls") or 0))[:15]
        print("  top shapes by calls:")
        for r in top:
            print(f"    m={M(r,'m'):5d} n={M(r,'n'):5d} k={M(r,'k'):4d} b={M(r,'batch'):5d} "
                  f"{TR[M(r,'transA')]}{TR[M(r,'transB')]} calls={g(r,'calls'):>6} -> {g(r,'chosen_origin')}/{g(r,'chosen_algo')}")

main(*sys.argv[1:])
