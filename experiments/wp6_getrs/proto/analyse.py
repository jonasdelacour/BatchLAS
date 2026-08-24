#!/usr/bin/env python3
# Ratio tables from the interleaved grid. Cells whose relative sd exceeds 10% are
# NAMED rather than silently averaged in.
import sys, collections, math
rows=[]
for L in open(sys.argv[1]):
    f=L.strip().split(',')
    if len(f)<16 or f[0]=='skip': 
        if f and f[0]=='skip': rows.append(('skip',f[1],int(f[2]),int(f[3]),int(f[4]),None,None,None,None))
        continue
    rows.append((f[0],f[1],int(f[2]),int(f[3]),int(f[4]),float(f[8]),float(f[10]),float(f[11]),float(f[13])))
cells=collections.defaultdict(dict)
for a,t,n,r,b,med,relsd,gbs,res in rows:
    cells[(t,n,r,b)][a]=(med,relsd,gbs,res)
noisy=[]
print("type      n    nrhs batch  vendor_ms  comp_ms   fblk_ms  fstr_ms | fblk/vendor fblk/comp fstr/fblk | GB/s  %peak resid")
geo=collections.defaultdict(list)
for k in sorted(cells, key=lambda k:(k[0],k[2],k[3])):
    t,n,r,b=k; c=cells[k]
    if 'fblock' not in c:
        print(f"{t:9s} {n:4d} {r:4d} {b:6d}  -- fused arm SKIPPED: RHS does not fit local memory --")
        continue
    v=c.get('vendor'); cp=c.get('comp'); fb=c['fblock']; fs=c.get('fstream')
    for nm,val in c.items():
        if val[1]>0.10: noisy.append((nm,k,val[1]))
    rv=v[0]/fb[0] if v else 0; rc=cp[0]/fb[0] if cp else 0; rs=(fs[0]/fb[0] if fs else 0)
    geo[('vendor',t)].append(rv); geo[('comp',t)].append(rc)
    geo[('vendor_all','')].append(rv); geo[('comp_all','')].append(rc)
    if r==1: geo[('vendor_r1','')].append(rv); geo[('comp_r1','')].append(rc)
    print(f"{t:9s} {n:4d} {r:4d} {b:6d}  {v[0]:9.4f} {cp[0]:9.4f} {fb[0]:8.4f} {(fs[0] if fs else 0):8.4f} |"
          f" {rv:10.3f} {rc:9.3f} {rs:9.3f} | {fb[2]:6.1f} {100*fb[2]/1008:5.1f} {fb[3]:.1e}")
def gm(v):
    v=[x for x in v if x>0]
    return math.exp(sum(math.log(x) for x in v)/len(v)) if v else 0
print()
for k in sorted(geo):
    print(f"geomean {k[0]:12s} {k[1]:9s} : {gm(geo[k]):.3f}  over {len(geo[k])} cells")
print()
if noisy:
    print("RELATIVE SD > 10%:", noisy)
else:
    print("no cell exceeded 10% relative sd")
