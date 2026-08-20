#!/usr/bin/env python3
import csv, sys, collections, math
rows=list(csv.DictReader(open(sys.argv[1])))
bad=[r for r in rows if float(r["rel_sd"])>0.10]
if bad:
    print(f"REJECTED (rel sd > 10%): {len(bad)} cells")
    for r in bad: print("   ", r["route"],r["tag"],r["type"],r["beta"],r["rel_sd"])
key=lambda r:(r["tag"],r["type"],r["beta"],r["padA"])
d=collections.defaultdict(dict)
for r in rows:
    if float(r["rel_sd"])>0.10: continue
    d[key(r)][r["route"]]=r
print(f"{'tag':<20}{'type':<9}{'shape':<22}{'beta':<5}{'pad':<5}{'vendor GF':>10}{'native GF':>10}{'native/vendor':>15}")
ratios=collections.defaultdict(list)
out=[]
for k,v in sorted(d.items()):
    if "vendor" not in v or "native" not in v: continue
    ve,na=v["vendor"],v["native"]
    ratio=float(na["gflops"])/float(ve["gflops"])
    shape=f"{ve['m']}x{ve['n']}x{ve['k']} b{ve['batch']} {ve['tA']}{ve['tB']}"
    print(f"{k[0]:<20}{k[1]:<9}{shape:<22}{k[2]:<5}{k[3]:<5}{float(ve['gflops']):>10.0f}{float(na['gflops']):>10.0f}{ratio:>14.3f}x")
    ratios[k[1]].append(ratio); ratios["all"].append(ratio)
    ratios[(k[1],"transposed" if not (ve['tA']=='N' and ve['tB']=='N') else "NN")].append(ratio)
    out.append([k[0],k[1],shape,k[2],k[3],ve['gflops'],na['gflops'],f"{ratio:.4f}",ve['median_ms'],na['median_ms']])
print()
for k,v in ratios.items():
    g=math.exp(sum(math.log(x) for x in v)/len(v))
    print(f"  geomean native/vendor [{k}] over {len(v)} cells: {g:.3f}x   (min {min(v):.3f}x, max {max(v):.3f}x)")
if len(sys.argv)>2:
    with open(sys.argv[2],"w",newline="") as fh:
        w=csv.writer(fh)
        w.writerow("tag type shape beta ld_pad vendor_gflops native_gflops native_over_vendor vendor_ms native_ms".split())
        for r in out: w.writerow(r)
