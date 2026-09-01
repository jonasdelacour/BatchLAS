#!/usr/bin/env python3
"""How close is each arm to the 1008 GB/s DRAM roof, on rows where DRAM is
actually the medium?

THE L2 TRAP. A 4090 has 72 MB of L2. `analyse.py` flags a row whose whole-batch
footprint fits inside it, but "just over 72 MB" is not DRAM-resident either --
partial residency inflates the effective bandwidth well past the pin (the pass-1
table has cfloat rows reading 3.1x the roof at an 87 MB footprint). So the roof
comparison here is restricted to rows whose footprint exceeds 4x L2 = 288 MB,
where whatever L2 holds is a small fraction of the traffic.
"""
import csv, glob, sys, collections

D = '/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/sparse_spmm'
ROOF = 1008.0
MIN_FP_MB = 288.0


def main(passes):
    rows = []
    for p in passes:
        for r in csv.DictReader(open(f'{D}/{p}/joined.csv')):
            if float(r['fp_mb']) < MIN_FP_MB or r['agree'] != 'True':
                continue
            rows.append((p, r))
    print(f"rows with footprint > {MIN_FP_MB:.0f} MB (4x the 72 MB L2): {len(rows)}")
    for ta in (0, 1):
        for typ in ('float', 'double', 'cfloat', 'cdouble'):
            st = [r for _, r in rows if r['typ'] == typ and int(r['transA']) == ta]
            if not st:
                continue
            v = [float(r['gbs_v']) for r in st]
            n = [float(r['gbs_n']) for r in st]
            print(f"  transA={ta} {typ:8s} n={len(st):3d}  vendor "
                  f"{min(v):6.1f}-{max(v):6.1f} GB/s ({min(v)/ROOF:.2f}-{max(v)/ROOF:.2f} "
                  f"x roof)   native {min(n):6.1f}-{max(n):6.1f} GB/s "
                  f"({min(n)/ROOF:.2f}-{max(n)/ROOF:.2f} x roof)")
    print()
    print("The lanczos shape at unambiguous DRAM residency (sat1/sat2, m=1024, "
          "3 nnz/row, batch 4096 and 8192, footprint 151-604 MB):")
    for p in ('sat1', 'sat2'):
        for r in csv.DictReader(open(f'{D}/{p}/joined.csv')):
            if int(r['batch']) < 4096:
                continue
            print(f"  {p} {r['typ']:8s} nrhs={r['nrhs']} pat={r['pattern']} "
                  f"b={r['batch']:5s} fp={float(r['fp_mb']):7.1f}MB  "
                  f"vendor {float(r['gbs_v']):6.1f} GB/s ({float(r['gbs_v'])/ROOF:.2f} x roof)  "
                  f"native {float(r['gbs_n']):6.1f} GB/s ({float(r['gbs_n'])/ROOF:.2f} x roof)  "
                  f"ratio {float(r['ratio']):.3f}")


if __name__ == '__main__':
    main(sys.argv[1:] or ['pass1', 'pass2'])
