#!/usr/bin/env python3
"""Pull the four counters that answer the sector-vs-line question out of ncu's
raw CSV, which is ~250 columns wide. Read BY HEADER NAME, never by position."""
import sys, csv

arm, t, n, b = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = [r for r in csv.reader(sys.stdin) if r]
# ncu prepends progress/warning lines that are not CSV; FIND the header row
# rather than assuming it is the first, and drop the units row that follows it.
h = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
if h is None or len(rows) < h + 3:
    print("%s %s n=%s b=%s : NO ROWS (kernel filter matched nothing?)" % (arm, t, n, b))
    sys.exit(0)
idx = {name: i for i, name in enumerate(rows[h])}
rows = rows[h:]


def num(r, key):
    return float(r[idx[key]].replace(",", ""))


print("=== arm=%s %s n=%s batch=%s ===" % (arm, t, n, b))
print("kernel,grid,block,ld_sectors,st_sectors,dram_bytes,ns,dram_B_per_sector")
for r in rows[2:]:
    k = r[idx["Kernel Name"]]
    k = "gather" if "Gather" in k else "walk"
    ld = num(r, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    st = num(r, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
    dr = num(r, "dram__bytes.sum")
    ns = num(r, "gpu__time_duration.sum")
    tot = ld + st
    print("%s,%s,%s,%d,%d,%d,%d,%.2f" %
          (k, r[idx["Grid Size"]].replace(",", " "), r[idx["Block Size"]].replace(",", " "),
           ld, st, dr, ns, (dr / tot) if tot else 0.0))
