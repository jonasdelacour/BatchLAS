#!/usr/bin/env python3
"""Turn one ncu --csv dump into the two ratios that decide the diagnosis.

  sectors / request   -- the L1 over-fetch. A fully coalesced 32-lane float
                         load moves 128 B = 4 sectors in one request. One
                         sector per lane (32) is the fully-scattered worst case,
                         so this number IS the over-fetch factor over 4.

  dram bytes / floor  -- the DRAM over-fetch. floor = 2*q*n*sizeof(T)*batch,
                         B read once and written once. Near 1.0 means the
                         scatter is being absorbed by cache and the defect is
                         NOT at the DRAM level.
"""
import csv, sys

path, n, q, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

vals = {}
with open(path, errors='ignore') as f:
    # ncu prefixes junk lines before the real header; find it.
    lines = [l for l in f if l.startswith('"')]
if not lines:
    print('   NO METRIC ROWS -- the kernel filter matched nothing')
    sys.exit(0)
for row in csv.DictReader(lines):
    name = row.get('Metric Name')
    raw = (row.get('Metric Value') or '').replace(',', '')
    if not name or not raw:
        continue
    try:
        vals[name] = float(raw)
    except ValueError:
        pass

ld_req = vals.get('l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum', 0)
ld_sec = vals.get('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', 0)
st_req = vals.get('l1tex__t_requests_pipe_lsu_mem_global_op_st.sum', 0)
st_sec = vals.get('l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum', 0)
dram = vals.get('dram__bytes_read.sum', 0) + vals.get('dram__bytes_write.sum', 0)
dur = vals.get('gpu__time_duration.sum', 0)

floor = 2.0 * q * n * 4 * batch          # float, B touched exactly twice

print(f'   load : {ld_sec/ld_req:6.2f} sectors/request   ({ld_sec/4/ld_req:5.2f}x over the coalesced 4)'
      if ld_req else '   load : n/a')
print(f'   store: {st_sec/st_req:6.2f} sectors/request   ({st_sec/4/st_req:5.2f}x over the coalesced 4)'
      if st_req else '   store: n/a')
print(f'   dram : {dram/1e6:8.2f} MB vs {floor/1e6:8.2f} MB floor  = {dram/floor:5.2f}x')
print(f'   time : {dur/1e6:8.3f} ms')
