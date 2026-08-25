#!/usr/bin/env bash
# Repair-agent mechanism probe for BODY 4 (segmented NoTrans), same metrics and
# same shapes as experiments/wp7_gemv/audit/mechanism.sh used for body 1, so the
# two are directly comparable row for row.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-1}
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
B=$W/experiments/wp7_gemv/ab/gemvab_v
NCU=/usr/local/cuda-13.2/bin/ncu
OUT=${OUT:-/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7/repair/mech4.csv}
export WARM_S=0.01
M_SECT=l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
M_OCC=sm__warps_active.avg.pct_of_peak_sustained_active
M_DRAM=dram__throughput.avg.pct_of_peak_sustained_elapsed

echo "arm,type,out_len,red_len,batch,transA,ld,kernel,grid,block,sectors_per_ld,occupancy_pct,dram_pct" > "$OUT"

probe () {
  local arm=$1 ty=$2 ol=$3 rl=$4 b=$5 tr=$6 ldp=$7
  local m n
  if [ "$tr" = "N" ]; then m=$ol; n=$rl; else m=$rl; n=$ol; fi
  LD=$ldp BATCHLAS_GEMV_ROUTE="$arm" "$NCU" --csv --print-summary per-kernel \
    --metrics "$M_SECT,$M_OCC,$M_DRAM,launch__grid_size,launch__block_size" \
    --launch-count 2 --launch-skip 3 \
    "$B" "$ty" "$m" "$n" "$b" "$tr" 2 2>/dev/null \
  | ARM="$arm" TY="$ty" OL="$ol" RL="$rl" BB="$b" TR="$tr" LDP="$ldp" python3 -c "
import sys,csv,os
rows=[r for r in csv.reader(sys.stdin) if len(r)>14 and r[0].isdigit()]
acc={}
for r in rows:
    k=r[3]
    if 'Gemv' not in k: continue
    acc.setdefault(k,{})
    acc[k]['grid']=r[5]; acc[k]['block']=r[4]
    acc[k][r[10]]=r[14]
e=os.environ
for k,v in acc.items():
    kn=k.split('<')[0].split('::')[-1]
    print(','.join([e['ARM'],e['TY'],e['OL'],e['RL'],e['BB'],e['TR'],e['LDP'],kn,
        v.get('grid','?').replace(',',''),v.get('block','?').replace(',',''),
        v.get('$M_SECT','?'),v.get('$M_OCC','?'),v.get('$M_DRAM','?')]))
" >> "$OUT"
}

for ol in 1 2 4 8 12 16; do
  probe native:direct float "$ol" 2048 512 N 0
  probe native:direct cdouble "$ol" 2048 512 N 0
done
cat "$OUT"
