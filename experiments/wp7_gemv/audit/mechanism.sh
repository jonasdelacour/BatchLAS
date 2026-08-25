#!/usr/bin/env bash
# WP7 AUDIT -- the MECHANISM behind the sub-0.50x cells, measured rather than
# argued.
#
# The parity sweep found 15 cells below the lead's 0.50x blocker line, and 13 of
# them are the SAME shape family: Algorithm::Direct, transA = NoTrans, with a
# SHORT OUTPUT LENGTH (out_len <= 16). The hypothesis this script tests is that
# they are a COALESCING collapse caused by the very flattening B5 mandated:
#
#     b = gid / out_len        i = gid % out_len
#
# Consecutive work-items hold consecutive `i` -- and therefore adjacent elements
# of one column -- ONLY while they stay inside one batch item. The moment
# out_len < 32 a warp straddles batch items, whose rows are `stride_a` elements
# apart, so one 32-lane load touches up to 32 different sectors instead of 4.
#
# The prediction is sharp and falsifiable: sectors per global load request should
# be ~4 (perfect, 32 lanes x 16 B = 512 B = 4 sectors) whenever out_len >= 32,
# and should climb toward 32 as out_len falls below the warp width -- with the
# transition AT 32, not at some smooth roll-off. Achieved occupancy should
# collapse in the same place, because out_len * batch is the entire launch.
#
# The `ld` column is the control. LD pads the leading dimension without changing
# the shape, the traffic or the reference, so if the effect were an alignment
# artefact rather than a warp-straddling one, padding would move it.
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/audit"
B=$W/experiments/wp7_gemv/ab/gemvab_v
NCU=/usr/local/cuda-13.2/bin/ncu
OUT="${OUT:-$D/mechanism.csv}"
export WARM_S=0.01

M_SECT=l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
M_OCC=sm__warps_active.avg.pct_of_peak_sustained_active
M_DRAM=dram__throughput.avg.pct_of_peak_sustained_elapsed

echo "arm,type,out_len,red_len,batch,transA,ld,kernel,grid,block,sectors_per_ld,occupancy_pct,dram_pct" > "$OUT"

probe () {  # arm type out red batch tr ldpad
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

# BODY 1, NoTrans, out_len swept ACROSS the 32-lane warp width. Reduction and
# batch are held fixed so out_len is the only thing that moves.
for ol in 1 2 4 8 16 24 31 32 33 48 64 128 256; do
  probe native:direct cdouble "$ol" 2048 512 N 0
done
# The same ladder for a 4-byte scalar: the warp width is in LANES, not bytes, so
# if the transition is at out_len = 32 for both, it is the warp and not a sector.
for ol in 1 8 16 32 64 128; do
  probe native:direct float "$ol" 2048 512 N 0
done
# CONTROL: pad ld away from m at two points on the ladder. An alignment story
# predicts movement here; a warp-straddling story predicts none.
probe native:direct cdouble 16 2048 512 N 17
probe native:direct cdouble 16 2048 512 N 24
probe native:direct cdouble 64 2048 512 N 65
probe native:direct cdouble 64 2048 512 N 72

# BODY 3, the CTA arm, for known weakness 1: short REDUCTION. Here the sweep is
# on red_len, which under a transposed transA is m.
for rl in 32 48 64 96 128 192 256 512; do
  probe native:cta cdouble 256 "$rl" 512 T 0
done
cat "$OUT"
