#!/usr/bin/env bash
# B5 AUDIT -- the launch geometry of the three native gemv bodies, read from the
# HARDWARE PROFILER rather than from the source.
#
# The defect B5 names is "batch-only parallelism": an nd_range<2> of
# global={batch, ceil(out/wg)*wg}, local={1,wg} produces EXACTLY `batch`
# work-groups whenever out_len <= wg. This script reports the ACTUAL grid size
# and block size ncu observes, so the claim that the extent was flattened is
# checked against the launch, not against the comment above it.
#
# It also reports static and dynamic shared memory per block. Zero on both is
# the "no local memory" property that makes the recorded 48 KB launch hole
# structurally unreachable -- verified here by the profiler, independently of
# any grep over the source.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
B=$W/experiments/wp7_gemv/ab/gemvab_v
NCU=/usr/local/cuda-13.2/bin/ncu
OUT="${OUT:-$W/experiments/wp7_gemv/audit/geometry.csv}"
export WARM_S=0.01

echo "arm,type,m,n,batch,transA,kernel,grid,block,smem_static,smem_dynamic" > "$OUT"

probe () {  # arm type m n batch tr
  local arm=$1 ty=$2 m=$3 n=$4 b=$5 tr=$6
  BATCHLAS_GEMV_ROUTE="$arm" "$NCU" --csv --print-summary per-kernel \
    --metrics launch__grid_size,launch__block_size,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic \
    --launch-count 2 --launch-skip 3 \
    "$B" "$ty" "$m" "$n" "$b" "$tr" 2 2>/dev/null \
  | ARM="$arm" TY="$ty" M="$m" N="$n" BB="$b" TR="$tr" python3 -c "
import sys,csv,os
rows=[r for r in csv.reader(sys.stdin) if len(r)>14 and r[0].isdigit()]
acc={}
for r in rows:
    k=r[3]
    if 'Gemv' not in k: continue
    acc.setdefault(k,{})
    acc[k]['grid_raw']=r[5]; acc[k]['block_raw']=r[4]
    acc[k][r[10]]=r[14]
e=os.environ
for k,v in acc.items():
    kn=k.split('<')[0].split('::')[-1]
    print(','.join([e['ARM'],e['TY'],e['M'],e['N'],e['BB'],e['TR'],kn,
        v.get('launch__grid_size','?'),v.get('launch__block_size','?'),
        v.get('launch__shared_mem_per_block_static','?'),
        v.get('launch__shared_mem_per_block_dynamic','?')]))
" >> "$OUT"
}

# THE REQUIRED PROOF: Body 1 at m = 64, batch = 128 on a 128-SM box.
probe native:direct cdouble 64 128 128 N
probe native:direct float   64 128 128 N
# Batch BELOW the SM count -- where a batch-only extent would be most starved.
probe native:direct cdouble 64 128  32 N
probe native:direct cdouble 64 128  16 N
# Output length below the work-group size, the exact trigger condition.
probe native:direct cdouble 16 512 128 N
probe native:direct cdouble  1 512 128 N
# Long output: the ladder should climb back to a wide work-group.
probe native:direct cdouble 4096 128 128 N
# Body 2 (Direct on a transposed shape) and Body 3 (CTA), same trigger shapes.
probe native:direct cdouble 128 64 128 T
probe native:direct cdouble 128 64  32 T
probe native:cta    cdouble 128 64 128 T
probe native:cta    cdouble 128 64  32 T
probe native:cta    cdouble 256 256 1024 T
probe native:cta    cdouble 256 4096 128 C
cat "$OUT"
