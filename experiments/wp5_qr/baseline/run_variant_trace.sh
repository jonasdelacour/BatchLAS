#!/usr/bin/env bash
# The KernelVariant half of the trailing-update question, OBSERVED rather than
# reasoned: run the same G1/G3 shapes under BATCHLAS_KERNEL_TRACE and read which
# kernel actually ran. The resolver Route (routeq_qr.csv) cannot answer this --
# it records Origin/Algorithm only, and KernelVariant is chosen a layer below.
#
# NEVER TIMED under the tracer (~60% inflation) -- these runs are for names only,
# and the JSON goes to the scratch dir, never into the repo.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
S=/home/jonaslacour/.claude/jobs/20812aa0/tmp
mkdir -p "$S"
export CUDA_VISIBLE_DEVICES=1 WARM_S=0.05
echo "build,which,type,N,nb,j0,kernels"
for bld in v nv; do
  for t in float double cfloat cdouble; do
    for cell in "256 24 512" "1024 56 128"; do
      set -- $cell; N=$1; nb=$2; b=$3
      for w in G1 G3; do
        f="$S/trace_${bld}_${t}_${N}_${w}.json"
        rm -f "$f"
        BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$f" \
          "$D/gemmtrail_$bld" "$t" "$N" "$nb" 0 "$b" "$w" 1 >/dev/null 2>&1
        names="$(python3 - "$f" <<'PY'
import json,sys,collections
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("NOTRACE"); raise SystemExit
def walk(o,acc):
    if isinstance(o,dict):
        for kk in ("name","kernel","label"):
            if kk in o and isinstance(o[kk],str): acc[o[kk]]+=1
        for v in o.values(): walk(v,acc)
    elif isinstance(o,list):
        for v in o: walk(v,acc)
c=collections.Counter(); walk(d,c)
print(";".join(f"{k}x{v}" for k,v in c.most_common(6)))
PY
)"
        echo "$bld,$w,$t,$N,$nb,0,$names"
        rm -f "$f"
      done
    done
  done
done
