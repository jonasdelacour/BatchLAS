#!/usr/bin/env bash
# WP7 AUDIT -- is the cuBLAS dip TYPE-EXCLUSIVE at the prize shapes?
#
# This matters for preferred(): a clause written on (m, n, batch) alone would
# fire for all four scalar types. The recon phase claimed the dip is
# complex<double> only, at IDENTICAL bytes -- checked here at three shapes drawn
# from the middle of the winning region, with the byte count held constant per
# type by keeping (m, n, batch) fixed and letting the footprint vary with the
# scalar width, and then again at MATCHED BYTES by scaling batch.
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/audit"
BIN="${BIN:-$W/experiments/wp7_gemv/ab/gemvab_v}"
OUT="${OUT:-$D/typecheck.csv}"
REPS="${REPS:-11}"

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld,note" > "$OUT"
row () { # type m n batch tr note
  for arm in vendor native:cta; do
    r=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" "$1" "$2" "$3" "$4" "$5" "$REPS" 2>/dev/null)
    [ -n "$r" ] && echo "$arm,$r,$6" >> "$OUT"
  done
}

# SAME (m, n, batch): the scalar width is the only thing that moves.
for ty in float double cfloat cdouble; do
  row "$ty" 256 256 512 T same_shape
  row "$ty" 288 384 384 T same_shape
  row "$ty" 160 1024 384 T same_shape
done
# MATCHED BYTES: batch scaled by 16/sizeof(T) so every type reads ~1 GB.
row float   256 256 8192 T matched_bytes
row double  256 256 4096 T matched_bytes
row cfloat  256 256 4096 T matched_bytes
row cdouble 256 256 2048 T matched_bytes

python3 - "$OUT" <<'PY'
import csv,sys,collections
by=collections.defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    by[(r["note"],r["type"],r["m"],r["n"],r["batch"])][r["arm"]]=(float(r["median_ms"]),float(r["GBs"]))
print("\n%-14s %-8s %5s %5s %6s %12s %12s %7s"%("case","type","m","n","batch","vendor GB/s","native GB/s","ratio"))
for k in sorted(by):
    v=by[k]
    if "vendor" not in v or "native:cta" not in v: continue
    print("%-14s %-8s %5s %5s %6s %12.1f %12.1f %7.2f"%(
        k[0],k[1],k[2],k[3],k[4],v["vendor"][1],v["native:cta"][1],
        v["vendor"][0]/v["native:cta"][0]))
PY
