#!/usr/bin/env bash
# WP7 AUDIT -- a THIRD, POST-REBUILD pass over the 15 blocker cells only.
#
# WHY THIS EXISTS. Halfway through prize_p1 another agent rebuilt
# build/src/libbatchlas_sycl.so (two rows died with "invalid ELF header" at the
# moment of the swap, and src/sycl/gemv_native.cc carries a 17:24 mtime). The
# native arm's own timings agree across that boundary to a median of 1.0010 and
# a worst of 1.054 over 197 paired cells, so the swap did not change behaviour --
# but "agrees with itself" is weaker than "re-measured after the fact", and the
# blocker list is the part of this audit that a repair depends on. So the
# blockers are re-run here against whatever binary is on disk NOW.
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
export OPENBLAS_CORETYPE=SKYLAKEX
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/audit"
BIN="${BIN:-$W/experiments/wp7_gemv/ab/gemvab_v}"
OUT="${OUT:-$D/blockers_p3.csv}"
REPS="${REPS:-11}"

# type out_len red_len batch transA -- the 15 cells the two-pass gate put below
# 0.50x, in (out_len, red_len) form.
CELLS="
cfloat 1 2048 512 N
float 1 2048 512 N
double 1 2048 512 N
cfloat 1 512 512 N
cfloat 4 1024 512 N
cdouble 1 2048 512 N
double 4 1024 512 N
cdouble 4 1024 512 N
float 16 2048 512 N
cdouble 1 512 512 N
double 1 512 512 N
float 1 512 512 N
float 4 1024 512 N
cdouble 64 64 512 T
cdouble 64 64 512 C
"

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld,out_len,red_len" > "$OUT"
while read -r ty ol rl b tr; do
  [ -z "$ty" ] && continue
  if [ "$tr" = "N" ]; then m=$ol; n=$rl; else m=$rl; n=$ol; fi
  dflt=$([ "$tr" = "N" ] && echo native:direct || echo native:cta)
  for arm in vendor "$dflt"; do
    row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" "$ty" "$m" "$n" "$b" "$tr" "$REPS" 2>>"$D/parity_err.txt")
    [ -z "$row" ] && row="$ty,$m,$n,$b,$tr,FAILED,,,,,,,"
    echo "$arm,$row,$ol,$rl" >> "$OUT"
  done
done <<< "$CELLS"

python3 - "$OUT" <<'PY'
import csv,sys,collections
rows=list(csv.DictReader(open(sys.argv[1])))
by=collections.defaultdict(dict)
for r in rows:
    if r["route"]=="FAILED": continue
    by[(r["type"],r["transA"],r["out_len"],r["red_len"],r["batch"])][r["arm"]]=(float(r["median_ms"]),float(r["GBs"]),r["route"])
print("\n%-8s %-2s %6s %6s %7s %11s %11s %-14s %7s"%("type","tr","out","red","batch","vendor GB/s","native GB/s","route","ratio"))
for k,v in by.items():
    d=[a for a in v if a!="vendor"]
    if not d or "vendor" not in v: continue
    n=v[d[0]]
    print("%-8s %-2s %6s %6s %7s %11.1f %11.1f %-14s %7.2f"%(k[0],k[1],k[2],k[3],k[4],v["vendor"][1],n[1],n[2],v["vendor"][0]/n[0]))
PY
