#!/usr/bin/env bash
# WP7 AUDIT -- the shapes ortho.cc ACTUALLY issues, including the corner the
# main parity ladder does not reach.
#
# ortho.cc:227-232, transA = NoTrans branch, iterating i = 1 .. k-1:
#
#   call 1   gemv(A_i, A_next, C,  transA = inv_trans)   out_len = i,  red_len = m
#   call 2   gemv(A_i, C, A_next,  transA = NoTrans)     out_len = m,  red_len = i
#
# The parity ladder covers call 1 well (it is the transposed CTA arm at short
# output length, which is healthy) and covers call 2 only down to red_len = 64.
# But `i` STARTS AT 1. So call 2's real regime is a LARGE output against a
# reduction of 1..32 -- one to thirty-two flops per output -- and nothing in this
# audit or in ../ab/ had measured it. That corner is measured here.
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/audit"
BIN="${BIN:-$W/experiments/wp7_gemv/ab/gemvab_v}"
OUT="${OUT:-$D/ortho_shapes.csv}"
REPS="${REPS:-11}"

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld,out_len,red_len,call" > "$OUT"
emit () { # type out red batch tr call
  local ty=$1 ol=$2 rl=$3 b=$4 tr=$5 call=$6 m n dflt
  if [ "$tr" = "N" ]; then m=$ol; n=$rl; else m=$rl; n=$ol; fi
  dflt=$([ "$tr" = "N" ] && echo native:direct || echo native:cta)
  for arm in vendor "$dflt"; do
    r=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" "$ty" "$m" "$n" "$b" "$tr" "$REPS" 2>/dev/null)
    [ -n "$r" ] && echo "$arm,$r,$ol,$rl,$call" >> "$OUT"
  done
}

for ty in float cdouble; do
  for m in 512 2048; do
    for i in 1 2 4 8 16 32 64; do
      emit "$ty" "$m" "$i" 512 N call2      # y(m) = A(m,i) * c(i)
      emit "$ty" "$i" "$m" 512 C call1      # c(i) = A(m,i)^H * y(m)
    done
  done
done

python3 - "$OUT" <<'PY'
import csv,sys,collections
by=collections.defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    if r["route"]=="FAILED": continue
    by[(r["call"],r["type"],int(r["out_len"]),int(r["red_len"]))][r["arm"]]=(float(r["median_ms"]),float(r["GBs"]),r["route"])
print("\n%-6s %-8s %6s %6s %12s %12s %-14s %7s"%("call","type","out","red","vendor GB/s","native GB/s","route","ratio"))
for k in sorted(by):
    v=by[k]; d=[a for a in v if a!="vendor"]
    if not d or "vendor" not in v: continue
    n=v[d[0]]
    print("%-6s %-8s %6d %6d %12.1f %12.1f %-14s %7.2f"%(
        k[0],k[1],k[2],k[3],v["vendor"][1],n[1],n[2],v["vendor"][0]/n[0]))
PY
