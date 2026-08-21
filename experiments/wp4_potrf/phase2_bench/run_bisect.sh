#!/usr/bin/env bash
# Which injected kernel makes the vendor-free blocked potrf report info != 0?
# Four route configurations x repeats, because the failure is NON-DETERMINISTIC
# (8, 12, 15 and 17 failing items out of 128 on four runs of one command) and a
# single clean run therefore proves nothing.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
BIN=${BIN:-./bench}
OUT=${OUT:-$D/bisect.csv}
REP=${REP:-5}
echo "cfg,rep,variant,type,n,batch,nb,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
for t in float double cfloat cdouble; do
for n in 256 512 1024; do
for cfg in vv nv vn nn; do
  unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
  case $cfg in
    nv) export BATCHLAS_GEMM_ROUTE=native ;;
    vn) export BATCHLAS_TRSM_ROUTE=native ;;
    nn) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native ;;
  esac
  for r in $(seq 1 $REP); do
    CUDA_VISIBLE_DEVICES=$GPU BENCH_WARM_S=0.2 $BIN ab "$t" "$n" 128 2 2>>"$D/bisect.err" \
      | sed "s/^/$cfg,$r,/" >> "$OUT"
  done
done
done
done
unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
echo "wrote $OUT"
