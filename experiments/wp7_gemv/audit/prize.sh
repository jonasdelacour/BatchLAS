#!/usr/bin/env bash
# WP7 AUDIT, items 3 and 4 -- THE ONE PRIZE, and ConjTrans.
#
# The recon phase found exactly one region where cuBLAS is off the DRAM roof:
# complex<double>, transposed, m in [64, 320], n >= 128, A larger than L2, where
# it reads 310-380 GB/s against a ~950 GB/s roof. This sweeps the region the
# lead specified -- m across 11 values x n in {128,256,512} x batch in
# {128,256,512} -- for BOTH transposed spellings.
#
# ConjTrans (D2) had never been measured in this tree and is the LIVE path:
# ortho.cc selects it for all four complex types. Running it as a full peer of
# Trans rather than as a spot check is what decides whether a preferred() clause
# may say `transA != NoTrans` or must say `transA == Trans` only.
#
# The footprint column is recorded so that residency is VISIBLE rather than
# assumed. B4 forbids an L2-residency term in preferred(); the way to keep that
# honest is to have the number in the CSV and check whether it predicts anything.
set -uo pipefail
# THIS BOX HAS TWO RTX 4090s AND ANOTHER AGENT WAS MEASURED ON DEVICE 0 while
# the first pass of this sweep was running (syr2k_tests, 470 MB, 31% util),
# which produced sustained 2.2 ms rows at a 0.5 MB shape. So the audit runs on
# device 1 -- identical part, identical max SM and memory clocks, identical
# power limit -- and every row carries the number of FOREIGN compute processes
# seen on that device, so a contaminated row can be identified rather than
# averaged in. GPU=0 restores the campaign default when device 0 is free.
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
foreign () {  # compute processes on the target device other than this harness
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc "gemvab_v" | head -1
}
export OPENBLAS_CORETYPE=SKYLAKEX
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/audit"
BIN="${BIN:-$W/experiments/wp7_gemv/ab/gemvab_v}"
OUT="${OUT:-$D/prize_p1.csv}"
REPS="${REPS:-11}"
TY="${TY:-cdouble}"

MS="32 48 64 80 128 192 256 320 384 448 512"
NS="128 256 512"
BS="128 256 512"

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld,MB,foreign" > "$OUT"
for tr in T C; do
  for m in $MS; do
    for n in $NS; do
      for b in $BS; do
        mb=$(( m * n * b * 16 / 1048576 ))
        for arm in vendor native:cta; do
          f0=$(foreign)
          row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" "$TY" "$m" "$n" "$b" "$tr" "$REPS" 2>>"$D/prize_err.txt")
          f1=$(foreign); fc=$(( f0 > f1 ? f0 : f1 ))
          if [ -z "$row" ]; then
            echo "$arm,$TY,$m,$n,$b,$tr,FAILED,,,,,,,,$mb,$fc" >> "$OUT"
          else
            echo "$arm,$row,$mb,$fc" >> "$OUT"
          fi
        done
      done
    done
  done
done
echo "wrote $OUT"
