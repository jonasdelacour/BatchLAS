#!/usr/bin/env bash
# WP7 AUDIT, item 1 -- THE PARITY GATE.
#
# WHAT IS DIFFERENT FROM experiments/wp7_gemv/ab/run.sh, and why this file
# exists at all: that sweep is a SQUARE ladder plus two aspect extremes, and it
# defines its cells in (m, n). This one defines every cell in
# (out_len, red_len) and maps it onto (m, n) per transA:
#
#     NoTrans          m = out_len   n = red_len
#     Trans/ConjTrans  m = red_len   n = out_len
#
# so one row of the ladder is the SAME amount of work and the SAME reduction
# length under all three transA, and a "skinny" cell stays skinny when the
# operation is transposed instead of silently becoming its own opposite. It
# also covers the regime ortho.cc actually issues -- an output length of 1 to
# ~1024 against a reduction length of 64 to 2048 -- which the square ladder
# never enters.
#
# METHOD (campaign rules): one DEDICATED RTX 4090 (see the GPU note below); arms
# INTERLEAVED within one cell so a clock drift has to hit all of them; every arm
# pinned EXPLICITLY (a bare `native` resolves to the first supported native
# route, which is CTA -- campaign trap 3); the RESOLVED ROUTE printed as a
# column on every row (a kernel being linked is not evidence it ran); 11 reps,
# median; a host correctness check over items 0 and batch-1 in the same process,
# so a fast wrong answer cannot enter the record.
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
OUT="${OUT:-$D/parity_p1.csv}"
REPS="${REPS:-11}"

# out_len red_len batch. Batch is held in [128, 512] as the task specifies and
# chosen so that A stays under ~2.2 GB.
CELLS="
1 64 512
1 512 512
1 2048 512
4 1024 512
16 512 512
16 2048 512
64 64 512
64 512 512
64 2048 512
128 128 512
256 256 512
256 1024 512
512 512 256
1024 128 512
1024 1024 128
2048 64 512
"

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld,out_len,red_len,foreign" > "$OUT"
while read -r ol rl b; do
  [ -z "$ol" ] && continue
  for tr in N T C; do
    if [ "$tr" = "N" ]; then m=$ol; n=$rl; else m=$rl; n=$ol; fi
    for ty in float double cfloat cdouble; do
      for arm in vendor native:direct native:cta; do
        # There is no NoTrans CTA body; supports() refuses it and the row would
        # be the vendor row wearing a native label.
        if [ "$tr" = "N" ] && [ "$arm" = "native:cta" ]; then continue; fi
        f0=$(foreign)
        row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" "$ty" "$m" "$n" "$b" "$tr" "$REPS" 2>>"$D/parity_err.txt")
        f1=$(foreign); fc=$(( f0 > f1 ? f0 : f1 ))
        if [ -z "$row" ]; then
          echo "$arm,$ty,$m,$n,$b,$tr,FAILED,,,,,,,,$ol,$rl,$fc" >> "$OUT"
        else
          echo "$arm,$row,$ol,$rl,$fc" >> "$OUT"
        fi
      done
    done
  done
done <<< "$CELLS"
echo "wrote $OUT"
