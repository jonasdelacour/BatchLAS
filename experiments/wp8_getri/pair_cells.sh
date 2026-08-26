#!/usr/bin/env bash
# PAIRED LU runner for the WP8 routing pass.
#
# GATE-B asks for the two arms interleaved WITHIN one session. For a
# vendor-vs-native LU comparison that is impossible: the arms are two BUILDS
# (lubench6_nv has no cuBLAS linked at all) and cannot share a process. D4 names
# the substitute -- experiments/wp6_perf/regcap/run.sh's cross-pass template --
# and this script tightens it one notch further: instead of two SWEEPS minutes
# apart, the two arms are run BACK TO BACK ON EACH CELL, so clock state, thermal
# state and any foreign process drift the same way for both arms of a ratio.
#
# Neither arm is pinned. That is deliberate and is how sat_native/sat_vendor were
# produced: with preferred() all-false the vendor-present binary resolves
# vendor:auto and the vendor-free binary resolves the native tier, so the BINARY
# selects the arm and no pin can silently fall through (route_resolve.hh:165).
# The resolved route is printed on every row and analyse.py REFUSES any row whose
# route does not match its arm.
#
# !! THAT REASONING IS NOW DEAD, AS OF THIS SAME PASS. It held only while
# preferred() was all-false. getri, getrf, getrs and gemv now all ship windows,
# so the vendor-PRESENT binary resolves NATIVE on every admitted cell and both
# arms of the ratio become the same kernel -- every admitted ratio silently
# collapses toward 1.00. analyse.py's route check is what saves you: it will
# REFUSE the rows rather than print a wrong number. But do not read a thin
# result from this script as a measurement; PIN BOTH ARMS EXPLICITLY
# (BATCHLAS_GETRI_ROUTE=vendor:auto / native:blocked -- never a bare `native`)
# before reusing it. Left in place, with the trap named, because the CSVs it
# already produced are cited by route_getri.hh and were taken when the premise
# was still true.
#
# usage: OUT=x.csv CELLFILE=cells.txt GPU=1 REPS=11 pair_cells.sh
set -u
D="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:?OUT=out.csv}"
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export WARM_S="${WARM_S:-1.0}"
export NPROBE="${NPROBE:-1}"
export NTRANS="${NTRANS:-1}"
REPS="${REPS:-11}"

unset BATCHLAS_GETRF_ROUTE BATCHLAS_GETRS_ROUTE BATCHLAS_GETRI_ROUTE
unset BATCHLAS_GETRS_LASWP BATCHLAS_GETRF_LASWP

UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
foreign () {
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc lubench6 || true
}

CELLS="$(grep -v '^#' "${CELLFILE:?CELLFILE=cells.txt}" | grep -v '^[[:space:]]*$')"

: > "$OUT"
echo "arm,op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws,route,e1,e2,e3,e4,flag,foreign" >> "$OUT"
for c in $CELLS; do
  IFS=: read -r op t n nrhs b <<< "$c"
  for arm in nv v; do
    f0=$(foreign)
    row=$(timeout "${TMO:-3000}" "$D/lubench6_$arm" "$op" "$t" "$n" "$nrhs" "$b" "$REPS" \
            2>>"${OUT%.csv}_err.txt") \
      || row="$op,$t,$n,$nrhs,$b,TIMEOUT_OR_THROW,-,-,-,-,-,-,-,-,-,-,BAD"
    f1=$(foreign)
    fc=$(( f0 > f1 ? f0 : f1 ))
    echo "$arm,$row,$fc" >> "$OUT"
  done
done
echo "wrote $OUT"
