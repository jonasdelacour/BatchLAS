#!/usr/bin/env bash
# WHERE THE TIME GOES -- on a WINNER and on a LOSER, because a split taken from a
# cell the kernel wins does not explain the cell it loses.
#
# Captures are of the VENDOR-FREE binary (so the split is of the native
# composition), plus one vendor capture on the losing cell so that what cuBLAS
# does differently is an observation and not an inference.
#
# NOTHING HERE IS A TIMING NUMBER. nsys inflates wall time; WARM_S is cut to 0.2
# and reps to 2 because these runs produce a SPLIT. Every millisecond quoted in
# the README comes from the unprofiled runners.
#
# *.nsys-rep and *.sqlite are NOT committed -- see .gitignore. Only the derived
# kernsum/*_kern.txt tables are.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp6_lu/bench"
N="$D/nsys"
mkdir -p "$N" "$D/kernsum"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=0.2 NPROBE=1 NTRANS=1

cap() {   # tag binary pin op type n nrhs batch
  local tag=$1 bin=$2 pin=$3; shift 3
  if [ "$pin" = vendor ]; then
    export BATCHLAS_GETRF_ROUTE=vendor BATCHLAS_GETRS_ROUTE=vendor BATCHLAS_GETRI_ROUTE=vendor
  else
    unset BATCHLAS_GETRF_ROUTE BATCHLAS_GETRS_ROUTE BATCHLAS_GETRI_ROUTE
  fi
  /usr/local/cuda-13.2/bin/nsys profile --trace=cuda --force-overwrite=true \
      -o "$N/$tag" "$D/$bin" "$@" 2 > "$N/$tag.run" 2>&1
  /usr/local/cuda-13.2/bin/nsys stats --report cuda_gpu_kern_sum --format table \
      "$N/$tag.nsys-rep" > "$D/kernsum/${tag}_kern.txt" 2>&1
  echo "== $tag"
  grep -v '^$' "$N/$tag.run" | tail -2
  sed -n '1,28p' "$D/kernsum/${tag}_kern.txt"
  echo
}

# --- WINNERS
cap win_getrf_float_2048   lubench6_nv none getrf float   2048 1 32
cap win_getri_float_2048   lubench6_nv none getri float   2048 1 32

# --- LOSERS
cap lose_getrf_double_128  lubench6_nv none getrf double  128  1 4096
cap lose_getrf_cdouble_128 lubench6_nv none getrf cdouble 128  1 4096
cap lose_getrs_float_512   lubench6_nv none getrs float   512  1 512
cap win_getrs_float_512    lubench6_nv none getrs float   512  128 256
cap lose_getrs_cdouble_512 lubench6_nv none getrs cdouble 512  1 512
cap lose_getri_cfloat_32   lubench6_nv none getri cfloat  32   1 8192

# --- the complex prediction: does the trailing NN update land on Tiled16?
cap cx_getrf_cdouble_1024  lubench6_nv none getrf cdouble 1024 1 128
cap cx_getrf_float_1024    lubench6_nv none getrf float   1024 1 128

# --- what the vendor does on one cell we lose, for contrast
cap vend_getrf_double_128  lubench6_v vendor getrf double 128 1 4096
