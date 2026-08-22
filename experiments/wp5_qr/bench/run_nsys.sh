#!/usr/bin/env bash
# WHERE THE TIME GOES. nsys on the VENDOR-FREE binary, one float cell and one
# complex<double> cell, geqrf and orgqr.
#
# The complex<double> cell is the one that decides the standing PREDICTION on
# record (baseline finding G1): that complex loses because the transposed panel
# GEMM W = V^H A22 short-circuits to Tiled16 for every scalar type
# (gemm_kernels.cc:470-482), and that G3, the NN update, is at parity. The
# kernel-name column of cuda_gpu_kern_sum names the variant, so this confirms or
# refutes it by OBSERVATION rather than by reading the router.
#
# WARM_S is cut to 0.2 s and reps to 2, because the profile is a SPLIT and not a
# timing: nsys inflates wall time and these captures must never be quoted as
# performance numbers. The timing tables come from run_order.sh, unprofiled.
#
# Captures (*.nsys-rep, *.sqlite) are NOT committed -- see .gitignore here. Only
# the derived *_kern.txt tables are.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/bench"
N="$D/nsys"
mkdir -p "$N" "$D/kernsum"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=0.2

cap() {   # tag op type m n batch
  local tag=$1; shift
  /usr/local/cuda-13.2/bin/nsys profile --trace=cuda --force-overwrite=true \
      -o "$N/$tag" "$D/qrbench_nv" "$@" 2 > "$N/$tag.run" 2>&1
  /usr/local/cuda-13.2/bin/nsys stats --report cuda_gpu_kern_sum --format table \
      "$N/$tag.nsys-rep" > "$D/kernsum/${tag}_kern.txt" 2>&1
  echo "== $tag"; sed -n '1,40p' "$D/kernsum/${tag}_kern.txt"
}

cap geqrf_float_1024   geqrf float   1024 1024 128
cap geqrf_cdouble_1024 geqrf cdouble 1024 1024 128
cap geqrf_float_64     geqrf float   64   64   8192
cap geqrf_cdouble_64   geqrf cdouble 64   64   8192
cap orgqr_float_1024   orgqr float   1024 1024 128
cap orgqr_cdouble_1024 orgqr cdouble 1024 1024 128
