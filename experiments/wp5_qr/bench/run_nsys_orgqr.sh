#!/usr/bin/env bash
# RE-CAPTURE the two orgqr profiles with SYNTH=1.
#
# WHY THE FIRST PAIR HAD TO BE THROWN OUT. `qrbench_nv orgqr` builds the factor
# with an UNTIMED geqrf call before it times anything, and at n = 1024 that call
# issues 32 panels x (panel + pack_v + larft + 3 GEMMs). cuda_gpu_kern_sum
# aggregates by kernel NAME, and the gemm kernels carry no tag naming their
# caller -- so `GemmTiledGeneralKernel<float,16,...>` in the first orgqr capture
# is the SUM of orgqr's applies and geqrf's trailing updates, and the split it
# implies is wrong. The larft/pack_v rows ARE separable (OrmqrWyTag vs
# GeqrfWyTag) and that is how the contamination was spotted.
#
# SYNTH=1 fabricates the reflectors on the host instead, so the process makes NO
# geqrf call and every kernel in the capture belongs to orgqr. ormqr's cost is a
# function of the shape, not of the reflector values, so the work profiled is the
# work the real call does.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/bench"
N="$D/nsys"
mkdir -p "$N" "$D/kernsum"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=0.2
export SYNTH=1

cap() {
  local tag=$1; shift
  /usr/local/cuda-13.2/bin/nsys profile --trace=cuda --force-overwrite=true \
      -o "$N/$tag" "$D/qrbench_nv" "$@" 2 > "$N/$tag.run" 2>&1
  /usr/local/cuda-13.2/bin/nsys stats --report cuda_gpu_kern_sum --format table \
      "$N/$tag.nsys-rep" > "$D/kernsum/${tag}_kern.txt" 2>&1
  echo "== $tag"; sed -n '1,40p' "$D/kernsum/${tag}_kern.txt"
}

cap orgqr_float_1024_synth   orgqr float   1024 1024 128
cap orgqr_cdouble_1024_synth orgqr cdouble 1024 1024 128
cap orgqr_float_64_synth     orgqr float   64   64   8192
