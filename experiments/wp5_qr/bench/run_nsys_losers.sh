#!/usr/bin/env bash
# PROFILE THE CELLS THAT LOSE, not only the ones that win.
#
# run_nsys.sh profiled float and complex<double> at n = 1024, where BOTH win. A
# split taken at a winning cell does not explain a losing one, and saying "the
# cdouble loss at n <= 256 is the Tiled16 transposed GEMM" on the strength of an
# n = 1024 profile would be an inference dressed as a measurement. These three
# cells are losses in order.csv:
#
#   cdouble n=256  b=2048   0.84x   native:blocked  -- the largest losing blocked cell
#   cdouble n=64   b=8192   0.54x   native:cta
#   double  n=64   b=8192   0.53x   native:cta      -- and 0.92x against its OWN blocked tier
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/bench"
N="$D/nsys"
mkdir -p "$N" "$D/kernsum"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=0.2

cap() {
  local tag=$1; shift
  /usr/local/cuda-13.2/bin/nsys profile --trace=cuda --force-overwrite=true \
      -o "$N/$tag" "$D/qrbench_nv" "$@" 2 > "$N/$tag.run" 2>&1
  /usr/local/cuda-13.2/bin/nsys stats --report cuda_gpu_kern_sum --format table \
      "$N/$tag.nsys-rep" > "$D/kernsum/${tag}_kern.txt" 2>&1
  echo "== $tag"; sed -n '1,40p' "$D/kernsum/${tag}_kern.txt"
}

cap geqrf_cdouble_256 geqrf cdouble 256 256 2048
cap geqrf_double_64   geqrf double  64  64  8192
# The same cdouble n=256 cell on the blocked tier is what the route already picks
# (m*n = 65536 > cta_max_elems = 6080), so no pin is needed and none is used.
