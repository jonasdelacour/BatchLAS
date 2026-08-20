#!/usr/bin/env bash
# WP4 follow-up: separate the two effects the first sweep exposed.
#
# Sweep 1 showed the curve is NOT one thing. For the outer shape (m=128) pads
# 1/2/3 cost ~1.9x while pad 4 costs 1.05x, and the cost then climbs again from
# pad 32 to pad 384. So there is an ALIGNMENT cliff (ld % 4, the float packet
# width) sitting on top of a genuine SLOPE in the pad magnitude. This run
# separates them:
#
#   * pads 128/129/130/132 -- is the ld%4 cliff still there at a LARGE pad, i.e.
#     are the two effects independent and additive?
#   * pads 5..24 -- the inner shape (m=32) was slow at pad 4 and 8 but fine at
#     16 and 32, which the ld%4 story does not explain. Fine-grained resolution.
#   * pads 192/256/512 -- where the slope goes past the pad=384 endpoint.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu0/raw2"
mkdir -p "$OUT"
GPU="${GPU:-0}"
PADS="${PADS:-5 6 7 10 12 20 24 48 96 129 130 132 192 256 512}"

run() { # route beta pad tag m n klist batch
    local route=$1 beta=$2 pad=$3 tag=$4 m=$5 n=$6 klist=$7 batch=$8
    local f="$OUT/${tag}-${route}-b${beta}-pad${pad}"
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_BENCH_BETA="$beta" BATCHLAS_BENCH_LD_PAD="$pad" \
    GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$f.csv" "$m" "$n" "$klist" "$batch" > "$f.log" 2>&1
    echo "  $tag $route beta=$beta pad=$pad exit=$?"
}

for pad in $PADS; do
  run native 1 "$pad" outer 128 1024 128,256,128,256 512
  run native 1 "$pad" inner 32  1024 32,96,32,96     512
done
for pad in 129 132 256 512; do
  run vendor 1 "$pad" outer 128 1024 128,256,128,256 512
  run vendor 1 "$pad" inner 32  1024 32,96,32,96     512
done
echo "sweep2 complete"
