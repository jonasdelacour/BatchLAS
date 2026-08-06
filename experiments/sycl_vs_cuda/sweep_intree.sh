#!/usr/bin/env bash
# Sweep the in-tree GEMM routes over the shapes the campaign notes use, so the
# new 128x128x8 kernel can be compared against the incumbent and the vendor
# under one timing regime. Runs strictly serially: concurrent GPU work makes
# every number here meaningless.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/build/benchmarks/gemm_steady_benchmark"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SHAPES=(
    "128 128 128 4096"
    "256 256 256 1024"
    "512 512 512 512"
    "512 256 512 512"
    "512 64 512 512"
    "1024 1024 1024 64"
)

run_one() {
    local label="$1"; shift
    local shape="$1"; shift
    # shellcheck disable=SC2086
    "$BIN" --backend=CUDA --type=float --name=BM_GEMM_STEADY_EVENT \
        --warmup=5 --min_iters=10 --max_iters=20 --min_time=0 $shape 2>/dev/null \
        | awk '/BM_GEMM_STEADY_EVENT/ {print $(NF-1)}'
}

printf '%-22s %12s %12s %12s %10s\n' shape vendor 128x64x32 128x128x8 "new/vendor"
for shape in "${SHAPES[@]}"; do
    v=$(BATCHLAS_GEMM_VARIANT=vendor run_one vendor "$shape")
    o=$(BATCHLAS_GEMM_VARIANT=sycl BATCHLAS_GEMM_SYCL_KERNEL=128x64x32large run_one old "$shape")
    n=$(BATCHLAS_GEMM_VARIANT=sycl BATCHLAS_GEMM_SYCL_KERNEL=128x128x8 run_one new "$shape")
    pct=$(awk -v a="$n" -v b="$v" 'BEGIN{ if (b+0>0) printf "%.1f%%", 100*a/b; else print "n/a" }')
    printf '%-22s %12s %12s %12s %10s\n' "${shape// /x}" "$v" "$o" "$n" "$pct"
done
