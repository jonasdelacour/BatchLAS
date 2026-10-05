#!/usr/bin/env bash
# One gemm_steady_benchmark CSV per pinned gemm choice. Pins are BATCHLAS_GEMM_ROUTE spellings
# (src/ops/gemm/choice.hh): `vendor`, `native` (the best native entry of the table), or a
# family spelling. A spelling whose form is not instantiated throws instead of timing Auto.
# The P3.4-deleted s2_u2 and persistent cases are gone with their kernels.

build_dir=${1:-build}
output_prefix=${2:-output/gemm_steady_phase1_cuda}

backend=${BATCHLAS_BENCH_BACKEND:-CUDA}
scalar_type=${BATCHLAS_BENCH_TYPE:-float}
warmup=${BATCHLAS_BENCH_WARMUP:-5}
min_iters=${BATCHLAS_BENCH_MIN_ITERS:-3}
max_iters=${BATCHLAS_BENCH_MAX_ITERS:-6}
min_time=${BATCHLAS_BENCH_MIN_TIME:-0}

bench="$build_dir/benchmarks/gemm_steady_benchmark"

if [ ! -x "$bench" ]; then
    echo "missing benchmark binary: $bench" >&2
    exit 1
fi

output_dir=$(dirname "$output_prefix")
mkdir -p "$output_dir"

# run_case LABEL ROUTE [m n k batch]
run_case() {
    label=$1
    route=$2
    shift 2

    csv_path="${output_prefix}_${label}.csv"
    txt_path="${output_prefix}_${label}.txt"

    echo
    echo "=== $label ==="
    echo "csv=$csv_path"
    echo "BATCHLAS_GEMM_ROUTE=$route TERM=dumb $bench --backend=$backend --type=$scalar_type --warmup=$warmup --min_iters=$min_iters --max_iters=$max_iters --min_time=$min_time --csv=$csv_path $*"

    BATCHLAS_GEMM_ROUTE="$route" TERM=dumb "$bench" \
        --backend="$backend" \
        --type="$scalar_type" \
        --warmup="$warmup" \
        --min_iters="$min_iters" \
        --max_iters="$max_iters" \
        --min_time="$min_time" \
        --csv="$csv_path" "$@" | tee "$txt_path"
}

echo "backend=$backend type=$scalar_type build_dir=$build_dir output_prefix=$output_prefix"

run_case vendor vendor
run_case native_default native
run_case reg_128x32x32 "reg:m=128:n=32:k=32:u=1"
run_case reg_128x64x32_u4 "reg:m=128:n=64:k=32:u=4"
