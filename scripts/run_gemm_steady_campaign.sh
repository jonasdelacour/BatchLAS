#!/usr/bin/env bash

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

run_case() {
    label=$1
    variant_mode=$2
    forced_kernel=$3
    experimental=$4

    csv_path="${output_prefix}_${label}.csv"
    txt_path="${output_prefix}_${label}.txt"

    echo
    echo "=== $label ==="
    echo "csv=$csv_path"

    cmd="TERM=dumb $bench --backend=$backend --type=$scalar_type --warmup=$warmup --min_iters=$min_iters --max_iters=$max_iters --min_time=$min_time --csv=$csv_path"
    if [ -n "$variant_mode" ]; then
        cmd="BATCHLAS_GEMM_VARIANT=$variant_mode $cmd"
    fi
    if [ -n "$forced_kernel" ]; then
        cmd="BATCHLAS_GEMM_SYCL_KERNEL=$forced_kernel $cmd"
    fi
    if [ "$experimental" = "1" ]; then
        cmd="BATCHLAS_GEMM_EXPERIMENTAL=1 $cmd"
    fi

    echo "$cmd"

    if [ -n "$variant_mode" ]; then
        export BATCHLAS_GEMM_VARIANT="$variant_mode"
    else
        unset BATCHLAS_GEMM_VARIANT
    fi

    if [ -n "$forced_kernel" ]; then
        export BATCHLAS_GEMM_SYCL_KERNEL="$forced_kernel"
    else
        unset BATCHLAS_GEMM_SYCL_KERNEL
    fi

    if [ "$experimental" = "1" ]; then
        export BATCHLAS_GEMM_EXPERIMENTAL=1
    else
        unset BATCHLAS_GEMM_EXPERIMENTAL
    fi

    TERM=dumb "$bench" \
        --backend="$backend" \
        --type="$scalar_type" \
        --warmup="$warmup" \
        --min_iters="$min_iters" \
        --max_iters="$max_iters" \
        --min_time="$min_time" \
        --csv="$csv_path" | tee "$txt_path"

    unset BATCHLAS_GEMM_VARIANT
    unset BATCHLAS_GEMM_SYCL_KERNEL
    unset BATCHLAS_GEMM_EXPERIMENTAL
}

run_case_dims() {
    label=$1
    variant_mode=$2
    forced_kernel=$3
    experimental=$4
    shift 4

    csv_path="${output_prefix}_${label}.csv"
    txt_path="${output_prefix}_${label}.txt"

    echo
    echo "=== $label ==="
    echo "csv=$csv_path"

    cmd="TERM=dumb $bench --backend=$backend --type=$scalar_type --warmup=$warmup --min_iters=$min_iters --max_iters=$max_iters --min_time=$min_time --csv=$csv_path $*"
    if [ -n "$variant_mode" ]; then
        cmd="BATCHLAS_GEMM_VARIANT=$variant_mode $cmd"
    fi
    if [ -n "$forced_kernel" ]; then
        cmd="BATCHLAS_GEMM_SYCL_KERNEL=$forced_kernel $cmd"
    fi
    if [ "$experimental" = "1" ]; then
        cmd="BATCHLAS_GEMM_EXPERIMENTAL=1 $cmd"
    fi

    echo "$cmd"

    if [ -n "$variant_mode" ]; then
        export BATCHLAS_GEMM_VARIANT="$variant_mode"
    else
        unset BATCHLAS_GEMM_VARIANT
    fi

    if [ -n "$forced_kernel" ]; then
        export BATCHLAS_GEMM_SYCL_KERNEL="$forced_kernel"
    else
        unset BATCHLAS_GEMM_SYCL_KERNEL
    fi

    if [ "$experimental" = "1" ]; then
        export BATCHLAS_GEMM_EXPERIMENTAL=1
    else
        unset BATCHLAS_GEMM_EXPERIMENTAL
    fi

    TERM=dumb "$bench" \
        --backend="$backend" \
        --type="$scalar_type" \
        --warmup="$warmup" \
        --min_iters="$min_iters" \
        --max_iters="$max_iters" \
        --min_time="$min_time" \
        --csv="$csv_path" "$@" | tee "$txt_path"

    unset BATCHLAS_GEMM_VARIANT
    unset BATCHLAS_GEMM_SYCL_KERNEL
    unset BATCHLAS_GEMM_EXPERIMENTAL
}

echo "backend=$backend type=$scalar_type build_dir=$build_dir output_prefix=$output_prefix"

run_case vendor vendor "" 0
run_case sycl_default sycl "" 0
run_case s2_u1_aligned sycl 128x32x32_s2_u1_aligned 0
run_case s2_u2 sycl 128x32x32_s2_u2 0
run_case k32_large sycl 128x64x32large 0
run_case_dims persistent_256 sycl 128x32x32_persistent 1 256 256 256 1024
run_case_dims persistent_512 sycl 128x32x32_persistent 1 512 512 512 512