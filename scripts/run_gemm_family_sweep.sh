#!/usr/bin/env bash

build_dir=${1:-build}
backend=${BATCHLAS_BENCH_BACKEND:-CUDA}
scalar_type=${BATCHLAS_BENCH_TYPE:-float}

nn_bench="$build_dir/benchmarks/gemm_benchmark"
transpose_bench="$build_dir/benchmarks/gemm_transpose_benchmark"

if [ ! -x "$nn_bench" ]; then
    echo "missing benchmark binary: $nn_bench" >&2
    exit 1
fi

if [ ! -x "$transpose_bench" ]; then
    echo "missing benchmark binary: $transpose_bench" >&2
    exit 1
fi

variants=(
    reg128x32k16
    reg128x32k32
    reg128x32k32s2u1
    reg128x32k32s2u2
)

nn_cases=(
    "128 128 128 4096"
    "256 256 256 1024"
    "512 512 512 512"
)

transpose_cases=(
    "128 128 128 4096 1 1"
)

echo "backend=$backend type=$scalar_type build_dir=$build_dir"

for variant in "${variants[@]}"; do
    echo
    echo "=== $variant : NN sweep ==="
    for dims in "${nn_cases[@]}"; do
        echo "BATCHLAS_GEMM_VARIANT=sycl BATCHLAS_GEMM_SYCL_KERNEL=$variant $nn_bench $dims --backend=$backend --type=$scalar_type --warmup=5"
        BATCHLAS_GEMM_VARIANT=sycl \
        BATCHLAS_GEMM_SYCL_KERNEL="$variant" \
        "$nn_bench" $dims --backend="$backend" --type="$scalar_type" --warmup=5
    done

    echo
    echo "=== $variant : transpose sweep ==="
    for dims in "${transpose_cases[@]}"; do
        echo "BATCHLAS_GEMM_VARIANT=sycl BATCHLAS_GEMM_SYCL_KERNEL=$variant $transpose_bench $dims --backend=$backend --type=$scalar_type --warmup=5"
        BATCHLAS_GEMM_VARIANT=sycl \
        BATCHLAS_GEMM_SYCL_KERNEL="$variant" \
        "$transpose_bench" $dims --backend="$backend" --type="$scalar_type" --warmup=5
    done
done