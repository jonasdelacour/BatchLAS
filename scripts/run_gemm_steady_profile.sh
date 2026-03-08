#!/usr/bin/env bash

build_dir=${1:-build}
output_dir=${2:-output/profiling}

backend=${BATCHLAS_BENCH_BACKEND:-CUDA}
scalar_type=${BATCHLAS_BENCH_TYPE:-float}
warmup=${BATCHLAS_BENCH_WARMUP:-5}

bench="$build_dir/benchmarks/gemm_steady_benchmark"
nsys_bin=${NSYS_BIN:-/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin/nsys}
ncu_bin=${NCU_BIN:-/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin/ncu}

if [ ! -x "$bench" ]; then
    echo "missing benchmark binary: $bench" >&2
    exit 1
fi

mkdir -p "$output_dir"

run_profile_pair() {
    label=$1
    kernel=$2

    local nsys_out="$output_dir/${label}_steady_512_nsys"
    local ncu_out="$output_dir/${label}_steady_512_ncu"

    echo
    echo "=== $label ==="

    BATCHLAS_GEMM_VARIANT=sycl \
    BATCHLAS_GEMM_SYCL_KERNEL="$kernel" \
    "$nsys_bin" profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        --stats=true \
        --force-overwrite=true \
        -o "$nsys_out" \
        "$bench" \
        --backend="$backend" \
        --type="$scalar_type" \
        --name=BM_GEMM_STEADY_EVENT \
        --warmup="$warmup" \
        --min_iters=1 \
        --max_iters=1 \
        --min_time=0 \
        512 512 512 512

    BATCHLAS_GEMM_VARIANT=sycl \
    BATCHLAS_GEMM_SYCL_KERNEL="$kernel" \
    "$ncu_bin" \
        --section SchedulerStats \
        --section WarpStateStats \
        --section SourceCounters \
        --section MemoryWorkloadAnalysis \
        --section ComputeWorkloadAnalysis \
        --kernel-name-base demangled \
        --kernel-name regex:GemmRegisterTiledKernel \
        --launch-count 1 \
        --target-processes all \
        --force-overwrite \
        --export "$ncu_out" \
        "$bench" \
        --backend="$backend" \
        --type="$scalar_type" \
        --name=BM_GEMM_STEADY_EVENT \
        --warmup="$warmup" \
        --min_iters=1 \
        --max_iters=1 \
        --min_time=0 \
        512 512 512 512
}

run_profile_pair s2_u2 128x32x32_s2_u2
run_profile_pair k32_large 128x64x32large