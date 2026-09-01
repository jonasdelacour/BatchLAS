#!/usr/bin/env bash
# THE FAIRNESS DECOMPOSITION: how much of each arm's wall time is GPU kernel and
# how much is the per-call host chain.
#
# WHY THIS EXISTS. src/backends/cusparse.cc:283 calls handle.setStream and then
# spmm_planned_buffer_size -- a fresh cusparseSpMM_bufferSize -- on EVERY spmm
# call, and (since this session's heterogeneous-nnz fix) also builds an
# SpmmCsrBatchPlan that walks all `batch` row-offset pairs ON THE HOST, per call.
# None of that is hoistable by a caller, so the wall-clock number is what lanczos
# actually pays -- but a wall-clock ratio that is really a host-overhead ratio
# must not be reported as a kernel win. nsys separates the two: the CUDA kernel
# summary's average instance duration is the arm's kernel time, independent of
# how many calls the process made.
#
# One process at a time, device 1, exactly like every other runner here.
#
# usage: nsys_split.sh <outdir> <type> <route> ARGS...
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$D/../.." && pwd)"
BIN="$ROOT/build/benchmarks/spmm_benchmark"
OUTDIR="${1:?outdir}"; shift
TYPE="${1:?type}"; shift
ROUTE="${1:?route}"; shift
mkdir -p "$OUTDIR"
SAFE="${ROUTE//:/_}"
TAG="$(echo "$*" | tr ' ' '_')"
REP="$OUTDIR/nsys_${TYPE}_${SAFE}_${TAG}"

used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -gt 200 ] && { echo "REFUSING: GPU 1 holds ${used} MiB" >&2; exit 3; }

# Short measurement: the kernel-instance AVERAGE does not need many iterations,
# and a long trace is a large .nsys-rep for no extra information.
CUDA_VISIBLE_DEVICES=1 BATCHLAS_SPMM_ROUTE="$ROUTE" BATCHLAS_SPMM_WARM_MS=150 \
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
       -o "$REP" \
       "$BIN" --name=BM_SPMM_Grid --type="$TYPE" --warmup=1 --warmup_internal=1 \
              --min_time=1 --min_iters=3 --max_iters=3 \
              --csv="$REP.csv" "$@" > "$REP.stdout" 2>&1

nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
     -o "$REP" "$REP.nsys-rep" > /dev/null 2>&1 || \
nsys stats --report cuda_gpu_kern_sum --format csv "$REP.nsys-rep" > "$REP"_cuda_gpu_kern_sum.csv 2>/dev/null
echo "== $TYPE $ROUTE $*"
cat "$REP"*cuda_gpu_kern_sum.csv 2>/dev/null | head -12
