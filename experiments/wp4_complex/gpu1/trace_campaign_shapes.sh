#!/usr/bin/env bash
# Confirm, per campaign cell, WHICH kernel the native arm actually ran.
# Never assume: BATCHLAS_GEMM_VARIANT does not force a kernel and an
# unrecognised value silently means vendor. Run separately from the timing --
# BATCHLAS_KERNEL_TRACE inflates wall time ~60%.
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
TR="$D/trace"; mkdir -p "$TR"
run() {
    local tag="$1"; shift
    rm -f "$TR/c_$tag.json"
    BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$TR/c_$tag.json" \
    BATCHLAS_BENCH_WARM_S=0 BATCHLAS_GEMM_ROUTE=native \
        "$D/cx_gemm_bench" "$@" 1 >/dev/null 2>&1
    printf '%-22s %-34s %s\n' "$tag" "$*" "$(python3 -c "
import json,collections
d=json.load(open('$TR/c_$tag.json'))
ev=d['traceEvents'] if isinstance(d,dict) else d
print([n for n,_ in collections.Counter(e.get('name') for e in ev).most_common(2) if n!='sycl_parallel_for'])
")"
}
for ty in cfloat cdouble; do
run panel_herk129_cn  $ty 129 129 48   64  C N
run panel_herk128_cn  $ty 128 128 48   64  C N
run panel_her2k96_nc  $ty 96  96  64   64  N C
run panel_syevx184_nc $ty 184 184 16   64  N C
run panel_2stage97_nc $ty 97  97  16   64  N C
run expand_trmm129_nn $ty 129 96  129  64  N N
run skinny_2stage_nn  $ty 16  64  16   64  N N
run skinny_2stage_cn  $ty 16  64  16   64  C N
run square256_nn      $ty 256 256 256  16  N N
run square512_nn      $ty 512 512 512  8   N N
run square1024_nn     $ty 1024 1024 1024 4 N N
done
